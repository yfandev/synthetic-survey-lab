"""System 1 watches what runs: while commands run, it is asked on a schedule what each is doing, one ask for
all of them and none while nothing runs, and what it finds stalled, failing or waited on reaches the agent: at the
next turn, or between runs as a run of its own. Nothing here stops or moves a command."""

from __future__ import annotations

import asyncio

from intern.decisions import Answers, ChoiceAnswer, ChoiceQuestion, Question
from intern.events import Nudge, ProcessCheck, RunStart
from intern.messages import UserMessage
from intern.providers.fake import Turn
from intern.session import JobNotices, Session

from .support import Harness, Quiet
from .test_job_notices import _call, _scripted, _session

SCHEDULE = (0.3, 0.6, 1.2)


class Sees(Quiet):
    """A System 1 that finds every running command `verdict`, and is quiet about the rest; `down` fails the looks."""

    def __init__(self, verdict: str) -> None:
        super().__init__()
        self.verdict, self.down = verdict, False

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        looks = [n for n, q in questions.items() if isinstance(q, ChoiceQuestion) and "stalled" in q.criteria]
        if looks and self.down:
            raise RuntimeError("backend down")
        answers = await super().decide(state, questions)
        for name in looks:
            answers.answers[name] = ChoiceAnswer(choice=self.verdict, probabilities={self.verdict: 0.9, "unclear": 0.1})
        return answers


def _watched(harness: Harness, notices: JobNotices = "quiet") -> Session:
    session, _ = _session(harness, notices)
    assert session._watch is not None  # pyright: ignore[reportPrivateUsage]
    session._watch.schedule, session._watch.every = SCHEDULE, 30  # pyright: ignore[reportPrivateUsage]
    return session


def _told(harness: Harness) -> list[str]:
    return [n.text for n in harness.recorder.of(Nudge) if n.reason == "process_check"]


async def test_looks_come_on_the_schedule_one_ask_for_all_and_none_once_nothing_runs(make_harness):
    harness = make_harness([], system1=(system1 := Sees("progressing")))
    session = _watched(harness)
    jobs = session.deps.processes
    assert jobs is not None
    try:
        await jobs.start("sleep 1.6", cwd=harness.cwd)
        await jobs.start("sleep 1.6", cwd=harness.cwd)
        await asyncio.sleep(2.6)
    finally:
        await session.aclose()
        await jobs.aclose()
    checks = harness.recorder.of(ProcessCheck)
    # At 0.3, 0.6 and 1.2 s of age, each look over both jobs; after they ended at 1.6 s, none.
    assert [len(c.verdicts) for c in checks] == [2, 2, 2]
    assert all(v.verdict == "progressing" and not v.told for c in checks for v in c.verdicts)
    assert 1 <= len([q for _, q in system1.asks if "p0" in q]) <= 3, "a look the same as the last is the cache's"
    assert _told(harness) == [], "progressing tells the agent nothing"


async def test_a_command_found_stalled_is_told_at_the_next_turn_the_latest_look_only(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="echo compiling; sleep 30"), _call("bash", command="sleep 1.5")
        ),
        system1=Sees("stalled"),
    )
    session = _watched(harness)
    try:
        assert (await session.run("build it")).reason == "completed"
    finally:
        await session.aclose()
        assert session.deps.processes is not None
        await session.deps.processes.aclose()
    assert len(harness.recorder.of(ProcessCheck)) >= 2, "several looks during the 1.5 s call"
    [told] = _told(harness)
    assert "background job" in told and "looks stalled" in told and "Its last line: compiling" in told
    # It reached the agent before its next turn.
    assert any(isinstance(m, UserMessage) and told in m.content for m in harness.provider.requests[2].messages)


async def test_between_runs_a_stalled_command_starts_a_run_that_tells_the_agent(make_harness):
    harness = make_harness(
        _scripted(_call("process", action="start", command="sleep 30"), lambda r: Turn(text="started")),
        system1=Sees("stalled"),
    )
    session = _watched(harness, "wake")
    jobs = session.deps.processes
    assert jobs is not None
    try:
        await session.run("start it")
        for _ in range(60):
            if len(harness.recorder.of(RunStart)) == 2 and not session.running:
                break
            await asyncio.sleep(0.05)
        assert len(harness.recorder.of(RunStart)) == 2 and _told(harness)
        assert harness.recorder.of(ProcessCheck)[0].turn is None, "looked at between runs"
    finally:
        await session.aclose()
        await jobs.aclose()


async def test_a_look_system1_cannot_answer_during_a_run_stops_it(make_harness):
    harness = make_harness(
        _scripted(_call("process", action="start", command="sleep 30"), _call("bash", command="sleep 1")),
        system1=(system1 := Sees("progressing")),
    )
    system1.down = True
    harness.deps.decisions._retry_window_s = 0.1  # pyright: ignore[reportPrivateUsage]
    session = _watched(harness)
    try:
        outcome = await session.run("start it")
    finally:
        await session.aclose()
        assert session.deps.processes is not None
        await session.deps.processes.aclose()
    assert outcome.reason == "system1_unavailable" and outcome.turns == 2
