"""A long `bash` command never holds the agent's turn: it goes on in the background, the same process not
started again, when it reaches `move_after_s`, when System 1 finds it waits or serves, or when System 1's check finds
it stuck. It keeps its deadline, and the run that moved it does not end before its result is the agent's."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from intern.decisions import Answers, ChoiceAnswer, ChoiceQuestion, Question
from intern.events import Nudge, ProcessEvent, RunStart
from intern.messages import ToolResultMessage
from intern.processes import ProcessManager
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn
from intern.session import JobNotices, Session
from intern.tools import shell
from intern.tools.process import ProcessTool

from .support import Harness, Quiet
from .test_job_notices import _call, _handles, _scripted


class Judges(Quiet):
    """A System 1 that calls every command still running `kind` (finishes, works_long, waits, serves) and every
    command it looks at `verdict`."""

    def __init__(self, kind: str = "finishes", verdict: str = "progressing") -> None:
        super().__init__()
        self.kind, self.verdict = kind, verdict

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        answers = await super().decide(state, questions)
        for name, q in questions.items():
            if isinstance(q, ChoiceQuestion) and "works_long" in q.criteria:
                answers.answers[name] = ChoiceAnswer(choice=self.kind, probabilities={self.kind: 0.9, "finishes": 0.1})
            elif isinstance(q, ChoiceQuestion) and "stalled" in q.criteria:
                answers.answers[name] = ChoiceAnswer(choice=self.verdict, probabilities={self.verdict: 0.9})
        return answers


def _session(
    harness: Harness, move_after_s: float, schedule: tuple[float, ...] = (30.0,), notices: JobNotices = "quiet"
) -> Session:
    jobs = ProcessManager(move_after_s=move_after_s)
    harness.deps.processes = jobs
    harness.deps.tools = [ProcessTool(jobs) if tool.name == "process" else tool for tool in harness.deps.tools]
    session = Session(harness.deps, job_notices=notices)
    assert session._watch is not None  # pyright: ignore[reportPrivateUsage]
    session._watch.schedule = schedule  # pyright: ignore[reportPrivateUsage]
    return session


def _results(harness: Harness) -> list[str]:
    return [
        m.result.content
        for m in harness.provider.requests[-1].messages
        if isinstance(m, ToolResultMessage) and m.result.tool_name == "bash"
    ]


def _told(harness: Harness, reason: str = "process_ended") -> list[str]:
    return [n.text for n in harness.recorder.of(Nudge) if n.reason == reason]


async def _closed(session: Session) -> None:
    await session.aclose()
    assert session.deps.processes is not None
    await session.deps.processes.aclose()


async def test_a_command_still_running_goes_on_without_a_restart_and_the_run_waits_for_its_result(make_harness):
    command = "echo started >> runs.txt; echo working; sleep 1.5; echo finished"
    harness = make_harness(
        _scripted(
            _call("bash", command=command),
            lambda r: Turn(text="waiting on it"),
            lambda r: Turn(text="done: it finished"),
        )
    )
    session = _session(harness, move_after_s=0.6)
    try:
        outcome = await session.run("run it")
    finally:
        await _closed(session)
    [moved] = _results(harness)
    assert moved.startswith("[Running in the background as job") and "still running at" in moved and "working" in moved
    # The same process went on: it ran once, and its end is the agent's before the run ended.
    assert (harness.cwd / "runs.txt").read_text() == "started\n"
    assert outcome.final_text == "done: it finished" and outcome.turns == 3
    [told] = _told(harness)
    assert "exited, exit code 0" in told and "finished" in told
    calls = [e.process for e in harness.recorder.of(ProcessEvent) if e.process.kind == "call"]
    assert [c.state for c in calls if c.ended_at] == ["moved"]


async def test_in_the_terminal_ui_the_run_ends_at_once_and_the_command_s_end_wakes_the_agent(make_harness):
    harness = make_harness(
        _scripted(
            _call("bash", command="echo working; sleep 1.5; echo finished"),
            lambda r: Turn(text="it is running; I will report back"),
            lambda r: Turn(text="it finished"),
        )
    )
    session = _session(harness, move_after_s=0.4, notices="wake")
    try:
        started = time.monotonic()
        outcome = await session.run("run it")
        # The user has the session back while the command runs.
        assert outcome.final_text == "it is running; I will report back" and time.monotonic() - started < 1.4
        for _ in range(80):
            if len(harness.recorder.of(RunStart)) == 2 and not session.running:
                break
            await asyncio.sleep(0.05)
    finally:
        await _closed(session)
    assert session.last_outcome is not None and session.last_outcome.final_text == "it finished"
    [told] = _told(harness)
    assert "finished" in told


async def test_a_command_that_serves_is_not_waited_for_nor_stopped_at_its_timeout(make_harness, monkeypatch):
    monkeypatch.setattr(shell, "JUDGE_AFTER_S", 0.3)
    harness = make_harness(
        _scripted(
            _call("bash", command="echo listening; sleep 30", timeout_s=2), lambda r: Turn(text="the server is up")
        ),
        system1=Judges(kind="serves"),
    )
    session = _session(harness, move_after_s=30)
    try:
        started = time.monotonic()
        outcome = await asyncio.wait_for(session.run("start the server"), 10)
        assert outcome.final_text == "the server is up" and time.monotonic() - started < 5
        # The 2 s were for a call the agent expected to block on; a server does not end, and they would only kill it.
        await asyncio.sleep(max(0.0, 3 - (time.monotonic() - started)))
        assert session.deps.processes is not None and session.deps.processes.running(), "and it still serves"
    finally:
        await _closed(session)
    [moved] = [m for m in harness.provider.requests[1].messages if isinstance(m, ToolResultMessage)]
    assert "session closes" in moved.result.content


async def test_a_moved_command_found_to_serve_is_freed_of_its_timeout(make_harness):
    harness = make_harness(_scripted(_call("bash", command="sleep 30", timeout_s=2), lambda r: Turn(text="waiting")))
    session = _session(harness, move_after_s=0.4, notices="wake")  # the run ends; the job goes on
    jobs = session.deps.processes
    assert jobs is not None
    try:
        started = time.monotonic()
        await session.run("start the watcher")
        [job] = jobs.jobs_running()
        jobs.release(job.id)  # as a look that finds it up and waited on does
        await asyncio.sleep(max(0.0, 3 - (time.monotonic() - started)))
        assert jobs.job_running(job.id), "past its 2 s and still running"
    finally:
        await _closed(session)


async def test_a_moved_command_keeps_its_deadline(make_harness):
    harness = make_harness(_scripted(_call("bash", command="sleep 30", timeout_s=2), lambda r: Turn(text="waiting")))
    session = _session(harness, move_after_s=0.4)
    started = time.monotonic()
    try:
        await session.run("run it")
    finally:
        await _closed(session)
    assert time.monotonic() - started < 10, "stopped at its 2 s timeout, not run to 30 s"
    [told] = _told(harness)
    assert "timed out" in told


# One that serves moves too: test_a_command_that_serves_is_not_waited_for_nor_stopped_at_its_timeout.
@pytest.mark.parametrize(("kind", "moved"), [("waits", True), ("finishes", False)])
async def test_system1_moves_a_command_that_waits_at_once_and_leaves_one_that_works(
    make_harness, monkeypatch, kind, moved
):
    monkeypatch.setattr(shell, "JUDGE_AFTER_S", 0.3)
    harness = make_harness(
        _scripted(_call("bash", command="sleep 2; echo done"), lambda r: Turn(text="ok"), lambda r: Turn(text="ok")),
        system1=Judges(kind=kind),
    )
    session = _session(harness, move_after_s=30)
    try:
        await session.run("watch the build")
    finally:
        await _closed(session)
    result = next(m.result for r in harness.provider.requests for m in r.messages if isinstance(m, ToolResultMessage))
    assert result.content.startswith("[Running in the background as job") is moved
    assert (result.duration_ms < 1_500) is moved, "moved at the judgment, not at its end"


async def test_a_command_system1_finds_stalled_goes_on_in_the_background_now(make_harness):
    harness = make_harness(
        _scripted(_call("bash", command="echo waiting for input; sleep 3"), lambda r: Turn(text="ok")),
        system1=Judges(verdict="stalled"),
    )
    session = _session(harness, move_after_s=30, schedule=(0.4, 30.0))
    try:
        await session.run("run it")
    finally:
        await _closed(session)
    result = next(m.result for r in harness.provider.requests for m in r.messages if isinstance(m, ToolResultMessage))
    assert result.content.startswith("[Running in the background as job") and result.duration_ms < 2_000
    [check] = _told(harness, "process_check")
    assert "looks stalled" in check


async def test_with_moving_off_a_command_runs_to_its_end_as_before(make_harness):
    harness = make_harness(_scripted(_call("bash", command="sleep 1; echo done")))
    session = _session(harness, move_after_s=0)
    try:
        await session.run("run it")
    finally:
        await _closed(session)
    assert _results(harness) == ["done\n[exit code 0]"]


async def test_terminating_a_moved_command_stops_its_process(make_harness, tmp_path: Path):
    pid = tmp_path / "pid"

    def terminate(request: ProviderRequest) -> Turn:
        return Turn(tool_calls=[("process", {"action": "terminate", "handle": _handles_of_moved(request)})])

    harness = make_harness(
        _scripted(_call("bash", command=f"echo $$ > {pid}; sleep 30"), terminate, lambda r: Turn(text="stopped it"))
    )
    session = _session(harness, move_after_s=0.4)
    try:
        outcome = await session.run("run it")
        assert outcome.final_text == "stopped it", "nothing left running: the run ends at once"
    finally:
        await _closed(session)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


async def test_esc_while_the_run_waits_stops_the_run_and_the_command(make_harness, tmp_path: Path):
    pid = tmp_path / "pid"
    harness = make_harness(
        _scripted(_call("bash", command=f"echo $$ > {pid}; sleep 30"), lambda r: Turn(text="waiting"))
    )
    session = _session(harness, move_after_s=0.4)
    try:
        run = asyncio.create_task(session.run("run it"))
        await asyncio.sleep(1.5)
        assert session.running, "held for the moved command"
        session.abort()
        assert (await run).reason == "aborted"
        await asyncio.sleep(0.5)
    finally:
        await _closed(session)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid.read_text()), 0)


def _handles_of_moved(request: ProviderRequest) -> str:
    for message in request.messages:
        if isinstance(message, ToolResultMessage) and "background as job " in message.result.content:
            return message.result.content.split("background as job ", 1)[1].split(" ", 1)[0]
    return _handles(request)[0]
