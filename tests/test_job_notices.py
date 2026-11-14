"""A job's end is news the agent gets without asking: told at the next turn as the harness's message, with
how it ended and the end of its output; never twice, and never for an end the agent already saw. A run does not end
while such news is untold; between runs it waits for the next prompt (`quiet`) or starts a run itself (`wake`)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from intern.events import Nudge, RunStart
from intern.journal import Journal, MessageEntry
from intern.messages import JsonValue, ToolResultMessage, UserMessage
from intern.processes import ProcessManager
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn
from intern.session import JobNotices, Session
from intern.tools.process import ProcessTool

from .support import Harness


def _session(harness: Harness, notices: JobNotices = "quiet") -> tuple[Session, ProcessManager]:
    """The session's job manager, shared by its `process` tool, as session setup builds them."""
    jobs = ProcessManager()
    harness.deps.processes = jobs
    harness.deps.tools = [ProcessTool(jobs) if tool.name == "process" else tool for tool in harness.deps.tools]
    return Session(harness.deps, job_notices=notices), jobs


def _handles(request: ProviderRequest) -> list[str]:
    """The handles of the jobs started so far, from the `process` results in the request."""
    found = []
    for message in request.messages:
        if isinstance(message, ToolResultMessage) and message.result.tool_name == "process":
            handle = json.loads(message.result.content).get("handle")
            if handle and handle not in found:
                found.append(handle)
    return found


def _told(harness: Harness) -> list[str]:
    return [event.text for event in harness.recorder.of(Nudge) if event.reason == "process_ended"]


def _scripted(*steps: Callable[[ProviderRequest], Turn]) -> Callable[[ProviderRequest], Turn]:
    """The agent's turns in order; past the last, it answers `done`."""
    calls: list[int] = []

    def script(request: ProviderRequest) -> Turn:
        calls.append(1)
        return steps[len(calls) - 1](request) if len(calls) <= len(steps) else Turn(text="done")

    return script


def _call(name: str, **arguments: JsonValue) -> Callable[[ProviderRequest], Turn]:
    return lambda request: Turn(tool_calls=[(name, dict(arguments))])


async def test_a_job_that_ends_during_the_run_is_told_at_the_next_turn_as_the_harness_s(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="sleep 0.3; echo built-ok; exit 3"),
            _call("bash", command="sleep 1"),
        )
    )
    session, jobs = _session(harness)
    try:
        assert (await session.run("build it in the background")).reason == "completed"
    finally:
        await jobs.aclose()
    [told] = _told(harness)
    assert "exited, exit code 3" in told and "built-ok" in told
    # Before the agent's next turn, and as the harness's word, not the user's: it is no request of theirs.
    third = harness.provider.requests[2].messages
    notice = next(m for m in third if isinstance(m, UserMessage) and m.content == told)
    assert notice.source == "harness"


async def test_an_end_the_agent_already_saw_is_not_told(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="echo polled"),
            # It ends during the bash call, and the poll after it in the same turn shows the end. The call is long
            # enough for `echo` to end on a loaded machine too: at 0.3 s, a busy one ran out of time.
            lambda request: Turn(
                tool_calls=[
                    ("bash", {"command": "sleep 2"}),
                    ("process", {"action": "poll", "handle": _handles(request)[0]}),
                ]
            ),
            _call("process", action="start", command="sleep 30"),
            lambda request: Turn(tool_calls=[("process", {"action": "terminate", "handle": _handles(request)[1]})]),
            _call("bash", command="sleep 0.3"),
        )
    )
    session, jobs = _session(harness)
    try:
        assert (await session.run("start two jobs")).reason == "completed"
    finally:
        await jobs.aclose()
    assert _told(harness) == []


async def test_a_reopened_session_tells_the_agent_of_the_job_its_close_stopped(make_harness, tmp_path):
    # The user quits with a job running and resumes later: the job was stopped at the close, and the agent hears so
    # before its first turn, instead of polling a handle no manager knows and guessing what became of it.
    path = tmp_path / "session.jsonl"
    before = make_harness(_scripted(_call("process", action="start", command="sleep 30"), lambda r: Turn(text="ok")))
    before.deps.journal = Journal(cwd=str(before.cwd), path=path)
    session, jobs = _session(before, "wake")
    assert (await session.run("start the soak test")).reason == "completed"
    [handle] = _handles(before.provider.requests[-1])
    await session.aclose()
    await jobs.aclose()  # as the session's resources close it: a job still running is stopped

    after = make_harness(_scripted(_call("process", action="poll", handle=handle)))
    after.deps.journal = Journal(cwd=str(after.cwd), path=path)
    session, jobs = _session(after, "wake")
    try:
        assert (await session.run("did the soak test finish?")).reason == "completed"
    finally:
        await jobs.aclose()
    [told] = _told(after)
    assert handle in told and "sleep 30" in told
    # Before the agent's first turn: with the user's question, so the first answer is already right.
    assert any(isinstance(m, UserMessage) and told in m.content for m in after.provider.requests[0].messages)
    # Its handle is explained, not taken for another session's.
    poll = [m for m in after.provider.requests[1].messages if isinstance(m, ToolResultMessage)][-1]
    assert "sleep 30" in poll.result.content and "Unknown process handle" not in poll.result.content


async def test_a_run_does_not_end_while_a_job_s_end_is_untold(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="sleep 0.3; echo late"),
            lambda request: Turn(text="started it"),
            lambda request: Turn(text="it finished: late"),
        )
    )
    session, jobs = _session(harness)
    provider, stream = harness.provider, harness.provider.stream

    async def slow_final(request: ProviderRequest):
        if len(provider.requests) == 1:  # the reply that would end the run comes after the job ended
            await asyncio.sleep(0.8)
        async for item in stream(request):
            yield item

    provider.stream = slow_final  # type: ignore[method-assign]
    try:
        outcome = await session.run("start it and tell me")
    finally:
        await jobs.aclose()
    assert outcome.final_text == "it finished: late" and outcome.turns == 3
    assert len(_told(harness)) == 1 and "late" in _told(harness)[0]


async def test_quiet_holds_an_end_between_runs_for_the_next_prompt(make_harness):
    harness = make_harness(
        _scripted(_call("process", action="start", command="sleep 0.2; echo idle-done"), lambda r: Turn(text="started"))
    )
    session, jobs = _session(harness, "quiet")
    try:
        await session.run("start it")
        await asyncio.sleep(1.2)
        assert len(harness.recorder.of(RunStart)) == 1, "no run starts by itself"
        await session.run("anything new?")
    finally:
        await jobs.aclose()
    [told] = _told(harness)
    assert "idle-done" in told
    # Told with the prompt that came next, right after it, as the harness's word.
    said = [e.message for e in session.entries() if isinstance(e, MessageEntry) and isinstance(e.message, UserMessage)]
    assert said[-2:] == [UserMessage(content="anything new?"), UserMessage(content=told, source="harness")]
    last = harness.provider.requests[-1].messages[-1]
    assert isinstance(last, UserMessage) and last.content == f"anything new?\n\n{told}"


async def test_wake_starts_a_run_for_an_end_between_runs_but_not_for_one_an_abort_caused(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="sleep 0.2; echo woke"),
            lambda r: Turn(text="started"),
            lambda r: Turn(text="it ended"),
            _call("process", action="start", command="sleep 30"),
            _call("bash", command="sleep 10"),
        )
    )
    session, jobs = _session(harness, "wake")
    try:
        await session.run("start it")
        for _ in range(100):
            if len(harness.recorder.of(RunStart)) == 2 and not session.running:
                break
            await asyncio.sleep(0.05)
        # The second run is the session's own: no message of the user's, only what ended.
        assert len(harness.recorder.of(RunStart)) == 2 and "woke" in _told(harness)[0]
        assert session.last_outcome is not None and session.last_outcome.final_text == "it ended"
        assert not any(m.content == "" for m in harness.provider.requests[2].messages if isinstance(m, UserMessage))

        # Esc stops a run and the job it started: that end is the user's doing, and wakes nothing.
        run = asyncio.create_task(session.run("start another"))
        await asyncio.sleep(0.5)
        session.abort()
        assert (await run).reason == "aborted"
        await asyncio.sleep(1.5)
        assert len(harness.recorder.of(RunStart)) == 3
    finally:
        await jobs.aclose()
