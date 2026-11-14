"""Waiting is the harness's, not the agent's: `process poll` never blocks; a turn that only polls running
jobs hands over (the terminal UI gets the session back, a quiet session holds the run for news, System 1's looks among
it); plain waits start in the background; and a message from the user never queues behind a running command."""

from __future__ import annotations

import asyncio
import time

from intern.events import Nudge, ProcessEvent, RunEnd, RunStart
from intern.messages import ToolResultMessage, UserMessage
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn
from intern.tools import process as process_tool
from intern.tools import shell

from .test_job_notices import _call, _handles, _scripted, _session, _told


def _poll(request: ProviderRequest) -> Turn:
    """The agent says where things stand and polls the job, with nothing else to do."""
    return Turn(text="It is running.", tool_calls=[("process", {"action": "poll", "handle": _handles(request)[0]})])


def _silent_poll(request: ProviderRequest) -> Turn:
    return Turn(tool_calls=[("process", {"action": "poll", "handle": _handles(request)[0]})])


def _results(harness, tool: str) -> list:
    """The tool's results, in the order they came, as the last request carries them."""
    last = harness.provider.requests[-1].messages
    return [m.result for m in last if isinstance(m, ToolResultMessage) and m.result.tool_name == tool]


async def test_a_quiet_run_holds_for_the_job_instead_of_the_agent_polling(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="sleep 1; echo built"), _silent_poll, lambda r: Turn(text="built")
        )
    )
    session, jobs = _session(harness)
    try:
        started = time.monotonic()
        outcome = await session.run("build it and tell me")
        took = time.monotonic() - started
    finally:
        await session.aclose()
        await jobs.aclose()
    assert outcome.final_text == "built" and 0.9 < took < 5
    [_, poll] = _results(harness, "process")
    assert poll.duration_ms < 500, "the agent's poll read and returned; the run did the waiting"
    # Never told to end its turn first: in -p that would end the run, and the session would stop the job.
    assert [n.reason for n in harness.recorder.of(Nudge) if n.reason.startswith("process_")] == ["process_ended"]
    [told] = _told(harness)
    assert "built" in told


async def test_a_message_from_the_user_ends_a_quiet_run_s_wait_for_a_job(make_harness):
    harness = make_harness(
        _scripted(_call("process", action="start", command="sleep 30"), _poll, lambda r: Turn(text="stopped waiting"))
    )
    session, jobs = _session(harness)
    try:
        run = asyncio.create_task(session.run("start it and wait"))
        await asyncio.sleep(0.5)
        started = time.monotonic()
        session.steer("are you stuck?")
        assert (await run).final_text == "stopped waiting"
        assert time.monotonic() - started < 5, "the wait ended on the message, not at the job's end"
        last = harness.provider.requests[-1].messages
        assert any(isinstance(m, UserMessage) and m.content == "are you stuck?" for m in last)
    finally:
        await session.aclose()
        await jobs.aclose()


async def test_a_job_that_never_ends_cannot_hold_a_quiet_run_system1_s_look_brings_the_agent_back(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="echo serving; sleep 30"), _poll, lambda r: Turn(text="it serves")
        )
    )
    session, jobs = _session(harness)
    assert session._watch is not None  # pyright: ignore[reportPrivateUsage]
    session._watch.schedule = (0.5, 30.0)  # pyright: ignore[reportPrivateUsage]
    try:
        started = time.monotonic()
        outcome = await session.run("start the server")
        assert outcome.final_text == "it serves" and time.monotonic() - started < 5
        assert jobs.jobs_running(), "the job still runs"
    finally:
        await session.aclose()
        await jobs.aclose()
    looks = [n.text for n in harness.recorder.of(Nudge) if n.reason == "process_look"]
    assert len(looks) == 1 and "System 1 looked at what runs" in looks[0]


async def test_in_the_terminal_ui_a_turn_that_only_polls_hands_the_session_back(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="sleep 1; echo built"),
            _poll,
            lambda r: Turn(text="built, as you asked"),
        )
    )
    session, jobs = _session(harness, "wake")
    try:
        started = time.monotonic()
        first = await session.run("build it and tell me")
        assert first.reason == "waiting" and time.monotonic() - started < 0.9, "back to the user before the job ends"
        for _ in range(60):
            if len(harness.recorder.of(RunStart)) == 2 and not session.running:
                break
            await asyncio.sleep(0.05)
    finally:
        await session.aclose()
        await jobs.aclose()
    assert session.last_outcome is not None and session.last_outcome.final_text == "built, as you asked"
    assert [e.reason for e in harness.recorder.of(RunEnd)] == ["waiting", "completed"]


async def test_a_plain_wait_starts_in_the_background(make_harness, monkeypatch):
    monkeypatch.setattr(shell, "WAIT_MIN_SLEEP_S", 0.5)
    harness = make_harness(
        _scripted(
            _call("bash", command="sleep 1; echo slept"), lambda r: Turn(text="waiting"), lambda r: Turn(text="ok")
        )
    )
    session, jobs = _session(harness)
    jobs.move_after_s = 30
    try:
        await session.run("wait a second")
    finally:
        await session.aclose()
        await jobs.aclose()
    [bash] = _results(harness, "bash")
    assert bash.content.startswith("[Running in the background as job") and bash.duration_ms < 500
    moved = next(e.process for e in harness.recorder.of(ProcessEvent) if e.process.state == "moved")
    assert moved.note == "`sleep 1` is a wait"


async def test_a_message_from_the_user_never_waits_behind_a_running_command(make_harness):
    def stop_it(request: ProviderRequest) -> Turn:
        [moved] = [m for m in request.messages if isinstance(m, ToolResultMessage)]
        handle = moved.result.content.split("background as job ", 1)[1].split(" ", 1)[0]
        return Turn(tool_calls=[("process", {"action": "terminate", "handle": handle})])

    hang = 'python3 -c "import time; time.sleep(30)"'  # not a plain wait: it runs in the foreground
    harness = make_harness(_scripted(_call("bash", command=hang), stop_it, lambda r: Turn(text="stopped it")))
    session, jobs = _session(harness)
    jobs.move_after_s = 30
    try:
        run = asyncio.create_task(session.run("run it"))
        await asyncio.sleep(0.5)
        session.steer("are you stuck?")
        outcome = await asyncio.wait_for(run, 10)
    finally:
        await session.aclose()
        await jobs.aclose()
    assert outcome.final_text == "stopped it"
    [bash] = _results(harness, "bash")
    assert "a message from the user came in" in bash.content and bash.duration_ms < 2_000
    second = harness.provider.requests[1].messages
    assert any(isinstance(m, UserMessage) and m.content == "are you stuck?" for m in second)


async def test_before_handing_over_the_agent_answers_what_does_not_wait_on_the_job(make_harness):
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="sleep 1; echo ci passed"),
            _silent_poll,  # it would hand over without a word to the user: told to answer first, once
            lambda r: Turn(text="CI is running. The tests cannot run: nothing listens on port 5433."),
            lambda r: Turn(text="CI passed."),
        )
    )
    session, jobs = _session(harness, "wake")
    try:
        first = await session.run("run CI, and tell me why the tests hang")
        assert first.final_text.startswith("CI is running") and first.reason == "completed"
        for _ in range(60):
            if len(harness.recorder.of(RunStart)) == 2 and not session.running:
                break
            await asyncio.sleep(0.05)
    finally:
        await session.aclose()
        await jobs.aclose()
    assert [n.reason for n in harness.recorder.of(Nudge) if n.reason.startswith("process_")] == [
        "process_waiting",
        "process_ended",
    ]
    assert session.last_outcome is not None and session.last_outcome.final_text == "CI passed."


async def test_asked_how_a_job_is_going_the_agent_reads_its_log_before_it_hands_over(make_harness):
    # The job writes to a log, as `docker build … > build.log 2>&1` does, so a poll shows nothing. Nudged before it
    # hands over, the agent gets a turn to read the log, and the run does not end on the nudge.
    harness = make_harness(
        _scripted(
            _call("process", action="start", command="echo 'step 2/12: load metadata' > build.log; sleep 30"),
            _silent_poll,
            _call("bash", command="for i in $(seq 50); do [ -s build.log ] && break; sleep 0.1; done; cat build.log"),
            lambda r: Turn(text="It is at step 2 of 12, loading metadata."),
        )
    )
    session, jobs = _session(harness, "wake")
    try:
        outcome = await asyncio.wait_for(session.run("how is the build going?"), 15)
    finally:
        await session.aclose()
        await jobs.aclose()
    assert outcome.reason == "completed" and outcome.final_text.startswith("It is at step 2")
    [read] = _results(harness, "bash")
    assert "step 2/12" in read.content
    assert [n.reason for n in harness.recorder.of(Nudge) if n.reason.startswith("process_")] == ["process_waiting"]


async def test_a_quick_command_started_as_a_job_costs_one_turn(make_harness, monkeypatch):
    monkeypatch.setattr(process_tool, "QUICK_S", 3.0)
    harness = make_harness(_scripted(_call("process", action="start", command="echo quick"), lambda r: Turn(text="ok")))
    session, jobs = _session(harness)
    try:
        assert (await session.run("run it")).turns == 2  # the start and the answer: no poll
    finally:
        await session.aclose()
        await jobs.aclose()
    [start] = _results(harness, "process")
    assert '"state":"exited"' in start.content and "quick" in start.content
    assert _told(harness) == [], "its end came with the start, not as a notice"
