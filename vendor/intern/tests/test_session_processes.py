"""What runs in a session is known in one place: the session's job manager records every command from its
start to its end (a job, a `bash` call, what a command leaves running), reports each as a journaled event, and the
State written at a compaction keeps the jobs still running, so the agent does not lose a job whose start it cut."""

from __future__ import annotations

import asyncio
import json

import pytest

from intern.events import ProcessEvent, State
from intern.journal import EventEntry
from intern.processes import ProcessManager
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn
from intern.session import Session
from intern.session_cut import STATE_SHARE, cut_view, estimate_tokens
from intern.tools.process import ProcessTool

from .fake_openai import Reply
from .support import Quiet, is_state_edit
from .test_loop import Opinionated
from .test_session_setup import configured
from .test_state_edit_writes import WINDOW, agent


def _journaled(session: Session) -> list[ProcessEvent]:
    return [e.event for e in session.entries() if isinstance(e, EventEntry) and isinstance(e.event, ProcessEvent)]


async def test_every_command_of_a_default_session_is_reported_from_start_to_end_and_journaled(tmp_path):
    job, call = "sleep 0.3; echo job-done", "sleep 1 > /dev/null 2>&1 & echo started"
    replies = [
        Reply(tool_calls=[("process", {"action": "start", "command": job})]),
        Reply(tool_calls=[("bash", {"command": call})]),
        Reply(text="done"),
    ]
    # No tool packs configured: the job manager and the process tool are the session's own.
    async with configured(tmp_path, replies, "") as (session, _, _, _):
        await session.run("start a job, and leave a command running")
        jobs = session.deps.processes
        assert jobs is not None
        # The job and what the command left running end after the run did: reported all the same.
        for _ in range(200):
            if not jobs.running():
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail(f"still running after 10 s: {jobs.running()}")
        events = _journaled(session)
        started = [(e.process.kind, e.process.command) for e in events if e.action == "started"]
        assert started == [("job", job), ("call", call), ("left_running", call)]
        ended = {e.process.kind: e.process for e in events if e.action == "ended"}
        assert set(ended) == {"job", "call", "left_running"}
        assert ended["job"].state == "exited" and ended["job"].exit_code == 0
        assert ended["call"].state == "exited" and ended["call"].exit_code == 0
        # Left running a second past the call's end, and seen to end only once its group was empty.
        left = ended["left_running"]
        assert left.ended_at is not None and left.ended_at - ended["call"].started_at >= 900


async def test_a_job_still_running_at_a_compaction_is_in_the_state_and_one_that_ended_is_not(make_harness):
    writes: list[ProviderRequest] = []
    harness = make_harness(agent(6, lambda n: "No change.", writes=writes), window=WINDOW, system1=Opinionated())
    harness.deps.processes = jobs = ProcessManager()
    try:
        running = await jobs.start("sleep 30", cwd=harness.cwd)
        finished = await jobs.start("true", cwd=harness.cwd)
        assert (await jobs.poll(finished.handle, wait_s=5)).state == "exited"
        assert (await harness.run()).reason == "completed"
        states = [s for s in harness.recorder.of(State) if not s.dropped]
        assert len(states) >= 2
        for state in states:
            # What the agent needs to reach the job again, after the turn that started it is cut.
            assert f"job {running.handle}" in state.text and "`sleep 30`" in state.text
            assert finished.handle not in state.text
            # Code's sections come last, the user's requests the very last.
            assert state.text.index("## Running processes") < state.text.index("## User requests")
        # The harness keeps the section: the writer is never given it to edit.
        edits = [w for w in writes if is_state_edit(w)]
        assert edits and all(running.handle not in str(w.messages[-1]) for w in edits)
    finally:
        await jobs.aclose()


async def test_system1_reads_what_runs_beside_the_goal_and_it_is_never_cut_to_fit(make_harness):
    long = "x" * 4_000
    harness = make_harness(
        [
            Turn(tool_calls=[("process", {"action": "start", "command": "echo compiling; sleep 30"})]),
            *[Turn(tool_calls=[("bash", {"command": f"sleep 0.05; printf '{long}'"})]) for _ in range(12)],
            Turn(text="done"),
        ],
        system1=(system1 := Quiet()),
    )
    harness.deps.processes = jobs = ProcessManager()
    harness.deps.tools = [ProcessTool(jobs) if tool.name == "process" else tool for tool in harness.deps.tools]
    # A window this small cuts the history (past the first message and the newest six, which stay whole): what runs
    # stays whole beside the goal.
    harness.deps.decisions._max_state_tokens = 2_000  # pyright: ignore[reportPrivateUsage]
    try:
        assert (await harness.run()).reason == "completed"
        limit = int(2_000 * STATE_SHARE)
        running = jobs.for_system1()
        assert cut_view(harness.deps.journal.active_path(), limit, running).whole_tokens > limit, "the cut had to cut"
    finally:
        await jobs.aclose()
    asked = [state for state, questions in system1.asks if "mode" in questions]  # the supervisor's, every turn
    last = json.loads(asked[-1])
    assert estimate_tokens(asked[-1]) <= limit and last["goal"] == "do the task"
    [job] = last["processes"]
    assert job["kind"] == "job" and job["command"] == "echo compiling; sleep 30"
    assert job["last_output"] == "compiling" and job["quiet_s"] >= 0 and job["elapsed_s"] >= 0
