"""The loop's own behaviour, end to end over a scripted provider and real tools."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress

import pytest

from intern import hooks as h
from intern import loop
from intern.context import STATE_FILE, requests_section
from intern.decisions import Answer, Answers, ChoiceAnswer, ChoiceQuestion, NoulAnswer, NoulQuestion, Question
from intern.events import (
    Compaction,
    Decision,
    HookError,
    MessageEvent,
    Nudge,
    State,
    StateDelta,
    StateWriting,
    ToolCallEnd,
)
from intern.journal import CompactionEntry, Journal, MessageEntry, ObservationEntry
from intern.loop import RunControl
from intern.messages import (
    AssistantMessage,
    JsonObject,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from intern.models.profile import Profile
from intern.providers.base import ProviderRequest
from intern.providers.fake import FakeProvider, Turn
from intern.session import Session

from .support import GreetTool, WithTools, is_state_write


def history(harness) -> list:
    return [e.message for e in harness.deps.journal.active_path() if isinstance(e, MessageEntry)]


async def test_run_without_tools_completes_with_the_reply(make_harness):
    harness = make_harness([Turn(text="All done.")])
    outcome = await harness.run("say hi")
    assert outcome.reason == "completed"
    assert outcome.final_text == "All done."
    assert harness.recorder.end.turns == 1


async def test_tool_calls_run_and_their_results_reach_the_next_request(make_harness):
    harness = make_harness(
        [
            Turn(tool_calls=[("write", {"path": "notes.txt", "content": "hello\n"})]),
            Turn(tool_calls=[("read", {"path": "notes.txt"})]),
            Turn(text="The file says hello."),
        ]
    )
    outcome = await harness.run()
    assert outcome.reason == "completed"
    assert (harness.cwd / "notes.txt").read_text() == "hello\n"
    last_request = harness.provider.requests[-1]
    tool_results = [m for m in last_request.messages if isinstance(m, ToolResultMessage)]
    assert [r.result.status for r in tool_results] == ["ok", "ok"]
    assert tool_results[-1].result.content.startswith("hello")


async def test_tool_call_stop_prevents_later_writes_in_the_batch(make_harness):
    seen = []

    class StopFirst(h.Extension):
        async def tool_call(self, call, ctx):
            seen.append(call.id)
            return h.Stop(reason="stopped_by_hook", detail="do not continue")

    harness = make_harness(
        [Turn(tool_calls=[("write", {"path": name, "content": "x"}) for name in ("first", "second")])],
        extensions=[StopFirst()],
    )
    outcome = await harness.run()
    assert outcome.reason == "stopped_by_hook"
    assert len(seen) == 1
    assert not (harness.cwd / "first").exists() and not (harness.cwd / "second").exists()
    assert [event.status for event in harness.recorder.of(ToolCallEnd)] == ["refused", "refused"]
    assert len([m for m in history(harness) if isinstance(m, ToolResultMessage)]) == 2


async def test_read_only_batch_cannot_race_a_tool_call_stop(make_harness):
    seen = []

    class StopAfterYield(h.Extension):
        async def tool_call(self, call, ctx):
            seen.append(call.id)
            await asyncio.sleep(0)
            return h.Stop(reason="stopped_by_hook")

    harness = make_harness(
        [Turn(tool_calls=[("read", {"path": name}) for name in ("first", "second")])],
        extensions=[StopAfterYield()],
    )
    outcome = await harness.run()
    assert outcome.reason == "stopped_by_hook" and len(seen) == 1
    assert [event.status for event in harness.recorder.of(ToolCallEnd)] == ["refused", "refused"]


async def test_unknown_tool_returns_an_error_naming_the_real_tools(make_harness):
    harness = make_harness([Turn(tool_calls=[("open_file", {"path": "x"})]), Turn(text="ok")])
    await harness.run()
    result = harness.recorder.of(ToolCallEnd)[0]
    assert result.status == "error"
    assert "there is no tool named 'open_file'" in result.content
    assert "read" in result.content


async def test_invalid_arguments_come_back_as_an_actionable_error(make_harness):
    harness = make_harness([Turn(tool_calls=[("read", {"offset": 3})]), Turn(text="ok")])
    await harness.run()
    end = harness.recorder.of(ToolCallEnd)[0]
    assert end.status == "error"
    assert "required: path" in end.content


async def test_unparsable_tool_calls_stay_out_of_history_and_the_model_is_told(make_harness):
    harness = make_harness([Turn(invalid_tool_calls=[("edit", '{"path": "a.py", ')]), Turn(text="ok")])
    outcome = await harness.run()
    assert outcome.reason == "completed"
    second = harness.provider.requests[1]
    assistant = [m for m in second.messages if isinstance(m, AssistantMessage)]
    assert all(not m.tool_calls for m in assistant)
    assert "Your call to edit was not run" in second.messages[-1].content
    assert harness.recorder.of(Nudge)[0].reason == "invalid_tool_call"


async def test_a_reply_cut_off_at_the_output_limit_is_continued(make_harness):
    harness = make_harness([Turn(text="The answer is", stop="length"), Turn(text="forty-two.")])
    outcome = await harness.run()
    assert outcome.reason == "completed"
    assert harness.recorder.of(Nudge)[0].reason == "output_limit"
    assert len(harness.provider.requests) == 2


async def test_continuations_are_capped_by_the_profile_and_then_the_run_stops_on_the_limit(make_harness):
    harness = make_harness([Turn(text="part", stop="length")] * 5, profile=Profile(max_continuations=2))
    outcome = await harness.run()
    assert outcome.reason == "output_limit" and "3 times" in (outcome.detail or "")
    assert len(harness.recorder.of(Nudge)) == 2
    assert len(harness.provider.requests) == 3


async def test_a_window_cap_limits_the_prompt_but_not_the_reply(make_harness):
    # A 131k model run at --window 16384: the prompt budget is the working window, the reply may use the
    # model's own context. Before, the reply cap was 0.15 x 16384 = 2457 tokens.
    harness = make_harness([Turn(text="done")], window=131_072, profile=Profile(max_output_tokens=32_768))
    window = harness.deps.system2.window
    harness.deps.system2.window = window.model_copy(update={"working": 16_384, "user_cap": 16_384})
    await harness.run()
    request = harness.provider.requests[0]
    assert request.max_output_tokens == int(131_072 * 0.15)
    assert request.max_output_tokens > 16_384, "the reply is not held under the working window"


async def test_a_small_advertised_window_still_bounds_the_reply(make_harness):
    harness = make_harness([Turn(text="done")], window=16_384, profile=Profile(max_output_tokens=32_768))
    await harness.run()
    assert harness.provider.requests[0].max_output_tokens == int(16_384 * 0.15)


async def test_a_reply_that_fails_with_a_retryable_error_is_requested_again(make_harness, monkeypatch):
    monkeypatch.setattr(loop, "REPLY_RETRY_WAITS_S", (0.0, 0.0))
    harness = make_harness([Turn(error="stream broke", retryable=True), Turn(text="done")])
    outcome = await harness.run()
    assert outcome.reason == "completed" and outcome.final_text == "done"
    assert len(harness.provider.requests) == 2


async def test_turn_budget_ends_with_a_tool_less_reply_to_the_user(make_harness):
    def script(request: ProviderRequest) -> Turn:
        if not request.tools:
            return Turn(text="Edited money.py; I did not get to run the tests.")
        return Turn(tool_calls=[("bash", {"command": "true"})])

    harness = make_harness(script, profile=Profile(max_turns=3))
    outcome = await harness.run()
    assert outcome.reason == "turn_budget"
    # Three budgeted turns of tool calls, then one wrap-up turn with no tools.
    assert harness.recorder.end.turns == 4
    assert [bool(r.tools) for r in harness.provider.requests] == [True, True, True, False]
    assert outcome.final_text == "Edited money.py; I did not get to run the tests."
    nudge = harness.recorder.of(Nudge)[-1]
    assert nudge.reason == "turn_budget" and "no turns left" in nudge.text


async def test_turn_budget_still_ends_when_the_wrap_up_call_fails(make_harness):
    def script(request: ProviderRequest) -> Turn:
        return Turn(error="down") if not request.tools else Turn(tool_calls=[("bash", {"command": "true"})])

    harness = make_harness(script, profile=Profile(max_turns=2))
    outcome = await harness.run()
    assert outcome.reason == "turn_budget" and outcome.final_text == ""


async def test_provider_errors_end_the_run_with_a_typed_reason(make_harness):
    harness = make_harness([Turn(error="HTTP 500 from fake")])
    outcome = await harness.run()
    assert outcome.reason == "provider_error"
    assert "HTTP 500" in (outcome.detail or "")


async def test_abort_stops_a_run_waiting_on_a_slow_tool(make_harness):
    harness = make_harness(lambda _: Turn(tool_calls=[("bash", {"command": "sleep 30"})]))
    control = RunControl()
    task = asyncio.create_task(harness.run(control=control))
    await asyncio.sleep(0.5)
    control.abort.set()
    outcome = await asyncio.wait_for(task, 10)
    assert outcome.reason == "aborted"


async def test_steering_messages_join_before_the_next_model_call(make_harness):
    control = RunControl()

    def script(request: ProviderRequest) -> Turn:
        if len(script_calls) == 0:
            control.steering.append("Also mention the date.")
            script_calls.append(1)
            return Turn(tool_calls=[("bash", {"command": "echo hi"})])
        return Turn(text="done")

    script_calls: list[int] = []
    harness = make_harness(script)
    await harness.run(control=control)
    second = harness.provider.requests[1].messages
    assert any(isinstance(m, UserMessage) and "Also mention the date." in m.content for m in second)


async def test_read_only_calls_run_together_and_results_keep_call_order(make_harness):
    batches = []

    class Capture(h.Extension):
        async def turn_end(self, turn, ctx):
            if turn.results:
                batches.append([result.tool_name for result in turn.results])

    harness = make_harness(
        [
            Turn(
                tool_calls=[
                    ("grep", {"pattern": "x"}),
                    ("find", {"pattern": "*.txt"}),
                    ("read", {"path": "."}),
                ]
            ),
            Turn(text="ok"),
        ],
        extensions=[Capture()],
    )
    await harness.run()
    assert sorted(e.name for e in harness.recorder.of(ToolCallEnd)) == ["find", "grep", "read"]
    recorded = [m.result.tool_name for m in history(harness) if isinstance(m, ToolResultMessage)]
    assert sorted(recorded) == ["find", "grep", "read"]
    assert batches == [["grep", "find", "read"]]


@pytest.mark.parametrize("parallel", [False, True])
async def test_completed_tool_receipt_is_persisted_while_another_call_waits(make_harness, tmp_path, parallel):
    entered, completed = asyncio.Event(), asyncio.Event()

    class Waiting(GreetTool):
        name = "waiting"

        async def run(self, args, ctx):
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    quick: tuple[str, JsonObject] = (
        ("read", {"path": "source"}) if parallel else ("write", {"path": "target", "content": "effect"})
    )
    waiting: tuple[str, JsonObject] = ("waiting", {"who": "later"})
    harness = make_harness([Turn(tool_calls=[waiting, quick] if parallel else [quick, waiting])])
    harness.deps.tools.append(Waiting())
    (harness.cwd / "source").write_text("evidence")
    path = tmp_path / "session.jsonl"
    harness.deps.journal = Journal(cwd=str(harness.cwd), path=path)
    emit = harness.deps.emit

    def observe(event):
        emit(event)
        if isinstance(event, ToolCallEnd) and event.name == quick[0]:
            completed.set()

    harness.deps.emit = observe
    task = asyncio.create_task(harness.run())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.wait_for(completed.wait(), 2)
        saved = Journal(cwd="ignored", path=path)
        results = [
            e.message.result
            for e in saved.entries()
            if isinstance(e, MessageEntry) and isinstance(e.message, ToolResultMessage)
        ]
        assert len(results) == 1 and results[0].tool_name == quick[0] and results[0].status == "ok"
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def test_resume_closes_unknown_tool_outcomes_without_reexecuting(make_harness, tmp_path):
    path = tmp_path / "interrupted.jsonl"
    journal = Journal(cwd=str(tmp_path), path=path)
    journal.append(MessageEntry(message=UserMessage(content="old task")))
    journal.append(
        MessageEntry(
            message=AssistantMessage(
                tool_calls=[
                    ToolCall(id="unknown", name="write", arguments={"path": "must-not-exist", "content": "effect"}),
                ]
            )
        )
    )
    harness = make_harness([Turn(text="check previous effects first")])
    harness.deps.journal = Journal(cwd="ignored", path=path)
    assert (await harness.run("continue carefully")).reason == "completed"
    assert not (harness.cwd / "must-not-exist").exists()
    result = next(m.result for m in harness.provider.requests[0].messages if isinstance(m, ToolResultMessage))
    assert result.call_id == "unknown" and result.error is not None and not result.error.retry_safe
    assert "unknown" in result.content


async def test_initial_journal_write_failure_ends_the_run_without_dispatch(make_harness, tmp_path, monkeypatch):
    import intern.journal as journal_module

    harness = make_harness([Turn(text="must not run")])
    harness.deps.journal = Journal(cwd=str(harness.cwd), path=tmp_path / "failed.jsonl")
    before = harness.deps.journal.entries()

    def failed_sync(fd):
        raise OSError("journal disk failure")

    monkeypatch.setattr(journal_module.os, "fsync", failed_sync)
    outcome = await harness.run()
    assert outcome.reason == harness.recorder.end.reason == "internal_error"
    assert "journal disk failure" in (outcome.detail or "")
    assert not harness.provider.requests and harness.deps.journal.entries() == before


async def test_a_large_window_does_not_let_one_tool_result_fill_it(make_harness):
    # 600K characters on a 1M window: the 0.15 share alone (157K tokens) showed all of it.
    command = "python3 -c \"print('x' * 600_000); print('THE-TAIL')\""
    harness = make_harness([Turn(tool_calls=[("bash", {"command": command})]), Turn(text="ok")], window=1_000_000)
    await harness.run()
    result = next(m for m in history(harness) if isinstance(m, ToolResultMessage)).result
    assert harness.deps.system2.profile.max_tool_result_tokens == 20_000
    # 20K tokens at the meter's chars per token, not the 600K. The fake provider counts messages, not tool schemas, so
    # its ratio grows with the tools: about 6 with `process` among them.
    assert len(result.content) < 150_000
    assert "characters omitted" in result.content and "THE-TAIL" in result.content
    observation = harness.deps.journal.get(result.observation_id or "")
    assert isinstance(observation, ObservationEntry) and "THE-TAIL" in observation.content


async def test_large_tool_output_is_bounded_and_kept_in_full(make_harness):
    harness = make_harness([Turn(tool_calls=[("bash", {"command": "seq 1 50000"})]), Turn(text="ok")], window=16_384)
    await harness.run()
    result = next(m for m in history(harness) if isinstance(m, ToolResultMessage)).result
    assert result.observation_id is not None
    assert "characters omitted" in result.content
    assert "50000" in result.content  # the tail is kept
    observation = harness.deps.journal.get(result.observation_id)
    assert isinstance(observation, ObservationEntry)
    assert observation.content.count("\n") >= 49_999
    spilled = list((harness.deps.spill_dir).glob("*.txt"))
    assert len(spilled) == 1


async def test_agent_can_read_the_missing_middle_of_a_captured_command(make_harness):
    import re
    import shlex
    import sys

    script = 'import sys; sys.stdout.write("A"*800000+"\\nMIDDLE_EVIDENCE\\n"+"B"*800000)'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    def respond(request):
        results = [m.result for m in request.messages if isinstance(m, ToolResultMessage)]
        if not results:
            return Turn(tool_calls=[("bash", {"command": command})])
        if results[-1].tool_name == "bash":
            result = results[-1]
            assert result.omitted_bytes > 0 and result.resources
            path = re.search(r"Complete command output: ([^;]+);", result.content)
            assert path is not None
            return Turn(tool_calls=[("read", {"path": path.group(1), "offset": 2, "limit": 1})])
        assert "MIDDLE_EVIDENCE" in results[-1].content
        return Turn(text="recovered the middle")

    harness = make_harness(respond)
    outcome = await harness.run()
    assert outcome.reason == "completed" and outcome.final_text == "recovered the middle"
    # The capture is in the session's spill folder: the working directory holds only the task's files.
    assert list(harness.cwd.iterdir()) == []
    assert len(list((harness.deps.spill_dir / "command-output").glob("*.txt"))) == 1


async def test_compaction_keeps_the_run_inside_a_small_window(make_harness):
    big = "x" * 6_000

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(text="## Plan\nprint things\n## Now\nkeep printing")
        calls.append(1)
        if len(calls) > 12:
            return Turn(text="finished")
        return Turn(tool_calls=[("bash", {"command": f"printf '{big}'"})])

    calls: list[int] = []
    harness = make_harness(script, window=12_000)
    outcome = await harness.run()
    assert outcome.reason == "completed"
    assert harness.recorder.of(Compaction), "expected at least one compaction"
    entries = [e for e in harness.deps.journal.active_path() if isinstance(e, CompactionEntry)]
    assert entries[0].method == "state", "the State written by the trigger stood; compaction cut to it"
    for request in harness.provider.requests:
        chars = sum(len(str(m.model_dump())) for m in request.messages)
        assert chars // 4 < 12_000
        for i, m in enumerate(request.messages):
            if isinstance(m, ToolResultMessage):
                assert isinstance(request.messages[i - 1], AssistantMessage | ToolResultMessage)
    agent_requests = [r for r in harness.provider.requests if not is_state_write(r)]
    writes = [r for r in harness.provider.requests if is_state_write(r)]
    assert "print things" in agent_requests[-1].messages[1].content
    # The State comes from the System 2 model itself, at low effort, with an output cap that leaves a
    # thinking model room to reason before the text: what the context leaves, not a tenth of the window.
    assert writes
    assert all(r.reasoning_effort == "low" for r in writes)
    assert all(12_000 // 10 < r.max_output_tokens <= loop.MAX_STATE_TOKENS for r in writes)


async def test_replayed_reasoning_counts_toward_compaction_so_the_run_compacts_instead_of_failing(make_harness):
    # Small tool results, long reasoning: the request grows by what the model thought, which it replays.
    thought = "z" * 6_000

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(text="## Goal\nthink\n## Outstanding work\nkeep thinking")
        calls.append(1)
        if len(calls) > 12:
            return Turn(text="finished")
        return Turn(reasoning=thought, tool_calls=[("bash", {"command": "printf ok"})])

    calls: list[int] = []
    harness = make_harness(script, window=12_000)
    outcome = await harness.run()
    assert outcome.reason == "completed", outcome.detail
    assert harness.recorder.of(Compaction)


async def test_compaction_falls_back_to_dropping_when_the_state_write_fails(make_harness):
    big = "y" * 6_000

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(error="down")
        calls.append(1)
        if len(calls) > 8:
            return Turn(text="done")
        return Turn(tool_calls=[("bash", {"command": f"printf '{big}'"})])

    calls: list[int] = []
    harness = make_harness(script, window=12_000)
    outcome = await harness.run()
    assert outcome.reason == "completed"
    compactions = harness.recorder.of(Compaction)
    assert compactions and {e.method for e in compactions} == {"drop"}
    # The user's request comes through every cut, once.
    assert all(c.summary.endswith("\n\n" + requests_section(["do the task"], 10_000)) for c in compactions)
    assert all(c.summary.count("## User requests") == 1 for c in compactions)


class RefuseBash(h.Extension):
    name = "refuse-bash"

    async def tool_call(self, call: ToolCall, ctx: h.RunContext) -> h.ToolCallDecision | None:
        if call.name == "bash":
            return h.Refuse(reason="shell is disabled here", next_action="Use read instead.")
        return None


async def test_tool_call_hook_can_refuse(make_harness):
    harness = make_harness([Turn(tool_calls=[("bash", {"command": "ls"})]), Turn(text="ok")], extensions=[RefuseBash()])
    await harness.run()
    end = harness.recorder.of(ToolCallEnd)[0]
    assert end.status == "refused"
    assert "Use read instead." in end.content


class RewritePath(h.Extension):
    name = "rewrite-path"

    async def tool_call(self, call: ToolCall, ctx: h.RunContext) -> h.ToolCallDecision | None:
        return h.ReplaceArguments(arguments={**call.arguments, "path": "fixed.txt"})


async def test_tool_call_hook_can_repair_arguments(make_harness):
    harness = make_harness(
        [Turn(tool_calls=[("write", {"path": "wrong.txt", "content": "x"})]), Turn(text="ok")],
        extensions=[RewritePath()],
    )
    await harness.run()
    assert (harness.cwd / "fixed.txt").exists()
    assert not (harness.cwd / "wrong.txt").exists()


class Annotate(h.Extension):
    name = "annotate"

    async def tool_result(self, call: ToolCall, result: ToolResult, ctx: h.RunContext) -> ToolResult | None:
        return result.model_copy(update={"content": result.content + "\n[checked]"})


async def test_tool_result_hooks_chain_in_order(make_harness):
    harness = make_harness(
        [Turn(tool_calls=[("bash", {"command": "echo a"})]), Turn(text="ok")], extensions=[Annotate(), Annotate()]
    )
    await harness.run()
    assert harness.recorder.of(ToolCallEnd)[0].content.endswith("[checked]\n[checked]")


async def test_extension_tools_join_the_packs_and_replace_same_name_tools(make_harness):
    harness = make_harness(
        [Turn(tool_calls=[("greet", {"who": "bob"}), ("read", {"path": "a.txt"})]), Turn(text="ok")],
        extensions=[WithTools()],
    )
    (harness.cwd / "a.txt").write_text("quiet\n")
    await harness.run()
    names = [t.name for t in harness.provider.requests[0].tools]
    assert names.count("read") == 1 and names[-1] == "greet"
    assert names.index("read") < names.index("bash")  # the replacement keeps the pack tool's place
    greeted, read = harness.recorder.of(ToolCallEnd)[:2]
    assert (greeted.content, read.content) == ("hello bob", "QUIET")


class RequireTests(h.Extension):
    name = "require-tests"

    def __init__(self) -> None:
        self.asked = False

    async def turn_end(self, turn: h.TurnInfo, ctx: h.RunContext) -> h.Verdict | None:
        if not turn.message.tool_calls and not self.asked:
            self.asked = True
            return h.Nudge(text="Run the tests before finishing.", reason="finish_check")
        return None


async def test_turn_end_nudge_reopens_a_finished_turn(make_harness):
    harness = make_harness([Turn(text="done?"), Turn(text="tests pass, done.")], extensions=[RequireTests()])
    outcome = await harness.run()
    assert outcome.final_text == "tests pass, done."
    assert harness.recorder.of(Nudge)[0].reason == "finish_check"


class StopAfterFirst(h.Extension):
    name = "stop-after-first"

    async def turn_end(self, turn: h.TurnInfo, ctx: h.RunContext) -> h.Verdict | None:
        return h.Stop(reason="stopped_by_hook", detail="one turn only")


async def test_turn_end_stop_ends_the_run(make_harness):
    harness = make_harness(lambda _: Turn(tool_calls=[("bash", {"command": "true"})]), extensions=[StopAfterFirst()])
    outcome = await harness.run()
    assert (outcome.reason, outcome.detail) == ("stopped_by_hook", "one turn only")


def printing(n: int, size: int):
    """A run of n tool turns with `size`-character outputs; the State writer answers with a plan."""

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(text="## Plan\n- print things")
        calls.append(1)
        return (
            Turn(text="done") if len(calls) > n else Turn(tool_calls=[("bash", {"command": f"printf '{'x' * size}'"})])
        )

    calls: list[int] = []
    return script


async def test_the_state_is_written_for_each_compaction_and_compaction_cuts_to_it(make_harness):
    seen = SeesState()
    harness = make_harness(printing(12, 3_000), window=12_000, extensions=[seen], system1=Opinionated())
    outcome = await harness.run()
    assert outcome.reason == "completed"
    events = [e for e in harness.recorder.events if isinstance(e, State | Compaction)]
    # System 2 writes the State for a compaction, just before its cut, and at no other time.
    assert len(events) >= 4 and [type(e) for e in events] == [State, Compaction] * (len(events) // 2)
    states = harness.recorder.of(State)
    assert states[0].text.startswith("## Plan")
    compactions = harness.recorder.of(Compaction)
    assert all(c.method == "state" for c in compactions), "every cut is to the State written for it"
    # One State write per State event: compaction itself asked the model for nothing.
    writes = [r for r in harness.provider.requests if is_state_write(r)]
    assert len(writes) == len(states)
    # System 1 reads the session, not the State: after the cuts it still sees the turns they took away.
    assert '"id":"t1"' in seen.states[-1] and "## Plan" not in seen.states[-1]
    # The compaction kept whole turns, in order.
    for request in harness.provider.requests:
        for i, m in enumerate(request.messages):
            if isinstance(m, ToolResultMessage):
                assert isinstance(request.messages[i - 1], AssistantMessage | ToolResultMessage)


async def test_the_state_write_is_the_agents_own_request_with_the_state_asked_for(make_harness):
    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(text="## Plan\n- print things")
        calls.append(1)
        if len(calls) > 8:
            return Turn(text="done")
        return Turn(
            reasoning=f"step {len(calls)}: print more", tool_calls=[("bash", {"command": f"printf '{'x' * 3_000}'"})]
        )

    calls: list[int] = []
    harness = make_harness(script, window=12_000, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    requests = harness.provider.requests
    at = next(i for i, r in enumerate(requests) if is_state_write(r))
    write, agent = requests[at], requests[at - 1]
    # The agent's last request and the turn since (its reply and the tool's result), reasoning included: the same
    # system prompt, tools and turns (an old tool output may be shown shorter as it ages); the State request is the
    # only addition.
    assert not is_state_write(agent) and write.tools == agent.tools and write.messages[0] == agent.messages[0]
    assert [m.role for m in write.messages[: len(agent.messages)]] == [m.role for m in agent.messages]
    assert len(write.messages) == len(agent.messages) + 3
    assert write.reasoning_replay == agent.reasoning_replay
    assert any(isinstance(m, AssistantMessage) and m.reasoning for m in write.messages)


async def test_a_state_write_announces_itself_and_streams_its_text_before_the_state_lands(make_harness):
    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(text="## Plan\n- print things\n## Now\nprinting")
        calls.append(1)
        if len(calls) > 8:
            return Turn(text="done")
        return Turn(tool_calls=[("bash", {"command": f"printf '{'x' * 3_000}'"})])

    calls: list[int] = []
    harness = make_harness(script, window=12_000, system1=Opinionated())
    assert (await harness.run()).reason == "completed"
    events = [e for e in harness.recorder.events if isinstance(e, StateWriting | StateDelta | State)]
    states = harness.recorder.of(State)
    assert states, "System 1 had a State written"
    for state in states:
        before = events[: events.index(state)]
        start = max(i for i, e in enumerate(before) if isinstance(e, StateWriting))
        writing = before[start]
        assert isinstance(writing, StateWriting) and writing.turn == state.turn
        assert writing.turns_covered == state.turns_covered
        deltas = [e for e in before[start:] if isinstance(e, StateDelta)]
        if state.method == "edit":  # edits are not the State's text: nothing streams
            assert not deltas
            continue
        assert deltas and all(d.turn == state.turn for d in deltas)
        # What streams is the writer's; the user's requests, which code keeps, land after it.
        written, asked = state.text.split("\n\n## User requests")
        assert " ".join("".join(d.text for d in deltas).split()) == " ".join(written.split())
        assert f"## User requests{asked}" == requests_section(["do the task"], 10_000)
    # One announcement per write, the text between, the State after: nothing else in that stretch.
    kinds = [type(e).__name__ for e in events if not isinstance(e, StateDelta)]
    assert kinds == ["StateWriting", "State"] * len(states)


async def test_state_and_number_repairs_keep_extension_boundaries_without_mutating_actor_view(make_harness):
    hook_calls = []

    class Boundary(h.Extension):
        async def context(self, view, ctx):
            view.system = view.system.replace("PRIVATE", "filtered")
            view.pinned.append("extension constraint")
            message = next(m for m in view.history if isinstance(m, AssistantMessage))
            message.text = "filtered history"

        async def before_provider_request(self, request, ctx):
            hook_calls.append(request)
            return request.model_copy(update={"tools": [s for s in request.tools if s.name != "write"]})

    harness = make_harness(
        [
            Turn(text="## Plan\nkeep value 999"),
            Turn(tool_calls=[("edit", {"path": "STATE.md", "old_string": "999", "new_string": "unknown"})]),
        ],
        extensions=[Boundary()],
    )
    harness.deps.instructions = harness.deps.instructions.model_copy(update={"system": "PRIVATE system"})
    harness.deps.journal.append(MessageEntry(message=UserMessage(content="task")))
    harness.deps.journal.append(MessageEntry(message=AssistantMessage(text="PRIVATE history")))
    run = loop.Run(harness.deps, harness.meter, RunControl())
    view = run._view()
    before = [m.model_dump() for m in view.history]
    write = run._update_state(view)
    assert write is not None
    await write
    assert harness.recorder.of(State)[-1].text.startswith("## Plan\nkeep value unknown")
    assert len(hook_calls) == len(harness.provider.requests) == 2
    for request in harness.provider.requests:
        assert "PRIVATE" not in str(request.messages)
        assert "extension constraint" in str(request.messages)
        assert "write" not in {spec.name for spec in request.tools}
    assert view.system == "PRIVATE system" and "extension constraint" not in view.pinned
    assert [m.model_dump() for m in view.history] == before


async def test_a_state_write_that_calls_a_tool_is_not_a_state_and_runs_nothing(make_harness):
    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(tool_calls=[("write", {"path": "stray.txt", "content": "no"})])
        calls.append(1)
        return (
            Turn(text="done") if len(calls) > 8 else Turn(tool_calls=[("bash", {"command": f"printf '{'x' * 3_000}'"})])
        )

    calls: list[int] = []
    harness = make_harness(script, window=12_000)
    assert (await harness.run()).reason == "completed"
    states = harness.recorder.of(State)
    assert states and all(s.dropped == "write_failed" for s in states)
    assert not (harness.cwd / "stray.txt").exists()
    assert all(e.name == "bash" for e in harness.recorder.of(ToolCallEnd))


async def test_a_state_number_the_run_never_showed_is_fixed_by_line_edits_and_what_is_left_is_blanked(make_harness):
    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            last = request.messages[-1]
            if isinstance(last, UserMessage) and "4,233,903" in last.content:  # the fix, in the same fork
                fixes.append(request)
                return Turn(
                    tool_calls=[
                        ("write", {"path": STATE_FILE, "content": "## Plan\n- all new"}),  # a rewrite: refused
                        ("edit", {"path": STATE_FILE, "old_string": "## Plan", "new_string": "## Goal"}),  # no number
                        (
                            "edit",
                            {"path": STATE_FILE, "old_string": "- sales 4,233,903", "new_string": "- sales 4,233,003"},
                        ),
                    ]
                )
            if isinstance(last, UserMessage) and "Still not" in last.content:
                return Turn(text="done")
            # The State miscopies 4,233,003 and gives a sum of its own.
            return Turn(text="## Plan\n- print: done\n## Facts\n- sales 4,233,903\n- total 777")
        calls.append(1)
        return Turn(text="done") if len(calls) > 8 else Turn(tool_calls=[("bash", {"command": cmd})])

    cmd = f"printf 'sales 4,233,003 {'x' * 3_000}'"
    calls: list[int] = []
    fixes: list[ProviderRequest] = []
    harness = make_harness(script, window=12_000)
    assert (await harness.run()).reason == "completed"
    state = harness.recorder.of(State)[0]
    assert state.unseen_numbers == ["4,233,903", "777"] and state.blanked == ["777"]
    # Only the line with the miscopied number changed; the rewrite and the edit of a line without one did not.
    # The user's request, which code adds after the check, is no number of the writer's to check.
    written = "## Plan\n- print: done\n## Facts\n- sales 4,233,003\n- total ?"
    assert state.text == f"{written}\n\n{requests_section(['do the task'], 10_000)}"
    # The fix was asked in the same fork, after the State the writer gave.
    fix = fixes[0].messages
    assert isinstance(fix[-2], AssistantMessage) and fix[-2].text.endswith("- total 777")
    assert not (harness.cwd / STATE_FILE).exists(), "the State is not on disk"
    assert state.usage is not None and state.usage.input_tokens > 0


async def test_compaction_cuts_what_the_plan_wants_not_everything_the_state_covers(make_harness):
    # keep_recent (35% of 12k) stays raw after the cut, even though the State was written a turn ago and
    # covers most of it; the cut is free (no State write) because the State covers the cut turns.
    harness = make_harness(printing(12, 3_000), window=12_000)
    await harness.run()
    first = harness.recorder.of(Compaction)[0]
    assert first.method == "state"
    assert first.tokens_after >= 0.3 * 12_000, "the recent turns were kept"
    states_before = [s for s in harness.recorder.of(State) if s.turn <= first.turn]
    assert sum(s.turns_covered for s in states_before) > first.turns_compacted, "the State covered more than was cut"


async def test_without_system1_compaction_writes_the_state_and_cuts_to_it(make_harness):
    harness = make_harness(printing(12, 3_000), window=12_000)
    await harness.run()
    states, compactions = harness.recorder.of(State), harness.recorder.of(Compaction)
    assert compactions and all(c.method == "state" for c in compactions)
    assert states and all(s.landed == s.turn for s in states)
    # The write covered everything the agent saw, so the turns kept raw need no second write right after.
    assert len(states) == len(compactions)
    assert all(c.summary.startswith("## Plan") for c in compactions)
    assert Session(harness.deps).state().state == compactions[-1].summary


async def test_hosts_and_event_listeners_see_the_latest_state(make_harness):
    harness = make_harness(printing(12, 3_000), window=12_000)
    session = Session(harness.deps)
    assert session.state().state is None
    await harness.run()
    states, first = harness.recorder.of(State), harness.recorder.of(Compaction)[0]
    assert first.method == "state"
    assert first.summary == [s.text for s in states if s.turn <= first.turn][-1], "the cut was to the standing State"
    assert session.state().state == states[-1].text


class SeesState(h.Extension):
    name = "sees-state"

    def __init__(self) -> None:
        self.states: list[str] = []

    async def turn_end(self, turn: h.TurnInfo, ctx: h.RunContext) -> h.Verdict | None:
        self.states.append(ctx.state())
        return None


async def test_ctx_state_is_the_session_in_order_with_notes_for_tool_outputs(make_harness):
    seen = SeesState()
    harness = make_harness([Turn(tool_calls=[("bash", {"command": "echo hi"})]), Turn(text="ok")], extensions=[seen])
    await harness.run()
    state = json.loads(seen.states[0])
    assert state["goal"] == "do the task" and state["history"][0] == {"i": 0, "role": "user", "text": "do the task"}
    (call,) = state["history"][1]["tool_calls"]
    assert (call["tool"], call["input"]) == ("bash", '{"command":"echo hi"}')
    assert call["result"].startswith("ok, ") and " chars (omitted): starts " in call["result"]
    assert json.loads(seen.states[1])["history"][-1] == {"i": 3, "role": "assistant", "text": "ok"}


class Broken(h.Extension):
    name = "broken"

    async def context(self, view, ctx: h.RunContext) -> None:
        raise RuntimeError("boom")


async def test_hooks_fail_open(make_harness):
    harness = make_harness([Turn(text="fine")], extensions=[Broken()])
    outcome = await harness.run()
    assert outcome.reason == "completed"
    error = harness.recorder.of(HookError)[0]
    assert (error.hook, error.extension) == ("context", "broken")


class FinishGate(h.Extension):
    name = "finish-gate"

    def __init__(self) -> None:
        self.calls = 0

    async def agent_end(self, outcome: h.RunOutcome, ctx: h.RunContext) -> h.Nudge | None:
        self.calls += 1
        return h.Nudge(text="Not yet: check the output file.", reason="finish_gate")


async def test_agent_end_gate_reopens_a_bounded_number_of_times(make_harness):
    gate = FinishGate()
    harness = make_harness([Turn(text="done")] * 5, extensions=[gate])
    outcome = await harness.run()
    assert outcome.reason == "completed"
    assert len(harness.provider.requests) == 3
    assert gate.calls == 3


async def test_every_message_is_journaled_and_announced(make_harness):
    harness = make_harness([Turn(tool_calls=[("bash", {"command": "echo hi"})]), Turn(text="ok")])
    await harness.run()
    roles = [e.message.role for e in harness.recorder.of(MessageEvent)]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert [m.role for m in history(harness)] == ["user", "assistant", "tool", "assistant"]


async def test_a_second_prompt_continues_the_same_conversation(make_harness):
    harness = make_harness([Turn(text="first"), Turn(text="second")])
    await harness.run("one")
    await harness.run("two")
    messages = harness.provider.requests[-1].messages
    assert any(isinstance(m, AssistantMessage) and m.text == "first" for m in messages)
    assert isinstance(messages[-1], UserMessage) and messages[-1].content == "two"


async def test_steering_during_the_final_reply_is_answered(make_harness):
    control = RunControl()
    replies: list[str] = []

    def script(request: ProviderRequest) -> Turn:
        replies.append("x")
        if len(replies) == 1:
            control.steering.append("One more thing: say bye.")
            return Turn(text="Done.")
        return Turn(text="Bye.")

    harness = make_harness(script)
    outcome = await harness.run(control=control)
    assert outcome.final_text == "Bye."
    last = harness.provider.requests[-1].messages
    assert isinstance(last[-1], UserMessage) and last[-1].content == "One more thing: say bye."


async def test_abort_skips_the_remaining_calls_of_the_same_reply(make_harness):
    harness = make_harness(
        [Turn(tool_calls=[("bash", {"command": "sleep 30"}), ("write", {"path": "late.txt", "content": "x"})])]
    )
    control = RunControl()
    task = asyncio.create_task(harness.run(control=control))
    await asyncio.sleep(0.5)
    control.abort.set()
    outcome = await asyncio.wait_for(task, 10)
    assert outcome.reason == "aborted"
    assert not (harness.cwd / "late.txt").exists()
    assert [e.status for e in harness.recorder.of(ToolCallEnd)] == ["error", "refused"]


class SlowStateWrites(FakeProvider):
    """A provider whose State writes take real time, so a loop that did not wait for them would overtake them."""

    def __init__(self, script, delay_s: float = 0.05) -> None:
        super().__init__(script)
        self.delay_s = delay_s
        self.order: list[str] = []  # "state:start", "agent", "state:done" in wall-clock order
        self.writing = asyncio.Event()  # set when the first write starts

    async def stream(self, request: ProviderRequest):
        write = is_state_write(request)
        self.order.append("state:start" if write else "agent")
        if write:
            self.writing.set()
            await asyncio.sleep(self.delay_s)
        async for item in super().stream(request):
            yield item
        if write:
            self.order.append("state:done")


async def test_compaction_waits_for_the_state_write_and_the_agent_never_sees_it(make_harness):
    harness = make_harness(printing(12, 3_000), window=12_000)
    harness.provider = harness.deps.system2.provider = SlowStateWrites(printing(12, 3_000))  # type: ignore[attr-defined]
    outcome = await harness.run()
    assert outcome.reason == "completed"
    states = harness.recorder.of(State)
    assert states and harness.recorder.of(Compaction)
    order = harness.provider.order  # type: ignore[attr-defined]
    starts = [i for i, o in enumerate(order) if o == "state:start"]
    assert all(order[i + 1] == "state:done" for i in starts), "no request went out while a write was in flight"
    # Every write landed at the turn that asked for it and none was dropped.
    assert len(starts) == len(states) and all(s.landed == s.turn and s.dropped is None for s in states)
    # System 2's prompt did not change because of a write: no State text before a compaction cut to it.
    agent = [r for r in harness.provider.requests if not is_state_write(r)]
    cut = next(
        (
            i
            for i, r in enumerate(agent)
            if any(isinstance(m, UserMessage) and "earlier turns were compacted" in m.content for m in r.messages)
        ),
        len(agent),
    )
    assert not any("## Plan" in m.content for r in agent[:cut] for m in r.messages if isinstance(m, UserMessage))


async def test_abort_stops_a_run_waiting_on_its_compactions_state_write(make_harness):
    # Without System 1 the State is written when a compaction needs it, and the run waits for it: Esc stops it.
    harness = make_harness(printing(12, 3_000), window=12_000)
    harness.provider = harness.deps.system2.provider = SlowStateWrites(printing(12, 3_000), delay_s=30)  # type: ignore[attr-defined]
    control = RunControl()
    task = asyncio.create_task(harness.run(control=control))
    await asyncio.wait_for(harness.provider.writing.wait(), 10)  # type: ignore[attr-defined]
    control.abort.set()
    outcome = await asyncio.wait_for(task, 5)
    assert outcome.reason == "aborted"
    assert len(harness.recorder.of(StateWriting)) == 1
    assert "state:done" not in harness.provider.order  # type: ignore[attr-defined]


class Opinionated:
    """A System 1 that always finds the step complete and every read worked on: forks and kept reads happen."""

    name = "opinionated"

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        answers: dict[str, Answer] = {}
        for name, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                first = next(iter(q.criteria))
                answers[name] = ChoiceAnswer(choice=first, probabilities={c: float(c == first) for c in q.criteria})
            elif isinstance(q, NoulQuestion):
                answers[name] = NoulAnswer(noul=0.9)
        return Answers(answers=answers, cost_usd=0.001, input_tokens=50, output_tokens=5)


def reading(n: int):
    """A run of n `read` turns over big.txt (so compactions cut file reads, which `keep` asks are about)."""

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(text="## Plan\n- read things")
        calls.append(1)
        return Turn(text="done") if len(calls) > n else Turn(tool_calls=[("read", {"path": "big.txt"})])

    calls: list[int] = []
    return script


async def test_every_state_write_is_accounted_for_in_the_stream_and_decisions_name_their_turn(make_harness):
    harness = make_harness(reading(14), window=12_000, system1=Opinionated())
    (harness.cwd / "big.txt").write_text("".join(f"line {i} {'x' * 60}\n" for i in range(40)))
    outcome = await harness.run()
    assert outcome.reason == "completed"
    end = harness.recorder.end
    states, compactions = harness.recorder.of(State), harness.recorder.of(Compaction)
    assert compactions and states
    # The agent's own replies plus every State write (landed, failed, or made for a compaction) add up to
    # the run's usage, so a listener can attribute cost while the run goes on; no call is hidden in the total.
    replies = sum((m.usage for m in history(harness) if isinstance(m, AssistantMessage)), Usage())
    writes = sum((e.usage for e in states if e.usage is not None), Usage())
    assert replies + writes == end.usage
    assert end.agent_usage == replies and end.state_usage == writes and end.agent_usage + end.state_usage == end.usage
    for e in states:
        if e.usage is not None:
            assert e.duration_ms is not None and e.model == "fake/model"
    # Every System 1 ask says which turn it belongs to and what it answered; keep asks name the read.
    decisions = harness.recorder.of(Decision)
    assert decisions and all(d.turn is not None and d.answers for d in decisions)
    supervise = [d for d in decisions if d.purpose == "supervise"]
    # One question a turn, the turns' mode: System 2 writes the State only for a compaction.
    assert supervise and all(set(d.answers or {}) == {"mode"} for d in supervise)
    keeps = [d for d in decisions if d.purpose == "keep"]
    assert keeps and all(d.subject and (d.answers or {})["w"]["choice"] == "working_on" for d in keeps)
    assert {d.turn for d in keeps} <= {c.turn for c in compactions}
    assert {s.turn for s in states} <= {c.turn for c in compactions}  # the State is written only for a compaction


async def test_a_state_write_that_fails_is_reported_with_its_reason(make_harness):
    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(error="down")
        calls.append(1)
        return (
            Turn(text="done")
            if len(calls) > 10
            else Turn(tool_calls=[("bash", {"command": "printf '%s'" % ("y" * 4_000)})])
        )

    calls: list[int] = []
    harness = make_harness(script, window=12_000)
    assert (await harness.run()).reason == "completed"
    failed = [s for s in harness.recorder.of(State) if s.dropped == "write_failed"]
    assert failed and all(s.landed == 0 and s.text == "" and s.usage == Usage() for s in failed)
