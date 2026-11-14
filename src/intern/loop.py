"""The agent loop: turn phases that each end in a verdict (continue, nudge, or stop with a reason).

A turn is: take steering messages, build the request (compacting at the boundary if needed), call the
model, record the reply, run its tool calls, then decide. Everything product-specific is a hook; the
loop only owns what every model needs: the turn budget, continuation after an output-limit cut-off,
keeping unparsable tool calls out of history, window-scaled tool output, and compaction.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import traceback
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, NamedTuple

from intern import hooks as h
from intern.async_tasks import join_tasks
from intern.context import (
    INITIAL_CHARS_PER_TOKEN,
    CompactionPlan,
    ContextView,
    StateEdit,
    TokenMeter,
    bound_output,
    build_view,
    dropped_turns,
    estimate_tokens,
    message_chars,
    plan_compaction,
    reasoning_chars,
    squeeze,
    state_edit_request,
    state_request,
    tools_chars,
    uncovered_tokens,
    with_requests,
    without_code_sections,
    write_state,
)
from intern.decisions import DecisionClient, System1Unavailable
from intern.events import (
    Compaction,
    ContextBudget,
    EventSink,
    MessageEvent,
    Nudge,
    ProcessInfo,
    ReasoningDelta,
    RunEnd,
    RunStart,
    State,
    StateDelta,
    StateWriting,
    StopReason,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnStart,
    WindowInfo,
)
from intern.evidence import keep_reads, reduce_output
from intern.executor import Executor, mount_of
from intern.guard import NEXT_ACTION, Guard, Scope
from intern.instructions import SYSTEM1_LABEL, SYSTEM1_SHARE, Instructions
from intern.journal import (
    CompactionEntry,
    Journal,
    MessageEntry,
    ObservationEntry,
    StateEntry,
    SystemPromptEntry,
    new_id,
)
from intern.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolError,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from intern.models.profile import Profile, effort_choices, effort_levels
from intern.processes import ProcessManager
from intern.providers.base import (
    Completed,
    Provider,
    ProviderError,
    ProviderRequest,
    ReasoningChunk,
    TextChunk,
    ToolSpec,
)
from intern.session_cut import STATE_SHARE, session_cut
from intern.state_edits import parse as parse_state
from intern.supervise import Supervisor
from intern.tools.base import Tool, ToolContext, ToolOutput, fail
from intern.tools.files import EditTool

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")
MIN_OUTPUT_TOKENS = 1_024
MAX_STATE_TOKENS = 12_288  # the State and the thinking that writes it: Qwen3.8 at low effort, up to 4K + 4K
# A thinking model left at its default effort spends the whole State budget reasoning and returns no
# text (Qwen3.8: 2,048 tokens of thought, empty State, so every compaction fell back to "drop").
STATE_EFFORT = "low"
MAX_FINISH_REOPENS = 2
# A reply that fails partway with a retryable error is requested again; nothing of it was recorded.
REPLY_RETRY_WAITS_S = (5.0, 20.0)

CUT_OFF_NUDGE = "Your reply was cut off at the output limit. Continue exactly where it stopped."
BUDGET_NUDGE = (
    "You have no turns left and cannot call tools any more. Reply to the user now: say what is done, what is "
    "not done, and what they should check or do themselves. Do not claim anything you did not do."
)


class StateCall(NamedTuple):
    """A State write: its text (None when the call failed, called a tool or returned nothing) and its cost."""

    text: str | None
    usage: Usage
    duration_ms: int
    model: str
    unseen_numbers: list[str]  # numbers the State gave that the conversation never showed
    blanked: list[str]  # those still there after the fix, replaced by `?`
    method: Literal["edit", "rewrite"] = "rewrite"
    fallback: str | None = None


@dataclass
class ModelBinding:
    """A model in a role: how to reach it, its profile and its working window."""

    provider: Provider
    ref: str
    wire_model: str
    profile: Profile
    window: WindowInfo

    @property
    def effort_levels(self) -> tuple[str, ...]:
        """The reasoning efforts that can be set, least to most (see `effort_levels`)."""
        return effort_levels(getattr(self.provider, "effort_levels", ()), self.profile)

    @property
    def effort_choices(self) -> tuple[str, ...]:
        """What the thought level can be set to (see `effort_choices`)."""
        return effort_choices(getattr(self.provider, "effort_levels", ()), self.profile)

    @property
    def takes_effort(self) -> bool:
        """Whether a reasoning effort can be set at all."""
        return bool(self.effort_levels)


@dataclass
class EngineDeps:
    """The seams a run reaches everything through."""

    system2: ModelBinding
    tools: list[Tool]
    hooks: h.Hooks
    journal: Journal
    instructions: Instructions
    decisions: DecisionClient
    executor: Executor
    cwd: Path
    spill_dir: Path
    emit: EventSink
    redact: Callable[[str], str] | None = None
    """Replaces every stored secret's value with `[REDACTED]`; applied to each tool's output before the model,
    the journal, clients or a spill file see it, and to a run's end detail. None when no secret is stored."""
    system_writes: bool = False
    processes: ProcessManager | None = None
    """The session's job manager (`processes.py`): the `process` tool's jobs, and a record of every command running in
    the session. None when the executor runs no background jobs, or for a session built without setup."""


CompactOutcome = Literal["compacted", "fits", "stopped"]


def system1_state(deps: EngineDeps) -> str:
    """What System 1 is asked about: the session as it stands, cut to its share of System 1's window, with what runs
    now beside the goal, then the user's SYSTEM1.md when they keep one."""
    max_tokens = deps.decisions.max_state_tokens
    running = deps.processes.for_system1() if deps.processes is not None else ()
    state = session_cut(deps.journal.active_path(), int(max_tokens * STATE_SHARE), processes=running)
    if not deps.instructions.system1:
        return state
    text, _ = bound_output(deps.instructions.system1, int(max_tokens * SYSTEM1_SHARE * INITIAL_CHARS_PER_TOKEN))
    return f"{state}\n\n{SYSTEM1_LABEL}{text}"


@dataclass
class RunControl:
    """How a client steers or stops a run that is in progress."""

    abort: asyncio.Event = field(default_factory=asyncio.Event)
    steering: list[str] = field(default_factory=list)
    ended: list[ProcessInfo] = field(default_factory=list)
    """Processes that ended since the last turn began (the session's job manager reports them): the agent is told at
    the next turn, of those it has not seen end, and the run does not end while one is untold."""
    checks: list[str] = field(default_factory=list)
    """What System 1 found of the commands running (`process_watch`): told at the next turn, like `ended`."""
    failure: System1Unavailable | None = None
    """An ask the session made for this run that System 1 did not answer: the run stops at the next turn with it."""
    hold: bool = True
    """Whether a final answer waits for a command the run moved to the background: for a session that tells
    the agent nothing between runs (`quiet`: -p, ACP, JSON-RPC). One that wakes it (the terminal UI) ends the run."""
    news: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when something for the agent arrives (steering, a process's end, a check): a run held for a command it
    moved to the background takes its next turn."""
    waiting: bool = False
    """The agent handed over (a turn that only polled running jobs) and the run holds for news: each of System 1's
    looks at what runs is news then (`looks`), so a job that never ends, a server, cannot hold the run forever."""
    looks: list[str] = field(default_factory=list)
    """System 1's routine looks while the agent waits, told at the next turn."""


class Run:
    """One run of the loop, from a prompt to a stop reason. Created per prompt; holds only per-run state."""

    def __init__(self, deps: EngineDeps, meter: TokenMeter, control: RunControl) -> None:
        self.deps = deps
        self.meter = meter
        self.control = control
        self.run_id = new_id()
        self.turn = 0
        self.usage = Usage()
        self.state_usage = Usage()  # the State writes' share of `usage`
        self.continuations = 0
        self.final_text = ""
        self._prompt_chars = 0
        self._pending_stop: h.Stop | None = None
        self._answered = False
        """The agent has written to the user since the user's latest message."""
        self._urged = False
        """The agent was told, since the user's latest message, to answer what does not depend on a job first."""
        self._waiting_on: list[str] = []
        """The commands the agent handed over on at the last turn, in a quiet session: the run holds before the next."""
        self.supervisor = Supervisor()
        self.guard = Guard(
            Scope.of(deps.cwd, deps.executor, Path.home(), system_writes=deps.system_writes),
            deps.executor,
            deps.cwd,
            deps.decisions,
            deps.emit,
            self._system1_state,
            guided=bool(deps.instructions.system1),
        )

    # -- helpers --------------------------------------------------------------------------------

    @property
    def binding(self) -> ModelBinding:
        return self.deps.system2

    @property
    def window(self) -> int:
        return self.binding.window.working

    def _ctx(self, turn: int | None = None) -> h.RunContext:
        return h.RunContext(
            run_id=self.run_id,
            turn=self.turn if turn is None else turn,
            cwd=str(self.deps.cwd),
            profile=self.binding.profile,
            working_window=self.window,
            chars_per_token=self.meter.chars_per_token,
            journal=self.deps.journal,
            decisions=self.deps.decisions,
            emit=self.deps.emit,
            state=self._system1_state,
        )

    def _system1_state(self) -> str:
        return system1_state(self.deps)

    def _record(self, message: Message) -> str:
        if isinstance(message, UserMessage) and message.source == "user":
            self._answered = self._urged = False  # a new message from the user: the agent owes it a reply
        elif isinstance(message, AssistantMessage) and message.text.strip():
            self._answered = True
        entry = self.deps.journal.append(MessageEntry(message=message, run_id=self.run_id))
        self.deps.emit(MessageEvent(entry_id=entry.id, message=message))
        return entry.id

    def _nudge(self, text: str, reason: str) -> None:
        self.deps.emit(Nudge(turn=self.turn, reason=reason, text=text))
        self._record(UserMessage(content=text, source="harness"))

    def _untold(self) -> list[ProcessInfo] | list[str]:
        """The processes that ended since the last turn whose end the agent has not been shown, else what System 1
        found of the commands running and the agent has not been told."""
        jobs = self.deps.processes
        untold = jobs.unseen(self.control.ended) if jobs is not None and self.control.ended else []
        return untold or self.control.checks

    def _tell_ended(self) -> None:
        """Tell the agent which of its processes ended since the last turn, before it is asked for the next one, so it
        never waits on one with `sleep` or a poll loop."""
        jobs, ended = self.deps.processes, self.control.ended[:]
        self.control.ended.clear()
        if jobs is not None and ended and (text := jobs.end_notice(ended)) is not None:
            self._nudge(self.deps.redact(text) if self.deps.redact else text, "process_ended")
        checks = self.control.checks[:]
        self.control.checks.clear()
        for text in checks:
            self._nudge(self.deps.redact(text) if self.deps.redact else text, "process_check")
        looks = self.control.looks[-1:]  # the latest look is enough
        self.control.looks.clear()
        for text in looks:
            self._nudge(self.deps.redact(text) if self.deps.redact else text, "process_look")

    def _moved_running(self) -> bool:
        return self.control.hold and self.deps.processes is not None and bool(self.deps.processes.held_running())

    async def _hold(self, *, handed_over: bool = False) -> None:
        """Wait for news for the agent (a moved command's end among it) or an abort, the run kept open meanwhile. After
        a hand-over, System 1's next look at what runs is news too."""
        if self.control.news.is_set() or self.control.abort.is_set():
            return
        self.control.waiting = handed_over
        waits = [asyncio.create_task(self.control.news.wait()), asyncio.create_task(self.control.abort.wait())]
        try:
            await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self.control.waiting = False
            await join_tasks(*waits, cancel=True)

    def _handed_over(self, calls: list[ToolCall]) -> list[str]:
        """The commands the agent waits on, when its turn only polled (or listed) jobs that still run: waiting is the
        harness's, not the agent's. Empty for any other turn, or when a polled job has ended (the agent has
        its result to act on)."""
        jobs = self.deps.processes
        if jobs is None or not calls or any(c.name != "process" for c in calls):
            return []
        actions = [c.arguments.get("action") for c in calls]
        if any(action not in ("poll", "list") for action in actions):
            return []
        running = {info.id: info.command for info in jobs.jobs_running()}
        polled = [str(c.arguments.get("handle", "")) for c in calls if c.arguments.get("action") == "poll"]
        if not polled:
            return list(running.values())
        waited = [running[handle] for handle in polled if handle in running]
        return waited if len(waited) == len(polled) else []

    def _requests(self) -> list[str]:
        """What the user asked in this session: their messages only, never the harness's or the model's."""
        return [
            entry.message.content
            for entry in self.deps.journal.active_path()
            if isinstance(entry, MessageEntry)
            and isinstance(entry.message, UserMessage)
            and entry.message.source == "user"
        ]

    def _with_requests(self, state: str, requests: list[str]) -> str:
        """`state` with the commands running now and the user's latest requests as its last sections, the requests
        within the profile's share of the window."""
        limit = int(self.meter.chars_per_token * self.window * self.binding.profile.thresholds.requests)
        running = self.deps.processes.running() if self.deps.processes is not None else []
        return with_requests(state, requests, limit, running)

    def _tool_result_chars(self) -> int:
        """How much of one tool result the model is shown: its share of the window, at most the profile's cap."""
        profile = self.binding.profile
        tokens = self.window * profile.thresholds.tool_result
        if profile.max_tool_result_tokens is not None:
            tokens = min(tokens, profile.max_tool_result_tokens)
        return int(self.meter.chars_per_token * tokens)

    def _tool_specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self.deps.tools]

    # -- run ------------------------------------------------------------------------------------

    async def execute(self, prompt: str | None) -> h.RunOutcome:
        """Run from `prompt`, the user's message; None for a run the session starts itself to tell the agent of
        processes that ended while it was idle (`RunControl.ended`)."""
        started = time.monotonic()
        system1_before = self.deps.decisions.usage.model_copy()
        self.deps.emit(
            RunStart(
                session_id=self.deps.journal.session_id,
                run_id=self.run_id,
                model=self.binding.ref,
                cwd=str(self.deps.cwd),
                window=self.binding.window,
                tools=[t.name for t in self.deps.tools],
            )
        )
        try:
            self._record_system_prompt()
            self._recover_unfinished_calls()
            if prompt is not None:
                self._record(UserMessage(content=prompt))
            outcome = await self._run_with_finish_gate()
        except System1Unavailable as e:
            outcome = self._outcome("system1_unavailable", str(e))
        except Exception as e:
            # A bug must still end the run with a typed reason, so clients and evals never hang.
            traceback.print_exc(file=sys.stderr)
            outcome = self._outcome("internal_error", f"{type(e).__name__}: {e}")

        self.deps.emit(
            RunEnd(
                run_id=self.run_id,
                reason=outcome.reason,
                detail=self.deps.redact(outcome.detail) if self.deps.redact and outcome.detail else outcome.detail,
                turns=self.turn,
                usage=self.usage,
                agent_usage=self.usage - self.state_usage,
                state_usage=self.state_usage,
                system1=self.deps.decisions.usage.since(system1_before),
                final_text=outcome.final_text,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        )
        return outcome

    async def _run_with_finish_gate(self) -> h.RunOutcome:
        await self.deps.hooks.call("before_agent_start", lambda e: e.before_agent_start(self._ctx()))
        outcome = await self._loop()
        reopens = 0
        while True:
            gates = await self.deps.hooks.call("agent_end", lambda e, o=outcome: e.agent_end(o, self._ctx()))
            gate = next((verdict for _, verdict in gates if isinstance(verdict, h.Nudge)), None)
            if gate is None or outcome.reason != "completed" or reopens >= MAX_FINISH_REOPENS:
                return outcome
            reopens += 1
            self._nudge(gate.text, gate.reason)
            outcome = await self._loop()

    def _record_system_prompt(self) -> None:
        system = self.deps.instructions.system
        tools = [t.name for t in self.deps.tools]
        last = next((e for e in reversed(self.deps.journal.active_path()) if isinstance(e, SystemPromptEntry)), None)
        if last is None or last.content != system or last.tools != tools:
            last = self.deps.journal.append(SystemPromptEntry(content=system, tools=tools))
        self.deps.emit(MessageEvent(entry_id=last.id, message=SystemMessage(content=system)))

    def _recover_unfinished_calls(self) -> None:
        """Close interrupted protocol turns without executing or guessing the outcome of a tool."""
        pending: dict[str, ToolCall] = {}
        for entry in self.deps.journal.active_path():
            if not isinstance(entry, MessageEntry):
                continue
            message = entry.message
            if isinstance(message, AssistantMessage):
                pending.update((call.id, call) for call in message.tool_calls)
            elif isinstance(message, ToolResultMessage):
                pending.pop(message.result.call_id, None)
        for call in pending.values():
            error = ToolError(
                what_failed="The previous run ended without a persisted result for this call; its outcome is unknown.",
                next_action="Check files or remote state before retrying. The harness has not re-executed this call.",
                retry_safe=False,
            )
            self._record(
                ToolResultMessage(
                    result=ToolResult(
                        call_id=call.id,
                        tool_name=call.name,
                        status="error",
                        error=error,
                        content=f"Error: {error.what_failed}\nNext: {error.next_action}",
                    )
                )
            )

    def _outcome(self, reason: StopReason, detail: str | None = None) -> h.RunOutcome:
        return h.RunOutcome(reason=reason, detail=detail, turns=self.turn, final_text=self.final_text)

    async def _loop(self) -> h.RunOutcome:
        return await self._turns()

    async def _turns(self) -> h.RunOutcome:
        while True:
            if self.control.abort.is_set():
                return self._outcome("aborted")
            if self.turn >= self.binding.profile.max_turns:
                await self._wrap_up()
                return self._outcome("turn_budget", f"stopped after {self.turn} turns")
            if self.control.failure is not None:
                raise self.control.failure
            self.turn += 1
            self.deps.emit(TurnStart(turn=self.turn))
            self.control.news.clear()
            while self.control.steering:
                self._record(UserMessage(content=self.control.steering.pop(0)))
            self._tell_ended()

            try:
                request = await self._prepare_request()
                reply = await self._call_model_with_retries(request)
            except ProviderError as e:
                return self._outcome("provider_error", str(e))
            if reply is None:
                return self._outcome("aborted")

            verdict = await self._handle_reply(reply)
            if not isinstance(verdict, h.Stop) and self._waiting_on:
                # The agent handed over: the run holds for news (a job's end, System 1's look, the user) instead.
                self._waiting_on = []
                await self._hold(handed_over=True)
                continue
            if isinstance(verdict, h.Stop):
                # A steering message or a job's end that arrived during the final reply still gets an answer.
                if verdict.reason in ("completed", "waiting") and (self.control.steering or self._untold()):
                    continue
                # A command this run moved to the background still runs: its result is the agent's before the run
                # ends, so a `-p` run gets it as it would have, by the command's own deadline.
                if verdict.reason == "completed" and self._moved_running():
                    await self._hold()
                    continue
                return self._outcome(verdict.reason, verdict.detail)

    async def _wrap_up(self) -> None:
        """One tool-less turn when the budget runs out, so the user gets a reply instead of silence.

        Without it the run ends on whatever the last message carried: nothing after a tool call, or mid-work
        narration. The model is told it is out of turns and asked to say what is done and what is not."""
        self.turn += 1
        self.deps.emit(TurnStart(turn=self.turn))
        self._nudge(BUDGET_NUDGE, "turn_budget")
        try:
            request = await self._prepare_request()
            request.tools = []
            reply = await self._call_model_with_retries(request)
        except ProviderError:
            return
        if reply is None:
            return
        self.usage += reply.usage
        self.meter.observe(self._prompt_chars, reply.usage.input_tokens)
        self._record(reply)
        if reply.text:
            self.final_text = reply.text

    # -- phase: request -------------------------------------------------------------------------

    async def _prepare_request(self) -> ProviderRequest:
        profile = self.binding.profile
        specs = self._tool_specs()
        view = self._view()
        plan = plan_compaction(view, specs, self.meter, self.window, profile.thresholds)
        if plan is not None:
            view = await self._compact(view, plan, specs)

        prompt_limit = int(self.window * (1 - profile.thresholds.output_reserve))
        if estimate_tokens(view, specs, self.meter) > prompt_limit:
            squeeze(view, specs, self.meter, prompt_limit)
        await self.deps.hooks.call("context", lambda e: e.context(view, self._ctx()))
        squeeze(view, specs, self.meter, prompt_limit)

        prompt_tokens = estimate_tokens(view, specs, self.meter)
        # The working window caps the prompt (compaction, squeeze); the reply gets what the model's own
        # context leaves. Sized from the working window, a --window cap starved thinking models: at 16k it
        # left 2457 tokens, less than their reasoning on a hard turn, and the reply came back empty.
        advertised = self.binding.window.advertised
        room = max(advertised - prompt_tokens, MIN_OUTPUT_TOKENS)
        request = ProviderRequest(
            model=self.binding.wire_model,
            messages=view.messages(),
            tools=specs,
            max_output_tokens=min(profile.output_cap(advertised), room),
            sampling=profile.sampling,
            reasoning_effort=profile.reasoning_effort,
            reasoning_replay=profile.reasoning_replay,
            compat=profile.compat,
            session_id=self.deps.journal.session_id,
        )
        request = await self.deps.hooks.chain(
            "before_provider_request", request, lambda e, r: e.before_provider_request(r, self._ctx())
        )
        self._prompt_chars = self._admit_request(request)
        chars = view.segment_chars()
        self.deps.emit(
            ContextBudget(
                turn=self.turn,
                tokens=self.meter.tokens(self._prompt_chars),
                working_window=self.window,
                system=self.meter.tokens(chars["system"]),
                pinned=self.meter.tokens(chars["pinned"]),
                history=self.meter.tokens(chars["history"]),
                tail=self.meter.tokens(chars["tail"]),
            )
        )
        return request

    def _admit_request(self, request: ProviderRequest, binding: ModelBinding | None = None) -> int:
        """Check the final messages and schemas, including changes made by request hooks, for `binding`'s model
        (System 2's by default; a State write's, which a model switch may have replaced since it began).

        This is a calibrated estimate, not a provider tokenizer. Never silently cut immutable input
        or an extension's constraints to make a request fit.
        """
        binding = binding or self.binding
        window = binding.window.working
        ref, replay = binding.ref, request.reasoning_replay
        chars = sum(message_chars(m) + reasoning_chars(m, ref, replay) for m in request.messages)
        chars += tools_chars(request.tools)
        tokens = self.meter.tokens(chars)
        limit = int(window * (1 - binding.profile.thresholds.output_reserve))
        if tokens > limit:
            raise ProviderError(
                f"Final request is estimated at {tokens} input tokens, exceeding the working input budget "
                f"of {limit} (window {window}). Reduce the task, instruction files, tool schemas or "
                "extension context, or choose a larger working window. No provider call was made."
            )
        room = binding.window.advertised - tokens
        request.max_output_tokens = min(max(1, request.max_output_tokens), max(1, room))
        return chars

    def _view(self) -> ContextView:
        view = build_view(self.deps.journal.active_path(), self.deps.instructions, self.meter, self.window)
        view.model, view.replay = self.binding.ref, self.binding.profile.reasoning_replay
        return view

    def _update_state(self, view: ContextView) -> Coroutine[object, object, None] | None:
        """The update of the State over the turns it does not cover yet, as of now (None: it covers them all): edits
        on its items once it has them, else the State written whole. The write is announced and what it reads
        taken now; the coroutine returned makes it. Only a compaction writes the State, and waits for it; System 1
        reads its own cut of the session (`session_cut`). A write that fails is reported too, with what it cost.
        The run's abort stops it."""
        if not view.history_ids or view.covered >= len(view.history):
            return None
        turn, covers_through_id, tokens_before = self.turn, view.history_ids[-1], uncovered_tokens(view, self.meter)
        requests = self._requests()
        turns_covered = sum(isinstance(m, AssistantMessage) for m in view.history[view.covered :])
        # The write's own copy: request hooks may change any segment or message in place.
        view = replace(
            view,
            pinned=list(view.pinned),
            history=[message.model_copy(deep=True) for message in view.history],
            history_ids=list(view.history_ids),
            tail=list(view.tail),
            kept=[item.model_copy(deep=True) for item in view.kept],
        )
        self.deps.emit(StateWriting(turn=turn, turns_covered=turns_covered))
        return self._land_state(
            view, turn, requests, covers_through_id, tokens_before, turns_covered, self.control.abort
        )

    async def _land_state(
        self,
        view: ContextView,
        turn: int,
        requests: list[str],
        covers_through_id: str,
        tokens_before: int,
        turns_covered: int,
        abort: asyncio.Event,
    ) -> None:
        call = await self._write_state(view, turn, abort)
        self.usage += call.usage
        self.state_usage += call.usage
        dropped: Literal["write_failed", "branched"] | None = "write_failed" if call.text is None else None
        # It lands on the branch it was written for: the turns it covers must still lead to the leaf.
        if dropped is None and covers_through_id not in {e.id for e in self.deps.journal.active_path()}:
            dropped = "branched"
        if dropped is None and call.text is not None:
            text = self._with_requests(call.text, requests)
            call = call._replace(text=text)
            self.deps.journal.append(
                StateEntry(text=text, covers_through_id=covers_through_id, tokens_before=tokens_before)
            )
        self.deps.emit(
            State(
                turn=turn,
                landed=turn if dropped is None else 0,
                turns_covered=turns_covered,
                tokens_before=tokens_before,
                text=call.text if dropped is None and call.text else "",
                dropped=dropped,
                usage=call.usage,
                duration_ms=call.duration_ms,
                model=call.model,
                unseen_numbers=call.unseen_numbers,
                blanked=call.blanked,
                method=call.method,
                fallback=call.fallback,
            )
        )

    async def compact(self) -> CompactOutcome:
        """Compact now, between runs, as the loop does once a request passes `compact_at`: the same cut (the turns
        before the most recent `keep_recent` of the window), the State written first if it does not cover them,
        the reads System 1 keeps, the extensions' say. `fits`: every turn fits in `keep_recent`, so there is
        nothing to cut; `stopped`: aborted, or an extension vetoed it."""
        path = self.deps.journal.active_path()
        # The turn the conversation stands at, the last run's last, for the events: a State lands at a turn.
        last = next((e.run_id for e in reversed(path) if isinstance(e, MessageEntry) and e.run_id), None)
        replies = [e for e in path if isinstance(e, MessageEntry) and isinstance(e.message, AssistantMessage)]
        self.turn = max(sum(e.run_id == last for e in replies), 1)
        specs = self._tool_specs()
        view = self._view()
        plan = plan_compaction(view, specs, self.meter, self.window, self.binding.profile.thresholds, now=True)
        if plan is None:
            return "fits"
        await self._compact(view, plan, specs)
        cut = any(isinstance(e, CompactionEntry) for e in self.deps.journal.active_path()[len(path) :])
        return "compacted" if cut else "stopped"

    async def _compact(self, view: ContextView, plan: CompactionPlan, specs: list[ToolSpec]) -> ContextView:
        older, first_kept_id = plan.older, plan.first_kept_id
        if view.covered < len(older) and (write := self._update_state(view)) is not None:
            # The State does not cover the cut: write it now, over everything the agent sees.
            await write
            if self.control.abort.is_set():
                return view  # stopped while the State was being written: cut nothing rather than drop the turns
            view = self._view()
        # The cut is the plan's, not the State's: the recent turns stay raw even where the State describes them.
        summary, method = (view.state, "state") if view.covered >= len(older) else (None, "drop")
        # System 1 says which of the reads being cut the current step works on; those stay verbatim.
        kept = await keep_reads(
            older, view.kept, self._system1_state(), self.deps.decisions, self.window, turn=self.turn
        )
        if summary is None:
            # No State covers the cut: list what the cut turns did, with the user's requests as a State has them.
            previous = without_code_sections(view.state) if view.state else None
            summary = self._with_requests(dropped_turns(previous, older), self._requests())
        proposal = h.CompactionProposal(
            summary=summary,
            method=method,
            turns=sum(isinstance(m, AssistantMessage) for m in older),
            kept=kept,
        )
        for extension in self.deps.hooks.extensions:
            decision = await self.deps.hooks.invoke(
                "session_compact", extension, lambda e, p=proposal: e.session_compact(p, self._ctx())
            )
            if isinstance(decision, h.Veto):
                return view
            if decision is not None:
                proposal = decision

        self.deps.journal.append(
            CompactionEntry(
                summary=proposal.summary,
                first_kept_id=first_kept_id,
                tokens_before=plan.tokens_before,
                method=proposal.method,
                kept=proposal.kept,
            )
        )
        compacted = self._view()
        self.deps.emit(
            Compaction(
                turn=self.turn,
                method=proposal.method,
                tokens_before=plan.tokens_before,
                tokens_after=estimate_tokens(compacted, specs, self.meter),
                turns_compacted=proposal.turns,
                kept=len(proposal.kept),
                summary=proposal.summary,
            )
        )
        return compacted

    async def _write_state(self, view: ContextView, turn: int, abort: asyncio.Event) -> StateCall:
        """One State write, at the end of `turn`, stopped by `abort`. System 2 is forked: the request is the one the
        agent would send next (its system prompt, tools, and context with the reasoning it replays) plus the State
        request, so the writer sees all the agent sees and the provider can reuse the agent's cached prefix. The
        agent never sees it. `view` is the write's own copy (`_update_state`)."""
        binding = self.binding
        started = time.monotonic()
        specs = self._tool_specs()
        if "edit" not in {s.name for s in specs}:  # the number fix is an edit; the agent normally has the tool
            specs.append(EditTool().spec())
        # A State with its sections is brought up to date with edits, for every model; the first State, or one without
        # them, is written whole. The user's requests are code's to keep, not the writer's to edit.
        previous = parse_state(without_code_sections(view.state)) if view.state else None
        if previous is not None and not previous.editable():
            previous = None
        ask = state_edit_request(previous) if previous is not None else state_request(view)
        fork = replace(view, tail=[*view.tail, ask])
        await self.deps.hooks.call("context", lambda e: e.context(fork, self._ctx(turn)))
        squeeze(fork, specs, self.meter, int(self.window * (1 - binding.profile.thresholds.output_reserve)))
        room = binding.window.advertised - estimate_tokens(fork, specs, self.meter)
        base = ProviderRequest(
            model=binding.wire_model,
            messages=fork.messages(),
            # Offered so the prompt starts as the agent's does; a reply that only calls one is not a State (only
            # the number fix that may follow uses edit). The fork runs no tool.
            tools=specs,
            # A thinking model spends output tokens on reasoning before the State's text (Qwen3.8 at low
            # effort: 2-5K), so the cap is what the model's context leaves, not a share of the working window.
            max_output_tokens=max(min(MAX_STATE_TOKENS, room), MIN_OUTPUT_TOKENS),
            sampling=binding.profile.sampling,
            reasoning_effort=STATE_EFFORT,
            reasoning_replay=binding.profile.reasoning_replay,
            compat=binding.profile.compat,
            session_id=self.deps.journal.session_id,
        )
        request = await self.deps.hooks.chain(
            "before_provider_request", base, lambda e, r: e.before_provider_request(r, self._ctx(turn))
        )
        edit = None
        if previous is not None:
            # Should the edits fail: the same request asking for the whole State (hooked when it is sent).
            whole = replace(fork, tail=[*fork.tail[:-1], state_request(view)])
            edit = StateEdit(previous=previous, rewrite=base.model_copy(update={"messages": whole.messages()}))

        async def reply(candidate: ProviderRequest) -> AssistantMessage | None:
            # The first request has already been transformed, so number validation sees exactly its
            # filtered messages. Repairs are new requests and need the same boundary hooks too.
            whole = candidate is (edit.rewrite if edit is not None else request)
            if candidate is not request:
                candidate = await self.deps.hooks.chain(
                    "before_provider_request", candidate, lambda e, r: e.before_provider_request(r, self._ctx(turn))
                )
            return await self._state_reply(candidate, whole, turn, abort, binding)

        write = await write_state(request, reply, edit)
        return StateCall(
            text=write.text,
            usage=write.usage,
            duration_ms=int((time.monotonic() - started) * 1000),
            model=binding.ref,
            unseen_numbers=write.unseen_numbers,
            blanked=write.blanked,
            method=write.method,
            fallback=write.fallback,
        )

    async def _state_reply(
        self, request: ProviderRequest, stream: bool, turn: int, abort: asyncio.Event, binding: ModelBinding
    ) -> AssistantMessage | None:
        """One call of a State write to `binding`'s model; None if it failed or `abort` was set while waiting for
        it. A reply with the whole State streams out as `state_delta` events; edits and the number fix's do not."""

        def on_text(text: str) -> None:
            self.deps.emit(StateDelta(turn=turn, text=text))

        try:
            return await self._abortable(self._complete(request, binding, on_text if stream else None), abort)
        except ProviderError:
            return None

    async def _complete(
        self, request: ProviderRequest, binding: ModelBinding, on_text: Callable[[str], None] | None = None
    ) -> AssistantMessage | None:
        self._admit_request(request, binding)
        message = None
        async for item in binding.provider.stream(request):
            if isinstance(item, TextChunk) and on_text is not None and item.text:
                on_text(item.text)
            if isinstance(item, Completed):
                message = item.message
        return message

    async def _abortable[T](self, work: Coroutine[object, object, T], abort: asyncio.Event | None = None) -> T | None:
        """The work's result, or None if the run was aborted (`abort`, when given) while waiting for it. The work
        runs in a task of its own, so a result it gives after a cancel is dropped."""
        abort = self.control.abort if abort is None else abort
        if abort.is_set():
            # The abort callback may run after scheduling the caller but before its first turn.
            work.close()
            return None
        task = asyncio.create_task(work)
        aborted = asyncio.create_task(abort.wait())
        try:
            done, _ = await asyncio.wait({task, aborted}, return_when=asyncio.FIRST_COMPLETED)
            return task.result() if task in done else None
        finally:
            await join_tasks(task, aborted, cancel=True)

    # -- phase: model ---------------------------------------------------------------------------

    async def _consume(self, request: ProviderRequest) -> AssistantMessage | None:
        # The abort callback may run after scheduling this task but before its
        # first turn. Do not dispatch a new provider request in that interval.
        if self.control.abort.is_set():
            return None
        self._admit_request(request)
        message: AssistantMessage | None = None
        async for item in self.binding.provider.stream(request):
            match item:
                case TextChunk(text=text):
                    self.deps.emit(TextDelta(turn=self.turn, text=text))
                case ReasoningChunk(text=text):
                    self.deps.emit(ReasoningDelta(turn=self.turn, text=text))
                case Completed(message=completed):
                    message = completed
        if message is None:
            raise ProviderError("the stream ended without a reply")
        return message

    async def _call_model_with_retries(self, request: ProviderRequest) -> AssistantMessage | None:
        """Providers retry failures before any output; this covers a stream that breaks partway."""
        for wait in REPLY_RETRY_WAITS_S:
            try:
                return await self._call_model(request)
            except ProviderError as e:
                if not e.retryable:
                    raise
                await asyncio.sleep(wait)
        return await self._call_model(request)

    async def _call_model(self, request: ProviderRequest) -> AssistantMessage | None:
        """The reply, or None if the run was aborted while waiting for it."""
        return await self._abortable(self._consume(request))

    # -- phase: reply and tools -----------------------------------------------------------------

    async def _handle_reply(self, message: AssistantMessage) -> h.Verdict:
        self.usage += message.usage
        self.meter.observe(self._prompt_chars, message.usage.input_tokens)
        self._record(message)
        if message.text:
            self.final_text = message.text

        results = await self._run_tools(message.tool_calls) if message.tool_calls else []
        waiting_on = self._handed_over(message.tool_calls) if not message.invalid_tool_calls else []
        nudges: list[tuple[str, str]] = []
        if waiting_on and not self._answered and not self._urged and not self.control.hold:
            # It would hand the terminal UI's user the session back without a word since their message: first, once,
            # what does not wait. Not in a quiet session (-p, ACP): there "end your turn" would end the run, and the
            # session would stop the job the agent waits on.
            self._urged = True
            jobs = ", ".join(f"`{c}`" for c in waiting_on)
            nudges.append(
                (
                    f"{jobs} still runs; you are told when it ends, so do not wait on it. Answer now whatever the "
                    "user asked that does not depend on it; if they asked how it is going or whether it is stuck, "
                    "read its output or the file it writes to and answer from that. Say what you are waiting for, "
                    "then end your turn.",
                    "process_waiting",
                )
            )
            waiting_on = []
        if message.invalid_tool_calls:
            lines = [
                f"Your call to {c.name or 'a tool'} was not run: {c.error}. "
                "Send it again with its arguments as one valid JSON object."
                for c in message.invalid_tool_calls
            ]
            if message.stop == "length":
                lines.append(
                    "Your reply hit the output limit; write smaller pieces, e.g. several edits instead of one write."
                )
            nudges.append(("\n".join(lines), "invalid_tool_call"))
        elif message.stop == "length" and not message.tool_calls:
            if self.continuations < self.binding.profile.max_continuations:
                self.continuations += 1
                nudges.append((CUT_OFF_NUDGE, "output_limit"))

        verdict: h.Verdict
        if self._pending_stop is not None:
            verdict = self._pending_stop
        elif message.stop == "error":
            verdict = h.Stop(reason="provider_error", detail=message.error or "the provider refused the reply")
        elif waiting_on and not self.control.hold:
            # A session that wakes the agent when a job ends (the terminal UI) gives the user the session back now.
            verdict = h.Stop(reason="waiting", detail="waiting on " + ", ".join(f"`{c}`" for c in waiting_on))
        elif message.tool_calls or nudges:
            verdict = h.Continue()
        elif message.stop == "length":
            verdict = h.Stop(
                reason="output_limit",
                detail=f"the reply was cut off at the output limit {self.continuations + 1} times in a row",
            )
        else:
            verdict = h.Stop(reason="completed")

        if not isinstance(verdict, h.Stop) and not self.control.abort.is_set():
            # The turn's boundary: System 1 reviews the turns and the loop acts on the answer now (the nudge goes into
            # the next request). An abort during the ask ends the run instead of the ask.
            nudge = await self._abortable(
                self.supervisor.after_turn(self.turn, self.deps.decisions, self._system1_state())
            )
            if nudge:
                nudges.append(nudge)

        info = h.TurnInfo(turn=self.turn, message=message, results=results)
        for _, hook_verdict in await self.deps.hooks.call("turn_end", lambda e: e.turn_end(info, self._ctx())):
            if isinstance(hook_verdict, h.Stop):
                verdict = hook_verdict
                break
            if isinstance(hook_verdict, h.Nudge):
                nudges.append((hook_verdict.text, hook_verdict.reason))

        if isinstance(verdict, h.Stop) and (verdict.reason != "completed" or not nudges):
            return verdict
        for text, reason in nudges:
            self._nudge(text, reason)
        if waiting_on and not nudges:
            self._waiting_on = waiting_on  # quiet: `_turns` holds for news before the next turn
        return h.Continue()

    async def _run_tools(self, calls: list[ToolCall]) -> list[ToolResult]:
        by_name = {tool.name: tool for tool in self.deps.tools}
        # A tool-call hook can stop the run. Preserve call order in that case so a later call cannot
        # race ahead of its verdict; ordinary read-only batches still execute concurrently.
        has_call_hooks = any(type(e).tool_call is not h.Extension.tool_call for e in self.deps.hooks.extensions)
        parallel = (
            not has_call_hooks
            and len(calls) > 1
            and all(c.name in by_name and by_name[c.name].read_only for c in calls)
        )
        if parallel:
            tasks = [asyncio.create_task(self._run_tool(c, by_name)) for c in calls]
            try:
                results = list(await asyncio.gather(*tasks))
            finally:
                await join_tasks(*tasks, cancel=True)
        else:
            results = [await self._run_tool(c, by_name) for c in calls]
        return results

    async def _run_tool(self, call: ToolCall, by_name: dict[str, Tool]) -> ToolResult:
        self.deps.emit(ToolCallStart(turn=self.turn, call_id=call.id, name=call.name, arguments=call.arguments))
        started = time.monotonic()
        output = self._redacted(await self._tool_output(call, by_name))
        result = ToolResult(
            call_id=call.id,
            tool_name=call.name,
            status=output.status,
            content=output.content,
            error=output.error,
            resources=output.resources,
            images=output.images,
            duration_ms=int((time.monotonic() - started) * 1000),
            omitted_bytes=output.omitted_bytes,
            capture_omitted_bytes=output.capture_omitted_bytes,
        )
        # A big test/build log becomes a receipt of its evidence; hooks see the whole output; the window
        # bound applies to what is left.
        result = reduce_output(call, result, self.deps.spill_dir, self.turn, self.deps.emit) or result
        result = await self.deps.hooks.chain("tool_result", result, lambda e, r: e.tool_result(call, r, self._ctx()))
        limit = self._tool_result_chars()
        content, cut = bound_output(result.content, limit)
        if cut:
            observation_id = self.deps.journal.append(ObservationEntry(call_id=call.id, content=result.content)).id
            spill = self.deps.spill_dir / f"{_SAFE_NAME.sub('_', call.id)}-{observation_id}.txt"
            try:
                spill.parent.mkdir(parents=True, exist_ok=True)
                spill.write_text(result.content)
                spill.chmod(0o600)
                label = "Executor excerpt" if result.omitted_bytes else "Full output"
                content += f"\n[{label} saved to {spill}; read it with offset/limit or grep.]"
            except OSError as e:
                content += f"\n[The full output could not be saved: {e}]"
            result = result.model_copy(update={"content": content, "observation_id": observation_id})
        self._record(ToolResultMessage(result=result))
        self.deps.emit(
            ToolCallEnd(
                turn=self.turn,
                call_id=call.id,
                name=call.name,
                status=result.status,
                content=result.content,
                duration_ms=result.duration_ms,
                diff=output.diff,
            )
        )
        return result

    def _redacted(self, output: ToolOutput) -> ToolOutput:
        """The output with every stored secret's value replaced, from every tool (`read`, MCP, the browser and
        extensions as much as `bash`), so no copy of it holds a value: not the model's, the journal's, a client's
        or a spill file's."""
        redact = self.deps.redact
        if redact is None:
            return output
        update: dict[str, object] = {"content": redact(output.content)}
        if (error := output.error) is not None:
            update["error"] = error.model_copy(
                update={"what_failed": redact(error.what_failed), "next_action": redact(error.next_action)}
            )
        if (diff := output.diff) is not None:
            old_text = None if diff.old_text is None else redact(diff.old_text)
            update["diff"] = diff.model_copy(update={"old_text": old_text, "new_text": redact(diff.new_text)})
        return output.model_copy(update=update)

    async def _tool_output(self, call: ToolCall, by_name: dict[str, Tool]) -> ToolOutput:
        if self._pending_stop is not None:
            return fail("the run was stopped before this call ran", "Stop.").model_copy(update={"status": "refused"})
        if self.control.abort.is_set():
            return fail("the run was aborted before this call ran", "Stop.").model_copy(update={"status": "refused"})
        for extension in self.deps.hooks.extensions:
            decision = await self.deps.hooks.invoke(
                "tool_call", extension, lambda e, c=call: e.tool_call(c, self._ctx())
            )
            match decision:
                case h.ReplaceArguments(arguments=arguments):
                    call = call.model_copy(update={"arguments": arguments})
                case h.Refuse(reason=reason, next_action=next_action):
                    refused = fail(reason, next_action)
                    return refused.model_copy(update={"status": "refused"})
                case h.Stop():
                    self._pending_stop = decision
                    return fail(f"the run was stopped: {decision.detail or decision.reason}", "Stop.").model_copy(
                        update={"status": "refused"}
                    )

        tool = by_name.get(call.name)
        if tool is None:
            return fail(
                f"there is no tool named {call.name!r}",
                f"Call one of: {', '.join(sorted(by_name))}.",
                argument="name",
            )
        parsed = tool.parse(call.arguments)
        if isinstance(parsed, ToolOutput):
            return parsed
        # After the hooks, so the guard judges the arguments the tool will get; it cannot undo a hook's refusal.
        block = await self._abortable(
            self.guard.check(tool, call.id, call.arguments, requests=self._requests(), turn=self.turn)
        )
        if self.control.abort.is_set():
            return fail("the run was aborted before this call ran", "Stop.").model_copy(update={"status": "refused"})
        if block is not None:
            if self._pending_stop is None and (why := self.guard.exhausted()):
                self._pending_stop = h.Stop(reason="stopped_by_guard", detail=why)
            refused = fail(f"the guard refused this call: {block.reason}", NEXT_ACTION)
            return refused.model_copy(update={"status": "refused"})
        limit = self._tool_result_chars()
        ctx = ToolContext(
            cwd=self.deps.cwd,
            executor=self.deps.executor,
            abort=self.control.abort,
            max_output_chars=limit,
            image_dir=self.deps.spill_dir / "images" if self.binding.profile.vision else None,
            # Agents list the working directory and read what the harness left there; a runner's file tools
            # reach only its workspace, so there the captures stay in it.
            output_dir=None if mount_of(self.deps.executor) else self.deps.spill_dir / "command-output",
            processes=self.deps.processes,
            call_id=call.id,
        )
        try:
            return await tool.run(parsed, ctx)
        except System1Unavailable:
            raise
        except Exception as e:
            return fail(
                f"{tool.name} failed unexpectedly: {type(e).__name__}: {e}",
                "Try different arguments or a different approach.",
                retry_safe=False,
            )
