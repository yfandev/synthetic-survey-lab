"""The protocol's event union: everything the engine emits, one pydantic model per event type.

Clients in other languages generate their types from `event_json_schema()`. A breaking change to any
event bumps `PROTOCOL_VERSION`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from intern.messages import FileDiff, JsonObject, Message, System1Usage, Usage

# 2: tool_call_end carries an optional `diff`.
PROTOCOL_VERSION = 2

StopReason = Literal[
    "completed",
    "turn_budget",
    "output_limit",
    "aborted",
    "provider_error",
    "system1_unavailable",
    "stopped_by_hook",
    "stopped_by_tool",
    "stopped_by_guard",
    "internal_error",
    "waiting",
]
"""`waiting`: the agent handed over to the harness, waiting on a background job (a turn that only polled running jobs),
in a session that wakes it when the job ends (the terminal UI). Quiet sessions hold the run instead."""


def now_ms() -> int:
    return int(time.time() * 1000)


class _EventBase(BaseModel):
    timestamp: int = Field(default_factory=now_ms)


class WindowInfo(BaseModel):
    """How the working window was chosen: the smallest of the three limits that are known."""

    advertised: int
    advertised_source: Literal["profile", "provider", "llama.cpp", "default"]
    effective: int | None
    user_cap: int | None
    working: int


class RunStart(_EventBase):
    type: Literal["run_start"] = "run_start"
    protocol_version: int = PROTOCOL_VERSION
    session_id: str
    run_id: str
    model: str
    cwd: str
    window: WindowInfo
    tools: list[str]


class TurnStart(_EventBase):
    type: Literal["turn_start"] = "turn_start"
    turn: int


class TextDelta(_EventBase):
    type: Literal["text_delta"] = "text_delta"
    turn: int
    text: str


class ReasoningDelta(_EventBase):
    type: Literal["reasoning_delta"] = "reasoning_delta"
    turn: int
    text: str


class MessageEvent(_EventBase):
    """A message was added to the conversation: the system prompt, a prompt, a reply or a tool result."""

    type: Literal["message"] = "message"
    entry_id: str
    message: Message


class ToolCallStart(_EventBase):
    type: Literal["tool_call_start"] = "tool_call_start"
    turn: int
    call_id: str
    name: str
    arguments: JsonObject


class ToolCallEnd(_EventBase):
    type: Literal["tool_call_end"] = "tool_call_end"
    turn: int
    call_id: str
    name: str
    status: Literal["ok", "error", "refused"]
    content: str
    duration_ms: int
    diff: FileDiff | None = Field(
        default=None, description="For a write or edit on the host: the file before and after. Not journaled."
    )


class ContextBudget(_EventBase):
    """The request about to be sent, measured after pruning; tokens are calibrated against provider usage."""

    type: Literal["context_budget"] = "context_budget"
    turn: int
    tokens: int
    working_window: int
    system: int
    pinned: int
    history: int
    tail: int


class State(_EventBase):
    """The State was updated by a fork of System 2's own session over the turns up to `turn`, for a compaction:
    with edits on its items, or written whole. A write that did not land is reported too, with `dropped` saying why
    and `landed` 0."""

    type: Literal["state"] = "state"
    turn: int
    landed: int = Field(default=0, description="The turn the State landed at: `turn`, or 0 when it did not.")
    trigger: str = Field(default="compaction", description="Always `compaction`.")
    turns_covered: int
    tokens_before: int
    text: str
    dropped: Literal["write_failed", "branched"] | None = Field(
        default=None,
        description="Why the write did not land: the call failed or gave no State (`write_failed`), or it came after "
        "the run and the session had moved to another branch meanwhile (`branched`).",
    )
    usage: Usage | None = Field(default=None, description="The write's usage, when there was a call.")
    duration_ms: int | None = Field(default=None, description="The write's duration.")
    model: str | None = Field(default=None, description="The model the write went to.")
    unseen_numbers: list[str] = Field(
        default_factory=list,
        description="Numbers the State gave that the conversation never showed; the writer fixed them with "
        "line edits on the State.",
    )
    blanked: list[str] = Field(default_factory=list, description="Those still there after the fix, replaced by `?`.")
    method: Literal["edit", "rewrite"] = Field(
        default="rewrite",
        description="`edit`: the writer edited the previous State's items; `rewrite`: it wrote the State whole (the "
        "first State, one without its sections, or edits that could not be applied).",
    )
    fallback: str | None = Field(
        default=None, description="Why edits were asked for and the State was written whole instead."
    )


class Compaction(_EventBase):
    type: Literal["compaction"] = "compaction"
    turn: int
    method: Literal["summary", "drop", "state"]
    tokens_before: int
    tokens_after: int
    turns_compacted: int
    summary: str
    """What the agent continues from: the State, or with `drop` a list of the dropped tool calls."""
    kept: int = Field(default=0, description="Reads kept verbatim after the State, by System 1's judgement.")


class Reduced(_EventBase):
    """A big test/build output was replaced by a receipt of verbatim evidence, or left alone (`applied` false)."""

    type: Literal["reduced"] = "reduced"
    turn: int
    call_id: str
    source_chars: int
    receipt_chars: int
    items: int
    applied: bool
    reason: str | None = Field(default=None, description="Why the output was left alone.")


class MemoryMount(_EventBase):
    type: Literal["memory_mount"] = "memory_mount"
    drive: str
    commit: str


class MemoryRecall(_EventBase):
    type: Literal["memory_recall"] = "memory_recall"
    commit: str
    pages: list[str]
    paths: list[str]
    chars: int
    steps: int
    backend: str
    reused: bool = Field(description="The session already had a recall on this path; System 1 was not asked.")
    duration_ms: int


class Decision(_EventBase):
    type: Literal["decision"] = "decision"
    purpose: str
    backend: str
    questions: int
    ok: bool
    cached: bool
    cost_usd: float
    input_tokens: int = Field(
        default=0,
        description="System 1's tokens for this attempt, as the backend reported them; 0 for a cached answer. They "
        "add up to `run_end.system1`, so a run that never reaches `run_end` still has its System 1 usage.",
    )
    output_tokens: int = 0
    duration_ms: int
    error: str | None = None
    truncated: bool = Field(default=False, description="The state was cut to System 1's max_state_tokens.")
    turn: int | None = Field(default=None, description="The turn the ask belongs to, when the caller knows it.")
    subject: str | None = Field(default=None, description="What was judged, when the caller names it: a read's path.")
    answers: dict[str, JsonObject] | None = Field(
        default=None,
        description="What the caller got, question by question, in compact form: a choice with its probabilities, "
        "a noul's probability, a score with its level probabilities. None when `ok` is false: a failed attempt, "
        "which is retried.",
    )


class StateWriting(_EventBase):
    """A State write began at the end of `turn`: a fork of System 2's own session is writing it while a compaction
    waits for it. A `state` event follows, landed or dropped. Listeners show the write, so a pause for it reads as
    the write it is."""

    type: Literal["state_writing"] = "state_writing"
    turn: int
    trigger: str = Field(default="compaction", description="Always `compaction`.")
    turns_covered: int = Field(description="The turns the write covers that the standing State does not.")


class StateDelta(_EventBase):
    """A piece of the State's text as the write streams it; the `state` event carries the whole, checked."""

    type: Literal["state_delta"] = "state_delta"
    turn: int
    text: str


class Indexing(_EventBase):
    """A search index (code search's, before a run's first turn) is being built or refreshed: `done` of `total`
    files so far, a few times a second, and once more with `finished` and how long it took."""

    type: Literal["indexing"] = "indexing"
    name: str = Field(description="The store: `code-search`.")
    root: str
    done: int
    total: int
    finished: bool = False
    duration_ms: int | None = None


class ExtensionEvent(_EventBase):
    """An extension's own report as it works (a search's progress, say), in a payload whose type the extension
    owns: `data` is the extension's pydantic model dumped to JSON (`ExtensionEvent.of`), and the extension's view
    in the terminal UI (`Extension.tui_view`) validates it back. The engine passes it on and never reads it. Not
    journaled; ACP sends nothing for it."""

    type: Literal["extension"] = "extension"
    extension: str = Field(description="The extension's name: `code-search`.")
    kind: str = Field(description="What the payload is, in the extension's words: `search_trace`.")
    data: JsonObject

    @classmethod
    def of(cls, extension: str, kind: str, payload: BaseModel) -> ExtensionEvent:
        return cls(extension=extension, kind=kind, data=payload.model_dump(mode="json"))


class Nudge(_EventBase):
    """The engine added a message of its own, e.g. after a cut-off reply or an unparsable tool call."""

    type: Literal["nudge"] = "nudge"
    turn: int
    reason: str
    text: str


class ToolGuard(_EventBase):
    """The guard refused a tool call: a fixed rule matched, or System 1 judged it risky and unrequested."""

    type: Literal["tool_guard"] = "tool_guard"
    turn: int
    call_id: str
    name: str
    layer: Literal["rule", "system1"]
    rule: str = Field(description="The fixed rule, or the risk System 1 saw.")
    reason: str
    risk: float | None = Field(default=None, description="System 1's P(any risk).")
    asked: float | None = Field(default=None, description="System 1's P(the user asked for the call).")


ProcessKind = Literal["call", "job", "left_running"]
ProcessState = Literal["running", "exited", "timed_out", "terminated", "aborted", "out_of_memory", "moved"]
"""`moved`: a `bash` call that went on as a background job; the job's record goes on from there."""


class ProcessInfo(BaseModel):
    """One command that runs, or ran, in the session: a `bash` call while it runs (`call`), a job the session's job
    manager owns (`job`), or a process group a finished command left running (`left_running`, a `cmd &`)."""

    id: str = Field(description="A job's handle, a call's id, or `group-<pgid>` for what a command left running.")
    kind: ProcessKind
    command: str
    state: ProcessState
    exit_code: int | None = None
    started_at: int = Field(description="When it started, in ms since the epoch.")
    ended_at: int | None = Field(default=None, description="When it ended; None while it runs.")
    last_output_at: int | None = Field(
        default=None, description="When it last wrote output, for a job; None for what is not read as it runs."
    )
    note: str | None = Field(
        default=None,
        description="Why a `bash` call went on in the background, on the call (state `moved`) and on its job: "
        "System 1's judgment and its probability, the timer, a stuck finding, or a rule.",
    )


class ProcessEvent(_EventBase):
    """A command started or ended in the session (`ProcessInfo`): what is running at any moment, from one stream.
    Journaled. ACP sends nothing for it."""

    type: Literal["process"] = "process"
    action: Literal["started", "ended"]
    process: ProcessInfo


class ProcessVerdict(BaseModel):
    """What System 1 found of one running command at a check (`process_watch`)."""

    id: str
    command: str
    elapsed_s: int
    verdict: str = Field(description="Its pick: progressing, waiting_as_expected, stalled, failing, ready or unclear.")
    probabilities: dict[str, float] = Field(default_factory=dict)
    told: bool = Field(description="The verdict calls for the agent's attention, and it is told.")


class ProcessCheck(_EventBase):
    """System 1 looked at the commands running: one verdict each. Those it found stalled, failing, or up and
    waited on as if they would end are told to the agent. Journaled. ACP sends nothing for it."""

    type: Literal["process_check"] = "process_check"
    turn: int | None = Field(default=None, description="The run's turn when one was in progress; None between runs.")
    verdicts: list[ProcessVerdict]


class SecretUnavailable(_EventBase):
    """A `[secrets]` entry could not be read; the session runs without it. `message` names the entry and the fix,
    never the value."""

    type: Literal["secret_unavailable"] = "secret_unavailable"
    name: str
    reason: str
    message: str


class HookError(_EventBase):
    """An extension raised; hooks fail open, so the run continues without its effect."""

    type: Literal["hook_error"] = "hook_error"
    hook: str
    extension: str
    error: str


class RunEnd(_EventBase):
    type: Literal["run_end"] = "run_end"
    run_id: str
    reason: StopReason
    detail: str | None = None
    turns: int
    usage: Usage
    agent_usage: Usage | None = Field(default=None, description="The agent's own replies; usage = agent + state.")
    state_usage: Usage | None = Field(default=None, description="State writes, the ones made for compaction included.")
    system1: System1Usage = Field(
        default_factory=System1Usage, description="This run's System 1 asks: answered, failed after retries, tokens."
    )
    final_text: str
    duration_ms: int


Event = Annotated[
    RunStart
    | TurnStart
    | TextDelta
    | ReasoningDelta
    | MessageEvent
    | ToolCallStart
    | ToolCallEnd
    | ContextBudget
    | State
    | StateWriting
    | StateDelta
    | Compaction
    | Reduced
    | MemoryMount
    | MemoryRecall
    | Indexing
    | ExtensionEvent
    | Decision
    | Nudge
    | ToolGuard
    | ProcessEvent
    | ProcessCheck
    | HookError
    | SecretUnavailable
    | RunEnd,
    Field(discriminator="type"),
]
EventSink = Callable[[Event], None]
event_adapter: TypeAdapter[Event] = TypeAdapter(Event)


def event_json_schema() -> dict[str, Any]:
    return event_adapter.json_schema()
