"""The eight hooks extensions attach to, and how they are called.

Hook names follow pi's lifecycle so extension designs written against it port one to one. Extensions
run in the order they were configured and fail open: an exception is reported as a `hook_error`
event and the run continues as if that extension had returned nothing. The one exception is
`System1Unavailable`: System 1 is required, so it stops the run wherever it is raised.

Beside the hooks, an extension may declare tools (pi's `registerTool`). No hook can do it: the tool
list is fixed before the first request, when the specs go to the model. It may also name its own view
in the terminal UI (`tui_view`), fed by the `ExtensionEvent`s it emits; the view lives with the
extension, the TUI only hosts it (`intern.tui.views`).
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from pydantic import BaseModel, Field

from intern.decisions import System1Unavailable
from intern.events import EventSink, HookError, StopReason
from intern.journal import Kept
from intern.messages import AssistantMessage, JsonObject, ToolCall, ToolResult
from intern.models.registry import ConfigError
from intern.needs import Needs

if TYPE_CHECKING:
    from intern.context import ContextView
    from intern.decisions import DecisionClient
    from intern.journal import Journal
    from intern.models.profile import Profile
    from intern.providers.base import ProviderRequest
    from intern.tools.base import Tool

HookName = Literal[
    "before_agent_start",
    "context",
    "before_provider_request",
    "tool_call",
    "tool_result",
    "turn_end",
    "session_compact",
    "agent_end",
]


class Continue(BaseModel):
    kind: Literal["continue"] = "continue"


class Nudge(BaseModel):
    """Keep going and tell the model why, in a message the engine adds to the conversation."""

    kind: Literal["nudge"] = "nudge"
    text: str
    reason: str


class Stop(BaseModel):
    kind: Literal["stop"] = "stop"
    reason: StopReason
    detail: str | None = None


Verdict = Continue | Nudge | Stop


class Refuse(BaseModel):
    """Do not run the tool; the model receives `reason` as a refused tool result."""

    kind: Literal["refuse"] = "refuse"
    reason: str
    next_action: str


class ReplaceArguments(BaseModel):
    kind: Literal["replace"] = "replace"
    arguments: JsonObject


ToolCallDecision = Refuse | ReplaceArguments | Stop


class CompactionProposal(BaseModel):
    summary: str
    method: Literal["summary", "drop", "state"]
    turns: int
    kept: list[Kept] = Field(
        default_factory=list, description="Reads shown verbatim after the State; System 1's choice."
    )


class Veto(BaseModel):
    """Skip this compaction; the engine falls back to shortening old tool output in the view."""

    kind: Literal["veto"] = "veto"
    reason: str


class TurnInfo(BaseModel):
    turn: int
    message: AssistantMessage
    results: list[ToolResult]


class RunOutcome(BaseModel):
    reason: StopReason
    detail: str | None
    turns: int
    final_text: str


@dataclass
class RunContext:
    """What an extension may see and use. Extensions keep their own state on themselves."""

    run_id: str
    turn: int
    cwd: str
    profile: Profile
    working_window: int
    chars_per_token: float
    journal: Journal
    decisions: DecisionClient
    emit: EventSink
    # The session as System 1 should see it (`session_cut`: the user's requests and the whole session, cut to fit): pass
    # it as `state` to `decisions.ask`, so every System 1 question reads the same picture.
    state: Callable[[], str]


class Extension:
    """Base class; override only the hooks you need. One instance lives for one session."""

    name: ClassVar[str] = "extension"
    tui_view: ClassVar[str | None] = None
    """`module:Class` of the extension's view in the terminal UI, a `intern.tui.views.ExtensionView`: it is
    given every live event and draws what the extension reports (`ExtensionEvent`s it emits through `ctx.emit`).
    Only the TUI imports it, so a headless or ACP run never loads view code."""
    needs: ClassVar[Needs | None] = None
    """What it needs installed beyond the base install: its `intern` extra and the modules that show the extra is
    there. Setup refuses it without them, and the TUI's `/caps` installs them when it switches it on."""

    def tools(self) -> list[Tool]:
        """Tools this extension adds to the session; one named like a pack tool replaces it."""
        return []

    async def before_agent_start(self, ctx: RunContext) -> None:
        return None

    async def context(self, view: ContextView, ctx: RunContext) -> None:
        return None

    async def before_provider_request(self, request: ProviderRequest, ctx: RunContext) -> ProviderRequest | None:
        return None

    async def tool_call(self, call: ToolCall, ctx: RunContext) -> ToolCallDecision | None:
        return None

    async def tool_result(self, call: ToolCall, result: ToolResult, ctx: RunContext) -> ToolResult | None:
        return None

    async def turn_end(self, turn: TurnInfo, ctx: RunContext) -> Verdict | None:
        return None

    async def session_compact(self, proposal: CompactionProposal, ctx: RunContext) -> CompactionProposal | Veto | None:
        return None

    async def agent_end(self, outcome: RunOutcome, ctx: RunContext) -> Nudge | None:
        """May return a Nudge to reopen a completed run once more (a finish gate)."""
        return None


class Hooks:
    def __init__(self, extensions: list[Extension], emit: EventSink) -> None:
        self.extensions = extensions
        self._emit = emit

    async def invoke[T](
        self, hook: HookName, extension: Extension, invoke: Callable[[Extension], Awaitable[T | None]]
    ) -> T | None:
        """Call one extension's hook; an exception becomes a `hook_error` event and a None result."""
        try:
            return await invoke(extension)
        except System1Unavailable:
            raise
        except Exception as e:
            self._emit(HookError(hook=hook, extension=extension.name, error=f"{type(e).__name__}: {e}"))
            return None

    async def call[T](
        self, hook: HookName, invoke: Callable[[Extension], Awaitable[T | None]]
    ) -> list[tuple[Extension, T]]:
        """Call `hook` on every extension in order; returns the non-None results."""
        results: list[tuple[Extension, T]] = []
        for extension in self.extensions:
            value = await self.invoke(hook, extension, invoke)
            if value is not None:
                results.append((extension, value))
        return results

    async def chain[T](self, hook: HookName, value: T, invoke: Callable[[Extension, T], Awaitable[T | None]]) -> T:
        """Pass `value` through every extension in order; each sees the previous one's replacement."""
        for extension in self.extensions:
            replaced = await self.invoke(hook, extension, lambda e, v=value: invoke(e, v))
            if replaced is not None:
                value = replaced
        return value


def resolve_extension(spec: str, registry: dict[str, type[Extension]]) -> type[Extension]:
    """The class a packaged extension name or a `module:Class` names, not yet built."""
    if spec in registry:
        return registry[spec]
    module_name, sep, class_name = spec.partition(":")
    if not sep:
        known = ", ".join(sorted(registry)) or "none packaged yet"
        raise ConfigError(f"unknown extension {spec!r}; use a packaged name ({known}) or module:Class")
    try:
        cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as e:
        raise ConfigError(f"cannot load extension {spec!r}: {e}") from e
    if not (isinstance(cls, type) and issubclass(cls, Extension)):
        raise ConfigError(f"{spec!r} is not an Extension subclass")
    return cls


def build_extension(spec: str, cls: type[Extension]) -> Extension:
    """One instance of `cls`. An extension is built with no arguments; one that cannot be (it needs arguments, or
    its constructor fails) is a `ConfigError` naming it, with the original error as its cause."""
    try:
        return cls()
    except Exception as e:
        raise ConfigError(f"cannot build extension {spec!r}: {type(e).__name__}: {e}") from e


def load_extension(spec: str, registry: dict[str, type[Extension]]) -> Extension:
    """Load a packaged extension by name, or any class by `module:Class`; every failure is a `ConfigError`."""
    return build_extension(spec, resolve_extension(spec, registry))
