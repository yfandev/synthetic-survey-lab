"""Agent Client Protocol (ACP v1) over stdio, for t3code, Zed and other ACP clients.

One process serves any number of ACP sessions, each a harness `Session` built by `Bootstrap` like the
CLI's and the JSON-RPC server's. The contract is docs/acp.md; the event mapping is acp_mapping.py.
Needs the `acp` extra (the official `agent-client-protocol` SDK).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import signal
import stat
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from acp import (
    PROTOCOL_VERSION,
    Client,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    SetSessionConfigOptionResponse,
    SetSessionModeResponse,
    run_agent,
    stdio_streams,
    update_agent_message_text,
)
from acp.connection import StreamDirection, StreamEvent
from acp.schema import (
    AgentCapabilities,
    AudioContentBlock,
    ClientCapabilities,
    CloseSessionResponse,
    ConfigOptionUpdate,
    EmbeddedResourceContentBlock,
    ForkSessionResponse,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    ListSessionsResponse,
    McpServerStdio,
    PromptCapabilities,
    ResourceContentBlock,
    ResumeSessionResponse,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionConfigSelectGroup,
    SessionConfigSelectOption,
    SessionMode,
    SessionModeState,
    SetSessionModelResponse,
    SseMcpServer,
    TextContentBlock,
    TextResourceContents,
    Usage,
)
from pydantic import BaseModel, StrictBool, ValidationError

from intern import __version__, capability_toggles, names
from intern.capability_settings import CapabilitySettings
from intern.config import Bootstrap, Options, System2Unavailable, home_dir, process_env
from intern.events import ContextBudget, Event, EventSink, RunEnd
from intern.journal import CapabilitiesChangeEntry, Journal, ModelChangeEntry
from intern.memory.spend import read_spend
from intern.models.catalog import CACHE_TTL_S, Catalog, ModelInfo
from intern.models.profile import DEFAULT_EFFORT, Profile
from intern.models.registry import ConfigError
from intern.models.window import recapped
from intern.needs import InstallFailed, Needs, Say, install
from intern.protocol import acp_mapping
from intern.session import Session, SessionBusy
from intern.session_setup import (
    capability_needs,
    capability_problem,
    runner_choices,
    runner_config,
    runner_problem,
)
from intern.tools import load_packs

AGENT_NAME = names.NAME
AGENT_TITLE = names.TITLE
FULL_ACCESS = "full-access"
INDEX_FILE = "index.json"
CONTEXT_WINDOW = "context_window"
AUTO_WINDOW = "auto"
WINDOW_PRESETS = (32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000)
CAPABILITY_CATEGORY = "_nace_capability"
RUNNER = "runner"
RUNNER_CATEGORY = "_nace_runner"
"""ACP keeps categories starting with `_` for custom ones."""
DEFAULT_PACKS = tuple(Profile().tool_packs)
"""The packs toggles start from before a session's profile is known."""
# Prompts carry images inline as base64; the SDK's own default.
STDIO_LIMIT_BYTES = 50 * 1024 * 1024
# How long replies to requests already received may take once stdin closes; runs are aborted first.
DRAIN_TIMEOUT_S = 10.0

SESSION_BUSY = -32000
PROVIDER_ERROR = -32001

ConfigOption = SessionConfigOptionSelect | SessionConfigOptionBoolean
McpServer = HttpMcpServer | SseMcpServer | McpServerStdio
PromptBlock = (
    TextContentBlock | ImageContentBlock | AudioContentBlock | ResourceContentBlock | EmbeddedResourceContentBlock
)


@dataclass
class OpenRequest:
    """What a session is opened with: the client's cwd, the journal to create or resume, and models."""

    cwd: Path
    journal: Path | None
    system2: str | None
    emit: EventSink
    capabilities: CapabilitySettings | None = None
    """A whole replacement for the configured capabilities, from the client's toggles; None keeps them."""


SessionOpener = Callable[[OpenRequest], Awaitable[Session]]
CatalogLoader = Callable[[], Awaitable[Catalog]]
Installer = Callable[[Needs, Say], Awaitable[None]]
"""Installs what a capability needs and this environment lacks, saying what it does (`needs.install`)."""


def bootstrap_opener(options: Options, env: Mapping[str, str] | None = None) -> SessionOpener:
    """Open sessions the way the CLI does, with `options` as every session's defaults."""

    async def open_session(request: OpenRequest) -> Session:
        update: dict[str, Any] = {"cwd": request.cwd, "session_file": request.journal}
        if request.system2:
            update["system2"] = request.system2
        if request.capabilities is not None:
            # Built from the configured settings, which already hold --memory and --no-memory.
            update |= {"capabilities": request.capabilities, "memory": None, "memory_off": False}
        return await Bootstrap(options.model_copy(update=update), env).session(request.emit)

    return open_session


class SessionIndex:
    """Session id to journal file, so `session/load` finds a journal by the id the client kept."""

    def __init__(self, sessions: Path) -> None:
        self.sessions = sessions

    @property
    def path(self) -> Path:
        return self.sessions / INDEX_FILE

    def _read(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def remember(self, session_id: str, journal: Path) -> None:
        entries = self._read()
        entries[session_id] = str(journal)
        self.sessions.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entries, indent=1))
        temporary.replace(self.path)

    def find(self, session_id: str) -> Path | None:
        known = self._read().get(session_id)
        if known and Path(known).is_file():
            return Path(known)
        # Journals written before the index existed, or by the CLI: the first entry's id is the session id.
        for journal in sorted(self.sessions.glob("*.jsonl"), reverse=True):
            try:
                with journal.open() as f:
                    first = json.loads(f.readline() or "{}")
            except (OSError, ValueError):
                continue
            if isinstance(first, dict) and first.get("id") == session_id:
                self.remember(session_id, journal)
                return journal
        return None


def window_cap(value: str) -> int | None:
    """A `context_window` value as a cap: `auto` (None, the `--window` default) or a positive token count."""
    if value == AUTO_WINDOW:
        return None
    if not value.isascii() or not value.isdigit() or int(value) <= 0:
        raise RequestError(-32602, f"{CONTEXT_WINDOW} takes {AUTO_WINDOW} or a positive number of tokens")
    return int(value)


def tokens_label(tokens: int) -> str:
    return f"{tokens / 1_000_000:.3g}M" if tokens >= 1_000_000 else f"{round(tokens / 1000)}K"


class SessionMeta(BaseModel):
    """`_meta.nace` on `session/new` and `session/load`: the models and capability toggles the client wants the
    session on."""

    system2: str | None = None
    system1: str | None = None
    capabilities: dict[str, StrictBool] | None = None
    runner: str | None = None


def session_meta(meta: Mapping[str, Any]) -> SessionMeta:
    try:
        return SessionMeta.model_validate(meta.get("nace") or {})
    except ValidationError as e:
        raise RequestError(-32602, f"_meta.nace is invalid: {e}") from None


@dataclass
class UnboundSession:
    """An ACP session no System 2 could be bound for (none configured, or its key is missing). It takes
    configuration but no prompts, and opens as a harness session once the client picks a model that binds."""

    id: str
    cwd: str
    journal: Path | None
    system2: str
    """The model it would have run on, shown as the picker's current value; empty when none is configured."""
    system1: str
    reason: str
    capabilities: CapabilitySettings
    """What it opens on once a model binds: the configured settings with the client's toggles."""
    opening: bool = False
    window: str | None = None
    """A `context_window` picked before a model bound, applied once one does."""


@dataclass
class LiveSession:
    """An ACP session: the harness session and the queue its events go through to the client, in order. Its usage
    totals outlive a reopen."""

    id: str
    session: Session
    capabilities: CapabilitySettings
    """What the harness session is open on."""
    packs: tuple[str, ...]
    """The packs toggles start from when `tool_packs` is unset: the profile's. Bootstrap replaces a bound
    profile's packs with those in force, so they are read once, from a session opened without a list."""
    queue: asyncio.Queue[Event] = field(default_factory=asyncio.Queue)
    pump: asyncio.Task[None] | None = None
    context_tokens: int = 0
    cost_usd: float = 0.0
    pending: CapabilitySettings | None = None
    """Toggles set since it opened, applied by reopening it at the next prompt."""
    reopening: bool = False
    cancelled: bool = False
    """A cancel arrived while it reopened, before the prompt's run started."""
    installing: asyncio.Task[None] | None = None
    """The install a switched-on capability waits for before the reopen; a cancel stops it."""

    def observe(self, event: Event) -> None:
        """Keep the running totals the usage updates report."""
        match event:
            case ContextBudget(tokens=tokens):
                self.context_tokens = tokens
            case RunEnd(usage=usage, system1=system1):
                self.cost_usd += usage.cost_usd + system1.cost_usd
            case _:
                pass

    def mapping(self) -> acp_mapping.MappingContext:
        deps = self.session.deps
        return acp_mapping.MappingContext(
            cwd=deps.cwd,
            effects={tool.name: tool.effects for tool in deps.tools},
            working_window=deps.system2.window.working,
            context_tokens=self.context_tokens,
            cost_usd=self.cost_usd,
        )

    async def flush(self) -> None:
        """Wait until every event emitted so far has been sent."""
        await self.queue.join()


def _replay_context(cwd: Path) -> acp_mapping.MappingContext:
    """How an unbound session's history is shown: no profile picked its tool packs yet, so kinds come from base."""
    effects = {tool.name: tool.effects for tool in load_packs(["base"])}
    return acp_mapping.MappingContext(cwd=cwd, effects=effects, working_window=0, context_tokens=0, cost_usd=0.0)


def prompt_text(blocks: list[PromptBlock], attachments: Path) -> str:
    """One prompt from ACP content blocks: text in order, resources inline, images saved for `read`."""
    parts: list[str] = []
    for index, block in enumerate(blocks):
        match block:
            case TextContentBlock(text=text):
                parts.append(text)
            case ResourceContentBlock(uri=uri, name=name):
                parts.append(f"[{name}]({uri})")
            case EmbeddedResourceContentBlock(resource=TextResourceContents(uri=uri, text=text)):
                parts.append(f"{uri}:\n```\n{text}\n```")
            case EmbeddedResourceContentBlock(resource=resource):
                parts.append(f"[attached binary resource {resource.uri}, not shown]")
            case ImageContentBlock(data=data, mime_type=mime):
                suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
                path = attachments / f"{uuid.uuid4().hex[:8]}-{index}{suffix.get(mime, '.img')}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(base64.b64decode(data))
                parts.append(f"[The user attached an image: {path}. Read it to look at it.]")
            case AudioContentBlock():
                raise RequestError.invalid_params({"message": "audio prompts are not supported"})
    return "\n\n".join(part for part in parts if part)


class AcpAgent:
    """The ACP methods, over a session opener. Tests pass an opener that builds sessions on fake providers."""

    def __init__(
        self,
        open_session: SessionOpener,
        home: Path,
        *,
        persist: bool = True,
        catalog: CatalogLoader | None = None,
        capabilities: CapabilitySettings | None = None,
        unavailable: Callable[[str], str | None] = lambda toggle_id: None,
        install: Installer | None = None,
    ) -> None:
        self._open = open_session
        self._home = home
        self._persist = persist
        self._capabilities = capabilities if capabilities is not None else CapabilitySettings()
        """The configured settings every session starts from; toggles are set over them."""
        self._unavailable = unavailable
        """Why a capability toggle cannot switch on in this install and environment as it is now, else None."""
        self._install = install
        """What a prompt installs a switched-on capability's missing extra with; None installs nothing."""
        self._installs = asyncio.Lock()
        """One install at a time: two sessions switching the same capability on install it once."""
        self._index = SessionIndex(home / "sessions")
        self._client: Client | None = None
        self.sessions: dict[str, LiveSession] = {}
        self.unbound: dict[str, UnboundSession] = {}
        self._load_catalog = catalog
        self.catalog: Catalog | None = None
        self._catalog_at = 0.0
        self.catalog_task: asyncio.Task[None] | None = None
        self.stopping = False
        """Set when stdin closed: runs in progress are aborted and no new one starts."""

    def on_connect(self, conn: Client) -> None:
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=min(protocol_version, PROTOCOL_VERSION),
            agent_capabilities=AgentCapabilities(
                load_session=True,
                prompt_capabilities=PromptCapabilities(image=True, embedded_context=True),
                session_capabilities=SessionCapabilities(close=SessionCloseCapabilities()),
            ),
            agent_info=Implementation(name=AGENT_NAME, title=AGENT_TITLE, version=__version__),
        )

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        return None

    # Sessions

    def _request(
        self, cwd: Path, journal: Path | None, system2: str | None, emit: EventSink, caps: CapabilitySettings
    ) -> OpenRequest:
        return OpenRequest(cwd, journal, system2, emit, None if caps == self._capabilities else caps)

    async def _start(
        self,
        cwd: str,
        journal: Path | None,
        system2: str | None,
        caps: CapabilitySettings,
        session_id: str | None = None,
    ) -> LiveSession:
        """Open a harness session. Raises System2Unavailable when its System 2 cannot bind."""
        queue: asyncio.Queue[Event] = asyncio.Queue()
        try:
            session = await self._open(self._request(Path(cwd), journal, system2, queue.put_nowait, caps))
        except System2Unavailable:
            raise
        except ConfigError as e:
            raise RequestError(-32602, str(e)) from None
        packs = tuple(session.deps.system2.profile.tool_packs) if caps.tool_packs is None else DEFAULT_PACKS
        live = LiveSession(session_id or session.deps.journal.session_id, session, caps, packs, queue)
        live.pump = asyncio.create_task(self._pump(live), name=f"acp-updates-{live.id}")
        self.sessions[live.id] = live
        return live

    def _picks(
        self, journal: Journal | None, picks: Mapping[str, bool] | None, runner: str | None = None
    ) -> CapabilitySettings:
        """The configured settings with the journal's last toggles (those still offered), then the client's,
        leaving off each that cannot switch on here, so the session still opens and its options show it off."""
        caps = self._capabilities
        if journal is not None and (last := journal.last_capabilities()) is not None:
            offered = self._values(caps, DEFAULT_PACKS)
            kept = {toggle_id: on for toggle_id, on in last.items() if toggle_id in offered}
            caps = capability_toggles.applied(caps, self._capabilities, kept, DEFAULT_PACKS)
        try:
            caps = capability_toggles.applied(caps, self._capabilities, picks or {}, DEFAULT_PACKS)
        except KeyError as e:
            raise RequestError(-32602, f"_meta.nace.capabilities: {e.args[0]}") from None
        caps = self._startable(self._capabilities, caps, DEFAULT_PACKS)[0]
        if runner is not None:
            return self._with_runner(caps, runner, "_meta.nace.runner: ")
        last = journal.last_runner() if journal is not None else None
        if last is not None and last != caps.runner.name and runner_problem(last, self._capabilities.runner) is None:
            # A loaded session continues on the runner it last ran on, while that can still run here.
            return self._with_runner(caps, last)
        return caps

    def _with_runner(self, caps: CapabilitySettings, name: str, prefix: str = "") -> CapabilitySettings:
        """`caps` on runner `name`; RequestError -32602 with the reason when it cannot run here."""
        if not isinstance(name, str) or not name:
            raise RequestError(-32602, f"{prefix}{RUNNER} takes a runner name")
        if (reason := runner_problem(name, self._capabilities.runner)) is not None:
            raise RequestError(-32602, f"{prefix}{reason}")
        return caps.model_copy(update={"runner": runner_config(name, self._capabilities.runner)})

    def _state(self, caps: CapabilitySettings, packs: Sequence[str]) -> tuple[dict[str, bool], str]:
        """What a session is open on, for telling whether a change changes it: its toggles and its runner."""
        return self._values(caps, packs), caps.runner.name

    def _values(self, caps: CapabilitySettings, packs: Sequence[str]) -> dict[str, bool]:
        return capability_toggles.values(caps, self._capabilities, packs)

    def _record(self, live: LiveSession) -> None:
        """Journal the toggles the session is open on when they differ from the last ones journaled (or from the
        configured ones), so a load opens on them."""
        journal = live.session.deps.journal
        now = self._values(live.capabilities, live.packs)
        runner = live.capabilities.runner.name
        last = journal.last_capabilities()
        last_runner = journal.last_runner() or self._capabilities.runner.name
        if now != (last if last is not None else self._values(self._capabilities, live.packs)) or runner != last_runner:
            journal.append(CapabilitiesChangeEntry(toggles=now, runner=runner))

    async def _open_session(
        self,
        cwd: str,
        journal: Path | None,
        system2: str | None,
        system1: str | None,
        caps: CapabilitySettings,
        session_id: str | None = None,
    ) -> LiveSession | UnboundSession:
        """A live session on `system2` (the configured default when None), or an unbound one when it cannot bind."""
        try:
            live = await self._start(cwd, journal, system2, caps, session_id)
        except System2Unavailable as e:
            # A new session's entry is written now, so the id the client keeps finds this journal on load.
            unbound = UnboundSession(
                id=session_id or Journal(cwd=cwd, path=journal).session_id,
                cwd=cwd,
                journal=journal,
                system2=e.ref,
                system1=system1 or e.system1,
                reason=str(e),
                capabilities=caps,
            )
            self.unbound[unbound.id] = unbound
            return unbound
        await self._settle(live, picked=system2 is not None, system1=system1)
        return live

    async def _settle(self, live: LiveSession, *, picked: bool, system1: str | None, window: str | None = None) -> None:
        """Journal a System 2 and toggles the client picked at open, so a later load continues on them; switch
        System 1 and apply a window picked while unbound."""
        session = live.session
        ref = session.deps.system2.ref
        if picked and session.deps.journal.last_models().get("system2") != ref:
            session.deps.journal.append(ModelChangeEntry(role="system2", model=ref))
        self._record(live)
        if system1 and system1 != session.system1:
            await session.set_model(system1, "system1")
        if window is not None:
            session.set_window_cap(window_cap(window))

    async def _bind(self, unbound: UnboundSession, system2: str) -> LiveSession:
        """Open an unbound session on the model the client picked; it stays unbound if that cannot bind either."""
        if unbound.opening:
            raise RequestError(SESSION_BUSY, "the session is opening on another model")
        unbound.opening = True
        try:
            live = await self._start(unbound.cwd, unbound.journal, system2, unbound.capabilities, unbound.id)
        except System2Unavailable as e:
            raise RequestError(-32602, str(e)) from None
        finally:
            unbound.opening = False
        self.unbound.pop(unbound.id, None)
        await self._settle(live, picked=True, system1=unbound.system1, window=unbound.window)
        return live

    async def _pump(self, live: LiveSession) -> None:
        while True:
            event = await live.queue.get()
            try:
                live.observe(event)
                for update in acp_mapping.event_updates(event, live.mapping()):
                    await self._send(live.id, update)
            except Exception as e:  # a client that cannot take an update must not stop the run
                print(f"{names.NAME}: could not send an update: {type(e).__name__}: {e}", file=sys.stderr)
            finally:
                live.queue.task_done()

    async def _send(self, session_id: str, update: Any) -> None:
        if self._client is not None:
            await self._client.session_update(session_id=session_id, update=update)

    def _journal_path(self) -> Path | None:
        if not self._persist:
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return self._home / "sessions" / f"{stamp}-{os.getpid()}-{uuid.uuid4().hex[:6]}.jsonl"

    async def new_session(
        self, cwd: str, mcp_servers: list[McpServer] | None = None, **kwargs: Any
    ) -> NewSessionResponse:
        meta = session_meta(kwargs)
        caps = self._picks(None, meta.capabilities, meta.runner)
        path = self._journal_path()
        opened = await self._open_session(cwd, path, meta.system2 or None, meta.system1 or None, caps)
        self._refresh_catalog()
        if path is not None:
            self._index.remember(opened.id, path)
        return NewSessionResponse(session_id=opened.id, config_options=self._options(opened), modes=self._modes())

    async def load_session(
        self, cwd: str, session_id: str, mcp_servers: list[McpServer] | None = None, **kwargs: Any
    ) -> LoadSessionResponse:
        meta = session_meta(kwargs)
        if session_id in self.sessions:
            await self._close(self.sessions.pop(session_id))
        self.unbound.pop(session_id, None)
        path = self._index.find(session_id)
        if path is None:
            raise RequestError.resource_not_found(session_id)
        journal = Journal(cwd=cwd, path=path)
        models = journal.last_models()
        system2 = meta.system2 or models.get("system2")
        caps = self._picks(journal, meta.capabilities, meta.runner)
        opened = await self._open_session(cwd, path, system2, meta.system1 or models.get("system1"), caps, session_id)
        self._refresh_catalog()
        if isinstance(opened, LiveSession):
            path_entries, mapping = opened.session.deps.journal.active_path(), opened.mapping()
        else:
            path_entries, mapping = journal.active_path(), _replay_context(Path(cwd))
        for update in acp_mapping.replay_updates(path_entries, mapping):
            await self._send(opened.id, update)
        spent = read_spend(self._home, path)
        if isinstance(opened, LiveSession) and spent.runs:
            opened.cost_usd += spent.cost_usd
            await self._send(opened.id, acp_mapping.steward_update(opened.mapping(), spent))
        return LoadSessionResponse(config_options=self._options(opened), modes=self._modes())

    def _live(self, session_id: str) -> LiveSession:
        live = self.sessions.get(session_id)
        if live is None:
            raise RequestError.resource_not_found(session_id)
        return live

    async def _close(self, live: LiveSession) -> None:
        try:
            await live.session.aclose()
        finally:
            with contextlib.suppress(Exception):
                async with asyncio.timeout(1):
                    await live.flush()
            if live.pump is not None:
                live.pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await live.pump

    # The model catalog: fetched in the background, so opening a session never waits on provider listings.

    def _refresh_catalog(self) -> None:
        if self._load_catalog is None or (self.catalog_task is not None and not self.catalog_task.done()):
            return
        if self.catalog is not None and time.monotonic() - self._catalog_at < CACHE_TTL_S:
            return
        self.catalog_task = asyncio.create_task(self._fetch_catalog(self._load_catalog), name="acp-catalog")

    async def _fetch_catalog(self, load: CatalogLoader) -> None:
        try:
            self.catalog = await load()
        except Exception as e:
            print(f"{names.NAME}: could not list models: {type(e).__name__}: {e}", file=sys.stderr)
            return
        self._catalog_at = time.monotonic()
        for opened in [*self.sessions.values(), *self.unbound.values()]:
            await self._send_options(opened)

    async def _send_options(self, opened: LiveSession | UnboundSession) -> None:
        update = ConfigOptionUpdate(session_update="config_option_update", config_options=self._options(opened))
        await self._send(opened.id, update)

    def stop(self) -> None:
        """The client is gone: abort every run in progress, so each pending prompt answers `cancelled`."""
        self.stopping = True
        for live in self.sessions.values():
            if live.installing is not None:
                live.installing.cancel()
            if live.session.running:
                live.session.abort()

    async def aclose(self) -> None:
        """Abort every run in progress and close every session."""
        if self.catalog_task is not None:
            self.catalog_task.cancel()
        self.unbound = {}
        sessions, self.sessions = list(self.sessions.values()), {}
        await asyncio.gather(*(self._close(live) for live in sessions), return_exceptions=True)

    # Turns

    async def prompt(
        self, prompt: list[PromptBlock], session_id: str, message_id: str | None = None, **kwargs: Any
    ) -> PromptResponse:
        if (unbound := self.unbound.get(session_id)) is not None:
            raise RequestError(-32602, f"no System 2 model can run: {unbound.reason}; pick a model")
        live = self._live(session_id)
        if self.stopping:
            return PromptResponse(stop_reason="cancelled")
        if live.pending is not None:
            live.cancelled = False
            await self._reopen(live)
            if live.cancelled or self.stopping:
                return PromptResponse(stop_reason="cancelled")
        session = live.session
        text = prompt_text(prompt, session.deps.spill_dir / "attachments")
        before = session.usage
        # Shared across System 1 switches, so the difference is this prompt's asks, tools' own included.
        system1_before = session.deps.decisions.usage.model_copy()
        try:
            # No await before run(): a cancel that follows this prompt finds the run started.
            outcome = await session.run(text)
        except SessionBusy as e:
            raise RequestError(SESSION_BUSY, str(e)) from None
        await live.flush()
        reason = acp_mapping.stop_reason(outcome.reason)
        if reason is None:
            if outcome.reason == "provider_error":
                failure = acp_mapping.provider_failure(outcome.detail)
                raise RequestError(PROVIDER_ERROR, failure.detail, failure.model_dump())
            raise RequestError(-32603, outcome.detail or outcome.reason, {"reason": outcome.reason})
        used = session.usage - before
        usage = acp_mapping.prompt_usage(
            session.deps.system2.ref,
            used,
            session.system1,
            session.deps.decisions.usage.since(system1_before),
        )
        return PromptResponse(
            field_meta={"nace": {"usage": usage.model_dump(by_alias=True)}},
            stop_reason=reason,
            usage=Usage(
                input_tokens=used.input_tokens,
                output_tokens=used.output_tokens,
                cached_read_tokens=used.cache_read_tokens,
                cached_write_tokens=used.cache_write_tokens,
                thought_tokens=used.reasoning_tokens,
                total_tokens=used.input_tokens + used.output_tokens,
            ),
        )

    async def _reopen(self, live: LiveSession) -> None:
        """Close the harness session and open it again on the same journal with the pending toggles, keeping its
        models, thinking level, window cap and usage. A toggle switched on gets its missing extra installed first.
        A toggle that still cannot switch on here stays off and the prompt fails naming it, after the other changes
        apply. When the session cannot open on the rest, it opens on the toggles it had and the prompt fails with
        the reason. A cancel during an install stops it and keeps the changes pending."""
        assert live.pending is not None
        old = live.session
        live.reopening = True
        try:
            installed = await self._install_needs(live, live.pending)
            if installed is None:
                return
            pending, failed = installed
            picks, refused = self._startable(live.capabilities, pending, live.packs)
            refused = {**failed, **refused}
            live.pending = None
            changed = self._state(picks, live.packs) != self._state(live.capabilities, live.packs)
            if changed:
                await old.aclose(reopen=True)
                try:
                    live.session = await self._open_again(live, old, picks)
                except Exception as e:
                    reason = (str(e) or type(e).__name__).rstrip(".")
                    try:
                        live.session = await self._open_again(live, old, live.capabilities)
                    except Exception as again:
                        self.sessions.pop(live.id, None)
                        if live.pump is not None:
                            live.pump.cancel()
                        raise RequestError(-32603, f"the session could not reopen: {again}") from None
                    await self._send_options(live)  # the client was shown the change as pending
                    raise RequestError(
                        -32602, f"capabilities not switched: {reason}; the session continues on the ones it had"
                    ) from None
                live.capabilities = picks
                self._record(live)
            if refused:
                await self._send_options(live)
                names = " and ".join(refused)
                applied = "; the other capability changes applied" if changed else ""
                reasons = "; ".join(refused.values())
                raise RequestError(
                    -32602, f"{reasons}. {names} {'stays' if len(refused) == 1 else 'stay'} off{applied}"
                )
        finally:
            live.reopening = False

    async def _install_needs(
        self, live: LiveSession, after: CapabilitySettings
    ) -> tuple[CapabilitySettings, dict[str, str]] | None:
        """`after` once each toggle it switches on has what its capability needs installed (its extra, then setup
        such as the browser's Chromium), with progress in the session. A failed install leaves its toggle off, and
        why, by toggle name. None when the prompt was cancelled meanwhile."""
        was = self._values(live.capabilities, live.packs)
        failed: dict[str, str] = {}
        for toggle in capability_toggles.toggles(after, self._capabilities, live.packs):
            if not toggle.on or was.get(toggle.id, False) or (needs := self._installable(toggle.id)) is None:
                continue
            try:
                if not await self._run_install(live, needs):
                    return None
            except InstallFailed as error:
                after = self._flip(after, toggle.id, False, live.packs)
                failed[toggle.name] = str(error).rstrip(".")
        return after, failed

    def _installable(self, toggle_id: str) -> Needs | None:
        """What switching `toggle_id` on installs: its capability's needs, when this server installs and can."""
        needs = capability_needs(toggle_id)
        return needs if self._install is not None and needs is not None and needs.installable() else None

    async def _run_install(self, live: LiveSession, needs: Needs) -> bool:
        """Install `needs` as a task `session/cancel` can stop, saying what it does as agent text. False when the
        prompt was cancelled first; raises InstallFailed."""
        assert self._install is not None
        if live.cancelled or self.stopping:
            return False
        run_install = self._install

        async def say(line: str) -> None:
            await self._send(live.id, update_agent_message_text(f"{line}\n\n"))

        async def run() -> None:
            async with self._installs:
                await run_install(needs, say)

        task = asyncio.ensure_future(run())
        live.installing = task
        try:
            await asyncio.wait({task})
        finally:
            live.installing = None
            task.cancel()  # a no-op once done; otherwise the prompt itself was cancelled
        if task.cancelled():
            return False
        task.result()
        return True

    def _offered(self, toggle_id: str, on: bool) -> bool:
        """Whether a session lists the toggle: one that can work here, at most after an install, or one that is on,
        so it can be switched off."""
        return on or self._unavailable(toggle_id) is None or self._installable(toggle_id) is not None

    def _startable(
        self, before: CapabilitySettings, after: CapabilitySettings, packs: Sequence[str]
    ) -> tuple[CapabilitySettings, dict[str, str]]:
        """`after` with each toggle it switches on that cannot switch on here left off, and why, by toggle name."""
        was = self._values(before, packs)
        refused: dict[str, str] = {}
        for toggle in capability_toggles.toggles(after, self._capabilities, packs):
            if toggle.on and not was.get(toggle.id, False) and (reason := self._unavailable(toggle.id)) is not None:
                after = self._flip(after, toggle.id, False, packs)
                refused[toggle.name] = reason.rstrip(".")
        return after, refused

    async def _open_again(self, live: LiveSession, old: Session, caps: CapabilitySettings) -> Session:
        deps = old.deps
        request = self._request(deps.cwd, deps.journal.path, deps.system2.ref, live.queue.put_nowait, caps)
        session = await self._open(request)
        try:
            if old.system1 != session.system1:
                await session.set_model(old.system1, "system1")
            effort = deps.system2.profile.reasoning_effort
            binding = session.deps.system2
            if deps.system2.takes_effort and binding.takes_effort:
                binding.profile = binding.profile.model_copy(update={"reasoning_effort": effort})
        except BaseException:
            await session.aclose(reopen=True)
            raise
        return session

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        live = self.sessions.get(session_id)
        if live is not None and live.reopening:
            live.cancelled = True
            if live.installing is not None:
                live.installing.cancel()
        elif live is not None and live.session.running:
            live.session.abort()

    # Configuration

    def _modes(self) -> SessionModeState:
        mode = SessionMode(id=FULL_ACCESS, name="Full access", description="The agent runs every tool without asking.")
        return SessionModeState(available_modes=[mode], current_mode_id=FULL_ACCESS)

    def _model_choices(self, current: str) -> list[SessionConfigSelectGroup]:
        """System 2 models grouped by provider, the current one always among them."""
        models = list(self.catalog.system2) if self.catalog is not None else []
        if current and all(model.id != current for model in models):
            models.append(ModelInfo(id=current, name=current))
        groups: dict[str, list[SessionConfigSelectOption]] = {}
        for model in models:
            provider = model.id.partition("/")[0]
            groups.setdefault(provider, []).append(SessionConfigSelectOption(value=model.id, name=model.name))
        return [
            SessionConfigSelectGroup(group=provider, name=provider, options=choices)
            for provider, choices in sorted(groups.items())
        ]

    def _system1_choices(self, current: str) -> list[SessionConfigSelectOption]:
        """System 1 models from the catalog (the console's decision models, then the decisions API's), the
        current one always among them."""
        models = list(self.catalog.system1) if self.catalog is not None else []
        if all(model.id != current for model in models):
            models.append(ModelInfo(id=current, name=current))
        return [SessionConfigSelectOption(value=model.id, name=model.name) for model in models]

    def _options(self, opened: LiveSession | UnboundSession) -> list[ConfigOption]:
        if isinstance(opened, UnboundSession):
            options = self._config_options(opened.system2, opened.system1, None)
            caps = opened.capabilities
            return [*options, self._runner_option(caps), *self._capability_options(caps, DEFAULT_PACKS)]
        binding = opened.session.deps.system2
        options = self._config_options(
            binding.ref, opened.session.system1, binding.profile.reasoning_effort, binding.effort_choices
        )
        caps = opened.pending or opened.capabilities
        return [
            *options,
            self._window_option(opened.session),
            self._runner_option(caps),
            *self._capability_options(caps, opened.packs),
        ]

    def _runner_option(self, caps: CapabilitySettings) -> SessionConfigOptionSelect:
        """Where the session's commands run: one choice among every runner, each value saying why it cannot run here
        when it cannot. A change not yet applied shows as the current value."""
        choices = [
            SessionConfigSelectOption(
                value=choice.id,
                name=choice.name,
                description=f"Unavailable: {choice.unavailable}" if choice.unavailable else choice.description,
            )
            for choice in runner_choices(self._capabilities.runner)
        ]
        return SessionConfigOptionSelect(
            id=RUNNER,
            name="Runner",
            category=RUNNER_CATEGORY,
            description="Where the agent's commands, file tools and background jobs run.",
            type="select",
            current_value=caps.runner.name,
            options=choices,
        )

    def _capability_options(self, caps: CapabilitySettings, packs: Sequence[str]) -> list[ConfigOption]:
        """One boolean per capability toggle that can work here, showing a change not yet applied as its value."""
        return [
            SessionConfigOptionBoolean(
                id=toggle.id,
                name=toggle.name,
                category=CAPABILITY_CATEGORY,
                description=toggle.note,
                type="boolean",
                current_value=toggle.on,
            )
            for toggle in capability_toggles.toggles(caps, self._capabilities, packs)
            if self._offered(toggle.id, toggle.on)
        ]

    def _window_option(self, session: Session) -> SessionConfigOptionSelect:
        """`auto` (the `--window` default) and the presets the model can use, the current cap always among them."""
        binding = session.deps.system2
        window = binding.window
        usable = min(window.advertised, window.effective or window.advertised)
        auto = recapped(window, binding.profile, session.default_window_cap).working
        caps = {preset for preset in WINDOW_PRESETS if preset <= usable}
        if session.window_cap is not None:
            caps.add(session.window_cap)
        choices = [SessionConfigSelectOption(value=AUTO_WINDOW, name=f"Auto ({tokens_label(auto)})")]
        choices += [SessionConfigSelectOption(value=str(cap), name=tokens_label(cap)) for cap in sorted(caps)]
        return SessionConfigOptionSelect(
            id=CONTEXT_WINDOW,
            name="Context window",
            category="model_config",
            description="The most context System 2 is given; auto is the model's window, or --window when set.",
            type="select",
            current_value=AUTO_WINDOW if session.window_cap is None else str(session.window_cap),
            options=choices,
        )

    def _config_options(
        self, system2: str, system1: str, effort: str | None, choices: tuple[str, ...] = ()
    ) -> list[ConfigOption]:
        options: list[ConfigOption] = [
            SessionConfigOptionSelect(
                id="model",
                name="Model",
                category="model",
                description="System 2: the agent model that reasons and calls tools.",
                type="select",
                current_value=system2,
                options=self._model_choices(system2),
            ),
            SessionConfigOptionSelect(
                id="system1",
                name="System 1",
                category="model_config",
                description="The fast decision model: a decision model from the console, or decisions:<model> on the "
                "decisions API.",
                type="select",
                current_value=system1,
                options=self._system1_choices(system1),
            ),
        ]
        if choices:
            options.append(
                SessionConfigOptionSelect(
                    id="thought_level",
                    name="Thinking",
                    category="thought_level",
                    description="The reasoning effort sent with each request; default sends none.",
                    type="select",
                    current_value=effort if effort in choices else choices[0],
                    options=[SessionConfigSelectOption(value=level, name=level.title()) for level in choices],
                )
            )
        return options

    async def set_config_option(
        self, config_id: str, session_id: str, value: str | bool, **kwargs: Any
    ) -> SetSessionConfigOptionResponse:
        if config_id.startswith(capability_toggles.PREFIX):
            return await self._set_capability(config_id, session_id, value)
        if config_id == RUNNER:
            return await self._set_runner(session_id, value)
        if (unbound := self.unbound.get(session_id)) is not None:
            if not isinstance(value, str):
                raise RequestError(-32602, f"{config_id} takes a string value")
            opened = await self._configure_unbound(unbound, config_id, value)
            return SetSessionConfigOptionResponse(config_options=self._options(opened))
        live = self._live(session_id)
        session = live.session
        if not isinstance(value, str):
            raise RequestError(-32602, f"{config_id} takes a string value")
        if session.running:
            raise RequestError(SESSION_BUSY, "cannot change configuration while a run is in progress")
        try:
            if config_id == "model":
                await session.set_model(value, "system2")
            elif config_id == "system1":
                if not value:
                    raise RequestError(-32602, "system1 takes a model name")
                await session.set_model(value, "system1")
            elif config_id == "thought_level" and session.deps.system2.takes_effort:
                binding = session.deps.system2
                if value not in binding.effort_choices:
                    raise RequestError(-32602, f"thought_level takes one of {', '.join(binding.effort_choices)}")
                effort = None if value == DEFAULT_EFFORT else value
                binding.profile = binding.profile.model_copy(update={"reasoning_effort": effort})
            elif config_id == CONTEXT_WINDOW:
                session.set_window_cap(window_cap(value))
            else:
                raise RequestError(-32602, f"unknown config option {config_id!r}")
        except SessionBusy as e:
            raise RequestError(SESSION_BUSY, str(e)) from None
        except (ConfigError, ValueError, KeyError) as e:
            raise RequestError(-32602, str(e)) from None
        return SetSessionConfigOptionResponse(config_options=self._options(live))

    async def _set_capability(
        self, config_id: str, session_id: str, value: str | bool
    ) -> SetSessionConfigOptionResponse:
        """Mark a toggle pending; the next prompt reopens the session once for every change made by then. Setting
        the value in force changes nothing, nor does restating the runner in force while another is pending."""
        if not isinstance(value, bool):
            raise RequestError(-32602, f"{config_id} takes true or false")
        if (unbound := self.unbound.get(session_id)) is not None:
            unbound.capabilities = self._flip(unbound.capabilities, config_id, value, DEFAULT_PACKS)
            return SetSessionConfigOptionResponse(config_options=self._options(unbound))
        live = self._live(session_id)
        self._pend(live, self._flip(live.pending or live.capabilities, config_id, value, live.packs))
        return SetSessionConfigOptionResponse(config_options=self._options(live))

    async def _set_runner(self, session_id: str, value: str | bool) -> SetSessionConfigOptionResponse:
        """Pick where the session's commands run. A runner that cannot run here is refused now, with why; one that
        can applies at the next prompt, which reopens the session once for every change made by then."""
        if not isinstance(value, str):
            raise RequestError(-32602, f"{RUNNER} takes a runner name")
        if (unbound := self.unbound.get(session_id)) is not None:
            unbound.capabilities = self._with_runner(unbound.capabilities, value)
            return SetSessionConfigOptionResponse(config_options=self._options(unbound))
        live = self._live(session_id)
        before = live.pending or live.capabilities
        if value != before.runner.name:
            self._pend(live, self._with_runner(before, value))
        return SetSessionConfigOptionResponse(config_options=self._options(live))

    def _pend(self, live: LiveSession, after: CapabilitySettings) -> None:
        """Hold `after` for the next prompt; setting what is in force (or pending) changes nothing."""
        if live.session.running or live.reopening:
            raise RequestError(SESSION_BUSY, "cannot change configuration while a run is in progress")
        before = live.pending or live.capabilities
        if self._state(after, live.packs) == self._state(before, live.packs):
            return
        if live.session.deps.journal.path is None:
            raise RequestError(-32602, "capabilities and the runner can switch only on a session kept in a journal")
        unchanged = self._state(after, live.packs) == self._state(live.capabilities, live.packs)
        live.pending = None if unchanged else after

    def _flip(self, caps: CapabilitySettings, config_id: str, on: bool, packs: Sequence[str]) -> CapabilitySettings:
        if config_id not in self._values(caps, packs) and config_id not in capability_toggles.RETIRED:
            raise RequestError(-32602, f"unknown config option {config_id!r}")
        return capability_toggles.flip(caps, self._capabilities, config_id, on, packs)

    async def _configure_unbound(
        self, unbound: UnboundSession, config_id: str, value: str
    ) -> LiveSession | UnboundSession:
        """`model` opens the session on the picked model; `system1` and `context_window` are kept for when it
        opens."""
        if config_id == "model":
            return await self._bind(unbound, value)
        if config_id == "system1":
            if not value:
                raise RequestError(-32602, "system1 takes a model name")
            unbound.system1 = value
            return unbound
        if config_id == CONTEXT_WINDOW:
            window_cap(value)
            unbound.window = value
            return unbound
        raise RequestError(-32602, f"unknown config option {config_id!r}")

    async def set_session_mode(self, mode_id: str, session_id: str, **kwargs: Any) -> SetSessionModeResponse:
        if session_id not in self.unbound:
            self._live(session_id)
        if mode_id != FULL_ACCESS:
            raise RequestError(-32602, f"only the {FULL_ACCESS} mode is available")
        return SetSessionModeResponse()

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        """Unstable in ACP; advertised so clients close sessions instead of killing the process."""
        if self.unbound.pop(session_id, None) is not None:
            return CloseSessionResponse()
        await self._close(self._live(session_id))
        self.sessions.pop(session_id, None)
        return CloseSessionResponse()

    # Not offered in v1; the router answers clients that call them anyway.

    async def list_sessions(
        self, cursor: str | None = None, cwd: str | None = None, **kwargs: Any
    ) -> ListSessionsResponse:
        raise RequestError.method_not_found("session/list")

    async def set_session_model(self, model_id: str, session_id: str, **kwargs: Any) -> SetSessionModelResponse:
        raise RequestError.method_not_found("session/set_model")

    async def fork_session(
        self, cwd: str, session_id: str, mcp_servers: list[McpServer] | None = None, **kwargs: Any
    ) -> ForkSessionResponse:
        raise RequestError.method_not_found("session/fork")

    async def resume_session(
        self, cwd: str, session_id: str, mcp_servers: list[McpServer] | None = None, **kwargs: Any
    ) -> ResumeSessionResponse:
        raise RequestError.method_not_found("session/resume")

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


def _streamable(stream: TextIO) -> bool:
    """A pipe, socket or terminal: what the event loop can stream. Files and /dev/null it cannot (on POSIX)."""
    if os.name != "posix":
        return True
    try:
        mode = os.fstat(stream.fileno()).st_mode
    except (OSError, ValueError):
        return False
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stream.isatty()


class PendingReplies:
    """Requests the client sent that have not been answered yet, seen through the SDK's stream observers: a
    request is counted when it arrives and settled once its reply has been written."""

    def __init__(self) -> None:
        self._ids: set[Any] = set()
        self._settled = asyncio.Event()
        self._settled.set()

    def observe(self, event: StreamEvent) -> None:
        message = event.message
        if "id" not in message:
            return  # a notification
        if event.direction is StreamDirection.INCOMING and "method" in message:
            self._ids.add(message["id"])
        elif event.direction is StreamDirection.OUTGOING and "method" not in message:
            self._ids.discard(message["id"])
        if self._ids:
            self._settled.clear()
        else:
            self._settled.set()

    async def settled(self) -> None:
        await self._settled.wait()


async def serve(agent: AcpAgent) -> None:
    """Serve ACP on stdin/stdout until stdin closes or SIGTERM, then close every session.

    When stdin closes, requests already received still get their replies: runs in progress are aborted (their
    prompts answer `cancelled`), and the server exits once every reply is written, or after DRAIN_TIMEOUT_S.
    """
    loop = asyncio.get_running_loop()
    main = asyncio.current_task()
    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGTERM, lambda: main.cancel() if main else None)
    stdout = sys.stdout
    try:
        if not _streamable(sys.stdin) or not _streamable(sys.stdout):
            raise ConfigError(
                "acp talks over pipes: start it from an ACP client, not with stdin or stdout redirected to a file"
            )
        reader, writer = await stdio_streams(limit=STDIO_LIMIT_BYTES)
        if os.name == "posix":
            # The transport holds the stdout descriptor now; stray prints go to stderr, never into the protocol.
            sys.stdout = sys.stderr
        replies = PendingReplies()
        # session/close, which stewards a session's memory, is in the library's unstable protocol.
        await run_agent(
            agent, input_stream=writer, output_stream=reader, observers=[replies.observe], use_unstable_protocol=True
        )
        agent.stop()
        with contextlib.suppress(TimeoutError):
            async with asyncio.timeout(DRAIN_TIMEOUT_S):
                await replies.settled()
    except asyncio.CancelledError:
        pass
    finally:
        sys.stdout = stdout
        await agent.aclose()
        with contextlib.suppress(NotImplementedError):
            loop.remove_signal_handler(signal.SIGTERM)


def main(options: Options, env: Mapping[str, str] | None = None) -> None:
    environment = process_env() if env is None else dict(env)
    bootstrap = Bootstrap(options, environment)
    capabilities = bootstrap.capability_settings()
    agent = AcpAgent(
        bootstrap_opener(options, environment),
        home_dir(environment),
        persist=options.persist,
        catalog=bootstrap.catalog,
        capabilities=capabilities,
        unavailable=lambda toggle_id: capability_problem(toggle_id, capabilities, environment),
        install=install,
    )
    asyncio.run(serve(agent))
