"""The ACP server: in process over scripted providers, and as a process driven by the ACP SDK's client."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from acp import RequestError, spawn_agent_process, text_block
from acp.schema import (
    AgentMessageChunk,
    AgentThoughtChunk,
    ConfigOptionUpdate,
    ContentToolCallContent,
    FileEditToolCallContent,
    ImageContentBlock,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    UsageUpdate,
    UserMessageChunk,
)
from pydantic import BaseModel

from intern import capability_toggles
from intern.capability_settings import CapabilitySettings, McpConfig
from intern.config import Bootstrap, Options, System2Unavailable
from intern.decisions import (
    Answers,
    DecisionBackend,
    DecisionClient,
    DecisionsApiBackend,
    NoulAnswer,
    NoulQuestion,
    Question,
)
from intern.events import ContextBudget, Nudge, ReasoningDelta, RunEnd, RunStart, TextDelta, TurnStart, WindowInfo
from intern.events import ToolCallEnd as ToolCallEndEvent
from intern.events import ToolCallStart as ToolCallStartEvent
from intern.executor import LocalExecutor
from intern.hooks import Extension, Hooks, RunContext, TurnInfo
from intern.instructions import FileInstructions
from intern.journal import Journal, MessageEntry
from intern.loop import EngineDeps, ModelBinding
from intern.messages import (
    AssistantMessage,
    FileDiff,
    ToolCall,
    ToolResult,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from intern.models import catalog
from intern.models.profile import EFFORT_LEVELS, Profile
from intern.models.registry import ConfigError, ProviderSpec
from intern.models.window import working_window
from intern.needs import InstallFailed, Needs, Say
from intern.protocol import acp_mapping
from intern.protocol.acp import AcpAgent, Installer, OpenRequest, SessionIndex, prompt_text
from intern.providers.fake import FakeProvider, Script, Turn
from intern.session import Session
from intern.tools import load_packs
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput
from intern.tools.web_search import WebSearchTool

from .fake_openai import MODEL, FakeOpenAI, Reply
from .support import Quiet, binding


class WaitArgs(BaseModel):
    pass


class WaitTool(Tool):
    """Runs until the run is aborted, so a cancel always lands mid-run."""

    name = "wait"
    description = "Wait."
    Args = WaitArgs
    effects: ClassVar[frozenset[Effect]] = frozenset()

    def __init__(self, waiting: asyncio.Event) -> None:
        self.waiting = waiting

    async def run(self, args: WaitArgs, ctx: ToolContext) -> ToolOutput:
        self.waiting.set()
        await ctx.abort.wait()
        return ToolOutput(content="stopped waiting")


@dataclass
class FakeClient:
    """Records every update the agent sends, per session."""

    updates: list[tuple[str, Any]] = field(default_factory=list)

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))

    def of(self, kind: type) -> list[Any]:
        return [u for _, u in self.updates if isinstance(u, kind)]

    def text(self) -> str:
        return "".join(u.content.text for u in self.of(AgentMessageChunk))


@dataclass
class Opener:
    """Opens sessions over a scripted provider; models resolve to bindings with that ref and profile."""

    script: Script
    tmp: Path
    profiles: dict[str, Profile] = field(default_factory=dict)
    unbindable: set[str] = field(default_factory=set)
    """Models whose key is missing: opening a session on one fails as Bootstrap's does."""
    providers: dict[str, type[FakeProvider]] = field(default_factory=dict)
    """A provider of their own for some refs (one whose API takes effort levels); the scripted one for the rest."""
    windows: dict[str, int] = field(default_factory=dict)
    """Advertised windows by model; 32k for the rest."""
    window_cap: int | None = None
    """`--window`: every binding is capped by it, as Bootstrap's are."""
    opened: list[OpenRequest] = field(default_factory=list)
    waiting: asyncio.Event = field(default_factory=asyncio.Event)
    system1: DecisionBackend = field(default_factory=Quiet)
    """The System 1 sessions open on."""
    extensions: list[Extension] = field(default_factory=list)
    """Set once a run is inside the `wait` tool."""
    broken: dict[str, str] = field(default_factory=dict)
    """Capability toggles a session cannot open with, and the error it fails with."""
    closed: list[str] = field(default_factory=list)
    """The journal of each session whose close hook (what stewards memory) ran."""

    def bind(self, ref: str, provider: FakeProvider) -> ModelBinding:
        model = binding(provider, window=self.windows.get(ref, 32_768), profile=self.profiles.get(ref))
        model.ref = ref
        model.window = working_window(model.window.advertised, "provider", model.profile, self.window_cap)
        return model

    async def __call__(self, request: OpenRequest) -> Session:
        self.opened.append(request)
        ref = request.system2 or "fake/model"
        if ref in self.unbindable:
            raise System2Unavailable(f"{ref} needs an API key", ref=ref, system1=self.system1.name)
        caps = request.capabilities or CapabilitySettings()
        for toggle_id, on in capability_toggles.values(caps, caps, ["base"]).items():
            if on and toggle_id in self.broken:
                raise ConfigError(self.broken[toggle_id])
        provider = FakeProvider(self.script)

        async def resolve(ref: str) -> ModelBinding:
            if not ref.startswith("fake/") or ref in self.unbindable:
                raise ValueError(f"unknown provider in {ref!r}")
            own = self.providers.get(ref)
            return self.bind(ref, own(self.script) if own is not None else provider)

        system2 = self.bind(ref, provider)
        # As Bootstrap does: the profile's packs become the ones in force.
        packs = caps.effective_tool_packs(system2.profile.tool_packs)
        system2.profile = system2.profile.model_copy(update={"tool_packs": packs})
        deps = EngineDeps(
            system2=system2,
            tools=[*load_packs(packs), WaitTool(self.waiting)],
            hooks=Hooks(self.extensions, request.emit),
            journal=Journal(cwd=str(request.cwd), path=request.journal),
            instructions=FileInstructions(self.tmp / "home", context_files=False).load(request.cwd),
            decisions=DecisionClient(self.system1, request.emit),
            executor=LocalExecutor(),
            cwd=request.cwd,
            spill_dir=self.tmp / "spill",
            emit=request.emit,
        )
        return Session(
            deps,
            resolve_model=resolve,
            resolve_system1=self.resolve_system1,
            system1=self.system1.name,
            on_close=lambda session: self.closed.append(str(session.deps.journal.path)),
        )

    def resolve_system1(self, choice: str) -> DecisionBackend:
        return NamedBackend(choice)


class NamedBackend(Quiet):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name


@pytest.fixture
def work(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def make_agent(
    tmp_path: Path,
    script: Script,
    capabilities: CapabilitySettings | None = None,
    unavailable: dict[str, str] | None = None,
    install: Installer | None = None,
    **opener: Any,
) -> tuple[AcpAgent, FakeClient, Opener]:
    """`unavailable`: the capability toggles that cannot switch on here, and why."""
    open_session = Opener(script, tmp_path, **opener)
    agent = AcpAgent(
        open_session,
        tmp_path / "home",
        capabilities=capabilities,
        unavailable=(unavailable or {}).get,
        install=install,
    )
    client = FakeClient()
    agent.on_connect(client)  # pyright: ignore[reportArgumentType]
    return agent, client, open_session


async def test_initialize_advertises_load_session_images_and_the_harness(tmp_path: Path):
    agent, _, _ = make_agent(tmp_path, [])
    response = await agent.initialize(protocol_version=1)
    assert response.protocol_version == 1
    assert response.agent_info is not None
    assert (response.agent_info.name, response.agent_info.version) == (
        "intern",
        __import__("intern").__version__,
    )
    capabilities = response.agent_capabilities
    assert capabilities is not None and capabilities.load_session is True
    assert capabilities.prompt_capabilities is not None and capabilities.prompt_capabilities.image is True


async def test_a_prompt_streams_the_reply_and_ends_the_turn(tmp_path: Path, work: Path):
    agent, client, _ = make_agent(tmp_path, [Turn(text="hello there")])
    session = await agent.new_session(cwd=str(work))
    assert session.modes is not None and [m.id for m in session.modes.available_modes] == ["full-access"]
    response = await agent.prompt(prompt=[text_block("say hello")], session_id=session.session_id)
    assert response.stop_reason == "end_turn"
    assert client.text() == "hello there "
    assert response.usage is not None and response.usage.output_tokens == 10
    assert {sid for sid, _ in client.updates} == {session.session_id}
    await agent.aclose()


@dataclass
class PricedBackend:
    name: str = "drex-test"

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        return Answers(answers={"done": NoulAnswer(noul=1)}, input_tokens=300, output_tokens=20, cost_usd=0.002)


class AsksEachTurn(Extension):
    def __init__(self) -> None:
        self.asked = 0

    async def turn_end(self, turn: TurnInfo, ctx: RunContext) -> None:
        # A distinct State each time: a cached answer counts in no usage.
        self.asked += 1
        question: dict[str, Question] = {"done": NoulQuestion(instructions="Is the task done?")}
        await ctx.decisions.ask("check", f"ask {self.asked}", question)


async def test_a_prompt_reports_each_systems_usage_for_that_prompt(tmp_path: Path, work: Path):
    script = [Turn(text="one"), Turn(text="two")]
    agent, _, _ = make_agent(tmp_path, script, system1=PricedBackend(), extensions=[AsksEachTurn()])
    session = await agent.new_session(cwd=str(work))
    first = await agent.prompt(prompt=[text_block("one")], session_id=session.session_id)
    second = await agent.prompt(prompt=[text_block("two")], session_id=session.session_id)
    for response in (first, second):
        assert response.field_meta is not None and response.usage is not None
        usage = response.field_meta["nace"]["usage"]
        assert usage["system2"]["model"] == "fake/model"
        assert usage["system2"]["outputTokens"] == response.usage.output_tokens > 0
        assert usage["system1"] == {
            "model": "drex-test",
            "asks": 1,
            "failed": 0,
            "inputTokens": 300,
            "outputTokens": 20,
            "costUsd": 0.002,
        }
    await agent.aclose()


async def test_sessions_in_one_process_are_independent(tmp_path: Path, work: Path):
    agent, client, _ = make_agent(tmp_path, lambda request: Turn(text=f"{len(request.messages)} messages"))
    first = await agent.new_session(cwd=str(work))
    second = await agent.new_session(cwd=str(work))
    assert first.session_id != second.session_id
    await agent.prompt(prompt=[text_block("one")], session_id=first.session_id)
    await agent.prompt(prompt=[text_block("two")], session_id=first.session_id)
    await agent.prompt(prompt=[text_block("three")], session_id=second.session_id)
    by_session = {
        sid: "".join(u.content.text for s, u in client.updates if s == sid and isinstance(u, AgentMessageChunk))
        for sid in (first.session_id, second.session_id)
    }
    # The first session's second prompt sees its own history; the second session starts fresh.
    assert by_session[first.session_id] == "2 messages 4 messages "
    assert by_session[second.session_id] == "2 messages "
    await agent.aclose()


async def test_cancel_stops_the_run_and_the_prompt_resolves_cancelled(tmp_path: Path, work: Path):
    agent, _, opener = make_agent(tmp_path, [Turn(tool_calls=[("wait", {})]), Turn(text="never")])
    session = await agent.new_session(cwd=str(work))
    prompt = asyncio.create_task(agent.prompt(prompt=[text_block("wait")], session_id=session.session_id))
    await opener.waiting.wait()
    await agent.cancel(session_id=session.session_id)
    assert (await prompt).stop_reason == "cancelled"
    await agent.aclose()


async def test_cancel_right_after_prompt_cancels_that_prompt(tmp_path: Path, work: Path):
    agent, client, _ = make_agent(tmp_path, [Turn(text="too late")])
    session = await agent.new_session(cwd=str(work))
    # The SDK starts one task per message, in arrival order.
    prompt = asyncio.create_task(agent.prompt(prompt=[text_block("go")], session_id=session.session_id))
    cancel = asyncio.create_task(agent.cancel(session_id=session.session_id))
    response, _ = await asyncio.gather(prompt, cancel)
    assert response.stop_reason == "cancelled"
    assert client.text() == ""
    # A cancel with nothing running is ignored, and the session takes the next prompt.
    await agent.cancel(session_id=session.session_id)
    assert (await agent.prompt(prompt=[text_block("again")], session_id=session.session_id)).stop_reason == "end_turn"
    await agent.aclose()


async def test_a_second_prompt_while_one_runs_is_refused_as_busy(tmp_path: Path, work: Path):
    agent, _, opener = make_agent(tmp_path, [Turn(tool_calls=[("wait", {})])])
    session = await agent.new_session(cwd=str(work))
    first = asyncio.create_task(agent.prompt(prompt=[text_block("wait")], session_id=session.session_id))
    await opener.waiting.wait()
    with pytest.raises(RequestError) as busy:
        await agent.prompt(prompt=[text_block("again")], session_id=session.session_id)
    assert busy.value.code == -32000
    await agent.cancel(session_id=session.session_id)
    await first
    await agent.aclose()


@pytest.mark.parametrize(
    ("error", "kind", "status"),
    [
        ("HTTP 429 from openrouter: slow down", "rate_limit", 429),
        ("HTTP 402 from openrouter: Insufficient credits", "quota", 402),
        ("HTTP 401 from openrouter: bad key", "auth", 401),
        ("HTTP 500 from openrouter: boom", "provider", 500),
    ],
)
async def test_provider_errors_are_json_rpc_errors_with_their_kind(
    tmp_path: Path, work: Path, error: str, kind: str, status: int
):
    agent, _, _ = make_agent(tmp_path, [Turn(error=error)])
    session = await agent.new_session(cwd=str(work))
    with pytest.raises(RequestError) as failed:
        await agent.prompt(prompt=[text_block("go")], session_id=session.session_id)
    assert failed.value.code == -32001
    assert failed.value.data == {"kind": kind, "detail": error, "status": status}
    await agent.aclose()


async def test_unknown_sessions_are_not_found(tmp_path: Path, work: Path):
    agent, _, _ = make_agent(tmp_path, [])
    with pytest.raises(RequestError) as missing:
        await agent.prompt(prompt=[text_block("go")], session_id="nope")
    assert missing.value.code == -32002
    with pytest.raises(RequestError) as unloadable:
        await agent.load_session(cwd=str(work), session_id="nope")
    assert unloadable.value.code == -32002


async def test_load_replays_the_conversation_and_continues_on_its_model(tmp_path: Path, work: Path):
    script = lambda request: Turn(text=f"reply {len(request.messages)}")  # noqa: E731
    agent, _, _ = make_agent(tmp_path, script)
    session = await agent.new_session(cwd=str(work))
    await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/other")
    await agent.prompt(prompt=[text_block("first question")], session_id=session.session_id)
    await agent.aclose()

    # A new process: only the journal on disk and the index are left.
    agent, client, opener = make_agent(tmp_path, script)
    loaded = await agent.load_session(cwd=str(work), session_id=session.session_id)
    assert [u.content.text for u in client.of(UserMessageChunk)] == ["first question"]
    assert client.text() == "reply 2"
    assert opener.opened[-1].system2 == "fake/other"
    assert loaded.config_options is not None and loaded.config_options[0].current_value == "fake/other"
    response = await agent.prompt(prompt=[text_block("second")], session_id=session.session_id)
    assert response.stop_reason == "end_turn" and client.text().endswith("reply 4 ")
    await agent.aclose()


# Sessions whose configured System 2 cannot run (its key is missing): they still open.


async def test_a_session_opens_on_the_model_its_meta_asks_for(tmp_path: Path, work: Path):
    script = lambda request: Turn(text=f"reply {len(request.messages)}")  # noqa: E731
    agent, client, opener = make_agent(tmp_path, script, unbindable={"fake/model"})
    session = await agent.new_session(cwd=str(work), nace={"system2": "fake/other", "system1": "drex-latest"})
    assert session.config_options is not None
    assert [o.current_value for o in settings(session.config_options)] == ["fake/other", "drex-latest", "auto"]
    assert opener.opened[-1].system2 == "fake/other"
    response = await agent.prompt(prompt=[text_block("go")], session_id=session.session_id)
    assert response.stop_reason == "end_turn" and client.text() == "reply 2 "
    await agent.aclose()

    # The picked model is journaled: a load without _meta continues on it.
    agent, _, opener = make_agent(tmp_path, script, unbindable={"fake/model"})
    loaded = await agent.load_session(cwd=str(work), session_id=session.session_id)
    assert opener.opened[-1].system2 == "fake/other"
    assert loaded.config_options is not None
    assert [o.current_value for o in settings(loaded.config_options)] == ["fake/other", "drex-latest", "auto"]
    await agent.aclose()


async def test_an_unbindable_default_opens_unbound_until_a_model_is_picked(tmp_path: Path, work: Path):
    agent, client, opener = make_agent(tmp_path, [Turn(text="ok")], unbindable={"fake/model", "fake/keyless"})
    session = await agent.new_session(cwd=str(work))
    assert session.config_options is not None
    assert [(o.id, o.current_value) for o in settings(session.config_options)] == [
        ("model", "fake/model"),
        ("system1", "quiet"),
    ]
    assert session.modes is not None
    await agent.set_session_mode(mode_id="full-access", session_id=session.session_id)

    with pytest.raises(RequestError) as unbound:
        await agent.prompt(prompt=[text_block("go")], session_id=session.session_id)
    assert unbound.value.code == -32602
    assert str(unbound.value) == "no System 2 model can run: fake/model needs an API key; pick a model"
    with pytest.raises(RequestError) as still:
        await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/keyless")
    assert still.value.code == -32602
    with pytest.raises(RequestError) as unknown:
        await agent.set_config_option(config_id="thought_level", session_id=session.session_id, value="low")
    assert unknown.value.code == -32602

    with pytest.raises(RequestError) as bad_window:
        await agent.set_config_option(config_id="context_window", session_id=session.session_id, value="big")
    assert bad_window.value.code == -32602

    # System 1, a window and toggles picked while unbound are kept; picking a model opens the session on them.
    await agent.set_config_option(config_id="cap.browser", session_id=session.session_id, value=True)
    await agent.set_config_option(config_id="system1", session_id=session.session_id, value="drex-latest")
    await agent.set_config_option(config_id="context_window", session_id=session.session_id, value="32000")
    picked = await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/other")
    assert [o.current_value for o in settings(picked.config_options)] == ["fake/other", "drex-latest", "32000"]
    live = agent.sessions[session.session_id]
    assert live.session.deps.system2.ref == "fake/other" and live.session.system1 == "drex-latest"
    assert live.session.deps.system2.window.working == 32_000
    assert live.capabilities.browser.enabled and opener.opened[-1].capabilities == live.capabilities
    response = await agent.prompt(prompt=[text_block("go")], session_id=session.session_id)
    assert response.stop_reason == "end_turn" and client.text() == "ok "
    await agent.aclose()


async def test_a_loaded_session_whose_model_cannot_run_opens_unbound(tmp_path: Path, work: Path):
    script = lambda request: Turn(text=f"reply {len(request.messages)}")  # noqa: E731
    agent, _, _ = make_agent(tmp_path, script)
    session = await agent.new_session(cwd=str(work))
    await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/other")
    await agent.prompt(prompt=[text_block("first question")], session_id=session.session_id)
    await agent.aclose()

    # Its key is gone now: the session still loads and replays.
    agent, client, _ = make_agent(tmp_path, script, unbindable={"fake/other"})
    loaded = await agent.load_session(cwd=str(work), session_id=session.session_id)
    assert [u.content.text for u in client.of(UserMessageChunk)] == ["first question"]
    assert client.text() == "reply 2"
    assert loaded.config_options is not None and loaded.config_options[0].current_value == "fake/other"
    with pytest.raises(RequestError) as unbound:
        await agent.prompt(prompt=[text_block("second")], session_id=session.session_id)
    assert unbound.value.code == -32602
    await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/model")
    response = await agent.prompt(prompt=[text_block("second")], session_id=session.session_id)
    assert response.stop_reason == "end_turn" and client.text().endswith("reply 4 ")
    await agent.aclose()


def test_bootstrap_reports_a_system2_that_cannot_bind(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text('model = "anthropic/claude-haiku-5-5"\n')
    bootstrap = Bootstrap(Options(cwd=tmp_path), {"INTERN_HOME": str(home)})
    with pytest.raises(System2Unavailable) as missing:
        asyncio.run(bootstrap.session(lambda _: None))
    # The System 1 the unbound session will open on is the default, a decision model from the console.
    assert (missing.value.ref, missing.value.system1) == ("anthropic/claude-haiku-5-5", "drex-latest")
    assert "ANTHROPIC_API_KEY" in str(missing.value)
    unset = Bootstrap(Options(cwd=tmp_path), {"INTERN_HOME": str(tmp_path / "empty")})
    with pytest.raises(System2Unavailable) as none:
        asyncio.run(unset.session(lambda _: None))
    assert none.value.ref == "" and "no model given" in str(none.value)


async def test_the_server_opens_sessions_when_the_configured_model_has_no_key(tmp_path: Path, work: Path):
    """The live failure: config.toml's default needs a key the client did not pass."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text('model = "anthropic/claude-haiku-5-5"\n')
    client = FakeClient()
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "INTERN_SYSTEM2")}
    env |= {"INTERN_HOME": str(home)}
    with FakeOpenAI([Reply(text="from meta"), Reply(text="from the picker")]) as server:
        env |= {"VLLM_BASE_URL": server.url, **server.system1_env}
        async with spawn_agent_process(
            lambda _agent: client,  # pyright: ignore[reportArgumentType, reportReturnType]
            sys.executable, "-m", "intern", "acp", "--no-context-files",
            env=env, cwd=work,
        ) as (conn, process):  # fmt: skip
            await conn.initialize(protocol_version=1)
            meta = await conn.new_session(cwd=str(work), mcp_servers=[], nace={"system2": f"vllm/{MODEL}"})
            response = await conn.prompt(prompt=[text_block("hi")], session_id=meta.session_id)
            assert response.stop_reason == "end_turn" and client.text() == "from meta "

            unbound = await conn.new_session(cwd=str(work), mcp_servers=[])
            assert unbound.config_options is not None
            assert unbound.config_options[0].current_value == "anthropic/claude-haiku-5-5"
            with pytest.raises(RequestError) as refused:
                await conn.prompt(prompt=[text_block("hi")], session_id=unbound.session_id)
            assert refused.value.code == -32602 and "ANTHROPIC_API_KEY" in str(refused.value)
            await conn.set_config_option(config_id="model", session_id=unbound.session_id, value=f"vllm/{MODEL}")
            response = await conn.prompt(prompt=[text_block("hi")], session_id=unbound.session_id)
            assert response.stop_reason == "end_turn" and client.text().endswith("from the picker ")
            assert process.stdin is not None
            process.stdin.close()
            assert await asyncio.wait_for(process.wait(), 20) == 0


def test_the_index_finds_journals_by_their_first_entry_when_it_has_no_entry(tmp_path: Path):
    sessions = tmp_path / "sessions"
    journal = Journal(cwd=str(tmp_path), path=sessions / "20260101-000000-1.jsonl")
    index = SessionIndex(sessions)
    assert index.find(journal.session_id) == journal.path
    assert index.find("unknown") is None
    # Found once by a scan, it is indexed.
    assert SessionIndex(sessions)._read() == {journal.session_id: str(journal.path)}  # pyright: ignore[reportPrivateUsage]


def test_prompt_content_blocks_become_one_prompt(tmp_path: Path):
    png = "iVBORw0KGgo="
    text = prompt_text(
        [
            TextContentBlock(type="text", text="look at this"),
            ImageContentBlock(type="image", data=png, mime_type="image/png"),
        ],
        tmp_path / "attachments",
    )
    saved = next((tmp_path / "attachments").iterdir())
    assert saved.suffix == ".png" and saved.read_bytes() == b"\x89PNG\r\n\x1a\n"
    assert text == f"look at this\n\n[The user attached an image: {saved}. Read it to look at it.]"


async def test_the_server_process_speaks_acp_over_stdio(tmp_path: Path, work: Path):
    """The smoke test: `intern acp` as a process, driven by the ACP SDK's own client."""
    replies = [Reply(tool_calls=[("write", {"path": "hi.txt", "content": "hi\n"})]), Reply(text="wrote it")]
    client = FakeClient()
    env = {**os.environ, "VLLM_BASE_URL": "", "INTERN_HOME": str(tmp_path / "home")}
    with FakeOpenAI(replies) as server:
        env |= {"VLLM_BASE_URL": server.url, **server.system1_env}
        async with spawn_agent_process(
            lambda _agent: client,  # pyright: ignore[reportArgumentType, reportReturnType]
            sys.executable,
            "-m",
            "intern",
            "acp",
            "--no-context-files",
            "--system2",
            f"vllm/{MODEL}",
            env=env,
            cwd=work,
        ) as (conn, process):
            init = await conn.initialize(protocol_version=1)
            assert init.agent_info is not None and init.agent_info.name == "intern"
            session = await conn.new_session(cwd=str(work), mcp_servers=[])
            response = await conn.prompt(prompt=[text_block("write hi.txt")], session_id=session.session_id)
            assert response.stop_reason == "end_turn"
            assert (work / "hi.txt").read_text() == "hi\n"
            assert client.text() == "wrote it "

            # Cancel right after prompt, over the wire.
            prompt = asyncio.create_task(conn.prompt(prompt=[text_block("again")], session_id=session.session_id))
            await asyncio.sleep(0)  # the prompt goes on the wire first, as a client sends them
            await conn.cancel(session_id=session.session_id)
            assert (await prompt).stop_reason == "cancelled"
            assert process.stdin is not None
            process.stdin.close()
            assert await asyncio.wait_for(process.wait(), 20) == 0


# The event → session/update table (docs/acp.md, "Updates"), row by row.

CTX = acp_mapping.MappingContext(
    cwd=Path("/work"),
    effects={tool.name: tool.effects for tool in [*load_packs(["base"]), WebSearchTool()]},
    working_window=32_768,
    context_tokens=1200,
    cost_usd=0.25,
)
WINDOW = WindowInfo(advertised=32_768, advertised_source="profile", effective=None, user_cap=None, working=32_768)


def test_text_and_reasoning_deltas_are_message_and_thought_chunks():
    [text] = acp_mapping.event_updates(TextDelta(turn=1, text="hi"), CTX)
    [thought] = acp_mapping.event_updates(ReasoningDelta(turn=1, text="hmm"), CTX)
    assert isinstance(text, AgentMessageChunk) and isinstance(text.content, TextContentBlock)
    assert isinstance(thought, AgentThoughtChunk) and isinstance(thought.content, TextContentBlock)
    assert (text.content.text, thought.content.text) == ("hi", "hmm")
    assert acp_mapping.event_updates(TextDelta(turn=1, text=""), CTX) == []


@pytest.mark.parametrize(
    ("name", "arguments", "kind", "title"),
    [
        ("read", {"path": "a.py", "offset": 10}, "read", "read a.py"),
        ("write", {"path": "a.py", "content": "x"}, "edit", "write a.py"),
        ("edit", {"path": "a.py", "old_string": "x", "new_string": "y"}, "edit", "edit a.py"),
        ("bash", {"command": "pytest -q\necho done"}, "execute", "bash pytest -q"),
        ("grep", {"pattern": "needle"}, "search", "grep needle"),
        ("find", {"pattern": "*.py"}, "search", "find *.py"),
        ("web_search", {"query": "acp"}, "fetch", "web_search acp"),
        ("mystery", {}, "other", "mystery"),
    ],
)
def test_tool_call_start_takes_its_kind_from_the_tools_effects(name: str, arguments: dict, kind: str, title: str):
    [update] = acp_mapping.event_updates(ToolCallStartEvent(turn=1, call_id="c1", name=name, arguments=arguments), CTX)
    assert isinstance(update, ToolCallStart)
    assert (update.tool_call_id, update.kind, update.title, update.status) == ("c1", kind, title, "in_progress")
    assert update.raw_input == arguments
    if "path" in arguments:
        assert update.locations is not None
        assert update.locations[0].path == "/work/a.py"
        assert update.locations[0].line == arguments.get("offset")
    else:
        assert update.locations is None


@pytest.mark.parametrize(("status", "acp_status"), [("ok", "completed"), ("error", "failed"), ("refused", "failed")])
def test_tool_call_end_is_an_update_with_the_result(status: str, acp_status: str):
    event = ToolCallEndEvent(turn=1, call_id="c1", name="bash", status=status, content="output", duration_ms=3)  # pyright: ignore[reportArgumentType]
    [update] = acp_mapping.event_updates(event, CTX)
    assert isinstance(update, ToolCallProgress)
    assert (update.tool_call_id, update.status) == ("c1", acp_status)
    assert update.content is not None and update.content[0].content.text == "output"  # pyright: ignore[reportAttributeAccessIssue]


def test_context_budget_and_run_end_are_usage_updates():
    budget = ContextBudget(turn=1, tokens=900, working_window=32_768, system=1, pinned=1, history=1, tail=1)
    [during] = acp_mapping.event_updates(budget, CTX)
    assert isinstance(during, UsageUpdate) and (during.used, during.size, during.cost) == (900, 32_768, None)
    end = RunEnd(run_id="r", reason="completed", turns=1, usage=Usage(), final_text="", duration_ms=1)
    [after] = acp_mapping.event_updates(end, CTX)
    assert isinstance(after, UsageUpdate) and (after.used, after.size) == (1200, 32_768)
    assert after.cost is not None and (after.cost.amount, after.cost.currency) == (0.25, "USD")


def test_other_events_send_nothing():
    start = RunStart(session_id="s", run_id="r", model="m", cwd="/work", window=WINDOW, tools=[])
    assert acp_mapping.event_updates(start, CTX) == []
    assert acp_mapping.event_updates(TurnStart(turn=1), CTX) == []
    assert acp_mapping.event_updates(Nudge(turn=1, reason="r", text="t"), CTX) == []


@pytest.mark.parametrize(
    ("reason", "acp_reason"),
    [
        ("completed", "end_turn"),
        ("stopped_by_hook", "end_turn"),
        ("stopped_by_tool", "end_turn"),
        ("stopped_by_guard", "end_turn"),
        ("aborted", "cancelled"),
        ("turn_budget", "max_turn_requests"),
        ("output_limit", "max_tokens"),
        ("provider_error", None),
        ("internal_error", None),
    ],
)
def test_stop_reasons(reason: str, acp_reason: str | None):
    assert acp_mapping.stop_reason(reason) == acp_reason  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("detail", "kind", "status"),
    [
        ("HTTP 429 from groq: slow down", "rate_limit", 429),
        ("stream error from openrouter: {'message': 'Rate limit exceeded'}", "rate_limit", None),
        ("HTTP 402 from openrouter: payment required", "quota", 402),
        ("error from fireworks: account quota exhausted", "quota", None),
        ("HTTP 403 from anthropic: forbidden", "auth", 403),
        ("the stream ended without a reply", "provider", None),
        (None, "provider", None),
    ],
)
def test_provider_failures_are_classified(detail: str | None, kind: str, status: int | None):
    failure = acp_mapping.provider_failure(detail)
    assert (failure.kind, failure.status) == (kind, status)


def test_replay_shows_prompts_replies_and_tool_calls_but_not_nudges():
    journal = Journal(cwd="/work")
    for message in [
        UserMessage(content="fix it"),
        AssistantMessage(reasoning="think", tool_calls=[ToolCall(id="c1", name="read", arguments={"path": "a.py"})]),
        ToolResultMessage(result=ToolResult(call_id="c1", tool_name="read", status="ok", content="x = 1")),
        UserMessage(content="keep going", source="harness"),
        AssistantMessage(text="fixed"),
    ]:
        journal.append(MessageEntry(message=message))
    updates = acp_mapping.replay_updates(journal.active_path(), CTX)
    assert [u.session_update for u in updates] == [
        "user_message_chunk",
        "agent_thought_chunk",
        "tool_call",
        "tool_call_update",
        "agent_message_chunk",
    ]
    start, end = updates[2], updates[3]
    assert isinstance(start, ToolCallStart) and (start.kind, start.title) == ("read", "read a.py")
    assert isinstance(end, ToolCallProgress) and end.status == "completed"


async def test_a_turn_with_a_tool_streams_the_call_and_usage(tmp_path: Path, work: Path):
    script = [Turn(tool_calls=[("write", {"path": "a.txt", "content": "one\n"})]), Turn(text="done")]
    agent, client, _ = make_agent(tmp_path, script)
    session = await agent.new_session(cwd=str(work))
    await agent.prompt(prompt=[text_block("write a.txt")], session_id=session.session_id)
    [start] = client.of(ToolCallStart)
    assert (start.kind, start.title) == ("edit", "write a.txt")
    assert start.locations is not None and start.locations[0].path == str(work / "a.txt")
    [end] = client.of(ToolCallProgress)
    assert end.status == "completed"
    usage = client.of(UsageUpdate)
    assert usage and usage[-1].cost is not None and usage[-1].size == 32_768
    assert all(u.used > 0 for u in usage)
    await agent.aclose()


# The model catalog and `intern models --json`.


@pytest.fixture(autouse=True)
def fresh_catalog_cache():
    catalog._cache.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    catalog._cache.clear()  # pyright: ignore[reportPrivateUsage]


async def test_the_catalog_lists_what_each_reachable_provider_serves():
    asked: list[str] = []

    def listing(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.host)
        if request.url.host == "openrouter.ai" and request.url.params.get("output_modalities") == "decisions":
            assert request.headers["Authorization"] == "Bearer or-key"
            decisions = [{"id": "~typesafe/jev-latest", "name": "TypeSafe: Jev Latest"}, {"id": "nace-ai/drex-max"}]
            return httpx.Response(200, json={"data": decisions})
        if request.url.host == "openrouter.ai":
            assert request.headers["Authorization"] == "Bearer or-key"
            qwen = {"id": "qwen/qwen3.8-27b", "name": "Qwen3.8 27B", "context_length": 262_144}
            qwen["architecture"] = {"input_modalities": ["text"], "output_modalities": ["text"]}
            painter = {"id": "acme/painter", "architecture": {"output_modalities": ["image"]}}
            return httpx.Response(200, json={"data": [qwen, painter]})
        if request.url.host == "console.nace.ai":
            assert request.headers["Authorization"] == "Bearer console-key"
            models = [
                {"name": "drex-v2.0", "description": "Drex 2.0 decision model.", "alias_for": None},
                {"name": "drex-latest", "description": "Points to drex-v2.0.", "alias_for": "drex-v2.0"},
                {"name": "drex-v1.0", "description": "Retired on 2026-10-21.", "alias_for": "drex-max-v1.1"},
            ]
            return httpx.Response(200, json={"models": models})
        if request.url.host == "api.anthropic.com":
            assert request.headers["x-api-key"] == "an-key"
            return httpx.Response(200, json={"data": [{"id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5"}]})
        if request.url.host == "inference.example":
            served = [
                {"id": "mine"},
                {"id": "text-embedding-3-small"},
                {"id": "gpt-4o-mini-tts"},
                {"id": "base-only", "supports_chat": False},
                {"id": "ranker", "type": "rerank"},
                # Fireworks marks its embedding models chat-capable; their kind says otherwise.
                {"id": "qwen3-embedding-8b", "kind": "EMBEDDING_MODEL", "supports_chat": True},
            ]
            return httpx.Response(200, json={"data": served})
        return httpx.Response(500)

    found = await catalog.list_models(
        {"OPENROUTER_API_KEY": "or-key", "ANTHROPIC_API_KEY": "an-key", "MY_KEY": "k"},
        providers={"mine": ProviderSpec(base_url="https://inference.example/v1", api_key_env="MY_KEY")},
        extra_system2=[
            catalog.ModelInfo(id=ref, name=ref)
            for ref in ("openrouter/qwen/qwen3.8-27b", "groq/llama-x", "nowhere/model", "vllm/local")
        ],
        system1=catalog.System1Sources(
            console_key="console-key", console_models=("drex-latest", "drex-latest", "drex-2"), decisions_key="or-key"
        ),
        transport=httpx.MockTransport(listing),
    )
    # Only models System 2 can chat with: what the listing says where it says, else what the id names. The
    # image, embedding, speech, completions-only and rerank models are left out.
    assert [(m.id, m.name) for m in found.system2] == [
        ("anthropic/claude-haiku-5-5", "Claude Haiku 5.5"),
        ("mine/mine", "mine"),
        ("openrouter/qwen/qwen3.8-27b", "Qwen3.8 27B"),
        ("vllm/local", "vllm/local"),
    ]
    # A listing that says how much context a model takes gives its window; an extra's is not known here.
    assert [m.window for m in found.system2] == [None, None, 262_144, None]
    # Providers without a key are never asked, nor listed from the extras; keyless local ones are asked, and
    # their failures are left out.
    assert "api.openai.com" not in asked and "api.groq.com" not in asked and "localhost" in asked
    # System 1: the console's decision models, the configured names first and retired ones left out, then the
    # decisions API's decision models as decisions:<model>.
    assert [(m.id, m.name) for m in found.system1] == [
        ("drex-latest", "drex-latest"),
        ("drex-2", "drex-2"),
        ("drex-v2.0", "drex-v2.0"),
        ("decisions:~typesafe/jev-latest", "TypeSafe: Jev Latest"),
        ("decisions:nace-ai/drex-max", "nace-ai/drex-max"),
    ]


async def test_system_1_sources_without_keys_are_not_asked_and_the_configured_decision_models_stand_in():
    asked: list[str] = []

    def listing(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.host)
        return httpx.Response(500)

    sources = catalog.System1Sources(console_models=("drex-latest", "drex-max-v1.1"))
    found = await catalog.list_models({}, system1=sources, transport=httpx.MockTransport(listing))
    assert "console.nace.ai" not in asked and "openrouter.ai" not in asked
    assert [m.id for m in found.system1] == ["drex-latest", "drex-max-v1.1"]


def test_models_json_lists_the_catalog_without_opening_a_session(tmp_path: Path, work: Path):
    home = tmp_path / "home"
    home.mkdir()
    deployment = "fireworks/accounts/example/deployments/qwen3p8-27b"
    claude = "anthropic/claude-unprofiled"
    config = f'model = "{claude}"\n[system1]\nmodels = ["drex-latest", "drex-mini"]\n'
    # A dedicated deployment reaches the catalog through a user profile's concrete id.
    (home / "profiles").mkdir()
    (home / "profiles" / "mine.toml").write_text(
        f'match = ["{deployment}"]\nname = "Qwen3.8 27B (Fireworks)"\nwindow = 131072\nreasoning_effort = "medium"\n'
    )
    (home / "config.toml").write_text(config + "[capabilities.browser]\nenabled = true\n")
    with FakeOpenAI([]) as server:
        env = {"PATH": os.environ["PATH"], "INTERN_HOME": str(home), "VLLM_BASE_URL": server.url}
        # Fireworks lists its serverless models but not dedicated deployments, as the real one does.
        env |= {"FIREWORKS_API_KEY": "fw-key", "FIREWORKS_BASE_URL": server.url}
        env |= {"ANTHROPIC_API_KEY": "sk-ant", "ANTHROPIC_BASE_URL": server.url}
        result = subprocess.run(
            [sys.executable, "-m", "intern", "models", "--json"],
            env=env, cwd=work, capture_output=True, text=True, timeout=60,
        )  # fmt: skip
        assert server.requests == []
    assert result.returncode == 0, result.stderr
    listed = json.loads(result.stdout)
    # A listed model's window is the listing's; one the provider does not list gets its profile's.
    assert {"id": f"vllm/{MODEL}", "name": MODEL, "window": 40_960} in listed["system2"]
    assert {"id": f"fireworks/{MODEL}", "name": MODEL, "window": 40_960} in listed["system2"]
    # The configured default is marked. Its API takes an effort for every model, so with no profile naming one it
    # still lists the efforts, after `default`; the flags are only ever added.
    [default] = [m for m in listed["system2"] if m.get("default")]
    assert default == {
        "id": claude,
        "name": claude,
        "default": True,
        "reasoning": True,
        "efforts": ["default", "low", "medium", "high", "xhigh", "max"],
    }
    # An OpenAI-compatible model takes one only on its profile's word, and always sends it.
    [qwen] = [m for m in listed["system2"] if m["id"] == deployment]
    assert qwen == {
        "id": deployment,
        "name": "Qwen3.8 27B (Fireworks)",
        "reasoning": True,
        "efforts": ["low", "medium", "high"],
        "window": 131_072,
    }
    assert all(set(m) <= {"id", "name", "default", "reasoning", "efforts", "window"} for m in listed["system2"])
    # The capability toggles sessions offer, each on as the config has it.
    assert [(c["id"], c["enabled"]) for c in listed["capabilities"]] == [
        ("cap.web-search", False),
        ("cap.web-read", False),
        ("cap.browser", True),
        ("cap.code-search", False),
        ("cap.workspace", False),
        ("cap.memory", True),
    ]
    assert all(c["name"] and c["description"] for c in listed["capabilities"])
    # Where commands run is one choice, listed apart: the built-ins and OpenSandbox, the configured one the default.
    runners = {r["id"]: r for r in listed["runners"]}
    assert list(runners) == ["local", "docker", "opensandbox"]
    assert runners["local"]["default"] is True and not runners["docker"]["default"]
    assert all(r["name"] and r["description"] for r in listed["runners"])
    # Without its extra or its key OpenSandbox cannot run here, and says why.
    reason = runners["opensandbox"].get("unavailable") or ""
    assert "--extra opensandbox" in reason or "OPEN_SANDBOX_API_KEY" in reason
    # This environment names no search provider, so Web search says why it cannot switch on.
    assert {c["id"]: c.get("unavailable") for c in listed["capabilities"]}["cap.web-search"] == (
        "Web search is not set up: set INTERN_SEARCH_PROVIDER to brave, tavily, exa or searxng"
    )
    # No console or decisions-API key in this environment: the configured decision models stand in.
    assert listed["system1"] == [{"id": "drex-latest", "name": "drex-latest"}, {"id": "drex-mini", "name": "drex-mini"}]
    assert not (home / "sessions").exists()


async def test_the_model_picker_fills_from_the_catalog_once_it_arrives(tmp_path: Path, work: Path):
    listed = catalog.Catalog(
        system2=[
            catalog.ModelInfo(id="fake/big", name="Big"),
            catalog.ModelInfo(id="other/x", name="X"),
        ],
        system1=[catalog.ModelInfo(id="drex-latest", name="drex-latest")],
    )

    async def load() -> catalog.Catalog:
        return listed

    open_session = Opener([Turn(text="ok")], tmp_path)
    agent = AcpAgent(open_session, tmp_path / "home", catalog=load)
    client = FakeClient()
    agent.on_connect(client)  # pyright: ignore[reportArgumentType]
    session = await agent.new_session(cwd=str(work))
    # Before the catalog: only the current model.
    assert session.config_options is not None
    assert option_values(session.config_options, "model") == ["fake/model"]
    assert agent.catalog_task is not None
    await agent.catalog_task
    [update] = client.of(ConfigOptionUpdate)
    assert option_values(update.config_options, "model") == ["fake/big", "fake/model", "other/x"]
    picked = await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/big")
    model = next(o for o in picked.config_options if o.id == "model")
    assert model.current_value == "fake/big"
    assert agent.sessions[session.session_id].session.deps.system2.ref == "fake/big"
    with pytest.raises(RequestError) as unresolvable:
        await agent.set_config_option(config_id="model", session_id=session.session_id, value="nowhere/x")
    assert unresolvable.value.code == -32602
    await agent.aclose()


def settings(options: list[Any]) -> list[Any]:
    """The model options: not the capability toggles, nor the runner (tests/test_runner_choice.py covers it)."""
    return [o for o in options if o.category not in ("_nace_capability", "_nace_runner")]


def option_values(options: list[Any], config_id: str) -> list[str]:
    option = next(o for o in options if o.id == config_id)
    values: list[str] = []
    for entry in option.options:
        values += [c.value for c in entry.options] if hasattr(entry, "group") else [entry.value]
    return values


# System 1 and thought level, switched through set_config_option.


async def test_system1_is_switched_from_the_picker_and_restored_on_load(tmp_path: Path, work: Path):
    agent, _, _ = make_agent(tmp_path, [Turn(text="ok")])
    session = await agent.new_session(cwd=str(work))
    assert session.config_options is not None
    system1 = next(o for o in session.config_options if o.id == "system1")
    assert (system1.category, system1.current_value) == ("model_config", "quiet")
    picked = await agent.set_config_option(config_id="system1", session_id=session.session_id, value="drex-latest")
    assert next(o for o in picked.config_options if o.id == "system1").current_value == "drex-latest"
    decisions = agent.sessions[session.session_id].session.deps.decisions
    assert isinstance(decisions.backend, NamedBackend) and decisions.backend.name == "drex-latest"
    await agent.aclose()

    agent, _, _ = make_agent(tmp_path, [Turn(text="ok")])
    loaded = await agent.load_session(cwd=str(work), session_id=session.session_id)
    assert loaded.config_options is not None
    assert next(o for o in loaded.config_options if o.id == "system1").current_value == "drex-latest"
    backend = agent.sessions[session.session_id].session.deps.decisions.backend
    assert isinstance(backend, NamedBackend) and backend.name == "drex-latest"
    await agent.aclose()


class Ladder(FakeProvider):
    """A provider whose API takes the whole run of efforts for every model, as the Anthropic provider does."""

    effort_levels = EFFORT_LEVELS


async def test_thought_level_is_offered_only_when_the_model_takes_an_effort_and_lists_what_it_takes(
    tmp_path: Path, work: Path
):
    agent, _, _ = make_agent(
        tmp_path,
        [Turn(text="ok")] * 3,
        profiles={"fake/thinker": Profile(reasoning_effort="high")},
        providers={"fake/claude": Ladder},
    )
    session = await agent.new_session(cwd=str(work))
    assert session.config_options is not None
    assert [o.id for o in settings(session.config_options)] == ["model", "system1", "context_window"]
    with pytest.raises(RequestError) as unknown:
        await agent.set_config_option(config_id="thought_level", session_id=session.session_id, value="low")
    assert unknown.value.code == -32602

    switched = await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/thinker")
    assert [o.id for o in settings(switched.config_options)] == ["model", "system1", "thought_level", "context_window"]
    level = switched.config_options[2]
    assert (level.category, level.current_value) == ("thought_level", "high")
    assert option_values(switched.config_options, "thought_level") == ["low", "medium", "high"]
    await agent.set_config_option(config_id="thought_level", session_id=session.session_id, value="low")
    await agent.prompt(prompt=[text_block("go")], session_id=session.session_id)
    provider = agent.sessions[session.session_id].session.deps.system2.provider
    assert isinstance(provider, FakeProvider) and provider.requests[-1].reasoning_effort == "low"
    with pytest.raises(RequestError) as bad:
        await agent.set_config_option(config_id="thought_level", session_id=session.session_id, value="max")
    assert bad.value.code == -32602 and "low, medium, high" in str(bad.value)
    # A provider whose API takes the whole run lists it with no profile naming an effort, starting at `default`
    # (none sent); the top of it is accepted, and `default` goes back to sending none.
    switched = await agent.set_config_option(config_id="model", session_id=session.session_id, value="fake/claude")
    level = next(o for o in switched.config_options if o.id == "thought_level")
    assert level.current_value == "default"
    assert option_values(switched.config_options, "thought_level") == ["default", *EFFORT_LEVELS]
    await agent.set_config_option(config_id="thought_level", session_id=session.session_id, value="max")
    await agent.prompt(prompt=[text_block("again")], session_id=session.session_id)
    provider = agent.sessions[session.session_id].session.deps.system2.provider
    assert isinstance(provider, Ladder) and provider.requests[-1].reasoning_effort == "max"
    reset = await agent.set_config_option(config_id="thought_level", session_id=session.session_id, value="default")
    assert next(o for o in reset.config_options if o.id == "thought_level").current_value == "default"
    await agent.prompt(prompt=[text_block("once more")], session_id=session.session_id)
    assert provider.requests[-1].reasoning_effort is None
    await agent.aclose()


async def test_the_context_window_is_capped_per_session_across_models_and_loads(tmp_path: Path, work: Path):
    windows = {"fake/model": 200_000, "fake/small": 48_000}
    script = [Turn(text="ok")] * 2
    agent, client, _ = make_agent(tmp_path, script, windows=windows, window_cap=100_000)
    session = await agent.new_session(cwd=str(work))
    sid = session.session_id
    assert session.config_options is not None
    window = next(o for o in session.config_options if o.id == "context_window")
    assert (window.category, window.current_value) == ("model_config", "auto")
    # The presets the model can use; auto is the --window default.
    assert [(c.value, c.name) for c in window.options] == [
        ("auto", "Auto (100K)"),
        ("32000", "32K"),
        ("64000", "64K"),
        ("128000", "128K"),
    ]
    live = agent.sessions[sid].session

    # Above --window but within the model's window: the session's cap wins, and the meter follows it.
    picked = await agent.set_config_option(config_id="context_window", session_id=sid, value="128000")
    assert next(o for o in picked.config_options if o.id == "context_window").current_value == "128000"
    assert live.deps.system2.window.working == 128_000
    await agent.prompt(prompt=[text_block("go")], session_id=sid)
    assert client.of(UsageUpdate)[-1].size == 128_000

    # A smaller model clamps it; the cap is kept, listed, and back in force on the bigger model.
    switched = await agent.set_config_option(config_id="model", session_id=sid, value="fake/small")
    assert live.deps.system2.window.working == 48_000
    assert option_values(switched.config_options, "context_window") == ["auto", "32000", "128000"]
    await agent.set_config_option(config_id="model", session_id=sid, value="fake/model")
    assert live.deps.system2.window.working == 128_000

    for bad in ("0", "-1", "1.5", "big", "", "１２８"):
        with pytest.raises(RequestError) as refused:
            await agent.set_config_option(config_id="context_window", session_id=sid, value=bad)
        assert refused.value.code == -32602, bad
    await agent.aclose()

    # A new process continues on the cap; auto goes back to the --window default.
    agent, client, _ = make_agent(tmp_path, script, windows=windows, window_cap=100_000)
    loaded = await agent.load_session(cwd=str(work), session_id=sid)
    assert loaded.config_options is not None
    assert next(o for o in loaded.config_options if o.id == "context_window").current_value == "128000"
    live = agent.sessions[sid].session
    assert live.deps.system2.window.working == 128_000
    await agent.set_config_option(config_id="context_window", session_id=sid, value="auto")
    assert live.deps.system2.window.working == 100_000
    await agent.prompt(prompt=[text_block("go")], session_id=sid)
    assert client.of(UsageUpdate)[-1].size == 100_000
    await agent.aclose()


async def test_configuration_waits_for_the_run_to_end(tmp_path: Path, work: Path):
    agent, _, opener = make_agent(tmp_path, [Turn(tool_calls=[("wait", {})])])
    session = await agent.new_session(cwd=str(work))
    prompt = asyncio.create_task(agent.prompt(prompt=[text_block("wait")], session_id=session.session_id))
    await opener.waiting.wait()
    picks = (("model", "fake/other"), ("system1", "drex-latest"), ("context_window", "32000"), ("cap.browser", True))
    for config_id, value in picks:
        with pytest.raises(RequestError) as busy:
            await agent.set_config_option(config_id=config_id, session_id=session.session_id, value=value)
        assert busy.value.code == -32000
    await agent.cancel(session_id=session.session_id)
    await prompt
    await agent.aclose()


# Capability toggles: boolean cap.* options, applied by reopening the session at the next prompt.

DOCS_MCP = McpConfig(
    namespace="docs", transport="http", url="https://mcp.example/docs", remote_tools={"lookup": frozenset()}
)


def toggle_values(options: list[Any]) -> dict[str, bool]:
    return {o.id: o.current_value for o in options if o.category == "_nace_capability"}


async def test_capability_toggles_apply_at_the_next_prompt_with_one_reopen_and_survive_a_load(
    tmp_path: Path, work: Path
):
    configured = CapabilitySettings.model_validate({"browser": {"enabled": True}, "mcp": [DOCS_MCP]})
    script = lambda request: Turn(text=f"reply {len(request.messages)}")  # noqa: E731
    agent, client, opener = make_agent(tmp_path, script, configured)
    session = await agent.new_session(cwd=str(work))
    sid = session.session_id
    assert session.config_options is not None
    # One boolean per capability after the other options, each on as the config has it; extensions are not toggles.
    assert toggle_values(session.config_options) == {
        "cap.web-search": False,
        "cap.web-read": False,
        "cap.browser": True,
        "cap.code-search": False,
        "cap.workspace": False,
        "cap.memory": True,
        "cap.mcp.docs": True,
    }
    browser = next(o for o in session.config_options if o.id == "cap.browser")
    assert browser.type == "boolean" and browser.name == "Browser" and "Playwright" in (browser.description or "")
    await agent.prompt(prompt=[text_block("first")], session_id=sid)
    live = agent.sessions[sid]
    first, spent = live.session, live.context_tokens

    # Setting the values in force, as a client does on a thread's first turn, changes nothing.
    for config_id, value in (("cap.browser", True), ("cap.web-search", False), ("cap.mcp.docs", True)):
        await agent.set_config_option(config_id=config_id, session_id=sid, value=value)
    # A toggle switched and switched back before a prompt is no change either.
    await agent.set_config_option(config_id="cap.web-read", session_id=sid, value=True)
    await agent.set_config_option(config_id="cap.web-read", session_id=sid, value=False)
    # Nor is `cap.process` from a client of 0.22 or before: no longer offered (a base tool), accepted, and ignored.
    await agent.set_config_option(config_id="cap.process", session_id=sid, value=True)
    await agent.prompt(prompt=[text_block("second")], session_id=sid)
    assert len(opener.opened) == 1 and live.session is first

    # Changes are pending until the next prompt, which reopens once on the same journal for all of them.
    await agent.set_config_option(config_id="cap.web-search", session_id=sid, value=True)
    picked = await agent.set_config_option(config_id="cap.mcp.docs", session_id=sid, value=False)
    assert toggle_values(picked.config_options)["cap.web-search"] is True
    assert toggle_values(picked.config_options)["cap.mcp.docs"] is False
    assert len(opener.opened) == 1
    response = await agent.prompt(prompt=[text_block("third")], session_id=sid)
    assert response.stop_reason == "end_turn" and client.text().endswith("reply 6 ")
    assert len(opener.opened) == 2 and agent.sessions[sid] is live and live.session is not first
    reopened = opener.opened[-1]
    assert reopened.journal == first.deps.journal.path and reopened.system2 == "fake/model"
    assert reopened.capabilities is not None and reopened.capabilities.tool_packs == ["base", "web-search"]
    assert reopened.capabilities.mcp == [] and reopened.capabilities.browser.enabled
    assert "web_search" in [tool.name for tool in live.session.deps.tools]
    assert live.context_tokens >= spent > 0  # the usage totals carry on
    # A reopen is not the session's end: it is not stewarded into memory.
    assert opener.closed == []

    for config_id, value in (("cap.browser", "yes"), ("cap.nothing", True), ("cap.mcp.other", True)):
        with pytest.raises(RequestError) as refused:
            await agent.set_config_option(config_id=config_id, session_id=sid, value=value)
        assert refused.value.code == -32602, config_id
    await agent.aclose()
    assert opener.closed == [str(first.deps.journal.path)]

    # A new process opens on the journaled toggles, with no reopen at the first prompt.
    agent, client, opener = make_agent(tmp_path, script, configured)
    loaded = await agent.load_session(cwd=str(work), session_id=sid)
    assert loaded.config_options is not None
    assert toggle_values(loaded.config_options)["cap.web-search"] is True
    assert toggle_values(loaded.config_options)["cap.mcp.docs"] is False
    assert opener.opened[-1].capabilities == reopened.capabilities
    await agent.prompt(prompt=[text_block("fourth")], session_id=sid)
    assert len(opener.opened) == 1
    await agent.aclose()


async def test_a_capability_switch_the_session_cannot_open_on_fails_the_prompt_and_keeps_the_old_ones(
    tmp_path: Path, work: Path
):
    broken = {"cap.browser": "browser could not start: Playwright is not installed"}
    agent, client, opener = make_agent(tmp_path, [Turn(text="ok")], broken=broken)
    session = await agent.new_session(cwd=str(work))
    sid = session.session_id
    await agent.set_config_option(config_id="cap.browser", session_id=sid, value=True)
    with pytest.raises(RequestError) as failed:
        await agent.prompt(prompt=[text_block("go")], session_id=sid)
    assert failed.value.code == -32602
    assert "browser could not start: Playwright is not installed" in str(failed.value)
    # Tried on the switch, then opened again on the configured toggles.
    assert len(opener.opened) == 3 and opener.opened[-1].capabilities is None
    live = agent.sessions[sid]
    assert not live.capabilities.browser.enabled and live.pending is None
    # The client, shown the change as pending, is told it did not happen.
    assert toggle_values(client.of(ConfigOptionUpdate)[-1].config_options)["cap.browser"] is False
    response = await agent.prompt(prompt=[text_block("again")], session_id=sid)
    assert response.stop_reason == "end_turn" and len(opener.opened) == 3
    assert live.session.deps.journal.last_capabilities() is None
    await agent.aclose()


async def test_a_switch_that_cannot_start_here_stays_off_and_is_named_while_the_others_apply(
    tmp_path: Path, work: Path
):
    reasons = {"cap.web-search": "Web search is not set up: set INTERN_SEARCH_PROVIDER to brave or exa."}
    script = [Turn(text="ok")] * 3
    agent, client, opener = make_agent(tmp_path, script, unavailable=reasons)
    session = await agent.new_session(cwd=str(work))
    sid = session.session_id
    # A switch that only ever fails here is not offered; one a client sends anyway is refused at the prompt.
    assert session.config_options is not None and "cap.web-search" not in toggle_values(session.config_options)
    await agent.set_config_option(config_id="cap.browser", session_id=sid, value=True)
    await agent.set_config_option(config_id="cap.web-search", session_id=sid, value=True)
    with pytest.raises(RequestError) as failed:
        await agent.prompt(prompt=[text_block("search")], session_id=sid)
    assert failed.value.code == -32602
    assert str(failed.value) == (
        "Web search is not set up: set INTERN_SEARCH_PROVIDER to brave or exa. "
        "Web search stays off; the other capability changes applied"
    )
    # Browser applied in the one reopen; Web search was never tried, and the client is shown it gone.
    live = agent.sessions[sid]
    assert len(opener.opened) == 2 and opener.opened[-1].capabilities == live.capabilities
    assert live.capabilities.browser.enabled and live.capabilities.tool_packs == ["base"]
    assert "cap.web-search" not in toggle_values(client.of(ConfigOptionUpdate)[-1].config_options)

    # A client that asks again is refused again, with nothing to reopen; once it stops asking, prompts run.
    await agent.set_config_option(config_id="cap.web-search", session_id=sid, value=True)
    with pytest.raises(RequestError) as again:
        await agent.prompt(prompt=[text_block("search")], session_id=sid)
    assert str(again.value).endswith("Web search stays off") and len(opener.opened) == 2
    assert (await agent.prompt(prompt=[text_block("go")], session_id=sid)).stop_reason == "end_turn"
    await agent.aclose()

    # A journaled toggle that can no longer start here is left off at a load instead of failing it.
    agent, _, opener = make_agent(tmp_path, [Turn(text="ok")], unavailable={"cap.browser": "no browser extra"})
    loaded = await agent.load_session(cwd=str(work), session_id=sid)
    assert loaded.config_options is not None and "cap.browser" not in toggle_values(loaded.config_options)
    caps = opener.opened[-1].capabilities
    assert caps is None or not caps.browser.enabled
    await agent.aclose()


async def test_a_switch_on_installs_what_its_capability_lacks_before_the_reopen(
    tmp_path: Path, work: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(Needs, "installable", lambda self: True)
    reasons = {
        "cap.browser": "Browser needs the browser extra: install intern-agent[browser]",
        "cap.code-search": "Code search needs the code-search extra: install intern-agent[code-search]",
        "cap.web-search": "Web search is not set up: set INTERN_SEARCH_PROVIDER",
    }
    started, release = asyncio.Event(), asyncio.Event()
    installed: list[str] = []
    stopped: list[str] = []

    async def install(needs: Needs, say: Say) -> None:
        await say(f"Installing the {needs.extra} extra…")
        if needs.extra == "code-search":
            raise InstallFailed("Installing the code-search extra failed: offline. By hand: uv pip install scipy")
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            stopped.append(needs.extra)
            raise
        installed.append(needs.extra)
        reasons.pop("cap.browser")

    agent, client, opener = make_agent(tmp_path, [Turn(text="ok")] * 2, unavailable=reasons, install=install)
    session = await agent.new_session(cwd=str(work))
    sid = session.session_id
    # An extra this server installs does not keep a switch from being offered; a missing search provider does.
    assert session.config_options is not None
    offered = toggle_values(session.config_options)
    assert "cap.browser" in offered and "cap.code-search" in offered and "cap.web-search" not in offered

    # A cancel during the install stops it, and the switch stays pending for the next prompt.
    await agent.set_config_option(config_id="cap.browser", session_id=sid, value=True)
    prompt = asyncio.create_task(agent.prompt(prompt=[text_block("browse")], session_id=sid))
    await started.wait()
    await agent.cancel(session_id=sid)
    assert (await prompt).stop_reason == "cancelled"
    live = agent.sessions[sid]
    assert stopped == ["browser"] and installed == [] and len(opener.opened) == 1
    assert live.pending is not None and live.pending.browser.enabled

    # The next prompt installs, says so as it goes, and reopens on it; a failed install stays off, with why.
    started.clear()
    release.set()
    await agent.set_config_option(config_id="cap.code-search", session_id=sid, value=True)
    with pytest.raises(RequestError) as failed:
        await agent.prompt(prompt=[text_block("browse")], session_id=sid)
    assert str(failed.value) == (
        "Installing the code-search extra failed: offline. By hand: uv pip install scipy. "
        "Code search stays off; the other capability changes applied"
    )
    assert installed == ["browser"] and "Installing the browser extra…" in client.text()
    assert len(opener.opened) == 2 and live.capabilities.browser.enabled and not live.capabilities.code_search.enabled
    assert (await agent.prompt(prompt=[text_block("browse")], session_id=sid)).stop_reason == "end_turn"
    await agent.aclose()


async def test_a_session_opens_on_the_toggles_its_meta_asks_for(tmp_path: Path, work: Path):
    agent, _, opener = make_agent(tmp_path, [Turn(text="ok")] * 2)
    session = await agent.new_session(cwd=str(work), nace={"capabilities": {"cap.web-read": True}})
    sid = session.session_id
    assert session.config_options is not None
    assert toggle_values(session.config_options)["cap.web-read"] is True
    caps = opener.opened[-1].capabilities
    assert caps is not None and caps.tool_packs == ["base", "web-read"]
    await agent.prompt(prompt=[text_block("go")], session_id=sid)
    assert len(opener.opened) == 1
    # A client of 0.22 or before may still ask for `cap.process`: the session opens, and it changes nothing.
    older = await agent.new_session(cwd=str(work), nace={"capabilities": {"cap.process": True}})
    assert older.config_options is not None and "cap.process" not in toggle_values(older.config_options)
    caps = opener.opened[-1].capabilities
    assert caps is None or caps.tool_packs is None
    for picks in ({"cap.nothing": True}, {"cap.web-read": "true"}, {"cap.web-read": 1}):
        with pytest.raises(RequestError) as refused:
            await agent.new_session(cwd=str(work), nace={"capabilities": picks})
        assert refused.value.code == -32602, picks
    await agent.aclose()

    # Journaled: a load without _meta opens on them, and a load's _meta is set over them.
    agent, _, opener = make_agent(tmp_path, [Turn(text="ok")])
    await agent.load_session(cwd=str(work), session_id=sid, nace={"capabilities": {"cap.memory": False}})
    caps = opener.opened[-1].capabilities
    assert caps is not None and caps.tool_packs == ["base", "web-read"] and not caps.memory.enabled
    await agent.aclose()


def test_bootstrap_requires_system_1_and_serves_picks_from_the_console_or_the_decisions_api(tmp_path: Path):
    options = Options(cwd=tmp_path)
    home = {"INTERN_HOME": str(tmp_path / "home")}
    # The default is a decision model from the console; without its key the session is refused.
    assert Bootstrap(options, home).system1_choice() == "drex-latest"
    with pytest.raises(ConfigError, match="NACE_API_KEY"):
        Bootstrap(options, home).system1_backend("drex-latest")
    keyed = Bootstrap(options, {**home, "NACE_API_KEY": "k"})
    default = keyed.system1_backend("drex-latest")
    assert isinstance(default, DecisionsApiBackend) and default.name == "decisions-api:drex-latest"
    assert default._url == "https://console.nace.ai/v1/systemone"  # pyright: ignore[reportPrivateUsage]
    other = keyed.system1_backend("drex-max-v1.1")
    assert isinstance(other, DecisionsApiBackend) and other.name == "decisions-api:drex-max-v1.1"
    # Heuristics are gone: there is no System 1 that is not a model.
    with pytest.raises(ConfigError, match="System 1 is required"):
        Bootstrap(options.model_copy(update={"system1": "heuristic"}), home).system1_choice()
    # A decision model on the decisions API, picked as decisions:<model> or started on with --system1, with its
    # own endpoint, fallbacks and key.
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        keyed.system1_backend("decisions:~typesafe/jev-latest")
    started = Bootstrap(
        options.model_copy(update={"system1": "decisions:nace-ai/drex-x,~typesafe/jev-latest"}),
        {**home, "OPENROUTER_API_KEY": "o"},
    )
    assert started.system1_choice() == "decisions:nace-ai/drex-x"
    configured = started.system1_backend("decisions:nace-ai/drex-x")
    assert isinstance(configured, DecisionsApiBackend)
    assert configured._models == ["nace-ai/drex-x", "~typesafe/jev-latest"]  # pyright: ignore[reportPrivateUsage]
    assert configured._url == "https://openrouter.ai/api/v1/systemone"  # pyright: ignore[reportPrivateUsage]
    # drex:<model>[,...] names console models too; $DREX_BASE_URL and $DREX_API_KEY stand in for the config.
    prefixed = Bootstrap(
        options.model_copy(update={"system1": "drex:drex-v2.0,drex-max-v1.1"}),
        {**home, "DREX_BASE_URL": "https://console.example", "DREX_API_KEY": "d"},
    )
    assert prefixed.system1_choice() == "drex-v2.0"
    console = prefixed.system1_backend("drex-v2.0")
    assert isinstance(console, DecisionsApiBackend)
    assert console._models == ["drex-v2.0", "drex-max-v1.1"]  # pyright: ignore[reportPrivateUsage]
    assert console._url == "https://console.example/v1/systemone"  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ConfigError, match="must be"):
        Bootstrap(options.model_copy(update={"system1": "what:ever"}), home).system1_choice()


# Diffs: write and edit carry the file before and after to the client.


def test_a_write_or_edit_end_carries_an_acp_diff():
    diff = FileDiff(path="/work/a.py", old_text="x = 1\n", new_text="x = 2\n")
    event = ToolCallEndEvent(
        turn=1, call_id="c1", name="edit", status="ok", content="Replaced", duration_ms=1, diff=diff
    )
    [update] = acp_mapping.event_updates(event, CTX)
    assert isinstance(update, ToolCallProgress) and update.content is not None
    text, edit = update.content
    assert isinstance(text, ContentToolCallContent)
    assert isinstance(edit, FileEditToolCallContent)
    assert (edit.type, edit.path, edit.old_text, edit.new_text) == ("diff", "/work/a.py", "x = 1\n", "x = 2\n")


async def test_an_edit_in_a_turn_reaches_the_client_as_a_diff(tmp_path: Path, work: Path):
    (work / "a.txt").write_text("old line\n")
    script = [Turn(tool_calls=[("edit", {"path": "a.txt", "old_string": "old", "new_string": "new"})]), Turn(text="ok")]
    agent, client, _ = make_agent(tmp_path, script)
    session = await agent.new_session(cwd=str(work))
    await agent.prompt(prompt=[text_block("edit")], session_id=session.session_id)
    [end] = client.of(ToolCallProgress)
    assert end.content is not None
    [edit] = [c for c in end.content if isinstance(c, FileEditToolCallContent)]
    assert (edit.path, edit.old_text, edit.new_text) == (str((work / "a.txt").resolve()), "old line\n", "new line\n")
    # Diffs reach clients only: the journal keeps the tool result without one.
    journal = agent.sessions[session.session_id].session.deps.journal
    assert journal.path is not None and "old line" not in journal.path.read_text()
    await agent.aclose()


# Stdin closing: requests already received still get their replies.


def test_requests_sent_before_stdin_closes_are_all_answered(tmp_path: Path, work: Path):
    # The model asks for a long command; stdin closes while it runs, so that prompt answers cancelled.
    replies = [Reply(tool_calls=[("bash", {"command": "sleep 30"})]), Reply(text="never")]
    requests = [
        {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": 1}},
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": str(work), "mcpServers": []}},
    ]
    with FakeOpenAI(replies) as server:
        env = {**os.environ, "VLLM_BASE_URL": server.url, "INTERN_HOME": str(tmp_path / "home")}
        env |= server.system1_env
        process = subprocess.Popen(
            [sys.executable, "-m", "intern", "acp", "--no-context-files", "--system2", f"vllm/{MODEL}"],
            env=env, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )  # fmt: skip
        stdin, stdout = process.stdin, process.stdout
        assert stdin is not None and stdout is not None
        stdin.write("".join(json.dumps(r) + "\n" for r in requests))
        stdin.flush()
        answered: dict[int, dict] = {}
        while 1 not in answered:
            message = json.loads(stdout.readline())
            if "id" in message:
                answered[message["id"]] = message
        session_id = answered[1]["result"]["sessionId"]
        prompt = {"sessionId": session_id, "prompt": [{"type": "text", "text": "wait"}]}
        stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "session/prompt", "params": prompt}) + "\n")
        stdin.flush()
        while not json.loads(stdout.readline()).get("params", {}).get("update", {}).get("toolCallId"):
            pass  # until the bash call has started
        stdin.close()
        rest = [json.loads(line) for line in stdout]
        assert process.wait(timeout=20) == 0
    [reply] = [m for m in rest if m.get("id") == 2]
    assert reply["result"]["stopReason"] == "cancelled"


def test_acp_refuses_stdio_that_is_not_a_pipe(tmp_path: Path, work: Path):
    with (tmp_path / "out.txt").open("w") as out:
        result = subprocess.run(
            [sys.executable, "-m", "intern", "acp", "--system2", "vllm/x"],
            env={**os.environ, "INTERN_HOME": str(tmp_path / "home")},
            cwd=work, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.PIPE, text=True, timeout=30,
        )  # fmt: skip
    assert result.returncode == 2
    assert result.stderr.strip() == (
        "intern: acp talks over pipes: start it from an ACP client, not with stdin or stdout redirected to a file"
    )
