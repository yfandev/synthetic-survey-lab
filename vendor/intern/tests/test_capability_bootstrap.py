"""Saved capability settings through the real shared Bootstrap/Session boundary."""

from __future__ import annotations

import builtins
import importlib
import json
import traceback
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import intern.config as config_module
import intern.session_setup as capability_setup
from intern.capability_settings import CapabilitySettings
from intern.config import Bootstrap, Options
from intern.decisions import DecisionsApiBackend, NoulAnswer, NoulQuestion, Question
from intern.events import WindowInfo
from intern.loop import ModelBinding
from intern.models.profile import Profile
from intern.models.registry import ConfigError
from intern.providers.fake import FakeProvider, Turn
from intern.tools.web_read import WebReadTool
from intern.web.search import HTTPSearchProvider

from .fake_openai import quiet
from .support import Quiet


def make_binding(
    ref: str, script: list[Turn], *, profile: Profile | None = None, advertised: int = 65_536, working: int = 32_768
) -> tuple[ModelBinding, FakeProvider]:
    provider = FakeProvider(script, model=ref)
    return (
        ModelBinding(
            provider=provider,
            ref=ref,
            wire_model=ref.split("/", 1)[1],
            profile=profile or Profile(),
            window=WindowInfo(
                advertised=advertised, advertised_source="profile", effective=None, user_cap=working, working=working
            ),
        ),
        provider,
    )


def make_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bindings: dict[str, ModelBinding],
    text: str = 'system2 = "fake/primary"\n',
    *,
    capabilities: CapabilitySettings | None = None,
    env: dict[str, str] | None = None,
    system1: str | None = None,
    quiet: bool = True,
) -> Bootstrap:
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir(exist_ok=True)
    work.mkdir(exist_ok=True)
    (home / "config.toml").write_text(text)

    async def bind(self: Bootstrap, ref: str) -> ModelBinding:
        original = bindings[ref]
        return replace(original, profile=original.profile.model_copy(deep=True))

    monkeypatch.setattr(Bootstrap, "bind", bind)
    if quiet:
        monkeypatch.setattr(Bootstrap, "_system1", lambda self, spec=None: Quiet())
    return Bootstrap(
        Options(
            cwd=work,
            system1=system1,
            persist=False,
            context_files=False,
            extensions=False,
            capabilities=capabilities,
            memory_off=True,
        ),
        {**(env or {}), "INTERN_HOME": str(home)},
    )


async def test_default_bootstrap_keeps_the_base_tools_and_never_opens_optional_services(tmp_path, monkeypatch):
    primary, provider = make_binding("fake/primary", [Turn(text="done")])
    bootstrap = make_bootstrap(tmp_path, monkeypatch, {primary.ref: primary})
    ordinary_import = importlib.import_module
    builtin_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in {"playwright", "mcp", "jsonschema"}
        return ordinary_import(name, *args, **kwargs)

    def guarded_builtin(name, *args, **kwargs):
        assert name.split(".", 1)[0] not in {"playwright", "mcp", "jsonschema"}
        return builtin_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled optional service was started")

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    monkeypatch.setattr(builtins, "__import__", guarded_builtin)
    monkeypatch.setattr(capability_setup, "BrowserSession", forbidden)
    monkeypatch.setattr(capability_setup, "stdio_connection", forbidden)
    async with await bootstrap.session(lambda event: None) as session:
        # The job manager is the session's, not an optional service: it starts nothing until a command runs.
        tools = ["read", "write", "edit", "bash", "grep", "find", "process"]
        assert [tool.name for tool in session.deps.tools] == tools
        assert session.deps.processes is not None and session.deps.processes.running() == []
        assert session.deps.system2.provider is provider
        assert session.model_budget is None
        assert (await session.run("finish the task")).final_text == "done"
        assert len(provider.requests[0].tools) == len(tools)
    assert session.closed
    assert not (bootstrap.home / "session-resources").exists()


async def test_saved_toml_and_whole_object_programmatic_override_have_clear_precedence(tmp_path, monkeypatch):
    primary, _ = make_binding("fake/primary", [])
    text = """system2 = "fake/primary"
[capabilities]
tool_packs = ["base", "web-read"]
[capabilities.reader]
timeout_s = 11
"""
    bootstrap = make_bootstrap(tmp_path, monkeypatch, {primary.ref: primary}, text)
    async with await bootstrap.session(lambda event: None) as saved:
        reader = next(tool for tool in saved.deps.tools if isinstance(tool, WebReadTool))
        assert reader.reader.settings.timeout_s == 11
    override = CapabilitySettings(tool_packs=["base", "web-read"])
    bootstrap = make_bootstrap(tmp_path, monkeypatch, {primary.ref: primary}, text, capabilities=override)
    async with await bootstrap.session(lambda event: None) as explicit:
        reader = next(tool for tool in explicit.deps.tools if isinstance(tool, WebReadTool))
        assert reader.reader.settings.timeout_s == 30  # Object replacement, not a hidden partial merge.
    bootstrap = make_bootstrap(
        tmp_path, monkeypatch, {primary.ref: primary}, text, capabilities=CapabilitySettings(tool_packs=[])
    )
    async with await bootstrap.session(lambda event: None) as empty:
        assert empty.deps.tools == []


@pytest.mark.parametrize("profile_packs", [["base"], ["base", "web-read"]])
async def test_capability_options_configure_profile_selected_packs_without_enabling_them(
    tmp_path, monkeypatch, profile_packs
):
    primary, _ = make_binding("fake/primary", [], profile=Profile(tool_packs=profile_packs))
    bootstrap = make_bootstrap(
        tmp_path,
        monkeypatch,
        {primary.ref: primary},
        'system2 = "fake/primary"\n[capabilities.reader]\ntimeout_s = 12\n',
    )
    async with await bootstrap.session(lambda event: None) as session:
        readers = [tool for tool in session.deps.tools if isinstance(tool, WebReadTool)]
        assert bool(readers) == ("web-read" in profile_packs)
        if readers:
            assert readers[0].reader.settings.timeout_s == 12
    assert primary.profile.tool_packs == profile_packs


@pytest.mark.parametrize("explicit_backend", [False, True])
async def test_search_uses_bootstrap_environment_snapshot_and_explicit_backend_precedence(
    tmp_path, monkeypatch, explicit_backend
):
    primary, _ = make_binding(
        "fake/primary", [Turn(tool_calls=[("web_search", {"query": "release notes"})]), Turn(text="done")]
    )
    seen: list[httpx.Request] = []

    def reply(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []} if explicit_backend else {"grounding": {"generic": []}})

    monkeypatch.setattr(
        capability_setup,
        "HTTPSearchProvider",
        lambda settings: HTTPSearchProvider(settings, transport=httpx.MockTransport(reply)),
    )
    text = 'system2 = "fake/primary"\n[capabilities]\ntool_packs = ["base", "web-search"]\n'
    if explicit_backend:
        text += '[capabilities.search]\nbackend = "tavily"\napi_key_env = "SELECTED_SEARCH_KEY"\n'
    env = {
        "INTERN_SEARCH_PROVIDER": "brave",
        "BRAVE_SEARCH_API_KEY": "snapshot-brave",
        "SELECTED_SEARCH_KEY": "snapshot-tavily",
    }
    bootstrap = make_bootstrap(tmp_path, monkeypatch, {primary.ref: primary}, text, env=env)
    env.update(INTERN_SEARCH_PROVIDER="exa", BRAVE_SEARCH_API_KEY="changed-source")
    monkeypatch.setenv("INTERN_SEARCH_PROVIDER", "exa")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "changed-global")
    async with await bootstrap.session(lambda event: None) as session:
        assert (await session.run("search")).reason == "completed"
    assert len(seen) == 1
    if explicit_backend:
        assert seen[0].url.host == "api.tavily.com" and seen[0].headers["Authorization"] == "Bearer snapshot-tavily"
    else:
        assert (
            seen[0].url.host == "api.search.brave.com" and seen[0].headers["X-Subscription-Token"] == "snapshot-brave"
        )


async def test_native_system1_fallback_and_shared_session_budget_survive_model_switch(tmp_path, monkeypatch):
    primary, primary_provider = make_binding("fake/primary", [Turn(error="unavailable", retryable=True)])
    backup, backup_provider = make_binding("fake/backup", [Turn(text="backup answer")])
    replacement, replacement_provider = make_binding("fake/replacement", [Turn(text="replacement answer")])
    native_calls: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        native_calls.append(body["model"])
        if body["model"] == "s1-primary":
            return httpx.Response(503, text="unavailable")
        answers = quiet(body["questions"])
        if "ready" in answers:
            answers["ready"] = {"type": "noul", "noul": 0.9}
        return httpx.Response(200, json={"answers": answers, "usage": {"cost": 0.02}})

    def native_backend(base_url, api_key, models, **kwargs):
        return DecisionsApiBackend(base_url, api_key, models, transport=httpx.MockTransport(answer), **kwargs)

    monkeypatch.setattr(config_module, "DecisionsApiBackend", native_backend)
    bootstrap = make_bootstrap(
        tmp_path,
        monkeypatch,
        {b.ref: b for b in (primary, backup, replacement)},
        """system2 = "fake/primary"
[system1]
backend = "decisions-api"
model = "s1-primary"
fallback_models = ["s1-backup"]
api_key_env = "DECISION_KEY"
[capabilities]
system2_fallbacks = ["fake/backup"]
[capabilities.session_budget]
max_calls = 4
max_observed_cost_usd = 0.1
""",
        env={"DECISION_KEY": "synthetic-native-key"},
        quiet=False,
    )
    async with await bootstrap.session(lambda event: None) as session:
        budget = session.model_budget
        assert budget is not None
        questions: dict[str, Question] = {"ready": NoulQuestion(instructions="Is this ready?")}
        # The check at open reaches the native list but no run's budget.
        assert native_calls == ["s1-primary", "s1-backup"]
        assert (await session.deps.decisions.ask("review", "same State", questions))["ready"] == NoulAnswer(noul=0.9)
        assert (await session.run("first task")).final_text == "backup answer"
        first = await budget.snapshot()
        assert first.calls == 3 and first.completed_calls == 2 and first.unknown_usage_calls == 1
        assert first.unknown_output_calls == 1 and first.usage.cost_usd == pytest.approx(0.02)
        assert native_calls == ["s1-primary", "s1-backup"] * 2  # Native list is one logical S1 dispatch.
        await session.set_model("fake/replacement")
        assert session.model_budget is budget
        assert (await session.run("second task")).final_text == "replacement answer"
        await session.deps.decisions.ask("review", "same State", questions)
        assert native_calls == ["s1-primary", "s1-backup"] * 2  # Same DecisionClient cache remains authoritative.
        assert (await budget.snapshot()).calls == 4
        assert (await session.run("budget must remain exhausted")).reason == "provider_error"
        assert (
            len(primary_provider.requests) == len(backup_provider.requests) == len(replacement_provider.requests) == 1
        )
        assert (await budget.snapshot()).calls == 4


@pytest.mark.parametrize("incompatibility", ["working", "advertised", "sampling", "effort", "compat", "vision"])
async def test_incompatible_backup_is_refused_before_optional_resource_side_effects(
    tmp_path, monkeypatch, incompatibility
):
    primary, _ = make_binding("fake/primary", [])
    profile = Profile()
    if incompatibility == "sampling":
        profile.sampling.temperature = 0.5
    elif incompatibility == "effort":
        profile.reasoning_effort = "high"
    elif incompatibility == "compat":
        profile.compat.stream = False
    elif incompatibility == "vision":
        profile.vision = True
    backup, _ = make_binding(
        "fake/backup",
        [],
        profile=profile,
        advertised=32_768 if incompatibility == "advertised" else 65_536,
        working=16_384 if incompatibility == "working" else 32_768,
    )
    started: list[str] = []

    def forbidden(*args, **kwargs):
        started.append("resource startup")
        raise AssertionError("incompatible model must be rejected before resource creation")

    monkeypatch.setattr(capability_setup, "BrowserSession", forbidden)
    monkeypatch.setattr(capability_setup, "stdio_connection", forbidden)
    bootstrap = make_bootstrap(
        tmp_path,
        monkeypatch,
        {primary.ref: primary, backup.ref: backup},
        """system2 = "fake/primary"
[capabilities]
system2_fallbacks = ["fake/backup"]
[capabilities.browser]
enabled = true
[[capabilities.mcp]]
namespace = "never"
transport = "stdio"
command = "must-not-start"
remote_tools = { echo = [] }
""",
    )
    with pytest.raises(ConfigError, match="compatible request settings"):
        await bootstrap.session(lambda event: None)
    assert not started


@pytest.mark.parametrize(
    "invalid",
    [
        '[capabilities.search]\napi_key = "private-synthetic-credential"\n',
        '[[capabilities.mcp]]\nnamespace="docs"\ntransport="http"\n'
        'url="https://user:private-synthetic-credential@example.test/mcp"\nremote_tools={echo=[]}\n',
    ],
)
def test_saved_settings_errors_withhold_credentials_even_in_formatted_tracebacks(tmp_path, monkeypatch, invalid):
    primary, _ = make_binding("fake/primary", [])
    with pytest.raises(ConfigError) as caught:
        make_bootstrap(tmp_path, monkeypatch, {primary.ref: primary}, 'system2="fake/primary"\n' + invalid)
    assert "private-synthetic-credential" not in str(caught.value)
    assert "private-synthetic-credential" not in "".join(traceback.format_exception(caught.value))
