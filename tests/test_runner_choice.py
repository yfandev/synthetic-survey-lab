"""The runner as one per-session choice: `[capabilities.runner]` in config, the ACP `runner` select, `/runner`."""

from __future__ import annotations

from pathlib import Path

import pytest
from acp import RequestError
from pydantic import BaseModel, ConfigDict

import intern.runners as registry
from intern.capability_settings import CapabilitySettings, RunnerConfig
from intern.executor import Executor, LocalExecutor
from intern.journal import CapabilitiesChangeEntry, Journal
from intern.runners import Runner
from intern.session_setup import runner_choices, runner_config, runner_problem

from .test_acp import Turn, make_agent, text_block
from .test_runners import CopyBackend, CopyOptions, EntryPoint


class NeedyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint: str


class NeedyRunner(Runner):
    """A runner that cannot run without options of its own, as a remote sandbox needs its server."""

    name = "needy"
    title = "Needy sandbox"
    note = "commands run on a server the options name"
    Options = NeedyOptions
    isolated = False

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        return LocalExecutor()


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    points = [EntryPoint("copy", CopyBackend), EntryPoint("needy", NeedyRunner)]
    monkeypatch.setattr(registry, "entry_points", lambda group: points)
    registry.registered.cache_clear()
    yield
    registry.registered.cache_clear()


def configured_copy(root: Path) -> CapabilitySettings:
    return CapabilitySettings.model_validate({"runner": {"name": "copy", "options": {"root": str(root)}}})


def test_every_runner_is_one_choice_with_the_configured_one_as_default(tmp_path):
    choices = runner_choices(configured_copy(tmp_path).runner)
    assert [(c.id, c.default) for c in choices] == [
        ("local", False),
        ("docker", False),
        ("copy", True),
        ("needy", False),
    ]
    local, docker, copy, needy = choices
    assert (local.name, docker.name, needy.name) == ("Local", "Docker", "Needy sandbox")
    assert needy.description == "commands run on a server the options name"
    # The configured runner has its options; one that cannot run on defaults says so before anyone picks it.
    assert copy.unavailable is None and local.unavailable is None and docker.unavailable is None
    assert needy.unavailable == (
        "needy needs its options: set [capabilities.runner] name = 'needy' and its "
        "[capabilities.runner.options] in config.toml"
    )


def test_a_picked_runner_gets_its_own_options_never_another_runners(tmp_path):
    configured = configured_copy(tmp_path).runner
    again = runner_config("copy", configured)
    assert isinstance(again.options, CopyOptions) and again.options.root == tmp_path
    docker = runner_config("docker", configured)
    assert docker.name == "docker" and docker.options.model_dump()["memory_mb"] == 512  # Docker's defaults
    with pytest.raises(ValueError, match="needy needs its options"):
        runner_config("needy", configured)
    # Without options the copy runner cannot run either, unless config.toml gives them.
    assert runner_problem("copy", RunnerConfig()) is not None
    assert runner_problem("copy", configured) is None


def test_a_configured_module_class_runner_is_listed_too():
    spec = "tests.test_runner_choice:NeedyRunner"
    configured = RunnerConfig.model_validate({"name": spec, "options": {"endpoint": "x"}})
    assert [c.id for c in runner_choices(configured)][-1] == spec


async def test_an_acp_thread_picks_its_runner_once_and_it_applies_at_the_next_prompt(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    configured = configured_copy(tmp_path)
    agent, _, opener = make_agent(tmp_path, lambda request: Turn(text="ok"), configured)
    session = await agent.new_session(cwd=str(work))
    sid = session.session_id
    assert session.config_options is not None
    (option,) = [o for o in session.config_options if o.id == "runner"]
    assert option.type == "select" and option.category == "_nace_runner" and option.current_value == "copy"
    assert [v.value for v in option.options] == ["local", "docker", "copy", "needy"]
    needy = next(v for v in option.options if v.value == "needy")
    assert (needy.description or "").startswith("Unavailable: needy needs its options")
    # No runner is a capability switch any more.
    assert not any(o.id.startswith(("cap.docker", "cap.executor")) for o in session.config_options)
    await agent.prompt(prompt=[text_block("first")], session_id=sid)

    picked = await agent.set_config_option(config_id="runner", session_id=sid, value="docker")
    assert next(o for o in picked.config_options if o.id == "runner").current_value == "docker"
    assert len(opener.opened) == 1  # pending until the next prompt
    await agent.prompt(prompt=[text_block("second")], session_id=sid)
    reopened = opener.opened[-1].capabilities
    assert reopened is not None and reopened.runner.name == "docker"

    # A runner that cannot run here is refused at once, with why, and the session keeps the one it has.
    for value, reason in (("needy", "needy needs its options"), ("nowhere", "Unknown runner 'nowhere'")):
        with pytest.raises(RequestError) as refused:
            await agent.set_config_option(config_id="runner", session_id=sid, value=value)
        assert refused.value.code == -32602 and reason in str(refused.value.data or refused.value)

    # Back to the configured runner: the session reopens on the configured settings themselves.
    await agent.set_config_option(config_id="runner", session_id=sid, value="copy")
    await agent.prompt(prompt=[text_block("third")], session_id=sid)
    assert len(opener.opened) == 3 and opener.opened[-1].capabilities is None
    await agent.aclose()


async def test_a_loaded_session_continues_on_the_runner_it_last_ran_on(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    agent, _, opener = make_agent(tmp_path, lambda request: Turn(text="ok"))
    sid = (await agent.new_session(cwd=str(work))).session_id
    await agent.set_config_option(config_id="runner", session_id=sid, value="docker")
    await agent.prompt(prompt=[text_block("first")], session_id=sid)
    journal = agent.sessions[sid].session.deps.journal
    assert isinstance(journal, Journal) and journal.last_runner() == "docker"
    assert any(isinstance(e, CapabilitiesChangeEntry) and e.runner == "docker" for e in journal.active_path())
    await agent.aclose()

    agent, _, opener = make_agent(tmp_path, lambda request: Turn(text="ok"))
    loaded = await agent.load_session(cwd=str(work), session_id=sid)
    assert loaded.config_options is not None
    assert next(o for o in loaded.config_options if o.id == "runner").current_value == "docker"
    reopened = opener.opened[-1].capabilities
    assert reopened is not None and reopened.runner.name == "docker"
    await agent.aclose()


async def test_a_session_opens_on_the_runner_its_meta_asks_for(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    agent, _, opener = make_agent(tmp_path, lambda request: Turn(text="ok"))
    session = await agent.new_session(cwd=str(work), nace={"runner": "docker"})
    assert session.config_options is not None
    assert next(o for o in session.config_options if o.id == "runner").current_value == "docker"
    opened = opener.opened[-1].capabilities
    assert opened is not None and opened.runner.name == "docker"
    with pytest.raises(RequestError) as refused:
        await agent.new_session(cwd=str(work), nace={"runner": "needy"})
    assert refused.value.code == -32602
    await agent.aclose()
