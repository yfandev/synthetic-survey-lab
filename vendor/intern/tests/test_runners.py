"""Runners by name: the built-ins, a package's runner, and a sandbox the whole session follows."""

from __future__ import annotations

import asyncio
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import intern.runners as registry
from intern.capability_settings import CapabilitySettings
from intern.executor import ExecResult, Executor, LocalExecutor, WorkspaceMount, mount_of
from intern.guard import Scope
from intern.models.registry import ConfigError
from intern.runners import DockerRunner, LocalRunner, Runner, load_runner
from intern.runners.docker import DockerConfig, DockerExecutor
from intern.secrets import EnvironmentSecrets, SecretExecutor

from .fake_openai import Reply
from .test_session_setup import configured, outputs


class CopyOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    root: Path


class CopySandbox:
    """A stand-in for a remote sandbox: the workspace is copied to another folder on first use, commands run there,
    and the copy is synced back after each one. The tools see the copy's path, never the host's."""

    instances: ClassVar[list[CopySandbox]] = []

    def __init__(self, workspace: Path, root: Path) -> None:
        # Kept as given, symlinks and all: the mount must work without the backend resolving its paths.
        self.workspace, self.root = workspace, root / "sandbox"
        self.started = self.closed = False
        self.commands: list[str] = []
        CopySandbox.instances.append(self)

    @property
    def mount(self) -> WorkspaceMount:
        return WorkspaceMount(self.workspace, PurePosixPath(self.root))

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_s: float,
        max_output_bytes: int,
        abort: asyncio.Event | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult:
        if not self.started:
            shutil.copytree(self.workspace, self.root)
            self.started = True
        self.commands.append(command)
        inner = Path(self.mount.to_inner(cwd))
        result = await LocalExecutor().run(
            command, cwd=inner, timeout_s=timeout_s, max_output_bytes=max_output_bytes, abort=abort, env=env
        )
        shutil.copytree(self.root, self.workspace, dirs_exist_ok=True)
        return result

    async def aclose(self) -> None:
        self.closed = True
        shutil.rmtree(self.root, ignore_errors=True)


class CopyBackend(Runner):
    name = "copy"
    Options = CopyOptions

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        assert isinstance(options, CopyOptions)
        if not workspace.is_dir():
            raise ValueError("The copy sandbox needs an existing workspace")
        return CopySandbox(workspace, options.root)


class EntryPoint:
    def __init__(self, name: str, target: object) -> None:
        self.name, self.target = name, target

    def load(self) -> object:
        return self.target


@pytest.fixture(autouse=True)
def fresh_registry():
    """Tests replace the installed entry points; the registry is read again for each."""
    registry.registered.cache_clear()
    yield
    registry.registered.cache_clear()


@pytest.fixture
def copy_registered(monkeypatch):
    CopySandbox.instances.clear()
    monkeypatch.setattr(registry, "entry_points", lambda group: [EntryPoint("copy", CopyBackend)])
    registry.registered.cache_clear()
    yield
    registry.registered.cache_clear()


def test_builtins_load_by_name_and_classes_by_module_path():
    assert isinstance(load_runner("local"), LocalRunner) and not LocalRunner.isolated
    assert isinstance(load_runner("docker"), DockerRunner) and DockerRunner.Options is DockerConfig
    assert isinstance(load_runner("intern.runners.docker:DockerRunner"), DockerRunner)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("e2b", "Unknown runner 'e2b'; use one of local, docker, opensandbox"),
        ("intern.missing_module:Backend", "Cannot load runner"),
        ("intern.executor:LocalExecutor", "is not a Runner subclass"),
    ],
)
def test_a_backend_that_cannot_load_names_what_is_wrong(spec, message):
    with pytest.raises(ValueError, match=message):
        load_runner(spec)


def test_a_package_backend_is_found_by_its_entry_point_name(copy_registered):
    assert isinstance(load_runner("copy"), CopyBackend)
    with pytest.raises(ValueError, match="use one of local, docker, copy"):
        load_runner("e2b")


def test_a_package_backend_validates_its_own_options_when_the_config_loads(copy_registered, tmp_path):
    settings = CapabilitySettings.model_validate({"runner": {"name": "copy", "options": {"root": str(tmp_path)}}})
    assert isinstance(settings.runner.options, CopyOptions) and settings.runner.options.root == tmp_path
    assert settings.model_dump()["runner"]["options"] == {"root": tmp_path}
    with pytest.raises(ValidationError, match="Invalid options for runner 'copy': root: Field required"):
        CapabilitySettings.model_validate({"runner": {"name": "copy"}})
    # `process` is a base tool, which a runner that runs no background jobs goes without: listing it is no error.
    CapabilitySettings.model_validate(
        {"tool_packs": ["base", "process"], "runner": {"name": "copy", "options": {"root": str(tmp_path)}}}
    )


def test_option_errors_name_the_field_but_never_echo_its_value():
    with pytest.raises(ValidationError) as caught:
        CapabilitySettings.model_validate(
            {"runner": {"name": "docker", "options": {"image": "private synthetic value"}}}
        )
    assert "image" in str(caught.value) and "private synthetic value" not in str(caught.value)


def test_a_workspace_only_applies_to_an_isolated_backend():
    with pytest.raises(ValidationError, match="workspace applies to an isolated runner, not 'local'"):
        CapabilitySettings.model_validate({"runner": {"workspace": "/tmp/work"}})


def test_the_mount_is_passed_through_wrappers_and_drives_the_guard_scope(tmp_path):
    work = tmp_path / "work" / "pkg"
    work.mkdir(parents=True)
    docker = DockerExecutor(tmp_path / "work")
    wrapped = SecretExecutor(docker, EnvironmentSecrets([]), scope="executor", inject={})
    assert mount_of(LocalExecutor()) is None
    assert mount_of(wrapped) == WorkspaceMount((tmp_path / "work").resolve(), PurePosixPath("/workspace"))
    scope = Scope.of(work, wrapped, Path.home())
    assert scope.cwd == PurePosixPath("/workspace/pkg")
    assert scope.roots == (PurePosixPath(work.resolve()), PurePosixPath("/workspace/pkg"))
    with pytest.raises(ValueError):
        docker.mount.to_inner(tmp_path)


async def test_shell_and_file_tools_follow_a_package_sandbox_and_it_closes_with_the_session(copy_registered, tmp_path):
    (tmp_path / "remote").mkdir()
    sandbox_root = tmp_path / "remote-link"  # a symlinked sandbox root, as /home can be inside an image
    sandbox_root.symlink_to(tmp_path / "remote")
    toml = f"""[capabilities]
tool_packs = ["base", "process"]
[capabilities.runner]
name = "copy"
[capabilities.runner.options]
root = "{sandbox_root}"
"""
    replies = [
        Reply(tool_calls=[("write", {"path": "notes.txt", "content": "from the sandbox"})]),
        Reply(tool_calls=[("bash", {"command": "pwd; cat notes.txt"})]),
        Reply(tool_calls=[("read", {"path": "notes.txt"})]),
        Reply(text="done"),
    ]
    async with configured(tmp_path, replies, toml) as (session, _, _, work):
        (sandbox,) = CopySandbox.instances
        assert not sandbox.started  # nothing starts at setup
        # It runs no background jobs: the session opens without the process tool rather than refusing.
        assert "process" not in [tool.name for tool in session.deps.tools] and session.deps.processes is None
        # What each run's guard is built from: it sees the sandbox's path for the working directory.
        assert Scope.of(session.deps.cwd, session.deps.executor, Path.home()).cwd == PurePosixPath(sandbox.root)
        await session.run("write a note and read it back")
        _, bash, read = outputs(session)
        assert sandbox.started and len(sandbox.commands) == 3  # write and read went through the sandbox too
        assert str(sandbox.root.resolve()) in bash and "from the sandbox" in bash
        assert "from the sandbox" in read
        assert (work / "notes.txt").read_text() == "from the sandbox"  # synced back to the host workspace
        assert not sandbox.closed
    assert sandbox.closed and not sandbox.root.exists()


async def test_a_backend_that_cannot_build_its_executor_fails_setup_with_its_message(copy_registered, tmp_path):
    toml = f"""[capabilities.runner]
name = "copy"
workspace = "missing"
[capabilities.runner.options]
root = "{tmp_path}"
"""
    with pytest.raises(ConfigError, match="The copy sandbox needs an existing workspace"):
        async with configured(tmp_path, [], toml):
            pytest.fail("setup must fail before the session opens")
    assert CopySandbox.instances == []


class OutsideBackend(CopyBackend):
    """Mounts a folder that does not contain the working directory."""

    name = "outside"

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        assert isinstance(options, CopyOptions)
        return CopySandbox(options.root, options.root)


async def test_a_mount_that_misses_the_working_directory_fails_setup_plainly(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "entry_points", lambda group: [EntryPoint("outside", OutsideBackend)])
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    toml = f"""[capabilities.runner]
name = "outside"
[capabilities.runner.options]
root = "{elsewhere}"
"""
    with pytest.raises(ConfigError, match=r"^The outside runner's workspace must contain the session working dir"):
        async with configured(tmp_path, [], toml):
            pytest.fail("setup must fail before the session opens")


class RecordingExecutor:
    """A non-isolated runner (an ssh host, say): commands go elsewhere, files stay with the host tools."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.closed = False

    async def run(self, command: str, **kwargs) -> ExecResult:
        self.commands.append(command)
        return await LocalExecutor().run(command, **kwargs)

    async def aclose(self) -> None:
        self.closed = True


class RecordingBackend(Runner):
    name = "recording"
    isolated = False
    made: ClassVar[list[RecordingExecutor]] = []

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        executor = RecordingExecutor()
        RecordingBackend.made.append(executor)
        return executor


async def test_a_non_isolated_package_backend_still_runs_the_commands(monkeypatch, tmp_path):
    RecordingBackend.made.clear()
    monkeypatch.setattr(registry, "entry_points", lambda group: [EntryPoint("recording", RecordingBackend)])
    toml = """[capabilities]
tool_packs = ["base"]
[capabilities.runner]
name = "recording"
"""
    replies = [Reply(tool_calls=[("bash", {"command": "echo ran-there"})]), Reply(text="done")]
    async with configured(tmp_path, replies, toml) as (session, _, _, _):
        (executor,) = RecordingBackend.made
        assert session.deps.executor is executor
        assert type(session.deps.tools[0]).__module__ == "intern.tools.files"  # host file tools stay
        await session.run("run it")
        assert executor.commands == ["echo ran-there"] and "ran-there" in outputs(session)[0]
    assert executor.closed


async def test_a_backend_this_install_lacks_loads_but_fails_the_session_plainly(tmp_path):
    settings = CapabilitySettings.model_validate({"runner": {"name": "e2b", "options": {"template": "x"}}})
    assert settings.runner.name == "e2b"  # `models --json` and the rest of the config still load
    toml = """[capabilities.runner]
name = "e2b"
[capabilities.runner.options]
template = "base"
"""
    with pytest.raises(ConfigError, match="Unknown runner 'e2b'"):
        async with configured(tmp_path, [], toml):
            pytest.fail("setup must fail before the session opens")


def test_a_mount_maps_paths_whether_or_not_its_host_is_resolved(tmp_path):
    (tmp_path / "real" / "pkg").mkdir(parents=True)
    (tmp_path / "link").symlink_to(tmp_path / "real")
    mount = WorkspaceMount(tmp_path / "link", PurePosixPath("/sandbox"))
    assert mount.to_inner(tmp_path / "real" / "pkg") == PurePosixPath("/sandbox/pkg")
    assert mount.to_inner(tmp_path / "link" / "pkg") == PurePosixPath("/sandbox/pkg")
