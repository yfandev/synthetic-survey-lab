"""The OpenSandbox backend against a stand-in SDK that runs commands in a folder playing the sandbox, so the
upload, the copy back, background jobs, deadlines and cleanup all run for real; and live against a server when one
is configured."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import intern.runners.opensandbox as opensandbox
from intern.capability_settings import CapabilitySettings
from intern.models.registry import ConfigError
from intern.runners import load_runner
from intern.runners.opensandbox import OpenSandboxExecutor, OpenSandboxOptions, OpenSandboxRunner

from .fake_openai import Reply
from .test_session_setup import configured, outputs


def opts(**kw: Any) -> SimpleNamespace:
    return SimpleNamespace(**({"background": False, "working_directory": None, "timeout": None, "envs": None} | kw))


class FakeCommands:
    def __init__(self) -> None:
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self.logs: dict[str, bytearray] = {}
        self.interrupted: list[str] = []
        self.ran: list[Any] = []

    async def _spawn(self, command: Any, options: SimpleNamespace) -> asyncio.subprocess.Process:
        argv = command if isinstance(command, list) else ["bash", "-c", command]
        return await asyncio.create_subprocess_exec(
            *argv,
            cwd=options.working_directory,
            env={**os.environ, **(options.envs or {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )

    async def run(self, command: Any, *, opts: SimpleNamespace | None = None, handlers: Any = None) -> Any:
        # The real server (execd v1.1.0) takes shell text only: an argv list is a 400.
        assert isinstance(command, str), "OpenSandbox's execd takes shell text, not an argv list"
        options = opts or globals()["opts"]()
        self.ran.append(command)
        execution_id = uuid.uuid4().hex
        process = await self._spawn(command, options)
        self.processes[execution_id] = process
        assert process.stdout is not None
        if options.background:
            log = self.logs[execution_id] = bytearray()

            async def pump() -> None:
                assert process.stdout is not None
                while chunk := await process.stdout.read(4096):
                    log.extend(chunk)

            asyncio.get_running_loop().create_task(pump())
            return SimpleNamespace(id=execution_id)
        if handlers is not None:
            await handlers.on_init(SimpleNamespace(id=execution_id))
        collected = []
        while chunk := await process.stdout.read(4096):
            message = SimpleNamespace(text=chunk.decode())
            collected.append(message)
            if handlers is not None:
                await handlers.on_stdout(message)
        code = await process.wait()
        return SimpleNamespace(id=execution_id, exit_code=code, error=None, logs=SimpleNamespace(stdout=collected))

    async def interrupt(self, execution_id: str) -> None:
        self.interrupted.append(execution_id)
        process = self.processes[execution_id]
        if process.returncode is None:
            os.killpg(process.pid, signal.SIGKILL)

    async def get_command_status(self, execution_id: str) -> Any:
        process = self.processes[execution_id]
        return SimpleNamespace(running=process.returncode is None, exit_code=process.returncode)

    async def get_background_command_logs(self, execution_id: str, cursor: int | None = None) -> Any:
        await asyncio.sleep(0)  # let the pump and the process's exit be seen
        with_exit = self.processes[execution_id]
        with __import__("contextlib").suppress(TimeoutError):
            await asyncio.wait_for(asyncio.shield(with_exit.wait()), timeout=0.01)
        log = self.logs[execution_id]
        start = cursor or 0
        return SimpleNamespace(content=log[start:].decode(), cursor=len(log))


class FakeFiles:
    async def write_file(self, path: str, data: bytes) -> None:
        await asyncio.to_thread(Path(path).write_bytes, data)

    async def read_bytes(self, path: str) -> bytes:
        return await asyncio.to_thread(Path(path).read_bytes)


class FakeVault:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}

    async def create(self, *, credentials: list[Any], bindings: list[Any]) -> Any:
        self.state = {"credentials": credentials, "bindings": bindings}
        return SimpleNamespace(revision=1)


class FakeSandbox:
    made: list[FakeSandbox] = []  # noqa: RUF012 - a test registry

    def __init__(self, image: str, **kw: Any) -> None:
        self.image, self.kw = image, kw
        self.commands, self.files = FakeCommands(), FakeFiles()
        self.credential_vault = FakeVault()
        self.killed = self.closed = False

    @classmethod
    async def create(cls, image: str, **kw: Any) -> FakeSandbox:
        cls.made.append(cls(image, **kw))
        return cls.made[-1]

    async def kill(self) -> None:
        self.killed = True
        for process in self.commands.processes.values():
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)

    async def close(self) -> None:
        self.closed = True


SDK = SimpleNamespace(
    Sandbox=FakeSandbox,
    ConnectionConfig=lambda **kw: SimpleNamespace(**kw),
    RunCommandOpts=opts,
    ExecutionHandlers=lambda **kw: SimpleNamespace(**kw),
    Volume=lambda **kw: SimpleNamespace(**kw),
    Host=lambda **kw: SimpleNamespace(**kw),
    **{
        name: (lambda **kw: SimpleNamespace(**kw))
        for name in (
            "NetworkPolicy",
            "NetworkRule",
            "CredentialProxyConfig",
            "Credential",
            "InlineCredentialSource",
            "CredentialBinding",
            "CredentialMatch",
            "CredentialAuth",
        )
    },
)


@pytest.fixture
def sdk(monkeypatch):
    FakeSandbox.made.clear()
    monkeypatch.setattr(opensandbox, "_sdk", lambda: SDK)
    monkeypatch.setattr(opensandbox.OpenSandboxJob, "poll_s", 0.02)
    return SDK


def test_it_is_registered_and_its_options_are_strict():
    assert isinstance(load_runner("opensandbox"), OpenSandboxRunner)
    settings = CapabilitySettings.model_validate(
        {"runner": {"name": "opensandbox", "options": {"domain": "sandbox.internal:8080"}}}
    )
    assert isinstance(settings.runner.options, OpenSandboxOptions)
    for bad in ({"domain": "http://x"}, {"workspace_mode": "rsync"}, {"timeout_s": 5}, {"mystery": 1}):
        with pytest.raises(ValidationError):
            CapabilitySettings.model_validate({"runner": {"name": "opensandbox", "options": bad}})
    # The process tool can run there.
    process = CapabilitySettings.model_validate({"tool_packs": ["process"], "runner": {"name": "opensandbox"}})
    assert process.tool_packs == ["process"]


def test_setup_names_a_missing_key_or_sdk(monkeypatch, tmp_path):
    backend = OpenSandboxRunner()
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"needs \$OPEN_SANDBOX_API_KEY"):
        backend.create(OpenSandboxOptions(), workspace=tmp_path, cwd=tmp_path)
    monkeypatch.setattr(opensandbox, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="uv sync --extra opensandbox"):
        backend.create(OpenSandboxOptions(api_key_env=""), workspace=tmp_path, cwd=tmp_path)


def sandbox_toml(root: Path, *packs: str, mode: str = "copy") -> str:
    (root / "remote").mkdir(exist_ok=True)
    (root / "scratch").mkdir(exist_ok=True)
    return f"""[capabilities]
tool_packs = {json.dumps(["base", *packs])}
[capabilities.runner]
name = "opensandbox"
[capabilities.runner.options]
api_key_env = ""
workspace_mode = "{mode}"
sandbox_path = "{(root / "remote" / "workspace").resolve()}"
scratch_path = "{(root / "scratch").resolve()}"
image = "test-image"
"""


async def test_a_session_runs_in_the_sandbox_and_its_changes_come_back(sdk, tmp_path):
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "before.txt").write_text("from the host")
    replies = [
        Reply(tool_calls=[("bash", {"command": "pwd; cat before.txt; echo made > made.txt"})]),
        Reply(tool_calls=[("write", {"path": "notes.txt", "content": "written there"})]),
        Reply(text="done"),
    ]
    async with configured(tmp_path, replies, sandbox_toml(tmp_path)) as (session, _, _, work):
        assert FakeSandbox.made == []  # nothing starts at setup
        await session.run("work in the sandbox")
        (sandbox,) = FakeSandbox.made
        bash = outputs(session)[0]
        assert str((tmp_path / "remote" / "workspace").resolve()) in bash and "from the host" in bash
        assert (work / "made.txt").read_text().strip() == "made"  # copied back after the command
        assert (work / "notes.txt").read_text() == "written there"  # the file tools go through the sandbox too
        assert sandbox.image == "test-image" and sandbox.kw["volumes"] is None
        assert not sandbox.killed
    assert sandbox.killed and sandbox.closed


async def test_background_jobs_run_in_the_sandbox_and_stop_with_it(sdk, tmp_path):
    replies = [
        Reply(tool_calls=[("process", {"action": "start", "command": "echo serving; sleep 30"})]),
        Reply(text="started"),
    ]
    async with configured(tmp_path, replies, sandbox_toml(tmp_path, "process")) as (session, model, _, _):
        await session.run("start it")
        started = json.loads(outputs(session)[0])
        await asyncio.sleep(0.5)  # its first line arrives
        poll = ("process", {"action": "poll", "handle": started["handle"]})
        model.replies.extend(
            [
                # With other work in the turn: a turn that only polls a running job hands over.
                Reply(tool_calls=[poll, ("process", {"action": "write", "handle": started["handle"], "data": "x"})]),
                Reply(tool_calls=[("process", {"action": "terminate", "handle": started["handle"]})]),
                Reply(text="ok"),
            ]
        )
        await session.run("poll, feed and stop it")
        polled, written, ended = outputs(session)[1:4]
        assert "serving" in json.loads(polled)["output"]
        assert "take no input" in written
        assert json.loads(ended)["state"] == "terminated"
        (sandbox,) = FakeSandbox.made
        assert sandbox.commands.interrupted  # stopped on the server


async def test_a_command_past_its_deadline_is_interrupted(sdk, tmp_path):
    (tmp_path / "remote").mkdir()
    executor = OpenSandboxExecutor(
        tmp_path,
        OpenSandboxOptions(
            api_key_env="",
            sandbox_path=str((tmp_path / "remote" / "workspace").resolve()),
            scratch_path=str(tmp_path.resolve()),
        ),
    )
    result = await executor.run("echo begun; sleep 10", cwd=tmp_path, timeout_s=0.5, max_output_bytes=4096)
    assert result.timed_out and result.exit_code is None
    (sandbox,) = FakeSandbox.made
    assert sandbox.commands.interrupted
    abort = asyncio.Event()
    asyncio.get_running_loop().call_later(0.2, abort.set)
    stopped = await executor.run("sleep 10", cwd=tmp_path, timeout_s=30, max_output_bytes=4096, abort=abort)
    assert stopped.aborted and not stopped.timed_out
    await executor.aclose()
    assert sandbox.killed


async def test_mount_mode_binds_the_host_folder_and_uploads_nothing(sdk, tmp_path):
    executor = OpenSandboxExecutor(tmp_path, OpenSandboxOptions(api_key_env="", workspace_mode="mount"))
    sandbox = await executor._sandbox_ready()  # pyright: ignore[reportPrivateUsage]
    (volume,) = sandbox.kw["volumes"]
    assert volume.host.path == str(tmp_path.resolve()) and volume.mount_path == "/workspace"
    assert sandbox.commands.ran == []
    await executor.aclose()


async def test_a_workspace_too_large_to_upload_fails_plainly(sdk, tmp_path):
    (tmp_path / "big.bin").write_bytes(b"x" * (2 << 20))
    executor = OpenSandboxExecutor(tmp_path, OpenSandboxOptions(api_key_env="", max_upload_mb=1))
    with pytest.raises(ValueError, match="over max_upload_mb"):
        await executor.run("true", cwd=tmp_path, timeout_s=5, max_output_bytes=1024)
    assert FakeSandbox.made[0].killed  # a sandbox that could not be set up is not left running


live = pytest.mark.skipif(
    not os.environ.get("INTERN_OPENSANDBOX_LIVE"),
    reason="set INTERN_OPENSANDBOX_LIVE=1 with OPEN_SANDBOX_DOMAIN and OPEN_SANDBOX_API_KEY for a running server",
)


@live
async def test_live_session_on_a_real_server(tmp_path):
    image = os.environ.get("INTERN_OPENSANDBOX_IMAGE", "python:3.12-slim")
    toml = f"""[capabilities]
tool_packs = ["base", "process"]
[capabilities.runner]
name = "opensandbox"
[capabilities.runner.options]
image = {json.dumps(image)}
"""
    replies = [
        Reply(tool_calls=[("bash", {"command": "echo live > live.txt; cat live.txt; pwd"})]),
        Reply(tool_calls=[("read", {"path": "live.txt"})]),
        Reply(tool_calls=[("process", {"action": "start", "command": "echo job-ran; sleep 2"})]),
        Reply(text="done"),
    ]
    env = {k: os.environ[k] for k in ("OPEN_SANDBOX_DOMAIN", "OPEN_SANDBOX_API_KEY") if k in os.environ}
    async with configured(tmp_path, replies, toml, env=env) as (session, _, _, work):
        await session.run("use the sandbox")
        assert "/workspace" in outputs(session)[0]
        assert "live" in outputs(session)[1]
        assert (work / "live.txt").read_text().strip() == "live"


def test_setup_refuses_a_server_key_that_is_not_set(monkeypatch, tmp_path):
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY", raising=False)

    async def open_it() -> None:
        async with configured(tmp_path, [], '[capabilities.runner]\nname = "opensandbox"\n'):
            pytest.fail("setup must fail before the session opens")

    with pytest.raises(ConfigError, match="OPEN_SANDBOX_API_KEY"):
        asyncio.run(open_it())


def copy_executor(tmp_path: Path) -> OpenSandboxExecutor:
    (tmp_path / "remote").mkdir(exist_ok=True)
    (tmp_path / "work").mkdir(exist_ok=True)
    return OpenSandboxExecutor(
        tmp_path / "work",
        OpenSandboxOptions(
            api_key_env="",
            sandbox_path=str((tmp_path / "remote" / "workspace").resolve()),
            scratch_path=str(tmp_path.resolve()),
        ),
    )


async def run(executor: OpenSandboxExecutor, command: str) -> str:
    result = await executor.run(command, cwd=executor.workspace, timeout_s=20, max_output_bytes=8192)
    assert result.exit_code == 0, result.output
    return result.output


async def test_files_the_harness_writes_reach_the_sandbox_before_each_command(sdk, tmp_path):
    executor = copy_executor(tmp_path)
    await run(executor, "true")  # the upload is done: later host edits are not sent, except the harness's own
    own = executor.workspace / ".intern-resources" / "web" / "page.txt"
    own.parent.mkdir(parents=True)
    own.write_text("retained source")
    (executor.workspace / "user-edit.txt").write_text("not sent")
    shown = await run(executor, "cat .intern-resources/web/page.txt; ls")
    assert "retained source" in shown and "user-edit.txt" not in shown
    await executor.aclose()


async def test_renamed_and_timestamp_preserving_files_come_back(sdk, tmp_path):
    executor = copy_executor(tmp_path)
    (executor.workspace / "a.txt").write_text("moved")
    await run(executor, "true")
    # A rename keeps the old mtime; a file written with an old mtime (as tar -x and cp -p do) is older than the mark.
    await run(
        executor,
        "mv a.txt b.txt && python3 -c \"import os; open('old.txt','w').write('kept'); os.utime('old.txt', (0, 0))\"",
    )
    assert (executor.workspace / "b.txt").read_text() == "moved"
    assert (executor.workspace / "old.txt").read_text() == "kept"
    await executor.aclose()


async def test_dependency_folders_are_not_uploaded(sdk, tmp_path):
    executor = copy_executor(tmp_path)
    for folder in (".venv", "node_modules", "pkg/__pycache__"):
        (executor.workspace / folder).mkdir(parents=True)
        (executor.workspace / folder / "x").write_text("host build")
    (executor.workspace / "pkg" / "code.py").write_text("print()")
    listed = await run(executor, "find . -type f | sort")
    assert "./pkg/code.py" in listed and "x" not in listed.replace("./pkg/code.py", "")
    await executor.aclose()


async def test_a_failed_final_copy_back_is_reported_and_the_sandbox_still_killed(sdk, tmp_path):
    executor = copy_executor(tmp_path)
    await run(executor, "true")
    (sandbox,) = FakeSandbox.made

    async def broken(path: str) -> bytes:
        raise OSError("server went away")

    sandbox.files.read_bytes = broken
    with pytest.raises(OSError, match="server went away"):
        await executor.aclose()
    assert sandbox.killed and sandbox.closed


def test_its_toggle_says_why_it_cannot_switch_on(monkeypatch):
    from intern.capability_settings import RunnerConfig
    from intern.session_setup import runner_problem

    monkeypatch.delenv("OPEN_SANDBOX_API_KEY", raising=False)
    reason = runner_problem("opensandbox", RunnerConfig()) or ""
    assert "OPEN_SANDBOX_API_KEY" in reason
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "set")
    assert runner_problem("opensandbox", RunnerConfig()) is None


def live_options(**kw: Any) -> OpenSandboxOptions:
    return OpenSandboxOptions.model_validate(
        {"image": os.environ.get("INTERN_OPENSANDBOX_IMAGE", "python:3.12-slim"), "timeout_s": 600, **kw}
    )


@live
async def test_live_copy_mode_syncs_both_ways_and_bounds_commands(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text("from the host")
    (work / ".venv").mkdir()
    (work / ".venv" / "big").write_text("host build")
    executor = OpenSandboxExecutor(work, live_options())
    try:
        first = await executor.run("cat a.txt; ls -a; pwd", cwd=work, timeout_s=60, max_output_bytes=8192)
        assert first.exit_code == 0 and "from the host" in first.output
        assert ".venv" not in first.output and "/workspace" in first.output
        # Renames and mtime-preserving writes come back.
        moved = await executor.run(
            "mv a.txt b.txt && python3 -c "
            "\"import os; open('old.txt','w').write('kept'); os.utime('old.txt', (0, 0))\"",
            cwd=work,
            timeout_s=60,
            max_output_bytes=8192,
        )
        assert moved.exit_code == 0, moved.output
        assert (work / "b.txt").read_text() == "from the host" and (work / "old.txt").read_text() == "kept"
        # What the harness writes into .intern-resources/ is sent before the next command.
        (work / ".intern-resources" / "web").mkdir(parents=True)
        (work / ".intern-resources" / "web" / "page.txt").write_text("retained")
        shown = await executor.run("cat .intern-resources/web/page.txt", cwd=work, timeout_s=60, max_output_bytes=8192)
        assert shown.output.strip() == "retained"
        # The exit code, the deadline and abort.
        failed = await executor.run("exit 3", cwd=work, timeout_s=60, max_output_bytes=8192)
        assert failed.exit_code == 3
        slow = await executor.run("sleep 30", cwd=work, timeout_s=2, max_output_bytes=8192)
        assert slow.timed_out and slow.exit_code is None and slow.duration_ms < 15_000
        abort = asyncio.Event()
        asyncio.get_running_loop().call_later(1, abort.set)
        stopped = await executor.run("sleep 30", cwd=work, timeout_s=60, max_output_bytes=8192, abort=abort)
        assert stopped.aborted
        # Background jobs: output by cursor, the exit code, stopping one, and no stdin.
        job = await executor.start_job("echo job-ran; exit 4", cwd=work)
        out = b""
        async with asyncio.timeout(30):
            while chunk := await job.read():
                out += chunk
        assert b"job-ran" in out and await job.wait() == 4
        long = await executor.start_job("sleep 60", cwd=work)
        with pytest.raises(NotImplementedError):
            await long.write(b"x")
        await long.kill()
        async with asyncio.timeout(30):
            await long.wait()
    finally:
        await executor.aclose()
    assert executor._sandbox is None  # pyright: ignore[reportPrivateUsage]


@live
async def test_live_mount_mode_shares_the_host_folder(tmp_path):
    """Needs a server on this machine whose [storage] allowed_host_paths include pytest's temporary folders."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "shared.txt").write_text("one copy")
    executor = OpenSandboxExecutor(work, live_options(workspace_mode="mount"))
    try:
        result = await executor.run(
            "cat shared.txt; echo made > made.txt", cwd=work, timeout_s=60, max_output_bytes=4096
        )
        assert result.exit_code == 0 and "one copy" in result.output, result.output
        assert (work / "made.txt").read_text().strip() == "made"
    finally:
        await executor.aclose()


GUARDED = {
    "egress": [{"action": "allow", "target": "*.github.com"}, {"action": "allow", "target": "pypi.org"}],
    "credentials": {"gh": {"env": "GH_TOKEN_FOR_TEST", "hosts": ["api.github.com"]}},
}


def test_egress_rules_and_credentials_are_checked_when_the_config_loads():
    settings = CapabilitySettings.model_validate({"runner": {"name": "opensandbox", "options": GUARDED}})
    options = settings.runner.options
    assert isinstance(options, OpenSandboxOptions) and options.egress_default == "deny"
    bad = [
        {"egress": [{"action": "allow"}]},  # neither target nor ports
        {"egress": [{"action": "allow", "target": "http://x"}]},
        {"credentials": {"k": {"env": "K", "hosts": ["x.com"], "auth": "api_key"}}},  # api_key needs a header
        {"credentials": {"k": {"env": "K", "hosts": ["x.com"], "header": "X-Key"}}},  # only api_key takes one
        # A credential for a host the rules block could never be used.
        {"egress": [{"action": "allow", "target": "pypi.org"}], "credentials": {"k": {"env": "K", "hosts": ["x.com"]}}},
    ]
    for options in bad:
        with pytest.raises(ValidationError):
            CapabilitySettings.model_validate({"runner": {"name": "opensandbox", "options": options}})


def test_a_missing_credential_variable_makes_the_runner_unavailable(monkeypatch):
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "set")
    monkeypatch.delenv("GH_TOKEN_FOR_TEST", raising=False)
    options = OpenSandboxOptions.model_validate(GUARDED)
    assert OpenSandboxRunner().problem(options) == "OpenSandbox credentials need $GH_TOKEN_FOR_TEST"
    monkeypatch.setenv("GH_TOKEN_FOR_TEST", "ghp-synthetic")
    assert OpenSandboxRunner().problem(options) is None


async def test_the_sandbox_gets_the_rules_and_the_vault_gets_the_secret_from_this_machine(sdk, monkeypatch, tmp_path):
    monkeypatch.setenv("GH_TOKEN_FOR_TEST", "ghp-synthetic-value")
    (tmp_path / "remote").mkdir()
    executor = OpenSandboxExecutor(
        tmp_path,
        OpenSandboxOptions.model_validate(
            GUARDED
            | {
                "api_key_env": "",
                "sandbox_path": str((tmp_path / "remote" / "workspace").resolve()),
                "scratch_path": str(tmp_path.resolve()),
            }
        ),
    )
    result = await executor.run("env", cwd=tmp_path, timeout_s=10, max_output_bytes=8192)
    (sandbox,) = FakeSandbox.made
    policy = sandbox.kw["network_policy"]
    assert policy.default_action == "deny"
    assert [(r.action, r.target) for r in policy.egress] == [("allow", "*.github.com"), ("allow", "pypi.org")]
    assert sandbox.kw["credential_proxy"].enabled is True
    (credential,) = sandbox.credential_vault.state["credentials"]
    (binding,) = sandbox.credential_vault.state["bindings"]
    assert credential.name == "gh" and credential.source.value == "ghp-synthetic-value"
    assert binding.match.hosts == ["api.github.com"] and binding.auth.type == "bearer"
    # The value goes only to the vault: no command the backend sends carries it (the live test checks the
    # sandbox's environment; this stand-in runs on the host, whose environment the test set).
    assert result.exit_code == 0
    assert all("ghp-synthetic-value" not in str(command) for command in sandbox.commands.ran)
    await executor.aclose()


async def test_no_rules_or_credentials_leaves_the_network_as_the_server_has_it(sdk, tmp_path):
    executor = OpenSandboxExecutor(tmp_path, OpenSandboxOptions(api_key_env="", workspace_mode="mount"))
    sandbox = await executor._sandbox_ready()  # pyright: ignore[reportPrivateUsage]
    assert sandbox.kw["network_policy"] is None and sandbox.kw["credential_proxy"] is None
    assert sandbox.credential_vault.state == {}
    await executor.aclose()


live_network = pytest.mark.skipif(
    not os.environ.get("INTERN_OPENSANDBOX_LIVE_NETWORK"),
    reason="needs a live server with [egress] mode = dns+nft and internet access (example.com, httpbin.org)",
)


@live
@live_network
async def test_live_rules_block_other_hosts_and_the_vault_signs_requests_without_the_sandbox_seeing_it(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("INTERN_LIVE_VAULT_TOKEN", "vault-only-value")
    options = live_options(
        egress=[{"action": "allow", "target": "example.com"}, {"action": "allow", "target": "httpbin.org"}],
        credentials={"echo": {"env": "INTERN_LIVE_VAULT_TOKEN", "hosts": ["httpbin.org"]}},
    )
    probe = (
        "import json, urllib.request\n"
        "def get(url):\n"
        "    try:\n"
        "        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 harness-test'})\n"
        "        return urllib.request.urlopen(req, timeout=15).read().decode()\n"
        "    except Exception as e:\n"
        "        return 'BLOCKED ' + type(e).__name__\n"
        "print('example', get('https://example.com')[:40].replace(chr(10), ' '))\n"
        "print('github', get('https://github.com')[:40])\n"
        "print('auth', json.loads(get('https://httpbin.org/headers'))['headers'].get('Authorization'))\n"
    )
    executor = OpenSandboxExecutor(tmp_path, options)
    try:
        (tmp_path / "probe.py").write_text(probe)
        result = await executor.run("python3 probe.py; env", cwd=tmp_path, timeout_s=90, max_output_bytes=16384)
        assert "github BLOCKED" in result.output and "example BLOCKED" not in result.output, result.output
        assert "auth Bearer vault-only-value" in result.output
        assert result.output.count("vault-only-value") == 1  # in the echoed header only, never in `env`
    finally:
        await executor.aclose()
