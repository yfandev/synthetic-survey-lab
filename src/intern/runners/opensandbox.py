"""OpenSandbox (github.com/opensandbox-group/OpenSandbox) as an runner: a session's shell commands, file
tools and background jobs run in one sandbox on an OpenSandbox server, on your Docker or Kubernetes.

Needs the `opensandbox` extra. The sandbox starts on the session's first command and is killed when the session
closes. The workspace reaches it one of two ways (`workspace_mode`):

- `copy` (default): uploaded on first use; from then on the sandbox is the source of truth, and what commands
  change is copied back after each one and at close. Host edits made after the upload are not seen there, except
  the files the harness itself writes (`.intern-resources/`), which are sent before each command. Files the sandbox
  deletes stay on the host.
- `mount`: the host folder is bind-mounted, for a server on this machine whose `allowed_host_paths` include it.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import os
import re
import shlex
import tarfile
import time
from collections.abc import Mapping
from datetime import timedelta
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intern import names
from intern.executor import ExecResult, Executor, WorkspaceMount, _HeadTail  # pyright: ignore[reportPrivateUsage]
from intern.runners.base import Runner

EXTRA_HINT = "OpenSandbox needs uv sync --extra opensandbox (or intern-agent[opensandbox])"
SKIPPED = frozenset({".venv", "node_modules", "__pycache__"})
"""Not uploaded: built for the host, not the sandbox's image, and often most of a workspace's size."""
OWN = names.RESOURCES_DIR
"""Files the harness itself writes into the workspace (the web reader's retained sources): sent before each command."""
LOG_POLL_S = 0.5


HOST = r"^(\*\.)?[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*$"
TARGET = r"^((\*\.)?[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*|[0-9a-fA-F:.]+(/[0-9]{1,3})?)$"


class EgressRule(BaseModel):
    """One network rule, matched in order: a domain (`example.com`, `*.example.com`), an IP or a CIDR."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    action: Literal["allow", "deny"]
    target: str | None = Field(default=None, pattern=TARGET, max_length=253)
    ports: list[int] | None = Field(default=None, min_length=1, max_length=256)
    """TCP ports; with no target the rule applies to every address on them. A domain target takes no ports."""

    @model_validator(mode="after")
    def target_or_ports(self) -> EgressRule:
        if self.target is None and not self.ports:
            raise ValueError("an egress rule needs a target, ports, or both")
        if self.ports and any(not 0 < port < 65536 for port in self.ports):
            raise ValueError("ports are 1-65535")
        return self


class VaultCredential(BaseModel):
    """A secret the sandbox uses without seeing it: the server's proxy adds it to requests to `hosts`."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    """The variable on this machine holding the value; the value is never written in config.toml."""
    hosts: list[str] = Field(min_length=1, max_length=64)
    """Exact hosts or `*.example.com`, reached over https."""
    auth: Literal["bearer", "basic", "api_key"] = "bearer"
    """`bearer`: `Authorization: Bearer <value>`; `basic`: the value is `user:password`; `api_key`: the value in
    `header`."""
    header: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]+$")

    @model_validator(mode="after")
    def header_for_api_key(self) -> VaultCredential:
        if any(not re.fullmatch(HOST, host) for host in self.hosts):
            raise ValueError("credential hosts are exact host names or *.example.com")
        if (self.auth == "api_key") != (self.header is not None):
            raise ValueError("header names the api_key header, and only api_key takes one")
        return self


class OpenSandboxOptions(BaseModel):
    """`[capabilities.runner.options]` for `name = "opensandbox"`."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    domain: str | None = Field(default=None, pattern=r"^[A-Za-z0-9.-]+(:[0-9]{1,5})?$")
    """The server, `host:port`; unset is `$OPEN_SANDBOX_DOMAIN`, else the SDK's default (`localhost:8080`)."""
    protocol: Literal["http", "https"] = "http"
    api_key_env: str = Field(default="OPEN_SANDBOX_API_KEY", pattern=r"^([A-Za-z_][A-Za-z0-9_]*)?$")
    """The environment variable holding the server's API key; empty for a server without one."""
    image: str = Field(default="python:3.12-slim", pattern=r"^[A-Za-z0-9][A-Za-z0-9_./:@-]*$")
    """Needs bash, tar and Python 3 (the file tools run Python in the sandbox)."""
    workspace_mode: Literal["copy", "mount"] = "copy"
    sandbox_path: str = Field(default="/workspace", pattern=r"^/[^\x00\n]*$")
    """Where the workspace is in the sandbox."""
    scratch_path: str = Field(default="/tmp", pattern=r"^/[^\x00\n]*$")
    """Where in the sandbox, outside the workspace, the backend keeps its sync archives and marks."""
    timeout_s: int = Field(default=3600, ge=60, le=86_400)
    """How long the server keeps the sandbox alive."""
    cpu: str | None = Field(default=None, pattern=r"^[0-9.]+m?$")
    memory: str | None = Field(default=None, pattern=r"^[0-9]+(Ki|Mi|Gi)?$")
    max_upload_mb: int = Field(default=256, ge=1, le=4096)
    """The most workspace `copy` uploads at start."""
    request_timeout_s: float = Field(default=60, gt=0, le=600)
    egress: list[EgressRule] | None = Field(default=None, max_length=256)
    """Where the sandbox may connect, rules matched in order; unset leaves the server's network as it is. Needs the
    server's egress sidecar ([egress] in its config, Docker `bridge` networking)."""
    egress_default: Literal["allow", "deny"] = "deny"
    """What happens to a connection no rule matches, when `egress` is set."""
    credentials: dict[str, VaultCredential] = Field(default_factory=dict, max_length=32)
    """Secrets by name, held in the sandbox's credential vault. Needs the server's [egress] mode = "dns+nft"."""

    @model_validator(mode="after")
    def credentials_reachable(self) -> OpenSandboxOptions:
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name) for name in self.credentials):
            raise ValueError("credential names are letters, digits, _ . -")
        if self.egress is not None and self.egress_default == "deny":
            allowed = [rule.target for rule in self.egress if rule.action == "allow" and rule.target and not rule.ports]
            for name, credential in self.credentials.items():
                for host in credential.hosts:
                    if not any(_covers(target, host) for target in allowed):
                        raise ValueError(f"credential {name!r} is for {host}, which no egress rule allows")
        return self


def _covers(target: str, host: str) -> bool:
    """Whether an allow rule's domain target reaches `host`."""
    if target.startswith("*."):
        return host == target[2:] or host.endswith(target[1:]) or host == target
    return host == target


def _sdk() -> Any:
    """The OpenSandbox SDK, imported on first use so the harness loads without it."""
    import opensandbox  # pyright: ignore[reportMissingImports]
    from opensandbox.config import ConnectionConfig  # pyright: ignore[reportMissingImports]
    from opensandbox.models import execd, sandboxes  # pyright: ignore[reportMissingImports]

    return type(
        "OpenSandboxSDK",
        (),
        {
            "Sandbox": opensandbox.Sandbox,
            "ConnectionConfig": ConnectionConfig,
            "RunCommandOpts": execd.RunCommandOpts,
            "ExecutionHandlers": execd.ExecutionHandlers,
            "Volume": sandboxes.Volume,
            "Host": sandboxes.Host,
            "NetworkPolicy": sandboxes.NetworkPolicy,
            "NetworkRule": sandboxes.NetworkRule,
            "CredentialProxyConfig": sandboxes.CredentialProxyConfig,
            "Credential": sandboxes.Credential,
            "InlineCredentialSource": sandboxes.InlineCredentialSource,
            "CredentialBinding": sandboxes.CredentialBinding,
            "CredentialMatch": sandboxes.CredentialMatch,
            "CredentialAuth": sandboxes.CredentialAuth,
        },
    )


class OpenSandboxRunner(Runner):
    name = "opensandbox"
    title = "OpenSandbox"
    note = "bash, the file tools and background jobs run in an OpenSandbox sandbox on your own server"
    Options = OpenSandboxOptions
    jobs = True
    isolated = True

    def problem(self, options: BaseModel) -> str | None:
        assert isinstance(options, OpenSandboxOptions)
        if find_spec("opensandbox") is None:
            return EXTRA_HINT
        if options.api_key_env and not os.environ.get(options.api_key_env):
            return f'OpenSandbox needs ${options.api_key_env}; a server without a key takes api_key_env = ""'
        missing = sorted({c.env for c in options.credentials.values() if not os.environ.get(c.env)})
        if missing:
            return "OpenSandbox credentials need " + ", ".join(f"${name}" for name in missing)
        return None

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        assert isinstance(options, OpenSandboxOptions)
        if (problem := self.problem(options)) is not None:
            raise ValueError(problem)
        if not workspace.is_dir():
            raise ValueError("OpenSandbox needs an existing workspace containing the session working directory")
        return OpenSandboxExecutor(workspace, options)


class OpenSandboxExecutor:
    """One sandbox per session, started on first use."""

    def __init__(self, workspace: Path, options: OpenSandboxOptions, *, sdk: Any = None) -> None:
        self.workspace = workspace.resolve()
        self.options = options.model_copy(deep=True)
        self._sdk = sdk
        self._sandbox: Any = None
        self._starting = asyncio.Lock()
        self._syncing = asyncio.Lock()
        self._closed = False
        self._own_sent = 0
        """The newest `.intern-resources/` mtime (ns) already in the sandbox."""

    @property
    def mount(self) -> WorkspaceMount:
        return WorkspaceMount(self.workspace, PurePosixPath(self.options.sandbox_path))

    def _inner(self, cwd: Path) -> str:
        return str(self.mount.to_inner(cwd))

    async def _sandbox_ready(self) -> Any:
        async with self._starting:
            if self._closed:
                raise RuntimeError("The OpenSandbox sandbox is closed")
            if self._sandbox is not None:
                return self._sandbox
            sdk = self._sdk = self._sdk or _sdk()
            options = self.options
            key = os.environ.get(options.api_key_env) if options.api_key_env else None
            connection = sdk.ConnectionConfig(
                domain=options.domain,
                protocol=options.protocol,
                api_key=key,
                request_timeout=timedelta(seconds=options.request_timeout_s),
            )
            resource = {k: v for k, v in (("cpu", options.cpu), ("memory", options.memory)) if v is not None}
            volumes = (
                [sdk.Volume(name="workspace", host=sdk.Host(path=str(self.workspace)), mount_path=options.sandbox_path)]
                if options.workspace_mode == "mount"
                else None
            )
            policy = (
                sdk.NetworkPolicy(
                    default_action=options.egress_default,
                    egress=[sdk.NetworkRule(**rule.model_dump(exclude_none=True)) for rule in options.egress],
                )
                if options.egress is not None
                else None
            )
            sandbox = await sdk.Sandbox.create(
                options.image,
                timeout=timedelta(seconds=options.timeout_s),
                resource=resource or None,
                volumes=volumes,
                network_policy=policy,
                credential_proxy=sdk.CredentialProxyConfig(enabled=True) if options.credentials else None,
                metadata={"created-by": "intern"},
                connection_config=connection,
            )
            try:
                if options.credentials:
                    await self._fill_vault(sandbox)
                if options.workspace_mode == "copy":
                    await self._upload(sandbox)
            except BaseException:
                with contextlib.suppress(Exception):
                    await sandbox.kill()
                raise
            self._sandbox = sandbox
            return sandbox

    async def _fill_vault(self, sandbox: Any) -> None:
        """The configured secrets, read from this machine's environment now, into the sandbox's vault. Only the
        server's proxy holds them; nothing in the sandbox can read them back."""
        sdk = self._sdk
        kinds = {"bearer": "bearer", "basic": "basic", "api_key": "apiKey"}
        credentials, bindings = [], []
        for name, credential in self.options.credentials.items():
            value = os.environ.get(credential.env)
            if not value:
                raise ValueError(f"OpenSandbox credential {name!r} needs ${credential.env}")
            credentials.append(sdk.Credential(name=name, source=sdk.InlineCredentialSource(value=value)))
            bindings.append(
                sdk.CredentialBinding(
                    name=name,
                    match=sdk.CredentialMatch(hosts=list(credential.hosts), schemes=["https"]),
                    auth=sdk.CredentialAuth(type=kinds[credential.auth], credential=name, name=credential.header),
                )
            )
        await sandbox.credential_vault.create(credentials=credentials, bindings=bindings)

    async def _upload(self, sandbox: Any) -> None:
        """The workspace as one tar, unpacked in the sandbox; then the sync mark, so only later changes come back."""
        archive = await asyncio.to_thread(self._pack)
        target, upload = shlex.quote(self.options.sandbox_path), self._scratch("harness-workspace.tar")
        await sandbox.files.write_file(upload, archive)
        await self._must(
            sandbox,
            f"mkdir -p {target} && tar -xf {shlex.quote(upload)} -C {target} && rm {shlex.quote(upload)} "
            f"&& touch {shlex.quote(self._scratch('.intern-synced'))}",
        )

    def _scratch(self, name: str) -> str:
        return str(PurePosixPath(self.options.scratch_path) / name)

    def _pack(self) -> bytes:
        buffer = io.BytesIO()
        limit = self.options.max_upload_mb << 20
        total = 0
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for path in sorted(self.workspace.rglob("*")):
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or SKIPPED.intersection(path.relative_to(self.workspace).parts)
                ):
                    continue
                total += path.stat().st_size
                if total > limit:
                    raise ValueError(f"The workspace is over max_upload_mb ({self.options.max_upload_mb} MB)")
                archive.add(path, arcname=str(path.relative_to(self.workspace)), recursive=False)
        return buffer.getvalue()

    async def _sync_back(self, sandbox: Any) -> None:
        """Copy back the files changed since the last sync (`copy` mode). Deletions are not copied back."""
        if self.options.workspace_mode != "copy":
            return
        async with self._syncing:
            target = shlex.quote(self.options.sandbox_path)
            mark, after = shlex.quote(self._scratch(".intern-synced")), shlex.quote(self._scratch(".intern-sync-next"))
            changes = self._scratch("harness-changed.tar")
            # The next mark is set before listing, so a change made while this runs is sent again next time,
            # never lost.
            changed = await self._must(
                sandbox,
                f"touch {after} && cd {target} && find . -type f \\( -newer {mark} -o -cnewer {mark} \\) -print0 "
                f"| tar --null -cf {shlex.quote(changes)} -T - && mv {after} {mark} && echo ok",
            )
            if "ok" not in changed:
                raise RuntimeError("OpenSandbox could not list the workspace's changes")
            archive = await sandbox.files.read_bytes(changes)
            await asyncio.to_thread(self._unpack, archive)

    async def _send_own(self, sandbox: Any) -> None:
        """`copy` mode: the sandbox is the source of truth, so host files reach it only here, and only the ones the
        harness itself writes into the workspace (`.intern-resources/`, the web reader's retained sources)."""
        if self.options.workspace_mode != "copy":
            return
        archive = await asyncio.to_thread(self._pack_own)
        if archive is None:
            return
        upload = self._scratch("harness-own.tar")
        await sandbox.files.write_file(upload, archive)
        target = shlex.quote(self.options.sandbox_path)
        await self._must(sandbox, f"tar -xf {shlex.quote(upload)} -C {target} && rm {shlex.quote(upload)}")

    def _pack_own(self) -> bytes | None:
        root = self.workspace / OWN
        files = sorted(root.rglob("*")) if root.is_dir() else []
        changed = [p for p in files if p.is_file() and not p.is_symlink() and p.stat().st_mtime_ns > self._own_sent]
        if not changed:
            return None
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as archive:
            for path in changed:
                archive.add(path, arcname=str(path.relative_to(self.workspace)), recursive=False)
        self._own_sent = max(path.stat().st_mtime_ns for path in changed)
        return buffer.getvalue()

    def _unpack(self, archive: bytes) -> None:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r") as tar:
            tar.extractall(self.workspace, filter="data")

    async def _must(self, sandbox: Any, command: str) -> str:
        """A setup command that must succeed; its stdout."""
        sdk = self._sdk
        execution = await sandbox.commands.run(
            command, opts=sdk.RunCommandOpts(timeout=timedelta(seconds=self.options.request_timeout_s))
        )
        code = execution.exit_code if execution.exit_code is not None else getattr(execution.error, "exit_code", None)
        if code not in (0, None) or execution.error is not None:
            raise RuntimeError("An OpenSandbox setup command failed")
        return "".join(message.text for message in execution.logs.stdout)

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
        started = time.monotonic()
        sandbox = await self._sandbox_ready()
        await self._send_own(sandbox)
        sdk = self._sdk
        buffer = _HeadTail(max_output_bytes)
        execution_id: list[str] = []

        async def output(message: Any) -> None:
            buffer.add(message.text.encode())

        async def init(event: Any) -> None:
            execution_id.append(event.id)

        handlers = sdk.ExecutionHandlers(on_stdout=output, on_stderr=output, on_init=init, skip_accumulation=True)
        opts = sdk.RunCommandOpts(
            working_directory=self._inner(cwd),
            timeout=timedelta(seconds=timeout_s + 5),  # the server's backstop; the deadline below is the one
            envs=dict(env) if env else None,
        )
        running = asyncio.create_task(
            sandbox.commands.run(f"bash -c {shlex.quote(command)}", opts=opts, handlers=handlers)
        )
        stoppers: set[asyncio.Future[Any]] = {running}
        aborted_task = asyncio.create_task(abort.wait()) if abort is not None else None
        if aborted_task is not None:
            stoppers.add(aborted_task)
        timed_out = aborted = False
        try:
            done, _ = await asyncio.wait(stoppers, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if running not in done:
                timed_out = not done
                aborted = not timed_out
                if execution_id:
                    with contextlib.suppress(Exception):
                        await sandbox.commands.interrupt(execution_id[0])
                running.cancel()
                with contextlib.suppress(BaseException):
                    await running
                exit_code = None
            else:
                execution = running.result()
                exit_code = execution.exit_code
                if exit_code is None and execution.error is not None:
                    exit_code = getattr(execution.error, "exit_code", None) or 1
                    buffer.add(f"\n{execution.error.name}: {execution.error.value}".encode())
        finally:
            if aborted_task is not None:
                aborted_task.cancel()
            if not running.done():
                running.cancel()
        await self._sync_back(sandbox)
        return ExecResult(
            exit_code=exit_code,
            output=buffer.text(),
            omitted_bytes=buffer.omitted,
            timed_out=timed_out,
            aborted=aborted,
            duration_ms=int((time.monotonic() - started) * 1000),
        )

    async def start_job(self, command: str, *, cwd: Path, env: Mapping[str, str] | None = None) -> OpenSandboxJob:
        sandbox = await self._sandbox_ready()
        await self._send_own(sandbox)
        opts = self._sdk.RunCommandOpts(
            background=True, working_directory=self._inner(cwd), envs=dict(env) if env else None
        )
        execution = await sandbox.commands.run(f"bash -c {shlex.quote(command)}", opts=opts)
        if execution.id is None:
            raise RuntimeError("OpenSandbox started no background command")
        return OpenSandboxJob(self, sandbox, execution.id)

    async def aclose(self) -> None:
        """Copy the last changes back, then kill the sandbox."""
        async with self._starting:
            self._closed = True
            sandbox, self._sandbox = self._sandbox, None
        if sandbox is None:
            return
        try:
            # Not suppressed: a lost copy-back loses the session's work, and session close reports it.
            await self._sync_back(sandbox)
        finally:
            try:
                await sandbox.kill()
            finally:
                await sandbox.close()


class OpenSandboxJob:
    """A background command: its logs read by cursor, its status polled, interrupted to stop. Takes no input."""

    poll_s: ClassVar[float] = LOG_POLL_S

    def __init__(self, executor: OpenSandboxExecutor, sandbox: Any, execution_id: str) -> None:
        self.executor, self.sandbox, self.id = executor, sandbox, execution_id
        self._cursor: int | None = None
        self._ended = False
        self._exit: int | None = None
        self._stopped = False

    async def _status(self) -> bool:
        """Whether it still runs; records its exit code once it does not."""
        if self._ended:
            return False
        status = await self.sandbox.commands.get_command_status(self.id)
        if status.running is False:
            self._ended, self._exit = True, status.exit_code
        return not self._ended

    async def read(self) -> bytes:
        while True:
            logs = await self.sandbox.commands.get_background_command_logs(self.id, cursor=self._cursor)
            self._cursor = logs.cursor if logs.cursor is not None else self._cursor
            if logs.content:
                return logs.content.encode()
            if not await self._status():
                # One last read after it ended, then the stream is over.
                logs = await self.sandbox.commands.get_background_command_logs(self.id, cursor=self._cursor)
                self._cursor = logs.cursor if logs.cursor is not None else self._cursor
                return logs.content.encode() if logs.content else b""
            await asyncio.sleep(self.poll_s)

    async def write(self, data: bytes, *, eof: bool = False) -> None:
        raise NotImplementedError

    async def wait(self) -> int | None:
        while await self._status():  # noqa: ASYNC110 - the server's status is polled; there is no event to wait on
            await asyncio.sleep(self.poll_s)
        with contextlib.suppress(Exception):
            await self.executor._sync_back(self.sandbox)  # pyright: ignore[reportPrivateUsage]
        return self._exit

    async def kill(self) -> None:
        if not self._stopped and not self._ended:
            self._stopped = True
            with contextlib.suppress(Exception):
                await self.sandbox.commands.interrupt(self.id)
