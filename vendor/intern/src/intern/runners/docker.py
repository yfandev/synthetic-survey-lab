"""The `docker` runner: a fresh locked-down container per command, with one explicit workspace mount."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import signal
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from intern.executor import (
    CapturingExecutor,
    ExecResult,
    Executor,
    LocalExecutor,
    WorkspaceMount,
    run_command,
)
from intern.runners.base import Runner

MOUNT = PurePosixPath("/workspace")


class DockerOptions(BaseModel):
    image: str = Field(default="python:3.12.11-slim-bookworm", pattern=r"^[A-Za-z0-9][A-Za-z0-9_./:@-]*$")
    network: Literal["none", "bridge"] = "none"
    memory_mb: int = Field(default=512, ge=32)
    cpus: float = Field(default=1, gt=0)
    pids_limit: int = Field(default=128, ge=8)


class DockerConfig(DockerOptions):
    """`[capabilities.runner.options]` for `name = "docker"`: an unknown or non-finite option fails at load."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    cpus: float = Field(default=1, gt=0, allow_inf_nan=False)


class DockerExecutor:
    """Run commands in fresh containers; only workspace files persist between calls.

    No image pulls occur during execution. Prepare the image explicitly and use a
    digest in production. Docker's daemon is part of the trust boundary.
    """

    def __init__(
        self, workspace: Path, options: DockerOptions | None = None, *, launcher: Executor | None = None
    ) -> None:
        self.workspace = workspace.resolve()
        if "," in str(self.workspace) or "\n" in str(self.workspace):
            raise ValueError("Docker mount paths cannot contain commas or newlines")
        if not self.workspace.is_dir():
            raise ValueError("Docker workspace must be an existing directory")
        self.options = (options or DockerOptions()).model_copy(deep=True)
        self.launcher = launcher or LocalExecutor()

    @property
    def mount(self) -> WorkspaceMount:
        """The workspace is bind-mounted at `/workspace` in every container."""
        return WorkspaceMount(self.workspace, MOUNT)

    @property
    def output_capture(self) -> bool:
        return isinstance(self.launcher, CapturingExecutor) and self.launcher.output_capture

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_s: float,
        max_output_bytes: int,
        abort: asyncio.Event | None = None,
        env: Mapping[str, str] | None = None,
        capture_path: Path | None = None,
    ) -> ExecResult:
        relative = (await asyncio.to_thread(cwd.resolve)).relative_to(self.workspace)
        name = "harness-" + uuid.uuid4().hex
        argv = self._argv(name, relative, command, env, interactive=False)
        launch_env = dict(os.environ)
        launch_env.update(env or {})
        try:
            return await run_command(
                self.launcher,
                shlex.join(argv),
                cwd=self.workspace,
                timeout_s=timeout_s,
                max_output_bytes=max_output_bytes,
                abort=abort,
                env=launch_env,
                capture_path=capture_path,
            )
        finally:
            # Killing the docker client does not prove the container stopped.
            # Remove only this invocation's uniquely named owned container.
            cleanup = asyncio.create_task(
                self.launcher.run(
                    shlex.join(["docker", "rm", "-f", name]),
                    cwd=self.workspace,
                    timeout_s=10,
                    max_output_bytes=1024,
                    env=launch_env,
                )
            )
            await asyncio.shield(cleanup)

    def _argv(
        self, name: str, relative: Path, command: str, env: Mapping[str, str] | None, *, interactive: bool
    ) -> list[str]:
        """`docker run` for one command in its own uniquely named container; `interactive` keeps its stdin open."""
        if env is not None and any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) for key in env):
            raise ValueError("Invalid environment variable name")
        options = self.options
        argv = [
            "docker",
            "run",
            "--rm",
            *(["--interactive"] if interactive else []),
            "--pull",
            "never",
            "--name",
            name,
            "--network",
            options.network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            str(options.pids_limit),
            "--memory",
            f"{options.memory_mb}m",
            "--cpus",
            str(options.cpus),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=64m",
            "--mount",
            f"type=bind,source={self.workspace},target={MOUNT}",
            "--workdir",
            str(MOUNT / relative),
        ]
        for key in env or {}:
            argv += ["--env", key]
        return [*argv, options.image, "/bin/bash", "-c", command]

    async def start_job(self, command: str, *, cwd: Path, env: Mapping[str, str] | None = None) -> DockerJob:
        """A background command in its own container, attached through a local `docker run --interactive` client:
        its output and stdin go through the client, and the container's limits bound its memory."""
        relative = (await asyncio.to_thread(cwd.resolve)).relative_to(self.workspace)
        name = "harness-" + uuid.uuid4().hex
        argv = self._argv(name, relative, command, env, interactive=True)
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self.workspace,
            env={**os.environ, **(env or {})},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        return DockerJob(process, name, cwd=self.workspace)


class DockerJob:
    """One container's command, seen through the local docker client that started it."""

    def __init__(self, process: asyncio.subprocess.Process, name: str, *, cwd: Path) -> None:
        self.process, self.name, self.cwd = process, name, cwd
        self._removed = False

    async def read(self) -> bytes:
        assert self.process.stdout is not None
        return await self.process.stdout.read(16_384)

    async def write(self, data: bytes, *, eof: bool = False) -> None:
        stdin = self.process.stdin
        if stdin is None or stdin.is_closing():
            raise ValueError("The process no longer accepts input")
        stdin.write(data)
        await asyncio.wait_for(stdin.drain(), timeout=5)
        if eof:
            stdin.close()

    async def wait(self) -> int | None:
        return await self.process.wait()

    async def kill(self) -> None:
        """Remove the container (stopping the docker client does not stop it), then the client."""
        if not self._removed:
            self._removed = True
            remover = await asyncio.create_subprocess_exec(
                "docker",
                "rm",
                "--force",
                self.name,
                cwd=self.cwd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(remover.wait(), timeout=10)
            except TimeoutError:
                remover.kill()
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(self.process.pid, signal.SIGKILL)
        await self.process.wait()


class DockerRunner(Runner):
    """A fresh locked-down container per command, with only the workspace mounted (see docker-workspace.md)."""

    name = "docker"
    title = "Docker"
    note = "bash, the file tools and background jobs run in a container with the working directory mounted"
    jobs = True
    Options = DockerConfig

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        try:
            return DockerExecutor(workspace, DockerOptions.model_validate(options.model_dump()))
        except (OSError, ValueError):
            raise ValueError("Docker needs an existing workspace containing the session working directory") from None
