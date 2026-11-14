"""Where shell commands run. v0 has the local process; a container sandbox and remote runners plug in
behind the same protocol."""

from __future__ import annotations

import asyncio
import contextlib
import os
import posixpath
import signal
import sys
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Protocol, runtime_checkable

from pydantic import BaseModel

READ_CHUNK = 64 * 1024
KILL_GRACE_S = 2.0
DRAIN_S = 1.0
EXIT_POLL_S = 0.02
#: How often a command's memory is read. A python loading a large file grows by a GB or more a second, so a slower
#: poll lets it overshoot its budget by that much.
MEMORY_POLL_S = 0.25
#: How often the memory of what commands left running in the background is read: they run for long, and one ps a
#: second for all of them keeps the watching cheap.
BACKGROUND_POLL_S = 1.0
MAX_CAPTURE_BYTES = 64 << 20  # Per-command disk bound, separate from the in-memory display excerpt.


class ExecResult(BaseModel):
    exit_code: int | None
    output: str
    omitted_bytes: int
    timed_out: bool
    aborted: bool
    duration_ms: int
    capture_omitted_bytes: int = 0
    capture_complete: bool = True
    #: The command's process group went over the executor's memory budget and was stopped.
    memory_exceeded: bool = False
    #: The most resident memory the group was seen holding, in bytes (0 when not watched).
    memory_bytes: int = 0
    #: The command's process group, when the command returned and left members of it running (`cmd &`).
    left_running_group: int | None = None


@dataclass(frozen=True)
class WorkspaceMount:
    """A host folder and the path the shell and file tools see it at. Both hold the same files: a bind mount, or a
    remote copy the executor keeps in sync."""

    host: Path
    inner: PurePosixPath

    def to_inner(self, path: Path) -> PurePosixPath:
        """`path` (on the host) as the tools see it; ValueError when it is outside the mounted folder."""
        host = PurePosixPath(posixpath.normpath(self.host.resolve()))
        relative = PurePosixPath(posixpath.normpath(path.resolve())).relative_to(host)
        return self.inner / relative


def mount_of(executor: object) -> WorkspaceMount | None:
    """The executor's workspace mount, or None when commands run on the host as they are (the local executor).
    A wrapper such as SecretExecutor passes on the mount of the executor it wraps."""
    mount = getattr(executor, "mount", None)
    return mount if isinstance(mount, WorkspaceMount) else None


@runtime_checkable
class ClosableExecutor(Protocol):
    """An executor that owns a service, such as a remote sandbox: the session closes it when the session closes."""

    async def aclose(self) -> None: ...


class Job(Protocol):
    """A background command an executor started: what the `process` tool polls, feeds and stops."""

    async def read(self) -> bytes:
        """The next output, stdout and stderr together; `b""` once all of it has been read."""
        ...

    async def write(self, data: bytes, *, eof: bool = False) -> None:
        """Send `data` to its stdin, then close stdin when `eof`. Raise NotImplementedError where jobs take no
        input."""
        ...

    async def wait(self) -> int | None:
        """Its exit code once it has ended (None when the executor cannot say)."""
        ...

    async def kill(self) -> None:
        """Stop it and what it started. Safe to call again, and after it ended."""
        ...


@runtime_checkable
class JobExecutor(Protocol):
    """An executor that runs background commands where its other commands run, so the `process` tool can use it.
    The local executor is not one: the process manager runs local jobs itself, in owned process groups."""

    async def start_job(self, command: str, *, cwd: Path, env: Mapping[str, str] | None = None) -> Job:
        """Start `command` in bash at `cwd` (a host path, mapped as `run` maps it) and return at once."""
        ...


def _open_capture(path: Path) -> BinaryIO:
    # Unbuffered: what a command printed is in the file at once, for a command moved to the background.
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb", buffering=0)


class Executor(Protocol):
    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_s: float,
        max_output_bytes: int,
        abort: asyncio.Event | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ExecResult: ...


@runtime_checkable
class CapturingExecutor(Executor, Protocol):
    """Optional capture capability; existing Executor implementations need not implement it."""

    @property
    def output_capture(self) -> bool: ...

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
    ) -> ExecResult: ...


async def run_command(
    executor: Executor,
    command: str,
    *,
    cwd: Path,
    timeout_s: float,
    max_output_bytes: int,
    abort: asyncio.Event | None = None,
    env: Mapping[str, str] | None = None,
    capture_path: Path | None = None,
) -> ExecResult:
    if capture_path is not None:
        if not isinstance(executor, CapturingExecutor) or not executor.output_capture:
            raise ValueError("This executor does not support output capture")
        return await executor.run(
            command,
            cwd=cwd,
            timeout_s=timeout_s,
            max_output_bytes=max_output_bytes,
            abort=abort,
            env=env,
            capture_path=capture_path,
        )
    return await executor.run(
        command, cwd=cwd, timeout_s=timeout_s, max_output_bytes=max_output_bytes, abort=abort, env=env
    )


class _HeadTail:
    """Keeps the first and last part of a stream without holding all of it in memory."""

    def __init__(self, limit: int) -> None:
        self.head_limit = limit // 3
        self.tail_limit = limit - self.head_limit
        self.head = bytearray()
        self.tail: deque[bytes] = deque()
        self.tail_size = 0
        self.omitted = 0

    def add(self, chunk: bytes) -> None:
        room = self.head_limit - len(self.head)
        if room > 0:
            self.head += chunk[:room]
            chunk = chunk[room:]
        if not chunk:
            return
        self.tail.append(chunk)
        self.tail_size += len(chunk)
        while self.tail_size > self.tail_limit:
            excess = self.tail_size - self.tail_limit
            first = self.tail[0]
            if len(first) <= excess:
                self.tail.popleft()
                self.tail_size -= len(first)
                self.omitted += len(first)
            else:
                self.tail[0] = first[excess:]
                self.tail_size -= excess
                self.omitted += excess

    def text(self) -> str:
        head = self.head.decode(errors="replace")
        tail = b"".join(self.tail).decode(errors="replace")
        if not self.omitted:
            return head + tail
        return f"{head}\n[... {self.omitted} bytes omitted ...]\n{tail}"


def shell_env(
    environ: Mapping[str, str], prefix: str = sys.prefix, base_prefix: str = sys.base_prefix
) -> dict[str, str]:
    """The caller's environment minus Intern's own virtualenv.

    `uv run` and an activated venv put the harness's interpreter first on PATH, so the agent's `python3`
    would be the harness's, without the packages the user installed. Other venvs stay.
    """
    env = dict(environ)
    if prefix == base_prefix:
        return env
    own = Path(prefix).resolve()
    bin_dir = own / ("Scripts" if os.name == "nt" else "bin")
    path = [p for p in env.get("PATH", "").split(os.pathsep) if p and Path(p).resolve() != bin_dir]
    env["PATH"] = os.pathsep.join(path)
    if env.get("VIRTUAL_ENV") and Path(env["VIRTUAL_ENV"]).resolve() == own:
        env.pop("VIRTUAL_ENV")
        env.pop("VIRTUAL_ENV_PROMPT", None)
    return env


def physical_memory() -> int | None:
    """The machine's physical memory in bytes; None where the platform does not say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


async def memory_by_group() -> dict[int, int] | None:
    """Resident memory in bytes of every process group on the machine, by group id, from one ps (macOS and Linux
    alike: neither enforces a memory rlimit on a group, so it is watched). A group with no member is absent. None
    when it cannot be read: ps missing, not allowed to run (a sandbox) or failing."""
    try:
        ps = await asyncio.create_subprocess_exec(
            "ps", "-A", "-o", "pgid=,rss=", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        out, _ = await ps.communicate()
    except OSError:
        return None
    if ps.returncode != 0:
        return None
    groups: dict[int, int] = {}
    for line in out.decode(errors="replace").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            groups[int(fields[0])] = groups.get(int(fields[0]), 0) + int(fields[1]) * 1024  # ps reports KiB
    return groups


def memory_stopped(used: int, limit: int) -> str:
    """How a stop for memory is told: what it held and what a command may."""
    return f"stopped at {used / 2**30:.1f} GB of memory, over the {limit / 2**30:.1f} GB a command may hold"


def _kill_group(process: asyncio.subprocess.Process, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, sig)


async def _exited(process: asyncio.subprocess.Process) -> None:
    """Return once bash itself has exited. `Process.wait()` also waits for every child holding the
    output pipe, so a command that leaves a background process running would look like it hangs."""
    while process.returncode is None:  # noqa: ASYNC110 - asyncio exposes no exit event without pipe EOF
        await asyncio.sleep(EXIT_POLL_S)


async def stop_process_group(process: asyncio.subprocess.Process) -> None:
    """Stop a process started with `start_new_session=True` and everything in its group.

    Members get a grace period after SIGTERM, even if the leader exits or their pipes close first, then SIGKILL.
    """
    _kill_group(process, signal.SIGTERM)
    deadline = asyncio.get_running_loop().time() + KILL_GRACE_S
    try:
        while asyncio.get_running_loop().time() < deadline:
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            except PermissionError:
                # A zombie-only group can report EPERM on Darwin; still attempt the final kill.
                break
            await asyncio.sleep(EXIT_POLL_S)
    finally:
        _kill_group(process, signal.SIGKILL)


class LocalExecutor:
    """Runs commands with bash in their own process group, so a timeout kills the whole tree, and so does going over
    ``memory_limit`` bytes of resident memory (None: not watched).

    Background processes a command starts keep running after it returns, but their output is no
    longer read; they should write to a file.
    """

    output_capture = True

    def __init__(self, memory_limit: int | None = None) -> None:
        self.memory_limit = memory_limit
        # What finished commands left running (a `cmd &`), by process group, and what was said when one was stopped.
        self._background: dict[int, str] = {}
        self._notices: list[str] = []
        self._watcher: asyncio.Task[None] | None = None

    def take_notices(self) -> list[str]:
        """What happened to background processes since the last call: each one stopped for memory, once."""
        notices, self._notices = self._notices, []
        return notices

    def _keep_watching(self, pgid: int, command: str) -> None:
        """Go on watching the group a finished command left running, for as long as it has members."""
        self._background[pgid] = command
        if self._watcher is None or self._watcher.done():
            self._watcher = asyncio.create_task(self._watch_background())

    async def _watch_background(self) -> None:
        limit = self.memory_limit
        while self._background and limit:
            await asyncio.sleep(BACKGROUND_POLL_S)
            if (usage := await memory_by_group()) is None:
                self._background.clear()  # memory cannot be read here: nothing to watch with
                return
            for pgid, command in list(self._background.items()):
                used = usage.get(pgid, 0)
                if not used:
                    del self._background[pgid]
                elif used > limit:
                    del self._background[pgid]
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(pgid, signal.SIGKILL)
                    shown = command if len(command) <= 120 else command[:117] + "..."
                    self._notices.append(
                        f"A process the command `{shown}` left running in the background was "
                        f"{memory_stopped(used, limit)}."
                    )

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
        """Optionally spool output before excerpting; the caller owns the private capture file.

        A capture exceeding MAX_CAPTURE_BYTES is explicitly reported as incomplete.
        """
        started = time.monotonic()
        capture = _open_capture(capture_path) if capture_path is not None else None
        try:
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd,
                env={**shell_env(os.environ), **(env or {})},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
        except BaseException:
            if capture is not None:
                capture.close()
            raise
        buffer = _HeadTail(max_output_bytes)
        captured = capture_omitted = 0
        pump_failed: asyncio.Future[Exception] = asyncio.get_running_loop().create_future()

        async def pump() -> None:
            nonlocal captured, capture_omitted
            assert process.stdout is not None
            try:
                while chunk := await process.stdout.read(READ_CHUNK):
                    if capture is not None:
                        kept = chunk[: max(0, MAX_CAPTURE_BYTES - captured)]
                        capture.write(kept)
                        captured += len(kept)
                        capture_omitted += len(chunk) - len(kept)
                    buffer.add(chunk)
            except Exception as error:
                pump_failed.set_result(error)

        peak = 0

        async def watch(limit: int) -> None:
            """Return only when the group is over ``limit``: memory that cannot be read stops the watch, not the
            command (it waits until the command ends and the watch is cancelled)."""
            nonlocal peak
            while True:
                await asyncio.sleep(MEMORY_POLL_S)
                if (usage := await memory_by_group()) is None:
                    await asyncio.get_running_loop().create_future()
                used = usage.get(process.pid, 0) if usage is not None else 0
                peak = max(peak, used)
                if used > limit:
                    return

        reader = asyncio.create_task(pump())
        exited = asyncio.create_task(_exited(process))
        abort_task = asyncio.create_task(abort.wait()) if abort else None
        watcher = asyncio.create_task(watch(self.memory_limit)) if self.memory_limit else None
        timed_out = aborted = over_memory = False
        finished = False
        try:
            stoppers = [t for t in (exited, pump_failed, abort_task, watcher) if t is not None]
            done, _ = await asyncio.wait(stoppers, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if pump_failed in done:
                raise pump_failed.result()
            timed_out = not done
            aborted = abort_task is not None and abort_task in done and exited not in done
            over_memory = watcher is not None and watcher in done and exited not in done
            if timed_out or aborted or over_memory:
                await stop_process_group(process)
                await exited
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(reader), DRAIN_S)
            finished = True
        finally:
            capture_complete = reader.done() and capture_omitted == 0
            pump_failed.cancel()
            # Also reached on cancellation: never leave the command running behind a dead task.
            if not finished or process.returncode is None:
                _kill_group(process, signal.SIGKILL)
            for task in (reader, exited, abort_task, watcher):
                if task is not None:
                    task.cancel()
            transport = process._transport  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
            transport.close()  # pyright: ignore[reportUnknownMemberType]
            if capture is not None:
                capture.close()
        if pump_failed.done() and not pump_failed.cancelled():
            raise pump_failed.result()
        left_running = None
        if not (timed_out or aborted or over_memory):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(process.pid, 0)  # raises when the group is empty: nothing left running
                left_running = process.pid
                if self.memory_limit:
                    self._keep_watching(process.pid, command)
        return ExecResult(
            exit_code=None if timed_out or aborted or over_memory else process.returncode,
            output=buffer.text(),
            omitted_bytes=buffer.omitted,
            timed_out=timed_out,
            aborted=aborted,
            duration_ms=int((time.monotonic() - started) * 1000),
            capture_omitted_bytes=capture_omitted,
            capture_complete=capture_complete,
            memory_exceeded=over_memory,
            memory_bytes=peak,
            left_running_group=left_running,
        )
