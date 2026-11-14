"""Owned background processes with bounded output and explicit cleanup: local process groups, or jobs an executor
runs where its other commands run (a container, a sandbox).

Each session has one manager (session setup builds it whenever its executor runs background jobs), which the
`process` tool starts jobs in and which also keeps a record of every other command that runs in the session: a
`bash` call while it runs, and a process group a finished command left running (`cmd &`). Every start and end is
reported to the session as a `ProcessEvent`, so what is running at any moment is known in one place.

Handles belong to a live manager, not to a PID. They are deliberately not
reattached after a harness restart: process identity and stdin cannot be recovered
safely from a bare PID. Use a durable remote job backend for that guarantee.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import uuid
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from intern.events import ProcessEvent, ProcessInfo, ProcessKind, ProcessState, now_ms
from intern.executor import (
    BACKGROUND_POLL_S,
    ExecResult,
    Job,
    JobExecutor,
    memory_by_group,
    memory_stopped,
    shell_env,
)

COMMAND_CHARS = 200  # a command as records and snapshots show it
LAST_LINE_CHARS = 200  # a running job's last line of output, as System 1 reads it
SYSTEM1_PROCESSES = 8  # the running commands System 1's view lists, oldest first
CAPTURE_POLL_S = 0.25  # how often a moved command's capture file is read for what it printed since
JUDGE_AFTER_S = 2.0  # a `bash` command still running after this long is one System 1 is asked about
Judgment = tuple[Literal["waits", "serves"], float]
"""System 1's word that a running `bash` command waits or serves, with P(waits) + P(serves)."""
ENDED_LINES = ("Background job ", "A process the command ")  # how the line of each process in an end notice begins
NOTICE_OUTPUT_CHARS = 1_500  # the end of a job's output the notice of its end quotes


class OutputFilter(Protocol):
    """Rewrites a job's output as it arrives (redaction): what `feed` returns is logged, then `flush` at the end."""

    def feed(self, chunk: bytes) -> bytes: ...

    def flush(self) -> bytes: ...


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _shown(command: str) -> str:
    return _cut(command, COMMAND_CHARS)


def duration_text(seconds: int) -> str:
    """`42s`, `4m 05s`, `1h 02m`."""
    hours, rest = divmod(max(seconds, 0), 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


def _from_before(info: ProcessInfo) -> str:
    """What became of a job of a reopened session's, from its journal: no job outlives the session that ran it."""
    if info.state == "running":  # its end was never recorded: the session closed without it
        fate = "it was still running when the session closed"
    else:  # `terminated` after the agent's last turn: closing stopped it
        took = duration_text(((info.ended_at or info.started_at) - info.started_at) // 1000)
        code = f", exit code {info.exit_code}" if info.exit_code is not None else ""
        fate = f"it ended: {info.state.replace('_', ' ')}{code}, after {took}"
    return (
        f"Background job {info.id} (`{info.command}`) is from before this session was reopened, and no job outlives "
        f"its session: {fate}. Its output is gone; start it again if you still need it."
    )


class ProcessSnapshot(BaseModel):
    handle: str
    command: str = Field(default="", description="The command, cut to COMMAND_CHARS.")
    state: ProcessState
    exit_code: int | None
    elapsed_s: int = Field(default=0, description="Seconds from its start to now, or to its end.")
    output: str
    cursor: int = Field(description="Absolute byte offset of the first returned output byte.")
    next_cursor: int
    omitted_bytes: int


class RunningProcess(BaseModel):
    """A command running in the session, as System 1 reads it (`session_cut`'s `processes`)."""

    id: str
    kind: ProcessKind
    command: str
    elapsed_s: int
    quiet_s: int | None = Field(
        default=None, description="Seconds since it last wrote output; None for what is not read as it runs."
    )
    last_output: str | None = Field(default=None, description="Its last line of output, cut to LAST_LINE_CHARS.")


@dataclass
class _Job:
    handle: str
    command: str = ""
    started_at: int = field(default_factory=now_ms)
    ended_at: int | None = None
    last_output_at: int | None = None
    process: asyncio.subprocess.Process | None = None
    """A local job's owned process group; None for a job an executor runs."""
    remote: Job | None = None
    """A job an executor runs where its other commands run (a container, a sandbox)."""
    remote_exit: int | None = None
    """The exit code of a job an executor ran, or of a `bash` call moved here."""
    moved: asyncio.Task[ExecResult] | None = None
    """A `bash` call that went on as this job: its execution, which ends it when it returns."""
    capture: Path | None = None
    """The moved call's capture file, read as it grows; the job removes it when it ends."""
    held: bool = False
    """A moved call whose result the run that moved it waits for (not one that serves, which does not end)."""
    note: str | None = None
    """Why it went on in the background, for a moved `bash` call (`ProcessInfo.note`)."""
    lift: Callable[[], None] | None = None
    """Frees a moved `bash` call of its timeout_s, once it is found to serve."""
    state: ProcessState = "running"
    output: bytearray = field(default_factory=bytearray)
    base: int = 0
    done: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task[None] | None = None
    kill_sent: bool = False
    end_seen: bool = False
    """The agent was shown that it ended: a poll, a list or its terminate after the end, or the notice of it."""
    output_filter: OutputFilter | None = None
    takes_input: bool = False
    """It was started with its input open (`stdin`), for `write`; any other job reads end-of-file."""

    @property
    def returncode(self) -> int | None:
        return self.process.returncode if self.process is not None else self.remote_exit


class ProcessManager:
    def __init__(
        self,
        *,
        max_running: int = 4,
        max_records: int = 64,
        max_log_bytes: int = 64 * 1024,
        move_after_s: float = 0,
    ) -> None:
        if max_running < 1 or max_records < max_running or max_log_bytes < 1 or move_after_s < 0:
            raise ValueError("Invalid process capacity or output budget")
        self.max_running, self.max_records, self.max_log_bytes = max_running, max_records, max_log_bytes
        self.move_after_s = move_after_s
        """A `bash` call still running after this many seconds goes on as a job; 0 never moves one."""
        self.judge: Callable[[str], Coroutine[object, object, Judgment | None]] | None = None
        """Asks System 1 whether a command still running after JUDGE_AFTER_S waits or serves rather than works, so it
        moves now (the session sets it)."""
        self._moves: dict[str, asyncio.Event] = {}
        self._move_why: dict[str, str] = {}
        """Why a move was requested, by call id."""
        """A request to move each running `bash` call (System 1's check found it stuck), by call id."""
        self._jobs: dict[str, _Job] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._sink: Callable[[ProcessEvent], None] | None = None
        self._calls: dict[str, ProcessInfo] = {}
        """`bash` calls that are running, by call id."""
        self._groups: dict[int, ProcessInfo] = {}
        """Process groups finished commands left running, by group id, watched until they are empty."""
        self._watcher: asyncio.Task[None] | None = None
        self._past: dict[str, ProcessInfo] = {}
        """The session's jobs from before it was reopened (its journal's last record of each), by handle."""
        self._past_told: set[str] = set()

    def remember(self, past: Sequence[ProcessInfo]) -> None:
        """The jobs a reopened session's journal records: a handle of one is explained, not unknown, and an end the
        agent was not told is told with `end_notice`."""
        self._past = {info.id: info for info in past}

    def listen(self, sink: Callable[[ProcessEvent], None]) -> None:
        """Report every start and end to `sink` (the session's event sink) from now on."""
        self._sink = sink

    def _report(self, action: Literal["started", "ended"], info: ProcessInfo) -> None:
        if self._sink is not None:
            self._sink(ProcessEvent(action=action, process=info))

    @staticmethod
    def _info(job: _Job) -> ProcessInfo:
        return ProcessInfo(
            id=job.handle,
            kind="job",
            command=_shown(job.command),
            state=job.state,
            exit_code=job.returncode,
            started_at=job.started_at,
            ended_at=job.ended_at,
            last_output_at=job.last_output_at,
            note=job.note,
        )

    def _job_ended(self, job: _Job) -> None:
        job.ended_at = now_ms()
        self._report("ended", self._info(job))

    def running(self) -> list[ProcessInfo]:
        """Every command running in the session now, oldest first: jobs, `bash` calls and what commands left
        running."""
        jobs = [self._info(job) for job in self._jobs.values() if not job.done.is_set()]
        return sorted([*jobs, *self._calls.values(), *self._groups.values()], key=lambda info: info.started_at)

    def for_system1(self) -> list[RunningProcess]:
        """What runs now as System 1 reads it: the oldest SYSTEM1_PROCESSES, each with how long it has run and, for a
        job, how long it has been quiet and its last line of output."""
        now, shown = now_ms(), []
        for info in self.running()[:SYSTEM1_PROCESSES]:
            quiet = last = None
            if (job := self._jobs.get(info.id)) is not None and info.kind == "job":
                quiet = (now - (job.last_output_at or job.started_at)) // 1000
                lines = bytes(job.output[-4 * LAST_LINE_CHARS :]).decode("utf-8", errors="replace").strip().splitlines()
                last = _cut(lines[-1].strip(), LAST_LINE_CHARS) if lines else None
            elapsed = (now - info.started_at) // 1000
            shown.append(
                RunningProcess(
                    id=info.id, kind=info.kind, command=info.command, elapsed_s=elapsed, quiet_s=quiet, last_output=last
                )
            )
        return shown

    def begin_call(self, call_id: str, command: str) -> None:
        """A `bash` call started: it is running until `end_call` (or `adopt`)."""
        info = ProcessInfo(id=call_id, kind="call", command=_shown(command), state="running", started_at=now_ms())
        self._calls[call_id] = info
        self._moves[call_id] = asyncio.Event()
        self._report("started", info)

    def end_call(self, call_id: str, state: ProcessState, exit_code: int | None) -> None:
        self._moves.pop(call_id, None)
        self._move_why.pop(call_id, None)
        if (info := self._calls.pop(call_id, None)) is None:
            return
        self._report("ended", info.model_copy(update={"state": state, "exit_code": exit_code, "ended_at": now_ms()}))

    def move_requested(self, call_id: str) -> asyncio.Event:
        """Set when the running `bash` call `call_id` should go on in the background now."""
        return self._moves.setdefault(call_id, asyncio.Event())

    def request_move(self, call_id: str, why: str = "System 1 found it stuck") -> None:
        """Ask the running `bash` call `call_id` to go on in the background now: System 1 found it stuck, or a message
        from the user came in (`move_for_user`)."""
        if call_id in self._calls and self.move_after_s > 0:
            self._move_why.setdefault(call_id, why)
            self.move_requested(call_id).set()

    def move_for_user(self) -> None:
        """The user sent a message for the agent: every `bash` call still running goes on in the background, so the
        agent reads it at its next turn instead of after the command."""
        for call_id in list(self._calls):
            self.request_move(call_id, "a message from the user came in")

    def move_why(self, call_id: str) -> str:
        return self._move_why.get(call_id, "System 1 found it stuck")

    def adopt(
        self,
        call_id: str,
        command: str,
        execution: asyncio.Task[ExecResult],
        capture: Path | None,
        *,
        held: bool = True,
        note: str | None = None,
        lift: Callable[[], None] | None = None,
    ) -> ProcessSnapshot:
        """Go on with the running `bash` call `call_id` as a job: `execution` is its run, left running, and `capture`
        the file it writes its output to, if any. The job keeps the call's start, so its age counts from there, and
        its deadline, which `execution` holds, unless it serves: one no run waits for (not `held`) is freed of it with
        `lift`. Not counted against `max_running`: it was already running. `note` says why it moved, on the call's end
        and on the job."""
        call = self._calls.pop(call_id, None)
        self._moves.pop(call_id, None)
        self._move_why.pop(call_id, None)
        started = call.started_at if call is not None else now_ms()
        if call is not None:
            self._report("ended", call.model_copy(update={"state": "moved", "ended_at": now_ms(), "note": note}))
        job = _Job(
            handle=uuid.uuid4().hex,
            command=command,
            started_at=started,
            moved=execution,
            capture=capture,
            held=held,
            note=note,
            lift=lift,
        )
        if not held and lift is not None:
            lift()
        self._jobs[job.handle] = job
        self._report("started", self._info(job))
        job.task = asyncio.create_task(self._supervise_moved(job))
        return self._snapshot(job, None)

    def held_running(self) -> list[ProcessInfo]:
        """The `bash` calls moved here that still run and whose result the run waits for."""
        return [self._info(job) for job in self._jobs.values() if job.held and not job.done.is_set()]

    def release(self, handle: str) -> None:
        """No run waits for this moved call's result any more: it serves (System 1 found it up and ready), so it is
        freed of its call's timeout_s too."""
        if (job := self._jobs.get(handle)) is not None:
            job.held = False
            if job.lift is not None:
                job.lift()

    async def _read_capture(self, job: _Job, offset: int) -> int:
        """Append what the moved call's capture file holds past `offset`; returns the new offset."""
        if job.capture is None:
            return offset

        def read() -> bytes:
            try:
                with job.capture.open("rb") as stream:  # type: ignore[union-attr]
                    stream.seek(offset)
                    return stream.read()
            except OSError:
                return b""

        chunk = await asyncio.to_thread(read)
        if chunk:
            self._receive(job, chunk)
        return offset + len(chunk)

    async def _supervise_moved(self, job: _Job) -> None:
        """A moved `bash` call: read its capture as it grows until its execution returns, then end the job as the
        command ended, with what it printed and what it left running."""
        assert job.moved is not None
        offset = 0
        try:
            while not job.moved.done():
                offset = await self._read_capture(job, offset)
                await asyncio.wait({job.moved}, timeout=CAPTURE_POLL_S)
            result = None if job.moved.cancelled() or job.moved.exception() is not None else job.moved.result()
            if job.capture is not None:
                offset = await self._read_capture(job, offset)
            elif result is not None:
                self._receive(job, result.output.encode())
            if result is not None:
                job.remote_exit = result.exit_code
                if job.state == "running":
                    job.state = (
                        "out_of_memory"
                        if result.memory_exceeded
                        else "timed_out"
                        if result.timed_out
                        else "aborted"
                        if result.aborted
                        else "exited"
                    )
                if result.left_running_group is not None:
                    self.left_running(result.left_running_group, job.command)
            elif job.state == "running":
                job.state = "terminated"
        finally:
            self._end_of_output(job)
            if job.capture is not None:
                job.capture.unlink(missing_ok=True)
            job.done.set()
            self._job_ended(job)

    def left_running(self, group: int, command: str) -> None:
        """A finished command left process group `group` running (`cmd &`): watch it until it has no members.
        Nothing here stops it; the session does not own it."""
        if self._closed or group in self._groups or len(self._groups) >= self.max_records:
            return
        info = ProcessInfo(
            id=f"group-{group}", kind="left_running", command=_shown(command), state="running", started_at=now_ms()
        )
        self._groups[group] = info
        self._report("started", info)
        if self._watcher is None or self._watcher.done():
            self._watcher = asyncio.create_task(self._watch_groups())

    @classmethod
    def _group_alive(cls, group: int) -> bool:
        try:
            os.killpg(group, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            try:
                return cls._live_group(group)
            except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired):
                return True  # cannot tell: keep it until it can be told

    async def _watch_groups(self) -> None:
        while self._groups:
            await asyncio.sleep(BACKGROUND_POLL_S)
            for group, info in list(self._groups.items()):
                if not await asyncio.to_thread(self._group_alive, group):
                    del self._groups[group]
                    self._report("ended", info.model_copy(update={"state": "exited", "ended_at": now_ms()}))

    async def __aenter__(self) -> ProcessManager:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def start(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_s: float = 300,
        env: Mapping[str, str] | None = None,
        abort: asyncio.Event | None = None,
        memory_limit: int | None = None,
        executor: JobExecutor | None = None,
        output_filter: OutputFilter | None = None,
        shown_command: str | None = None,
        stdin: bool = False,
    ) -> ProcessSnapshot:
        """Start ``command`` as an owned job; its process group is stopped past ``timeout_s`` or, when given, past
        ``memory_limit`` bytes of resident memory (read every BACKGROUND_POLL_S). With ``executor`` the job runs
        where that executor runs commands, and the executor's own limits bound its memory. ``shown_command`` is the
        command as the agent wrote it, for the record, when ``command`` is a prepared one (secrets injected). Like a
        bash call, it reads end-of-file on its input unless ``stdin``: a command that reads its input (``docker exec
        -i``, a prompt) would otherwise wait on it forever; only a job started with ``stdin`` takes ``write``."""
        if not command.strip() or not 0 < timeout_s <= 86_400:
            raise ValueError("Provide a command and a timeout in (0, 86400] seconds")
        shown = shown_command or command
        async with self._lock:
            if self._closed:
                raise RuntimeError("Process manager is closed")
            if abort is not None and abort.is_set():
                raise RuntimeError("Run was aborted before process creation")
            if sum(not j.done.is_set() for j in self._jobs.values()) >= self.max_running:
                raise RuntimeError("The running-process limit is reached; terminate or wait for an owned job")
            for handle, old in list(self._jobs.items()):
                if len(self._jobs) < self.max_records:
                    break
                if old.done.is_set():
                    del self._jobs[handle]
            if executor is not None:
                remote = await executor.start_job(command, cwd=cwd, env=env)
                if not stdin:
                    with contextlib.suppress(NotImplementedError, ValueError):
                        await remote.write(b"", eof=True)
                job = _Job(
                    handle=uuid.uuid4().hex,
                    command=shown,
                    remote=remote,
                    output_filter=output_filter,
                    takes_input=stdin,
                )
                self._jobs[job.handle] = job
                self._report("started", self._info(job))
                job.task = asyncio.create_task(self._supervise_remote(job, timeout_s, abort))
                return self._snapshot(job, None)
            process = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                command,
                cwd=cwd,
                env=shell_env(os.environ if env is None else env),
                stdin=asyncio.subprocess.PIPE if stdin else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            job = _Job(
                handle=uuid.uuid4().hex, command=shown, process=process, output_filter=output_filter, takes_input=stdin
            )
            self._jobs[job.handle] = job
            self._report("started", self._info(job))
            job.task = asyncio.create_task(self._supervise(job, timeout_s, abort, memory_limit))
            return self._snapshot(job, None)

    def _append(self, job: _Job, chunk: bytes) -> None:
        if chunk:
            job.last_output_at = now_ms()
        job.output.extend(chunk)
        overflow = max(0, len(job.output) - self.max_log_bytes)
        if overflow:
            del job.output[:overflow]
            job.base += overflow

    def _receive(self, job: _Job, chunk: bytes) -> None:
        self._append(job, job.output_filter.feed(chunk) if job.output_filter is not None else chunk)

    def _end_of_output(self, job: _Job) -> None:
        if job.output_filter is not None:
            self._append(job, job.output_filter.flush())

    async def _drain(self, job: _Job) -> None:
        assert job.process is not None and job.process.stdout is not None
        try:
            while chunk := await job.process.stdout.read(16_384):
                self._receive(job, chunk)
        finally:
            self._end_of_output(job)

    @staticmethod
    def _live_group(group: int) -> bool:
        # Darwin can report EPERM for a group containing only zombies. Check
        # status without collecting process arguments or trusting a shell.
        result = subprocess.run(
            ["/bin/ps", "-o", "pgid=,stat=", "-g", str(group)],
            capture_output=True,
            check=False,
            timeout=2,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        if result.returncode not in {0, 1}:
            raise RuntimeError("Could not inspect process-group cleanup")
        for row in result.stdout.decode().splitlines():
            fields = row.split()
            if len(fields) != 2:
                raise RuntimeError("Unexpected process-group status")
            if int(fields[0]) == group and not fields[1].startswith(("Z", "X")):
                return True
        return False

    async def _signal(self, job: _Job, sig: signal.Signals | int) -> bool:
        assert job.process is not None
        try:
            os.killpg(job.process.pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            if not await asyncio.to_thread(self._live_group, job.process.pid):
                return False
            # Never treat live members with changed credentials as cleaned up.
            raise

    async def _kill(self, job: _Job) -> None:
        # terminate() and the supervisor can converge on the same cleanup.
        # Do not signal a reaped group again (zombie-only groups can return
        # EPERM on macOS, and a completed group ID could eventually be reused).
        if not job.kill_sent:
            await self._signal(job, signal.SIGKILL)
            job.kill_sent = True

    async def _stop(self, job: _Job) -> None:
        if job.moved is not None:
            # Cancelling its execution stops the command's process group, as a timeout would.
            job.moved.cancel()
            await asyncio.gather(job.moved, return_exceptions=True)
            return
        if job.remote is not None:
            await self._stop_remote(job)
            return
        assert job.process is not None
        if job.kill_sent:
            await job.process.wait()
            return
        await self._signal(job, signal.SIGTERM)
        try:
            # The shell may exit before a descendant does, and descendants can
            # close their output pipes. Neither parent exit nor EOF proves the
            # owned process group has stopped.
            deadline = asyncio.get_running_loop().time() + 1.0
            while not job.kill_sent and asyncio.get_running_loop().time() < deadline:
                if not await self._signal(job, 0):
                    break
                await asyncio.sleep(0.02)
        finally:
            # Also enforce cleanup when this grace period itself is cancelled.
            await self._kill(job)
        await job.process.wait()

    async def _watch_memory(self, job: _Job, limit: int) -> None:
        """Return once the job's process group holds more than ``limit`` bytes, having said so in its output. Memory
        that cannot be read stops the watch, not the job."""
        while True:
            await asyncio.sleep(BACKGROUND_POLL_S)
            if (usage := await memory_by_group()) is None:  # cannot be read here: stop watching, not the job
                await asyncio.get_running_loop().create_future()
            assert job.process is not None
            used = usage.get(job.process.pid, 0) if usage is not None else 0
            if used > limit:
                self._append(job, f"\n[{memory_stopped(used, limit)}]\n".encode())
                return

    async def _supervise(
        self, job: _Job, timeout_s: float, abort: asyncio.Event | None, memory_limit: int | None = None
    ) -> None:
        assert job.process is not None
        drain = asyncio.create_task(self._drain(job))
        exited = asyncio.create_task(job.process.wait())
        aborted = asyncio.create_task(abort.wait()) if abort is not None else None
        memory = asyncio.create_task(self._watch_memory(job, memory_limit)) if memory_limit else None
        pending = {exited} | ({aborted} if aborted is not None else set()) | ({memory} if memory is not None else set())
        try:
            done, _ = await asyncio.wait(pending, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if exited not in done:
                if memory is not None and memory in done:
                    job.state = "out_of_memory"
                else:
                    job.state = "aborted" if aborted is not None and aborted in done else "timed_out"
            elif job.state == "running":
                job.state = "exited"
            # Normal parent exit also ends ownership: do not leave quiet
            # descendants running after this handle becomes a completed record.
            await self._stop(job)
            try:
                await asyncio.wait_for(asyncio.shield(drain), timeout=1.0)
            except TimeoutError:
                await self._kill(job)
        except asyncio.CancelledError:
            job.state = "terminated"
            await self._stop(job)
            raise
        finally:
            try:
                await self._kill(job)
            finally:
                extra = [t for t in (aborted, memory) if t is not None]
                for task in [exited, drain, *extra]:
                    if not task.done():
                        task.cancel()
                try:
                    await asyncio.gather(exited, drain, *extra, return_exceptions=True)
                finally:
                    if job.process.stdin is not None:
                        job.process.stdin.close()
                    job.done.set()
                    self._job_ended(job)

    async def _pump(self, job: _Job) -> None:
        assert job.remote is not None
        try:
            while chunk := await job.remote.read():
                self._receive(job, chunk)
        finally:
            self._end_of_output(job)

    async def _stop_remote(self, job: _Job) -> None:
        assert job.remote is not None
        if not job.kill_sent:
            job.kill_sent = True
            await asyncio.wait_for(job.remote.kill(), timeout=15)

    async def _supervise_remote(self, job: _Job, timeout_s: float, abort: asyncio.Event | None) -> None:
        """A job an executor runs: the same deadline, abort and output bound as a local job; the executor stops it."""
        assert job.remote is not None
        pump = asyncio.create_task(self._pump(job))
        exited = asyncio.create_task(job.remote.wait())
        aborted = asyncio.create_task(abort.wait()) if abort is not None else None
        pending = {exited} | ({aborted} if aborted is not None else set())
        try:
            done, _ = await asyncio.wait(pending, timeout=timeout_s, return_when=asyncio.FIRST_COMPLETED)
            if exited not in done:
                job.state = "aborted" if aborted is not None and aborted in done else "timed_out"
                await self._stop_remote(job)
            elif job.state == "running":
                job.state = "exited"
            if exited.done() and not exited.cancelled() and exited.exception() is None:
                job.remote_exit = exited.result()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(pump), timeout=1.0)
        except asyncio.CancelledError:
            job.state = "terminated"
            await self._stop_remote(job)
            raise
        finally:
            try:
                # Like a local job's group, nothing it started outlives its record.
                await self._stop_remote(job)
            finally:
                extra = [aborted] if aborted is not None else []
                for task in [exited, pump, *extra]:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(exited, pump, *extra, return_exceptions=True)
                job.done.set()
                self._job_ended(job)

    def _get(self, handle: str) -> _Job:
        if (job := self._jobs.get(handle)) is not None:
            return job
        if (past := self._past.get(handle)) is not None:
            raise ValueError(_from_before(past))
        # Names the jobs there are: a small model that mistypes a 32-character handle corrects it in one call.
        jobs = "; ".join(f"{j.handle} (`{_cut(j.command, 60)}`, {j.state})" for j in list(self._jobs.values())[-5:])
        raise ValueError(
            f"Unknown process handle {handle!r} (a PID is not a handle). "
            + (f"This session's jobs: {jobs}" if jobs else "This session has no jobs")
        )

    @staticmethod
    def _snapshot(job: _Job, cursor: int | None) -> ProcessSnapshot:
        requested = job.base if cursor is None else cursor
        end = job.base + len(job.output)
        if requested < 0 or requested > end:
            raise ValueError("Output cursor is outside the available stream")
        start = max(requested, job.base)
        if job.done.is_set():
            job.end_seen = True
        return ProcessSnapshot(
            handle=job.handle,
            command=_shown(job.command),
            elapsed_s=((job.ended_at or now_ms()) - job.started_at) // 1000,
            state=job.state,
            exit_code=job.returncode,
            output=bytes(job.output[start - job.base :]).decode("utf-8", errors="replace"),
            cursor=start,
            next_cursor=end,
            omitted_bytes=max(0, job.base - requested),
        )

    async def poll(self, handle: str, *, cursor: int | None = None, wait_s: float = 0) -> ProcessSnapshot:
        """The job's output from `cursor`, after waiting up to `wait_s` for it to end. The agent's `process poll` never
        waits: waiting is the harness's; `wait_s` is for the harness's own code."""
        if not 0 <= wait_s <= 30:
            raise ValueError("wait_s must be between 0 and 30 seconds")
        job = self._get(handle)
        if wait_s and not job.done.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.done.wait(), timeout=wait_s)
        return self._snapshot(job, cursor)

    def job_running(self, handle: str) -> bool:
        """Whether `handle` names a job of this manager's that still runs."""
        job = self._jobs.get(handle)
        return job is not None and not job.done.is_set()

    def jobs_running(self) -> list[ProcessInfo]:
        """The jobs that still run (not foreground calls, nor what commands left running)."""
        return [self._info(job) for job in self._jobs.values() if not job.done.is_set()]

    def unseen(self, ended: Sequence[ProcessInfo]) -> list[ProcessInfo]:
        """Of `ended` (processes reported ended), those whose end the agent has not been shown: a job it has not polled,
        listed or stopped since, and what commands left running, whose end only a notice tells."""
        return [
            info
            for info in ended
            if info.kind == "left_running"
            or (info.kind == "job" and (job := self._jobs.get(info.id)) is not None and not job.end_seen)
            or (info.kind == "job" and info.id in self._past and info.id not in self._past_told)
        ]

    def end_notice(self, ended: Sequence[ProcessInfo]) -> str | None:
        """What to tell the agent of the processes in `ended` whose end it has not been shown, each job with the end
        of its output; None when there is nothing to tell. The jobs count as seen from here on. Each process's own
        line begins with `ENDED_LINES` (the terminal UI shows those)."""
        lines, jobs = [], False
        for info in self.unseen(ended):
            took = duration_text(((info.ended_at or now_ms()) - info.started_at) // 1000)
            if info.id in self._past:
                self._past_told.add(info.id)
                lines.append(_from_before(info))
            elif info.kind == "job":
                job, jobs = self._jobs[info.id], True
                job.end_seen = True
                status = info.state.replace("_", " ") + (
                    f", exit code {info.exit_code}" if info.exit_code is not None else ""
                )
                tail = bytes(job.output[-NOTICE_OUTPUT_CHARS:]).decode("utf-8", errors="replace").strip()
                said = f"\nThe end of its output:\n{tail}" if tail else "\nIt wrote no output."
                lines.append(f"Background job {info.id} (`{info.command}`) ended: {status}, after {took}.{said}")
            else:
                group = info.id.removeprefix("group-")
                lines.append(
                    f"A process the command `{info.command}` left running (process group {group}) ended after {took}; "
                    "its output went where the command sent it."
                )
        if not lines:
            return None
        if jobs:
            lines.append("Read more of a job's output with process poll and its handle.")
        return "\n\n".join(lines)

    async def write(self, handle: str, data: str = "", *, eof: bool = False) -> ProcessSnapshot:
        job = self._get(handle)
        if len(data.encode()) > 64 * 1024:
            raise ValueError("Input exceeds 64 KiB")
        if job.moved is not None:
            raise ValueError("A command moved to the background takes no input; start it with the process tool instead")
        if not job.takes_input:
            raise ValueError("This job was started to take no input; start it with stdin: true to write to it")
        if job.remote is not None:
            if job.done.is_set():
                raise ValueError("The process no longer accepts input")
            try:
                await job.remote.write(data.encode(), eof=eof)
            except NotImplementedError:
                raise ValueError(
                    "Jobs on this executor take no input; start the command with its input instead"
                ) from None
            return self._snapshot(job, None)
        assert job.process is not None
        if job.done.is_set() or job.process.returncode is not None or job.process.stdin is None:
            raise ValueError("The process no longer accepts input")
        job.process.stdin.write(data.encode())
        await asyncio.wait_for(job.process.stdin.drain(), timeout=5)
        if eof:
            job.process.stdin.close()
        return self._snapshot(job, None)

    async def terminate(self, handle: str) -> ProcessSnapshot:
        job = self._get(handle)
        if not job.done.is_set():
            job.state = "terminated"
            await self._stop(job)
            await job.done.wait()
        return self._snapshot(job, None)

    def list(self) -> list[ProcessSnapshot]:
        return [self._snapshot(job, None) for job in self._jobs.values()]

    async def aclose(self) -> None:
        self._closed = True
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
        await asyncio.gather(*(self.terminate(j.handle) for j in self._jobs.values() if not j.done.is_set()))
