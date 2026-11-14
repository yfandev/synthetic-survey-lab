"""The bash tool. Most real work reaches the agent as command-line programs, so this is
the one tool every task can use."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from intern import names
from intern.events import ProcessState
from intern.executor import CapturingExecutor, ExecResult, LocalExecutor, memory_stopped, run_command
from intern.messages import ResourceRef
from intern.processes import JUDGE_AFTER_S, Judgment, ProcessManager, ProcessSnapshot, duration_text
from intern.providers.base import ToolSpec
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 1800
SERVES_MAX_S = 86_400
"""How long a command that serves may run once its call's timeout_s no longer holds: as long as a `process` job."""
# Only this excerpt stays in memory; the stream is captured to disk before head/tail truncation.
CAPTURE_FACTOR = 8


def _capture_resource(path: Path) -> ResourceRef:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return ResourceRef(uri=path.resolve().as_uri(), sha256=digest)


WAIT_MIN_SLEEP_S = 10.0  # a `sleep` this long or longer, first in a command, is a wait by rule


def obvious_wait(command: str) -> str | None:
    """What kind of wait `command` plainly is, so it goes on in the background as it starts: a long `sleep`
    first, `tail -f`, `watch`, a CI watch. None for anything else; System 1 judges those once they run a while."""
    text = re.sub(r"^(\s*cd\s+\S+\s*(&&|;))+", "", command).strip()
    if (sleep := re.match(r"sleep\s+(\d+(?:\.\d+)?)\b", text)) is not None:
        return f"`sleep {sleep.group(1)}` is a wait" if float(sleep.group(1)) >= WAIT_MIN_SLEEP_S else None
    if re.search(r"\btail\s+-\S*[fF]", text):
        return "`tail -f` does not end"
    if re.match(r"watch\s", text):
        return "`watch` does not end"
    if re.search(r"\bgh\s+run\s+watch\b", text) or re.search(r"\bgh\s+pr\s+checks\b.*--watch\b", text):
        return "a CI watch is a wait"
    return None


async def _moves(
    execution: asyncio.Task[ExecResult], processes: ProcessManager, call_id: str, args: BashArgs
) -> tuple[Literal["rule", "time", "stuck", "waits", "serves"], str] | None:
    """Wait for the command to end (None), or for a reason to go on in the background, with why:
    it is plainly a wait (`rule`); System 1, asked once it has run JUDGE_AFTER_S, finds it `waits` or `serves`; it
    reached `move_after_s` (`time`); or a move was requested (`stuck`): System 1's check found it stuck, or a message
    from the user came in."""
    if (plain := obvious_wait(args.command)) is not None:
        return "rule", plain
    loop = asyncio.get_running_loop()
    deadline = loop.time() + processes.move_after_s
    requested = asyncio.create_task(processes.move_requested(call_id).wait())
    judging: asyncio.Task[Judgment | None] | None = None
    try:
        if processes.judge is not None and processes.move_after_s > JUDGE_AFTER_S:
            await asyncio.wait({execution, requested}, timeout=JUDGE_AFTER_S, return_when=asyncio.FIRST_COMPLETED)
            if not execution.done() and not requested.done():
                judging = asyncio.create_task(processes.judge(args.command))
        while True:
            if execution.done():
                return None
            if requested.done():
                return "stuck", processes.move_why(call_id)
            if judging is not None and judging.done():
                if judged := judging.result():  # System1Unavailable goes up: the run stops, and the command with it
                    kind, share = judged
                    return kind, f"System 1 judged it {'serves' if kind == 'serves' else 'waits'} ({share:.2f})"
                judging = None
            left = deadline - loop.time()
            if left <= 0:
                return "time", f"still running at {duration_text(round(processes.move_after_s))}"
            waits: set[asyncio.Future[object]] = {execution, requested}  # type: ignore[arg-type]
            if judging is not None:
                waits.add(judging)  # type: ignore[arg-type]
            await asyncio.wait(waits, timeout=left, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (requested, judging):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task


class _Deadline:
    """What stops a `bash` command: the run's abort, or its timeout_s. The executor waits on `stop` beside the command,
    so the timeout_s can be lifted after it starts: one found to serve (a server, a watcher) does not end, and the
    timeout_s the agent gave a call it expected to block on would only kill it."""

    def __init__(self, abort: asyncio.Event, timeout_s: float) -> None:
        self.stop = asyncio.Event()
        self.expired = False
        self._abort = abort
        self._timer = asyncio.get_running_loop().call_later(timeout_s, self._expire)

    def _expire(self) -> None:
        if not self.stop.is_set():
            self.expired = True
            self.stop.set()

    def lift(self) -> None:
        """No timeout_s from here on: it runs until it is stopped or its session closes (SERVES_MAX_S at most)."""
        self._timer.cancel()

    async def run(self, command: Callable[[], Awaitable[ExecResult]]) -> ExecResult:
        """The command's result, as a timeout when its deadline stopped it rather than the run's abort."""

        async def relay() -> None:
            await self._abort.wait()
            self.stop.set()

        relaying = asyncio.create_task(relay())
        try:
            result = await command()
        finally:
            relaying.cancel()
            self._timer.cancel()
        if self.expired and result.aborted:
            result = result.model_copy(update={"aborted": False, "timed_out": True})
        return result


def _moved(
    snapshot: ProcessSnapshot, capture: Path | None, max_chars: int, why: str, timeout_s: int, serves: bool
) -> ToolOutput:
    """What the model sees of a command that went on in the background: why, the job's handle, when it is still
    stopped, its output so far."""
    so_far = ""
    if capture is not None:
        try:
            with capture.open("rb") as stream:
                stream.seek(max(0, capture.stat().st_size - max_chars))
                so_far = stream.read().decode("utf-8", errors="replace").strip()
        except OSError:
            so_far = ""
    # A command that works keeps the timeout_s the agent gave a call it expected to block on: say so, so its deadline
    # surprises no one. One that serves is freed of it.
    stops = (
        "It serves, so its timeout_s no longer applies: it runs until it is stopped or the session closes"
        if serves
        else f"It is still stopped at its timeout_s, {duration_text(timeout_s)} after it started"
    )
    head = (
        f"[Running in the background as job {snapshot.handle} ({why}). {stops}. You are told when it ends: do other "
        "work, or end your turn. Read its output with process poll; stop it with process terminate.]"
    )
    return ToolOutput(content=f"{head}\n{so_far or '(no output yet)'}")


class BashArgs(BaseModel):
    command: str = Field(description="The command to run with bash in the working directory.")
    timeout_s: int = Field(
        default=DEFAULT_TIMEOUT_S, ge=1, le=MAX_TIMEOUT_S, description="Seconds before it is killed."
    )


class BashTool(Tool):
    name = "bash"
    description = (
        "Run a shell command and return its combined stdout and stderr with the exit code. "
        "Commands do not share state between calls, except through files. A command still running after a while "
        "goes on in the background: you get its output so far and a process handle, and are told when it ends."
    )
    STAY_DESCRIPTION = (
        "Run a shell command and return its combined stdout and stderr with the exit code. "
        "Commands do not share state between calls, except through files. To leave a command running while you "
        "work on, start it with the process tool; without one, in the background with its output redirected: "
        "cmd > out.log 2>&1 &"
    )
    """For a session whose commands never move to the background (`move_after_s = 0`, or no job manager)."""
    stays = False
    """Its commands never go on in the background: the model is shown STAY_DESCRIPTION (session setup sets it)."""

    def spec(self) -> ToolSpec:
        spec = super().spec()
        return spec.model_copy(update={"description": self.STAY_DESCRIPTION}) if self.stays else spec

    Args = BashArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read", "write", "exec", "network"})

    async def run(self, args: BashArgs, ctx: ToolContext) -> ToolOutput:
        directory = ctx.output_dir or ctx.cwd / names.PROJECT_DIR / "command-output"
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            with (directory / ".gitignore").open("x") as ignore:
                ignore.write("*\n")
        except FileExistsError:
            pass
        capture = directory / f"{uuid4().hex}.txt"
        retained = False
        try:
            output, moved = await self._run_captured(args, ctx, capture)
            retained = moved or bool(output.resources)  # a moved command's job reads it, and removes it
            return output
        finally:
            if not retained:
                capture.unlink(missing_ok=True)

    async def _run_captured(self, args: BashArgs, ctx: ToolContext, capture: Path) -> tuple[ToolOutput, bool]:
        """The command's result, and whether it went on in the background instead."""
        captures = isinstance(ctx.executor, CapturingExecutor) and ctx.executor.output_capture
        # The session records the call as running for as long as it runs, and what it leaves running after.
        processes, call_id = ctx.processes, ctx.call_id or uuid4().hex
        if processes is not None:
            processes.begin_call(call_id, args.command)
        deadline = _Deadline(ctx.abort, args.timeout_s)
        execution = asyncio.create_task(
            deadline.run(
                lambda: run_command(
                    ctx.executor,
                    args.command,
                    cwd=ctx.cwd,
                    timeout_s=SERVES_MAX_S,  # the executor's backstop; `deadline` holds the call's timeout_s
                    max_output_bytes=max(ctx.max_output_chars * CAPTURE_FACTOR, 1 << 20),
                    abort=deadline.stop,
                    capture_path=capture if captures else None,
                )
            )
        )
        try:
            moved = await _moves(execution, processes, call_id, args) if processes and processes.move_after_s else None
            if processes is not None and moved is not None:
                kind, why = moved
                # One that serves does not end: no run waits for it.
                snapshot = processes.adopt(
                    call_id,
                    args.command,
                    execution,
                    capture if captures else None,
                    held=kind != "serves",
                    note=why,
                    lift=deadline.lift,
                )
                return _moved(
                    snapshot, capture if captures else None, ctx.max_output_chars, why, args.timeout_s, kind == "serves"
                ), True
            result = await execution
        except BaseException:
            if not execution.done():
                execution.cancel()  # stops the command's process group
                await asyncio.gather(execution, return_exceptions=True)
            if processes is not None:
                processes.end_call(call_id, "aborted" if ctx.abort.is_set() else "terminated", None)
            raise
        if processes is not None:
            state: ProcessState = (
                "out_of_memory"
                if result.memory_exceeded
                else "timed_out"
                if result.timed_out
                else "aborted"
                if result.aborted
                else "exited"
            )
            processes.end_call(call_id, state, result.exit_code)
            if result.left_running_group is not None:
                processes.left_running(result.left_running_group, args.command)
        return await self._output(result, args, ctx, capture, captures), False

    async def _output(
        self, result: ExecResult, args: BashArgs, ctx: ToolContext, capture: Path, captures: bool
    ) -> ToolOutput:
        """What the model sees of a command that ended."""
        output = result.output.rstrip() or "(no output)"
        if isinstance(ctx.executor, LocalExecutor) and (notices := ctx.executor.take_notices()):
            output = "".join(f"[{notice}]\n" for notice in notices) + output
        resources = []
        if result.omitted_bytes and captures:
            resources.append(await asyncio.to_thread(_capture_resource, capture))
            label = (
                "Complete command output"
                if result.capture_complete
                else (
                    f"Partial command output archive ({result.capture_omitted_bytes} bytes beyond the disk limit "
                    "are unavailable, or the output stream remained open)"
                )
            )
            shown = capture.relative_to(ctx.cwd) if capture.is_relative_to(ctx.cwd) else capture
            output = f"[{label}: {shown}; read with offset/limit or grep.]\n{output}"
        elif result.omitted_bytes:
            output = "[Executor output is truncated; this executor does not support full output capture.]\n" + output
        metadata = {
            "resources": resources,
            "omitted_bytes": result.omitted_bytes,
            "capture_omitted_bytes": result.capture_omitted_bytes,
        }
        if result.aborted:
            return fail(
                "the command was aborted",
                "Check its effects before taking further action.",
                content=output,
                retry_safe=False,
            ).model_copy(update=metadata)
        if result.memory_exceeded:
            limit = ctx.executor.memory_limit if isinstance(ctx.executor, LocalExecutor) else None
            return fail(
                f"the command was {memory_stopped(result.memory_bytes, limit or result.memory_bytes)}",
                "Read less at once (fewer columns or rows, or in chunks), or query the data where it is (a database, "
                "or a tool that reads the files in place) rather than loading it whole.",
                content=output,
                retry_safe=False,
            ).model_copy(update=metadata)
        if result.timed_out:
            return fail(
                f"the command timed out after {args.timeout_s}s",
                "Check whether it changed files or remote state before retrying; then narrow it or adjust timeout_s.",
                argument="timeout_s",
                content=output,
                retry_safe=False,
            ).model_copy(update=metadata)
        if result.exit_code != 0:
            return fail(
                f"the command exited with code {result.exit_code}",
                "Read the output above and fix the cause before running it again.",
                content=output,
                retry_safe=False,
            ).model_copy(update=metadata)
        return ToolOutput(content=f"{output}\n[exit code 0]", **metadata)
