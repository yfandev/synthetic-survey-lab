"""Model-facing interface to an owned process manager."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from intern.executor import JobExecutor, LocalExecutor
from intern.processes import OutputFilter, ProcessManager
from intern.secrets import SecretAccessDenied, SecretExecutor
from intern.tools.base import Tool, ToolContext, ToolOutput, fail

QUICK_S = 3.0
"""How long `start` waits for a job before it returns its handle: one that ends by then comes back with its result,
as a bash call would, so a quick command costs the agent one turn, not a start and a poll."""


class ProcessArgs(BaseModel):
    action: Literal["start", "poll", "write", "terminate", "list"]
    command: str = Field(default="", max_length=100_000)
    handle: str = ""
    data: str = Field(default="", max_length=65_536)
    eof: bool = False
    cursor: int | None = Field(default=None, ge=0)
    stdin: bool = False
    timeout_s: float = Field(default=300, gt=0, le=86_400)


class ProcessTool(Tool):
    name = "process"
    description = (
        "Run a command in the background where this session runs commands. start returns a handle (not a PID), or "
        "the result if it ends within 3 s; poll reads new output (pass next_cursor as cursor) and state; write feeds "
        "only a job started with stdin: true; terminate stops it; list shows your jobs. Jobs stop at timeout_s. poll "
        "never waits: you are told when a job ends, and a turn that only polls running jobs ends your turn until then."
    )
    Args = ProcessArgs
    effects = frozenset({"read", "write", "exec", "network"})

    def __init__(self, manager: ProcessManager | None = None) -> None:
        self.manager = manager or ProcessManager()

    async def run(self, args: Any, ctx: ToolContext) -> ToolOutput:
        assert isinstance(args, ProcessArgs)
        # Jobs run where the session's commands run: never escape a sandbox onto the host. With secrets, the job is
        # prepared like a command (scrubbed environment, injected values) and runs where the wrapped executor runs.
        secrets = ctx.executor if isinstance(ctx.executor, SecretExecutor) else None
        inner = secrets.executor if secrets is not None else ctx.executor
        local = type(inner) is LocalExecutor
        executor = inner if not local and isinstance(inner, JobExecutor) else None
        if not local and executor is None:
            return fail(
                "This session's executor cannot run background jobs.", "Run the command with bash, or switch runner."
            )
        if ctx.abort.is_set():
            return fail("The run was aborted.", "Stop.").model_copy(update={"status": "refused"})
        try:
            match args.action:
                case "start":
                    command, env = args.command, None
                    output_filter: OutputFilter | None = None
                    if secrets is not None:
                        prepared = await secrets.prepare(args.command)
                        command, env = prepared.command, prepared.env
                        output_filter = prepared.redactor.stream()
                    result = await self.manager.start(
                        command,
                        cwd=ctx.cwd,
                        timeout_s=args.timeout_s,
                        abort=ctx.abort,
                        env=env,
                        memory_limit=inner.memory_limit if isinstance(inner, LocalExecutor) else None,
                        executor=executor,
                        output_filter=output_filter,
                        shown_command=args.command,
                        stdin=args.stdin,
                    )
                    if not args.stdin:  # one the agent will feed waits for its input, so it returns at once
                        result = await self.manager.poll(result.handle, wait_s=QUICK_S)
                case "poll":
                    result = await self.manager.poll(args.handle, cursor=args.cursor)
                case "write":
                    result = await self.manager.write(args.handle, args.data, eof=args.eof)
                case "terminate":
                    result = await self.manager.terminate(args.handle)
                case "list":
                    return ToolOutput(content="[" + ",".join(j.model_dump_json() for j in self.manager.list()) + "]")
            return ToolOutput(content=result.model_dump_json())
        except SecretAccessDenied as e:  # value-free by construction
            return fail(str(e), "Run it without the process tool, or ask for the secret's configuration to be fixed.")
        except (ValueError, RuntimeError, OSError, TimeoutError) as e:
            return fail(str(e), "Poll/list an owned handle, correct the arguments, or terminate a running job.")

    async def aclose(self) -> None:
        await self.manager.aclose()
