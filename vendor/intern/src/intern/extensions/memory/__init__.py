from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from typing import ClassVar

from intern import hooks as h
from intern.decisions import System1Unavailable
from intern.events import MemoryRecall
from intern.extensions.memory.reminders import MemoryActions
from intern.journal import MemoryRecallEntry, MessageEntry
from intern.memory.recall import recall
from intern.messages import ToolCall, UserMessage
from intern.tools.base import Tool
from intern.tools.search import MemorySearchTool

#: Tools that write a file at their `path` (`sql_export`: the workspace's, a sql result to a file).
WRITE_TOOLS = frozenset({"write", "edit", "sql_export"})
NEXT_WRITE = "Write somewhere else. Memory changes are proposed after the session by `intern steward`."


class Memory(MemoryActions):
    name: ClassVar[str] = "memory"

    def tools(self) -> list[Tool]:
        return [MemorySearchTool(self.drive.root)]

    async def before_agent_start(self, ctx: h.RunContext) -> None:
        await self._recall(ctx)
        await super().before_agent_start(ctx)

    async def _recall(self, ctx: h.RunContext) -> None:
        started = time.monotonic()
        path = ctx.journal.active_path()
        entry = next((e for e in reversed(path) if isinstance(e, MemoryRecallEntry)), None)
        reused = entry is not None
        if entry is None:
            task = next(
                e.message.content
                for e in path
                if isinstance(e, MessageEntry) and isinstance(e.message, UserMessage) and e.message.source == "user"
            )
            budget = int(ctx.chars_per_token * ctx.working_window * ctx.profile.thresholds.memory)
            spent = ctx.decisions.usage.cost_usd
            try:
                found = await recall(self.drive, task, ctx.decisions, budget, ctx.state())
            except System1Unavailable:
                raise
            except Exception:
                # A broken drive must not end the run; it runs without memory and recall is tried next session.
                traceback.print_exc(file=sys.stderr)
                return
            entry = ctx.journal.append(
                MemoryRecallEntry(
                    drive=str(self.drive.root),
                    commit=self.drive.commit,
                    pages=found.pages,
                    paths=found.paths,
                    text=found.text,
                    backend=found.backend,
                    steps=found.steps,
                    cost_usd=ctx.decisions.usage.cost_usd - spent,
                )
            )
        ctx.emit(
            MemoryRecall(
                commit=entry.commit,
                pages=entry.pages,
                paths=entry.paths,
                chars=len(entry.text),
                steps=entry.steps,
                backend=entry.backend,
                reused=reused,
                duration_ms=int((time.monotonic() - started) * 1000),
            )
        )

    async def tool_call(self, call: ToolCall, ctx: h.RunContext) -> h.ToolCallDecision | None:
        if refused := self._refuse_write(call, ctx):
            return refused
        return await super().tool_call(call, ctx)

    def _refuse_write(self, call: ToolCall, ctx: h.RunContext) -> h.Refuse | None:
        given = call.arguments.get("path")
        if call.name not in WRITE_TOOLS or not isinstance(given, str):
            return None
        path = Path(given).expanduser()
        real = (path if path.is_absolute() else Path(ctx.cwd) / path).resolve()
        root = self.drive.root.resolve()
        if real != root and root not in real.parents:
            return None
        return h.Refuse(
            reason=f"{given} is inside the memory drive {self.drive.root}, which task runs never write",
            next_action=NEXT_WRITE,
        )
