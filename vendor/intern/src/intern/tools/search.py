"""Search tools: grep and find. They use ripgrep when it is installed and fall back to Python."""

from __future__ import annotations

import asyncio
import fnmatch
import math
import os
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from intern import names
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail

SKIP_DIRS = frozenset(
    {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", names.PROJECT_DIR}
    | {".decision_workspace", ".drex_workspace"}  # the workspace index (and its earlier name): a copy of every text
)
MAX_LINE_CHARS = 300
SEARCH_TIMEOUT_S = 60


def _ripgrep() -> str | None:
    return shutil.which("rg")


async def _run(argv: list[str], cwd: Path) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), SEARCH_TIMEOUT_S)
    except TimeoutError:
        process.kill()
        await process.wait()
        return -1, "search timed out"
    code = process.returncode if process.returncode is not None else -1
    return code, (stdout if code in (0, 1) else stderr).decode(errors="replace")


def _walk(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            yield Path(folder) / name


def _relative(path: Path, cwd: Path) -> str:
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


class GrepArgs(BaseModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str = Field(default=".", description="File or directory to search.")
    glob: str | None = Field(default=None, description="Only search files matching this glob, e.g. '*.py'.")
    ignore_case: bool = False
    max_matches: int = Field(default=100, ge=1, le=1000)


class GrepTool(Tool):
    name = "grep"
    description = "Search file contents with a regular expression. Returns path:line:text for each match."
    Args = GrepArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read"})

    async def run(self, args: GrepArgs, ctx: ToolContext) -> ToolOutput:
        root = ctx.resolve(args.path)
        if not root.exists():
            return fail(f"{args.path} does not exist", "Search an existing directory, e.g. '.'.", argument="path")
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
        except re.error as e:
            return fail(f"invalid regular expression: {e}", "Fix or escape the pattern.", argument="pattern")

        rg = _ripgrep()
        if rg:
            argv = [rg, "--line-number", "--no-heading", "--color", "never"]
            argv += ["--max-columns", str(MAX_LINE_CHARS), "--max-columns-preview"]
            argv += ["-i"] if args.ignore_case else []
            argv += ["-g", args.glob] if args.glob else []
            # Always pass a path: without one ripgrep reads stdin when stdin is not a terminal.
            argv += ["-e", args.pattern, "--", _relative(root, ctx.cwd)]
            code, output = await _run(argv, ctx.cwd)
            if code not in (0, 1):
                return fail(f"grep failed: {output.strip()[:300]}", "Check the pattern and path.")
            matches = [line.removeprefix("./") for line in output.splitlines()]
        else:
            matches = []
            for file in _walk(root):
                if args.glob and not fnmatch.fnmatch(file.name, args.glob):
                    continue
                try:
                    text = file.read_text()
                except (UnicodeDecodeError, OSError):
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        matches.append(f"{_relative(file, ctx.cwd)}:{number}:{line[:MAX_LINE_CHARS]}")

        if not matches:
            return ToolOutput(content=f"No matches for {args.pattern!r} in {args.path}.")
        shown = matches[: args.max_matches]
        footer = (
            f"\n[{len(shown)} of {len(matches)} matches shown; narrow the pattern, path or glob.]"
            if len(matches) > len(shown)
            else ""
        )
        return ToolOutput(content="\n".join(shown) + footer)


MEMORY_SESSIONS = "sessions"
MAX_SESSION_LINES = 8
_QUERY_WORD = re.compile(r"[\w.@-]{2,}")


class MemorySearchArgs(BaseModel):
    query: str = Field(description="Words to look for: names, ids, values, commands or what was asked.")
    max_sessions: int = Field(default=5, ge=1, le=20)


class MemorySearchTool(Tool):
    name = "memory_search"
    description = (
        "Search the transcripts of earlier sessions in this world for words, when no memory page holds what the "
        "task needs. Returns the best-matching sessions, each with its task, outcome and matching lines."
    )
    Args = MemorySearchArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read"})

    def __init__(self, drive: Path) -> None:
        self.folder = drive / MEMORY_SESSIONS

    async def run(self, args: MemorySearchArgs, ctx: ToolContext) -> ToolOutput:
        words = {w.lower() for w in _QUERY_WORD.findall(args.query)}
        if not words:
            return fail("the query has no words to search for", "Pass names, ids or values.", argument="query")
        return ToolOutput(content=await asyncio.to_thread(self._search, words, args.max_sessions))

    def _search(self, words: set[str], most: int) -> str:
        sessions: list[tuple[str, str, set[str]]] = []
        for file in sorted(self.folder.glob("*.txt")) if self.folder.is_dir() else []:
            try:
                text = file.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            lowered = text.lower()
            sessions.append((text, lowered, {w for w in words if w in lowered}))
        # A word few sessions use tells them apart; one every session uses says little.
        weight = {w: math.log((1 + len(sessions)) / (1 + sum(w in s[2] for s in sessions))) + 1 for w in words}
        ranked: list[tuple[tuple[float, str], str, list[str]]] = []
        for text, _, found in sessions:
            if not found:
                continue
            head, _, body = text.partition("\n\n")
            lines = body.splitlines()
            scored = sorted(
                ((sum(weight[w] for w in found if w in line.lower()), i) for i, line in enumerate(lines)),
                key=lambda s: -s[0],
            )
            hits = [lines[i] for score, i in scored if score > 0]
            date = next((line[6:] for line in head.splitlines() if line.startswith("date: ")), "")
            ranked.append(((sum(weight[w] for w in found), date), head, hits))
        if not ranked:
            return f"No earlier session mentions {' '.join(sorted(words))}."
        ranked.sort(key=lambda r: r[0], reverse=True)
        parts = []
        for _, head, hits in ranked[:most]:
            lines = [f"  {line.strip()[:MAX_LINE_CHARS]}" for line in hits[:MAX_SESSION_LINES]]
            more = f"\n  [{len(hits) - len(lines)} more matching lines]" if len(hits) > len(lines) else ""
            short = "\n".join(line[:MAX_LINE_CHARS] for line in head.splitlines())
            parts.append(short + ("\n" + "\n".join(lines) if lines else "") + more)
        return "\n\n".join(parts)


class FindArgs(BaseModel):
    pattern: str = Field(description="Glob for file paths, e.g. '*.py' or 'src/**/test_*.py'.")
    path: str = Field(default=".", description="Directory to search.")
    max_results: int = Field(default=200, ge=1, le=2000)


class FindTool(Tool):
    name = "find"
    description = "Find files by name with a glob pattern. Returns matching paths, sorted."
    Args = FindArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read"})

    async def run(self, args: FindArgs, ctx: ToolContext) -> ToolOutput:
        root = ctx.resolve(args.path)
        if not root.is_dir():
            return fail(f"{args.path} is not a directory", "Pass a directory as path, e.g. '.'.", argument="path")
        rg = _ripgrep()
        if rg:
            code, output = await _run([rg, "--files", "--color", "never", "-g", args.pattern, "--", "."], root)
            if code not in (0, 1):
                return fail(f"find failed: {output.strip()[:300]}", "Check the glob pattern.", argument="pattern")
            found = sorted(_relative(root / line, ctx.cwd) for line in output.splitlines())
        else:
            name_only = "/" not in args.pattern
            found = sorted(
                _relative(file, ctx.cwd)
                for file in _walk(root)
                if fnmatch.fnmatch(file.name if name_only else _relative(file, root), args.pattern)
            )
        if not found:
            return ToolOutput(content=f"No files match {args.pattern!r} in {args.path}.")
        shown = found[: args.max_results]
        footer = f"\n[{len(shown)} of {len(found)} files shown; narrow the pattern.]" if len(found) > len(shown) else ""
        return ToolOutput(content="\n".join(shown) + footer)
