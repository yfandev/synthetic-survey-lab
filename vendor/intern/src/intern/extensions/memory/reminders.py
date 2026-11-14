from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import ClassVar

from intern import hooks as h
from intern.journal import MemoryRecallEntry
from intern.memory.drive import Drive
from intern.memory.format import Page, is_page_file, strip_source
from intern.memory.steward.extract import (
    READ_METHODS,
    READ_TOOLS,
    argument_calls,
    call_targets,
    http_calls,
    segments,
    shell_functions,
    shell_operation,
    shell_tokens,
    without_heredocs,
)
from intern.memory.steward.place import page_for_host, page_for_tool
from intern.messages import ToolCall

FOLDERS = ("map/", "user/")
RULE_BLOCKS = ("b-rules",)
PLACE_BLOCKS = ("b-where", "b-fails")
# Lines that say how to use something, wherever they are, and the names in them a call can use.
NAMED_FOLDERS = ("map/", "user/", "lessons/")
NAMED_BLOCKS = ("b-rules", "b-where", "b-fails", "b-how", "b-steps", "b-pitfalls", "b-checks")
_IDENT = re.compile(r"(?<![\w./-])[A-Za-z][\w-]*(?:[_./][\w-]+)+")
# A call that already holds the longest command a line quotes follows it, unless the line warns against one of them.
WARNING_BLOCKS = ("b-fails", "b-pitfalls")
_QUOTED = re.compile(r"`([^`]+)`")
_WARNS = re.compile(r"\b(never|not|don't|avoid|instead of|rather than|without|no longer|stop)\b[^`.;]{0,30}$", re.I)
MIN_IDENT = 6
# A line naming more names than this lists what exists rather than saying how to use one of them.
MAX_LINE_IDENTS = 3
MAX_LINES_CHARS = 2_000
READ_VERBS = frozenset(
    {
        "get",
        "list",
        "search",
        "find",
        "grep",
        "read",
        "show",
        "describe",
        "fetch",
        "lookup",
        "log",
        "logs",
        "query",
        "view",
        "status",
        "tools",
        "help",
        "info",
        "diff",
    }
)
WRITE_VERBS = frozenset(
    {
        "create",
        "update",
        "delete",
        "remove",
        "set",
        "add",
        "close",
        "open",
        "order",
        "apply",
        "submit",
        "send",
        "post",
        "put",
        "patch",
        "unlock",
        "lock",
        "transfer",
        "cancel",
        "write",
        "edit",
        "move",
        "rename",
        "push",
        "commit",
        "merge",
        "install",
        "give",
        "reset",
        "pay",
        "refund",
        "approve",
        "reject",
    }
)
_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
_NAME = re.compile(r"[A-Za-z][\w.-]*")
READ_WORDS = 3
MIN_NAME = 3
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_ISO_DATE = re.compile(r"\b20\d\d-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b")
_WORD_DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?!\d)", re.I)
REFUSAL = (
    "Memory for this action, from earlier sessions:\n{lines}\n"
    "Check the call against each line: an argument to change, a step to take first, or nothing. When a line names "
    "a document, record or page that holds what the call needs, read it before the call. Leaving out an argument a "
    "line is about does not follow the line."
)
NEXT = (
    "Say in one sentence what these lines change in the call, or that it already follows them, "
    "then make the call that follows them."
)


def _dates(text: str) -> set[tuple[int, int]]:
    found = {(int(m), int(d)) for m, d in _ISO_DATE.findall(text)}
    found |= {(MONTHS.index(m.lower()[:3]) + 1, int(d)) for m, d in _WORD_DATE.findall(text)}
    return found


def _base(block_id: str) -> str:
    return "-".join(block_id.split("-")[:2])


@dataclass
class _Entry:
    page: Page
    names: list[str]
    rules: list[str]
    place: list[str]
    dates: set[tuple[int, int]]


def _entries(drive: Drive) -> list[_Entry]:
    found: list[_Entry] = []
    for name in drive.files:
        if not is_page_file(name) or not name.startswith(FOLDERS) or "/history" in name or "/timeline/" in name:
            continue
        page = drive.page(name[:-3])
        if page is None:
            continue
        rules = [strip_source(b)[2:] for blk in page.blocks if _base(blk.id) in RULE_BLOCKS for b in blk.bullets]
        place = [strip_source(b)[2:] for blk in page.blocks if _base(blk.id) in PLACE_BLOCKS for b in blk.bullets]
        if not rules and not (page.kind == "subject" and place):
            continue
        aliases = [a.strip() for a in page.meta.get("aliases", "").split(",") if a.strip()]
        names = [n for n in [page.meta.get("name", ""), *aliases] if len(n) >= MIN_NAME and n != "the user"]
        found.append(_Entry(page, names, rules, place, {d for r in rules for d in _dates(r)}))
    return found


@dataclass
class _Named:
    path: str
    text: str
    idents: list[str]
    commands: list[str]


def _named(drive: Drive) -> list[_Named]:
    # A name with a separator in it (a tool, a path, a command, a host) is specific enough to tie a line to a call.
    found: list[_Named] = []
    for name in drive.files:
        if not is_page_file(name) or not name.startswith(NAMED_FOLDERS) or "/history" in name or "/timeline/" in name:
            continue
        page = drive.page(name[:-3])
        if page is None:
            continue
        for blk in page.blocks:
            if _base(blk.id) not in NAMED_BLOCKS:
                continue
            for bullet in blk.bullets:
                text = strip_source(bullet)[2:]
                idents = {i.rstrip(".") for i in _IDENT.findall(text)}
                keep = sorted(i for i in idents if len(i) >= MIN_IDENT)
                if 0 < len(keep) <= MAX_LINE_IDENTS:
                    quoted = [q for q in _QUOTED.finditer(text) if len(q[1]) >= MIN_IDENT]
                    warns = _base(blk.id) in WARNING_BLOCKS or any(_WARNS.search(text[: q.start()]) for q in quoted)
                    longest = max((" ".join(q[1].split()) for q in quoted), key=len, default="")
                    commands = [] if warns or not longest else [longest]
                    found.append(_Named(page.path, text, keep, commands))
    return found


def _call_text(call: ToolCall) -> str:
    command = call.arguments.get("command") if call.name == "bash" else None
    return command if isinstance(command, str) else f"{call.name} {json.dumps(dict(call.arguments))}"


def changes(call: ToolCall) -> bool:
    # Reads never change anything; a shell command changes something when it sends a write request or runs a
    # program beyond the basics that only print and move text.
    if call.name in READ_TOOLS:
        return False
    command = call.arguments.get("command") if call.name == "bash" else None
    if not isinstance(command, str):
        return call.name != "bash"
    functions = shell_functions(command)
    for segment in segments(without_heredocs(command)):
        tokens = shell_tokens(segment)
        found = http_calls(tokens)
        if any(key.split(" ", 1)[0] not in READ_METHODS for _, key, _ in found):
            return True
        operation = None if found else shell_operation(tokens)
        if operation is not None and operation.split()[0] not in functions and not _reads(operation, tokens):
            return True
    return False


def _reads(operation: str, tokens: list[str]) -> bool:
    # `gh pr view`, `git status`, `kubectl get pods`: the subcommand or tool it names says it only reads.
    program = operation.split()[0]
    start = next((i + 1 for i, t in enumerate(tokens) if t.rsplit("/", 1)[-1] == program), len(tokens))
    names = []
    for token in tokens[start : start + READ_WORDS]:
        if not _NAME.fullmatch(token):
            break
        names.append(token)
    words = {w.lower() for name in names for w in _WORD.findall(name)}
    return bool(words & READ_VERBS) and not words & WRITE_VERBS


def _places(call: ToolCall) -> set[str]:
    command = call.arguments.get("command") if call.name == "bash" else None
    calls = (
        [c for s in segments(without_heredocs(command)) for c in http_calls(shell_tokens(s))]
        if isinstance(command, str)
        else argument_calls(call.arguments)
    )
    pages = {page_for_host(host) for host, _, _ in calls}
    if call.name not in ("bash", "write", "edit"):
        pages.add(page_for_tool(call.name))
    return pages


# A call that may change something is held once per memory page about its targets, and the agent is shown that
# page's rules and where-entries it has not seen yet. It is also held once per line that names what the call uses,
# even when recall loaded that line's page: a line read at the start is far from the call it is about.
class MemoryActions(h.Extension):
    name: ClassVar[str] = "memory-actions"

    def __init__(self, drive: Drive) -> None:
        self.drive = drive
        self._entries: list[_Entry] | None = None
        self._named: list[_Named] | None = None
        self.shown: set[str] = set()
        self.shown_lines: set[str] = set()

    async def before_agent_start(self, ctx: h.RunContext) -> None:
        recalled = next((e for e in reversed(ctx.journal.active_path()) if isinstance(e, MemoryRecallEntry)), None)
        self.shown = set(recalled.pages) if recalled else set()

    def entries(self) -> list[_Entry]:
        if self._entries is None:
            self._entries = _entries(self.drive)
        return self._entries

    async def tool_call(self, call: ToolCall, ctx: h.RunContext) -> h.ToolCallDecision | None:
        if not changes(call):
            return None
        targets = " \n ".join(call_targets(call.name, call.arguments)).lower()
        places = _places(call)
        dates = _dates(" ".join(call_targets(call.name, call.arguments)))
        matched: list[_Entry] = []
        for entry in self.entries():
            if entry.page.path in self.shown:
                continue
            named = any(re.search(rf"(?<![\w]){re.escape(n.lower())}(?![\w])", targets) for n in entry.names)
            if named or entry.page.path in places or (dates & entry.dates):
                matched.append(entry)
        parts: list[str] = []
        for entry in matched:
            self.shown.add(entry.page.path)
            lines = [f"  - rule: {r}" for r in entry.rules] + [f"  - {p}" for p in entry.place]
            self.shown_lines.update([*entry.rules, *entry.place])
            parts.append(f"{entry.page.path}:\n" + "\n".join(lines))
        if self._named is None:
            self._named = _named(self.drive)
        used = _call_text(call)
        spaced = " ".join(used.split())
        by_page: dict[str, list[str]] = {}
        for line in self._named:
            if line.text in self.shown_lines:
                continue
            if line.commands and all(c in spaced for c in line.commands):
                self.shown_lines.add(line.text)
                continue
            if any(re.search(rf"(?<![\w./-]){re.escape(i)}(?![\w/-])", used) for i in line.idents):
                self.shown_lines.add(line.text)
                by_page.setdefault(line.path, []).append(f"  - {line.text}")
        parts += [f"{path}:\n" + "\n".join(lines) for path, lines in by_page.items()]
        if not parts:
            return None
        text = "\n".join(parts)
        if len(text) > MAX_LINES_CHARS:
            text = text[:MAX_LINES_CHARS] + "\n  [more on the pages above]"
        return h.Refuse(reason=REFUSAL.format(lines=text), next_action=NEXT)
