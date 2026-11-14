"""System 1's State: a cut of the raw session, built by code.

The session as fast-jev-compaction builds Jev's input (https://github.com/tamaratran/fast-jev-compaction,
`src/state.ts`, our own port): a goal from the user's last three prompts and the whole history in order. The history
holds the user's messages, the harness's, the agent's replies without their reasoning, and each tool call with its
input and a note in place of its output: its status and size (`ok, 48213 chars (omitted)`), and here also a check's
tally and the output's first HEAD_CHARS, where fast-jev gives Jev status and size only. Asked about the session, System
1 answered 95% of question probes with them, 84% without: what checks and errors showed was lost. It is
fitted to a token budget in fast-jev's stages, each tried only when the one before is not enough:

1. tool inputs cut to 1,000 characters, then 200, then 60;
2. long texts abridged to head and tail, oldest first, the pinned messages last;
3. old messages collapsed to `[… N chars omitted …]`;
4. old calls shortened to one line each;
5. old messages without calls left out;
6. runs of old call-only messages folded into one entry.

The first message and the newest PINNED are never collapsed, shortened or left out. Where fast-jev gives up, this
leaves out the oldest of the rest, since System 1 must still be asked. Nothing writes the State: it is rebuilt from
the journal whenever System 1 is asked, for supervision, kept reads, memory recall and extensions (`ctx.state()`).
`cut_view` is the same cut as a structure, with how each message was kept, for the TUI's State panel.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from intern.evidence import DIAGNOSTIC_COMMAND, STATUS_LINE, extract
from intern.journal import Entry, MessageEntry
from intern.messages import AssistantMessage, JsonObject, ToolResult, ToolResultMessage, UserMessage
from intern.processes import RunningProcess

CONTEXT = (
    "A coding agent's session so far. `goal` is the user's latest requests. `history` is the whole session, oldest "
    "first: the user's messages, the harness's, the agent's replies (its reasoning is not shown) and its tool calls, "
    "each with its input and a short `result` note in place of the output. Long texts may be abridged and old calls "
    "shortened to one line."
)
PROCESSES_CONTEXT = (
    " `processes` lists the commands running now, oldest first: a tool call still running (`call`), a background "
    "job (`job`) or what a finished command left running (`left_running`), each with how long it has run and, for a "
    "job, how long it has written nothing (`quiet_s`) and its last line of output."
)
STATE_SHARE = 0.75  # of System 1's max_state_tokens; the rest is the room an ask's own text needs
INPUT_CHARS = (1000, 200, 60)
TEXT_HEAD = 400
TEXT_TAIL = 150
GOAL_PROMPTS = 3
GOAL_CHARS = 500
PINNED = 6  # the newest messages; a turn's tool results count as one, as in Anthropic's format
TALLY_CHARS = 120
HEAD_CHARS = 300  # fast-jev's truncateHeadChars

_PIECES = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")
_LINT_COMMAND = re.compile(r"(?:^|[;&|()\s'\"])(?:ruff|pyright|basedpyright|mypy|tsc|eslint)(?:\s|$)")
_TALLY = re.compile(  # pytest -q's tally (no rule around it), and linters' and type checkers'
    r"^(?:(?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?),? ?)+in [\d.]+s.*"
    r"|no tests ran in [\d.]+s.*|\d+ errors?, \d+ warnings?.*|Found \d+ errors?.*|All checks passed!"
    r"|Success: no issues found.*|\d+ files? (?:would be )?reformatted.*|\d+ files? already formatted"
    r"|. \d+ problems? \(.*\))$"
)
_QUOTED = re.compile(r"^(?:fatal|failure|target|summary|warning) L\d+: ", re.M)  # a reduced output's quotes


def estimate_tokens(text: str) -> int:
    """fast-jev's estimate, no tokenizer: a word costs a token per six letters, a digit half a token, any other
    symbol nine tenths (2-18% above the true count on its transcripts)."""
    tokens = 0.0
    for piece in _PIECES.findall(text):
        if piece[0].isdigit():
            tokens += len(piece) / 2
        elif piece[0].isascii() and piece[0].isalpha():
            tokens += 1 + (len(piece) - 1) // 6
        else:
            tokens += 0.9
    return -int(-tokens // 1)


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: max(0, limit - 1)]}…"


def _abridge(text: str) -> str:
    if len(text) <= TEXT_HEAD + TEXT_TAIL + 40:
        return text
    return f"{text[:TEXT_HEAD]}\n[… {len(text) - TEXT_HEAD - TEXT_TAIL} chars omitted …]\n{text[-TEXT_TAIL:]}"


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tally(command: str | None, result: ToolResult) -> str | None:
    """A check's tally (the tests', a linter's), passed or failed; None for anything that is not a check."""
    if command is None or not (DIAGNOSTIC_COMMAND.search(command) or _LINT_COMMAND.search(command)):
        return None
    body = _QUOTED.sub("", result.content)
    found = [e.quote for e in extract(body) if e.kind == "summary" and not STATUS_LINE.match(e.quote)]
    found += [line.strip() for line in body.splitlines() if _TALLY.match(line.strip())]
    return truncate(found[-1].strip("= "), TALLY_CHARS) if found else None


def note(command: str | None, result: ToolResult) -> str | None:
    """What a result's note adds to its status and size: a check's tally, then the output's first HEAD_CHARS (as
    fast-jev leaves a result it drops from the compacted session, whitespace folded)."""
    return _note(tally(command, result), result)


def _note(found_tally: str | None, result: ToolResult) -> str | None:
    found = [found_tally] if found_tally else []
    if result.content.strip():
        found.append("starts " + json.dumps(truncate(" ".join(result.content.split()), HEAD_CHARS), ensure_ascii=False))
    return "; ".join(found) or None


@dataclass
class _Call:
    id: str
    tool: str
    input: JsonObject
    error: bool
    chars: int
    outcome: str | None
    tally: str | None = None

    def note(self) -> str:
        note = f"{'error' if self.error else 'ok'}, {self.chars} chars (omitted)"
        return f"{note}: {self.outcome}" if self.outcome else note

    def line(self) -> str:
        """The call as one line, for when the structured form is too costly."""
        args = " ".join(
            f"{key}={' '.join((value if isinstance(value, str) else _dump(value)[:200]).split())}"
            for key, value in self.input.items()
        )
        line = (
            f"{self.id} {self.tool} {truncate(args, INPUT_CHARS[2])} → {'error' if self.error else 'ok'} {self.chars}ch"
        )
        return f"{line}: {self.outcome}" if self.outcome else line


@dataclass
class _Message:
    role: str  # user, harness, assistant, tool (a turn's results: never an entry, but counted for the pins)
    text: str
    calls: list[_Call] = field(default_factory=list)
    entry_id: str = ""


def _session(entries: Sequence[Entry], notes: bool) -> list[_Message]:
    results = {
        e.message.result.call_id: e.message.result
        for e in entries
        if isinstance(e, MessageEntry) and isinstance(e.message, ToolResultMessage)
    }
    messages: list[_Message] = []
    count = 0
    for entry in entries:
        if not isinstance(entry, MessageEntry):
            continue
        m = entry.message
        if isinstance(m, UserMessage):
            messages.append(_Message("user" if m.source == "user" else "harness", m.content, entry_id=entry.id))
        elif isinstance(m, ToolResultMessage):
            if messages and messages[-1].role != "tool":
                messages.append(_Message("tool", ""))
        elif isinstance(m, AssistantMessage):
            calls = []
            for c in m.tool_calls:
                if (r := results.get(c.id)) is None:
                    continue  # no result yet: nothing to say of it
                count += 1
                command = c.arguments.get("command") if c.name == "bash" else None
                checked = tally(command if isinstance(command, str) else None, r) if notes else None
                found = _note(checked, r) if notes else None
                error = r.status != "ok"
                calls.append(_Call(f"t{count}", c.name, dict(c.arguments), error, len(r.content), found, checked))
            messages.append(_Message("assistant", m.text, calls, entry.id))
    return messages


class _Fit:
    """The history being fitted, with each entry's tokens kept up to date."""

    def __init__(self, base: int, history: list[dict], limit: int) -> None:
        self.base, self.history, self.limit = base, history, limit
        self.per = [self.tokens_of(e) for e in history]

    @staticmethod
    def tokens_of(entry: dict) -> int:
        return estimate_tokens(_dump(entry)) + 1

    @property
    def tokens(self) -> int:
        return self.base + sum(self.per)

    def fits(self) -> bool:
        return self.tokens <= self.limit

    def shrink(self, index: int, change: Callable[[dict], None]) -> None:
        change(self.history[index])
        self.per[index] = self.tokens_of(self.history[index])


def session_cut(
    entries: Sequence[Entry], max_tokens: int, *, notes: bool = True, processes: Sequence[RunningProcess] = ()
) -> str:
    """The session on `entries` (a journal's active path) as System 1's State, within `max_tokens` (estimated), with
    the commands running now (`processes`, the session's job manager's `for_system1()`). `notes=False` gives results
    as fast-jev does, status and size only."""
    return _fitted(entries, max_tokens, notes, processes).text()


@dataclass
class _Cut:
    messages: list[_Message]
    goal: list[str]
    processes: Sequence[RunningProcess]
    history: list[dict]
    input_chars: int
    whole_tokens: int

    def text(self) -> str:
        if not self.processes:
            return _dump({"context": CONTEXT, "goal": "\n".join(self.goal), "history": self.history})
        running = [p.model_dump(exclude_none=True) for p in self.processes]
        context = CONTEXT + PROCESSES_CONTEXT
        return _dump({"context": context, "goal": "\n".join(self.goal), "processes": running, "history": self.history})


def _fitted(entries: Sequence[Entry], max_tokens: int, notes: bool, processes: Sequence[RunningProcess] = ()) -> _Cut:
    messages = _session(entries, notes)
    requests = [m.text for m in messages if m.role == "user" and m.text.strip()]
    goal = [truncate(text, GOAL_CHARS) for text in requests[-GOAL_PROMPTS:]]
    input_chars, whole = INPUT_CHARS[-1], 0

    # Like the goal, what runs is never cut to fit: the history is fitted around both.
    def state(history: list[dict]) -> _Cut:
        return _Cut(messages, goal, processes, history, input_chars, whole)

    def entries_of(input_chars: int) -> list[dict]:
        history = []
        for i, m in enumerate(messages):
            calls = [{"id": c.id, "tool": c.tool, "input": truncate(_dump(c.input), input_chars), "result": c.note()}
                     for c in m.calls]  # fmt: skip
            if not m.text.strip() and not calls:
                continue
            entry: dict = {"i": i, "role": m.role, "text": m.text}
            if calls:
                entry["tool_calls"] = calls
            history.append(entry)
        return history

    base = estimate_tokens(state([]).text())
    fit = _Fit(base, [], max_tokens)
    for input_chars in INPUT_CHARS:
        fit = _Fit(base, entries_of(input_chars), max_tokens)
        whole = whole or fit.tokens
        if fit.fits():
            return state(fit.history)

    def pinned(entry: dict) -> bool:
        return entry["i"] == 0 or entry["i"] >= len(messages) - PINNED

    history = fit.history
    order = [k for k, e in enumerate(history) if not pinned(e)] + [k for k, e in enumerate(history) if pinned(e)]
    for k in order:
        if len(history[k]["text"]) > TEXT_HEAD + TEXT_TAIL + 40:
            fit.shrink(k, lambda e: e.update(text=_abridge(e["text"])))
            if fit.fits():
                return state(history)
    for k in order:
        if not pinned(history[k]) and history[k]["text"]:
            chars = len(messages[history[k]["i"]].text)
            fit.shrink(k, lambda e, n=chars: e.update(text=f"[… {n} chars omitted …]"))
            if fit.fits():
                return state(history)
    for k in order:
        if not pinned(history[k]) and (calls := messages[history[k]["i"]].calls):
            fit.shrink(k, lambda e, own=calls: e.update(tool_calls=[c.line() for c in own]))
            if fit.fits():
                return state(history)
    left: set[int] = set()
    for k in order:
        if not pinned(history[k]) and "tool_calls" not in history[k]:
            left.add(k)
            if base + sum(n for i, n in enumerate(fit.per) if i not in left) <= max_tokens:
                return state([e for i, e in enumerate(history) if i not in left])

    def foldable(e: dict) -> bool:
        return not pinned(e) and not e["text"] and isinstance((e.get("tool_calls") or [None])[0], str)

    merged: list[dict] = []
    for entry in (e for i, e in enumerate(history) if i not in left):
        if merged and foldable(merged[-1]) and foldable(entry) and merged[-1]["role"] == entry["role"]:
            merged[-1]["tool_calls"] = [*merged[-1]["tool_calls"], *entry["tool_calls"]]
        else:
            merged.append(entry)
    fit = _Fit(base, merged, max_tokens)
    if fit.fits():
        return state(merged)
    # Past fast-jev's last stage, where it throws: the oldest of the rest left out, a note in their place.
    fit.base += _Fit.tokens_of(_left_out(len(merged)))
    at, gone = None, 0
    while not fit.fits() and (k := next((k for k, e in enumerate(merged) if not pinned(e)), None)) is not None:
        del merged[k], fit.per[k]
        at, gone = k, gone + 1
    if at is not None:
        merged.insert(at, _left_out(gone))
    return state(merged)


def _left_out(count: int) -> dict:
    return {"role": "harness", "text": f"[… {count} earlier entries left out …]"}


Kept = Literal["whole", "abridged", "collapsed", "line", "left_out"]
"""How System 1 reads a message, from most to least: as it is, its text abridged to head and tail, its text
collapsed to the omission note, its calls shortened to one line, or not at all."""


class CutCall(BaseModel):
    id: str = Field(description="`t1`, `t2`, ... in the order the calls were made, as the cut names them.")
    tool: str
    input: JsonObject
    error: bool
    chars: int = Field(description="The output's length; the cut has a note in its place.")
    tally: str | None = Field(default=None, description="A check's tally (tests, a linter), passed or failed.")


class CutMessage(BaseModel):
    index: int = Field(description="Its place in the session, the history's `i`.")
    entry_id: str
    role: Literal["user", "harness", "assistant"]
    text: str = Field(description="As System 1 reads it: whole, abridged, the omission note; empty when left out.")
    chars: int = Field(description="The text's length in the session.")
    calls: list[CutCall]
    kept: Kept
    pinned: bool = Field(description="The first message or one of the newest PINNED: never collapsed or left out.")


class CutView(BaseModel):
    """System 1's State as the TUI shows it: the same cut as `session_cut`, message by message."""

    text: str = Field(description="What System 1 reads, exactly `session_cut`'s.")
    goal: list[str] = Field(description="The user's latest requests the goal is made of, oldest first.")
    messages: list[CutMessage] = Field(description="Every message of the session that has text or calls, in order.")
    tokens: int = Field(description="`text`, estimated as the budget is.")
    budget: int
    input_chars: int = Field(
        description="What each tool input was cut to: INPUT_CHARS' first unless it had to be less."
    )
    whole_tokens: int = Field(description="The history with every message whole: what the cut wore down from.")
    processes: list[RunningProcess] = Field(default_factory=list, description="The commands running when it was cut.")


def cut_view(entries: Sequence[Entry], max_tokens: int, processes: Sequence[RunningProcess] = ()) -> CutView:
    """`session_cut(entries, max_tokens, processes=processes)`, with how each message was kept."""
    cut = _fitted(entries, max_tokens, True, processes)
    text = cut.text()
    present = {e["i"]: e for e in cut.history if "i" in e}
    calls = [c for e in cut.history for c in e.get("tool_calls", [])]
    shown = {c["id"] for c in calls if isinstance(c, dict)}
    lined = {c.split(" ", 1)[0] for c in calls if isinstance(c, str)}
    last = len(cut.messages) - PINNED
    messages: list[CutMessage] = []
    for i, m in enumerate(cut.messages):
        if m.role == "tool" or (not m.text.strip() and not m.calls):
            continue
        ids = {c.id for c in m.calls}
        entry = present.get(i)
        kept_text = entry["text"] if entry is not None else ""
        kept: Kept
        if ids & lined:
            kept = "line"
        elif entry is None and not ids & shown:
            kept = "left_out"
        elif kept_text == m.text:
            kept = "whole"
        elif kept_text == f"[… {len(m.text)} chars omitted …]":
            kept = "collapsed"
        else:
            kept = "abridged"
        found = [CutCall(id=c.id, tool=c.tool, input=c.input, error=c.error, chars=c.chars, tally=c.tally)
                 for c in m.calls]  # fmt: skip
        role = "harness" if m.role == "harness" else "user" if m.role == "user" else "assistant"
        messages.append(
            CutMessage(index=i, entry_id=m.entry_id, role=role, text=kept_text, chars=len(m.text), calls=found,
                       kept=kept, pinned=i == 0 or i >= last)
        )  # fmt: skip
    return CutView(
        text=text, goal=cut.goal, messages=messages, tokens=estimate_tokens(text), budget=max_tokens,
        input_chars=cut.input_chars, whole_tokens=cut.whole_tokens, processes=list(cut.processes),
    )  # fmt: skip
