"""The State panel: System 1's State, the session cut it reads (`session_cut.py`), drawn.

System 1 is asked about the session as one text, rebuilt from the journal for every ask and fitted to its budget:
the user's latest requests as the goal, then the history, oldest first, each message whole, abridged, collapsed,
its calls shortened to one line, or left out. The panel draws that text as the reader would want it:

- the strip, one row for the whole session, oldest left: each cell a stretch of messages, as tall as System 1
  reads them (whole is a full block, left out the floor), so where the cut wore the session down shows at a
  glance. Until the cut wears something down, every message is read whole and the legend says so: there is no strip;
- the tokens the cut takes against its budget, what the history would take uncut, and a legend that counts each
  way a message was kept;
- the goal, a shaded block per request;
- the history: a row per prompt, reply and call (`✓`/`✗`, the tool, what it was about, a check's tally in green
  or red), a rail on the left in the strip's glyph for how the message was kept, runs left out folded into one
  row, and the point System 2's own context starts after a compaction. It follows the end as the session grows.

System 2's summary, the State it writes at a compaction and continues from, is not System 1's and is not here: the
compaction's card in the transcript shows it. `/state` opens System 1's State in full.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from rich.cells import cell_len, set_cell_size
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from intern.journal import CompactionEntry, Entry, MessageEntry
from intern.messages import AssistantMessage, UserMessage
from intern.processes import duration_text
from intern.session_cut import INPUT_CHARS, CutCall, CutMessage, CutView, Kept
from intern.tui.theme import (
    S_BAD,
    S_HEAD,
    S_INK,
    S_MUTED,
    S_MUTED_ITALIC,
    S_OK,
    S_RULE,
    S_TAB,
    S_WARN,
)
from intern.tui.widgets import (
    Following,
    Part,
    fit_path,
    meter,
    short_path,
    tokens_text,
)

KEPT_ORDER: tuple[Kept, ...] = ("whole", "abridged", "collapsed", "line", "left_out")
KEPT_GLYPH: dict[Kept, str] = {"whole": "█", "abridged": "▆", "collapsed": "▄", "line": "▂", "left_out": "▁"}
KEPT_LABEL: dict[Kept, str] = {
    "whole": "whole",
    "abridged": "abridged",
    "collapsed": "collapsed",
    "line": "one line",
    "left_out": "left out",
}
LEVEL: dict[Kept, int] = {"whole": 8, "abridged": 6, "collapsed": 4, "line": 2, "left_out": 0}
"""In eighths of the strip's row: the legend's glyphs."""
EIGHTHS = " ▁▂▃▄▅▆▇█"
HISTORY_ROWS = 400
"""The newest rows drawn; a long session's older ones are in `/state`."""
MIN_WIDTH = 24
GOAL_HEIGHT = 22
"""The panel's height under which the goal is not drawn: its requests are in the history too, which needs the room."""
FAILED = re.compile(
    r"\b[1-9]\d* (?:failed|errors?|problems?)\b|\bFound [1-9]|\b[1-9]\d* files? would be reformatted", re.I
)
OMITTED = re.compile(r"^\[… (\d+) chars omitted …\]$")
CD = re.compile(r"^cd \S+ && ")
EMPTY = "System 1's State is the session as it reads it: it shows here from the first prompt."


def clip(text: str, cells: int) -> str:
    """`text` in at most `cells` cells, `…` where it was cut."""
    if cell_len(text) <= cells:
        return text
    return set_cell_size(text, max(0, cells - 1)) + "…" if cells > 0 else ""


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


# -- the strip and the legend -------------------------------------------------------------------------------


def strip(messages: Sequence[CutMessage], width: int) -> Content:
    """The strip: `width` cells over the session, each as tall as System 1 reads the messages
    in it, on average, in gray so it maps the session without outweighing it. None while every message is whole:
    it would be a solid bar that says only what the legend does."""
    n = len(messages)
    if all(m.kept == "whole" for m in messages):
        return Content("")
    cells: list[Part] = []
    for c in range(width):
        lo = c * n // width
        part = messages[lo : max(lo + 1, (c + 1) * n // width)]
        level = round(sum(LEVEL[m.kept] for m in part) / len(part))
        cells.append((EIGHTHS[level] if level else "▁", S_MUTED))  # left out: the floor
    return Content.assemble(*cells)


def legend(view: CutView, width: int) -> Content:
    """The tokens against the budget and a meter of them; then how many messages were kept each way."""
    used = f"{tokens_text(view.tokens)} of {tokens_text(view.budget)} tokens"
    whole = tokens_text(view.whole_tokens)
    uncut = f"  ·  {whole} uncut" if whole != tokens_text(view.tokens) else ""
    bar = max(4, width - cell_len(used + uncut) - 2)
    parts: list[Part] = [(used, S_INK), (uncut, S_MUTED), "  ", *meter(view.tokens / max(view.budget, 1), S_INK, bar)]
    counts = Counter(m.kept for m in view.messages)
    items: list[list[Part]] = [
        [(KEPT_GLYPH[k], S_MUTED), f" {counts[k]} ", (KEPT_LABEL[k], S_MUTED)] for k in KEPT_ORDER if counts[k]
    ]
    if view.input_chars < INPUT_CHARS[0]:
        items.append([(f"inputs ≤ {view.input_chars} chars", S_MUTED)])
    line = 0  # the legend's items wrapped by hand: wrapping breaks at a non-breaking space too
    for item in items:
        size = Content.assemble(*item).cell_length
        if line and line + 3 + size > width:
            parts.append("\n")
            line = 0
        parts += (["   "] if line else ["\n"] if not parts or parts[-1] != "\n" else []) + item
        line += (3 if line else 0) + size
    return Content.assemble(*parts)


# -- the history --------------------------------------------------------------------------------------------


def subject(call: CutCall, cwd: Path | None) -> tuple[str, str]:
    """What a call was about, as (a muted lead, the rest): `$` and the command for bash, else its path or the
    argument that names what it is about."""
    arguments = call.input
    if isinstance(command := arguments.get("command"), str):
        return "$ ", CD.sub("", first_line(command))  # where it ran says little: the working tree, mostly
    if isinstance(pattern := arguments.get("pattern"), str):
        where = arguments.get("path")
        return "", f"{pattern} in {short_path(where, cwd)}" if isinstance(where, str) else pattern
    if isinstance(path := arguments.get("path"), str):
        return "", short_path(path, cwd)
    for key in ("query", "url", "question"):
        if isinstance(value := arguments.get(key), str):
            return "", first_line(value)
    return "", " ".join(f"{k}={v}" for k, v in arguments.items() if isinstance(v, str | int | float))


def tally_style(call: CutCall) -> str:
    return S_BAD if call.error or (call.tally and FAILED.search(call.tally)) else S_OK


def call_row(call: CutCall, width: int, cwd: Path | None, kept: Kept) -> list[Part]:
    glyph = ("✗", S_BAD) if call.error else ("✓", S_OK)
    lead, about = subject(call, cwd)
    tool = "" if lead else f"{call.tool} "  # a command is its own name
    right = clip(f"  {call.tally}", max(0, width // 2)) if call.tally else ""
    room = max(4, width - 2 - cell_len(tool) - cell_len(lead) - cell_len(right))
    about = fit_path(about, room) if "/" in about and not lead else clip(about, room)
    text_style = S_MUTED if kept == "line" else S_INK
    pad = " " * max(0, width - 2 - cell_len(tool) - cell_len(lead) - cell_len(about) - cell_len(right)) if right else ""
    return [glyph, " ", (tool, S_MUTED), (lead, S_MUTED), (about, text_style), pad, (right, tally_style(call))]


def message_rows(m: CutMessage, width: int, cwd: Path | None) -> list[list[Part]]:
    """A message's rows, without the rail: its text's first line, then its calls."""
    room = width - 2
    rows: list[list[Part]] = []
    if OMITTED.match(m.text):  # with calls, the rail says as much
        text: list[Part] = [] if m.calls else [(f"{m.chars:,} chars omitted", S_MUTED_ITALIC)]
    else:
        line = first_line(m.text)
        text = (
            [(clip(line, room), S_HEAD if m.role == "user" else S_WARN if m.role == "harness" else S_INK)]
            if line
            else []
        )
    lead: Part = ("› ", S_HEAD) if m.role == "user" else ("⚑ ", S_WARN) if m.role == "harness" else "  "
    if text:
        rows.append([lead, *text])
    elif m.role != "assistant":
        rows.append([lead])
    rows += [["  ", *call_row(call, room, cwd, m.kept)] for call in m.calls]
    return rows


def left_out_row(run: list[CutMessage]) -> list[Part]:
    calls = sum(len(m.calls) for m in run)
    said = f"{len(run)} message{'s' if len(run) != 1 else ''} left out"
    if calls:
        said += f" · {calls} call{'s' if calls != 1 else ''}"
    return [("  ┄ ", S_MUTED), (said, S_MUTED_ITALIC)]


def history(view: CutView, width: int, cwd: Path | None = None, system2_from: int | None = None) -> Content:
    """The history as rows, oldest first, each behind a rail in its message's strip glyph."""
    width = max(MIN_WIDTH, width)
    rows: list[list[Part]] = []

    def add(rail: Part, body: list[Part]) -> None:
        rows.append([rail, " ", *body])

    run: list[CutMessage] = []
    for m in [*view.messages, None]:
        if run and (m is None or m.kept != "left_out" or m.index == system2_from):
            add((KEPT_GLYPH["left_out"], S_MUTED), left_out_row(run))
            run = []
        if m is None:
            break
        if m.index == system2_from:
            said = "⟲ System 2 reads on from here "
            add(" ", [(said, S_MUTED), ("╌" * max(0, width - 2 - cell_len(said)), S_RULE)])
        if m.kept == "left_out":
            run.append(m)
            continue
        if m.role == "user" and rows:
            rows.append([])
        rail = (KEPT_GLYPH[m.kept], S_MUTED)
        for body in message_rows(m, width - 2, cwd):
            add(rail, body)
    if len(rows) > HISTORY_ROWS:
        hidden = len(rows) - HISTORY_ROWS
        rows = [[(f"… {hidden:,} earlier rows: /state has the whole of it", S_MUTED_ITALIC)], *rows[-HISTORY_ROWS:]]
    return Content("\n").join(Content.assemble(*row) for row in rows)


# -- the panel ----------------------------------------------------------------------------------------------


class StatePanel(Vertical, can_focus=False):
    """System 1's State, drawn (the module's docstring)."""

    def __init__(self, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.cut: CutView | None = None
        """System 1's State as last built, None before the session has any."""
        self.cwd: Path | None = None
        self._system2_from: str | None = None

    def compose(self) -> ComposeResult:
        # The "and" of Kahneman's title, where the two systems meet: what System 1 sees of System 2's work.
        yield Static(Content.assemble((" State ", S_TAB), ("  fast and slow", S_MUTED)), id="state-header")
        with Vertical(id="cut-top"):
            yield Static(id="cut-strip")
            yield Static(id="cut-legend")
            yield Static(id="cut-goal")
            yield Static(id="cut-running")
        with Following(id="cut-history"):
            yield Static(Content.styled(EMPTY, S_MUTED), id="cut-lines")

    # -- System 1's -------------------------------------------------------------------------------------------

    async def show_cut(self, cut: CutView, system2_from: str | None = None) -> None:
        """System 1's State as it now stands; `system2_from` the entry System 2's context starts at, after a
        compaction. A session with nothing said yet has none."""
        if not cut.messages:
            await self.clear()
            return
        self.cut, self._system2_from = cut, system2_from
        self._draw_cut()

    def on_resize(self) -> None:
        self._draw_cut()

    @property
    def _width(self) -> int:
        return max(MIN_WIDTH, self.content_size.width)

    def _draw_cut(self) -> None:
        cut = self.cut
        if cut is None:
            return
        scroll = self.query_one("#cut-history", Following)
        width = max(MIN_WIDTH, scroll.scrollable_content_region.width - 1)
        start = next((m.index for m in cut.messages if self._system2_from and m.entry_id == self._system2_from), None)
        drawn = strip(cut.messages, self._width)
        self.query_one("#cut-strip", Static).display = bool(drawn.plain)
        self.query_one("#cut-strip", Static).update(drawn)
        self.query_one("#cut-legend", Static).update(legend(cut, self._width))
        self._draw_goal()
        self._draw_running()
        self.query_one("#cut-lines", Static).update(history(cut, width, self.cwd, start))

    def _draw_goal(self) -> None:
        """The goal: the user's latest requests, a line each, newest last, shaded as the transcript's prompts."""
        goal = self.query_one("#cut-goal", Static)
        requests = self.cut.goal if self.cut else []
        goal.display = bool(requests) and self.size.height >= GOAL_HEIGHT
        room = self._width - 4
        lines = [Content.assemble(("› ", S_HEAD), clip(" ".join(text.split()), room - 2)) for text in requests]
        goal.update(Content("\n").join(lines))

    def _draw_running(self) -> None:
        """What runs as System 1 read it, a row each: the command, how long it has run and, for a job, how long it
        has been quiet."""
        running = self.query_one("#cut-running", Static)
        processes = self.cut.processes if self.cut else []
        running.display = bool(processes)
        room = self._width - 2
        rows = []
        for p in processes:
            took = duration_text(p.elapsed_s) + (f", quiet {duration_text(p.quiet_s)}" if p.quiet_s is not None else "")
            rows.append(
                Content.assemble(("⧗ ", S_MUTED), clip(p.command, max(room - len(took) - 4, 8)), (f"  {took}", S_MUTED))
            )
        running.update(Content("\n").join(rows))

    async def clear(self) -> None:
        """Back to none yet."""
        self.cut, self._system2_from = None, None
        self._draw_goal()
        for name in ("#cut-strip", "#cut-legend"):
            self.query_one(name, Static).update("")
        self.query_one("#cut-lines", Static).update(Content.styled(EMPTY, S_MUTED))
        self.query_one("#cut-history", Following).anchor()


def system2_start(entries: Sequence[Entry]) -> str | None:
    """Where System 2's own context starts, after the latest compaction: the first prompt, reply or harness message
    it keeps verbatim (the compaction names an entry, which may be a tool's result)."""
    kept = next((e.first_kept_id for e in reversed(entries) if isinstance(e, CompactionEntry)), None)
    at = next((i for i, e in enumerate(entries) if e.id == kept), None)
    if at is None:
        return None
    talk = (
        e for e in entries[at:] if isinstance(e, MessageEntry) and isinstance(e.message, UserMessage | AssistantMessage)
    )
    return next((e.id for e in talk), None)
