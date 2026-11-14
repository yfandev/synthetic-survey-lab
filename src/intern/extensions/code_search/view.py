"""`decision_code` in the terminal UI: the search drawn live right under the call, top-down, each score as System 1
gives it.

    decision_code where are malformed tool call arguments repaired?
    ▌ ⌕ System 1 walks the index                                                       ⠹ 2.0s
    ▌ ● descend 1.3s  ──  ⠹ parts  ──  ○ read  ──  ○ compare  ──  ○ ranked
    ▌ judging every part of the kept files by its card
    ▌
    ▌ ◉ src/intern               109 files      LEADERBOARD
    ▌ ├─ ◆ providers       ━━━┿━━━━━━━━━── 0.85 ✓       1 • providers/repair.py  ━━━━━─ 0.90
    ▌ │  ├─ ■ repair.py    ━━━┿━━━━━━━━━━─ 0.90 ✓       2 ▼ providers/fake.py    ━──── 0.30
    ▌ │  │  ├─ · def parse_arguments( ━━━━━━━━━━━━─ 0.87
    ▌ │  │  └─ … 8 more parts, best 0.21                 SYSTEM 1
    ▌ ├─ ◇ files 3/4, protocol ━━━┼─────────── 0.05 ✗   asks 62 · judged 191
    ▌ ┿ cut (best × 0.25)  ✓ kept  ✗ pruned  ⊘ budget  • read  ★ result
    ▌ ▾ fold   ▾ 54 more rows   latest 0.11 async def MemoryActions.t…
    running…

The panel fills from the code-search extension's `search_trace` events (`locate/trace.py`): the tree grows as
nodes are judged, each bar fills to its score, the tick in a bar is the survivor rule's cut among its siblings
(best x gap), pruned branches dim, the parts whose text was read show the text's score, the top files'
side-by-side votes come in one order at a time, and the ranking closes it. It stays in the transcript, between
the call and its output, and the transcript follows it as it grows; its control row folds it to one line,
shows the rows past the first screenful, and replays a finished search.

This module is the extension's view (`CodeSearch.tui_view`): the TUI hosts it and holds nothing of it. The
state and the drawing are plain functions of the trace; the panel only shows them.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static

from intern.events import Event, ExtensionEvent, ToolCallEnd, ToolCallStart
from intern.locate.search import GAP
from intern.locate.trace import KIND, PHASES, Phase, Ranked, SearchNode, SearchTrace, Vote
from intern.tui.theme import AMBER, GREEN, INK, INK_SOFT, MUTED, RED, RULE
from intern.tui.views import ExtensionView, ViewHost

TOOL = "decision_code"
FPS = 12
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
FILLED, HALF, TRACK = "━", "╸", "─"
SPARK = "▁▂▃▄▅▆▇█"
FLASH_S = 0.7
"""How long a node that just changed stays highlighted."""
MOVE_S = 2.0
"""How long a leaderboard move shows its arrow."""
EASE = 0.35
"""The share of the way a bar moves toward its score each frame."""
BAR = 14
READ_W = 12
PARTS_SHOWN = 4
PRUNED_SHOWN = 3
TOP = 8
FEED = 5
REPLAY_SPEED = 2.0
MAX_ROWS = 18
LEGEND = f"┿ cut (best × {GAP:g})  ✓ kept  ✗ pruned  ⊘ budget  • read  ★ result"
"""Tree rows shown until the panel's `more rows` is clicked."""
NARROW = 110
"""Below this width the panel stacks the tree over the leaderboard instead of beside it."""
PHASE_NAMES: dict[Phase, str] = {
    "descend": "descend",
    "parts": "parts",
    "read": "read",
    "compare": "compare",
    "done": "ranked",
}
PHASE_HINTS: dict[Phase, str] = {
    "descend": "judging folders and groups of files, best first, then the files of the groups kept",
    "parts": "judging every part of the kept files by its card",
    "read": "reading the text of the best parts",
    "compare": "the top files side by side, asked in three orders",
    "done": "ranked",
}

Part = tuple[str, str]


# -- the state ----------------------------------------------------------------------------------------------------


@dataclass
class SearchState:
    """One `decision_code` call and its search, merged from the traces as they come."""

    call_id: str
    question: str
    scope: str | None = None
    search: str | None = None
    """The trace's id, once a trace is bound to the call."""
    phase: Phase | None = None
    """None until the first trace: the index may be refreshing."""
    nodes: dict[str, SearchNode] = field(default_factory=dict)
    children: dict[str | None, list[str]] = field(default_factory=dict)
    last: SearchTrace | None = None
    votes: list[Vote] = field(default_factory=list)
    ranking: list[Ranked] = field(default_factory=list)
    finished: bool = False
    status: str | None = None
    """How the call ended: `ok`, `error`, `refused`; None while it runs."""
    error: str = ""
    history: list[SearchTrace] = field(default_factory=list)
    phase_started: dict[Phase, int] = field(default_factory=dict)
    changed_at: dict[str, float] = field(default_factory=dict)
    shown: dict[str, float] = field(default_factory=dict)
    """Each bar as drawn, easing toward its score."""
    moving: set[str] = field(default_factory=set)
    feed: deque[str] = field(default_factory=lambda: deque(maxlen=FEED))
    order: list[str] = field(default_factory=list)
    """The leaderboard as last drawn, file ids best first."""
    moves: dict[str, tuple[int, float]] = field(default_factory=dict)

    def merge(self, trace: SearchTrace, now: float) -> None:
        self.search = trace.search
        if trace.phase not in self.phase_started:
            self.phase_started[trace.phase] = trace.elapsed_ms
        self.phase = trace.phase
        for node in trace.nodes:
            old = self.nodes.get(node.id)
            if old is None:
                self.children.setdefault(node.parent, []).append(node.id)
            self.nodes[node.id] = node
            self.changed_at[node.id] = now
            if node.score is not None and (old is None or old.score != node.score):
                self.moving.add(node.id)
                self.feed.appendleft(node.id)
            elif node.read is not None and (old is None or old.read is None):
                self.feed.appendleft(node.id)
        self.votes += trace.votes
        if trace.finished:
            self.ranking = trace.ranking
            self.finished = True
        self.last = trace
        self.history.append(trace)
        self._rank(now)

    def _rank(self, now: float) -> None:
        order = [n.id for n in self.files()[:TOP]]
        if order == self.order:
            return
        before = {nid: i for i, nid in enumerate(self.order)}
        for i, nid in enumerate(order):
            was = before.get(nid)
            if was is None:
                self.moves[nid] = (TOP, now)  # new in the top
            elif was != i:
                self.moves[nid] = (was - i, now)
        self.order = order

    def ended(self, status: str, content: str) -> None:
        self.status = status
        if status != "ok":
            self.error = content

    def ease(self) -> bool:
        """Move each bar a step toward its score; True while any still moves."""
        for nid in list(self.moving):
            node = self.nodes[nid]
            target = node.score or 0.0
            now = self.shown.get(nid, 0.0)
            if abs(target - now) < 0.004:
                self.shown[nid] = target
                self.moving.discard(nid)
            else:
                self.shown[nid] = now + (target - now) * EASE
        return bool(self.moving)

    def files(self) -> list[SearchNode]:
        judged = [n for n in self.nodes.values() if n.kind == "file" and n.score is not None]
        return sorted(judged, key=lambda n: -value(n))

    def count(self, *kinds: str) -> int:
        return sum(1 for n in self.nodes.values() if n.kind in kinds)

    @property
    def elapsed_ms(self) -> int:
        return self.last.elapsed_ms if self.last is not None else 0

    @property
    def live(self) -> bool:
        return not self.finished and self.status is None


def value(node: SearchNode) -> float:
    """A node's place among its siblings: its score in the ranking once there is one, else its card's."""
    return node.final if node.final is not None else (node.score or 0.0)


# -- drawing --------------------------------------------------------------------------------------------------


def runs(parts: list[Part]) -> list[Part]:
    """Neighbouring parts of one style as one."""
    out: list[Part] = []
    for text, style in parts:
        if out and out[-1][1] == style:
            out[-1] = (out[-1][0] + text, style)
        else:
            out.append((text, style))
    return out


def bar(p: float, width: int, fill: str, cut: float | None = None) -> list[Part]:
    """A line that fills to `p` in half cells over a light track, as the TUI's meters do; `cut`, the survivor
    rule's bar among the node's siblings, as a tick across it."""
    halves = round(min(max(p, 0.0), 1.0) * width * 2)
    full, half = divmod(halves, 2)
    tick = None if cut is None else min(width - 1, int(min(max(cut, 0.0), 1.0) * width))
    parts: list[Part] = []
    for i in range(width):
        if i == tick:
            parts.append(("┿" if i < full else "┼", f"bold {AMBER}"))
        elif i < full:
            parts.append((FILLED, fill))
        elif i == full and half:
            parts.append((HALF, fill))
        else:
            parts.append((TRACK, RULE))
    return runs(parts)


def shimmer(width: int, frame: int) -> list[Part]:
    """An empty track with a dash running along it: the card is with System 1."""
    at = frame % (width + 3) - 2
    return runs([(FILLED, AMBER) if at <= i < at + 3 else (TRACK, RULE) for i in range(width)])


def fit(text: str, width: int, *, keep_end: bool = False) -> str:
    if width <= 1:
        return text[:width]
    if len(text) <= width:
        return text
    return "…" + text[-(width - 1) :] if keep_end else text[: width - 1] + "…"


def folder_of(state: SearchState, nid: str | None) -> str:
    """The folder the nearest tree node above `nid` stands for ('' at the top)."""
    while nid is not None:
        node = state.nodes.get(nid)
        if node is None:
            return ""
        if node.kind in ("root", "folder", "group"):
            head = node.label.partition(" · ")[0]
            return "" if head == "." else head
        nid = node.parent
    return ""


def short_label(state: SearchState, node: SearchNode) -> str:
    """A node's label without what its parents already say: `retry.py` under `providers`, a group by its
    members."""
    if node.kind in ("part", "root"):
        return node.label
    base = folder_of(state, node.parent)
    if node.kind == "file":
        return node.label[len(base) + 1 :] if base and node.label.startswith(base + "/") else node.label
    head, sep, rest = node.label.partition(" · ")
    rest = rest.strip("[]").replace("(files)", "files").replace("(files ", "files ").replace(")", "")
    if head in (base, "."):
        return rest or head
    if base and head.startswith(base + "/"):
        head = head[len(base) + 1 :]
    return head + ("/" if node.kind == "folder" else "") + (f" · {rest}" if sep else "")


def glyph(node: SearchNode, opened: bool) -> Part:
    match node.kind:
        case "root":
            return ("◉", INK)
        case "folder":
            return ("▾" if opened else "▸", INK)
        case "group":
            return ("◆" if opened else "◇", INK)
        case "file":
            return ("■" if opened else "□", GREEN if node.state == "chosen" else INK)
        case _:
            if node.state == "chosen":
                return ("★", GREEN)
            return ("•" if node.read is not None else "·", INK if node.read is not None else MUTED)


def mark(node: SearchNode, frame: int) -> Part:
    match node.state:
        case "asking" | "reading":
            return (SPINNER[frame % len(SPINNER)], AMBER)
        case "chosen":
            return ("★", f"bold {GREEN}")
        case "kept" | "opened" | "read":
            return ("✓", GREEN)
        case "pruned":
            return ("✗", MUTED)
        case "skipped":
            return ("⊘", AMBER)
        case _:
            return (" ", "")


def fill_of(node: SearchNode) -> str:
    match node.state:
        case "chosen":
            return GREEN
        case "kept" | "opened" | "read" | "reading":
            return INK
        case "pruned":
            return MUTED
        case "skipped":
            return AMBER
        case _:
            return INK_SOFT


def label_style(node: SearchNode) -> str:
    match node.state:
        case "chosen":
            return f"bold {GREEN}"
        case "pruned" | "skipped":
            return MUTED
        case "kept" | "opened" | "read":
            return f"bold {INK}" if node.kind in ("folder", "group", "file") else INK
        case _:
            return INK


def visible(state: SearchState, nid: str, show_pruned: bool) -> tuple[list[SearchNode], str | None]:
    """The children drawn under `nid`, best first, and a line for the rest: a file's best parts (and those read
    or chosen), a node's live children and its best pruned ones."""
    kids = sorted((state.nodes[k] for k in state.children.get(nid, [])), key=lambda n: (n.score is None, -value(n)))
    if kids and kids[0].kind == "part":
        shown = kids[:PARTS_SHOWN] + [n for n in kids[PARTS_SHOWN:] if n.state in ("reading", "read", "chosen")]
        rest = [n for n in kids if n not in shown]
        what = "parts"
    else:
        live = [n for n in kids if n.state != "pruned"]
        dead = [n for n in kids if n.state == "pruned"]
        shown_dead = dead[:PRUNED_SHOWN] if show_pruned else []
        shown = live + shown_dead
        rest = dead[len(shown_dead) :]
        what = "pruned"
    if not rest:
        return shown, None
    best = max((value(n) for n in rest), default=0.0)
    return shown, f"… {len(rest)} more {what}, best {best:.2f}"


Row = tuple[str, SearchNode | None, str]
"""A tree row: the guides before it, its node, or a summary line when there is no node."""


def tree_rows(state: SearchState, show_pruned: bool) -> list[Row]:
    rows: list[Row] = []

    def walk(nid: str, lead: str, last: bool, top: bool) -> None:
        rows.append((lead + ("" if top else "└─ " if last else "├─ "), state.nodes[nid], ""))
        kids, summary = visible(state, nid, show_pruned)
        under = lead + ("" if top else "   " if last else "│  ")
        total = len(kids) + (1 if summary else 0)
        for i, kid in enumerate(kids):
            walk(kid.id, under, i == total - 1, False)
        if summary:
            rows.append((under + "└─ ", None, summary))

    for root in state.children.get(None, []):
        walk(root, "", True, True)
    return rows


def node_line(state: SearchState, lead: str, node: SearchNode, width: int, now: float, frame: int) -> Content:
    opened = bool(state.children.get(node.id))
    right = BAR + 5 + 2 + READ_W
    label_w = max(10, width - len(lead) - 2 - right - 1)
    style = label_style(node)
    if now - state.changed_at.get(node.id, 0.0) < FLASH_S:
        style = f"bold {AMBER}"  # it just changed: a pulse
    label = fit(short_label(state, node), label_w, keep_end=node.kind != "part")
    parts: list[Part] = [(lead, MUTED), glyph(node, opened), (" ", ""), (label.ljust(label_w), style), (" ", "")]
    if node.kind == "root":
        parts.append((node.detail, MUTED))
        return Content.assemble(*parts)
    if node.score is None:
        parts += [*shimmer(BAR, frame), ("   … ", MUTED)]
    else:
        parent = state.nodes.get(node.parent) if node.parent else None
        cut = parent.cut if parent is not None and node.kind in ("folder", "group") else None
        if node.kind == "file" and state.last is not None:
            cut = state.last.file_cut
        shown = state.shown.get(node.id, node.score)
        style_score = f"bold {INK}" if node.score >= 0.5 else (MUTED if node.state == "pruned" else INK)
        parts += [*bar(shown, BAR, fill_of(node), cut), (f" {node.score:.2f}", style_score)]
    parts += [(" ", ""), mark(node, frame)]
    if node.kind == "part" and node.state == "reading":
        parts.append((f"  {SPINNER[frame % len(SPINNER)]} reading", AMBER))
    elif node.kind == "part" and node.read is not None:
        parts += [("  text ", MUTED), (f"{node.read:.2f}", f"bold {INK}" if node.read >= 0.5 else INK)]
    elif node.kind == "file" and node.final is not None and state.phase not in (None, "descend", "parts"):
        parts += [("  → ", MUTED), (f"{node.final:.2f}", f"bold {GREEN}" if node.state == "chosen" else INK)]
    return Content.assemble(*parts)


def tree(
    state: SearchState, width: int, now: float, frame: int, show_pruned: bool, max_rows: int | None = None
) -> tuple[Content, int]:
    """The tree as drawn, and how many of its rows `max_rows` left out (the last ones: best first, they are the
    least likely)."""
    if not state.nodes:
        if state.status not in (None, "ok"):
            return Content.assemble((f"✗ {TOOL} did not search: ", RED), (state.error.strip()[:600], INK)), 0
        if state.status == "ok":
            return Content.assemble(("Nothing to search: the index is empty under this scope.", MUTED)), 0
        spin = SPINNER[frame % len(SPINNER)]
        return Content.assemble((f"{spin} ", AMBER), ("waiting for the index to be fresh…", MUTED)), 0
    rows = tree_rows(state, show_pruned)
    shown = rows if max_rows is None else rows[:max_rows]
    lines: list[Content] = []
    for lead, node, summary in shown:
        if node is None:
            lines.append(Content.assemble((lead, MUTED), (summary, f"italic {MUTED}")))
        else:
            lines.append(node_line(state, lead, node, width, now, frame))
    return Content("\n").join(lines), len(rows) - len(shown)


def arrow(state: SearchState, nid: str, now: float) -> Part:
    move = state.moves.get(nid)
    if move is None or now - move[1] > MOVE_S:
        return ("  ", "")
    delta = move[0]
    if delta >= TOP:
        return ("• ", AMBER)
    return ("▲ ", GREEN) if delta > 0 else ("▼ ", RED)


def leaderboard(state: SearchState, width: int, now: float) -> Content:
    if state.finished and state.ranking:
        lines: list[Content] = []
        name_w = max(8, width - 4 - 8 - 6)
        names = short_paths([r.path for r in state.ranking[:TOP]])
        for i, r in enumerate(state.ranking[:TOP], 1):
            node = state.nodes.get(f"f:{r.path}")
            chosen = node is not None and node.state == "chosen"
            style = f"bold {GREEN}" if chosen else INK
            name = fit(names[r.path], name_w, keep_end=True)
            lines.append(
                Content.assemble(
                    (f"{i:>2} ", MUTED),
                    (name.ljust(name_w), style),
                    (" ", ""),
                    *bar(r.score, 6, GREEN if chosen else INK),
                    (f" {r.score:.2f}", style),
                )
            )
            if r.where and chosen:  # where the result points, for the files it stands behind
                lines.append(Content.assemble(("    ↳ ", MUTED), (fit(r.where, width - 6), MUTED)))
        return Content("\n").join(lines)
    files = state.files()[:TOP]
    if not files:
        return Content.assemble(("No file judged yet: the descent is still in the folders.", MUTED))
    name_w = max(8, width - 5 - 8 - 7)
    names = short_paths([n.label for n in files])
    lines = []
    for i, node in enumerate(files, 1):
        v = value(node)
        name = fit(names[node.label], name_w, keep_end=True)
        lines.append(
            Content.assemble(
                (f"{i:>2} ", MUTED),
                arrow(state, node.id, now),
                (name.ljust(name_w), label_style(node)),
                (" ", ""),
                *bar(state.shown.get(node.id, v) if node.final is None else v, 6, fill_of(node)),
                (f" {v:.2f}", INK),
            )
        )
    return Content("\n").join(lines)


def compare(state: SearchState, width: int) -> Content:
    if not state.votes:
        if state.finished:
            return Content.assemble(("No side-by-side choice: fewer than two files.", MUTED))
        return Content.assemble(
            ("The top files side by side, once their text is read: one choice in three orders.", MUTED)
        )
    paths = state.votes[0].order
    n = len(state.votes)
    names = short_paths(paths)
    name_w = max(8, width - n * 5 - 12)
    head = [("".ljust(name_w + 1), ""), *[(f"#{k + 1}".ljust(5), MUTED) for k in range(n)], ("mean", MUTED)]
    lines = [Content.assemble(*head)]
    for path in paths:
        probs = [v.probabilities.get(path, 0.0) for v in state.votes]
        mean = sum(probs) / len(probs)
        parts: list[Part] = [(fit(names[path], name_w, keep_end=True).ljust(name_w), INK), (" ", "")]
        for p in probs:
            parts += [*bar(p, 4, INK), (" ", "")]
        parts += [*bar(mean, 6, GREEN if path == paths_best(state) else INK), (f" {mean:.2f}", f"bold {INK}")]
        lines.append(Content.assemble(*parts))
    return Content("\n").join(lines)


def paths_best(state: SearchState) -> str | None:
    """The file the votes favour, on average."""
    if not state.votes:
        return None
    paths = state.votes[0].order
    return max(paths, key=lambda p: sum(v.probabilities.get(p, 0.0) for v in state.votes))


def short_paths(paths: list[str]) -> dict[str, str]:
    """Paths without the folders they all share: `providers/retry.py` for `src/pkg/providers/retry.py`."""
    if len(paths) < 2:
        return {p: p for p in paths}
    split = [p.split("/")[:-1] for p in paths]
    common = 0
    while all(len(s) > common for s in split) and len({s[common] for s in split}) == 1:
        common += 1
    return {p: "/".join(p.split("/")[common:]) for p in paths}


def pips(done: int, total: int, width: int = 10) -> Part:
    lit = 0 if total <= 0 else min(width, round(done / total * width))
    return ("▰" * lit + "▱" * (width - lit), INK)


def histogram(state: SearchState) -> list[Part]:
    """The scores System 1 gave, in ten bins from 0 to 1."""
    bins = [0] * 10
    for node in state.nodes.values():
        if node.score is not None and node.kind != "root":
            bins[min(9, int(node.score * 10))] += 1
    top = max(bins) or 1
    return [
        (SPARK[0] if b == 0 else SPARK[max(1, round(b / top * (len(SPARK) - 1)))], INK_SOFT if b else RULE)
        for b in bins
    ]


def stats(state: SearchState) -> Content:
    t = state.last
    if t is None:
        return Content.assemble(("No ask yet.", MUTED))
    secs = t.elapsed_ms / 1000
    return Content("\n").join(
        [
            Content.assemble(
                ("folders ", MUTED),
                (f"{t.opened}/{t.opened_budget}".ljust(8), INK),
                pips(t.opened, t.opened_budget),
            ),
            Content.assemble(
                ("files   ", MUTED),
                (f"{t.files}/{t.files_budget}".ljust(8), INK),
                pips(t.files, t.files_budget),
                ("  ", ""),
                (f"≈${t.cost_usd:.4f}", INK),
                (f" · {secs:.1f}s", MUTED),
            ),
            Content.assemble(
                ("asks ", MUTED),
                (str(t.asks), f"bold {INK}"),
                (" · judged ", MUTED),
                (str(t.judged), INK),
            ),
            Content.assemble(("scores  0 ", MUTED), *histogram(state), (" 1", MUTED)),
        ]
    )


def header(state: SearchState, width: int, frame: int, replay: bool) -> Content:
    """The panel's top: what the search is doing and how long it has taken, its phases, what the phase does. The
    question is on the call's line just above."""
    spin = SPINNER[frame % len(SPINNER)]
    secs = state.elapsed_ms / 1000
    if state.finished:
        status: list[Part] = [("✓ ranked", f"bold {GREEN}"), (f" in {secs:.1f}s", MUTED)]
    elif state.status not in (None, "ok"):
        status = [(f"✗ {state.status}", f"bold {RED}")]
    elif state.phase is None:
        status = [(spin, AMBER), (" waiting for the index", MUTED)]
    else:
        status = [(spin, AMBER), (f" {secs:.1f}s", INK)]
    if replay:
        status = [(f"⟲ replay ×{REPLAY_SPEED:g}  ", f"bold {AMBER}"), *status]
    first: list[Part] = [("⌕ ", f"bold {AMBER}"), ("System 1 walks the index", f"bold {INK}")]
    if state.scope:
        first.append((f"  under {state.scope}", MUTED))
    used = sum(len(p[0]) for p in first) + sum(len(p[0]) for p in status)
    first += [(" " * max(2, width - used), ""), *status]
    second: list[Part] = []
    current = PHASES.index(state.phase) if state.phase is not None else -1
    for i, phase in enumerate(PHASES):
        if i:
            second.append(("  ──  ", RULE))
        started = state.phase_started.get(phase)
        if state.finished or i < current:
            if started is None:
                second.append(("– " + PHASE_NAMES[phase], MUTED))  # skipped: one file, no compare
                continue
            later = [state.phase_started[p] for p in PHASES[i + 1 :] if p in state.phase_started]
            took = ((later[0] if later else state.elapsed_ms) - started) / 1000
            second += [("● ", GREEN), (PHASE_NAMES[phase], INK)]
            if phase != "done":
                second.append((f" {took:.1f}s", MUTED))
        elif i == current:
            second += [(f"{spin} ", AMBER), (PHASE_NAMES[phase], f"bold {INK}")]
        else:
            second += [("○ ", MUTED), (PHASE_NAMES[phase], MUTED)]
    lines = [Content.assemble(*first), Content.assemble(*second)]
    if state.phase is not None and state.live:
        lines.append(Content.assemble((PHASE_HINTS[state.phase], f"italic {MUTED}")))
    return Content("\n").join(lines)


def feed(state: SearchState, width: int, frame: int) -> Content:
    parts: list[Part] = [("latest  ", MUTED)]
    for i, nid in enumerate(state.feed):
        node = state.nodes[nid]
        if i:
            parts.append(("   ", ""))
        v = node.read if node.kind == "part" and node.read is not None else node.score
        what = "text " if node.kind == "part" and node.read is not None else ""
        parts += [
            (f"{what}{v or 0.0:.2f} ", f"bold {INK}"),
            mark(node, frame),
            (" ", ""),
            (fit(short_label(state, node), 26, keep_end=node.kind != "part"), INK),
        ]
    used = sum(len(p[0]) for p in parts)
    if used > width:
        return Content.assemble(*parts).truncate(width, ellipsis=True)
    return Content.assemble(*parts)


def card(state: SearchState, frame: int) -> Content:
    """The search in one line: the folded panel."""
    head: list[Part] = [("⌕ ", f"bold {AMBER}")]
    if state.finished:
        secs = state.elapsed_ms / 1000
        t = state.last
        parts: list[Part] = [
            *head,
            ("search tree", f"bold {INK}"),
            (
                f"  {state.count('folder', 'group')} folders and groups, {state.count('file')} files, "
                f"{state.count('part')} parts judged in {secs:.1f}s · {t.asks if t else 0} asks",
                MUTED,
            ),
        ]
        if state.ranking:
            best = state.ranking[0]
            parts += [("  ★ ", f"bold {GREEN}"), (best.path, f"bold {GREEN}"), (f" {best.score:.2f}", GREEN)]
        return Content.assemble(*parts)
    if state.status not in (None, "ok"):
        return Content.assemble(
            *head, ("search tree", f"bold {INK}"), ("  the search did not run (see the output)", MUTED)
        )
    spin = SPINNER[frame % len(SPINNER)]
    phase = PHASE_NAMES[state.phase] if state.phase else "waiting for the index"
    parts = [*head, ("search tree ", f"bold {INK}"), (spin, AMBER), (f" {phase}", INK)]
    if state.last is not None:
        parts.append((f" · {state.last.asks} asks · {state.last.files} files judged", MUTED))
    files = state.files()
    if files:
        parts += [("  best so far ", MUTED), (files[0].label, INK), (f" {value(files[0]):.2f}", INK)]
    return Content.assemble(*parts)


# -- the panel, the view ------------------------------------------------------------------------------------------


@dataclass
class Replay:
    source: SearchState
    state: SearchState
    started: float
    at: int = 0


class Control(Static):
    """A word in the panel's control row that does something when clicked."""

    DEFAULT_CSS = """
    Control {
        width: auto;
        height: 1;
        margin-right: 3;
        color: $muted;
    }
    Control:hover {
        color: $foreground;
        text-style: underline;
    }
    """

    def __init__(self, label: str, run: Callable[[], None], classes: str) -> None:
        super().__init__(label, classes=classes)
        self.run = run

    def on_click(self, event: events.Click) -> None:
        event.stop()  # the call's block would fold or unfold its output
        self.run()


class SearchPanel(Vertical):
    """The search, live, right under its `decision_code` call in the transcript: the phases, the tree, the leaderboard
    and then the result, the compare votes, System 1's counts, and a row of controls (fold, replay, all rows)."""

    DEFAULT_CSS = """
    SearchPanel {
        height: auto;
        background: $surface;
        border-left: thick $warning;
        padding: 0 1;
    }
    SearchPanel .decision-head, SearchPanel .decision-tree, SearchPanel .decision-section, SearchPanel .decision-feed {
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    SearchPanel .decision-head {
        height: auto;
    }
    SearchPanel .decision-body {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    SearchPanel.-narrow .decision-body {
        layout: vertical;
    }
    SearchPanel .decision-tree {
        width: 3fr;
        height: auto;
    }
    SearchPanel .decision-side {
        width: 2fr;
        min-width: 40;
        height: auto;
        margin-left: 3;
    }
    SearchPanel.-narrow .decision-tree, SearchPanel.-narrow .decision-side {
        width: 1fr;
        margin-left: 0;
    }
    SearchPanel.-narrow .decision-side {
        margin-top: 1;
    }
    SearchPanel .decision-section {
        height: auto;
        margin-bottom: 1;
    }
    SearchPanel .decision-controls {
        height: 1;
    }
    SearchPanel .decision-feed {
        width: 1fr;
        height: 1;
    }
    SearchPanel.-folded .decision-body, SearchPanel.-folded .decision-feed {
        display: none;
    }
    """

    def __init__(self, host: ViewHost, state: SearchState) -> None:
        super().__init__(classes="decision-panel")
        self.host = host
        self.state = state
        self.frame = 0
        self.folded = False
        self.all_rows = False
        self.replay: Replay | None = None
        self.dirty = True
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Static(classes="decision-head")
        with Horizontal(classes="decision-body"):
            yield Static(classes="decision-tree")
            with Vertical(classes="decision-side"):
                yield Static(classes="decision-section decision-board")
                yield Static(classes="decision-section decision-compare")
                yield Static(classes="decision-section decision-stats")
        with Horizontal(classes="decision-controls"):
            yield Control("▾ fold", self.toggle_fold, classes="decision-fold")
            yield Control("⟲ replay", self.start_replay, classes="decision-replay")
            yield Control("", self.toggle_rows, classes="decision-rows")
            yield Static(classes="decision-feed")

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / FPS, self.tick)
        self.draw(time.monotonic())

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < NARROW, "-narrow")
        self.wake()

    def on_click(self, event: events.Click) -> None:
        event.stop()  # a click in the panel is not one on the call's block

    @property
    def shown(self) -> SearchState:
        return self.replay.state if self.replay is not None else self.state

    def wake(self) -> None:
        """Something changed: draw on the next frame, and keep the frames coming while it moves."""
        self.dirty = True
        if self._timer is not None:
            self._timer.resume()

    def tick(self) -> None:
        self.frame += 1
        now = time.monotonic()
        if (replay := self.replay) is not None:
            due = (now - replay.started) * 1000 * REPLAY_SPEED
            history = replay.source.history
            while replay.at < len(history) and history[replay.at].elapsed_ms <= due:
                replay.state.merge(history[replay.at], now)
                replay.at += 1
            if replay.at >= len(history) and not replay.state.moving:
                self.replay = None
            self.dirty = True
        state = self.shown
        moving = state.ease()
        lively = (
            state.live
            or moving
            or any(now - t < FLASH_S for t in state.changed_at.values())
            or any(now - t < MOVE_S for _, t in state.moves.values())
        )
        if self.dirty or lively:
            self.dirty = False
            self.draw(now)
        if not lively and not self.dirty and self.replay is None and self._timer is not None:
            self._timer.pause()  # settled: no frames until something changes

    def draw(self, now: float) -> None:
        state = self.shown
        head = self.query_one(".decision-head", Static)
        replay = self.query_one(".decision-replay", Control)
        rows = self.query_one(".decision-rows", Control)
        fold = self.query_one(".decision-fold", Control)
        fold.update("▸ unfold" if self.folded else "▾ fold")
        replay.display = state.finished and not self.folded
        if self.folded:
            head.update(card(state, self.frame))
            rows.display = False
            return
        width = self.content_size.width or 100
        side = self.query_one(".decision-board", Static).content_size.width or 44
        tree_w = self.query_one(".decision-tree", Static).content_size.width or 70
        head.update(header(state, width, self.frame, self.replay is not None))
        rows_room = max(MAX_ROWS, self.query_one(".decision-side").size.height - 1)  # as tall as the column beside it
        drawn, hidden = tree(state, tree_w, now, self.frame, True, None if self.all_rows else rows_room)
        if state.nodes:
            drawn = Content("\n").join([drawn, Content.assemble((LEGEND, f"italic {MUTED}"))])
        self.query_one(".decision-tree", Static).update(drawn)
        rows.display = hidden > 0 or self.all_rows
        rows.update("▴ fewer rows" if self.all_rows else f"▾ {hidden} more rows")
        board = "RESULT" if state.finished and state.ranking else "LEADERBOARD"
        self.query_one(".decision-board", Static).update(section(board, leaderboard(state, side, now)))
        self.query_one(".decision-compare", Static).update(section("COMPARE", compare(state, side)))
        self.query_one(".decision-stats", Static).update(section("SYSTEM 1", stats(state)))
        self.query_one(".decision-feed", Static).update(feed(state, max(20, width - 40), self.frame))

    def now(self) -> None:
        """Draw at once, for a click: the next frame may be a while away."""
        self.draw(time.monotonic())
        self.wake()

    def toggle_fold(self) -> None:
        self.folded = not self.folded
        self.set_class(self.folded, "-folded")
        self.now()

    def toggle_rows(self) -> None:
        self.all_rows = not self.all_rows
        self.now()

    def start_replay(self) -> None:
        source = self.state
        if not source.finished or not source.history:
            return
        fresh = SearchState(call_id=source.call_id, question=source.question, scope=source.scope)
        fresh.status = source.status
        self.replay = Replay(source=source, state=fresh, started=time.monotonic())
        self.now()


def section(title: str, body: Content) -> Content:
    return Content("\n").join([Content.assemble((title, f"bold {MUTED}")), body])


class DecisionCodeView(ExtensionView):
    """The code-search extension's view: a live panel on its search under each `decision_code` call."""

    def __init__(self, host: ViewHost, extension: str) -> None:
        super().__init__(host, extension)
        self.searches: list[SearchState] = []
        self.panels: dict[str, SearchPanel] = {}

    async def on_event(self, event: Event) -> None:
        match event:
            case ToolCallStart(call_id=call_id, arguments=arguments) if event.name == TOOL:
                scope = arguments.get("scope")
                state = SearchState(
                    call_id=call_id,
                    question=str(arguments.get("question", "")),
                    scope=scope if isinstance(scope, str) and scope else None,
                )
                self.searches.append(state)
                self.panels[call_id] = SearchPanel(self.host, state)
                await self.host.attach(call_id, self.panels[call_id])
            case ExtensionEvent(kind=kind, data=data) if event.extension == self.extension and kind == KIND:
                trace = SearchTrace.model_validate(data)
                state = self.bind(trace)
                if state is None:
                    return
                state.merge(trace, time.monotonic())
                self.panels[state.call_id].wake()
            case ToolCallEnd(call_id=call_id, status=status, content=content) if call_id in self.panels:
                self.panels[call_id].state.ended(status, content)
                self.panels[call_id].wake()
            case _:
                pass

    def bind(self, trace: SearchTrace) -> SearchState | None:
        """The call a trace belongs to: the one bound to its search, else the oldest running call asked the same
        question that has none yet (searches in one turn run side by side), else the oldest without one."""
        for state in self.searches:
            if state.search == trace.search:
                return state
        free = [s for s in self.searches if s.search is None and s.status is None]
        same = [s for s in free if s.question == trace.question]
        return (same or free or [None])[0]
