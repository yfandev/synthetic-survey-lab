"""`decision_files` in the terminal UI: the union search drawn live right under the call, both lanes at once, every
judgment as System 1 gives it, the hop through the seeds' terms, and the podium.

    decision_files When did the board approve the Zorblax merger, and who compiled it?
    ▌ ⌕ System 1 searches the workspace                                                   ✓ ranked in 2.1s
    ▌ ● walk ∥ bm25 1.2s  ──  ● judge 0.4s  ──  ● seeds → terms 0.0s  ──  ● judge 0.3s  ──  ● ranked
    ▌
    ▌ WALK  folders → files → parts                    BM25  top 100 of 146 passages
    ▌ ◉ .                              38 files          1   § board/minutes_march.md · Minutes  ━━━━━━ 0.92 ✦
    ▌ ├─ ◆ board, staff   ━━━━━┿━━━━━━─ 0.90 ✓           2 ◆ § staff/quill.md · Staff          ━━━━━─ 0.80 ✦
    ▌ │  ├─ ■ doc_0.md     ━━━━━┿━━━━━━─ 0.90 ✓           3   § finance/2023/doc_0.md · Sect…  ⠹ (asking)
    ▌ │  │  ├─ • Section 0  ━━━━━━━━━━━━ 0.90 ✓ text 0.58  … 95 more, best 0.98
    ▌ ╰─ 20 from the walk ──────────────────▶ ⚖ ◀──────────────── 100 from BM25 · 15 in both ─╯
    ▌ JUDGE  ▰▰▰▰▰▰▰▰▰▰ 150/150 passages  scores 0 ▇▁▁▁▁▁▁▁▂▃ 1
    ▌ SEEDS  ✦ ops/vendors/doc_2.md 0.98  ✦ ops/vendors/doc_4.md 0.94  ✦ board/doc_3.md 0.93  … 7 more
    ▌ HOP    20 terms → 45 new passages, 43 in files the walk left
    ▌   board/minutes_march.md ─┬─ ada quill  ▮▮▮▮▯▯ ──▶  2  ↯ staff/quill.md 0.80
    ▌                            └─ 3          ▮▯▯▯▯▯ ──▶ 15  ↯ legal/doc_3.md 0.80
    ▌   … 15 reached nothing new
    ▌
    ▌ RESULT  top 10, in diversity order                          SYSTEM 1
    ▌  1 ★ § ops/vendors/doc_2.md · Section 2  bm25      ━━━━━━ 0.98     asks 65 · judged 223
    ▌  2 ★ § board/minutes_march.md · Minutes  bm25      ━━━━━━ 0.92 ▲3  ≈$0.0123 · 2.1s
    ▌  5   § staff/quill.md · Staff            walk+bm25 ━━━━━─ 0.80 ▲4  walk files 32/120 ▰▰▰▱▱▱▱▱▱▱
    ▌  6   § finance/2023/doc_0.md · Section 2 ↯ 2       ━━━━━━ 0.96 ▼4  passages 146 · 3 by card
    ▌ ▾ fold   ⟲ replay   ▾ 144 more rows   latest 0.43 · doc_4.md · Section 0
    Places for: When did the board approve …

The panel fills from the workspace's `search_trace` events (`trace.py`). The walk's tree grows on the left as
decision_code's does (the same drawing, `code_search/view.py`), the BM25 lane fills its ranked list on the right, and
the two converge on System 1's judge: each passage's bar fills as its ask comes back (◆: in both lanes). The ten
best light up as seeds; the hop shows each seed's terms (by weight), the passages each term reached and whether
the walk had pruned their file (↯). The podium closes it: the top ten in diversity order, each with the lane
that brought it (walk, bm25, or the term), its kind (▤ page, § section, ⊞ tab, ▦ table, ◩ image, □ file), and how
far the diversity order moved it. It stays in the transcript between the call and its output, the transcript
follows it as it grows, and its control row folds it to one line, shows every row, and replays a finished search.

This module is the workspace extension's view (`Workspace.tui_view`); the state and the drawing are plain
functions of the trace, the panel only shows them, as decision_code's view does (whose pieces it uses).
"""

from __future__ import annotations

import bisect
import time
from collections import deque
from dataclasses import dataclass, field

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.timer import Timer
from textual.widgets import Static

from intern.events import Event, ExtensionEvent, ToolCallEnd, ToolCallStart
from intern.extensions.code_search.view import (
    EASE,
    FLASH_S,
    FPS,
    NARROW,
    REPLAY_SPEED,
    SPARK,
    SPINNER,
    Control,
    Part,
    SearchState,
    bar,
    fit,
    pips,
    section,
    shimmer,
    short_paths,
    tree,
)
from intern.extensions.workspace.trace import STAGES, Hit, Stage, Term, UnionTrace
from intern.locate.trace import KIND
from intern.tui.theme import AMBER, GREEN, INK, INK_SOFT, MUTED, RED, RULE
from intern.tui.views import ExtensionView, ViewHost

TOOL = "decision_files"
LANE_ROWS = 10
"""Rows of each lane, and of the hop, until the panel's `more rows` is clicked."""
HOP_ROWS = 6
SETTLED_ROWS = 5
"""Rows of each lane and of the hop once the search has ranked: the podium is what is left to read."""
TOP = 10
LIVE_TOP = 6
FEED = 4
STAGE_NAMES: dict[Stage, str] = {
    "walk": "walk ∥ bm25",
    "judge": "judge",
    "expand": "seeds → terms",
    "rejudge": "judge",
    "done": "ranked",
}
STAGE_HINTS: dict[Stage, str] = {
    "walk": "System 1 walks folders, files and parts; the BM25 table of every passage opens beside it",
    "judge": "the walk's passages and the BM25 top 100 judged as evidence, eight to an ask, the walk's best in view",
    "expand": "the ten best are seeds: their rarest words and names, each searched with the question",
    "rejudge": "what the terms reached, judged with the seeds in the state",
    "done": "ranked",
}
KINDS = {"page": "▤", "pages": "▤", "section": "§", "table": "▦", "image": "◩", "file": "□"}


# -- the state ----------------------------------------------------------------------------------------------------


@dataclass
class FilesState:
    """One `decision_files` call and its search, merged from the traces as they come (a stale or repeated one changes
    nothing: each hit and node keeps its newest version)."""

    call_id: str
    question: str
    scope: str | None = None
    search: str | None = None
    walk: SearchState = field(init=False)
    """The walk's tree, drawn as decision_code draws its own."""
    stage: Stage | None = None
    stage_started: dict[Stage, int] = field(default_factory=dict)
    hits: dict[str, Hit] = field(default_factory=dict)
    terms: list[Term] = field(default_factory=list)
    last: UnionTrace | None = None
    finished: bool = False
    status: str | None = None
    error: str = ""
    history: list[UnionTrace] = field(default_factory=list)
    """Every trace once, in the order sent."""
    seen: dict[str, int] = field(default_factory=dict)
    """Each hit's and node's newest version: the trace it came in."""
    changed_at: dict[str, float] = field(default_factory=dict)
    shown: dict[str, float] = field(default_factory=dict)
    moving: set[str] = field(default_factory=set)
    feed: deque[str] = field(default_factory=lambda: deque(maxlen=FEED))

    def __post_init__(self) -> None:
        self.walk = SearchState(call_id=self.call_id, question=self.question, scope=self.scope)

    def merge(self, trace: UnionTrace, now: float) -> None:
        if any(t.seq == trace.seq for t in self.history):
            return
        bisect.insort(self.history, trace, key=lambda t: t.seq)
        self.search = trace.search
        latest = self.last is None or trace.seq > self.last.seq
        nodes = [n for n in trace.nodes if self.seen.get(n.id, -1) < trace.seq]
        for node in nodes:
            self.seen[node.id] = trace.seq
        walk = self.walk
        keep = (walk.phase, walk.last)
        walk.merge(trace.model_copy(update={"nodes": nodes, "finished": False}), now)
        if not latest:
            walk.phase, walk.last = keep
        if trace.finished:
            walk.finished, walk.ranking = True, trace.ranking
        for hit in trace.hits:
            if self.seen.get("h:" + hit.id, -1) >= trace.seq:
                continue
            self.seen["h:" + hit.id] = trace.seq
            old = self.hits.get(hit.id)
            self.hits[hit.id] = hit
            self.changed_at[hit.id] = now
            if hit.score is not None and (old is None or old.score != hit.score):
                self.moving.add(hit.id)
                if hit.state == "judged":
                    self.feed.appendleft(hit.id)
        if trace.stage not in self.stage_started or trace.elapsed_ms < self.stage_started[trace.stage]:
            self.stage_started[trace.stage] = trace.elapsed_ms
        if trace.terms:
            self.terms = trace.terms
        if latest:
            self.stage, self.last = trace.stage, trace
        self.finished = self.finished or trace.finished

    def ended(self, status: str, content: str) -> None:
        self.status = status
        self.walk.ended(status, content)
        if status != "ok":
            self.error = content

    def ease(self) -> bool:
        """Move each bar a step toward its score; True while any still moves."""
        walking = self.walk.ease()
        for hid in list(self.moving):
            target = self.hits[hid].score or 0.0
            now = self.shown.get(hid, 0.0)
            if abs(target - now) < 0.004:
                self.shown[hid] = target
                self.moving.discard(hid)
            else:
                self.shown[hid] = now + (target - now) * EASE
        return walking or bool(self.moving)

    def top(self) -> list[Hit]:
        """The ranking: the top ten in diversity order once there is one, else the best judged so far."""
        ranked = sorted((h for h in self.hits.values() if h.rank is not None), key=lambda h: h.rank or 0)
        if ranked:
            return ranked
        judged = [h for h in self.hits.values() if h.score is not None]
        return sorted(judged, key=lambda h: -(h.score or 0.0))[:LIVE_TOP]

    def bm25(self) -> list[Hit]:
        return sorted((h for h in self.hits.values() if h.bm25 is not None), key=lambda h: h.bm25 or 0)

    def candidates(self) -> list[Hit]:
        """The passages System 1 judges (or is to): the lanes' and what the terms reached."""
        return [h for h in self.hits.values() if h.state != "card" or h.lanes != ["walk"]]

    def missed(self, hit: Hit) -> bool:
        """The walk did not keep its file: pruned, out of budget, or never judged."""
        node = self.walk.nodes.get(f"f:{hit.path}")
        return node is None or node.state in ("pruned", "skipped")

    @property
    def elapsed_ms(self) -> int:
        return self.last.elapsed_ms if self.last is not None else 0

    @property
    def live(self) -> bool:
        return not self.finished and self.status is None


# -- drawing --------------------------------------------------------------------------------------------------


def kind_of(hit: Hit) -> Part:
    if hit.kind == "table" and hit.title.startswith("tab "):
        return ("⊞", INK_SOFT)
    return (KINDS.get(hit.kind, "·"), INK_SOFT)


def name_of(hit: Hit, names: dict[str, str], width: int) -> str:
    """`folder/file.pdf · page 4`, the path cut from the left before the part's title is."""
    title = fit(hit.title, min(24, max(6, width // 3))) if hit.title else ""
    room = width - (len(title) + 3 if title else 0)
    return fit(names.get(hit.path, hit.path), max(4, room), keep_end=True) + (f" · {title}" if title else "")


def score_of(state: FilesState, hit: Hit, width: int, frame: int) -> list[Part]:
    """A hit's bar and score; a running dash while it is with System 1, a dot while it waits."""
    if hit.state == "asking":
        return [*shimmer(width, frame), (f" {SPINNER[frame % len(SPINNER)]}   ", AMBER)]
    if hit.score is None:
        return [(" " * width, ""), ("    ·", MUTED)]
    fill = GREEN if hit.rank is not None and hit.rank <= 3 else INK if hit.score >= 0.5 else INK_SOFT
    style = f"bold {INK}" if hit.score >= 0.5 else INK
    return [*bar(state.shown.get(hit.id, hit.score), width, fill), (f" {hit.score:.2f}", style)]


def flash(state: FilesState, hid: str, now: float, style: str) -> str:
    return f"bold {AMBER}" if now - state.changed_at.get(hid, 0.0) < FLASH_S else style


def bm25_lane(state: FilesState, width: int, now: float, frame: int, max_rows: int | None) -> Content:
    hits = state.bm25()
    if not hits:
        if state.status not in (None, "ok"):
            return Content.assemble((f"✗ {TOOL} did not search: ", RED), (state.error.strip()[:400], INK))
        spin = SPINNER[frame % len(SPINNER)]
        return Content.assemble((f"{spin} ", AMBER), ("the passage table opens while the walk runs…", MUTED))
    shown = hits if max_rows is None else hits[:max_rows]
    names = short_paths([h.path for h in shown])
    bar_w = 6
    name_w = max(10, width - 4 - 2 - 2 - 1 - bar_w - 5 - 2)
    lines: list[Content] = []
    for hit in shown:
        both = "walk" in hit.lanes
        lines.append(
            Content.assemble(
                (f"{hit.bm25:>3} ", MUTED),
                ("◆ " if both else "  ", AMBER if both else ""),
                kind_of(hit),
                (" ", ""),
                (name_of(hit, names, name_w).ljust(name_w), flash(state, hit.id, now, INK)),
                (" ", ""),
                *score_of(state, hit, bar_w, frame),
                (" ✦" if hit.seed else "", f"bold {AMBER}"),
            )
        )
    rest = hits[len(shown) :]
    if rest:
        best = max((h.score or 0.0 for h in rest), default=0.0)
        lines.append(Content.assemble((f"    … {len(rest)} more, best {best:.2f}", f"italic {MUTED}")))
    return Content("\n").join(lines)


def walk_lane(state: FilesState, width: int, now: float, frame: int, max_rows: int | None) -> tuple[Content, int]:
    if not state.walk.nodes and state.status in (None, "ok") and state.last is None:
        spin = SPINNER[frame % len(SPINNER)]
        return Content.assemble((f"{spin} ", AMBER), ("waiting for the index to be fresh…", MUTED)), 0
    return tree(state.walk, width, now, frame, False, max_rows)


def converge(state: FilesState, width: int) -> Content:
    """The two lanes meeting at the judge."""
    walked = sum(1 for h in state.hits.values() if "walk" in h.lanes and h.state != "card")
    bm25 = sum(1 for h in state.hits.values() if "bm25" in h.lanes)
    both = sum(1 for h in state.hits.values() if {"walk", "bm25"} <= set(h.lanes))
    left = f" {walked} from the walk "
    right = f" {bm25} from BM25 · {both} in both "
    room = max(4, width - len(left) - len(right) - 9)
    a, b = room // 2, room - room // 2
    lit = MUTED if state.stage in (None, "walk") else INK
    return Content.assemble(
        ("╰─", RULE),
        (left, lit),
        ("─" * a + "▶ ", RULE),
        ("⚖", f"bold {AMBER}"),
        (" ◀" + "─" * b, RULE),
        (right, lit),
        ("─╯", RULE),
    )


def sparkline(scores: list[float]) -> list[Part]:
    bins = [0] * 10
    for s in scores:
        bins[min(9, int(s * 10))] += 1
    top = max(bins) or 1
    return [
        (SPARK[0] if b == 0 else SPARK[max(1, round(b / top * (len(SPARK) - 1)))], INK_SOFT if b else RULE)
        for b in bins
    ]


def judge_line(state: FilesState, frame: int) -> Content:
    cands = state.candidates()
    judged = sum(1 for h in cands if h.state == "judged")
    asking = sum(1 for h in cands if h.state == "asking")
    parts: list[Part] = [
        ("JUDGE  ", f"bold {MUTED}"),
        pips(judged, len(cands)),
        (f" {judged}/{len(cands)} passages", INK),
    ]
    if asking:
        parts.append((f"  {SPINNER[frame % len(SPINNER)]} {asking} with System 1", AMBER))
    scores = [h.score for h in cands if h.score is not None and h.state == "judged"]
    if scores:
        parts += [("  scores 0 ", MUTED), *sparkline(scores), (" 1", MUTED)]
    return Content.assemble(*parts)


def seeds_line(state: FilesState, width: int) -> Content:
    seeds = sorted((h for h in state.hits.values() if h.seed), key=lambda h: -(h.score or 0.0))
    parts: list[Part] = [("SEEDS  ", f"bold {MUTED}")]
    if not seeds:
        parts.append(("the ten best judged become seeds", f"italic {MUTED}"))
        return Content.assemble(*parts)
    names = short_paths([h.path for h in seeds])
    used = 7
    for n, hit in enumerate(seeds):
        piece = f"{name_of(hit, names, 30)} {hit.score or 0.0:.2f}"  # a seed is a passage: its part tells two apart
        if used + len(piece) + 4 > width:
            parts.append((f"… {len(seeds) - n} more", MUTED))
            break
        parts += [("✦ ", f"bold {AMBER}"), (piece, INK), ("  ", "")]
        used += len(piece) + 4
    return Content.assemble(*parts)


def hop(state: FilesState, width: int, frame: int, max_rows: int | None) -> Content:
    """Each seed's terms, by weight, and what each reached: the expansion's constellation."""
    if not state.terms:
        if state.stage in ("rejudge", "done") or state.finished:
            return Content.assemble(("HOP    ", f"bold {MUTED}"), ("no term outside the question's words", MUTED))
        hint = "their rarest words and names, searched with the question, reach what the walk pruned"
        return Content.assemble(("HOP    ", f"bold {MUTED}"), (hint, f"italic {MUTED}"))
    reached = [h for t in state.terms for i in t.hits if (h := state.hits.get(i)) is not None]
    missed = sum(1 for h in reached if state.missed(h))
    lines = [
        Content.assemble(
            ("HOP    ", f"bold {MUTED}"),
            (f"{len(state.terms)} terms → {len(reached)} new passages", INK),
            (f", {missed} in files the walk left" if missed else "", AMBER),
        )
    ]
    fruitful = [t for t in state.terms if t.hits]
    by_seed: dict[str, list[Term]] = {}
    for term in fruitful:
        by_seed.setdefault(term.seed, []).append(term)
    top_w = max(t.weight for t in state.terms) or 1.0
    seed_names = short_paths([s.partition("#")[0] for s in by_seed])
    seed_w = min(22, max(8, width // 5))
    term_w = min(18, max(8, width // 7))
    rows = 0
    budget = len(fruitful) if max_rows is None else max_rows
    for seed, terms in by_seed.items():
        for k, term in enumerate(terms):
            if rows >= budget:
                break
            rows += 1
            label = fit(seed_names[seed.partition("#")[0]], seed_w, keep_end=True) if k == 0 else ""
            joint = ("─┬─" if len(terms) > 1 else "───") if k == 0 else ("└─" if k == len(terms) - 1 else "├─").rjust(3)
            hits = [h for i in term.hits if (h := state.hits.get(i)) is not None]
            best = max(hits, key=lambda h: h.score or 0.0, default=None)
            weight = round(term.weight / top_w * 6)
            parts: list[Part] = [
                ("  ", ""),
                (label.rjust(seed_w), INK),
                (f" {joint} ", RULE),
                (fit(term.term, term_w).ljust(term_w), f"bold {INK}"),
                (" ", ""),
                ("▮" * weight, AMBER),
                ("▯" * (6 - weight), RULE),
                (" ──▶ ", RULE),
                (f"{len(hits):>2}", INK),
            ]
            if best is not None:
                far = state.missed(best)
                rest = width - sum(len(p[0]) for p in parts) - 14
                parts += [
                    ("  ↯ " if far else "  · ", f"bold {AMBER}" if far else MUTED),
                    (fit(best.path, max(6, rest), keep_end=True), INK),
                ]
                if best.state == "asking" or best.score is None:
                    parts.append((f" {SPINNER[frame % len(SPINNER)]}", AMBER))
                else:
                    parts.append((f" {best.score:.2f}", f"bold {GREEN}" if best.score >= 0.5 else INK))
            lines.append(Content.assemble(*parts))
    barren = len(state.terms) - len(fruitful)
    more = [f"{len(fruitful) - rows} more terms"] if rows < len(fruitful) else []
    more += [f"{barren} reached nothing new"] if barren else []
    if more:
        lines.append(Content.assemble((f"  … {', '.join(more)}", f"italic {MUTED}")))
    return Content("\n").join(lines)


def lane_of(state: FilesState, hit: Hit) -> Part:
    if "term" in hit.lanes and hit.term:
        return (f"↯ {hit.term}" if state.missed(hit) else f"⤳ {hit.term}", AMBER)
    if hit.state == "card":
        return ("walk card", MUTED)
    lanes = [lane for lane in ("walk", "bm25") if lane in hit.lanes]
    return ("+".join(lanes), INK_SOFT)


def podium(state: FilesState, width: int, now: float, frame: int) -> Content:
    hits = state.top()
    if not hits:
        return Content.assemble(("Nothing judged yet: the lanes are still filling.", MUTED))
    ranked = hits[0].rank is not None
    names = short_paths([h.path for h in hits])
    lane_w = min(18, max(9, width // 6))
    bar_w = 6
    name_w = min(56, max(10, width - 3 - 2 - 2 - lane_w - 1 - bar_w - 6 - 4))
    lines: list[Content] = []
    for n, hit in enumerate(hits, 1):
        medal = n <= 3 and ranked
        lane, lane_style = lane_of(state, hit)
        moved = (hit.scored or n) - n if ranked else 0
        lines.append(
            Content.assemble(
                (f"{n:>2} ", MUTED),
                ("★ " if medal else "  ", f"bold {GREEN}"),
                kind_of(hit),
                (" ", ""),
                (
                    name_of(hit, names, name_w).ljust(name_w),
                    flash(state, hit.id, now, f"bold {GREEN}" if medal else INK),
                ),
                (" ", ""),
                (fit(lane, lane_w).ljust(lane_w), lane_style),
                (" ", ""),
                *score_of(state, hit, bar_w, frame),
                (f" ▲{moved}" if moved > 0 else f" ▼{-moved}" if moved < 0 else "", GREEN if moved > 0 else MUTED),
            )
        )
    return Content("\n").join(lines)


def stats(state: FilesState) -> Content:
    t = state.last
    if t is None:
        return Content.assemble(("No ask yet.", MUTED))
    secs = t.elapsed_ms / 1000
    scores = [h.score for h in state.hits.values() if h.score is not None]
    return Content("\n").join(
        [
            Content.assemble(
                ("asks ", MUTED), (str(t.asks), f"bold {INK}"), (" · judged ", MUTED), (str(t.judged), INK)
            ),
            Content.assemble((f"≈${t.cost_usd:.4f}", INK), (f" · {secs:.1f}s", MUTED)),
            Content.assemble(
                ("walk files ", MUTED), (f"{t.files}/{t.files_budget}".ljust(8), INK), pips(t.files, t.files_budget)
            ),
            Content.assemble(("passages ", MUTED), (str(t.passages), INK), (f" · {t.unread} by card", MUTED)),
            Content.assemble(("scores 0 ", MUTED), *sparkline(scores), (" 1", MUTED)),
        ]
    )


def header(state: FilesState, width: int, frame: int, replay: bool) -> Content:
    """The panel's top: what the search does and how long it took, its stages, what the stage does. The question
    is on the call's line just above."""
    spin = SPINNER[frame % len(SPINNER)]
    secs = state.elapsed_ms / 1000
    if state.status not in (None, "ok"):
        status: list[Part] = [(f"✗ {state.status}", f"bold {RED}")]
    elif state.status == "ok" and state.last is None:
        status = [("nothing to search", MUTED)]
    elif state.finished and state.stage == "done":
        status = [("✓ ranked", f"bold {GREEN}"), (f" in {secs:.1f}s", MUTED)]
    elif state.finished:
        status = [("✗ stopped", f"bold {RED}"), (f" after {secs:.1f}s", MUTED)]
    elif state.stage is None:
        status = [(spin, AMBER), (" waiting for the index", MUTED)]
    else:
        status = [(spin, AMBER), (f" {secs:.1f}s", INK)]
    if replay:
        status = [(f"⟲ replay ×{REPLAY_SPEED:g}  ", f"bold {AMBER}"), *status]
    first: list[Part] = [("⌕ ", f"bold {AMBER}"), ("System 1 searches the workspace", f"bold {INK}")]
    if state.scope:
        first.append((f"  under {state.scope}", MUTED))
    used = sum(len(p[0]) for p in first) + sum(len(p[0]) for p in status)
    first += [(" " * max(2, width - used), ""), *status]
    for sep, times in (("  ──  ", True), (" ─ ", True), (" ─ ", False)):  # the widest that fits
        second = stage_chips(state, spin, sep, times)
        if sum(len(p[0]) for p in second) <= width:
            break
    lines = [Content.assemble(*first), Content.assemble(*second)]
    if state.stage is not None and state.live:
        lines.append(Content.assemble((STAGE_HINTS[state.stage], f"italic {MUTED}")))
    return Content("\n").join(lines)


def stage_chips(state: FilesState, spin: str, sep: str, times: bool) -> list[Part]:
    """walk ∥ bm25 → judge → seeds → terms → judge → ranked, each done with its time."""
    second: list[Part] = []
    current = STAGES.index(state.stage) if state.stage is not None else -1
    for i, stage in enumerate(STAGES):
        if i:
            second.append((sep, RULE))
        started = state.stage_started.get(stage)
        if (state.finished and started is not None) or i < current:
            if started is None:
                second.append(("– " + STAGE_NAMES[stage], MUTED))
                continue
            later = [state.stage_started[s] for s in STAGES[i + 1 :] if s in state.stage_started]
            took = ((later[0] if later else state.elapsed_ms) - started) / 1000
            second += [("● ", GREEN), (STAGE_NAMES[stage], INK)]
            if stage != "done" and times:
                second.append((f" {took:.1f}s", MUTED))
        elif i == current and state.live:
            second += [(f"{spin} ", AMBER), (STAGE_NAMES[stage], f"bold {INK}")]
        else:
            second += [("○ ", MUTED), (STAGE_NAMES[stage], MUTED)]
    return second


def card(state: FilesState, frame: int) -> Content:
    """The search in one line: the folded panel."""
    head: list[Part] = [("⌕ ", f"bold {AMBER}"), ("workspace search", f"bold {INK}")]
    t = state.last
    if state.status not in (None, "ok"):
        return Content.assemble(*head, ("  the search did not run (see the output)", MUTED))
    if state.finished:
        judged = sum(1 for h in state.hits.values() if h.state == "judged")
        parts: list[Part] = [
            *head,
            (
                f"  {judged} passages judged in {state.elapsed_ms / 1000:.1f}s · {t.asks if t else 0} asks · "
                f"{len(state.terms)} terms",
                MUTED,
            ),
        ]
        top = state.top()
        if top:
            parts += [("  ★ ", f"bold {GREEN}"), (top[0].path, f"bold {GREEN}"), (f" {top[0].score or 0:.2f}", GREEN)]
        return Content.assemble(*parts)
    stage = STAGE_NAMES[state.stage] if state.stage else "waiting for the index"
    parts = [*head, (f" {SPINNER[frame % len(SPINNER)]}", AMBER), (f" {stage}", INK)]
    if t is not None:
        parts.append((f" · {t.asks} asks", MUTED))
    top = state.top()
    if top:
        parts += [("  best so far ", MUTED), (top[0].path, INK), (f" {top[0].score or 0:.2f}", INK)]
    return Content.assemble(*parts)


def feed(state: FilesState, width: int) -> Content:
    parts: list[Part] = [("latest  ", MUTED)]
    for i, hid in enumerate(state.feed):
        hit = state.hits[hid]
        if i:
            parts.append(("   ", ""))
        score = hit.score or 0.0
        parts += [
            (f"{score:.2f} ", f"bold {INK}"),
            ("✓ " if score >= 0.5 else "· ", GREEN if score >= 0.5 else MUTED),
            (fit(f"{hit.path.rsplit('/', 1)[-1]}" + (f" · {hit.title}" if hit.title else ""), 30), INK),
        ]
    content = Content.assemble(*parts)
    return content.truncate(width, ellipsis=True) if len(content) > width else content


# -- the panel, the view ------------------------------------------------------------------------------------------


@dataclass
class Replay:
    source: FilesState
    state: FilesState
    started: float
    at: int = 0


class FilesPanel(Vertical):
    """The search, live, right under its `decision_files` call in the transcript: the stages, the two lanes, the judge,
    the seeds and their hop, the podium, System 1's counts, and a row of controls (fold, replay, all rows)."""

    DEFAULT_CSS = """
    FilesPanel {
        height: auto;
        background: $surface;
        border-left: thick $warning;
        padding: 0 1;
    }
    FilesPanel Static {
        height: auto;
        text-wrap: nowrap;
        text-overflow: ellipsis;
    }
    FilesPanel .dxf-lanes, FilesPanel .dxf-bottom {
        height: auto;
        layout: horizontal;
        margin-top: 1;
    }
    FilesPanel.-narrow .dxf-lanes, FilesPanel.-narrow .dxf-bottom {
        layout: vertical;
    }
    FilesPanel .dxf-walk, FilesPanel .dxf-bm25, FilesPanel .dxf-podium, FilesPanel .dxf-stats {
        width: 1fr;
    }
    FilesPanel .dxf-bm25, FilesPanel .dxf-stats {
        margin-left: 3;
    }
    FilesPanel .dxf-podium {
        width: 3fr;
    }
    FilesPanel .dxf-stats {
        width: 2fr;
        max-width: 48;
    }
    FilesPanel.-narrow .dxf-walk, FilesPanel.-narrow .dxf-bm25,
    FilesPanel.-narrow .dxf-podium, FilesPanel.-narrow .dxf-stats {
        width: 1fr;
        max-width: 100%;
        margin-left: 0;
    }
    FilesPanel.-narrow .dxf-bm25, FilesPanel.-narrow .dxf-stats {
        margin-top: 1;
    }
    FilesPanel .dxf-controls {
        height: 1;
    }
    FilesPanel .dxf-feed {
        width: 1fr;
        height: 1;
    }
    FilesPanel.-folded .dxf-lanes, FilesPanel.-folded .dxf-flow, FilesPanel.-folded .dxf-bottom,
    FilesPanel.-folded .dxf-feed {
        display: none;
    }
    """

    def __init__(self, host: ViewHost, state: FilesState) -> None:
        super().__init__(classes="dxf-panel")
        self.host = host
        self.state = state
        self.frame = 0
        self.folded = False
        self.all_rows = False
        self.replay: Replay | None = None
        self.dirty = True
        self._timer: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Static(classes="dxf-head")
        with Horizontal(classes="dxf-lanes"):
            yield Static(classes="dxf-walk")
            yield Static(classes="dxf-bm25")
        yield Static(classes="dxf-flow")
        with Horizontal(classes="dxf-bottom"):
            yield Static(classes="dxf-podium")
            yield Static(classes="dxf-stats")
        with Horizontal(classes="dxf-controls"):
            yield Control("▾ fold", self.toggle_fold, classes="dxf-fold")
            yield Control("⟲ replay", self.start_replay, classes="dxf-replay")
            yield Control("", self.toggle_rows, classes="dxf-rows")
            yield Static(classes="dxf-feed")

    def on_mount(self) -> None:
        self._timer = self.set_interval(1 / FPS, self.tick)
        self.draw(time.monotonic())

    def on_resize(self, event: events.Resize) -> None:
        self.set_class(event.size.width < NARROW, "-narrow")
        self.wake()

    def on_click(self, event: events.Click) -> None:
        event.stop()  # a click in the panel is not one on the call's block

    @property
    def shown(self) -> FilesState:
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
            or any(now - t < FLASH_S for t in state.walk.changed_at.values())
        )
        if self.dirty or lively:
            self.dirty = False
            self.draw(now)
        if not lively and not self.dirty and self.replay is None and self._timer is not None:
            self._timer.pause()  # settled: no frames until something changes

    def draw(self, now: float) -> None:
        state = self.shown
        head = self.query_one(".dxf-head", Static)
        replay = self.query_one(".dxf-replay", Control)
        rows = self.query_one(".dxf-rows", Control)
        fold = self.query_one(".dxf-fold", Control)
        fold.update("▸ unfold" if self.folded else "▾ fold")
        replay.display = state.finished and not self.folded
        if self.folded:
            head.update(card(state, self.frame))
            rows.display = False
            return
        width = self.content_size.width or 100
        lane_w = self.query_one(".dxf-walk", Static).content_size.width or 60
        bm25_w = self.query_one(".dxf-bm25", Static).content_size.width or 60
        podium_w = self.query_one(".dxf-podium", Static).content_size.width or 60
        cap = None if self.all_rows else SETTLED_ROWS if state.finished else LANE_ROWS
        hop_cap = None if self.all_rows else SETTLED_ROWS if state.finished else HOP_ROWS
        head.update(header(state, width, self.frame, self.replay is not None))
        walk, hidden = walk_lane(state, lane_w, now, self.frame, cap)
        self.query_one(".dxf-walk", Static).update(section("WALK  folders → files → parts", walk))
        passages = state.last.passages if state.last else 0
        bm25_title = f"BM25  top {len(state.bm25())} of {passages} passages" if passages else "BM25"
        self.query_one(".dxf-bm25", Static).update(section(bm25_title, bm25_lane(state, bm25_w, now, self.frame, cap)))
        hidden += max(0, len(state.bm25()) - cap) if cap else 0
        hidden += max(0, sum(1 for t in state.terms if t.hits) - hop_cap) if hop_cap else 0
        at = STAGES.index(state.stage) if state.stage is not None else -1
        flow = [converge(state, width), judge_line(state, self.frame)]
        if at >= STAGES.index("judge") or state.finished:
            flow.append(seeds_line(state, width))
        if at >= STAGES.index("expand") or state.finished:
            flow.append(hop(state, width, self.frame, hop_cap))
        self.query_one(".dxf-flow", Static).update(Content("\n").join(flow))
        title = "RESULT  top 10, in diversity order" if state.top() and state.top()[0].rank else "LEADERBOARD"
        self.query_one(".dxf-podium", Static).update(section(title, podium(state, podium_w, now, self.frame)))
        self.query_one(".dxf-stats", Static).update(section("SYSTEM 1", stats(state)))
        rows.display = hidden > 0 or self.all_rows
        rows.update("▴ fewer rows" if self.all_rows else f"▾ {hidden} more rows")
        self.query_one(".dxf-feed", Static).update(feed(state, max(20, width - 40)))

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
        fresh = FilesState(call_id=source.call_id, question=source.question, scope=source.scope)
        fresh.status = source.status
        self.replay = Replay(source=source, state=fresh, started=time.monotonic())
        self.now()


class DecisionFilesView(ExtensionView):
    """The workspace extension's view: a live panel on its search under each `decision_files` call."""

    def __init__(self, host: ViewHost, extension: str) -> None:
        super().__init__(host, extension)
        self.searches: list[FilesState] = []
        self.panels: dict[str, FilesPanel] = {}

    async def on_event(self, event: Event) -> None:
        match event:
            case ToolCallStart(call_id=call_id, arguments=arguments) if event.name == TOOL:
                scope = arguments.get("scope")
                state = FilesState(
                    call_id=call_id,
                    question=str(arguments.get("question", "")),
                    scope=scope if isinstance(scope, str) and scope else None,
                )
                self.searches.append(state)
                self.panels[call_id] = FilesPanel(self.host, state)
                await self.host.attach(call_id, self.panels[call_id])
            case ExtensionEvent(kind=kind, data=data) if event.extension == self.extension and kind == KIND:
                trace = UnionTrace.model_validate(data)
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

    def bind(self, trace: UnionTrace) -> FilesState | None:
        """The call a trace belongs to: the one bound to its search, else the oldest running call asked the same
        question that has none yet (searches in one turn run side by side), else the oldest without one."""
        for state in self.searches:
            if state.search == trace.search:
                return state
        free = [s for s in self.searches if s.search is None and s.status is None]
        same = [s for s in free if s.question == trace.question]
        return (same or free or [None])[0]
