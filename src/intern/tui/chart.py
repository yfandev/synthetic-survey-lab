"""System 1's panel as a chart of the session: three line plots on one turn axis, drawn in braille.

Watch: what the supervisor answered at each turn's end, a point a turn: P(advancing), P(stuck) (re-investigating
and retrying together: either way the work is not moving on), P(off-plan) and P(finished). Guard: at the
turns where System 1 judged a tool call, P(risk) and P(the user asked for it) of the turn's riskiest call, joined
across the turns between; a refused call is a red ✗ on its point. A line under FLOOR is not drawn, so the floor
stays clean and a line that shows means something; there are no threshold lines. Colour is kept for meaning, as
everywhere on the page; where lines cross in a cell the more alarming one shows.

A short session is spread out, a few dots a turn; a long one is squeezed, a few turns a point, the watch lines
averaged and the guard taking the riskiest call, so a risky call is never averaged away. Above the plots, the
turns and the prompts; each plot's legend gives each line's latest value (the current verdict's in bold), the
guard's how many calls it refused of those it judged. The transcript's cards say each turn in full; this says how
the work went.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from math import ceil

from textual.content import Content

from intern.tui.ledger import GuardJudgement, RunLedger, TurnJudgement
from intern.tui.theme import AMBER, DIM, GREEN, INK, INK_SOFT, MUTED, RED

Part = str | tuple[str, str]

FLOOR = 0.1
"""Under this a line is not drawn: most lines sit near zero most of the time, and a wiggle along the floor is
noise."""
STEP = 4
"""Dots between turns at most: a short session two cells a turn, not stretched across the panel."""
GUTTER = 3
"""The y axis: `1 ┤`."""
TICK_GAP = 8
"""Cells between the turn numbers under the axis, at least."""
BRAILLE = 0x2800
DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
"""A braille cell's dots by row, then column: two across, four down."""

RISK = "risk"
ASKED = "asked"
Line = tuple[str, str, str]
"""A plot's line: its key, its name in the legend, its colour."""
WATCH: tuple[Line, ...] = (
    ("off_plan", "off-plan", RED),
    ("stuck", "stuck", AMBER),
    ("finished", "finished", INK),
    ("advancing", "advancing", GREEN),
)
"""The watch plot's lines, the first winning a cell they share: the more alarming shows."""
GUARD: tuple[Line, ...] = ((RISK, RISK, AMBER), (ASKED, ASKED, INK_SOFT))
VERDICT = {
    "advancing": "advancing",
    "re_investigating": "stuck",
    "retrying": "stuck",
    "off_plan": "off_plan",
    "finished": "finished",
}
"""The supervisor's mode as the line that shows it, for the legend's bold."""


@dataclass
class Column:
    """One turn, or a few when the session has more turns than the plot has dots across."""

    values: dict[str, list[float]] = field(default_factory=dict)
    calls: list[tuple[float, float]] = field(default_factory=list)
    """Each judged call's (risk, asked)."""
    refused: bool = False
    verdict: str | None = None

    def value(self, name: str) -> float | None:
        if name in (RISK, ASKED):
            if not self.calls:
                return None
            risk, asked = max(self.calls)  # the riskiest call, its own asked
            return risk if name == RISK else asked
        values = self.values.get(name)
        return sum(values) / len(values) if values else None

    def take(self, other: Column) -> None:
        for name, values in other.values.items():
            self.values.setdefault(name, []).extend(values)
        self.calls += other.calls
        self.refused |= other.refused
        self.verdict = other.verdict or self.verdict


def mode_p(j: TurnJudgement, mode: str) -> float:
    """A mode's probability; a bare choice without probabilities counts as certain."""
    if not j.probabilities:
        return 1.0 if j.mode == mode else 0.0
    return j.probabilities.get(mode, 0.0)


def watch_values(j: TurnJudgement) -> dict[str, list[float]]:
    return {
        "advancing": [mode_p(j, "advancing")],
        "stuck": [mode_p(j, "re_investigating") + mode_p(j, "retrying")],
        "off_plan": [mode_p(j, "off_plan")],
        "finished": [mode_p(j, "finished")],
    }


def turns(runs: list[RunLedger]) -> list[Column]:
    """A column per turn of every run, in order: the supervisor's answers and the judged calls."""
    columns: list[Column] = []
    for run in runs:
        guards: dict[int, list[GuardJudgement]] = {}
        for g in run.guards:
            guards.setdefault(g.turn, []).append(g)
        for turn in sorted({*run.turns, *run.context, *guards}):
            column = Column()
            if (j := run.turns.get(turn)) is not None:
                column.values = watch_values(j)
                column.verdict = VERDICT.get(j.mode)
            for g in guards.get(turn, []):
                column.calls.append((g.risk, g.asked))
                column.refused |= g.refused
            columns.append(column)
    return columns


def squeeze(columns: list[Column], dots: int) -> tuple[list[Column], int]:
    """The columns, a few turns to each when they outnumber `dots`; and how many turns a column is."""
    per = max(1, ceil(len(columns) / max(1, dots)))
    if per == 1:
        return columns, 1
    merged: list[Column] = []
    for start in range(0, len(columns), per):
        column = Column()
        for each in columns[start : start + per]:
            column.take(each)
        merged.append(column)
    return merged, per


def spread(count: int, dots: int) -> float:
    """Dots from one column to the next: STEP at most, less once the columns fill the plot."""
    return 0.0 if count < 2 else min(float(STEP), (dots - 1) / (count - 1))


def segment(grid: list[list[int]], a: tuple[int, int], b: tuple[int, int]) -> None:
    """The dots from `a` to `b` (Bresenham)."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), -abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx + dy
    while True:
        dot(grid, x0, y0)
        if (x0, y0) == (x1, y1):
            return
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def dot(grid: list[list[int]], x: int, y: int) -> None:
    if 0 <= y < len(grid) * 4 and 0 <= x < len(grid[0]) * 2:
        grid[y // 4][x // 2] |= DOT_BITS[y % 4][x % 2]


def to_y(value: float, rows: int) -> int:
    return round((1 - min(1.0, max(0.0, value))) * (rows * 4 - 1))


def line(columns: list[Column], name: str, step: float, rows: int, cells: int) -> list[list[int]]:
    """One line's dots: its points joined, a stretch under FLOOR at both ends left out."""
    grid = [[0] * cells for _ in range(rows)]
    points = [(round(i * step), v) for i, c in enumerate(columns) if (v := c.value(name)) is not None]
    for (x0, v0), (x1, v1) in pairwise(points):
        if max(v0, v1) >= FLOOR:
            segment(grid, (x0, to_y(v0, rows)), (x1, to_y(v1, rows)))
    for x, v in points:
        if v >= FLOOR:
            dot(grid, x, to_y(v, rows))
    return grid


def plot(columns: list[Column], lines: tuple[Line, ...], step: float, rows: int, cells: int) -> list[list[Part]]:
    """The plot's rows, each line's dots in its colour; where lines share a cell the first in `lines` shows."""
    grids = [(line(columns, key, step, rows, cells), style) for key, _, style in lines]
    out: list[list[Part]] = []
    for r in range(rows):
        row: list[Part] = []
        for c in range(cells):
            bits = 0
            style = ""
            for grid, line_style in grids:
                if grid[r][c]:
                    bits |= grid[r][c]
                    style = style or line_style
            row.append((chr(BRAILLE + bits), style) if bits else " ")
        out.append(row)
    return out


def refusals(rows: list[list[Part]], columns: list[Column], step: float) -> None:
    """A refused call: a red ✗ where its point is."""
    for i, column in enumerate(columns):
        if column.refused and (risk := column.value(RISK)) is not None:
            cell, r = round(i * step) // 2, to_y(risk, len(rows)) // 4
            if cell < len(rows[r]):
                rows[r][cell] = ("✗", f"bold {RED}")


def latest(columns: list[Column], name: str) -> float | None:
    return next((v for c in reversed(columns) if (v := c.value(name)) is not None), None)


def legend(title: str, items: list[list[Part]], width: int) -> list[list[Part]]:
    """The plot's title and its lines' names and latest values, as many to a row as fit."""
    rows: list[list[Part]] = [[(title.ljust(6), f"bold {MUTED}")]]
    used = 6
    for item in items:
        size = sum(len(p if isinstance(p, str) else p[0]) for p in item)
        if used + size > width and used > 6:
            rows.append([" " * 6])
            used = 6
        rows[-1] += item
        used += size
    return rows


def item(name: str, style: str, value: float | None, now: bool) -> list[Part]:
    shown = "" if value is None else f" {value:.2f}"
    return [("━ ", style), (name, f"bold {INK}" if now else MUTED), (shown, f"bold {INK}" if now else INK), "  "]


def axis_rows(rows: list[list[Part]]) -> list[list[Part]]:
    """The plot's rows behind the y axis: 1 at the top, 0 at the bottom."""
    out: list[list[Part]] = []
    for r, row in enumerate(rows):
        digit = "1" if r == 0 else "0" if r == len(rows) - 1 else " "
        out.append([(f"{digit} ", MUTED), ("│" if digit == " " else "┤", DIM), *row])
    return out


def ticks(total: int, per: int, step: float, cells: int) -> list[Part]:
    """Turn numbers under the axis, a round number apart, TICK_GAP cells at least."""
    cells_a_turn = step / 2 / per
    every = next((n for n in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000) if n * cells_a_turn >= TICK_GAP), 2000)
    row = [" "] * cells
    free = 0
    for turn in (1, *range(every, total + 1, every)):
        cell = round(((turn - 1) // per) * step) // 2
        label = str(turn)
        if cell < free or cell + len(label) > cells:
            continue
        row[cell : cell + len(label)] = list(label)
        free = cell + len(label) + 1
    return [" " * GUTTER, ("".join(row), MUTED)]


@dataclass
class Layout:
    """What fits the panel's height: each plot's rows (0 leaves it out) and whether the spacer shows."""

    watch: int = 0
    guard: int = 0
    spacer: bool = False


MIN_ROWS = {"watch": 2, "guard": 2}
MAX_ROWS = {"watch": 12, "guard": 8}
SHARE = {"watch": 0.6, "guard": 0.4}


def fit(height: int, watch_legend: int, guard_legend: int, guarded: bool) -> Layout:
    """The rows each part takes, so the chart fits `height` and never loses its axis: the summary, the watch
    plot and the axis with its turns first, then the guard's plot and a spacer, each while it fits; the
    rows left over grow the plots, the larger share to the watch."""
    layout = Layout(watch=MIN_ROWS["watch"])
    used = 1 + watch_legend + layout.watch + 2  # the summary; the axis and its turns
    if guarded and used + guard_legend + MIN_ROWS["guard"] <= height:
        layout.guard = MIN_ROWS["guard"]
        used += guard_legend + layout.guard
    if used + 1 <= height:
        layout.spacer, used = True, used + 1
    shown = [name for name in ("watch", "guard") if getattr(layout, name)]
    spare = height - used
    for name in shown:
        grow = min(MAX_ROWS[name] - getattr(layout, name), int(spare * SHARE[name] / sum(SHARE[n] for n in shown)))
        setattr(layout, name, getattr(layout, name) + max(0, grow))
    spare = height - used - sum(getattr(layout, n) - MIN_ROWS[n] for n in shown)
    for name in shown:  # what rounding left, to the first with room
        grow = max(0, min(spare, MAX_ROWS[name] - getattr(layout, name)))
        setattr(layout, name, getattr(layout, name) + grow)
        spare -= grow
    return layout


def render_chart(runs: list[RunLedger], window: int, width: int, height: int) -> Content:
    """The panel's body, `width` cells across and `height` rows at most (`fit`): a summary line, the watch
    and guard plots over one turn axis. `window` is not needed: the context is the masthead's."""
    every = turns(runs)
    if not any(c.values or c.calls for c in every):
        return Content.assemble(("System 1 has not judged a turn yet.", MUTED))
    cells = max(10, width - GUTTER)
    columns, per = squeeze(every, cells * 2)
    step = spread(len(columns), cells * 2)
    prompts = sum(1 for run in runs if run.turns or run.context)
    guarded = sum(len(run.guards) for run in runs)
    head = f"{len(every)} turns · {prompts} prompt{'s' if prompts != 1 else ''}"
    head += f" · {per} turns a point" if per > 1 else ""

    verdict = next((c.verdict for c in reversed(columns) if c.values), None)
    watch_legend = legend(
        "watch", [item(name, style, latest(columns, key), key == verdict) for key, name, style in WATCH], width
    )
    guard_items = [item(name, style, latest(columns, key), False) for key, name, style in GUARD]
    refused = sum(run.refusals for run in runs)
    guard_items.append([("✗", f"bold {RED}" if refused else MUTED), (f" refused {refused} of {guarded}", MUTED)])
    guard_legend = legend("guard", guard_items, width)
    layout = fit(height, len(watch_legend), len(guard_legend) if guarded else 0, bool(guarded))

    lines: list[list[Part]] = [[(head, MUTED)], *([[]] if layout.spacer else [])]
    lines += [*watch_legend, *axis_rows(plot(columns, WATCH, step, layout.watch, cells))]
    if layout.guard:
        rows = plot(columns, GUARD, step, layout.guard, cells)
        refusals(rows, columns, step)
        lines += [*guard_legend, *axis_rows(rows)]
    lines.append([("  └" + "─" * cells, DIM)])
    lines.append(ticks(len(every), per, step, cells))
    out: list[Part] = []
    for i, row in enumerate(lines):
        out += [*row, "\n"] if i < len(lines) - 1 else row
    return Content.assemble(*out)
