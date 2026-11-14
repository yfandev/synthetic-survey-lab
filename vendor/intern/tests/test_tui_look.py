"""The look: the mark, the masthead, and System 1's cards."""

from __future__ import annotations

import re

from textual.content import Content

from intern.tui.ledger import CompactionJudgement, GuardJudgement, KeepVerdict, RuleRefusal, TurnJudgement
from intern.tui.theme import MASCOT_COLOURS, S_BOX, S_BOX_MUTED, S_RULE
from intern.tui.widgets import (
    CARD_LINES,
    FILLED,
    LANDED,
    MASCOT,
    METER_WIDTH,
    NBSP,
    TRACK,
    StatusBar,
    compaction_card,
    cost_text,
    guard_card,
    level,
    mark,
    meter,
    rule_card,
    settled,
    supervisor_card,
    thinking_box,
)

LEVEL = re.compile(r"#([0-9a-f]{2})[0-9a-f]{4}")


def judgement(**overrides: object) -> TurnJudgement:
    fields: dict[str, object] = {
        "turn": 7,
        "mode": "advancing",
        "probabilities": {
            "advancing": 0.92,
            "re_investigating": 0.06,
            "off_plan": 0.01,
            "finished": 0.0,
            "retrying": 0.01,
        },
        "backend": "decisions-api:nace-ai/drex-max",
        "cached": False,
        "duration_ms": 565,
        "cost_usd": 0.0001,
    }
    fields.update(overrides)
    return TurnJudgement(**fields)  # type: ignore[arg-type]


def levels(style: str) -> list[int]:
    return [int(level, 16) for level in LEVEL.findall(style)]


def test_the_mark_draws_the_mascot_two_pixels_to_a_cell_in_its_own_colours_with_the_page_around_it():
    assert {pixel for row in MASCOT for pixel in row} <= {*MASCOT_COLOURS, "."}  # no pixel drawn as the page by a typo
    rows = mark()
    assert len(rows) == 7  # the masthead's height
    pixel = {colour: name for name, colour in MASCOT_COLOURS.items()}
    drawn: list[str] = []
    for row in rows:
        styles = [""] * len(row.plain)
        for span in row.spans:
            styles[span.start : span.end] = [str(span.style)] * (span.end - span.start)
        upper = lower = ""
        for glyph, style in zip(row.plain, styles, strict=True):
            fg, _, bg = style.partition(" on ")
            top, bottom = (fg, bg) if glyph == "▀" else (bg, fg) if glyph == "▄" else ("", "")
            upper, lower = upper + pixel.get(top, "."), lower + pixel.get(bottom, ".")
        drawn += [upper, lower]
    assert drawn == list(MASCOT)


def test_the_masthead_lines_up_the_models_the_run_and_each_systems_cost_in_columns():
    status = StatusBar()
    status.system2, status.system1, status.thinking = "anthropic/claude-haiku-5-5", "drex:drex-max-v1.1", "xhigh"
    status.levels = ("low", "medium", "high", "xhigh", "max")
    status.context, status.window = 233_000, 1_000_000
    status.cost2, status.cost1 = 1.25, 0.0178
    status.refresh_line()
    content = status.content
    assert isinstance(content, Content)
    rows = content.plain.splitlines()
    assert len(rows) == 7 and "Intern" in rows[0] and rows[6].endswith("   /help commands")
    sys2, sys1 = rows[2], rows[3]
    assert sys2.index("System 2") == sys1.index("System 1")
    assert sys2.index("anthropic/") == sys1.index("drex-max-v1.1")
    assert sys2.index(" thinking xhigh") == sys1.index("console")
    assert sys2.index("$1.2500") == sys1.index("$0.0178")  # each system's cost on its row, in a column
    assert sys2.rstrip().endswith("$1.2500   ○ 23% of 1.0M")  # the run after it: the context as a share


def test_the_thinking_box_is_filled_up_to_the_level_and_as_wide_at_every_one():
    levels = ("low", "medium", "high")
    boxes = ["".join(p if isinstance(p, str) else p[0] for p in thinking_box(each, levels)) for each in levels]
    assert boxes == [" thinking low    ▰▱▱ ", " thinking medium ▰▰▱ ", " thinking high   ▰▰▰ "]
    assert {part[1] for part in thinking_box("low", levels) if isinstance(part, tuple)} == {S_BOX, S_BOX_MUTED}
    assert "".join(p[0] for p in thinking_box("default", levels) if isinstance(p, tuple)).endswith("default ▱▱▱ ")


def test_a_systems_cost_says_when_its_provider_reports_no_price():
    assert (cost_text(0.0), cost_text(0.01234)) == ("", "$0.0123")
    assert (cost_text(0.0, unpriced=True), cost_text(0.5, unpriced=True)) == ("price unknown", "≥ $0.5000")


def test_a_meter_is_a_fixed_width_line_filled_to_the_level():
    assert "".join(text for text, _ in meter(0.62, "x")) == FILLED * 6 + TRACK * 4
    assert meter(0.0, "x") == [(TRACK * METER_WIDTH, S_RULE)]
    assert meter(1.0, "x") == [(FILLED * METER_WIDTH, "x")]


def card_lines(content: Content) -> list[str]:
    """A card's text with its non-breaking spaces (inside a labelled level) read as spaces."""
    return content.plain.replace(NBSP, " ").splitlines()


def test_a_level_binds_its_label_and_number_with_non_breaking_spaces():
    text = Content.assemble(*level("rewrites shared history", 0.96, "x")).plain
    assert " " not in text and text.replace(NBSP, " ") == "rewrites shared history 0.96"


def test_the_supervisor_card_is_the_panel_row_in_one_line():
    (line,) = card_lines(supervisor_card(judgement(strikes=1)))
    # the call, then the two next likeliest; finished (0.00) is not among them
    assert line == "System 1 · 0.6s   advancing 0.92   re-investigating 0.06   off-plan 0.01   → strike 1/3"
    (cached,) = card_lines(supervisor_card(judgement(cached=True)))
    assert cached.startswith("System 1 · 0.6s · cached   advancing") and "→" not in cached  # nothing followed


def test_the_guard_card_says_what_the_call_would_do_and_why_it_was_refused():
    g = GuardJudgement(
        turn=3,
        subject='bash {"command": "git push --force"}',
        kind="rewrites_shared_history",
        probabilities={"routine": 0.02, "rewrites_shared_history": 0.96, "unclear": 0.02},
        asked=0.02,
        backend="decisions-api:nace-ai/drex-max",
        cached=False,
        duration_ms=380,
        cost_usd=0.00004,
    )
    (line,) = card_lines(guard_card(g))
    assert line == "✓ System 1   rewrites shared history 0.96   risk 0.96   asked 0.02   0.4s"
    g.refused, g.reason = True, "System 1 judged that it rewrites history that others share and the user did not ask"
    lines = card_lines(guard_card(g))
    assert lines[0].startswith("✗ System 1 refused   ") and lines[1] == g.reason
    rule = rule_card(RuleRefusal(turn=4, call_id="c4", name="bash", rule="force_push", reason="a force push"))
    assert rule.plain == "✗ refused   rule · force push\na force push"


def test_a_compaction_card_lands_each_read_as_judged_then_settles_with_the_kept_reads_first():
    c = CompactionJudgement(turn=9)
    c.verdicts = [KeepVerdict(f"src/m{n}.py (lines 1-30)", "done_with", {"working_on": 0.05}, False) for n in range(10)]
    c.verdicts.append(KeepVerdict("src/kept.py (lines 1-30)", "working_on", {"working_on": 0.9}, False))  # judged last
    # Live: a line shows once it has started landing, the newest still being judged.
    ages = {v.path: LANDED for v in c.verdicts[:3]} | {c.verdicts[3].path: 0}
    live = compaction_card(c, ages=ages).plain
    assert "m2.py" in live and "m3.py" in live and "m4.py" not in live and "judging" in live
    assert not settled(c, ages)
    # The cut made and every line landed: the kept read first though judged last, the rest behind a click.
    c.method, c.turns_compacted, c.tokens_before, c.tokens_after = "state", 6, 14_000, 6_000
    ages = {v.path: LANDED for v in c.verdicts}
    assert settled(c, ages)
    lines = compaction_card(c, ages=ages).plain.splitlines()
    reads = [line for line in lines if ".py" in line]
    assert "kept.py" in reads[0] and len(reads) == CARD_LINES and "3 more" in lines[-1]
    assert sum(".py" in line for line in compaction_card(c, expanded=True).plain.splitlines()) == len(c.verdicts)
