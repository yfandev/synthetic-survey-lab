"""The terminal UI's pure parts: the State outline and the System 1 ledger. No Textual needed."""

from __future__ import annotations

import pytest

from intern.events import Compaction, Decision, Nudge, RunStart, State, ToolGuard, WindowInfo
from intern.guard import RISK_THRESHOLD
from intern.messages import JsonObject
from intern.supervise import STRIKES
from intern.tui.ledger import CompactionJudgement, GuardJudgement, Ledger, RuleRefusal, TurnJudgement
from intern.tui.outline import parse_state, step_status

SAMPLE_STATE = """\
## Plan
1. Locate the failing test — done: tests/test_parser.py::test_dates fails on ISO weeks.
2. Fix the parser (in progress): add week handling to `parse_date`,
   which lives in src/parser.py.
3. Run the suite — pending.
## Done when
`pytest -q` passes.
## Now
Editing src/parser.py: the next action is to add the %V branch.
## Artifacts
- src/parser.py: the date parser
- tests/test_parser.py: the failing test
## Facts
- The failing input is 2026-W05-3.
"""

MODES = ("advancing", "re_investigating", "retrying", "off_plan", "finished")


def run_start(run_id: str) -> RunStart:
    window = WindowInfo(advertised=32_768, advertised_source="profile", effective=None, user_cap=None, working=32_768)
    return RunStart(session_id="s", run_id=run_id, model="fake/model", cwd=".", window=window, tools=[])


def supervise(turn: int, mode: str, *, ok: bool = True, error: str | None = None) -> Decision:
    probabilities: JsonObject = {m: (0.8 if m == mode else 0.05) for m in MODES}
    answers: dict[str, JsonObject] = {"mode": {"choice": mode, "probabilities": probabilities}}
    return Decision(
        purpose="supervise",
        backend="decisions-api:drex-latest",
        questions=1,
        ok=ok,
        cached=False,
        cost_usd=0.0003 if ok else 0.0,
        duration_ms=420,
        error=error,
        turn=turn,
        answers=answers,
    )


def guard(turn: int, subject: str, kind: str, asked: float, *, p: float = 0.96) -> Decision:
    """The guard's ask about one call: `kind` at `p` (routine, a risk, or unclear) and P(the user asked)."""
    probabilities: JsonObject = {"routine": round(1 - p, 2)} if kind != "routine" else {"unclear": round(1 - p, 2)}
    probabilities[kind] = p
    answers: dict[str, JsonObject] = {
        "risk": {"choice": kind, "probabilities": probabilities},
        "asked": {"noul": asked},
    }
    return Decision(
        purpose="guard",
        backend="decisions-api:nace-ai/drex-max",
        questions=2,
        ok=True,
        cached=False,
        cost_usd=0.00004,
        duration_ms=380,
        turn=turn,
        subject=subject,
        answers=answers,
    )


def test_the_ledger_follows_the_guard_call_by_call():
    ledger = Ledger()
    ledger.observe(run_start("r1"))
    allowed = ledger.observe(guard(2, 'bash {"command": "pytest -q"}', "routine", 0.8))
    assert isinstance(allowed, GuardJudgement) and (allowed.tool, allowed.kind) == ("bash", "routine")
    assert allowed.risk < RISK_THRESHOLD and not allowed.would_block and not allowed.refused
    judged = ledger.observe(guard(3, 'bash {"command": "git push --force"}', "rewrites_shared_history", 0.02))
    assert isinstance(judged, GuardJudgement) and judged.risk >= RISK_THRESHOLD and judged.would_block
    refused = ToolGuard(
        turn=3,
        call_id="c3",
        name="bash",
        layer="system1",
        rule="rewrites_shared_history",
        reason="System 1 judged that it rewrites history and that the user did not ask for it",
        risk=0.96,
        asked=0.02,
    )
    assert ledger.observe(refused) is judged and judged.refused and judged.reason.startswith("System 1 judged")
    rule = ToolGuard(turn=4, call_id="c4", name="bash", layer="rule", rule="force_push", reason="a force push")
    row = ledger.observe(rule)
    assert isinstance(row, RuleRefusal) and (row.call_id, row.rule, row.reason) == ("c4", "force_push", "a force push")
    assert ledger.current.refusals == 2 and len(ledger.current.guards) == 2
    assert ledger.current.rows() == []  # the guard's verdicts go under their calls, not on the panel


def keep(turn: int, path: str, choice: str) -> Decision:
    return Decision(
        purpose="keep",
        backend="decisions-api:drex-latest",
        questions=1,
        ok=True,
        cached=False,
        cost_usd=0.0001,
        duration_ms=90,
        turn=turn,
        subject=path,
        answers={"w": {"choice": choice, "probabilities": {choice: 0.71}}},
    )


# The State outline


def test_the_state_outline_has_its_sections_and_the_plan_s_steps_with_their_status():
    outline = parse_state(SAMPLE_STATE)
    assert [s.title for s in outline.sections] == ["Plan", "Done when", "Now", "Artifacts", "Facts"]
    assert [(s.status, s.depth) for s in outline.plan] == [("done", 0), ("in_progress", 0), ("pending", 0)]
    assert outline.plan[0].text.startswith("Locate the failing test")
    # A wrapped item is one step.
    assert outline.plan[1].text.endswith("which lives in src/parser.py.")
    assert outline.counts() == {"done": 1, "in_progress": 1, "pending": 1, "unknown": 0}
    artifacts = outline.section("artifacts")
    assert artifacts is not None
    assert artifacts.lines == ["- src/parser.py: the date parser", "- tests/test_parser.py: the failing test"]


@pytest.mark.parametrize(
    ("text", "status"),
    [
        ("[x] read the tests", "done"),
        ("Step one (done): found it", "done"),
        ("Completed the migration", "done"),
        ("Fix parser — in progress", "in_progress"),
        ("Started on the fix", "in_progress"),
        ("Write tests — pending", "pending"),
        ("Deploy: not started", "pending"),
        ("- [ ] ship it", "pending"),
        ("Finished the fix, pending review", "pending"),
        ("Refactor the module", "unknown"),
    ],
)
def test_a_step_s_words_give_its_status(text: str, status: str):
    assert step_status(text) == status


def test_a_state_in_another_shape_still_shows_as_lines_and_sub_steps_keep_their_depth():
    outline = parse_state("Some notes without headings\n- a bullet\n\n# Plan:\nfirst line\nsecond line\n")
    assert outline.sections[0].title == ""
    assert outline.sections[0].lines == ["Some notes without headings", "- a bullet"]
    assert [s.text for s in outline.plan] == ["first line", "second line"]
    nested = parse_state("## Plan\n- **step one** (done)\n  - sub-step, pending\n- step two\n")
    assert [(s.text, s.status, s.depth) for s in nested.plan] == [
        ("step one (done)", "done", 0),
        ("sub-step, pending", "pending", 1),
        ("step two", "unknown", 0),
    ]
    assert parse_state("").sections == []


# The System 1 ledger


def test_the_ledger_follows_the_supervisor_turn_by_turn_and_what_the_loop_did():
    ledger = Ledger()
    ledger.observe(run_start("r1"))
    row = ledger.observe(supervise(3, "advancing"))
    assert isinstance(row, TurnJudgement)
    assert (row.mode, row.confidence, row.strikes) == ("advancing", 0.8, 0)
    # Every turn counts: consecutive stuck verdicts are strikes, the third earns the nudge, the count starts over.
    first = ledger.observe(supervise(10, "re_investigating"))
    second = ledger.observe(supervise(11, "re_investigating"))
    assert isinstance(first, TurnJudgement) and isinstance(second, TurnJudgement)
    assert (first.strikes, second.strikes) == (1, 2)
    third = ledger.observe(supervise(12, "retrying"))
    assert isinstance(third, TurnJudgement) and third.stuck and third.strikes == STRIKES and not third.nudged
    nudge = Nudge(turn=12, reason="stuck_gate", text="Stop investigating.")
    assert ledger.observe(nudge) is third and third.nudged
    restart = ledger.observe(supervise(13, "retrying"))
    assert isinstance(restart, TurnJudgement) and restart.strikes == 1
    cleared = ledger.observe(supervise(14, "advancing"))
    assert isinstance(cleared, TurnJudgement) and cleared.strikes == 0
    # A failed attempt is retried: it makes no row, the answer that follows does.
    assert ledger.observe(supervise(41, "advancing", ok=False, error="TimeoutError: ")) is None
    assert 41 not in ledger.current.turns
    ledger.observe(supervise(41, "advancing"))
    # A second run starts its own ledger; the first is kept.
    ledger.observe(run_start("r2"))
    assert len(ledger.runs) == 2 and ledger.current.turns == {} and 41 in ledger.runs[0].turns
    assert ledger.observe(supervise(3, "finished")) is ledger.current.turns[3]


def test_the_ledger_groups_keep_verdicts_by_compaction_and_only_counts_other_asks():
    ledger = Ledger()
    ledger.observe(run_start("r1"))
    row = ledger.observe(keep(9, "src/a.py", "working_on"))
    assert isinstance(row, CompactionJudgement) and not row.done and row.turn == 9
    ledger.observe(keep(9, "src/b.py", "reference"))
    ledger.observe(keep(9, "src/b.py", "done_with"))  # judged again: the latest verdict stands
    for_it = State(turn=9, landed=9, turns_covered=6, tokens_before=5000, text="## Plan")
    assert ledger.observe(for_it) is row and row.state_written
    cut = Compaction(
        turn=9, method="state", tokens_before=14_000, tokens_after=6_000, turns_compacted=6, summary="", kept=1
    )
    assert ledger.observe(cut) is row
    assert (row.done, row.kept, row.turns_compacted, row.tokens_before, row.tokens_after) == (True, 1, 6, 14_000, 6_000)
    assert [(v.path, v.choice, v.kept) for v in row.verdicts] == [
        ("src/a.py", "working_on", True),
        ("src/b.py", "done_with", False),
    ]
    other = Decision(purpose="locate.files", backend="d", questions=4, ok=True, cached=False, cost_usd=0, duration_ms=1)
    assert ledger.observe(other) is None and ledger.current.other_asks == {"locate.files": 1}
    ledger.observe(supervise(9, "advancing"))
    assert [type(r).__name__ for r in ledger.current.rows()] == ["TurnJudgement", "CompactionJudgement"]
    # An event about a turn the ledger never saw changes nothing.
    assert ledger.observe(Nudge(turn=99, reason="stuck_gate", text="x")) is None
