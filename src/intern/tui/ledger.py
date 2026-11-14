"""System 1's decisions as a ledger the TUI shows: one judgement per turn, one per compaction, one per tool call
the guard asked about, per run.

Built from the event stream alone (`run_start`, `decision`, `state_writing`, `state_delta`, `state`, `nudge`,
`compaction`, `tool_guard`), the
way the stream is meant to be audited: a stuck strike, a dropped read, a State written or a refused call can be
followed from it. Stuck strikes are counted the way `supervise.py` counts them, from its constants, so the row
that earned a nudge says so; the guard's verdict is read the way `guard.py` reads it, from its thresholds.
Asks for other purposes (code search) are only counted; showing them is a follow-up.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from intern.events import (
    Compaction,
    ContextBudget,
    Decision,
    Event,
    Nudge,
    RunStart,
    State,
    StateDelta,
    StateWriting,
    ToolGuard,
)
from intern.evidence import WORKING
from intern.guard import ASKED_THRESHOLD, RISK_THRESHOLD, RISKS
from intern.messages import JsonObject
from intern.supervise import MAX_NUDGES, STRIKES, STUCK, is_strike

SUPERVISE = "supervise"
KEEP = "keep"
GUARD = "guard"
WORKING_ON = "working_on"
REASONS = tuple(WORKING.criteria)
"""System 1's choices for a read being cut, in the question's order: kept, then the two reasons to cut it."""


@dataclass
class TurnJudgement:
    """The supervisor's one call at a turn's end: what the latest turns are doing; then what the loop did about it."""

    turn: int
    mode: str
    probabilities: dict[str, float]
    backend: str
    cached: bool
    duration_ms: int
    cost_usd: float
    strikes: int | None = None
    """Consecutive stuck verdicts, this one included; None once the run's nudges are spent."""
    nudged: bool = False
    """A stuck nudge went into the next request."""

    @property
    def stuck(self) -> bool:
        return self.mode in STUCK

    @property
    def confidence(self) -> float:
        return self.probabilities.get(self.mode, 0.0)


@dataclass
class KeepVerdict:
    path: str
    choice: str
    probabilities: dict[str, float]
    cached: bool

    @property
    def kept(self) -> bool:
        return self.choice == WORKING_ON

    @property
    def keep(self) -> float:
        """P(working on): how close the read came to being kept, whatever won."""
        return self.probabilities.get(WORKING_ON, 1.0 if self.kept else 0.0)


@dataclass
class GuardJudgement:
    """System 1's judgement of one tool call before it ran: what the call would do (routine, one of the risks, or
    unclear) and whether the user asked for it; then whether the guard refused the call on it."""

    turn: int
    subject: str
    """The tool and its arguments, as the guard named them."""
    kind: str
    probabilities: dict[str, float]
    asked: float
    backend: str
    cached: bool
    duration_ms: int
    cost_usd: float
    refused: bool = False
    reason: str = ""

    @property
    def tool(self) -> str:
        return self.subject.split(" ", 1)[0]

    @property
    def risk(self) -> float:
        """P(any risk): the risks' probabilities together, the way the guard reads them."""
        return min(1.0, sum(self.probabilities.get(name, 0.0) for name in RISKS))

    @property
    def would_block(self) -> bool:
        return self.risk >= RISK_THRESHOLD and self.asked < ASKED_THRESHOLD


@dataclass
class RuleRefusal:
    """A fixed rule refused a tool call; no model was asked."""

    turn: int
    call_id: str
    name: str
    rule: str
    reason: str


@dataclass
class CompactionJudgement:
    """A compaction: System 2's summary being written, the keep-or-cut verdict on every read being cut, then the
    cut itself."""

    turn: int
    verdicts: list[KeepVerdict] = field(default_factory=list)
    writing: bool = False
    """System 2 is writing its summary (the State) for this compaction."""
    draft: str = ""
    """The write's text so far, as it streams (edits to a standing summary do not stream)."""
    state_written: bool = False
    """A State was written for this compaction, because the standing one did not cover the cut."""
    method: str | None = None
    turns_compacted: int = 0
    kept: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    summary: str = ""
    """What the agent continues from: the State, or with `drop` the list of the dropped calls."""

    @property
    def done(self) -> bool:
        return self.method is not None

    def kept_reads(self) -> list[KeepVerdict]:
        return [v for v in self.verdicts if v.kept]

    def cut_reads(self) -> dict[str, list[KeepVerdict]]:
        """The reads cut, by System 1's reason, the closest to being kept first: the reasons in the order the
        question gives them, so `reference` (read to understand) comes before `done with`."""
        by_reason: dict[str, list[KeepVerdict]] = {}
        for v in sorted((v for v in self.verdicts if not v.kept), key=lambda v: -v.keep):
            by_reason.setdefault(v.choice, []).append(v)
        order = {reason: n for n, reason in enumerate(REASONS)}
        return dict(sorted(by_reason.items(), key=lambda kv: order.get(kv[0], len(order))))


PanelRow = TurnJudgement | CompactionJudgement
"""What the System 1 panel lists, turn by turn."""
Row = PanelRow | GuardJudgement | RuleRefusal
"""What `observe` returns: a panel row, or a guard verdict that goes under its tool call in the transcript."""


@dataclass
class RunLedger:
    run_id: str
    turns: dict[int, TurnJudgement] = field(default_factory=dict)
    compactions: dict[int, CompactionJudgement] = field(default_factory=dict)
    guards: list[GuardJudgement] = field(default_factory=list)
    refusals: int = 0
    """Calls the guard refused, by a rule or on System 1's judgement."""
    other_asks: Counter[str] = field(default_factory=Counter)
    strikes: int = 0
    nudges: int = 0
    context: dict[int, int] = field(default_factory=dict)
    """System 2's request at each turn, in tokens: measured as it is sent, or on resume the reply's prompt."""

    def rows(self) -> list[PanelRow]:
        """Turn and compaction rows in turn order, a compaction after its turn's judgement."""
        rows: list[PanelRow] = [*self.turns.values(), *self.compactions.values()]
        return sorted(rows, key=lambda r: (r.turn, isinstance(r, CompactionJudgement)))


@dataclass
class Ledger:
    runs: list[RunLedger] = field(default_factory=list)

    @property
    def current(self) -> RunLedger:
        if not self.runs:
            self.runs.append(RunLedger(run_id=""))
        return self.runs[-1]

    def observe(self, event: Event) -> Row | None:
        """Take one event; the row it created or changed, for the panel to redraw, or None."""
        match event:
            case RunStart(run_id=run_id):
                self.runs.append(RunLedger(run_id=run_id))
            case ContextBudget(turn=turn, tokens=tokens):
                self.current.context[turn] = tokens
            case Decision(ok=False):
                pass  # a failed attempt, retried; the answer that follows is the one shown
            case Decision(purpose="supervise", turn=int() as turn, answers=dict() as answers):
                return self._supervised(turn, event, answers)
            case Decision(purpose="keep", turn=int() as turn, subject=str() as path, answers=dict() as answers):
                choice, probabilities = _choice(answers.get("w"))
                verdict = KeepVerdict(path, choice, probabilities, cached=event.cached)
                compaction = self._compaction(turn)
                compaction.verdicts = [v for v in compaction.verdicts if v.path != path] + [verdict]
                return compaction
            case Decision(purpose="guard", turn=int() as turn, subject=str() as subject, answers=dict() as answers):
                return self._guarded(turn, subject, event, answers)
            case Decision(purpose=purpose):
                self.current.other_asks[purpose] += 1
            case ToolGuard(layer="system1", reason=reason):
                self.current.refusals += 1
                if (judged := self.current.guards[-1:]) and not judged[0].refused:  # the one just asked about
                    judged[0].refused = True
                    judged[0].reason = reason
                    return judged[0]
            case ToolGuard(layer="rule", turn=turn, call_id=call_id, name=name, rule=rule, reason=reason):
                self.current.refusals += 1
                return RuleRefusal(turn=turn, call_id=call_id, name=name, rule=rule, reason=reason)
            case StateWriting(turn=turn):
                compaction = self._compaction(turn)
                compaction.writing, compaction.draft = True, ""
                return compaction
            case StateDelta(turn=turn, text=text):
                self._compaction(turn).draft += text  # the card's own frames draw it, not a redraw per delta
            case State(turn=turn):
                compaction = self._compaction(turn)
                compaction.state_written, compaction.writing, compaction.draft = True, False, ""
                return compaction
            case Nudge(reason="stuck_gate", turn=turn):
                if judgement := self.current.turns.get(turn):
                    judgement.nudged = True
                    return judgement
            case Compaction(turn=turn):
                compaction = self._compaction(turn)
                compaction.method = event.method
                compaction.turns_compacted = event.turns_compacted
                compaction.kept = event.kept
                compaction.tokens_before = event.tokens_before
                compaction.tokens_after = event.tokens_after
                compaction.summary = event.summary
                return compaction
            case _:
                pass
        return None

    def _compaction(self, turn: int) -> CompactionJudgement:
        return self.current.compactions.setdefault(turn, CompactionJudgement(turn=turn))

    def _guarded(self, turn: int, subject: str, event: Decision, answers: Mapping[str, JsonObject]) -> GuardJudgement:
        kind, probabilities = _choice(answers.get("risk"))
        if not probabilities and kind != "?":  # a bare choice, the way the guard reads one
            probabilities = {kind: 1.0}
        asked = answers.get("asked")
        judgement = GuardJudgement(
            turn=turn,
            subject=subject,
            kind=kind,
            probabilities=probabilities,
            asked=_number(asked.get("noul")) if isinstance(asked, dict) else 0.0,
            backend=event.backend,
            cached=event.cached,
            duration_ms=event.duration_ms,
            cost_usd=event.cost_usd,
        )
        self.current.guards.append(judgement)
        return judgement

    def _supervised(self, turn: int, event: Decision, answers: Mapping[str, JsonObject]) -> TurnJudgement:
        run = self.current
        mode, probabilities = _choice(answers.get("mode"))
        judgement = TurnJudgement(
            turn=turn,
            mode=mode,
            probabilities=probabilities,
            backend=event.backend,
            cached=event.cached,
            duration_ms=event.duration_ms,
            cost_usd=event.cost_usd,
        )
        # Strikes as the supervisor counts them: every turn, consecutive, starting over after a nudge.
        if run.nudges < MAX_NUDGES:
            run.strikes = run.strikes + 1 if is_strike(mode, probabilities) else 0
            judgement.strikes = run.strikes
            if run.strikes >= STRIKES:
                run.strikes = 0
                run.nudges += 1
        run.turns[turn] = judgement
        return judgement


def _number(value: object) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0


def _choice(answer: object) -> tuple[str, dict[str, float]]:
    """A compact choice answer's option and probabilities; `?` and none when the event carries no answer."""
    if not isinstance(answer, dict):
        return "?", {}
    choice = answer.get("choice")
    raw = answer.get("probabilities")
    probabilities = {str(k): _number(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    return (choice if isinstance(choice, str) else "?"), probabilities
