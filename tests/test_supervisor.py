from __future__ import annotations

import json
import re

import pytest

from intern.decisions import Answers, ChoiceAnswer
from intern.events import Decision, Nudge, State
from intern.messages import UserMessage
from intern.models.profile import Profile
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn
from intern.supervise import WRAP_NUDGE, WRAP_STRIKES

from .support import is_state_write


class Review:
    """System 1 answering the mode: a scripted mode per turn from the first (the last one repeating), each a bare
    choice or a (choice, probabilities) pair, and the same for the wrap-up question (`wraps`, "working" if none).
    It reads the turn off the session's last call."""

    name = "review"

    def __init__(self, *modes: str | tuple[str, dict[str, float]], wraps: tuple[str, ...] = ("working",)) -> None:
        self.modes = list(modes) or ["advancing"]
        self.wraps = list(wraps)
        self.asks: list[tuple[int, str]] = []
        self.wrap_asks: list[int] = []

    async def decide(self, state: str, questions: dict) -> Answers:
        if "wrap" in questions:
            self.wrap_asks.append(int(re.findall(r"turn (\d+)", state)[-1]))
            wrap = self.wraps.pop(0) if len(self.wraps) > 1 else self.wraps[0]
            return Answers(answers={"wrap": ChoiceAnswer(choice=wrap, probabilities={})})
        if "mode" not in questions:  # the tool-call guard's ask: no opinion
            return Answers(answers={})
        turn = int(re.findall(r"turn (\d+)", state)[-1])
        self.asks.append((turn, state))
        mode = self.modes.pop(0) if len(self.modes) > 1 else self.modes[0]
        choice, probabilities = (mode, {}) if isinstance(mode, str) else mode
        return Answers(answers={"mode": ChoiceAnswer(choice=choice, probabilities=probabilities)})


def churn():
    turns: list[int] = []

    def script(request: ProviderRequest) -> Turn:
        if is_state_write(request):
            return Turn(text="## Plan\n- steps\n## Now\nthe next step")
        turns.append(1)
        return Turn(tool_calls=[("bash", {"command": f"python repro.py --turn {len(turns)}"})])

    return script


async def test_three_stuck_turns_in_a_row_nudge_the_model_at_once_and_the_count_starts_over(make_harness):
    system1 = Review("re_investigating")
    harness = make_harness(churn(), window=200_000, profile=Profile(max_turns=13), system1=system1)
    await harness.run("fix the bug")
    # One ask per turn from the first, answered and acted on at that turn's end: strikes at every turn, a nudge
    # at 3, 6 and 9 (the count starts over after each), none after the third.
    assert [t for t, _ in system1.asks] == list(range(1, 14))
    assert all(e.ok for e in harness.recorder.of(Decision) if e.purpose == "supervise")
    assert [n.turn for n in harness.recorder.of(Nudge) if n.reason == "stuck_gate"] == [3, 6, 9]
    nudged = harness.provider.requests[3].messages[-1]
    assert isinstance(nudged, UserMessage) and "Your next tool call must be the change" in nudged.content
    assert '"goal":"fix the bug"' in system1.asks[0][1] and "repro.py --turn 1" in system1.asks[0][1]


@pytest.mark.parametrize("break_mode", ["advancing", "unclear"])
async def test_a_turn_that_is_not_stuck_resets_the_strikes(make_harness, break_mode):
    stuck = ("retrying", "re_investigating")
    system1 = Review(*stuck, break_mode, *stuck, break_mode, *stuck, "advancing")
    harness = make_harness(churn(), window=200_000, profile=Profile(max_turns=12), system1=system1)
    await harness.run("fix the bug")
    # The breaking verdict is taken as System 1 gave it, not replaced by the no-opinion fallback.
    reviews = [e.answers for e in harness.recorder.of(Decision) if e.purpose == "supervise" and e.answers]
    assert break_mode in [answers["mode"]["choice"] for answers in reviews]
    assert [n.reason for n in harness.recorder.of(Nudge)] == ["turn_budget"]


async def test_a_strike_needs_the_two_stuck_modes_together_at_the_threshold(make_harness):
    unsure = ("re_investigating", {"re_investigating": 0.5, "retrying": 0.2, "advancing": 0.3})  # 0.7: no strike
    split = ("advancing", {"advancing": 0.15, "re_investigating": 0.45, "retrying": 0.4})  # 0.85: a strike
    system1 = Review(unsure, unsure, unsure, unsure, split, split, split, "advancing")
    harness = make_harness(churn(), window=200_000, profile=Profile(max_turns=10), system1=system1)
    await harness.run("fix the bug")
    assert [n.turn for n in harness.recorder.of(Nudge) if n.reason == "stuck_gate"] == [7]


async def test_done_turns_in_a_row_send_one_wrap_up_nudge(make_harness):
    # Two done turns, then a working one: no nudge. Three done in a row from turn 4: the wrap-up at 6, once; System 1
    # is not asked again after it, and the stuck nudge's count is untouched by it.
    done = "done_polishing"
    system1 = Review(wraps=(done, done, "working", done))
    harness = make_harness(churn(), window=200_000, profile=Profile(max_turns=10), system1=system1)
    await harness.run("fix the bug")
    assert [(n.turn, n.reason) for n in harness.recorder.of(Nudge) if n.reason != "turn_budget"] == [(6, "wrap_up")]
    assert system1.wrap_asks == list(range(1, 7))
    nudged = harness.provider.requests[6].messages[-1]
    assert isinstance(nudged, UserMessage) and nudged.content == WRAP_NUDGE.format(turns=WRAP_STRIKES)


async def test_no_state_is_written_between_compactions_and_system_1_reads_the_whole_session(make_harness):
    system1 = Review()
    harness = make_harness(churn(), window=200_000, profile=Profile(max_turns=15), system1=system1)
    await harness.run("do the steps")
    # System 2 writes the State only at a compaction; System 1 reads the session itself, its first turn included.
    assert harness.recorder.of(State) == []
    assert not any(is_state_write(r) for r in harness.provider.requests)
    turn, state = system1.asks[-1]
    calls = [c["input"] for entry in json.loads(state)["history"] for c in entry.get("tool_calls", [])]
    assert turn == 15 and len(calls) == 15 and '--turn 1"' in calls[0]
