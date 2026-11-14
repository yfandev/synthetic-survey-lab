"""System 1 supervises the run: after every turn it says what the recent turns are doing, and being stuck earns a nudge.

Long capped runs spend their turn budget investigating what the session already established, with every call new and
no change ever made. One System 1 call per turn asks the mode of the recent turns (advancing, re-investigating,
retrying, off plan, finished, or unclear) over System 1's State, a cut of the session built by code (`session_cut`).
Unclear is the way out for turns that fit no single mode, so they are not forced into the nearest one; it is not a
stuck verdict, so it breaks a run of strikes like advancing does (drex-max never picked it over 1,485 reviews of 132
stored runs).

Stuck (STUCK: re_investigating and retrying together): every turn's verdict counts. A turn is a strike when System 1
puts P(re_investigating) + P(retrying) at STUCK_THRESHOLD or more; STRIKES in a row earn a nudge that asks for the
change as the next call, and the count starts over, so the agent has at least STRIKES turns to act on it before the
next one, judged like any other turns. The loop waits for the answer at the turn's end (about a second) and the nudge
goes into the next request.

Done (WRAP_UP, its own ask beside the mode's, so neither question's answers move): the request is met and checked, and
the latest turns change or re-check what it did not ask for. WRAP_STRIKES such turns in a row earn one wrap-up nudge
per run. The mode's `finished` does not see it: a done agent that keeps tuning makes changes, which reads as advancing.
The nudge never asks the agent to stop: the same verdict comes mid-run on work a grader fails, and stopping early is
how that work fails. It asks to keep what works working instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from intern.decisions import ChoiceAnswer, ChoiceQuestion, DecisionClient

STRIKES = 3  # stuck verdicts on consecutive turns before a nudge
STUCK_THRESHOLD = 0.8  # P(re_investigating) + P(retrying) that makes a turn a strike
MAX_NUDGES = 3
WRAP_STRIKES = 3  # done verdicts on consecutive turns before the wrap-up nudge, once per run
WRAP_THRESHOLD = 0.8  # P(done_polishing) that makes a turn count toward it

MODE = ChoiceQuestion(
    instructions="The state is the agent's session so far, oldest first; its last few entries are the latest turns. "
    "Compare the latest turns (about the last three replies and their tool calls) with what the session established "
    "before them. What are these turns doing?",
    criteria={
        "advancing": "they produce results the session did not have, and the work moves: a step completes, a change "
        "is made, a test confirms or refutes something",
        "re_investigating": "they read, grep or debug things earlier turns already covered, without a change to the "
        "code or the plan",
        "retrying": "they repeat variants of the same failing attempt",
        "off_plan": "they work on something the user did not ask for and the agent's plan does not need",
        "finished": "the user's request is met; the agent should stop and answer",
        "unclear": "the turns fit two of these equally well, or the session shows too little of them to tell",
    },
)
STUCK = frozenset({"re_investigating", "retrying"})

# It never tells the agent to stop: it fires because the turns are stuck, not because the work is done. "If no change is
# possible, give your answer now instead" and "Stop investigating" ended 8 h tasks at 6-10 minutes.
STUCK_NUDGE = """\
A review of your last {turns} turns found no progress: they re-read, re-ran or retried what your own notes \
already establish, and changed nothing. Do not repeat them. Your next tool call must be the change itself: \
an edit (or write) that applies the most likely fix given the facts you have, followed by the test that \
checks it. A tested wrong change tells you more than another read."""


WRAP_UP = ChoiceQuestion(
    instructions="The state is the agent's session so far, oldest first; its last few entries are the latest turns. "
    "Compare what the user asked with what the session's tool results already show. Which is true now?",
    criteria={
        "working": "a part of the request is not done yet, or not yet shown to work; the latest turns work toward it",
        "done_polishing": "every part of the request is done and shown to work on the real output, with each file the "
        "request names in place, and the latest turns change or re-check things the request did not ask for (speed, "
        "style, extra cases, refactoring, more tests)",
        "done_unsaved": "a working solution has been shown, but an output the request names (a file, a value in a "
        "file) is not yet where and how the request asks for it",
        "unclear": "the session shows too little to tell",
    },
)

WRAP_NUDGE = """\
A review of your last {turns} turns found the request done and checked: each part it asks for is in place and shown \
to work, and these turns change or re-check things it did not ask for. Keep what works: from here, make any further \
change on a copy, check it, and replace the original only once the check passes, so the deliverable works whenever \
the run ends. If a part of the request is not met after all, fix that first."""


def stuck_probability(choice: str, probabilities: Mapping[str, float]) -> float:
    """P(re_investigating) + P(retrying); a choice given without probabilities counts as certain."""
    if not probabilities:
        return 1.0 if choice in STUCK else 0.0
    return sum(probabilities.get(mode, 0.0) for mode in STUCK)


def is_strike(choice: str, probabilities: Mapping[str, float]) -> bool:
    """Whether a turn's mode verdict counts toward a stuck nudge (the TUI's ledger counts with this too)."""
    return stuck_probability(choice, probabilities) >= STUCK_THRESHOLD


def done_probability(answer: object) -> float:
    """P(done_polishing); a choice given without probabilities counts as certain."""
    if not isinstance(answer, ChoiceAnswer):
        return 0.0
    if not answer.probabilities:
        return 1.0 if answer.choice == "done_polishing" else 0.0
    return answer.probabilities.get("done_polishing", 0.0)


class Supervisor:
    """Per-run counters for the stuck and wrap-up nudges; `after_turn` asks System 1 and returns the nudge and its
    reason, if one is due."""

    def __init__(self) -> None:
        self.strikes = 0
        self.nudges = 0
        self.done_turns = 0
        self.wrapped = False

    async def after_turn(self, turn: int, decisions: DecisionClient, state: str) -> tuple[str, str] | None:
        asks = [decisions.ask("supervise", state, {"mode": MODE}, turn=turn)]
        if not self.wrapped:
            asks.append(decisions.ask("wrap_up", state, {"wrap": WRAP_UP}, turn=turn))
        answers = await asyncio.gather(*asks)
        mode = answers[0]["mode"]
        if not self.wrapped:
            self.done_turns = self.done_turns + 1 if done_probability(answers[1].get("wrap")) >= WRAP_THRESHOLD else 0
            if self.done_turns >= WRAP_STRIKES:
                self.wrapped = True
                self.strikes = 0
                return WRAP_NUDGE.format(turns=WRAP_STRIKES), "wrap_up"
        if self.nudges >= MAX_NUDGES:
            return None
        strike = isinstance(mode, ChoiceAnswer) and is_strike(mode.choice, mode.probabilities)
        self.strikes = self.strikes + 1 if strike else 0
        if self.strikes < STRIKES:
            return None
        self.strikes = 0
        self.nudges += 1
        return STUCK_NUDGE.format(turns=STRIKES), "stuck_gate"
