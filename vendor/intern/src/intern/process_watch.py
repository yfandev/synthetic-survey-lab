"""System 1 watches what runs: while any command runs in the session, it is asked, now and then, whether each
one is doing what it should, and one that looks stalled, failing, or up and waited on as if it would end is told to
the agent, so a user need not ask "are you stuck?".

When. Each running command is due for a look at the ages in SCHEDULE_S (30 s, 1, 2, 4, 8 and 16 minutes), then every
EVERY_S (30 minutes): hangs show early, and a command that has run fine for hours is not asked about every minute. When
any is due, one ask covers every command running, over System 1's view of the session, which lists them beside the
goal (`session_cut`'s `processes`). Nothing running, no ask. A command that runs ten hours costs about 25 asks; at a
supervisor ask's measured cost ($0.00027 mean over 2,095 local asks, median 0.54 s) that is under a cent.

What. One choice a command (CHECK). Those whose P(stalled) + P(failing) + P(ready) reach ACT_THRESHOLD are told to
the agent as one harness message (the session delivers it: at the run's next turn, or between runs as `job_notices`
says). There is no counter: a command that stays stuck is told again at its next look, each further apart. Every check
is a `process_check` event with every verdict, which the terminal UI shows. Nothing here stops or moves a command.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Callable, Sequence
from typing import Literal

from intern import names
from intern.decisions import ChoiceAnswer, ChoiceQuestion, DecisionClient, Question, System1Unavailable
from intern.events import ProcessCheck, ProcessVerdict, now_ms
from intern.processes import ProcessManager, RunningProcess, duration_text

SCHEDULE_S: tuple[float, ...] = (30, 60, 120, 240, 480, 960)
EVERY_S = 1800.0
ACT_THRESHOLD = 0.6  # P(stalled) + P(failing) + P(ready) that tells the agent; the guard's cut-off for risk
DUE_SLACK = 0.1  # a look due within this share of its age counts as come at a check

CHECK = {
    "progressing": "it writes output or produces results as a command like it should, and its last output shows "
    "the work moving",
    "waiting_as_expected": "it writes little or nothing, and that is what it does: it waits on something slow (a CI "
    "run, a sleep, a download, a long build or test run) or serves (a server, a watcher) as the session meant it to",
    "stalled": "it has written nothing for longer than a command like it should take, or it waits on something that "
    "will not come: input, a prompt, a lock, a network call that hangs",
    "failing": "its output shows it failing (errors repeating, a crash loop) while it keeps running",
    "ready": "it is a server or watcher that is up and ready, and the agent waits on it as if it would end",
    "unclear": "the session shows too little of it to tell",
}
ACT = frozenset({"stalled", "failing", "ready"})
SAID = {
    "stalled": "looks stalled: it has written nothing for longer than such a command should take, or waits on "
    "something that will not come",
    "failing": "looks to be failing: its output shows errors while it keeps running",
    "ready": "looks up and ready: a server or watcher, which you may be waiting on as if it would end",
}


# A `bash` command still running a few seconds on (JUDGE_AFTER_S): does it work, or wait or serve? One that waits or
# serves goes on in the background at once instead of at `move_after_s`. Quick commands end before the ask.
STARTED = ChoiceQuestion(
    instructions="The agent ran `command` and it is still running a few seconds on; `requests` are what the user "
    "asked. What kind of command is it?",
    criteria={
        "finishes": "it works and ends by itself within a minute or so: a script, a build step, a single test file, a "
        "query, a short install",
        "works_long": "it works and ends by itself, but takes minutes: a whole test suite, a large build or install, a "
        "migration, a data job",
        "waits": "it mostly waits on something outside it: a sleep, watching a CI run or a PR's checks, a poll loop, a "
        "slow download, a lock",
        "serves": "it does not end by itself: a server, a dev server, a watcher, a REPL, `tail -f`",
    },
)
MOVE_NOW = frozenset({"waits", "serves"})
MOVE_THRESHOLD = 0.6  # P(waits) + P(serves) that moves a command at once


async def waits_or_serves(
    decisions: DecisionClient, requests: Sequence[str], command: str, turn: int | None
) -> tuple[Literal["waits", "serves"], float] | None:
    """`waits` or `serves`, with P(waits) + P(serves), when System 1 finds `command` one that does, so it goes on in
    the background now; None for one that works. It sees only the user's requests and the command, as the guard does."""
    state = json.dumps({"requests": list(requests[-3:]), "command": command}, ensure_ascii=False)
    answer = (await decisions.ask("background", state, {"kind": STARTED}, turn=turn, subject=command[:200]))["kind"]
    if not isinstance(answer, ChoiceAnswer):
        return None
    p = answer.probabilities or {answer.choice: 1.0}
    share = sum(p.get(c, 0.0) for c in MOVE_NOW)
    if share < MOVE_THRESHOLD:
        return None
    return ("serves" if p.get("serves", 0.0) > p.get("waits", 0.0) else "waits"), share


def question(p: RunningProcess) -> ChoiceQuestion:
    quiet = f", nothing written for {duration_text(p.quiet_s)}" if p.quiet_s is not None else ""
    return ChoiceQuestion(
        instructions="The state is the agent's session; `processes` lists the commands running now. Look at the one "
        f"with id {p.id} (`{p.command}`), running {duration_text(p.elapsed_s)}{quiet}. Given what the session expects "
        "of it, what is it doing?",
        criteria=CHECK,
    )


def act_probability(answer: ChoiceAnswer) -> float:
    """P(stalled) + P(failing) + P(ready); a choice given without probabilities counts as certain."""
    if not answer.probabilities:
        return 1.0 if answer.choice in ACT else 0.0
    return sum(answer.probabilities.get(choice, 0.0) for choice in ACT)


def worst(answer: ChoiceAnswer) -> str:
    """Which of stalled, failing and ready System 1 put most on."""
    return max(sorted(ACT), key=lambda c: answer.probabilities.get(c, 1.0 if c == answer.choice else 0.0))


def note(flagged: Sequence[tuple[RunningProcess, str]]) -> str:
    """The harness message that tells the agent what System 1 found."""
    lines = ["System 1 looked at the commands running now:"]
    for p, verdict in flagged:
        where = (
            "the command you ran (moved to the background, if moving is on)"
            if p.kind == "call"
            else f"background job {p.id}"
            if p.kind == "job"
            else f"a process a command left running ({p.id})"
        )
        quiet = f", nothing written for {duration_text(p.quiet_s)}" if p.quiet_s is not None else ""
        last = f" Its last line: {p.last_output}" if p.last_output else ""
        lines.append(f"- {where}, `{p.command}` (running {duration_text(p.elapsed_s)}{quiet}), {SAID[verdict]}.{last}")
    lines.append("Check on it now: read its output, stop it, or carry on if you know it should still run.")
    return "\n".join(lines)


class ProcessWatch:
    """The session's watch over what runs: `poke` it when a command starts; it asks System 1 on the schedule, emits a
    `process_check` for every ask and hands `tell` what the agent should hear. `turn` gives the run's turn (None
    between runs); `fail` takes an ask System 1 did not answer during a run, which stops that run."""

    def __init__(
        self,
        processes: ProcessManager,
        decisions: DecisionClient,
        state: Callable[[], str],
        tell: Callable[[str], None],
        emit: Callable[[ProcessCheck], None],
        *,
        turn: Callable[[], int | None],
        fail: Callable[[System1Unavailable], None],
        schedule: Sequence[float] = SCHEDULE_S,
        every: float = EVERY_S,
    ) -> None:
        self.processes, self.decisions, self.state, self.tell, self.emit = processes, decisions, state, tell, emit
        self.turn, self.fail, self.schedule, self.every = turn, fail, tuple(schedule), every
        self._looks: dict[str, int] = {}
        """How many of each command's looks have come, by its id."""
        self._last: dict[str, str] = {}
        """System 1's verdict at each command's latest look, by its id."""
        self._changed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def poke(self) -> None:
        """A command started: watch it (and start watching, the first time)."""
        if self._closed:
            return
        self._changed.set()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._watch(), name="process-watch")

    async def aclose(self) -> None:
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def status(self) -> dict[str, tuple[str | None, float]]:
        """For each command running: System 1's verdict at its latest look (None before the first) and the seconds
        until its next look, for the terminal UI."""
        now = now_ms()
        return {
            p.id: (self._last.get(p.id), max(0.0, self._due(p.id) - (now - p.started_at) / 1000))
            for p in self.processes.running()
        }

    def _due(self, process_id: str) -> float:
        """The age, in seconds, at which a command's next look is due."""
        n = self._looks.get(process_id, 0)
        if n < len(self.schedule):
            return self.schedule[n]
        return self.schedule[-1] + self.every * (n - len(self.schedule) + 1)

    async def _watch(self) -> None:
        while not self._closed:
            running = self.processes.running()
            ids = {p.id for p in running}
            self._looks = {process_id: n for process_id, n in self._looks.items() if process_id in ids}
            self._last = {process_id: v for process_id, v in self._last.items() if process_id in ids}
            if not running:
                self._changed.clear()
                await self._changed.wait()
                continue
            now = now_ms()
            wait = min(self._due(p.id) - (now - p.started_at) / 1000 for p in running)
            if wait > 0:
                self._changed.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._changed.wait(), wait)
                continue
            try:
                await self._check()
            except System1Unavailable as error:
                if self.turn() is not None:
                    self.fail(error)
                self._advance()  # between runs: try again at the next look
            except Exception as error:  # a bug must not end the watch, nor the session; say so and go on
                print(f"{names.NAME}: the process check failed: {type(error).__name__}: {error}", file=sys.stderr)
                self._advance()

    def _advance(self) -> None:
        """Count a look for every command whose next look is due, or nearly (within DUE_SLACK of its step): two that
        started moments apart share one ask rather than two."""
        now = now_ms()
        for p in self.processes.running():
            while self._due(p.id) * (1 - DUE_SLACK) <= (now - p.started_at) / 1000:
                self._looks[p.id] = self._looks.get(p.id, 0) + 1

    async def _check(self) -> None:
        shown = self.processes.for_system1()
        if not shown:
            return
        turn = self.turn()
        questions: dict[str, Question] = {f"p{i}": question(p) for i, p in enumerate(shown)}
        answers = await self.decisions.ask("processes", self.state(), questions, turn=turn)
        self._advance()
        verdicts, flagged = [], []
        for i, p in enumerate(shown):
            answer = answers.get(f"p{i}")
            if not isinstance(answer, ChoiceAnswer):
                continue
            told = act_probability(answer) >= ACT_THRESHOLD
            self._last[p.id] = answer.choice
            if told:
                flagged.append((p, worst(answer)))
            verdicts.append(
                ProcessVerdict(
                    id=p.id,
                    command=p.command,
                    elapsed_s=p.elapsed_s,
                    verdict=answer.choice,
                    probabilities=dict(answer.probabilities),
                    told=told,
                )
            )
        self.emit(ProcessCheck(turn=turn, verdicts=verdicts))
        for p, verdict in flagged:
            if p.kind == "call":
                self.processes.request_move(p.id)  # the agent is blocked in it: it goes on in the background now
            elif verdict == "ready":
                self.processes.release(p.id)  # it serves: no run waits for it to end
        if flagged:
            self.tell(note(flagged))
