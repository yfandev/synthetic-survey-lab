"""A session: one journal, the runs made on it, and the commands clients send.

This is the engine's whole public surface. The headless CLI and the JSON-RPC server are thin clients of
it, and so are IDE and desktop clients later.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from intern.context import TokenMeter, build_view
from intern.decision_failover import BudgetDecisionBackend
from intern.decisions import System1Unavailable
from intern.events import (
    PROTOCOL_VERSION,
    ProcessCheck,
    ProcessEvent,
    ProcessInfo,
    StopReason,
    WindowInfo,
    now_ms,
)
from intern.hooks import RunOutcome
from intern.journal import Entry, EventEntry, Journal, MessageEntry, ModelChangeEntry, WindowCapEntry, recording
from intern.loop import CompactOutcome, EngineDeps, ModelBinding, Run, RunControl, system1_state
from intern.messages import AssistantMessage, ToolResultMessage, Usage, UserMessage
from intern.models.registry import Role
from intern.models.window import recapped
from intern.process_watch import ProcessWatch, waits_or_serves
from intern.processes import duration_text

if TYPE_CHECKING:
    from intern.decisions import DecisionBackend
    from intern.providers.failover import RunBudget
    from intern.session_resources import SessionResources

ModelResolver = Callable[[str], Awaitable[ModelBinding]]
System1Resolver = Callable[[str], "DecisionBackend"]
"""A System 1 choice (a model) to its backend."""

# A cancelled waiter may drop its Session. Retain teardown until it finishes,
# including the active run's cleanup, before releasing owned resources.
_CLOSING_SESSIONS: set[asyncio.Task[None]] = set()


def _close_finished(task: asyncio.Task[None]) -> None:
    _CLOSING_SESSIONS.discard(task)
    if not task.cancelled():
        task.exception()


COMPACTING = "the conversation is being compacted; send it once that is done"
JobNotices = Literal["wake", "quiet"]
"""What a process's end does while no run is in progress: `wake` starts a run that tells the agent (the terminal UI,
where the user sees it happen); `quiet` holds it for the next prompt (ACP, JSON-RPC and -p, whose runs answer a
client's request)."""


class JobStatus(BaseModel):
    """A background job as the terminal UI's strip shows it."""

    command: str
    elapsed_s: int
    verdict: str | None = None
    """System 1's verdict at its latest look; None before the first."""
    next_look_s: float | None = None


def past_jobs(journal: Journal) -> tuple[list[ProcessInfo], list[ProcessInfo]]:
    """A reopened session's jobs, from its journal: each one's last record, and of those, the ones the agent was not
    told ended (its end came after the last message, or was never recorded)."""
    last: dict[str, tuple[ProcessInfo, int]] = {}
    said = -1
    for i, entry in enumerate(journal.entries()):
        if isinstance(entry, MessageEntry):
            said = i
        elif isinstance(entry, EventEntry) and isinstance(e := entry.event, ProcessEvent) and e.process.kind == "job":
            last[e.process.id] = (e.process, i)
    past = [info for info, _ in last.values()]
    return past, [info for info, i in last.values() if i > said or info.state == "running"]


class SessionBusy(Exception):
    """A command that needs an idle session arrived while a run was in progress."""


class SessionIdle(Exception):
    """A command that needs a run in progress arrived while the session was idle."""


class SessionState(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    session_id: str
    running: bool
    model: str
    window: WindowInfo
    leaf_id: str | None
    entries: int
    usage: Usage
    last_reason: StopReason | None
    state: str | None = None
    """The latest State on the active branch: the brief to continue the work from."""


class Session:
    def __init__(
        self,
        deps: EngineDeps,
        resolve_model: ModelResolver | None = None,
        *,
        resources: SessionResources | None = None,
        model_budget: RunBudget | None = None,
        shutdown_timeout_s: float = 5,
        resolve_system1: System1Resolver | None = None,
        system1: str | None = None,
        on_close: Callable[[Session], None] | None = None,
        idle_s: float = 0,
        job_notices: JobNotices = "quiet",
    ) -> None:
        if not math.isfinite(shutdown_timeout_s) or shutdown_timeout_s <= 0:
            raise ValueError("session shutdown timeout must be finite and positive")
        if not math.isfinite(idle_s) or idle_s < 0:
            raise ValueError("session idle time must be finite and not negative")
        self.deps = deps
        # The journal keeps System 1's side of the session too, so a resumed session can show it.
        deps.emit = recording(deps.journal, deps.emit)
        deps.decisions.emit = deps.emit
        self.job_notices: JobNotices = job_notices
        self._ended: list[ProcessInfo] = []
        """Processes that ended while no run was in progress (or after a run's last turn), not yet told."""
        self._waking = False
        """A run `_wake` created has not begun yet: a prompt sent meanwhile runs right after it."""
        self._checks: list[str] = []
        """What System 1 found of the commands running while no run was in progress, not yet told."""
        self._run: Run | None = None
        self._watch: ProcessWatch | None = None
        if deps.processes is not None:
            # Reopened: no job outlives the session that ran it. One that ran then is explained if the agent polls it,
            # and an end it was not told (the session closed with the job running) is told at the next run.
            past, untold = past_jobs(deps.journal)
            deps.processes.remember(past)
            self._ended = untold
            # Commands start and end between runs too (a job the agent left running): reported whenever they do.
            deps.processes.listen(self._on_process)
            self._watch = ProcessWatch(
                deps.processes,
                deps.decisions,
                lambda: system1_state(self.deps),
                self._on_check,
                self._on_look,
                turn=lambda: self._run.turn if self._run is not None else None,
                fail=self._on_system1_failure,
            )
            deps.processes.judge = self._waits_or_serves
        self.meter = TokenMeter()
        self.usage = Usage()
        self.last_outcome: RunOutcome | None = None
        self._resolve_model = resolve_model
        self._resolve_system1 = resolve_system1
        self.system1 = system1 or deps.decisions.backend.name
        """The System 1 choice in force: a model name."""
        self._control: RunControl | None = None
        self._task: asyncio.Task[RunOutcome] | None = None
        self._active_task: asyncio.Task | None = None
        self._run_done = asyncio.Event()
        self._run_done.set()
        self._follow_ups: list[str] = []
        self._compacting = False
        self._resources = resources
        self.model_budget = model_budget
        self._shutdown_timeout_s = shutdown_timeout_s
        self._on_close = on_close
        self._idle_s = idle_s
        self._idle: asyncio.TimerHandle | None = None
        self._runs = 0
        self._handed_runs = 0
        """Runs `on_close` has already seen: a client that keeps a session open (T3 Code) gets it after `idle_s`
        without a run, and the close then runs it only for what came after."""
        self._closing = False
        self._close_task: asyncio.Task[None] | None = None
        self.default_window_cap = deps.system2.window.user_cap
        """The cap System 2 was bound with (`--window`), which a `window_cap` of None follows."""
        self.window_cap: int | None = None
        """The session's own cap on System 2's working window, journaled and kept across model switches."""
        restored = next((e for e in reversed(deps.journal.active_path()) if isinstance(e, WindowCapEntry)), None)
        if restored is not None and restored.cap is not None:
            self.window_cap = restored.cap
            self._recap(deps.system2)

    @property
    def closed(self) -> bool:
        task = self._close_task
        return task is not None and task.done() and not task.cancelled() and task.exception() is None

    def _ensure_open(self) -> None:
        if self._closing:
            raise SessionBusy("the session is closing or closed; create a new session")

    async def __aenter__(self) -> Session:
        self._ensure_open()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self, *, reopen: bool = False) -> None:
        """Stop active work before closing resources; safe to call from another task.

        Closing is permanent. Caller cancellation does not cancel shared teardown;
        a later aclose joins it and observes any cleanup failure. `reopen` closes it to open again on the same
        journal, which is not the session's end: `on_close` (stewarding it into memory) does not run.
        """
        if asyncio.current_task() is self._active_task:
            raise SessionBusy("close the session after its active run returns")
        if self._close_task is None or (
            self._close_task.done() and not self._close_task.cancelled() and self._close_task.exception() is not None
        ):
            if reopen:
                self._on_close = None
            self._cancel_idle()
            self._closing = True
            self._follow_ups.clear()
            self._close_task = asyncio.create_task(self._close(), name="harness-session-close")
            _CLOSING_SESSIONS.add(self._close_task)
            self._close_task.add_done_callback(_close_finished)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._shutdown_timeout_s
        try:
            if self._watch is not None:
                await self._watch.aclose()
            if self._control is not None:
                self._control.abort.set()
            active = self._active_task
            if active is not None:
                try:
                    async with asyncio.timeout_at(deadline):
                        await self._run_done.wait()
                except TimeoutError:
                    active.cancel()
                # Resource teardown must not race an engine finally block still
                # using its tools. Never claim it completed while that work lives.
                # The caller task can outlive run(); join the run boundary rather
                # than the whole host task (which may itself be awaiting aclose).
                await self._run_done.wait()
            elif self._task is not None and not self._task.done():
                self._task.cancel()
            if self._task is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._task
        finally:
            # Before teardown: a client may kill the process soon after asking it to stop.
            try:
                self._hand_over()
            finally:
                if self._resources is not None:
                    await self._resources.aclose()

    @property
    def running(self) -> bool:
        return self._control is not None

    def _on_process(self, event: ProcessEvent) -> None:
        """A process started or ended (the job manager's report): the event goes out, and an end the agent should hear
        of goes to the run in progress, told at its next turn, or waits for the next run, or wakes one."""
        self.deps.emit(event)
        info = event.process
        if event.action == "started" and self._watch is not None and not self._closing:
            self._watch.poke()
        if event.action != "ended" or info.kind == "call" or self._closing:
            return
        if self._control is not None and not self._compacting:
            self._control.ended.append(info)
            self._control.news.set()
            return
        self._ended.append(info)
        if self.job_notices == "wake":
            asyncio.get_running_loop().call_soon(self._wake)

    def _on_look(self, event: ProcessCheck) -> None:
        """System 1 looked at what runs: the event goes out; while the agent has handed over in a held run, a look that
        found nothing to tell is news too, so the agent decides again (a server it polled never ends)."""
        self.deps.emit(event)
        control = self._control
        if control is None or not control.waiting or any(v.told for v in event.verdicts):
            return
        seen = "; ".join(
            f"`{v.command}` {duration_text(v.elapsed_s)}: {v.verdict.replace('_', ' ')}" for v in event.verdicts
        )
        control.looks.append(
            f"System 1 looked at what runs ({seen}). You are told when a job ends: keep working, or poll it again to "
            "keep waiting; terminate it if you no longer need it."
        )
        control.news.set()

    def jobs_status(self) -> list[JobStatus]:
        """The background jobs running now, with System 1's latest verdict on each and when it looks next (the
        terminal UI's strip)."""
        if self.deps.processes is None:
            return []
        looks = self._watch.status() if self._watch is not None else {}
        now = now_ms()
        return [
            JobStatus(
                command=info.command,
                elapsed_s=(now - info.started_at) // 1000,
                verdict=looks.get(info.id, (None, None))[0],
                next_look_s=looks.get(info.id, (None, None))[1],
            )
            for info in self.deps.processes.jobs_running()
        ]

    def _on_check(self, text: str) -> None:
        """What System 1 found of the commands running: told at the run's next turn, or between runs held for
        the next one, or a run of its own (`job_notices = "wake"`)."""
        if self._closing:
            return
        # The latest look supersedes the ones before it: the agent hears only what System 1 found last.
        if self._control is not None and not self._compacting:
            self._control.checks[:] = [text]
            self._control.news.set()
            return
        self._checks[:] = [text]
        if self.job_notices == "wake":
            asyncio.get_running_loop().call_soon(self._wake)

    async def _waits_or_serves(self, command: str) -> tuple[Literal["waits", "serves"], float] | None:
        """System 1's word on a `bash` command still running a few seconds on: a kind moves it to the background now."""
        requests = [
            e.message.content
            for e in self.deps.journal.active_path()
            if isinstance(e, MessageEntry) and isinstance(e.message, UserMessage) and e.message.source == "user"
        ]
        turn = self._run.turn if self._run is not None else None
        return await waits_or_serves(self.deps.decisions, requests, command, turn)

    def _on_system1_failure(self, error: System1Unavailable) -> None:
        """An ask made for the run in progress went unanswered: the run stops with it, as for its own asks."""
        if self._control is not None:
            self._control.failure = error

    def _wake(self) -> None:
        """Start a run that tells the agent what ended while no run was in progress (`job_notices = "wake"`). An end
        the run's own abort caused wakes nothing: the user stopped it."""
        busy = self.running or (self._task is not None and not self._task.done())
        if self._closing or busy or self.deps.processes is None:
            return
        if self._checks or any(info.state != "aborted" for info in self.deps.processes.unseen(self._ended)):
            self._waking = True
            self._task = asyncio.create_task(self.run(None))

    async def run(self, prompt: str | None) -> RunOutcome:
        """Run `prompt` to the end, then any follow-ups queued meanwhile. Returns the last outcome. None runs without a
        user's message, to tell the agent of processes that ended while no run was in progress."""
        self._ensure_open()
        if self.running:
            raise SessionBusy("a run is in progress; use steer or follow_up")
        self._waking = False
        self._cancel_idle()
        self._active_task = asyncio.current_task()
        self._run_done.clear()
        try:
            outcome = await self._run_once(prompt)
            while self._follow_ups and not self._closing:
                outcome = await self._run_once(self._follow_ups.pop(0))
            return outcome
        finally:
            self._active_task = None
            self._run_done.set()
            if self._idle_s and self._on_close is not None and not self._closing:
                self._idle = asyncio.get_running_loop().call_later(self._idle_s, self._went_idle)
            if (self._ended or self._checks) and self.job_notices == "wake" and not self._closing:
                asyncio.get_running_loop().call_soon(self._wake)

    def _cancel_idle(self) -> None:
        if self._idle is not None:
            self._idle.cancel()
            self._idle = None

    def _went_idle(self) -> None:
        self._idle = None
        if not self._closing and not self.running:
            self._hand_over()

    def _hand_over(self) -> None:
        if self._on_close is not None and self._runs > self._handed_runs:
            self._handed_runs = self._runs
            self._on_close(self)

    async def _run_once(self, prompt: str | None) -> RunOutcome:
        # What ended while no run was in progress is told at this run's first turn.
        control = self._control = RunControl(ended=self._ended, checks=self._checks, hold=self.job_notices == "quiet")
        self._ended, self._checks = [], []
        run = self._run = Run(self.deps, self.meter, control)
        try:
            outcome = await run.execute(prompt)
        finally:
            # Steering that arrived after the last turn started becomes the next prompt; an end, the next run's news.
            self._follow_ups[:0] = control.steering
            self._ended[:0] = control.ended
            self._checks[:0] = control.checks
            self._run = None
            self._control = None
            self.usage += run.usage
            self._runs += 1
        self.last_outcome = outcome
        return outcome

    def start(self, prompt: str) -> asyncio.Task[RunOutcome]:
        """Start a run in the background, for clients that keep sending commands while it works."""
        self._ensure_open()
        if self._waking and self._task is not None and not self._task.done():
            # The session is starting a run to tell the agent of a job's end: the prompt runs right after it.
            self._follow_ups.append(prompt)
            return self._task
        if self.running or (self._task is not None and not self._task.done()):
            raise SessionBusy("a run is in progress; use steer or follow_up")
        self._task = asyncio.create_task(self.run(prompt))
        return self._task

    async def wait(self) -> None:
        """Wait for a run started with `start` (and its follow-ups) to finish. A waiter that is cancelled leaves the run
        going: stopping it is `abort`'s or `aclose`'s, which end it with its `run_end`."""
        if self._task is not None:
            await asyncio.shield(self._task)

    async def compact(self) -> CompactOutcome:
        """Compact the conversation now, as a run does once its request passes `compact_at` of the window (see
        `Run.compact`). The session is busy meanwhile, as during a run; `abort` stops it before anything is cut."""
        self._ensure_open()
        if self.running:
            raise SessionBusy("cannot compact while a run is in progress")
        control = self._control = RunControl()
        self._compacting = True
        self._active_task = asyncio.current_task()
        self._run_done.clear()
        run = Run(self.deps, self.meter, control)
        try:
            return await run.compact()
        finally:
            self._control, self._compacting = None, False
            self._active_task = None
            self._run_done.set()
            self.usage += run.usage
            if (self._ended or self._checks) and self.job_notices == "wake" and not self._closing:
                asyncio.get_running_loop().call_soon(self._wake)

    def steer(self, text: str) -> None:
        """Add a user message to the run in progress, before its next model call."""
        self._ensure_open()
        if self._control is None:
            raise SessionIdle("no run is in progress; use prompt")
        if self._compacting:
            raise SessionBusy(COMPACTING)
        self._control.steering.append(text)
        self._control.news.set()
        if self.deps.processes is not None:
            # Never behind a command: one still running goes on in the background, so the agent reads this now.
            self.deps.processes.move_for_user()

    def follow_up(self, text: str) -> asyncio.Task[RunOutcome] | None:
        """Queue `text` to run after the current run; starts it now when the session is idle."""
        self._ensure_open()
        if self._compacting:
            raise SessionBusy(COMPACTING)
        if self.running:
            self._follow_ups.append(text)
            return None
        return self.start(text)

    def abort(self) -> None:
        if self._control is None:
            raise SessionIdle("no run is in progress")
        self._follow_ups.clear()
        self._control.abort.set()

    async def set_model(self, ref: str, role: Role = "system2") -> ModelBinding | None:
        """Switch a role's model while idle and journal the change. System 2 returns its new binding; System 1
        (a model) rebuilds the decision client and returns None."""
        self._ensure_open()
        if role == "system1":
            self._set_system1(ref)
            return None
        if self._resolve_model is None:
            raise ValueError("this session was built without a model resolver")
        if self.running:
            raise SessionBusy("cannot change models while a run is in progress")
        binding = await self._resolve_model(ref)
        self._ensure_open()
        if self.running:
            raise SessionBusy("a run started while the model was resolving; retry while idle")
        if role == "system2":
            self._recap(binding)
            self.deps.system2 = binding
        else:
            raise ValueError(f"role {role!r} cannot be set on a running engine yet")
        self.deps.journal.append(ModelChangeEntry(role=role, model=binding.ref))
        return binding

    def _set_system1(self, ref: str) -> None:
        if self._resolve_system1 is None:
            raise ValueError("this session was built without a System 1 resolver")
        if self.running:
            raise SessionBusy("cannot change models while a run is in progress")
        backend = self._resolve_system1(ref)
        if self.model_budget is not None:
            backend = BudgetDecisionBackend(backend, self.model_budget)
        self.deps.decisions = self.deps.decisions.with_backend(backend)
        self.system1 = ref
        self.deps.journal.append(ModelChangeEntry(role="system1", model=ref))

    def set_window_cap(self, cap: int | None) -> WindowInfo:
        """Cap System 2's working window for this session while idle and journal it; None goes back to the
        default. A cap above what the model can use leaves its window as is, and still holds for the next one."""
        self._ensure_open()
        if cap is not None and cap <= 0:
            raise ValueError("a window cap must be a positive number of tokens")
        if self.running:
            raise SessionBusy("cannot change the window while a run is in progress")
        if cap != self.window_cap:
            self.deps.journal.append(WindowCapEntry(cap=cap))
            self.window_cap = cap
        self._recap(self.deps.system2)
        return self.deps.system2.window

    def _recap(self, binding: ModelBinding) -> None:
        cap = self.default_window_cap if self.window_cap is None else self.window_cap
        binding.window = recapped(binding.window, binding.profile, cap)

    def branch(self, entry_id: str) -> None:
        self._ensure_open()
        if self.running:
            raise SessionBusy("cannot branch while a run is in progress")
        pending: set[str] = set()
        for entry in self.deps.journal.path_to(entry_id):
            if isinstance(entry, MessageEntry):
                match entry.message:
                    case AssistantMessage(tool_calls=calls):
                        pending = {c.id for c in calls}
                    case ToolResultMessage(result=result):
                        pending.discard(result.call_id)
                    case _:
                        pass
        if pending:
            raise ValueError("cannot branch between a tool call and its result; pick a turn boundary")
        self.deps.journal.branch(entry_id)

    def entries(self, after_id: str | None = None) -> list[Entry]:
        entries = self.deps.journal.entries()
        if after_id is None:
            return entries
        ids = [e.id for e in entries]
        return entries[ids.index(after_id) + 1 :] if after_id in ids else entries

    def state(self) -> SessionState:
        path = self.deps.journal.active_path()
        view = build_view(path, self.deps.instructions, self.meter, self.deps.system2.window.working)
        return SessionState(
            session_id=self.deps.journal.session_id,
            running=self.running,
            model=self.deps.system2.ref,
            window=self.deps.system2.window,
            leaf_id=self.deps.journal.leaf_id,
            entries=len(self.deps.journal.entries()),
            usage=self.usage,
            last_reason=self.last_outcome.reason if self.last_outcome else None,
            state=view.state,
        )
