"""One session's async resources, entered and exited by one dedicated task.

MCP's AnyIO contexts cannot be entered by Bootstrap and exited by another task.
``open(setup)`` runs the entire sequential setup on a persistent owner; that same
owner later unwinds its AsyncExitStack. Tool calls still run on their usual tasks.
The host must stop active session work before calling ``aclose()``.

Timeouts bound caller waits and request cooperative cancellation. Python cannot
force a context manager that suppresses cancellation to finish. Such an owner is
kept tracked, reports incomplete cleanup, and can be joined by a later aclose().
No resource, task, future or event is created merely by constructing this object.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from typing import Literal, TypeVar, cast

from pydantic import BaseModel

ResourceState = Literal["new", "opening", "open", "closing", "closed", "incomplete"]
T = TypeVar("T")

# asyncio keeps weak task references. Retain unfinished owners even if a caller
# drops a session after a cleanup timeout. Resources themselves are never shared.
_ACTIVE_OWNERS: set[asyncio.Task[None]] = set()


class SessionResourceReport(BaseModel):
    state: ResourceState
    owner_done: bool
    cleanup_complete: bool
    setup_error_type: str | None = None
    owner_error_type: str | None = None
    cleanup_error_type: str | None = None
    timeout_reached: bool = False


class ResourceCleanupError(RuntimeError):
    """Cleanup failed or is still pending; details contain types, never raw errors."""

    def __init__(self, report: SessionResourceReport) -> None:
        self.report = report.model_copy(deep=True)
        kind = report.cleanup_error_type or report.owner_error_type or "cleanup is still pending"
        super().__init__(f"Session resource cleanup incomplete ({kind})")


class ResourceSetupError(RuntimeError):
    def __init__(self, error_type: str, report: SessionResourceReport) -> None:
        self.error_type = error_type
        self.report = report.model_copy(deep=True)
        super().__init__(f"Session resource setup failed ({error_type})")


@dataclass(frozen=True)
class _OpenResult:
    value: object = None
    error_type: str | None = None


class SessionResources:
    """Single-use resource owner; all callers must use the same event loop.

    setup receives an AsyncExitStack: enter contexts directly on it, and register
    aclose callbacks for lazy services such as the browser/process manager. Do not
    create separate tasks to enter/exit task-affine contexts. Failed __aenter__
    methods must clean their own partial startup, as required by that protocol.

    ``aclose`` is concurrent/idempotent and returns None only after successful
    cleanup. Cancellation of a close waiter leaves the owner running. A timeout
    raises ResourceCleanupError with state=incomplete; owner_task/report expose
    whether cleanup remains live. A late successful close can subsequently join.
    """

    def __init__(self, *, startup_timeout_s: float = 30, cleanup_timeout_s: float = 10) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (startup_timeout_s, cleanup_timeout_s)):
            raise ValueError("resource timeouts must be finite and positive")
        self.startup_timeout_s = startup_timeout_s
        self.cleanup_timeout_s = cleanup_timeout_s
        self._state: ResourceState = "new"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[_OpenResult] | None = None
        self._finished: asyncio.Future[None] | None = None
        self._stop: asyncio.Event | None = None
        self._started = False
        self._setup_complete = False
        self._close_requested = False
        self._cleanup_complete = False
        self._setup_error_type: str | None = None
        self._owner_error_type: str | None = None
        self._cleanup_error_type: str | None = None
        self._timeout_reached = False

    @property
    def owner_task(self) -> asyncio.Task[None] | None:
        """Tracked task for observation/joining; ordinary shutdown uses aclose()."""
        return self._task

    @property
    def report(self) -> SessionResourceReport:
        return SessionResourceReport(
            state=self._state,
            owner_done=self._task is None or self._task.done(),
            cleanup_complete=self._cleanup_complete,
            setup_error_type=self._setup_error_type,
            owner_error_type=self._owner_error_type,
            cleanup_error_type=self._cleanup_error_type,
            timeout_reached=self._timeout_reached,
        )

    def _check_loop(self) -> None:
        if self._loop is not None and asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("session resources belong to another event loop")

    async def open(self, setup: Callable[[AsyncExitStack], Awaitable[T]]) -> T:
        self._check_loop()
        if self._state != "new":
            raise RuntimeError("session resources can only be opened once")
        self._loop = asyncio.get_running_loop()
        self._ready = self._loop.create_future()
        self._finished = self._loop.create_future()
        self._stop = asyncio.Event()
        self._state = "opening"
        self._task = asyncio.create_task(self._run(setup), name=f"harness-session-resources-{id(self):x}")
        _ACTIVE_OWNERS.add(self._task)
        self._task.add_done_callback(self._owner_finished)
        try:
            # Only this readiness wait is timed out. Enter/exit never moves into
            # wait_for's helper task; the owner always executes both itself.
            async with asyncio.timeout(self.startup_timeout_s):
                result = await asyncio.shield(self._ready)
        except asyncio.CancelledError:
            self._request_close()
            # Preserve caller cancellation; report retains cleanup state.
            with suppress(ResourceCleanupError):
                await self.aclose()
            raise
        except TimeoutError:
            self._timeout_reached = True
            self._request_close()
            with suppress(ResourceCleanupError):
                await self.aclose()
            raise ResourceSetupError("TimeoutError", self.report) from None
        if result.error_type is not None or self._state != "open":
            with suppress(ResourceCleanupError):
                await self.aclose()
            raise ResourceSetupError(result.error_type or "ResourceClosed", self.report) from None
        return cast(T, result.value)

    def _request_close(self) -> None:
        if self._close_requested:
            return
        self._close_requested = True
        if self._task is None:
            self._cleanup_complete = True
            self._state = "closed"
            return
        if self._stop is not None:
            self._stop.set()
        if not self._task.done():
            self._state = "closing"
            if not self._setup_complete:
                self._task.cancel()

    async def aclose(self) -> None:
        self._check_loop()
        if self._task is not None and asyncio.current_task() is self._task:
            raise RuntimeError("a resource owner cannot wait for its own cleanup")
        self._request_close()
        if self._finished is not None:
            try:
                async with asyncio.timeout(self.cleanup_timeout_s):
                    await asyncio.shield(self._finished)
            except TimeoutError:
                self._timeout_reached = True
                if not self._finished.done():
                    self._state = "incomplete"
                    raise ResourceCleanupError(self.report) from None
        if not self._cleanup_complete or self._cleanup_error_type or self._owner_error_type:
            raise ResourceCleanupError(self.report) from None

    def _set_ready(self, result: _OpenResult) -> None:
        if self._ready is not None and not self._ready.done():
            self._ready.set_result(result)

    async def _run(self, setup: Callable[[AsyncExitStack], Awaitable[object]]) -> None:
        self._started = True
        stack = AsyncExitStack()
        cancelled: asyncio.CancelledError | None = None
        try:
            value = await setup(stack)
            self._setup_complete = True
            if self._close_requested:
                self._set_ready(_OpenResult(error_type="ResourceClosed"))
            else:
                self._state = "open"
                self._set_ready(_OpenResult(value=value))
                assert self._stop is not None
                await self._stop.wait()
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError):
                cancelled = error
            if not self._setup_complete:
                self._setup_error_type = type(error).__name__
                self._set_ready(_OpenResult(error_type=self._setup_error_type))
            elif not isinstance(error, asyncio.CancelledError) or not self._close_requested:
                self._owner_error_type = type(error).__name__
        finally:
            self._state = "incomplete" if self._timeout_reached else "closing"
            timer = asyncio.timeout(self.cleanup_timeout_s)
            try:
                # No task is spawned for exit: AnyIO scopes stay on their host.
                async with timer:
                    if cancelled is None:
                        await stack.aclose()
                    else:
                        # SDK taskgroups must see the original cancellation;
                        # aclose(None, None, None) can misclassify it as a new
                        # cleanup failure while their scopes are still active.
                        await stack.__aexit__(type(cancelled), cancelled, cancelled.__traceback__)
            except BaseException as error:
                if error is cancelled and not timer.expired():
                    # AnyIO taskgroups re-raise the body's original external
                    # cancellation after finishing their exits. It is not a new
                    # cleanup failure; a different error still fails explicitly.
                    self._cleanup_complete = True
                    self._state = "closed"
                else:
                    self._cleanup_error_type = type(error).__name__
                    self._state = "incomplete"
            else:
                self._cleanup_complete = True
                self._state = "closed"
            finally:
                self._timeout_reached |= timer.expired()
                self._set_ready(_OpenResult(error_type=self._setup_error_type or "ResourceClosed"))
                assert self._finished is not None
                self._finished.set_result(None)

    def _owner_finished(self, task: asyncio.Task[None]) -> None:
        _ACTIVE_OWNERS.discard(task)
        error_type = "CancelledError" if task.cancelled() else None
        if not task.cancelled():
            error = task.exception()  # Retrieve even unexpected internal failures.
            if error is not None:
                error_type = type(error).__name__
        if self._finished is None or self._finished.done():
            return
        # Cancellation before the task's first instruction never reaches finally.
        self._setup_error_type = error_type or "UnexpectedOwnerExit"
        if not self._started:
            self._cleanup_complete = True
            self._state = "closed"
        else:
            self._owner_error_type = error_type or "UnexpectedOwnerExit"
            self._state = "incomplete"
        self._set_ready(_OpenResult(error_type=self._setup_error_type))
        self._finished.set_result(None)
