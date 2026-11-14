"""Session teardown owns the run boundary and keeps resources live through cleanup."""

from __future__ import annotations

import asyncio
import gc
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, suppress
from typing import cast

import pytest
from pydantic import BaseModel

from intern.browser import BrowserSession, browser_tool
from intern.context import TokenMeter, build_view
from intern.decisions import Answers
from intern.events import Decision, RunStart
from intern.hooks import Extension, RunContext
from intern.journal import Entry, EventEntry, Journal, recording
from intern.mcp_client import McpConnection, _Description
from intern.messages import AssistantMessage, JsonObject
from intern.providers.base import Completed, ProviderRequest, StreamItem
from intern.providers.fake import FakeProvider, Turn
from intern.session import Session, SessionBusy
from intern.session_resources import ResourceCleanupError, SessionResources
from intern.tools.base import Tool, ToolContext, ToolOutput

from .support import binding


class _EmptyArgs(BaseModel):
    pass


class _WaitingTool(Tool):
    name = "waiting"
    description = "Wait for a controlled test boundary."
    Args = _EmptyArgs
    effects = frozenset({"exec"})

    def __init__(self, call: Callable[[ToolContext], Awaitable[ToolOutput]]) -> None:
        self.call = call

    async def run(self, args: _EmptyArgs, ctx: ToolContext) -> ToolOutput:
        return await self.call(ctx)


async def _resources(cleanup: Callable[[], Awaitable[None]], *, cleanup_timeout_s: float = 1) -> SessionResources:
    owner = SessionResources(cleanup_timeout_s=cleanup_timeout_s)

    async def setup(stack: AsyncExitStack) -> None:
        stack.push_async_callback(cleanup)

    await owner.open(setup)
    return owner


async def _tick() -> None:
    # Give shared teardown and a cancelled worker their next scheduling turn.
    for _ in range(4):
        await asyncio.sleep(0)


async def test_context_and_concurrent_close_release_resources_once(make_harness):
    harness = make_harness([Turn(text="done")])
    closed: list[str] = []

    async def cleanup() -> None:
        closed.append("resource")

    owner = await _resources(cleanup)
    session = Session(harness.deps, resources=owner)
    async with session as entered:
        assert entered is session
        assert (await session.run("work")).final_text == "done"
        assert not session.closed
    await asyncio.gather(*(session.aclose() for _ in range(5)))
    assert closed == ["resource"]
    assert owner.report.cleanup_complete and session.closed


async def test_close_joins_direct_run_boundary_not_the_host_task(make_harness):
    harness = make_harness([Turn(tool_calls=[("waiting", {})])])
    entered = asyncio.Event()
    host_after_run = asyncio.Event()
    release_host = asyncio.Event()
    order: list[str] = []

    async def tool(ctx: ToolContext) -> ToolOutput:
        entered.set()
        await ctx.abort.wait()
        order.append("tool-finished")
        return ToolOutput(content="stopped")

    async def cleanup() -> None:
        order.append("resource-closed")

    harness.deps.tools = [_WaitingTool(tool)]
    session = Session(harness.deps, resources=await _resources(cleanup))

    async def host() -> None:
        outcome = await session.run("work")
        assert outcome.reason == "aborted"
        host_after_run.set()
        await release_host.wait()

    task = asyncio.create_task(host())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(session.aclose(), 1)
        assert host_after_run.is_set()
        assert not task.done()
        assert order == ["tool-finished", "resource-closed"]
    finally:
        release_host.set()
        await asyncio.wait_for(task, 1)
        await session.aclose()


async def test_start_then_close_before_scheduled_run_cancels_queued_work(make_harness):
    harness = make_harness([Turn(text="must not run")])
    closed = asyncio.Event()

    async def cleanup() -> None:
        closed.set()

    session = Session(harness.deps, resources=await _resources(cleanup))
    task = session.start("work")
    await session.aclose()
    assert task.done()
    if not task.cancelled():
        with pytest.raises(SessionBusy):
            task.result()
    assert not harness.provider.requests
    assert closed.is_set() and session.closed


@pytest.mark.parametrize("cooperative", [True, False], ids=["abort-grace", "forced-cancel"])
async def test_tool_finally_finishes_before_resource_close(make_harness, cooperative: bool):
    harness = make_harness([Turn(tool_calls=[("waiting", {})])])
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def tool(ctx: ToolContext) -> ToolOutput:
        entered.set()
        try:
            await (ctx.abort.wait() if cooperative else asyncio.Event().wait())
            return ToolOutput(content="stopped")
        finally:
            cleaning.set()
            await release.wait()
            order.append("tool-cleaned")

    async def cleanup() -> None:
        order.append("resource-closed")

    harness.deps.tools = [_WaitingTool(tool)]
    session = Session(harness.deps, resources=await _resources(cleanup), shutdown_timeout_s=0.03)
    run = session.start("work")
    close: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        close = asyncio.create_task(session.aclose())
        await asyncio.wait_for(cleaning.wait(), 1)
        await _tick()
        assert not close.done()
        assert not order
        release.set()
        await asyncio.wait_for(close, 1)
        if cooperative:
            assert (await run).reason == "aborted"
        else:
            assert run.cancelled()
        assert order == ["tool-cleaned", "resource-closed"]
    finally:
        release.set()
        if close is not None:
            await asyncio.wait_for(close, 1)
        await session.aclose()


class _CleanupProvider(FakeProvider):
    def __init__(self) -> None:
        super().__init__([])
        self.entered = asyncio.Event()
        self.cleaning = asyncio.Event()
        self.release = asyncio.Event()
        self.cleaned = asyncio.Event()
        self.worker: asyncio.Task | None = None

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        self.worker = asyncio.current_task()
        self.entered.set()
        try:
            await asyncio.Event().wait()
            yield Completed(message=AssistantMessage(text="unreachable"))  # pragma: no cover
        finally:
            self.cleaning.set()
            await self.release.wait()
            self.cleaned.set()


async def test_provider_finally_finishes_before_resource_close(make_harness):
    harness = make_harness([])
    provider = _CleanupProvider()
    harness.deps.system2 = binding(provider)
    closed = asyncio.Event()

    async def cleanup() -> None:
        closed.set()

    session = Session(harness.deps, resources=await _resources(cleanup), shutdown_timeout_s=0.3)
    session.start("work")
    close: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(provider.entered.wait(), 1)
        close = asyncio.create_task(session.aclose())
        await asyncio.wait_for(provider.cleaning.wait(), 1)
        await _tick()
        assert not closed.is_set(), "resources closed while provider stream cleanup was still running"
        assert not close.done()
        provider.release.set()
        await asyncio.wait_for(close, 1)
        assert provider.cleaned.is_set() and closed.is_set()
    finally:
        provider.release.set()
        if provider.worker is not None:
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(provider.worker, 1)
        if close is not None:
            await asyncio.wait_for(close, 1)
        await session.aclose()


async def test_cancelled_run_caller_joins_provider_before_resources_close(make_harness):
    harness = make_harness([])
    provider = _CleanupProvider()
    harness.deps.system2 = binding(provider)
    closed = asyncio.Event()

    async def cleanup() -> None:
        closed.set()

    session = Session(harness.deps, resources=await _resources(cleanup))
    before = asyncio.all_tasks()
    run = session.start("work")
    close: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(provider.entered.wait(), 1)
        run.cancel()
        await asyncio.wait_for(provider.cleaning.wait(), 1)
        close = asyncio.create_task(session.aclose())
        await _tick()
        assert not run.done() and not closed.is_set()
        provider.release.set()
        with pytest.raises(asyncio.CancelledError):
            await run
        await asyncio.wait_for(close, 1)
        assert provider.cleaned.is_set() and closed.is_set()
        assert asyncio.all_tasks() <= before
    finally:
        provider.release.set()
        if provider.worker is not None:
            provider.worker.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(provider.worker, 1)
        with suppress(asyncio.CancelledError):
            await run
        if close is not None:
            await asyncio.wait_for(close, 1)
        await session.aclose()
        # The broken outer-cancellation path can strand its abort-wait child.
        # Keep a failing regression from leaking that child into loop shutdown.
        remaining = asyncio.all_tasks() - before
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


async def test_shutdown_deadline_and_repeated_cancel_do_not_interrupt_provider_finally(make_harness):
    harness = make_harness([])
    provider = _CleanupProvider()
    harness.deps.system2 = binding(provider)
    closed = asyncio.Event()

    async def cleanup() -> None:
        closed.set()

    session = Session(harness.deps, resources=await _resources(cleanup), shutdown_timeout_s=0.02)
    run = session.start("work")
    close: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(provider.entered.wait(), 1)
        close = asyncio.create_task(session.aclose())
        await asyncio.wait_for(provider.cleaning.wait(), 1)

        await asyncio.sleep(0.04)
        assert run.cancelling(), "session grace deadline did not request forced cancellation"
        for _ in range(2):
            run.cancel()
            await _tick()
        assert provider.worker is not None and not provider.worker.done()
        assert not closed.is_set() and not close.done()
        provider.release.set()
        await asyncio.wait_for(close, 1)
        assert provider.cleaned.is_set() and closed.is_set()
        assert run.cancelled()
    finally:
        provider.release.set()
        if close is not None:
            await asyncio.wait_for(close, 1)
        await session.aclose()


async def test_abort_during_request_preparation_dispatches_no_model_request(make_harness):
    preparing = asyncio.Event()
    release = asyncio.Event()

    class Preparing(Extension):
        async def before_provider_request(self, request: ProviderRequest, ctx: RunContext) -> ProviderRequest:
            preparing.set()
            await release.wait()
            return request

    harness = make_harness([Turn(text="must not dispatch")], extensions=[Preparing()])
    session = Session(harness.deps)
    run = session.start("work")
    try:
        await asyncio.wait_for(preparing.wait(), 1)
        session.abort()
        release.set()
        assert (await asyncio.wait_for(run, 1)).reason == "aborted"
        assert harness.provider.requests == []
    finally:
        release.set()
        await session.aclose()


async def test_abort_queued_before_consumer_first_instruction_prevents_dispatch(make_harness):
    class ScheduledAbort(Extension):
        async def before_provider_request(self, request: ProviderRequest, ctx: RunContext) -> ProviderRequest:
            # This callback runs before the newly created provider-consumer task,
            # even though the caller's synchronous pre-dispatch guard saw open.
            asyncio.get_running_loop().call_soon(session.abort)
            return request

    harness = make_harness([Turn(text="must not dispatch")], extensions=[ScheduledAbort()])
    session = Session(harness.deps)
    try:
        assert (await asyncio.wait_for(session.run("work"), 1)).reason == "aborted"
        assert harness.provider.requests == []
    finally:
        await session.aclose()


@pytest.mark.parametrize("adapter", ["mcp", "browser"])
async def test_session_grace_expiry_preserves_adapter_child_cleanup(make_harness, adapter: str):
    entered, cleaning, release, cleaned, closed = (asyncio.Event() for _ in range(5))

    class Service:
        worker: asyncio.Task | None = None

        async def operation(self) -> None:
            self.worker = asyncio.current_task()
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()
                cleaned.set()

        async def call_tool(self, *args, **kwargs):
            await self.operation()

        async def perform(self, *args, **kwargs):
            await self.operation()

        async def aclose(self) -> None:
            closed.set()

    class Validator:
        def validate(self, value) -> None:
            pass

    service = Service()
    arguments: JsonObject
    if adapter == "mcp":
        tool = McpConnection(service, "test")._tool(
            _Description(name="wait", inputSchema={"type": "object"}),
            "mcp__test__wait",
            frozenset({"network"}),
            Validator(),
        )
        arguments = {}
    else:
        tool = browser_tool(cast(BrowserSession, service))
        arguments = {"action": "read"}
    harness = make_harness([Turn(tool_calls=[(tool.name, arguments)])])
    harness.deps.tools = [tool]
    session = Session(harness.deps, resources=await _resources(service.aclose), shutdown_timeout_s=0.02)
    run = session.start("work")
    close: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        close = asyncio.create_task(session.aclose())
        await asyncio.wait_for(cleaning.wait(), 1)
        await asyncio.sleep(0.04)
        assert run.cancelling(), "session grace deadline did not request forced cancellation"
        assert service.worker is not None and not service.worker.done()
        assert not closed.is_set() and not close.done()
        release.set()
        await asyncio.wait_for(close, 1)
        assert cleaned.is_set() and closed.is_set()
    finally:
        release.set()
        if service.worker is not None:
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(service.worker, 1)
        if close is not None:
            await asyncio.wait_for(close, 1)
        await session.aclose()


async def test_a_cancelled_run_returns_only_after_its_system1_ask_has_cleaned_up(make_harness):
    entered, cleaning, release, cleaned, closed = (asyncio.Event() for _ in range(5))

    class Backend:
        name = "controlled-native-review"
        worker: asyncio.Task | None = None

        async def decide(self, state, questions) -> Answers:
            self.worker = asyncio.current_task()
            entered.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                cleaning.set()
                await release.wait()
                cleaned.set()

    async def tool(ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content="one step done")

    async def cleanup() -> None:
        closed.set()

    backend = Backend()
    harness = make_harness([], system1=backend)
    harness.deps.tools = [_WaitingTool(tool)]
    harness.deps.system2 = binding(FakeProvider([Turn(tool_calls=[("waiting", {})]) for _ in range(4)]))
    session = Session(harness.deps, resources=await _resources(cleanup))
    before = asyncio.all_tasks()
    run = session.start("work")
    try:
        await asyncio.wait_for(entered.wait(), 1)  # the loop waits for the ask at the end of turn 3
        run.cancel()
        await asyncio.wait_for(cleaning.wait(), 1)
        await _tick()
        assert not run.done(), "run returned while its native System 1 review was still cleaning up"
        assert not closed.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await run
        await session.aclose()
        assert cleaned.is_set() and closed.is_set()
        assert asyncio.all_tasks() <= before
    finally:
        release.set()
        if backend.worker is not None and not backend.worker.done():
            backend.worker.cancel()
            with suppress(asyncio.CancelledError):
                await backend.worker
        with suppress(asyncio.CancelledError):
            await run
        await session.aclose()
        remaining = asyncio.all_tasks() - before
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


async def test_closing_session_during_a_system1_ask_lets_the_backend_clean_up_once(make_harness):
    entered, cleaning, release, cleaned = (asyncio.Event() for _ in range(4))

    class Backend:
        name = "hanging-review"
        cleanups = 0

        async def decide(self, state, questions) -> Answers:
            entered.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("unreachable")
            finally:
                self.cleanups += 1
                cleaning.set()
                await release.wait()
                cleaned.set()

    async def tool(ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content="one completed step")

    backend = Backend()
    harness = make_harness([], system1=backend)
    harness.deps.tools = [_WaitingTool(tool)]
    harness.deps.system2 = binding(FakeProvider([Turn(tool_calls=[("waiting", {})]) for _ in range(4)]))
    session = Session(harness.deps)
    before = asyncio.all_tasks()
    run = session.start("work")
    close: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 1)  # the loop is waiting for the ask at the end of turn 3
        close = asyncio.create_task(session.aclose())
        await asyncio.wait_for(cleaning.wait(), 1)
        await _tick()
        assert not close.done()
        release.set()
        await asyncio.wait_for(close, 1)
        assert (await run).reason == "aborted"
        assert cleaned.is_set() and backend.cleanups == 1
        assert asyncio.all_tasks() <= before
    finally:
        release.set()
        if close is not None:
            await asyncio.wait_for(close, 1)
        await session.aclose()
        remaining = asyncio.all_tasks() - before
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


async def test_abandoned_close_waiter_cannot_let_gc_destroy_owned_run_cleanup(make_harness):
    entered, cleaning, cleaned, closed = (asyncio.Event() for _ in range(4))
    gates: list[weakref.ReferenceType[asyncio.Event]] = []

    class Provider(FakeProvider):
        async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                gate = asyncio.Event()
                gates.append(weakref.ref(gate))
                cleaning.set()
                await gate.wait()
                cleaned.set()
            yield Completed(message=AssistantMessage(text="unreachable"))  # pragma: no cover

    async def cleanup() -> None:
        closed.set()

    owner = await _resources(cleanup)

    async def abandon_waiter() -> tuple[weakref.ReferenceType[Session], weakref.ReferenceType[asyncio.Task]]:
        harness = make_harness([])
        harness.deps.system2 = binding(Provider([]))
        session = Session(harness.deps, resources=owner, shutdown_timeout_s=0.01)
        session.start("work")
        await asyncio.wait_for(entered.wait(), 1)
        waiter = asyncio.create_task(session.aclose())
        await asyncio.wait_for(cleaning.wait(), 1)
        await asyncio.sleep(0.02)
        weak_session = weakref.ref(session)
        assert session._close_task is not None
        weak_close = weakref.ref(session._close_task)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        return weak_session, weak_close

    before = asyncio.all_tasks()
    weak_session, weak_close = await abandon_waiter()
    try:
        for _ in range(3):
            await asyncio.sleep(0)
            gc.collect()
        session = weak_session()
        close = weak_close()
        assert session is not None and close is not None, "active Session teardown was garbage collected"
        assert not close.done() and not closed.is_set()
        gate = gates[0]()
        assert gate is not None
        gate.set()
        await asyncio.wait_for(session.aclose(), 1)
        assert cleaned.is_set() and closed.is_set()
        assert session.closed and asyncio.all_tasks() <= before
    finally:
        if gates and (gate := gates[0]()) is not None:
            gate.set()
        if (session := weak_session()) is not None:
            await asyncio.wait_for(session.aclose(), 1)
        await owner.aclose()


async def test_cancelled_close_waiter_can_join_shared_cleanup_later(make_harness):
    harness = make_harness([])
    cleaning = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def cleanup() -> None:
        calls.append("close")
        cleaning.set()
        await release.wait()

    owner = await _resources(cleanup)
    session = Session(harness.deps, resources=owner)
    first = asyncio.create_task(session.aclose())
    others: list[asyncio.Task[None]] = []
    try:
        await asyncio.wait_for(cleaning.wait(), 1)
        others = [asyncio.create_task(session.aclose()) for _ in range(4)]
        await _tick()
        assert all(not task.done() for task in others)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not session.closed and not owner.report.owner_done
        release.set()
        await asyncio.wait_for(asyncio.gather(*others), 1)
        await asyncio.wait_for(session.aclose(), 1)
        assert session.closed and calls == ["close"]
    finally:
        release.set()
        await asyncio.gather(*others)
        await session.aclose()


async def test_closed_session_rejects_new_work_and_mutations(make_harness):
    harness = make_harness([Turn(text="done")])
    resolver_calls: list[str] = []

    async def resolve(ref: str):
        resolver_calls.append(ref)
        return binding(FakeProvider([]))

    session = Session(harness.deps, resolve_model=resolve)
    await session.run("first")
    leaf = harness.deps.journal.leaf_id
    assert leaf is not None
    await session.aclose()
    with pytest.raises(SessionBusy, match="closing or closed"):
        await session.run("second")
    with pytest.raises(SessionBusy):
        session.start("second")
    with pytest.raises(SessionBusy):
        session.follow_up("second")
    with pytest.raises(SessionBusy):
        session.steer("second")
    with pytest.raises(SessionBusy):
        await session.set_model("fake/other")
    with pytest.raises(SessionBusy):
        session.branch(leaf)
    with pytest.raises(SessionBusy):
        await session.__aenter__()
    assert not resolver_calls
    assert len(harness.provider.requests) == 1
    assert session.state().running is False


async def test_wait_reports_an_already_completed_failed_start(make_harness):
    harness = make_harness([])

    class ClientDisconnected(RuntimeError):
        pass

    def emit(event) -> None:
        if isinstance(event, RunStart):
            raise ClientDisconnected("transport is gone")

    harness.deps.emit = emit
    session = Session(harness.deps)
    task = session.start("work")
    await asyncio.wait({task}, timeout=1)
    assert task.done()
    with pytest.raises(ClientDisconnected, match="transport is gone"):
        await session.wait()
    await session.aclose()
    assert session.closed


async def test_cleanup_failure_is_visible_and_retry_does_not_repeat_callbacks(make_harness):
    harness = make_harness([])
    calls: list[str] = []

    async def cleanup() -> None:
        calls.append("close")
        raise ValueError("credential must not escape")

    owner = await _resources(cleanup)
    session = Session(harness.deps, resources=owner)
    for _ in range(2):
        with pytest.raises(ResourceCleanupError) as error:
            await session.aclose()
        assert "credential" not in str(error.value)
        assert not session.closed
    assert calls == ["close"]
    assert owner.report.owner_done and not owner.report.cleanup_complete


async def test_late_resource_cleanup_can_be_joined_after_incomplete_close(make_harness):
    harness = make_harness([])
    release = asyncio.Event()
    timed_out = asyncio.Event()

    async def cleanup() -> None:
        try:
            await release.wait()
        except asyncio.CancelledError:
            timed_out.set()
            await release.wait()

    owner = await _resources(cleanup, cleanup_timeout_s=0.03)
    session = Session(harness.deps, resources=owner)
    try:
        with pytest.raises(ResourceCleanupError):
            await asyncio.wait_for(session.aclose(), 1)
        await asyncio.wait_for(timed_out.wait(), 1)
        assert not session.closed and not owner.report.owner_done
        release.set()
        await asyncio.wait_for(session.aclose(), 1)
        assert session.closed and owner.report.cleanup_complete
    finally:
        release.set()
        await session.aclose()


async def test_closing_one_session_keeps_the_other_resource_owner_live(make_harness):
    closed: list[str] = []

    async def first_cleanup() -> None:
        closed.append("first")

    async def second_cleanup() -> None:
        closed.append("second")

    first_owner = await _resources(first_cleanup)
    second_owner = await _resources(second_cleanup)
    first = Session(make_harness([]).deps, resources=first_owner)
    second = Session(make_harness([Turn(text="still usable")]).deps, resources=second_owner)
    try:
        await first.aclose()
        assert closed == ["first"]
        assert second_owner.report.state == "open" and not second.closed
        assert (await second.run("work")).final_text == "still usable"
    finally:
        await first.aclose()
        await second.aclose()
    assert closed == ["first", "second"]


async def test_a_session_journals_system_1s_side_where_it_happened_and_not_into_the_context(make_harness, tmp_path):
    def script(request: ProviderRequest) -> Turn:
        if sum(isinstance(m, AssistantMessage) for m in request.messages) < 3:
            return Turn(tool_calls=[("read", {"path": "notes.txt"})])
        return Turn(text="Done.")

    harness = make_harness([])
    (harness.cwd / "notes.txt").write_text("a note\n")
    harness.deps.system2 = binding(FakeProvider(script))
    harness.deps.journal = Journal(cwd=str(harness.cwd), path=tmp_path / "session.jsonl")
    session = Session(harness.deps)
    assert (await session.run("look at the notes")).reason == "completed"
    await session.aclose()

    path = Journal(cwd=str(harness.cwd), path=tmp_path / "session.jsonl").active_path()  # as a resume reads it
    recorded = [e.event for e in path if isinstance(e, EventEntry)]
    supervised = [e for e in recorded if isinstance(e, Decision) and e.purpose == "supervise"]
    assert [d.turn for d in supervised] == [1, 2, 3] and supervised[0].answers is not None  # what System 1 answered
    # Where it happened: each ask right after its turn's tool result.
    kinds = [type(e).__name__ for e in path]
    asks = [i for i, e in enumerate(path) if isinstance(e, EventEntry) and e.event in supervised]
    assert all(kinds[i - 1] == "MessageEntry" for i in asks[:2])
    # Never read into the context: the view is the one the path gives without them.
    plain: list[Entry] = [e for e in path if not isinstance(e, EventEntry)]
    meter = TokenMeter()
    assert build_view(path, harness.deps.instructions, meter, 32_768) == build_view(
        plain, harness.deps.instructions, meter, 32_768
    )


def test_a_decision_asked_for_no_turn_is_not_journaled():
    journal, events = Journal(cwd="/work"), []
    sink = recording(journal, events.append)
    sink(Decision(purpose="locate", backend="b", questions=1, ok=True, cached=False, cost_usd=0.0, duration_ms=3))
    assert len(events) == 1 and not any(isinstance(e, EventEntry) for e in journal.entries())
