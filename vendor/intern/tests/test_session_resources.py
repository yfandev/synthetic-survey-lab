from __future__ import annotations

import asyncio
import os
import sys
import traceback
from contextlib import AsyncExitStack, asynccontextmanager

import anyio
import pytest

from intern.session_resources import ResourceCleanupError, ResourceSetupError, SessionResources


async def test_initialization_and_unused_close_start_no_task():
    before = asyncio.all_tasks()
    owner = SessionResources()
    assert owner.owner_task is None and owner.report.state == "new"
    assert asyncio.all_tasks() == before
    await owner.aclose()
    await owner.aclose()
    assert owner.report.state == "closed" and owner.report.cleanup_complete
    assert owner.owner_task is None and asyncio.all_tasks() == before


async def test_anyio_contexts_enter_and_exit_on_same_owner_across_callers_in_lifo_order():
    events = []

    @asynccontextmanager
    async def resource(name):
        with anyio.CancelScope():
            events.append(("enter", name, asyncio.current_task()))
            try:
                yield name
            finally:
                events.append(("exit", name, asyncio.current_task()))

    async def setup(stack: AsyncExitStack):
        first = await stack.enter_async_context(resource("first"))
        second = await stack.enter_async_context(resource("second"))
        return first, second

    owner = SessionResources()
    opening = asyncio.create_task(owner.open(setup))
    assert await opening == ("first", "second")
    closing = asyncio.create_task(owner.aclose())
    await closing
    assert [(kind, name) for kind, name, _ in events] == [
        ("enter", "first"),
        ("enter", "second"),
        ("exit", "second"),
        ("exit", "first"),
    ]
    assert all(task is owner.owner_task and task not in {opening, closing} for _, _, task in events)
    assert owner.report.cleanup_complete and owner.report.owner_done
    copy = owner.report
    copy.state = "new"
    assert owner.report.state == "closed"


async def test_partial_startup_failure_rolls_back_registered_resources_before_return():
    closed = []

    @asynccontextmanager
    async def resource():
        with anyio.CancelScope():
            try:
                yield
            finally:
                closed.append(asyncio.current_task())

    async def setup(stack):
        await stack.enter_async_context(resource())
        raise ValueError("private-startup-detail")

    owner = SessionResources()
    with pytest.raises(ResourceSetupError) as captured:
        await owner.open(setup)
    assert captured.value.error_type == "ValueError"
    assert "private-startup" not in "".join(traceback.format_exception(captured.value))
    assert closed == [owner.owner_task]
    assert owner.report.cleanup_complete and owner.report.owner_done
    await owner.aclose()


async def test_cancelling_startup_waiter_cleans_same_task_anyio_context():
    started = asyncio.Event()
    closed = []

    @asynccontextmanager
    async def resource():
        with anyio.CancelScope():
            try:
                yield
            finally:
                closed.append(asyncio.current_task())

    async def setup(stack):
        await stack.enter_async_context(resource())
        started.set()
        await asyncio.Event().wait()

    owner = SessionResources()
    opening = asyncio.create_task(owner.open(setup))
    await started.wait()
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert closed == [owner.owner_task]
    assert owner.report.cleanup_complete and owner.report.owner_done


async def test_cancelled_close_waiter_does_not_cancel_or_abandon_cleanup():
    closing, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def finish():
        closing.set()
        await release.wait()
        closed.set()

    async def setup(stack):
        stack.push_async_callback(finish)

    owner = SessionResources()
    await owner.open(setup)
    waiter = asyncio.create_task(owner.aclose())
    await closing.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert owner.owner_task is not None and not owner.owner_task.done()
    release.set()
    await owner.aclose()
    assert closed.is_set() and owner.report.cleanup_complete and owner.report.owner_done


async def test_concurrent_and_repeated_close_run_callbacks_once():
    started, release = asyncio.Event(), asyncio.Event()
    closed = []

    async def finish():
        started.set()
        await release.wait()
        closed.append(True)

    async def setup(stack):
        stack.push_async_callback(finish)

    owner = SessionResources()
    await owner.open(setup)
    waiters = [asyncio.create_task(owner.aclose()) for _ in range(5)]
    await started.wait()
    release.set()
    assert await asyncio.gather(*waiters) == [None] * 5
    await owner.aclose()
    assert closed == [True]


async def test_cleanup_failure_is_sanitized_and_remaining_callbacks_still_run():
    closed = []

    async def finish():
        closed.append(True)

    async def broken():
        raise RuntimeError("private-cleanup-detail")

    async def setup(stack):
        stack.push_async_callback(finish)
        stack.push_async_callback(broken)

    owner = SessionResources()
    await owner.open(setup)
    for _ in range(2):
        with pytest.raises(ResourceCleanupError) as captured:
            await owner.aclose()
        assert "private-cleanup" not in "".join(traceback.format_exception(captured.value))
        assert captured.value.report.cleanup_error_type == "RuntimeError"
        assert captured.value.report.state == "incomplete" and captured.value.report.owner_done
    assert closed == [True] and not owner.report.cleanup_complete


async def test_startup_timeout_closes_registered_resource():
    closed = []

    async def finish():
        closed.append(True)

    async def setup(stack):
        stack.push_async_callback(finish)
        await asyncio.Event().wait()

    owner = SessionResources(startup_timeout_s=0.01)
    with pytest.raises(ResourceSetupError) as captured:
        await owner.open(setup)
    assert captured.value.error_type == "TimeoutError"
    assert closed == [True] and owner.report.cleanup_complete and owner.report.timeout_reached


async def test_cleanup_timeout_reports_incomplete_and_attempts_earlier_callbacks():
    closed = []

    async def finish():
        closed.append(True)

    async def waiting():
        await asyncio.Event().wait()

    async def setup(stack):
        stack.push_async_callback(finish)
        stack.push_async_callback(waiting)

    owner = SessionResources(cleanup_timeout_s=0.01)
    await owner.open(setup)
    with pytest.raises(ResourceCleanupError):
        await owner.aclose()
    assert owner.owner_task is not None
    async with asyncio.timeout(1):
        await asyncio.shield(owner.owner_task)
    assert owner.report.state == "incomplete" and owner.report.owner_done
    assert owner.report.cleanup_error_type == "TimeoutError" and closed == [True]


async def test_cancel_suppressing_closer_is_tracked_incomplete_until_late_completion():
    release = asyncio.Event()

    async def resistant():
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    async def setup(stack):
        stack.push_async_callback(resistant)

    owner = SessionResources(cleanup_timeout_s=0.01)
    await owner.open(setup)
    try:
        with pytest.raises(ResourceCleanupError) as captured:
            await owner.aclose()
        assert captured.value.report.state == "incomplete" and not captured.value.report.cleanup_complete
        assert owner.owner_task is not None and not owner.owner_task.done()
    finally:
        release.set()
        assert owner.owner_task is not None
        async with asyncio.timeout(1):
            await asyncio.shield(owner.owner_task)
    await owner.aclose()
    assert owner.report.state == "closed" and owner.report.cleanup_complete


async def test_separate_owners_do_not_share_values_tasks_or_cleanup():
    closed = set()

    def setup_for(name):
        async def setup(stack):
            async def finish():
                closed.add(name)

            stack.push_async_callback(finish)
            return object()

        return setup

    first, second = SessionResources(), SessionResources()
    a = await first.open(setup_for("first"))
    b = await second.open(setup_for("second"))
    assert a is not b and first.owner_task is not second.owner_task
    await first.aclose()
    assert closed == {"first"} and second.report.state == "open"
    await second.aclose()
    assert closed == {"first", "second"}


async def test_immediate_shutdown_before_owner_starts_does_not_leave_pending_futures():
    entered = []

    async def setup(stack):
        entered.append(True)

    owner = SessionResources()
    opening = asyncio.create_task(owner.open(setup))
    await asyncio.sleep(0)
    await owner.aclose()
    with pytest.raises(ResourceSetupError):
        await opening
    assert entered == [] and owner.report.cleanup_complete and owner.report.owner_done


async def test_real_mcp_sdk_context_can_open_and_close_from_different_callers():
    sdk = pytest.importorskip("mcp")

    async def setup(stack):
        incoming_sender, incoming = anyio.create_memory_object_stream(0)
        outgoing, outgoing_receiver = anyio.create_memory_object_stream(0)
        for stream in (incoming_sender, incoming, outgoing, outgoing_receiver):
            await stack.enter_async_context(stream)
        return await stack.enter_async_context(sdk.ClientSession(incoming, outgoing))

    owner = SessionResources()
    assert await asyncio.create_task(owner.open(setup)) is not None
    await asyncio.create_task(owner.aclose())
    assert owner.report.state == "closed" and owner.report.cleanup_complete


async def test_real_mcp_sdk_context_rolls_back_when_setup_waiter_is_cancelled():
    sdk = pytest.importorskip("mcp")
    started = asyncio.Event()

    async def setup(stack):
        sender, incoming = anyio.create_memory_object_stream(0)
        outgoing, receiver = anyio.create_memory_object_stream(0)
        for stream in (sender, incoming, outgoing, receiver):
            await stack.enter_async_context(stream)
        await stack.enter_async_context(sdk.ClientSession(incoming, outgoing))
        started.set()
        await asyncio.Event().wait()

    owner = SessionResources()
    opening = asyncio.create_task(owner.open(setup))
    await started.wait()
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert owner.report.cleanup_complete and owner.report.owner_done


async def test_setup_failure_keeps_cleanup_failure_visible_without_revealing_either_error():
    async def broken():
        raise RuntimeError("private-cleanup-detail")

    async def setup(stack):
        stack.push_async_callback(broken)
        raise ValueError("private-setup-detail")

    owner = SessionResources()
    with pytest.raises(ResourceSetupError) as captured:
        await owner.open(setup)
    assert captured.value.error_type == "ValueError"
    assert captured.value.report.cleanup_error_type == "RuntimeError"
    assert captured.value.report.state == "incomplete" and captured.value.report.owner_done
    assert "private-" not in "".join(traceback.format_exception(captured.value))


async def test_noncooperative_startup_is_tracked_and_never_returns_a_closed_resource():
    release, closed = asyncio.Event(), asyncio.Event()

    async def finish():
        closed.set()

    async def setup(stack):
        stack.push_async_callback(finish)
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return "must not be published"

    owner = SessionResources(startup_timeout_s=0.01, cleanup_timeout_s=0.01)
    try:
        with pytest.raises(ResourceSetupError) as captured:
            await owner.open(setup)
        assert captured.value.error_type == "TimeoutError"
        assert captured.value.report.state == "incomplete" and not captured.value.report.owner_done
    finally:
        release.set()
        assert owner.owner_task is not None
        async with asyncio.timeout(1):
            await asyncio.shield(owner.owner_task)
    await owner.aclose()
    assert closed.is_set() and owner.report.cleanup_complete


async def test_owner_close_callback_cannot_deadlock_waiting_for_itself():
    owner = SessionResources()

    async def setup(stack):
        stack.push_async_callback(owner.aclose)

    await owner.open(setup)
    async with asyncio.timeout(1):
        with pytest.raises(ResourceCleanupError) as captured:
            await owner.aclose()
    assert captured.value.report.cleanup_error_type == "RuntimeError" and owner.report.owner_done


@pytest.mark.parametrize("cancel_startup", [False, True])
async def test_real_stdio_mcp_roundtrip_and_child_cleanup_across_tasks(tmp_path, cancel_startup):
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    from intern.executor import LocalExecutor
    from intern.mcp_client import StdioConfig, stdio_connection
    from intern.tools.base import ToolContext, ToolOutput

    pidfile = tmp_path / "server.pid"
    server = tmp_path / "server.py"
    source = f"""import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
Path({str(pidfile)!r}).write_text(str(os.getpid()))
mcp=FastMCP("owned")
@mcp.tool()
def add(a:int,b:int)->int: return a+b
mcp.run(transport="stdio")
"""
    await asyncio.to_thread(server.write_text, source)
    started = asyncio.Event()
    connections = []

    async def setup(stack):
        connection = await stack.enter_async_context(
            stdio_connection(StdioConfig(command=sys.executable, args=[str(server)]), "owned", timeout_s=2)
        )
        connections.append(connection)
        started.set()
        if cancel_startup:
            await asyncio.Event().wait()
        return connection

    owner = SessionResources()
    opening = asyncio.create_task(owner.open(setup))
    await started.wait()
    try:
        if cancel_startup:
            opening.cancel()
            with pytest.raises(asyncio.CancelledError):
                await opening
        else:
            connection = await opening
            tool = (await connection.tools({"add": frozenset({"read"})}))[0]
            args = tool.parse({"a": 2, "b": 3})
            assert not isinstance(args, ToolOutput)
            context = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=1000)
            assert "5" in (await tool.run(args, context)).content
    finally:
        await asyncio.create_task(owner.aclose())
    assert owner.report.cleanup_complete and owner.report.owner_done and connections[0].closed
    pid = int(await asyncio.to_thread(pidfile.read_text))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
