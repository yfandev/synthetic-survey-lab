from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from intern.executor import LocalExecutor
from intern.processes import ProcessManager, ProcessSnapshot
from intern.tools import load_packs
from intern.tools.base import ToolContext, ToolOutput
from intern.tools.process import ProcessTool


async def test_real_stdin_output_and_cursor(tmp_path: Path):
    async with ProcessManager() as manager:
        job = await manager.start("read line; printf 'answer:%s' \"$line\"", cwd=tmp_path, stdin=True)
        await manager.write(job.handle, "hello\n")
        result = await manager.poll(job.handle, wait_s=3)
        assert result.state == "exited" and result.exit_code == 0 and result.output == "answer:hello"
        assert (await manager.poll(job.handle, cursor=result.next_cursor)).output == ""


async def test_bounded_logs_signal_exit_and_cross_manager_handles(tmp_path: Path):
    async with ProcessManager(max_log_bytes=100) as manager:
        job = await manager.start("printf '%05000d' 1", cwd=tmp_path)
        result = await manager.poll(job.handle, cursor=0, wait_s=3)
        assert len(result.output) == 100 and result.omitted_bytes == 4900 and result.next_cursor == 5000
        with pytest.raises(ValueError, match="Unknown process handle"):
            await ProcessManager().poll(job.handle)


async def test_a_mistyped_handle_s_error_names_the_session_s_jobs(tmp_path: Path):
    # A small model dropped characters from the middle of a handle; the error gives it the real one to call again with.
    async with ProcessManager() as manager:
        job = await manager.start("sleep 5", cwd=tmp_path)
        with pytest.raises(ValueError, match="Unknown process handle") as error:
            await manager.poll(job.handle[:12] + job.handle[18:])
        assert job.handle in str(error.value) and "sleep 5" in str(error.value)


HOG = f"{shlex.quote(sys.executable)} -c \"import time; x = b'1' * (1500 * 2**20); time.sleep(30)\""


async def test_a_job_past_its_memory_budget_is_stopped_and_says_so(tmp_path: Path, monkeypatch):
    from intern import processes

    monkeypatch.setattr(processes, "BACKGROUND_POLL_S", 0.1)
    async with ProcessManager() as manager:
        job = await manager.start(HOG, cwd=tmp_path, memory_limit=200 * 2**20)
        result = await manager.poll(job.handle, wait_s=20)
        assert result.state == "out_of_memory" and "of memory, over the 0.2 GB a command may hold" in result.output
        small = await manager.start("echo ok", cwd=tmp_path, memory_limit=200 * 2**20)
        assert (await manager.poll(small.handle, wait_s=5)).state == "exited"


async def test_a_job_whose_memory_cannot_be_read_runs_to_its_end(tmp_path: Path, monkeypatch):
    from intern import executor, processes

    async def unreadable() -> None:
        return None

    monkeypatch.setattr(processes, "BACKGROUND_POLL_S", 0.1)
    monkeypatch.setattr(processes, "memory_by_group", unreadable)
    assert executor.memory_by_group is not unreadable
    async with ProcessManager() as manager:
        job = await manager.start("sleep 0.6; echo done", cwd=tmp_path, memory_limit=200 * 2**20)
        result = await manager.poll(job.handle, wait_s=5)
        assert result.state == "exited" and result.output.strip() == "done"


async def test_the_process_tool_gives_jobs_the_executors_memory_budget(tmp_path: Path, monkeypatch):
    from intern import processes

    monkeypatch.setattr(processes, "BACKGROUND_POLL_S", 0.1)
    tool = ProcessTool()
    ctx = ToolContext(
        cwd=tmp_path, executor=LocalExecutor(memory_limit=200 * 2**20), abort=asyncio.Event(), max_output_chars=4000
    )
    started = await tool.run(tool.Args(action="start", command=HOG), ctx)
    handle = ProcessSnapshot.model_validate_json(started.content).handle
    await tool.manager.poll(handle, wait_s=20)  # the harness waits; the agent's poll only reads
    polled = await tool.run(tool.Args(action="poll", handle=handle), ctx)
    assert ProcessSnapshot.model_validate_json(polled.content).state == "out_of_memory"
    await tool.manager.aclose()


async def test_timeout_abort_and_cleanup_kill_owned_processes(tmp_path: Path):
    manager = ProcessManager()
    job = await manager.start("sleep 30", cwd=tmp_path, timeout_s=0.05)
    assert (await manager.poll(job.handle, wait_s=3)).state == "timed_out"
    abort = asyncio.Event()
    job = await manager.start("sleep 30", cwd=tmp_path, abort=abort)
    abort.set()
    assert (await manager.poll(job.handle, wait_s=3)).state == "aborted"
    job = await manager.start("sleep 30", cwd=tmp_path)
    await manager.aclose()
    assert (await manager.poll(job.handle)).state == "terminated"
    with pytest.raises(RuntimeError, match="closed"):
        await manager.start("true", cwd=tmp_path)


async def test_capacity_and_finished_record_eviction(tmp_path: Path):
    async with ProcessManager(max_running=1, max_records=1) as manager:
        job = await manager.start("sleep 30", cwd=tmp_path)
        with pytest.raises(RuntimeError, match="limit"):
            await manager.start("true", cwd=tmp_path)
        await manager.terminate(job.handle)
        next_job = await manager.start("true", cwd=tmp_path)
        await manager.poll(next_job.handle, wait_s=3)
        assert len(manager.list()) == 1


async def test_model_tool_pack_uses_same_handle_contract(tmp_path: Path):
    # A base tool, and still its own pack for configs that list it.
    [tool] = [tool for tool in load_packs(["base", "process"]) if isinstance(tool, ProcessTool)]
    ctx = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=10_000)
    try:
        args = tool.parse({"action": "start", "command": "printf ready"})
        assert not isinstance(args, ToolOutput)
        start = ProcessSnapshot.model_validate_json((await tool.run(args, ctx)).content)
        await tool.manager.poll(start.handle, wait_s=3)  # the harness waits; the agent's poll only reads
        args = tool.parse({"action": "poll", "handle": start.handle})
        assert not isinstance(args, ToolOutput)
        result = ProcessSnapshot.model_validate_json((await tool.run(args, ctx)).content)
        assert result.output == "ready" and result.state == "exited"
    finally:
        await tool.aclose()


def _is_running(pid: int) -> bool:
    # An orphan can briefly remain as a zombie until its OS parent reaps it.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, check=False).stdout.strip()
    return bool(state) and not state.startswith(b"Z")


async def _wait_for_pid(path: Path) -> int:
    def read() -> str:
        with contextlib.suppress(FileNotFoundError):
            return path.read_text()
        return ""

    async with asyncio.timeout(3):
        for _ in range(300):
            value = await asyncio.to_thread(read)
            if value:
                return int(value)
            await asyncio.sleep(0.01)
    pytest.fail("Descendant did not report its PID")


@pytest.mark.parametrize("completion", ["close", "abort", "timeout", "supervisor-cancel", "parent-exit"])
async def test_completion_stops_quiet_term_ignoring_descendant(tmp_path: Path, completion: str):
    pid_path = tmp_path / "descendant.pid"
    child = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(60)"
    )
    tail = (
        f"while [ ! -s {shlex.quote(str(pid_path))} ]; do sleep 0.01; done" if completion == "parent-exit" else "wait"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child)} </dev/null >/dev/null 2>&1 & {tail}"
    group: int | None = None
    abort = asyncio.Event()
    manager = ProcessManager()
    try:
        async with manager:
            job = await manager.start(
                command, cwd=tmp_path, timeout_s=1 if completion == "timeout" else 30, abort=abort
            )
            pid = await _wait_for_pid(pid_path)
            group = os.getpgid(pid)
            if completion == "abort":
                abort.set()
            elif completion == "supervisor-cancel":
                task = manager._jobs[job.handle].task
                assert task is not None
                task.cancel()
            if completion != "close":
                result = await manager.poll(job.handle, wait_s=5)
                assert (
                    result.state
                    == {
                        "abort": "aborted",
                        "timeout": "timed_out",
                        "supervisor-cancel": "terminated",
                        "parent-exit": "exited",
                    }[completion]
                )
        async with asyncio.timeout(3):
            for _ in range(300):
                if not await asyncio.to_thread(_is_running, pid):
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("Owned descendant is still running after cleanup")
        assert manager._jobs[job.handle].done.is_set()
    finally:
        if group is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(group, signal.SIGKILL)
        await manager.aclose()


def test_event_loop_shutdown_stops_quiet_descendant(tmp_path: Path):
    pid_path = tmp_path / "shutdown-descendant.pid"
    child = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"open({str(pid_path)!r},'w').write(str(os.getpid())); time.sleep(60)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(child)} </dev/null >/dev/null 2>&1 & wait"
    script = f"""
import asyncio
from pathlib import Path
from intern.processes import ProcessManager
async def main():
    manager = ProcessManager()
    await manager.start({command!r}, cwd=Path({str(tmp_path)!r}))
    while not Path({str(pid_path)!r}).exists():
        await asyncio.sleep(0.01)
asyncio.run(main())
"""
    pid: int | None = None
    try:
        subprocess.run([sys.executable, "-c", script], check=True, timeout=10)
        pid = int(pid_path.read_text())
        assert not _is_running(pid)
    finally:
        if pid is None and pid_path.exists():
            pid = int(pid_path.read_text())
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(os.getpgid(pid), signal.SIGKILL)


@pytest.mark.parametrize(("status", "expected"), [(b"101 Z\n", False), (b"101 Z\n101 S\n", True), (b"102 S\n", False)])
def test_group_status_distinguishes_zombies_and_live_members(monkeypatch, status: bytes, expected: bool):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, status))
    assert ProcessManager._live_group(101) is expected


async def test_permission_denial_ignores_only_completed_group(monkeypatch, tmp_path: Path):
    async with ProcessManager() as manager:
        started = await manager.start("sleep 30", cwd=tmp_path)
        job = manager._jobs[started.handle]

        def denied(*args):
            raise PermissionError("permission denied")

        with monkeypatch.context() as patch:
            patch.setattr(os, "killpg", denied)
            patch.setattr(manager, "_live_group", lambda group: False)
            assert await manager._signal(job, 0) is False
            assert await manager._signal(job, signal.SIGKILL) is False
            patch.setattr(manager, "_live_group", lambda group: True)
            with pytest.raises(PermissionError):
                await manager._signal(job, signal.SIGKILL)
