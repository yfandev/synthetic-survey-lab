from __future__ import annotations

import asyncio
import contextlib
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from intern.executor import LocalExecutor, shell_env


async def test_output_keeps_head_and_tail_within_the_cap(tmp_path: Path):
    result = await LocalExecutor().run("seq 1 100000", cwd=tmp_path, timeout_s=30, max_output_bytes=3_000)
    assert result.exit_code == 0
    assert result.output.startswith("1\n2\n")
    assert result.output.rstrip().endswith("100000")
    assert result.omitted_bytes > 0
    assert len(result.output) < 3_200


async def test_capture_keeps_middle_before_display_truncation_and_reports_disk_limit(tmp_path, monkeypatch):
    import intern.executor as executor_module

    capture = tmp_path / "complete.txt"
    result = await LocalExecutor().run(
        "seq 1 10000",
        cwd=tmp_path,
        timeout_s=5,
        max_output_bytes=100,
        capture_path=capture,
    )
    assert result.omitted_bytes and result.capture_complete
    assert "5000\n" not in result.output and b"\n5000\n" in capture.read_bytes()
    monkeypatch.setattr(executor_module, "MAX_CAPTURE_BYTES", 1000)
    limited = tmp_path / "limited.txt"
    result = await LocalExecutor().run(
        "seq 1 10000",
        cwd=tmp_path,
        timeout_s=5,
        max_output_bytes=100,
        capture_path=limited,
    )
    assert limited.read_bytes() == capture.read_bytes()[:1000]
    assert result.capture_omitted_bytes == capture.stat().st_size - 1000 and not result.capture_complete


async def test_failed_capture_stops_the_command_without_waiting_for_its_timeout(tmp_path, monkeypatch):
    import io

    import pytest

    import intern.executor as executor_module

    class FailedCapture(io.BytesIO):
        def write(self, data):
            raise OSError("capture disk full")

    monkeypatch.setattr(executor_module, "_open_capture", lambda path: FailedCapture())
    with pytest.raises(OSError, match="disk full"):
        await asyncio.wait_for(
            LocalExecutor().run(
                "printf output; sleep 60",
                cwd=tmp_path,
                timeout_s=60,
                max_output_bytes=100,
                capture_path=tmp_path / "failed.txt",
            ),
            3,
        )


async def test_timeout_kills_the_whole_process_tree(tmp_path: Path):
    marker = tmp_path / "still-running"
    started = time.monotonic()
    result = await LocalExecutor().run(
        f"(sleep 3 && touch {marker}) & sleep 30", cwd=tmp_path, timeout_s=0.5, max_output_bytes=1_000
    )
    assert result.timed_out and result.exit_code is None
    assert time.monotonic() - started < 6
    await asyncio.sleep(3.5)
    assert not marker.exists(), "the background child survived the timeout"


async def test_cancelling_a_run_kills_its_process_group(tmp_path: Path):
    marker = tmp_path / "survived"
    task = asyncio.create_task(
        LocalExecutor().run(f"sleep 2 && touch {marker}", cwd=tmp_path, timeout_s=60, max_output_bytes=1_000)
    )
    await asyncio.sleep(0.3)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await asyncio.sleep(2.5)
    assert not marker.exists(), "the command kept running after its task was cancelled"


async def test_abort_stops_a_command(tmp_path: Path):
    abort = asyncio.Event()
    asyncio.get_running_loop().call_later(0.3, abort.set)
    result = await LocalExecutor().run("sleep 30", cwd=tmp_path, timeout_s=60, max_output_bytes=1_000, abort=abort)
    assert result.aborted


@pytest.mark.parametrize("stop", ["timeout", "abort", "cancel"])
async def test_parent_exit_and_closed_pipes_do_not_hide_a_term_ignoring_child(tmp_path, stop):
    marker = tmp_path / "child.pid"
    script = (
        "import os,signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(marker)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} >/dev/null 2>&1 & wait"
    abort = asyncio.Event()
    task = asyncio.create_task(
        LocalExecutor().run(
            command,
            cwd=tmp_path,
            timeout_s=0.5 if stop == "timeout" else 60,
            max_output_bytes=1000,
            abort=abort,
        )
    )
    pid = None
    try:
        async with asyncio.timeout(3):
            for _ in range(300):
                if marker.exists():
                    break
                await asyncio.sleep(0.01)
        pid = int(marker.read_text())
        if stop == "abort":
            abort.set()
        elif stop == "cancel":
            task.cancel()
        if stop == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await asyncio.wait_for(task, 5)
            assert result.timed_out if stop == "timeout" else result.aborted
        async with asyncio.timeout(3):
            while True:
                status = (
                    await asyncio.to_thread(
                        subprocess.run, ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, check=False
                    )
                ).stdout.strip()
                if not status or status.startswith(b"Z"):
                    break
                await asyncio.sleep(0.02)
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if pid is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)


async def test_successful_background_command_keeps_its_process(tmp_path):
    marker = tmp_path / "background.pid"
    result = await LocalExecutor().run(
        f"sleep 60 >/dev/null 2>&1 & echo $! > {shlex.quote(str(marker))}",
        cwd=tmp_path,
        timeout_s=5,
        max_output_bytes=1000,
    )
    pid = int(marker.read_text())
    try:
        assert result.exit_code == 0
        os.kill(pid, 0)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def test_shell_env_drops_only_the_harness_venv(tmp_path: Path):
    own, other = tmp_path / "harness-venv", tmp_path / "user-venv"
    environ = {
        "PATH": f"{own}/bin:{other}/bin:/usr/bin",
        "VIRTUAL_ENV": str(own),
        "HOME": "/home/me",
    }
    env = shell_env(environ, prefix=str(own), base_prefix="/usr")
    assert env["PATH"] == f"{other}/bin:/usr/bin"
    assert "VIRTUAL_ENV" not in env
    assert env["HOME"] == "/home/me"
    # A user's own activated venv is theirs to keep.
    assert shell_env({**environ, "VIRTUAL_ENV": str(other)}, prefix=str(own), base_prefix="/usr")["VIRTUAL_ENV"] == str(
        other
    )
    # Not running from a venv: nothing changes.
    assert shell_env(environ, prefix="/usr", base_prefix="/usr") == environ
