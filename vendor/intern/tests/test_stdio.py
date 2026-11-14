"""Real pipes remain cancellable while their writer is still open."""

from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, TextIO

import pytest

from intern.protocol import stdio
from intern.protocol.stdio import read_lines

POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX pipe/terminal readiness")


@contextmanager
def pipe_input() -> Iterator[tuple[TextIO, BinaryIO]]:
    reader, writer = os.pipe()
    with os.fdopen(reader, encoding="utf-8") as stream, os.fdopen(writer, "wb", buffering=0) as output:
        yield stream, output


@pytest.mark.parametrize("text", ["", "one\n\ntwo\n", "one\n\nfinal", "é🙂\n"])
async def test_stringio_preserves_lines_and_borrowed_stream(text: str):
    stream = io.StringIO(text)
    assert [line async for line in read_lines(stream)] == text.splitlines(keepends=True)
    assert not stream.closed


async def test_regular_file_respects_text_stream_position(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("already read\né🙂\n\nfinal", encoding="utf-8")
    with path.open(encoding="utf-8") as stream:
        assert stream.readline() == "already read\n"
        assert [line async for line in read_lines(stream)] == ["é🙂\n", "\n", "final"]
        assert not stream.closed


@POSIX_ONLY
async def test_devnull_is_immediate_eof_and_stays_open():
    path = Path(os.devnull)
    with path.open() as stream:  # noqa: ASYNC230 -- /dev/null opens locally and cannot block on input.
        assert [line async for line in read_lines(stream)] == []
        assert not stream.closed


@POSIX_ONLY
async def test_pipe_decodes_split_utf8_and_preserves_blank_and_final_lines(monkeypatch: pytest.MonkeyPatch):
    read = os.read
    monkeypatch.setattr(stdio.os, "read", lambda descriptor, count: read(descriptor, min(count, 1)))
    with pipe_input() as (stream, writer):
        lines = read_lines(stream)
        first = asyncio.ensure_future(anext(lines))
        writer.write(b"\xc3")
        await asyncio.sleep(0)
        assert not first.done()
        writer.write(b"\xa9\xf0\x9f")
        await asyncio.sleep(0)
        assert not first.done()
        writer.write(b"\x99\x82\n\nfinal")
        assert await asyncio.wait_for(first, 2) == "é🙂\n"
        assert await anext(lines) == "\n"
        writer.close()
        assert await asyncio.wait_for(anext(lines), 2) == "final"
        with pytest.raises(StopAsyncIteration):
            await anext(lines)
        assert not stream.closed


@POSIX_ONLY
async def test_terminal_input_and_cancellation_leave_borrowed_stream_open():
    import pty

    master, slave = pty.openpty()
    with os.fdopen(slave, encoding="utf-8") as stream, os.fdopen(master, "wb", buffering=0) as writer:
        assert stream.isatty()
        lines = read_lines(stream)
        pending = asyncio.ensure_future(anext(lines))
        await asyncio.sleep(0)
        writer.write(b"terminal line\n")
        assert await asyncio.wait_for(pending, 2) == "terminal line\n"
        pending = asyncio.ensure_future(anext(lines))
        await asyncio.sleep(0)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not stream.closed
        assert os.get_blocking(stream.fileno())


@POSIX_ONLY
async def test_pipe_translates_split_universal_newlines():
    with pipe_input() as (stream, writer):
        lines = read_lines(stream)
        first = asyncio.ensure_future(anext(lines))
        writer.write(b"one\r")
        await asyncio.sleep(0)
        writer.write(b"\n\rtwo\r")
        writer.close()
        assert await asyncio.wait_for(first, 2) == "one\n"
        assert [line async for line in lines] == ["\n", "two\n"]


@POSIX_ONLY
async def test_cancel_removes_owned_descriptor_and_leaves_input_reusable(monkeypatch: pytest.MonkeyPatch):
    duplicates: list[int] = []
    duplicate = os.dup

    def record_duplicate(descriptor: int) -> int:
        owned = duplicate(descriptor)
        duplicates.append(owned)
        return owned

    monkeypatch.setattr(stdio.os, "dup", record_duplicate)
    with pipe_input() as (stream, writer):
        original_blocking = os.get_blocking(stream.fileno())
        lines = read_lines(stream)
        pending = asyncio.ensure_future(anext(lines))
        await asyncio.sleep(0)
        assert duplicates and not pending.done()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not stream.closed
        assert os.get_blocking(stream.fileno()) == original_blocking
        with pytest.raises(OSError):
            os.fstat(duplicates[0])

        # Reuse the same input (and normally the same duplicate number) after cancel.
        replacement = read_lines(stream)
        waiting = asyncio.ensure_future(anext(replacement))
        await asyncio.sleep(0)
        writer.write(b"still usable\n")
        assert await asyncio.wait_for(waiting, 2) == "still usable\n"
        await replacement.aclose()
        with pytest.raises(OSError):
            os.fstat(duplicates[-1])


@POSIX_ONLY
async def test_unsupported_loop_fails_without_closing_borrowed_input(monkeypatch: pytest.MonkeyPatch):
    loop = asyncio.get_running_loop()

    def unsupported(*_args: object) -> None:
        raise NotImplementedError

    with pipe_input() as (stream, _writer):
        monkeypatch.setattr(loop, "add_reader", unsupported)
        with pytest.raises(RuntimeError, match="does not support cancellable input readiness"):
            await anext(read_lines(stream))
        assert not stream.closed


@POSIX_ONLY
def test_cancelled_child_exits_while_parent_keeps_stdin_open():
    script = """
import asyncio
import sys
from intern.protocol.stdio import read_lines

async def main():
    task = asyncio.create_task(anext(read_lines(sys.stdin)))
    await asyncio.sleep(0)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        print('cancelled', flush=True)

asyncio.run(main())
print('exited', flush=True)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        assert process.wait(timeout=5) == 0
        assert not process.stdin.closed
        assert process.stdout is not None and process.stderr is not None
        assert process.stdout.read() == b"cancelled\nexited\n"
        assert process.stderr.read() == b""
    finally:
        if process.stdin is not None:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
