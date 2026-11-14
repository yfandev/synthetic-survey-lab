"""Cancellable text input without a background thread owning borrowed stdin."""

from __future__ import annotations

import asyncio
import codecs
import io
import os
import stat
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from typing import TextIO


async def read_lines(stream: TextIO) -> AsyncGenerator[str, None]:
    """Yield lines, including their newline and any final unterminated text.

    StringIO, regular files and POSIX /dev/null use synchronous reads. Pipes and terminals use
    readiness notifications and incremental decoding, with universal newlines like
    ordinary stdin. The caller must provide exclusive, previously unread input:
    descriptor reads cannot recover characters already buffered by a text wrapper.
    Cancellation ends this reader; partially consumed input is not replayed.

    Only an owned duplicate is registered and closed. The borrowed stream and its
    blocking mode are unchanged. Unsupported descriptors/event loops fail explicitly
    rather than starting an executor thread that could outlive cancellation. Windows
    console/pipe readiness is not supported by this implementation.
    """
    if isinstance(stream, io.StringIO):
        async for line in _file_lines(stream):
            yield line
        return

    try:
        descriptor = stream.fileno()
    except (AttributeError, io.UnsupportedOperation) as error:
        raise RuntimeError("cancellable input requires StringIO or a supported file descriptor") from error
    metadata = os.fstat(descriptor)
    null_input = False
    if os.name == "posix" and stat.S_ISCHR(metadata.st_mode):
        # The local null device is safe to read synchronously, unlike arbitrary devices.
        null_input = metadata.st_rdev == Path(os.devnull).stat().st_rdev  # noqa: ASYNC240
    if stat.S_ISREG(metadata.st_mode) or null_input:
        async for line in _file_lines(stream):
            yield line
        return
    if os.name != "posix":
        raise RuntimeError("cancellable pipe/terminal input requires POSIX readiness support")

    loop = asyncio.get_running_loop()
    decoder = io.IncrementalNewlineDecoder(
        codecs.getincrementaldecoder(stream.encoding or "utf-8")(errors=stream.errors or "strict"), translate=True
    )
    descriptor = os.dup(descriptor)
    pending: list[str] = []
    try:
        while True:
            ready: asyncio.Future[None] = loop.create_future()

            def readable(signal: asyncio.Future[None] = ready) -> None:
                if not signal.done():
                    signal.set_result(None)

            try:
                loop.add_reader(descriptor, readable)
            except (NotImplementedError, OSError) as error:
                raise RuntimeError("event loop does not support cancellable input readiness") from error
            try:
                await ready
            finally:
                loop.remove_reader(descriptor)
            try:
                chunk = os.read(descriptor, 65536)
            except (BlockingIOError, InterruptedError):
                continue
            pieces = decoder.decode(chunk, final=not chunk).split("\n")
            for piece in pieces[:-1]:
                pending.append(piece)
                yield "".join(pending) + "\n"
                pending.clear()
            if pieces[-1]:
                pending.append(pieces[-1])
            if not chunk:
                if pending:
                    yield "".join(pending)
                return
    finally:
        os.close(descriptor)


async def _file_lines(stream: TextIO) -> AsyncIterator[str]:
    while True:
        # Let cancellation run even if the consumer performs no other async work.
        await asyncio.sleep(0)
        line = stream.readline()
        if not line:
            return
        yield line
