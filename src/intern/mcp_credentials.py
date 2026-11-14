"""Credential commands: a host command that prints one HTTP header value for an MCP connection.

The value is fetched once when the session is set up and again only after the server answers 401,
never on a timer. The command is operator configuration: an absolute path with literal arguments, run
without a shell, in its own process group, in a directory the model does not write.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import AsyncGenerator, Callable, Generator, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import SecretStr

from intern.executor import stop_process_group

MAX_STDOUT_BYTES = 64 * 1024
MAX_VALUE_CHARS = 8192
STDERR_TAIL_CHARS = 512
_READ_CHUNK = 64 * 1024
_POLL_S = 0.02
_DRAIN_S = 1.0


class CredentialCommandError(Exception):
    """A credential command failed.

    `reason` is a fixed phrase (`exit N`, `timeout`, `output too large`, `invalid value`, `could not start`).
    `stderr_tail` is the command's raw output and must be redacted before anyone sees it.
    """

    def __init__(self, reason: str, stderr_tail: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.stderr_tail = stderr_tail


@dataclass(frozen=True, repr=False)
class CredentialCommand:
    command: str
    args: tuple[str, ...]
    env: Mapping[str, SecretStr]
    """The command's whole environment, already resolved."""
    cwd: Path
    timeout_s: float

    def __repr__(self) -> str:
        return f"CredentialCommand(command={self.command!r})"


async def _read_capped(stream: asyncio.StreamReader, limit: int, over: asyncio.Event) -> bytes:
    data = bytearray()
    while chunk := await stream.read(_READ_CHUNK):
        data.extend(chunk)
        if len(data) > limit:
            over.set()
            break
    return bytes(data)


async def _read_tail(stream: asyncio.StreamReader, keep: int) -> bytes:
    tail = bytearray()
    while chunk := await stream.read(_READ_CHUNK):
        tail.extend(chunk)
        del tail[:-keep]
    return bytes(tail)


def _value(stdout: bytes) -> str | None:
    try:
        text = stdout.decode("ascii")
    except UnicodeDecodeError:
        return None
    for newline in ("\r\n", "\n"):
        if text.endswith(newline):
            text = text[: -len(newline)]
            break
    if not 1 <= len(text) <= MAX_VALUE_CHARS or any(not 0x20 <= ord(c) <= 0x7E for c in text):
        return None
    return text


async def run_credential_command(
    spec: CredentialCommand, *, value: Callable[[bytes], str | None] | None = None
) -> SecretStr:
    """Run the command once and return the header value it printed (or what `value` makes of its output; None from
    `value` is an invalid value).

    Its process group is stopped before this returns or raises, including on cancellation, so a helper that
    leaves a background process behind cannot outlive the call.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            spec.command,
            *spec.args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=spec.cwd,
            env={name: value.get_secret_value() for name, value in spec.env.items()},
            start_new_session=True,
        )
    except OSError:
        raise CredentialCommandError("could not start") from None
    assert process.stdout is not None and process.stderr is not None
    over = asyncio.Event()
    reader = asyncio.create_task(_read_capped(process.stdout, MAX_STDOUT_BYTES, over))
    errors = asyncio.create_task(_read_tail(process.stderr, STDERR_TAIL_CHARS * 4))
    reason: str | None = None
    try:
        try:
            async with asyncio.timeout(spec.timeout_s):
                # Poll the leader's exit: `Process.wait()` also waits for any child still holding the pipes.
                while process.returncode is None and not over.is_set():  # noqa: ASYNC110
                    await asyncio.sleep(_POLL_S)
        except TimeoutError:
            reason = "timeout"
        if over.is_set():
            reason = "output too large"
    finally:
        await stop_process_group(process)
        _, pending = await asyncio.wait({reader, errors}, timeout=_DRAIN_S)
        for task in pending:
            task.cancel()
        for task in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await task
    stdout = reader.result() if not reader.cancelled() else b""
    stderr = errors.result() if not errors.cancelled() else b""
    tail = stderr.decode(errors="replace")[-STDERR_TAIL_CHARS:]
    if reason is not None:
        raise CredentialCommandError(reason, tail)
    if process.returncode != 0:
        raise CredentialCommandError(f"exit {process.returncode}", tail)
    printed = (value or _value)(stdout)
    if printed is None:
        raise CredentialCommandError("invalid value", tail)
    return SecretStr(printed)


_INVALID_TOKEN = re.compile(r'\berror\s*=\s*"?invalid_token"?', re.IGNORECASE)


def rejected(response: httpx.Response) -> bool:
    """Whether the server rejected the credential: a 401, or any status whose `WWW-Authenticate` says
    `error="invalid_token"` (RFC 6750), which some servers send with 400 or 403 instead."""
    return response.status_code == 401 or bool(_INVALID_TOKEN.search(response.headers.get("www-authenticate", "")))


class CommandCredential(httpx.Auth):
    """Set one header from a credential command, and run the command again once after a 401.

    Requests that get a 401 for the same value share one run: the generation check skips the run when another
    request already replaced the value. Each request is retried at most once; a second 401 is returned as is.
    """

    requires_request_body = True
    requires_response_body = False

    def __init__(self, spec: CredentialCommand, header: str, initial: SecretStr) -> None:
        self._spec = spec
        self._header = header
        self._value = initial
        self._generation = 0
        self._lock = asyncio.Lock()

    def __repr__(self) -> str:
        return f"CommandCredential(header={self._header!r})"

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        raise RuntimeError("CommandCredential supports only httpx.AsyncClient")

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        sent = self._generation
        request.headers[self._header] = self._value.get_secret_value()
        response = yield request
        if not rejected(response):
            return
        async with self._lock:
            if self._generation == sent:
                try:
                    self._value = await run_credential_command(self._spec)
                except CredentialCommandError as error:
                    raise CredentialCommandError(error.reason) from None
                self._generation += 1
        request.headers[self._header] = self._value.get_secret_value()
        yield request
