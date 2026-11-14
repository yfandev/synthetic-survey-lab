"""JSON-RPC 2.0 over stdio, one JSON object per line.

Clients send requests on stdin; the server answers on stdout and streams every engine event as an
`event` notification. Methods: prompt, steer, follow_up, abort, get_state, set_model, branch,
get_entries.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import sys
from collections.abc import Awaitable, Callable
from typing import Any, TextIO

from pydantic import BaseModel, ValidationError

from intern.events import Event
from intern.journal import Entry
from intern.models.registry import ConfigError, Role
from intern.protocol.stdio import read_lines
from intern.session import Session, SessionBusy, SessionIdle

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
SESSION_BUSY = -32000
SESSION_IDLE = -32001
CONFIG_ERROR = -32002
INTERNAL_ERROR = -32603


class TextParams(BaseModel):
    text: str


class SetModelParams(BaseModel):
    model: str
    role: Role = "system2"


class BranchParams(BaseModel):
    entry_id: str


class EntriesParams(BaseModel):
    after_id: str | None = None


class NoParams(BaseModel):
    pass


class RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


class Writer:
    """Serialises everything the server writes, so lines never interleave."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream

    def send(self, payload: dict[str, Any]) -> None:
        self._stream.write(json.dumps(payload) + "\n")
        self._stream.flush()

    def event(self, event: Event) -> None:
        self.send({"jsonrpc": "2.0", "method": "event", "params": event.model_dump(mode="json")})


class RpcServer:
    def __init__(self, writer: Writer) -> None:
        self.writer = writer
        self.session: Session | None = None
        self._methods: dict[str, tuple[type[BaseModel], Callable[[Any], Awaitable[Any]]]] = {
            "prompt": (TextParams, self._prompt),
            "steer": (TextParams, self._steer),
            "follow_up": (TextParams, self._follow_up),
            "abort": (NoParams, self._abort),
            "get_state": (NoParams, self._get_state),
            "set_model": (SetModelParams, self._set_model),
            "branch": (BranchParams, self._branch),
            "get_entries": (EntriesParams, self._get_entries),
        }

    def _session(self) -> Session:
        assert self.session is not None
        return self.session

    async def _prompt(self, params: TextParams) -> dict[str, Any]:
        self._session().start(params.text)
        return {"started": True}

    async def _steer(self, params: TextParams) -> dict[str, Any]:
        self._session().steer(params.text)
        return {}

    async def _follow_up(self, params: TextParams) -> dict[str, Any]:
        started = self._session().follow_up(params.text) is not None
        return {"started": started, "queued": not started}

    async def _abort(self, _: NoParams) -> dict[str, Any]:
        self._session().abort()
        return {}

    async def _get_state(self, _: NoParams) -> dict[str, Any]:
        return self._session().state().model_dump(mode="json")

    async def _set_model(self, params: SetModelParams) -> dict[str, Any]:
        binding = await self._session().set_model(params.model, params.role)
        if binding is None:
            return {"model": params.model}
        return {"model": binding.ref, "window": binding.window.model_dump(mode="json")}

    async def _branch(self, params: BranchParams) -> dict[str, Any]:
        self._session().branch(params.entry_id)
        return {"leaf_id": self._session().deps.journal.leaf_id}

    async def _get_entries(self, params: EntriesParams) -> dict[str, Any]:
        entries: list[Entry] = self._session().entries(params.after_id)
        return {"entries": [e.model_dump(mode="json") for e in entries]}

    async def handle(self, line: str) -> None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            self._error(None, PARSE_ERROR, f"invalid JSON: {e}")
            return
        if not isinstance(request, dict):
            self._error(None, INVALID_REQUEST, "not a JSON-RPC 2.0 request")
            return
        # A request without an id is a notification: it gets no response, not even an error.
        notification = "id" not in request
        request_id = request.get("id")
        reply = (lambda *_: None) if notification else self._error
        method_name = request.get("method")
        if request.get("jsonrpc") != "2.0" or not isinstance(method_name, str):
            reply(request_id, INVALID_REQUEST, "not a JSON-RPC 2.0 request")
            return
        method = self._methods.get(method_name)
        if method is None:
            reply(request_id, METHOD_NOT_FOUND, f"unknown method {method_name!r}")
            return
        params_type, handler = method
        try:
            params = params_type.model_validate(request.get("params") or {})
            result = await handler(params)
        except ValidationError as e:
            reply(request_id, INVALID_PARAMS, str(e.errors()[0]["msg"]))
            return
        except SessionBusy as e:
            reply(request_id, SESSION_BUSY, str(e))
            return
        except SessionIdle as e:
            reply(request_id, SESSION_IDLE, str(e))
            return
        except (ConfigError, KeyError, ValueError) as e:
            reply(request_id, CONFIG_ERROR, str(e))
            return
        except Exception as e:
            reply(request_id, INTERNAL_ERROR, f"{type(e).__name__}: {e}")
            return
        if not notification:
            self.writer.send({"jsonrpc": "2.0", "id": request_id, "result": result})

    def _error(self, request_id: Any, code: int, message: str) -> None:
        self.writer.send({"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}})


async def serve(
    make_session: Callable[[Callable[[Event], None]], Awaitable[Session]], stdin: TextIO = sys.stdin
) -> int | None:
    """Read requests until stdin closes, then let any run in progress finish. SIGTERM or SIGHUP closes the session at
    once, which aborts a run in progress and stops its command; the signal's number is returned."""
    loop = asyncio.get_running_loop()
    main = asyncio.current_task()
    ended_by: list[int] = []

    def end(sig: int) -> None:
        ended_by.append(sig)
        if main is not None:
            main.cancel()

    with contextlib.suppress(NotImplementedError):
        for sig in (signal.SIGTERM, signal.SIGHUP):
            loop.add_signal_handler(sig, end, sig)
    try:
        await _serve(make_session, stdin)
    except asyncio.CancelledError:
        if not ended_by:
            raise
    finally:
        with contextlib.suppress(NotImplementedError):
            for sig in (signal.SIGTERM, signal.SIGHUP):
                loop.remove_signal_handler(sig)
    return ended_by[0] if ended_by else None


async def _serve(make_session: Callable[[Callable[[Event], None]], Awaitable[Session]], stdin: TextIO) -> None:
    writer = Writer(sys.stdout)
    server = RpcServer(writer)
    server.session = await make_session(writer.event)
    async with server.session:
        if sys.platform == "win32":
            # Preserve the existing Windows input path. The cancellable POSIX
            # descriptor adapter cannot read Windows console/pipe handles.
            # A pending Windows read still requires EOF to stop its worker.
            loop = asyncio.get_running_loop()
            while line := await loop.run_in_executor(None, stdin.readline):
                if line.strip():
                    await server.handle(line)
        else:
            async with contextlib.aclosing(read_lines(stdin)) as lines:
                async for line in lines:
                    if line.strip():
                        await server.handle(line)
        # Normal EOF keeps the existing drain behavior. Exceptions/cancellation
        # instead enter Session.aclose, which aborts active work before teardown.
        await server.session.wait()
