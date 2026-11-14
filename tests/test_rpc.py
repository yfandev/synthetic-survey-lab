"""JSON-RPC request handling, in process."""

from __future__ import annotations

import io
import json

from intern.protocol.rpc import RpcServer, Writer
from intern.providers.fake import Turn
from intern.session import Session


def server_for(harness) -> tuple[RpcServer, io.StringIO]:
    out = io.StringIO()
    server = RpcServer(Writer(out))
    server.session = Session(harness.deps)
    return server, out


def replies(out: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in out.getvalue().splitlines()]


async def test_malformed_requests_get_errors_and_the_server_keeps_serving(make_harness):
    server, out = server_for(make_harness([Turn(text="ok")]))
    await server.handle('{"jsonrpc": "2.0", "id": 1, "method": ["get_state"]}')
    await server.handle("not json")
    await server.handle('{"jsonrpc": "2.0", "id": 2, "method": "get_state"}')
    first, second, third = replies(out)
    assert first["error"]["code"] == -32600 and first["id"] == 1
    assert second["error"]["code"] == -32700
    assert third["result"]["running"] is False


async def test_notifications_get_no_response(make_harness):
    server, out = server_for(make_harness([Turn(text="ok")]))
    await server.handle('{"jsonrpc": "2.0", "method": "nope"}')
    await server.handle('{"jsonrpc": "2.0", "method": "get_state"}')
    assert out.getvalue() == ""


async def test_unexpected_handler_errors_become_internal_errors(make_harness):
    server, out = server_for(make_harness([Turn(text="ok")]))

    async def explode(_params):
        raise OSError("disk full")

    server._methods["get_state"] = (server._methods["get_state"][0], explode)  # pyright: ignore[reportPrivateUsage]
    await server.handle('{"jsonrpc": "2.0", "id": 7, "method": "get_state"}')
    assert replies(out)[0]["error"] == {"code": -32603, "message": "OSError: disk full"}
