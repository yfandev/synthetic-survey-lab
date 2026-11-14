from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import pytest

from intern.executor import LocalExecutor
from intern.mcp_client import StdioConfig, http_connection, stdio_connection
from intern.tools.base import ToolContext, ToolOutput

pytest.importorskip("mcp")
pytest.importorskip("jsonschema")


async def test_real_sdk_stdio_discovery_schema_call_and_close(tmp_path: Path):
    server = tmp_path / "server.py"
    server.write_text("""from mcp.server.fastmcp import FastMCP
mcp=FastMCP("test")
@mcp.tool()
def add(a:int,b:int)->int: return a+b
@mcp.tool()
def fail()->str: raise ValueError("test failure")
mcp.run(transport="stdio")
""")
    config = StdioConfig(command=sys.executable, args=[str(server)])
    ctx = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=1000)
    async with stdio_connection(config, "test") as connection:
        tools = await connection.tools({"add": frozenset({"network"}), "fail": frozenset({"network"})})
        add, fail = tools
        assert add.name == "mcp__test__add" and not add.read_only
        assert add.spec().parameters["required"] == ["a", "b"]
        args = add.parse({"a": 2, "b": 3})
        assert not isinstance(args, ToolOutput)
        assert "5" in (await add.run(args, ctx)).content
        invalid = add.parse({"a": "bad", "b": 3})
        assert not isinstance(invalid, ToolOutput)
        assert (await add.run(invalid, ctx)).status == "error"
        args = fail.parse({})
        assert not isinstance(args, ToolOutput)
        assert (await fail.run(args, ctx)).status == "error"
        with pytest.raises(ValueError, match="absent"):
            await connection.tools({"missing": frozenset({"network"})})
    args = add.parse({"a": 1, "b": 1})
    assert not isinstance(args, ToolOutput)
    assert (await add.run(args, ctx)).status == "error"


async def test_aborted_call_never_reaches_server(tmp_path: Path):
    from intern.mcp_client import McpConnection, _Description

    class NoSession:
        def call_tool(self, *args, **kwargs):
            raise AssertionError("must not call remote service")

    class Validator:
        def validate(self, value):
            pass

    connection = McpConnection(NoSession(), "test")
    tool = connection._tool(
        _Description(name="do", inputSchema={"type": "object"}), "mcp__test__do", frozenset({"network"}), Validator()
    )
    abort = asyncio.Event()
    abort.set()
    ctx = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=abort, max_output_chars=1000)
    args = tool.parse({})
    assert not isinstance(args, ToolOutput)
    assert (await tool.run(args, ctx)).status == "refused"


@pytest.mark.parametrize("failure", ["timeout", "abort", "transport", "remote_error"])
async def test_mcp_failure_after_dispatch_requires_outcome_verification(tmp_path, failure):
    from types import SimpleNamespace

    from intern.mcp_client import McpConnection, _Description

    entered = asyncio.Event()
    effects = []

    class Remote:
        async def call_tool(self, *args, **kwargs):
            effects.append("effect")
            entered.set()
            if failure == "transport":
                raise ConnectionError("response lost after effect")
            if failure == "remote_error":
                return SimpleNamespace(model_dump=lambda **kwargs: {"isError": True, "content": []})
            await asyncio.Event().wait()
            return None

    class Validator:
        def validate(self, value):
            pass

    connection = McpConnection(Remote(), "test", timeout_s=0.05 if failure == "timeout" else 1)
    tool = connection._tool(
        _Description(name="mutate", inputSchema={"type": "object"}),
        "mcp__test__mutate",
        frozenset({"write"}),
        Validator(),
    )
    ctx = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=1000)
    args = tool.parse({})
    assert not isinstance(args, ToolOutput)
    pending = asyncio.create_task(tool.run(args, ctx))
    await asyncio.wait_for(entered.wait(), 1)
    if failure == "abort":
        ctx.abort.set()
    result = await asyncio.wait_for(pending, 2)
    assert effects == ["effect"]
    assert result.error is not None and not result.error.retry_safe


async def test_real_streamable_http_auth_inflight_abort_and_reuse(tmp_path: Path):
    import httpx

    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    server = tmp_path / "http_server.py"
    started = tmp_path / "slow.started"
    server.write_text(f"""import asyncio
import uvicorn
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
mcp=FastMCP("http-test",host="127.0.0.1",port={port})
@mcp.tool()
def add(a:int,b:int)->int: return a+b
@mcp.tool()
async def slow()->str:
    Path({str(started)!r}).write_text("started")
    await asyncio.sleep(30)
    return "too late"
inner=mcp.streamable_http_app()
async def app(scope,receive,send):
    if scope["type"]=="http" and (b"authorization",b"Bearer test-token") not in scope["headers"]:
        return await JSONResponse({{"error":"unauthorized"}},status_code=401)(scope,receive,send)
    await inner(scope,receive,send)
uvicorn.run(app,host="127.0.0.1",port={port},log_level="critical")
""")
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(server), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        async with httpx.AsyncClient(headers={"Authorization": "Bearer test-token"}, timeout=2) as client:
            for _ in range(100):
                try:
                    response = await client.get(url)
                    assert response.status_code != 401
                    break
                except httpx.ConnectError:
                    assert process.returncode is None, "MCP test server exited during startup"
                    await asyncio.sleep(0.05)
            else:
                pytest.fail("MCP test server did not start")
            async with httpx.AsyncClient() as unauthenticated:
                assert (await unauthenticated.get(url)).status_code == 401
            async with http_connection(url, "http", timeout_s=2, http_client=client) as connection:
                add, slow = await connection.tools({"add": frozenset({"network"}), "slow": frozenset({"network"})})
                ctx = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=1000)
                arguments = add.parse({"a": 4, "b": 5})
                assert not isinstance(arguments, ToolOutput)
                assert "9" in (await add.run(arguments, ctx)).content
                arguments = slow.parse({})
                assert not isinstance(arguments, ToolOutput)
                pending = asyncio.create_task(slow.run(arguments, ctx))
                async with asyncio.timeout(2):
                    for _ in range(200):
                        if await asyncio.to_thread(started.exists):
                            break
                        await asyncio.sleep(0.01)
                    else:
                        pytest.fail("Slow MCP tool never started")
                ctx.abort.set()
                assert "interrupted" in (await asyncio.wait_for(pending, 3)).content
                ctx.abort.clear()
                arguments = add.parse({"a": 10, "b": 1})
                assert not isinstance(arguments, ToolOutput)
                assert "11" in (await add.run(arguments, ctx)).content
            assert not client.is_closed
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await process.wait()
