"""Credential commands: one header value from a host command, fetched at setup and again after a 401."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from intern.mcp_credentials import (
    MAX_STDOUT_BYTES,
    CommandCredential,
    CredentialCommand,
    CredentialCommandError,
    run_credential_command,
)
from intern.models.registry import ConfigError
from intern.session_resources import ResourceCleanupError

from .fake_openai import Reply
from .test_session_setup import configured, mcp_toml, outputs, stdio_script


def helper(path: Path, body: str) -> Path:
    """An executable Python helper; `body` runs with `os`, `sys` and `Path` imported."""
    path.write_text(f"#!{sys.executable}\nimport os, sys\nfrom pathlib import Path\n{body}\n")
    path.chmod(0o755)
    return path


def spec(command: Path, cwd: Path, *, timeout_s: float = 10, env: dict[str, str] | None = None) -> CredentialCommand:
    return CredentialCommand(
        command=str(command),
        args=(),
        env={name: SecretStr(value) for name, value in (env or {}).items()},
        cwd=cwd,
        timeout_s=timeout_s,
    )


def counting(path: Path, counter: Path, token: Path) -> Path:
    """Counts its runs and prints `Bearer <token file>`."""
    return helper(
        path,
        f"""c = Path({str(counter)!r})
c.write_text(str(int(c.read_text()) + 1) if c.exists() else "1")
print("Bearer " + Path({str(token)!r}).read_text())""",
    )


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def same_place(a: str, b: Path) -> bool:
    return Path(a).resolve() == b.resolve()


def written(path: Path) -> bool:
    return path.exists() and bool(path.read_text())


async def wait_for_file(path: Path) -> None:
    async with asyncio.timeout(5):
        while not await asyncio.to_thread(written, path):  # noqa: ASYNC110 - the helper is another process
            await asyncio.sleep(0.01)


async def test_value_is_printed_stdout_without_the_trailing_newline(tmp_path):
    command = helper(tmp_path / "ok", 'print("Bearer synthetic-value")')
    value = await run_credential_command(spec(command, tmp_path))
    assert value.get_secret_value() == "Bearer synthetic-value"


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        ('sys.stderr.write("boom"); sys.exit(3)', "exit 3"),
        ('sys.stdout.write("x" * ' + str(MAX_STDOUT_BYTES + 10) + ")", "output too large"),
        ('print("")', "invalid value"),
        ('print("Bearer caf\\u00e9")', "invalid value"),
        ('print("Bearer a\\tb")', "invalid value"),
    ],
)
async def test_bad_runs_fail_with_a_fixed_reason(tmp_path, body, reason):
    command = helper(tmp_path / "bad", body)
    with pytest.raises(CredentialCommandError) as caught:
        await run_credential_command(spec(command, tmp_path))
    assert caught.value.reason == reason
    assert str(caught.value) == reason
    if reason == "exit 3":
        assert caught.value.stderr_tail == "boom"


async def test_missing_command_cannot_start(tmp_path):
    with pytest.raises(CredentialCommandError, match="could not start"):
        await run_credential_command(spec(tmp_path / "absent", tmp_path))


async def test_timeout_and_cancellation_stop_the_whole_process_group(tmp_path):
    pid_file = tmp_path / "child"
    body = f"""import subprocess, time
child = subprocess.Popen(["sleep", "100"])
Path({str(pid_file)!r}).write_text(str(child.pid))
time.sleep(100)"""
    command = helper(tmp_path / "slow", body)
    started = time.monotonic()
    with pytest.raises(CredentialCommandError, match="timeout"):
        await run_credential_command(spec(command, tmp_path, timeout_s=0.5))
    assert time.monotonic() - started < 5
    assert gone(int(pid_file.read_text()))

    pid_file.unlink()
    task = asyncio.create_task(run_credential_command(spec(command, tmp_path, timeout_s=30)))
    await wait_for_file(pid_file)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert gone(int(pid_file.read_text()))


async def test_a_background_child_left_by_a_successful_run_is_stopped(tmp_path):
    pid_file = tmp_path / "child"
    body = f"""import subprocess
child = subprocess.Popen(["sleep", "100"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
Path({str(pid_file)!r}).write_text(str(child.pid))
print("Bearer synthetic-value")"""
    command = helper(tmp_path / "leaves", body)
    value = await run_credential_command(spec(command, tmp_path))
    assert value.get_secret_value() == "Bearer synthetic-value"
    assert gone(int(pid_file.read_text()))


async def test_command_sees_exactly_the_given_environment_and_directory(tmp_path):
    seen, home = tmp_path / "seen.json", tmp_path / "home"
    home.mkdir()
    command = helper(
        tmp_path / "env",
        f"""import json
Path({str(seen)!r}).write_text(json.dumps({{"env": dict(os.environ), "cwd": os.getcwd()}}))
print("Bearer synthetic-value")""",
    )
    await run_credential_command(spec(command, home, env={"CLIENT_SECRET": "synthetic-secret", "HOME": str(home)}))
    observed = json.loads(seen.read_text())
    # Added by the helper itself, not passed: macOS's __CF_USER_TEXT_ENCODING and Python's C-locale coercion.
    added = {"LC_CTYPE"}
    assert {k: v for k, v in observed["env"].items() if not k.startswith("__CF") and k not in added} == {
        "CLIENT_SECRET": "synthetic-secret",
        "HOME": str(home),
    }
    assert same_place(observed["cwd"], home)


def gated(accepted: Path, hits: list[str]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        hits.append(request.headers.get("authorization", ""))
        ok = request.headers.get("authorization") == "Bearer " + accepted.read_text()
        return httpx.Response(200 if ok else 401)

    return httpx.MockTransport(handle)


@pytest.mark.parametrize(
    ("status", "challenge", "refreshes"),
    [
        (400, 'Bearer error="invalid_token", error_description="Invalid token"', True),
        (403, "Bearer error=invalid_token", True),
        (403, 'Bearer error="insufficient_scope"', False),
        (400, "", False),
    ],
)
async def test_an_invalid_token_challenge_counts_as_rejected_on_any_status(tmp_path, status, challenge, refreshes):
    counter, token = tmp_path / "count", tmp_path / "token"
    token.write_text("new")

    def handle(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") == "Bearer new":
            return httpx.Response(200)
        return httpx.Response(status, headers={"WWW-Authenticate": challenge} if challenge else {})

    credential = CommandCredential(
        spec(counting(tmp_path / "cmd", counter, token), tmp_path), "Authorization", SecretStr("Bearer old")
    )
    async with httpx.AsyncClient(auth=credential, transport=httpx.MockTransport(handle)) as client:
        response = await client.post("http://mcp.test/mcp", content=b"{}")
    assert (response.status_code == 200) is refreshes
    assert counter.exists() is refreshes


async def test_a_401_runs_the_command_once_and_retries_once(tmp_path):
    counter, token, accepted = tmp_path / "count", tmp_path / "token", tmp_path / "accepted"
    token.write_text("new")
    accepted.write_text("new")
    credential = CommandCredential(
        spec(counting(tmp_path / "cmd", counter, token), tmp_path), "Authorization", SecretStr("Bearer old")
    )
    hits: list[str] = []
    async with httpx.AsyncClient(auth=credential, transport=gated(accepted, hits)) as client:
        assert (await client.post("http://mcp.test/mcp", content=b"{}")).status_code == 200
        assert hits == ["Bearer old", "Bearer new"]
        assert counter.read_text() == "1"

        accepted.write_text("never")
        hits.clear()
        assert (await client.post("http://mcp.test/mcp", content=b"{}")).status_code == 401
        assert len(hits) == 2
        assert counter.read_text() == "2"
    assert "new" not in repr(credential)


async def test_concurrent_401s_share_one_run(tmp_path):
    counter, token, accepted = tmp_path / "count", tmp_path / "token", tmp_path / "accepted"
    token.write_text("new")
    accepted.write_text("new")
    command = helper(
        tmp_path / "cmd",
        f"""import time
time.sleep(0.2)
c = Path({str(counter)!r})
c.write_text(str(int(c.read_text()) + 1) if c.exists() else "1")
print("Bearer " + Path({str(token)!r}).read_text())""",
    )
    credential = CommandCredential(spec(command, tmp_path), "Authorization", SecretStr("Bearer old"))
    async with httpx.AsyncClient(auth=credential, transport=gated(accepted, [])) as client:
        responses = await asyncio.gather(*(client.post("http://mcp.test/mcp", content=b"{}") for _ in range(5)))
    assert [response.status_code for response in responses] == [200] * 5
    assert counter.read_text() == "1"


def test_sync_clients_are_refused(tmp_path):
    credential = CommandCredential(spec(Path("/bin/true"), tmp_path), "Authorization", SecretStr("Bearer x"))
    transport = httpx.MockTransport(lambda _: httpx.Response(200))
    with (
        httpx.Client(auth=credential, transport=transport) as client,
        pytest.raises(RuntimeError, match=r"only httpx\.AsyncClient"),
    ):
        client.get("http://mcp.test/")


@asynccontextmanager
async def rotating_server(root: Path) -> AsyncIterator[tuple[str, Path]]:
    """A Streamable HTTP MCP server that accepts only `Bearer <contents of the accepted file>`."""
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    accepted = root / "accepted"
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    script = root / "rotating_server.py"
    script.write_text(f"""import uvicorn
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
mcp=FastMCP("remote",host="127.0.0.1",port={port})
@mcp.tool()
def add(a:int,b:int)->int: return a+b
inner=mcp.streamable_http_app()
async def app(scope,receive,send):
    want = ("Bearer " + Path({str(accepted)!r}).read_text()).encode()
    if scope["type"]=="http" and (b"authorization",want) not in scope["headers"]:
        return await JSONResponse({{"error":"unauthorized"}},status_code=401)(scope,receive,send)
    await inner(scope,receive,send)
uvicorn.run(app,host="127.0.0.1",port={port},log_level="critical")
""")
    accepted.write_text("unset")
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(script), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        async with httpx.AsyncClient(timeout=1) as client, asyncio.timeout(8):
            while True:
                try:
                    assert (await client.get(url)).status_code == 401
                    break
                except httpx.ConnectError:
                    assert process.returncode is None, "MCP test server exited during startup"
                    await asyncio.sleep(0.02)
        yield url, accepted
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await process.wait()


def credential_toml(url: str, command: Path, timeout_s: float = 30) -> str:
    return f"""[[capabilities.mcp]]
namespace = "remote"
transport = "http"
url = "{url}"
timeout_s = {timeout_s}
remote_tools = {{ add = [] }}

[capabilities.mcp.credential_command]
command = {json.dumps(str(command))}
env_refs = {{ CLIENT_SECRET = "SYNTHETIC_CLIENT_SECRET" }}
"""


async def test_session_keeps_working_after_the_token_rotates(tmp_path):
    counter, token = tmp_path / "count", tmp_path / "token"
    # The helper reads its token file, unless a .env sits in its working directory.
    command = helper(
        tmp_path / "token-cmd",
        f"""c = Path({str(counter)!r})
c.write_text(str(int(c.read_text()) + 1) if c.exists() else "1")
assert os.environ["CLIENT_SECRET"] == "synthetic-client-secret"
print("Bearer planted" if Path(".env").exists() else "Bearer " + Path({str(token)!r}).read_text())""",
    )
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / ".env").write_text("planted by the model")
    async with rotating_server(tmp_path) as (url, accepted):
        token.write_text("synthetic-token-one")
        accepted.write_text("synthetic-token-one")
        replies = [
            Reply(tool_calls=[("mcp__remote__add", {"a": 2, "b": 5})]),
            Reply(text="done"),
            Reply(tool_calls=[("mcp__remote__add", {"a": 30, "b": 12})]),
            Reply(text="done"),
        ]
        env = {"SYNTHETIC_CLIENT_SECRET": "synthetic-client-secret"}
        async with configured(tmp_path, replies, credential_toml(url, command), env=env) as values:
            session, model, _, _ = values
            await session.run("add")
            assert "7" in outputs(session)[0]
            assert counter.read_text() == "1"  # only the setup run; the planted .env was not seen

            token.write_text("synthetic-token-two")
            accepted.write_text("synthetic-token-two")
            await session.run("add again")
            assert "42" in outputs(session)[1]
            assert counter.read_text() == "2"  # one run after the 401, then the retry succeeded

            recorded = json.dumps(model.requests) + "".join(entry.model_dump_json() for entry in session.entries())
            for secret in ("synthetic-token-one", "synthetic-token-two", "synthetic-client-secret"):
                assert secret not in recorded


async def test_a_refresh_failure_mid_session_fails_the_call_promptly(tmp_path):
    token, broken = tmp_path / "token", tmp_path / "broken"
    command = helper(
        tmp_path / "token-cmd",
        f"""if Path({str(broken)!r}).exists():
    sys.exit(4)
print("Bearer " + Path({str(token)!r}).read_text())""",
    )
    async with rotating_server(tmp_path) as (url, accepted):
        token.write_text("synthetic-token-one")
        accepted.write_text("synthetic-token-one")
        replies = [Reply(tool_calls=[("mcp__remote__add", {"a": 2, "b": 5})]), Reply(text="done")]
        env = {"SYNTHETIC_CLIENT_SECRET": "synthetic-client-secret"}
        # The MCP SDK runs each request in its transport's task group, so an error raised while sending ends the
        # transport and is reported again when the session closes. A stale static header does the same on main.
        with pytest.raises(ResourceCleanupError):
            async with configured(tmp_path, replies, credential_toml(url, command), env=env) as values:
                session = values[0]
                accepted.write_text("synthetic-token-two")
                broken.write_text("yes")
                started = time.monotonic()
                await session.run("add")
                elapsed = time.monotonic() - started
                assert "MCP" in outputs(session)[0] and outputs(session)[0].startswith("Error")
                assert elapsed < 10, f"refresh failure took {elapsed:.1f}s to reach the tool call"


async def test_failing_command_stops_setup_before_any_server_starts(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    script, marker = tmp_path / "server.py", tmp_path / "pid"
    stdio_script(script, marker)
    command = helper(
        tmp_path / "token-cmd",
        'sys.stderr.write("boom: rejected " + os.environ["CLIENT_SECRET"]); sys.exit(2)',
    )
    toml = mcp_toml(script, "first") + credential_toml("http://127.0.0.1:1/mcp", command)
    env = {"SYNTHETIC_CLIENT_SECRET": "synthetic-client-secret"}
    with pytest.raises(ConfigError) as caught:
        async with configured(tmp_path, [], toml, env=env):
            pytest.fail("a failing credential command must stop setup")
    message = str(caught.value)
    assert "MCP remote credential command failed (exit 2)" in message
    assert "boom: rejected [REDACTED]" in message
    assert "synthetic-client-secret" not in message
    assert not marker.exists()


async def test_missing_credential_env_ref_is_reported_before_any_server_starts(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    script, marker = tmp_path / "server.py", tmp_path / "pid"
    stdio_script(script, marker)
    command = helper(tmp_path / "token-cmd", 'print("Bearer x")')
    toml = mcp_toml(script, "first") + credential_toml("http://127.0.0.1:1/mcp", command)
    with pytest.raises(ConfigError, match="needs the environment variable SYNTHETIC_CLIENT_SECRET"):
        async with configured(tmp_path, [], toml):
            pytest.fail("a missing credential reference must stop setup")
    assert not marker.exists()
