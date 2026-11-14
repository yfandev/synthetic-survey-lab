"""MCP OAuth: sign in once with `intern mcp login`, then sessions use and refresh the saved tokens."""

from __future__ import annotations

import asyncio
import json
import socket
import stat
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath

import httpx
import pytest

from intern.capability_settings import McpConfig
from intern.guard import Scope, rule_hits
from intern.mcp_commands import mcp_main
from intern.mcp_oauth import OAuthStore, login, store_path
from intern.models.registry import ConfigError

from .fake_openai import Reply
from .test_session_setup import configured, mcp_toml, outputs, stdio_script


def free_port() -> int:
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        return reserved.getsockname()[1]


@asynccontextmanager
async def oauth_server(root: Path) -> AsyncIterator[tuple[str, Path]]:
    """An MCP server behind its own OAuth authorization server, all on one local port.

    `state.json` holds the tokens it accepts, how long new ones live, whether a refresh succeeds and which grants
    were used. Authorization redirects straight back with a code, as a browser would after the person signs in.
    """
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    port = free_port()
    state = root / "state.json"
    state.write_text(json.dumps({"accepted": [], "expires_in": 3600, "refresh_ok": True, "grants": [], "count": 0}))
    script = root / "oauth_server.py"
    script.write_text(f"""import json, uvicorn
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.responses import JSONResponse, RedirectResponse
from starlette.routing import Route
STATE = Path({str(state)!r})
BASE = "http://127.0.0.1:{port}"
def load(): return json.loads(STATE.read_text())
def save(s): STATE.write_text(json.dumps(s))
mcp = FastMCP("oauth-test", host="127.0.0.1", port={port})
@mcp.tool()
def add(a: int, b: int) -> int: return a + b
inner = mcp.streamable_http_app()
async def prm(request): return JSONResponse({{"resource": BASE + "/mcp", "authorization_servers": [BASE]}})
async def asm(request): return JSONResponse({{
    "issuer": BASE, "authorization_endpoint": BASE + "/authorize", "token_endpoint": BASE + "/token",
    "registration_endpoint": BASE + "/register", "response_types_supported": ["code"],
    "grant_types_supported": ["authorization_code", "refresh_token"], "code_challenge_methods_supported": ["S256"],
    "token_endpoint_auth_methods_supported": ["none", "client_secret_post"]}})
async def register(request):
    return JSONResponse({{**(await request.json()), "client_id": "registered-client"}}, status_code=201)
async def authorize(request):
    q = request.query_params
    return RedirectResponse(q["redirect_uri"] + "?code=the-code&state=" + q["state"], status_code=302)
async def token(request):
    form = await request.form(); s = load(); s["count"] += 1; s["grants"].append(form["grant_type"])
    if form["grant_type"] == "refresh_token" and not s["refresh_ok"]:
        save(s); return JSONResponse({{"error": "invalid_grant"}}, status_code=400)
    access = "access-" + str(s["count"]); s["accepted"] = [access]; save(s)
    return JSONResponse({{"access_token": access, "token_type": "Bearer", "expires_in": s["expires_in"],
                         "refresh_token": "refresh-" + str(s["count"])}})
oauth = Starlette(routes=[
    Route("/.well-known/oauth-protected-resource", prm), Route("/.well-known/oauth-protected-resource/mcp", prm),
    Route("/.well-known/oauth-authorization-server", asm), Route("/register", register, methods=["POST"]),
    Route("/authorize", authorize), Route("/token", token, methods=["POST"])])
async def app(scope, receive, send):
    if scope["type"] == "http" and scope["path"].startswith(("/.well-known", "/register", "/authorize", "/token")):
        return await oauth(scope, receive, send)
    if scope["type"] == "http":
        given = dict(scope["headers"]).get(b"authorization", b"").decode().removeprefix("Bearer ")
        if given not in load()["accepted"]:
            meta = 'Bearer resource_metadata="' + BASE + '/.well-known/oauth-protected-resource"'
            response = JSONResponse({{"error": "unauthorized"}}, status_code=401, headers={{"WWW-Authenticate": meta}})
            return await response(scope, receive, send)
    await inner(scope, receive, send)
uvicorn.run(app, host="127.0.0.1", port={port}, log_level="critical")
""")
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
                    assert process.returncode is None, "OAuth test server exited during startup"
                    await asyncio.sleep(0.02)
        yield url, state
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await process.wait()


def entry(url: str, callback_port: int) -> McpConfig:
    return McpConfig.model_validate(
        {
            "namespace": "remote",
            "transport": "http",
            "url": url,
            "remote_tools": {"add": []},
            "oauth": {"callback_port": callback_port},
        }
    )


def oauth_toml(url: str, callback_port: int) -> str:
    return f"""[[capabilities.mcp]]
namespace = "remote"
transport = "http"
url = "{url}"
remote_tools = {{ add = [] }}

[capabilities.mcp.oauth]
callback_port = {callback_port}
"""


def browser(opened: list[str]) -> Callable[[str], Awaitable[None]]:
    """Stands in for the person: follows the authorization URL back to the local callback."""

    async def open_url(url: str) -> None:
        opened.append(url)
        async with httpx.AsyncClient(timeout=5) as client:
            back = (await client.get(url)).headers["location"]
            assert (await client.get(back)).status_code == 200

    return open_url


def update(state: Path, **changes: object) -> None:
    state.write_text(json.dumps({**json.loads(state.read_text()), **changes}))


async def test_login_saves_private_tokens_and_a_session_uses_them(tmp_path):
    home = tmp_path / "home"
    async with oauth_server(tmp_path) as (url, state):
        port = free_port()
        opened: list[str] = []
        saved = await login(entry(url, port), home, {}, None, open_url=browser(opened))
        assert saved == store_path(home, "remote")
        assert len(opened) == 1
        assert f"redirect_uri=http%3A%2F%2F127.0.0.1%3A{port}%2Fcallback" in opened[0]
        assert "code_challenge_method=S256" in opened[0]
        assert stat.S_IMODE(saved.stat().st_mode) == 0o600
        assert stat.S_IMODE(saved.parent.stat().st_mode) == 0o700
        assert "access-1" in saved.read_text()

        replies = [Reply(tool_calls=[("mcp__remote__add", {"a": 2, "b": 5})]), Reply(text="done")]
        async with configured(tmp_path, replies, oauth_toml(url, port)) as (session, model, _, _):
            await session.run("add")
            assert "7" in outputs(session)[0]
            recorded = json.dumps(model.requests) + "".join(item.model_dump_json() for item in session.entries())
            assert "access-1" not in recorded and "refresh-1" not in recorded
        assert json.loads(state.read_text())["grants"] == ["authorization_code"]


async def test_an_expired_token_is_refreshed_without_signing_in_again(tmp_path):
    home = tmp_path / "home"
    async with oauth_server(tmp_path) as (url, state):
        port = free_port()
        update(state, expires_in=1)
        await login(entry(url, port), home, {}, None, open_url=browser([]))
        await asyncio.sleep(1.5)
        update(state, expires_in=3600)
        replies = [Reply(tool_calls=[("mcp__remote__add", {"a": 20, "b": 22})]), Reply(text="done")]
        async with configured(tmp_path, replies, oauth_toml(url, port)) as (session, _, _, _):
            await session.run("add")
            assert "42" in outputs(session)[0]
        assert json.loads(state.read_text())["grants"] == ["authorization_code", "refresh_token"]
        assert "access-2" in store_path(home, "remote").read_text()


async def test_a_session_never_starts_a_sign_in(tmp_path):
    async with oauth_server(tmp_path) as (url, state):
        port = free_port()
        await login(entry(url, port), tmp_path / "home", {}, None, open_url=browser([]))
        update(state, accepted=[], refresh_ok=False)  # the server revoked every token
        with pytest.raises(ConfigError, match="MCP remote setup failed"):
            async with configured(tmp_path, [], oauth_toml(url, port)):
                pytest.fail("a revoked sign-in must stop setup")
        assert json.loads(state.read_text())["grants"] == ["authorization_code"]


async def test_missing_sign_in_names_the_login_command_before_any_server_starts(tmp_path):
    pytest.importorskip("mcp")
    script, marker = tmp_path / "server.py", tmp_path / "pid"
    stdio_script(script, marker)
    toml = mcp_toml(script, "first") + oauth_toml("http://127.0.0.1:1/mcp", free_port())
    with pytest.raises(ConfigError, match=r"MCP remote needs sign-in: run `intern mcp login remote`"):
        async with configured(tmp_path, [], toml):
            pytest.fail("a missing sign-in must stop setup")
    assert not marker.exists()


async def test_saved_tokens_belong_to_one_server_url(tmp_path):
    path = tmp_path / "remote.json"
    pytest.importorskip("mcp")
    from mcp.shared.auth import OAuthToken

    await OAuthStore(path, "https://one.example/mcp").set_tokens(OAuthToken(access_token="synthetic", expires_in=60))
    assert OAuthStore(path, "https://one.example/mcp").signed_in()
    assert not OAuthStore(path, "https://two.example/mcp").signed_in()
    restored = await OAuthStore(path, "https://one.example/mcp").get_tokens()
    assert restored is not None and restored.expires_in is not None and 0 < restored.expires_in <= 60
    assert "synthetic" not in repr(OAuthStore(path, "https://one.example/mcp"))


def test_the_guard_keeps_the_agent_out_of_the_token_store():
    home = PurePosixPath("/Users/someone")
    scope = Scope(cwd=home / "project", roots=(home / "project",), home=home)
    tokens = "~/.intern/mcp-oauth/remote.json"
    assert [hit.rule for hit in rule_hits("read", {"path": tokens}, scope)] == ["credential_read"]
    assert [hit.rule for hit in rule_hits("bash", {"command": f"cat {tokens}"}, scope)] == ["credential_read"]
    assert [hit.rule for hit in rule_hits("write", {"path": tokens, "content": "{}"}, scope)] == ["protected_write"]


def write_config(home: Path, toml: str) -> dict[str, str]:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(toml)
    return {"INTERN_HOME": str(home)}


def test_login_command_needs_a_configured_oauth_entry(tmp_path, capsys):
    env = write_config(tmp_path, oauth_toml("https://mcp.example/mcp", 40000))
    assert mcp_main(["login", "missing"], env) == 2
    assert "No [[capabilities.mcp]] entry has namespace 'missing'" in capsys.readouterr().err
    plain = write_config(tmp_path / "plain", mcp_toml(Path("/server.py"), "local"))
    assert mcp_main(["login", "local"], plain) == 2
    assert "does not sign in" in capsys.readouterr().err


def test_logout_deletes_the_saved_tokens(tmp_path, capsys):
    env = write_config(tmp_path, oauth_toml("https://mcp.example/mcp", 40000))
    path = store_path(tmp_path, "remote")
    path.parent.mkdir()
    path.write_text(json.dumps({"server_url": "https://mcp.example/mcp", "tokens": {"access_token": "x"}}))
    assert mcp_main(["logout", "remote"], env) == 0
    assert not path.exists()
    assert "Signed out of MCP remote" in capsys.readouterr().err


def test_cli_login_signs_in_end_to_end(tmp_path, capsys):
    async def serve_and_login() -> int:
        async with oauth_server(tmp_path) as (url, _):
            env = write_config(tmp_path / "home", oauth_toml(url, free_port()))
            started = time.monotonic()
            code = await asyncio.to_thread(mcp_main, ["login", "remote"], env, browser([]))
            assert time.monotonic() - started < 30
            return code

    assert asyncio.run(serve_and_login()) == 0
    assert "Signed in to MCP remote; tokens saved in" in capsys.readouterr().err
    assert store_path(tmp_path / "home", "remote").is_file()
