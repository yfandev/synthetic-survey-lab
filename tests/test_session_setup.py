"""Saved capabilities exercised through Bootstrap, real transports and the agent loop."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import socket
import stat
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

import intern.session_setup as capability_setup
from intern.browser import BrowserSession
from intern.capability_settings import CapabilitySettings
from intern.capability_toggles import MCP_PREFIX, toggles
from intern.config import Bootstrap, Options
from intern.extensions import REGISTRY
from intern.hooks import Extension
from intern.journal import MessageEntry
from intern.messages import ToolResultMessage
from intern.models.registry import ConfigError
from intern.processes import ProcessManager
from intern.session import Session
from intern.tools.base import Tool
from intern.tools.files import ReadTool
from intern.tools.process import ProcessTool
from intern.tools.web_read import WebReadTool
from intern.tools.web_search import WebSearchTool
from intern.web.reader import ReadSettings, WebReader
from intern.web.search import HTTPSearchProvider, SearchSettings

from .fake_openai import MODEL, FakeOpenAI, Reply


async def test_new_persisted_sessions_are_isolated_and_explicit_paths_resume(tmp_path, monkeypatch):
    from datetime import datetime

    import intern.config as config
    from intern.messages import UserMessage

    class FrozenClock:
        @staticmethod
        def now():
            return datetime(2026, 11, 8, 12)

    monkeypatch.setattr(config, "datetime", FrozenClock)
    with FakeOpenAI([]) as model:
        bootstrap = Bootstrap(
            Options(cwd=tmp_path, system2=f"vllm/{MODEL}", context_files=False, extensions=False),
            {"INTERN_HOME": str(tmp_path / "home"), "VLLM_BASE_URL": model.url, **model.system1_env},
        )
        async with await bootstrap.session(lambda event: None) as first:
            first.deps.journal.append(MessageEntry(message=UserMessage(content="private first task")))
            first_path = first.deps.journal.path
            first_id = first.deps.journal.session_id
            async with await bootstrap.session(lambda event: None) as second:
                assert second.deps.journal.path != first_path
                assert second.deps.journal.session_id != first_id
                assert not any(isinstance(entry, MessageEntry) for entry in second.entries())
        assert first_path is not None
        resumed = Bootstrap(
            Options(
                cwd=tmp_path,
                system2=f"vllm/{MODEL}",
                session_file=first_path,
                context_files=False,
                extensions=False,
            ),
            {"INTERN_HOME": str(tmp_path / "home"), "VLLM_BASE_URL": model.url, **model.system1_env},
        )
        async with await resumed.session(lambda event: None) as session:
            assert session.deps.journal.session_id == first_id
            assert "private first task" in "".join(entry.model_dump_json() for entry in session.entries())


@asynccontextmanager
async def configured(
    root: Path,
    replies: list[Reply],
    toml: str,
    *,
    env: dict[str, str] | None = None,
    extensions: list[str] | None = None,
    profile: Path | None = None,
) -> AsyncIterator[tuple[Session, FakeOpenAI, Path, Path]]:
    home, work = root / "home", root / "work"
    home.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    (home / "config.toml").write_text(f'system2 = "vllm/{MODEL}"\n{toml}')
    with FakeOpenAI(replies) as model:
        bootstrap = Bootstrap(
            Options(
                cwd=work,
                persist=False,
                context_files=False,
                extensions=bool(extensions),
                extra_extensions=extensions or [],
                profile=profile,
                memory_off=True,
            ),
            {
                "INTERN_HOME": str(home),
                "VLLM_BASE_URL": model.url,
                **model.system1_env,
                "PATH": os.environ.get("PATH", ""),
                **(env or {}),
            },
        )
        async with await bootstrap.session(lambda event: None) as session:
            yield session, model, home, work


def outputs(session: Session) -> list[str]:
    return [
        entry.message.result.content
        for entry in session.entries()
        if isinstance(entry, MessageEntry) and isinstance(entry.message, ToolResultMessage)
    ]


@pytest.fixture
async def website():
    hits: list[str] = []
    port = 0

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            path = header.split(b" ")[1].decode()
            hits.append(path)
            if path.startswith("/search"):
                body = json.dumps(
                    {
                        "results": [
                            {"title": "Fixture", "url": f"http://127.0.0.1:{port}/page", "content": "demo evidence"}
                        ]
                    }
                ).encode()
                mime = "application/json"
            elif path == "/long":
                body = (
                    "<html><title>Long fixture</title><body>"
                    + "".join(
                        f"<p>Evidence line {i}: a useful detail for the retained source.</p>" for i in range(2000)
                    )
                    + "<p>TAIL_MARKER</p></body></html>"
                ).encode()
                mime = "text/html"
            else:
                body = b"""<html><title>Fixture</title><body><h1>demo evidence</h1>
<input id="name"><button id="go" onclick="document.querySelector('#result').textContent=
document.querySelector('#name').value">Go</button><p id="result"></p>
<script>document.cookie='fixture=value';</script></body></html>"""
                mime = "text/html"
            writer.write(
                (
                    f"HTTP/1.1 200 OK\r\nContent-Type:{mime}\r\nContent-Length:{len(body)}\r\nConnection:close\r\n\r\n"
                ).encode()
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", hits
    finally:
        server.close()
        await server.wait_closed()


async def test_saved_search_and_reader_reach_model_and_retain_private_sources(tmp_path, website):
    url, hits = website
    replies = [
        Reply(tool_calls=[("web_search", {"query": "evidence"}), ("web_read", {"url": f"{url}/page"})]),
        Reply(text="done"),
    ]
    toml = f"""[capabilities]
tool_packs = ["base", "web-search", "web-read"]
[capabilities.search]
backend = "searxng"
searxng_url = "{url}"
[capabilities.reader]
allow_private = true
storage_dir = "retained"
"""
    async with configured(tmp_path, replies, toml) as (session, model, _, work):
        assert (await session.run("find and read the source")).final_text == "done"
        assert any(path.startswith("/search?") for path in hits) and "/page" in hits
        assert "demo evidence" in json.dumps(model.requests[-1]["messages"])
        sources = list((work / "retained").glob("*/*.txt"))
        assert len(sources) == 1 and "demo evidence" in sources[0].read_text()
        assert stat.S_IMODE(sources[0].stat().st_mode) == 0o600
        assert stat.S_IMODE(sources[0].parent.stat().st_mode) == 0o700
    assert sources[0].is_file()  # Closing owned services does not delete retained evidence.


async def test_reader_keeps_default_private_address_denial(tmp_path, website):
    url, hits = website
    replies = [Reply(tool_calls=[("web_read", {"url": f"{url}/page"})]), Reply(text="done")]
    async with configured(tmp_path, replies, '[capabilities]\ntool_packs = ["web-read"]\n') as values:
        session, _, home, _ = values
        await session.run("read")
        assert "private" in outputs(session)[0].lower()
        assert not hits and not (home / "session-resources").exists()


def stdio_script(path: Path, marker: Path) -> None:
    path.write_text(f"""import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
Path({str(marker)!r}).write_text(str(os.getpid()))
mcp=FastMCP("configured-test")
@mcp.tool()
def add(a:int,b:int)->int: return a+b
mcp.run(transport="stdio")
""")


def mcp_toml(script: Path, namespace: str, tool: str = "add") -> str:
    return f"""[[capabilities.mcp]]
namespace = "{namespace}"
transport = "stdio"
command = {json.dumps(sys.executable)}
args = [{json.dumps(str(script))}]
remote_tools = {{ {tool} = [] }}
"""


def assert_reaped(marker: Path) -> None:
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


async def test_saved_stdio_mcp_is_ready_for_agent_and_closes_from_another_task(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    script, marker = tmp_path / "server.py", tmp_path / "pid"
    stdio_script(script, marker)
    replies = [Reply(tool_calls=[("mcp__math__add", {"a": 4, "b": 5})]), Reply(text="done")]
    async with configured(tmp_path, replies, mcp_toml(script, "math")) as (session, model, _, _):
        assert marker.is_file()
        assert (await session.run("add numbers")).reason == "completed"
        assert "9" in outputs(session)[0]
        assert "9" in json.dumps(model.requests[-1]["messages"])
        await asyncio.create_task(session.aclose())
        assert session.closed
    assert_reaped(marker)


async def test_failed_second_mcp_rolls_back_both_owned_processes_before_return(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    scripts = [tmp_path / f"server-{i}.py" for i in range(2)]
    markers = [tmp_path / f"pid-{i}" for i in range(2)]
    for script, marker in zip(scripts, markers, strict=True):
        stdio_script(script, marker)
    with pytest.raises(ConfigError, match="MCP second setup failed") as error:
        async with configured(tmp_path, [], mcp_toml(scripts[0], "first") + mcp_toml(scripts[1], "second", "absent")):
            pytest.fail("missing selected remote tool must fail startup")
    assert "Traceback" not in str(error.value)
    for marker in markers:
        assert marker.is_file()
        assert_reaped(marker)


async def test_missing_mcp_auth_is_validated_before_any_server_starts(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    script, marker = tmp_path / "server.py", tmp_path / "pid"
    stdio_script(script, marker)
    toml = (
        mcp_toml(script, "first")
        + """[[capabilities.mcp]]
namespace = "second"
transport = "http"
url = "http://127.0.0.1:1/mcp"
remote_tools = { add = [] }
header_env = { Authorization = "ABSENT_SYNTHETIC_AUTH" }
"""
    )
    with pytest.raises(ConfigError, match="needs the environment variable"):
        async with configured(tmp_path, [], toml):
            pytest.fail("missing credential must fail startup")
    assert not marker.exists()


async def test_saved_http_mcp_maps_auth_reference_and_does_not_stop_remote_server(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("jsonschema")
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    script = tmp_path / "mcp_http_server.py"
    script.write_text(f"""import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
mcp=FastMCP("remote",host="127.0.0.1",port={port})
@mcp.tool()
def add(a:int,b:int)->int: return a+b
inner=mcp.streamable_http_app()
async def app(scope,receive,send):
    if scope["type"]=="http" and (b"authorization",b"Bearer synthetic-mcp-token") not in scope["headers"]:
        return await JSONResponse({{"error":"unauthorized"}},status_code=401)(scope,receive,send)
    await inner(scope,receive,send)
uvicorn.run(app,host="127.0.0.1",port={port},log_level="critical")
""")
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(script), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    url = f"http://127.0.0.1:{port}/mcp"
    toml = f"""[[capabilities.mcp]]
namespace = "remote"
transport = "http"
url = "{url}"
remote_tools = {{ add = [] }}
header_env = {{ Authorization = "FIXTURE_MCP_AUTH" }}
"""
    try:
        async with httpx.AsyncClient(timeout=1) as client:
            async with asyncio.timeout(8):
                while True:
                    try:
                        assert (await client.get(url)).status_code == 401
                        break
                    except httpx.ConnectError:
                        assert process.returncode is None, "MCP test server exited during startup"
                        await asyncio.sleep(0.02)
            replies = [Reply(tool_calls=[("mcp__remote__add", {"a": 2, "b": 5})]), Reply(text="done")]
            async with configured(
                tmp_path, replies, toml, env={"FIXTURE_MCP_AUTH": "Bearer synthetic-mcp-token"}
            ) as values:
                session, model, _, _ = values
                await session.run("add numbers")
                assert "7" in outputs(session)[0]
                assert "synthetic-mcp-token" not in json.dumps(model.requests)
            assert process.returncode is None
            assert (await client.get(url)).status_code == 401
    finally:
        if process.returncode is None:
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except TimeoutError:
            process.kill()
            await process.wait()


@pytest.mark.skipif(not os.environ.get("INTERN_BROWSER_TEST"), reason="explicit prepared Chromium required")
async def test_saved_browser_is_lazy_usable_and_isolated_with_owned_cleanup(tmp_path, website, monkeypatch):
    url, _ = website
    browsers: list[BrowserSession] = []

    def capture(*args, **kwargs):
        browser = BrowserSession(*args, **kwargs)
        browsers.append(browser)
        return browser

    monkeypatch.setattr(capability_setup, "BrowserSession", capture)
    toml = """[capabilities]
tool_packs = ["base"]
[capabilities.browser]
enabled = true
allow_private_urls = true
artifact_dir = "screenshots"
"""
    replies = [
        Reply(tool_calls=[("browser", {"action": "navigate", "url": f"{url}/page"})]),
        Reply(tool_calls=[("browser", {"action": "fill", "selector": "#name", "text": "Ada"})]),
        Reply(tool_calls=[("browser", {"action": "click", "selector": "#go"})]),
        Reply(tool_calls=[("browser", {"action": "screenshot"})]),
        Reply(text="done"),
    ]
    async with configured(tmp_path, replies, toml) as (session, model, _, work):
        assert len(browsers) == 1 and browsers[0]._browser is None
        await session.run("use the form and capture the result")
        assert "Ada" in outputs(session)[2]
        assert "Ada" in json.dumps(model.requests[-1]["messages"])
        images = list((work / "screenshots").glob("*/*.png"))
        assert len(images) == 1 and images[0].read_bytes().startswith(b"\x89PNG")
        assert stat.S_IMODE(images[0].stat().st_mode) == 0o600
        assert await browsers[0]._context.cookies()
        async with configured(tmp_path / "other", [], toml) as (other, _, _, _):
            assert browsers[1]._browser is None
            await browsers[1].start()
            assert await browsers[1]._context.cookies() == []
            await session.aclose()
            assert browsers[0]._closed and browsers[0].proxy._server is None
            assert not browsers[1]._closed and not other.closed
    assert images[0].is_file()
    assert all(browser._closed and browser.proxy._server is None for browser in browsers)


async def test_saved_process_manager_survives_messages_but_not_session_close(tmp_path):
    marker = tmp_path / "command-pid"
    command = f"echo $$ > {shlex.quote(str(marker))}; printf ready; sleep 30"
    replies = [
        Reply(tool_calls=[("process", {"action": "start", "command": command})]),
        Reply(text="started"),
        Reply(tool_calls=[("process", {"action": "list"})]),
        Reply(text="listed"),
    ]
    toml = '[capabilities]\ntool_packs = ["base", "process"]\n'
    async with configured(tmp_path, replies, toml) as (session, _, _, _):
        assert not marker.exists()
        await session.run("start the background task")
        async with asyncio.timeout(3):
            for _ in range(300):
                if marker.is_file():
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("managed command never published its PID")
        handle = json.loads(outputs(session)[0])["handle"]
        os.kill(int(marker.read_text()), 0)
        await session.run("list the task")
        assert handle in outputs(session)[1]
        async with configured(tmp_path / "peer", [], toml) as (peer, _, _, _):
            peer_tool = next(tool for tool in peer.deps.tools if isinstance(tool, ProcessTool))
            assert peer_tool.manager.list() == []
            await session.aclose()
            assert not peer.closed
    assert_reaped(marker)


@pytest.mark.parametrize("docker", [False, True], ids=["local", "prepared-docker"])
async def test_saved_secrets_and_executor_compose_without_exposing_value_to_model(tmp_path, docker):
    image = os.environ.get("INTERN_DOCKER_IMAGE")
    if docker and not image:
        pytest.skip("explicit prepared local Docker image required")
    toml = """[capabilities]
tool_packs = ["base"]
[capabilities.secrets.bindings.demo]
source_env = "FIXTURE_COMMAND_KEY"
scopes = ["executor"]
[capabilities.secrets.inject]
DEMO_TOKEN = "demo"
"""
    if docker:
        toml += f"""[capabilities.runner]
name = "docker"
[capabilities.runner.options]
image = {json.dumps(image)}
"""
    replies = [
        Reply(tool_calls=[("write", {"path": "evidence.txt", "content": "workspace evidence"})]),
        Reply(tool_calls=[("bash", {"command": 'cat evidence.txt; printf " %s" "$DEMO_TOKEN"'})]),
        Reply(text="done"),
    ]
    async with configured(
        tmp_path, replies, toml, env={"FIXTURE_COMMAND_KEY": "synthetic-command-credential"}
    ) as values:
        session, model, _, work = values
        await session.run("write and read evidence")
        assert (work / "evidence.txt").read_text() == "workspace evidence"
        assert "workspace evidence" in outputs(session)[1] and "[REDACTED]" in outputs(session)[1]
        assert "synthetic-command-credential" not in json.dumps(model.requests)
        assert "synthetic-command-credential" not in "".join(entry.model_dump_json() for entry in session.entries())


async def test_process_jobs_get_injected_secrets_with_a_redacted_log(tmp_path):
    toml = """[capabilities]
tool_packs = ["base", "process"]
[capabilities.secrets.bindings.demo]
source_env = "FIXTURE_JOB_KEY"
scopes = ["executor"]
[capabilities.secrets.inject]
DEMO_TOKEN = "demo"
"""
    marker = tmp_path / "job-saw"
    command = f'printf "%s" "$DEMO_TOKEN" > {shlex.quote(str(marker))}; printf "token=%s;" "$DEMO_TOKEN"'
    replies = [Reply(tool_calls=[("process", {"action": "start", "command": command})]), Reply(text="started")]
    async with configured(tmp_path, replies, toml, env={"FIXTURE_JOB_KEY": "job-credential"}) as values:
        session, model, _, _ = values
        await session.run("start it")
        handle = json.loads(outputs(session)[0])["handle"]
        tool = next(tool for tool in session.deps.tools if isinstance(tool, ProcessTool))
        snapshot = await tool.manager.poll(handle, wait_s=5)
        async with asyncio.timeout(5):
            while snapshot.state == "running":
                snapshot = await tool.manager.poll(handle, wait_s=1)
        assert marker.read_text() == "job-credential"
        assert snapshot.output == "token=[REDACTED];"
        assert "job-credential" not in json.dumps(model.requests)


async def test_missing_command_secret_fails_without_running_command(tmp_path):
    toml = """[capabilities.secrets.bindings.demo]
source_env = "MISSING_FIXTURE_COMMAND_KEY"
scopes = ["executor"]
[capabilities.secrets.inject]
DEMO_TOKEN = "demo"
"""
    with pytest.raises(ConfigError, match="needs the environment variable") as error:
        async with configured(tmp_path, [Reply(tool_calls=[("bash", {"command": "touch should-not-exist"})])], toml):
            pytest.fail("missing credential must fail before run")
    assert not (tmp_path / "work" / "should-not-exist").exists()
    assert "Traceback" not in str(error.value)


async def test_command_secret_reaches_commands_redacted_and_failure_stops_setup(tmp_path):
    vault = tmp_path / "vault-cli"
    vault.write_text('#!/bin/sh\nprintf "%s-from-vault" "$VAULT_TOKEN"\n')
    vault.chmod(0o755)
    toml = f"""[capabilities]
tool_packs = ["base"]
[capabilities.secrets.bindings.demo]
command = {json.dumps(str(vault))}
env_refs = {{ VAULT_TOKEN = "FIXTURE_VAULT_TOKEN" }}
scopes = ["executor"]
[capabilities.secrets.inject]
DEMO_TOKEN = "demo"
"""
    replies = [Reply(tool_calls=[("bash", {"command": 'printf "%s" "$DEMO_TOKEN"'})]), Reply(text="done")]
    async with configured(tmp_path, replies, toml, env={"FIXTURE_VAULT_TOKEN": "unlock"}) as values:
        session, model, _, _ = values
        await session.run("use the secret")
        assert outputs(session)[0].startswith("[REDACTED]")
        assert "unlock-from-vault" not in json.dumps(model.requests)
    vault.write_text('#!/bin/sh\necho "vault sealed for $VAULT_TOKEN" >&2; exit 2\n')
    with pytest.raises(ConfigError, match="Secret demo command failed \\(exit 2\\)") as error:
        async with configured(tmp_path / "again", [], toml, env={"FIXTURE_VAULT_TOKEN": "unlock"}):
            pytest.fail("a failing secret command must stop setup")
    assert "unlock" not in str(error.value) and "vault sealed" in str(error.value)


@pytest.mark.skipif(not os.environ.get("INTERN_DOCKER_IMAGE"), reason="explicit prepared local Docker image required")
async def test_docker_reader_continuation_path_is_readable_inside_workspace(tmp_path, website):
    url, _ = website
    toml = f"""[capabilities]
tool_packs = ["base", "web-read"]
[capabilities.runner]
name = "docker"
[capabilities.runner.options]
image = {json.dumps(os.environ["INTERN_DOCKER_IMAGE"])}
[capabilities.reader]
allow_private = true
"""
    replies = [Reply(tool_calls=[("web_read", {"url": f"{url}/long"})]), Reply(text="retained")]
    async with configured(tmp_path, replies, toml) as (session, model, home, work):
        await session.run("read the large source")
        notice = re.search(r"Full UTF-8 source: (.*?); read", outputs(session)[0])
        assert notice is not None
        display = Path(notice.group(1))
        assert display.is_relative_to("/workspace")
        source = work / display.relative_to("/workspace")
        lines = source.read_text().splitlines()
        model.replies.extend(
            [Reply(tool_calls=[("read", {"path": str(display), "offset": max(1, len(lines) - 5)})]), Reply(text="done")]
        )
        await session.run("read the end of the retained source")
        assert "TAIL_MARKER" in outputs(session)[1]
        assert "TAIL_MARKER" in json.dumps(model.requests[-1]["messages"])
        assert not (home / "session-resources").exists()


async def test_docker_reader_refuses_unmounted_storage_before_startup(tmp_path):
    toml = """[capabilities]
tool_packs = ["base", "web-read"]
[capabilities.runner]
name = "docker"
[capabilities.reader]
storage_dir = "../unmounted"
"""
    with pytest.raises(ConfigError, match="inside its mounted workspace"):
        async with configured(tmp_path, [], toml):
            pytest.fail("unreadable retained sources must fail before startup")
    assert not (tmp_path / "unmounted").exists()


async def test_relative_config_home_keeps_default_artifacts_in_that_home(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home, work = tmp_path / "config-home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    (home / "config.toml").write_text('[capabilities]\ntool_packs = ["web-read"]\n')
    with FakeOpenAI([]) as model:
        bootstrap = Bootstrap(
            Options(
                cwd=work,
                system2=f"vllm/{MODEL}",
                persist=False,
                context_files=False,
                extensions=False,
            ),
            {"INTERN_HOME": "config-home", "VLLM_BASE_URL": model.url, **model.system1_env},
        )
        async with await bootstrap.session(lambda event: None) as session:
            reader = next(tool for tool in session.deps.tools if isinstance(tool, capability_setup.WebReadTool))
            assert reader.storage_dir is not None
            assert reader.storage_dir.parent == home / "session-resources" / "web"


@pytest.mark.parametrize("packs", [[], ["base", "web-read"]], ids=["no-pack-tools", "configured-reader"])
async def test_saved_capabilities_preserve_landed_extension_tools_and_replacements(tmp_path, packs):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    (work / "evidence.txt").write_text("quiet evidence")
    (home / "config.toml").write_text(
        f"[capabilities]\ntool_packs = {json.dumps(packs)}\n[capabilities.reader]\ntimeout_s = 11\n"
    )
    replies = [Reply(tool_calls=[("greet", {"who": "Ada"}), ("read", {"path": "evidence.txt"})]), Reply(text="done")]
    with FakeOpenAI(replies) as model:
        bootstrap = Bootstrap(
            Options(
                cwd=work,
                system2=f"vllm/{MODEL}",
                persist=False,
                context_files=False,
                extra_extensions=["tests.support:WithTools"],
            ),
            {"INTERN_HOME": str(home), "VLLM_BASE_URL": model.url, **model.system1_env},
        )
        async with await bootstrap.session(lambda event: None) as session:
            readers = [tool for tool in session.deps.tools if isinstance(tool, capability_setup.WebReadTool)]
            assert bool(readers) == bool(packs)
            if readers:
                assert readers[0].reader.settings.timeout_s == 11
            await session.run("greet and read")
            assert outputs(session) == ["hello Ada", "QUIET EVIDENCE"]
            assert "QUIET EVIDENCE" in json.dumps(model.requests[-1]["messages"])


@pytest.fixture
def configured_stock_extension(tmp_path, website, monkeypatch):
    url, _ = website
    reader = WebReadTool(
        WebReader(ReadSettings(timeout_s=7, allow_private=True)), storage_dir=tmp_path / "extension-sources"
    )
    search = WebSearchTool(HTTPSearchProvider(SearchSettings(backend="searxng", searxng_url=url)))
    process = ProcessTool(ProcessManager(max_running=1, max_records=1))
    calls: list[str] = []

    class ConfiguredStock(Extension):
        def tools(self) -> list[Tool]:
            calls.append("tools")
            return [reader, search, process]

    monkeypatch.setitem(REGISTRY, "configured-stock", ConfiguredStock)
    return reader, search, process, calls


async def test_extension_owned_stock_tools_keep_their_provider_settings_and_manager(
    tmp_path, website, configured_stock_extension
):
    url, hits = website
    reader, search, process, calls = configured_stock_extension
    replies = [
        Reply(
            tool_calls=[
                ("web_search", {"query": "evidence"}),
                ("web_read", {"url": f"{url}/page"}),
                ("process", {"action": "list"}),
            ]
        ),
        Reply(text="done"),
    ]
    try:
        async with configured(
            tmp_path, replies, "[capabilities]\ntool_packs = []\n", extensions=["configured-stock"]
        ) as (session, model, _, _):
            assert session.deps.tools == [reader, search, process]
            await session.run("search, read and list jobs")
            assert "demo evidence" in outputs(session)[0] and "demo evidence" in outputs(session)[1]
            assert outputs(session)[2] == "[]" and calls == ["tools"]
            assert "demo evidence" in json.dumps(model.requests[-1]["messages"])
            assert any(path.startswith("/search?") for path in hits) and "/page" in hits
            assert list((tmp_path / "extension-sources").glob("*.txt"))
    finally:
        # This manager belongs to the extension, not a pack-created service.
        await process.aclose()


@pytest.mark.parametrize("section", ["search", "reader", "process"])
async def test_settings_cannot_silently_reconfigure_extension_owned_stock_tool(
    tmp_path, configured_stock_extension, section
):
    setting = "max_running = 2" if section == "process" else "timeout_s = 11"
    toml = f"[capabilities]\ntool_packs = []\n[capabilities.{section}]\n{setting}\n"
    with pytest.raises(ConfigError, match="conflict with an extension replacement"):
        async with configured(tmp_path, [], toml, extensions=["configured-stock"]):
            pytest.fail("extension-owned service configuration must not be overwritten")


async def test_docker_does_not_replace_extension_owned_exact_stock_file_tool(tmp_path, monkeypatch):
    class OwnRead(Extension):
        def tools(self) -> list[Tool]:
            return [ReadTool()]

    monkeypatch.setitem(REGISTRY, "own-read", OwnRead)
    toml = '[capabilities]\ntool_packs = []\n[capabilities.runner]\nname = "docker"\n'
    with pytest.raises(ConfigError, match="cannot replace an extension's file/shell tools"):
        async with configured(tmp_path, [], toml, extensions=["own-read"]):
            pytest.fail("same class does not grant ownership of an extension's tool")


@pytest.mark.parametrize("attach_images", [True, False])
async def test_host_reader_image_mode_survives_bootstrap_and_preserves_host_views(tmp_path, monkeypatch, attach_images):
    from PIL import Image

    from intern.config import VISION_LINE
    from intern.messages import ImagePart
    from intern.tools.base import ToolOutput

    admitted_path = tmp_path / "admitted.png"
    Image.new("RGB", (10, 10), "white").save(admitted_path)
    admitted = ImagePart(
        path=str(admitted_path), mime="image/png", width=10, height=10, source="host view", sha256="test"
    )

    class ViewImages(ReadTool):
        name = "view_images"

        async def run(self, args, ctx):
            assert ctx.image_dir is not None
            return ToolOutput(content="Host-admitted image", images=[admitted])

    class HostImages(Extension):
        def tools(self) -> list[Tool]:
            return [ReadTool(attach_images=attach_images), ViewImages()]

    monkeypatch.setitem(REGISTRY, "host-images", HostImages)
    profile = tmp_path / "vision.toml"
    profile.write_text("vision = true\n")
    replies = [
        Reply(tool_calls=[("read", {"path": "chart.png"})]),
        Reply(tool_calls=[("view_images", {"path": "chart.png"})]),
        Reply(text="Viewed."),
        Reply(text="Follow-up."),
    ]
    async with configured(
        tmp_path, replies, '[capabilities]\ntool_packs = ["base"]\n', extensions=["host-images"], profile=profile
    ) as (session, model, _, work):
        Image.new("RGB", (10, 10), "black").save(work / "chart.png")
        await session.run("Read the chart, then view it through the host.")
        read_result = next(
            entry.message.result
            for entry in session.entries()
            if isinstance(entry, MessageEntry)
            and isinstance(entry.message, ToolResultMessage)
            and entry.message.result.tool_name == "read"
        )
        assert read_result.status == "ok" and len(read_result.images) == int(attach_images)
        assert json.dumps(model.requests[1]["messages"]).count('"type": "image_url"') == int(attach_images)
        assert json.dumps(model.requests[-1]["messages"]).count('"type": "image_url"') == 1 + int(attach_images)
        # Check which viewing route is advertised without pinning the instruction's wording.
        assert (VISION_LINE in session.deps.instructions.system) == attach_images
        await session.run("Continue without viewing the image again.")
        assert json.dumps(model.requests[-1]["messages"]).count('"type": "image_url"') == 1 + int(attach_images)


async def test_workspace_cannot_silently_restore_disabled_read_image_attachments(tmp_path, monkeypatch):
    class HostImages(Extension):
        def tools(self) -> list[Tool]:
            return [ReadTool(attach_images=False)]

    monkeypatch.setitem(REGISTRY, "host-images", HostImages)
    with pytest.raises(ConfigError, match=r"image attachments were disabled.*capabilities.workspace"):
        async with configured(tmp_path, [], "[capabilities.workspace]\nenabled = true\n", extensions=["host-images"]):
            pytest.fail("the host's image opt-out must not be lost during capability assembly")


async def test_workspace_keeps_default_image_attachment_through_bootstrap(tmp_path):
    from PIL import Image

    profile = tmp_path / "vision.toml"
    profile.write_text("vision = true\n")
    replies = [Reply(tool_calls=[("read", {"path": "chart.png"})]), Reply(text="Viewed.")]
    async with configured(tmp_path, replies, "[capabilities.workspace]\nenabled = true\n", profile=profile) as (
        session,
        model,
        _,
        work,
    ):
        Image.new("RGB", (10, 10), "black").save(work / "chart.png")
        await session.run("Read the chart.")
        assert json.dumps(model.requests[-1]["messages"]).count('"type": "image_url"') == 1


async def test_stored_secret_reaches_commands_redacted_over_the_users_environment(tmp_path, monkeypatch):
    fetch = tmp_path / "fetch"
    fetch.write_text("#!/bin/sh\nprintf stored-token-value\n")
    fetch.chmod(0o755)
    monkeypatch.setenv("SSH_AUTH_SOCK", "/tmp/agent.sock")
    toml = f'[secrets]\nDEMO_TOKEN = [{json.dumps(str(fetch))}]\n[capabilities]\ntool_packs = ["base"]\n'
    replies = [
        Reply(tool_calls=[("bash", {"command": 'printf "%s|%s" "$DEMO_TOKEN" "$SSH_AUTH_SOCK"'})]),
        Reply(text="done"),
    ]
    async with configured(tmp_path, replies, toml) as (session, model, _, _):
        await session.run("use the secret")
        assert outputs(session)[0].startswith("[REDACTED]|/tmp/agent.sock")
        assert "stored-token-value" not in json.dumps(model.requests)
    both = toml + '[capabilities.secrets.bindings.demo]\nsource_env = "X"\nscopes = ["executor"]\n'
    both += '[capabilities.secrets.inject]\nOTHER = "demo"\n'
    with pytest.raises(ConfigError, match=r"\[secrets\] and \[capabilities.secrets\] both"):
        async with configured(tmp_path / "again", [], both, env={"X": "value-of-x"}):
            pytest.fail("one source of command secrets at a time")


async def test_a_stored_key_the_harness_uses_itself_stays_out_of_the_agents_commands(tmp_path, monkeypatch):
    fetch = tmp_path / "fetch"
    fetch.write_text('#!/bin/sh\nprintf "%s-stored-value" "$1"\n')
    fetch.chmod(0o755)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    toml = (
        f'[secrets]\nDEMO_TOKEN = [{json.dumps(str(fetch))}, "demo"]\n'
        f'ANTHROPIC_API_KEY = [{json.dumps(str(fetch))}, "model"]\n[capabilities]\ntool_packs = ["base"]\n'
    )
    command = 'printf "%s|%s" "$DEMO_TOKEN" "${ANTHROPIC_API_KEY-unset}"'
    replies = [Reply(tool_calls=[("bash", {"command": command})]), Reply(text="done")]
    async with configured(tmp_path, replies, toml) as (session, _, _, _):
        await session.run("use the secrets")
        assert outputs(session)[0].startswith("[REDACTED]|unset")


async def test_a_stored_secret_is_redacted_from_every_tool_not_only_the_command_that_got_it(tmp_path):
    fetch = tmp_path / "fetch"
    fetch.write_text("#!/bin/sh\nprintf stored-token-value\n")
    fetch.chmod(0o755)
    toml = f'[secrets]\nDEMO_TOKEN = [{json.dumps(str(fetch))}]\n[capabilities]\ntool_packs = ["base"]\n'
    replies = [
        # A command writes the value into a file in the workspace; `read` and `grep` then show that file.
        Reply(tool_calls=[("bash", {"command": 'printf "token=%s\\n" "$DEMO_TOKEN" > leak.txt'})]),
        Reply(tool_calls=[("read", {"path": "leak.txt"})]),
        Reply(tool_calls=[("grep", {"pattern": "token=", "path": "."})]),
        Reply(text="done"),
    ]
    async with configured(tmp_path, replies, toml) as (session, model, home, work):
        await session.run("leak it")
        assert (work / "leak.txt").read_text() == "token=stored-token-value\n"  # the command's own file keeps it
        read_out, grep_out = outputs(session)[1:3]
        assert "token=[REDACTED]" in read_out and "token=[REDACTED]" in grep_out
        journal = json.dumps([entry.model_dump(mode="json") for entry in session.deps.journal.entries()])
        assert "stored-token-value" not in journal
        assert "stored-token-value" not in json.dumps(model.requests)
        written = [path for path in home.rglob("*") if path.is_file() and "stored-token-value" in path.read_text()]
        assert written == []


@pytest.mark.parametrize(
    ("profile_on", "expected"),
    [(True, ["with-tools", "extension"]), (False, ["extension"])],
    ids=["profile-and-flag", "no-extensions-keeps-the-flag"],
)
async def test_no_extensions_drops_only_the_profiles_extensions(tmp_path, profile_on, expected):
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    profile = tmp_path / "profile.toml"
    profile.write_text('extensions = ["tests.support:WithTools"]\n')
    with FakeOpenAI([Reply(text="ok")]) as model:
        bootstrap = Bootstrap(
            Options(
                cwd=work,
                system2=f"vllm/{MODEL}",
                profile=profile,
                persist=False,
                context_files=False,
                extensions=profile_on,
                extra_extensions=["intern.hooks:Extension"],
                memory_off=True,
            ),
            {"INTERN_HOME": str(home), "VLLM_BASE_URL": model.url, **model.system1_env},
        )
        async with await bootstrap.session(lambda event: None) as session:
            assert [extension.name for extension in session.deps.hooks.extensions] == expected


def test_every_capability_that_needs_an_install_names_an_extra_this_package_lists():
    """`/caps` installs a capability's extra as the installed package lists it: an extra renamed in pyproject.toml,
    or misspelt in a capability's `needs`, would leave nothing to install."""
    ids = [toggle.id for toggle in toggles(CapabilitySettings(), CapabilitySettings(), [], base=True)]
    declared = {key: needs for key in [*ids, f"{MCP_PREFIX}docs"] if (needs := capability_setup.capability_needs(key))}
    assert {"cap.browser", "cap.code-search", "cap.workspace", f"{MCP_PREFIX}docs"} <= set(declared)
    for key, needs in declared.items():
        assert needs.requirements(), f"{key}: intern lists no packages for the {needs.extra!r} extra"
