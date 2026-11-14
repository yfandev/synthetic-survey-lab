from __future__ import annotations

import asyncio
import base64
import os
import stat
from pathlib import Path
from typing import cast

import httpcore
import pytest

from intern.browser import BrowserAction, BrowserOptions, BrowserSession, browser_tool
from intern.browser_proxy import BrowserProxy
from intern.executor import LocalExecutor
from intern.tools.base import ToolContext
from intern.web.reader import PublicNetworkBackend


@pytest.fixture
async def website():
    hits: list[str] = []
    port: int = 0

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            path = header.split(b" ")[1].decode()
            hits.append(path)
            if b"proxy-authorization:" in header.lower():
                hits.append("LEAKED_PROXY_AUTH")
            if path == "/redirect":
                response = f"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{port}/secret\r\nContent-Length:0\r\n\r\n"
                writer.write(response.encode())
            else:
                body = f"""<html><title>Local browser fixture</title><body>
<h1>Browser ready</h1><input id="name"><button id="go"
onclick="document.querySelector('#result').textContent=document.querySelector('#name').value">Go</button>
<p id="result"></p><img src="http://127.0.0.1:{port}/secret">
<script>document.cookie='fixture=value';</script></body></html>""".encode()
                writer.write(
                    (
                        f"HTTP/1.1 200 OK\r\nContent-Type:text/html\r\nContent-Length:{len(body)}\r\n"
                        "Connection:close\r\n\r\n"
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
        yield port, hits
    finally:
        server.close()
        await server.wait_closed()


async def test_proxy_requires_auth_and_denies_private_connect(website):
    port, hits = website
    proxy = BrowserProxy(timeout_s=2)
    await proxy.start()
    proxy_port = int(proxy.address.rsplit(":", 1)[1])
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(f"GET http://127.0.0.1:{port}/ HTTP/1.1\r\nHost:127.0.0.1\r\n\r\n".encode())
        await writer.drain()
        assert b"407" in await reader.read()
        writer.close()
        await writer.wait_closed()
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        credential = base64.b64encode(f"{proxy.username}:{proxy.password}".encode()).decode()
        writer.write(f"CONNECT 127.0.0.1:443 HTTP/1.1\r\nProxy-Authorization: Basic {credential}\r\n\r\n".encode())
        await writer.drain()
        assert b"502" in await reader.read()
        writer.close()
        await writer.wait_closed()
        assert hits == []
    finally:
        await proxy.aclose()


browser_test = pytest.mark.skipif(
    not os.environ.get("INTERN_BROWSER_TEST"), reason="explicit prepared Chromium required"
)


@browser_test
async def test_real_chromium_local_actions_screenshot_and_fresh_context(tmp_path: Path, website):
    port, hits = website
    async with BrowserSession(tmp_path, BrowserOptions(allow_private_urls=True)) as browser:
        observation, _ = await browser.perform(BrowserAction(action="navigate", url=f"http://127.0.0.1:{port}/"))
        assert "Browser ready" in observation.text
        await browser.perform(BrowserAction(action="fill", selector="#name", text="Ada"))
        observation, _ = await browser.perform(BrowserAction(action="click", selector="#go"))
        assert "Ada" in observation.text
        observation, image = await browser.perform(BrowserAction(action="screenshot"))
        assert image is not None and (await asyncio.to_thread(Path(image.path).read_bytes)).startswith(b"\x89PNG")
        metadata = await asyncio.to_thread(Path(image.path).stat)
        assert stat.S_IMODE(metadata.st_mode) == 0o600
        assert observation.screenshot is not None
        assert await browser._context.cookies()
        assert await browser._page.evaluate("typeof RTCPeerConnection") == "undefined"
        assert await browser._page.evaluate("typeof WebTransport") == "undefined"
        await browser._page.evaluate(f"""() => new Promise(resolve => {{
            const ws = new WebSocket('ws://127.0.0.1:{port}/socket');
            ws.onerror = () => resolve('blocked'); ws.onclose = () => resolve('closed');
            setTimeout(() => resolve('no connection'), 100);
        }})""")
        assert "/socket" not in hits and "LEAKED_PROXY_AUTH" not in hits
        assert await browser._page.evaluate("alert('test dialog'); 'dismissed'") == "dismissed"
        await browser._page.evaluate("window.open('/', '_blank'); null")
        await asyncio.sleep(0.05)
        assert len(browser._context.pages) == 1
    assert browser.proxy._server is None and not browser.proxy._tasks
    async with BrowserSession(tmp_path, BrowserOptions(allow_private_urls=True)) as other:
        assert await other._context.cookies() == []


@browser_test
async def test_real_chromium_cannot_bypass_proxy_for_loopback(tmp_path: Path, website):
    port, hits = website
    async with BrowserSession(tmp_path, BrowserOptions(timeout_s=5)) as browser:
        ctx = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=1000)
        output = await browser_tool(browser).run(BrowserAction(action="navigate", url=f"http://127.0.0.1:{port}/"), ctx)
        assert output.status == "error" and hits == []
        assert browser.proxy.denied_connections > 0
        assert output.error is not None and not output.error.retry_safe
        await browser.proxy.aclose()
        output = await browser_tool(browser).run(BrowserAction(action="navigate", url=f"http://127.0.0.1:{port}/"), ctx)
        assert output.status == "error" and hits == []  # No DIRECT fallback when the fixed proxy is unavailable.


@browser_test
async def test_real_browser_redirects_and_subresources_use_pinned_policy(tmp_path: Path, website):
    port, hits = website
    fixture_port = port
    numeric_connections: list[str] = []

    async def resolve(host: str, destination_port: int) -> list[str]:
        return ["1.1.1.1"] if host == "public.test" else ["127.0.0.1"]

    class FixtureTransport(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
            numeric_connections.append(host)
            # Only the connector below the production policy is replaced: it maps
            # the already-approved public destination to our local fixture.
            backend = cast(httpcore.AsyncNetworkBackend, httpcore.AnyIOBackend())
            return await backend.connect_tcp("127.0.0.1", fixture_port, timeout, local_address, socket_options)

    async with BrowserSession(tmp_path, BrowserOptions(timeout_s=5)) as browser:
        browser.proxy.backend = PublicNetworkBackend(resolver=resolve, backend=FixtureTransport())
        observation, _ = await browser.perform(BrowserAction(action="navigate", url="http://public.test/"))
        assert "Browser ready" in observation.text
        # The requested subresource points at loopback and must never hit fixture.
        await asyncio.sleep(0.1)
        assert "/secret" not in hits
        with pytest.raises(ValueError, match="refused"):
            await browser.perform(BrowserAction(action="navigate", url="http://public.test/redirect"))
        assert "/redirect" in hits and "/secret" not in hits
        assert numeric_connections and set(numeric_connections) == {"1.1.1.1"}
        assert browser.proxy.denied_connections > 0


@pytest.mark.parametrize("unsafe", ["broad_directory", "symlink_directory", "symlink_file"])
async def test_screenshot_storage_refuses_unsafe_targets_without_overwriting(tmp_path, monkeypatch, unsafe):
    from types import SimpleNamespace

    class Page:
        url = "https://example.org/private"

        async def screenshot(self, **kwargs):
            return b"private screenshot"

        async def evaluate(self, *args):
            return "private text"

        async def title(self):
            return "private title"

    target = tmp_path / "private"
    target.mkdir(mode=0o700)
    folder = tmp_path / "artifacts"
    original = target / "original"
    original.write_bytes(b"private screenshot")
    original.chmod(0o600)
    if unsafe == "broad_directory":
        folder.mkdir(mode=0o755)
    elif unsafe == "symlink_directory":
        folder.symlink_to(target, target_is_directory=True)
    else:
        folder.mkdir(mode=0o700)
        monkeypatch.setattr("intern.browser.uuid.uuid4", lambda: SimpleNamespace(hex="collision"))
        (folder / "browser-collision.png").symlink_to(original)
    browser = BrowserSession(folder)
    browser._browser = object()  # Exercise publication without launching Chromium.
    browser._page = Page()
    try:
        with pytest.raises(OSError):
            await browser.perform(BrowserAction(action="screenshot"))
        assert original.read_bytes() == b"private screenshot"
        assert not list(folder.glob(".pending-*"))
        if unsafe != "symlink_file":
            assert not list(folder.glob("*.png"))
    finally:
        await browser.aclose()


@browser_test
async def test_real_browser_abort_closes_inflight_navigation_and_proxy(tmp_path: Path):
    started = asyncio.Event()
    handlers: set[asyncio.Task] = set()

    async def stalled(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        handlers.add(task)
        try:
            await reader.readuntil(b"\r\n\r\n")
            started.set()
            await reader.read()  # Browser cancellation must close this upstream socket.
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(task)

    server = await asyncio.start_server(stalled, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        async with BrowserSession(tmp_path, BrowserOptions(allow_private_urls=True, timeout_s=5)) as browser:
            context = ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=1000)
            pending = asyncio.create_task(
                browser_tool(browser).run(BrowserAction(action="navigate", url=f"http://127.0.0.1:{port}/"), context)
            )
            await asyncio.wait_for(started.wait(), timeout=5)
            context.abort.set()
            output = await asyncio.wait_for(pending, timeout=5)
            assert output.status == "error" and output.error is not None and not output.error.retry_safe
            assert browser._closed and browser.proxy._server is None and not browser.proxy._tasks
    finally:
        server.close()
        await server.wait_closed()
        for task in list(handlers):
            task.cancel()
        await asyncio.gather(*handlers, return_exceptions=True)
