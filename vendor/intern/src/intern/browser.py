"""Optional session-owned browser tools with an enforced egress proxy."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib
import os
import uuid
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, ClassVar, Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, Field

from intern.async_tasks import join_tasks
from intern.browser_proxy import BrowserProxy
from intern.messages import ImagePart, ResourceRef
from intern.needs import Needs
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail, file_resource
from intern.web.reader import checked_url
from intern.web.storage import publish_private_file

BROWSER_NEEDS = Needs(
    title="Browser", extra="browser", modules=("playwright",), setup=(("-m", "playwright", "install", "chromium"),)
)
"""Playwright, then the Chromium build it drives: Playwright downloads it, and does nothing when it is there."""


class BrowserOptions(BaseModel):
    allow_private_urls: bool = False
    timeout_s: float = Field(default=30, gt=0, le=120)
    viewport_width: int = Field(default=1280, ge=320, le=1920)
    viewport_height: int = Field(default=720, ge=240, le=1080)
    max_dom_chars: int = Field(default=1_000_000, ge=1000, le=4_000_000)
    max_screenshot_bytes: int = Field(default=4_000_000, ge=1000, le=8_000_000)


class BrowserAction(BaseModel):
    action: Literal["navigate", "read", "screenshot", "click", "fill"]
    url: str = Field(default="", max_length=8192)
    selector: str = Field(default="", max_length=2048)
    text: str = Field(default="", max_length=100_000)
    offset: int = Field(default=0, ge=0)


class BrowserObservation(BaseModel):
    url: str
    title: str
    text: str
    next_offset: int | None = None
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)
    screenshot: ResourceRef | None = None


class BrowserSession:
    def __init__(self, artifact_dir: Path, options: BrowserOptions | None = None) -> None:
        self.options = (options or BrowserOptions()).model_copy(deep=True)
        self.artifact_dir = artifact_dir
        self.proxy = BrowserProxy(allow_private=self.options.allow_private_urls, timeout_s=self.options.timeout_s)
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._closed = False
        self._lock = asyncio.Lock()
        self._background: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> BrowserSession:
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def _route(self, route: Any) -> None:
        try:
            checked_url(route.request.url)
        except ValueError:
            await route.abort("blockedbyclient")
        except Exception:
            await route.abort("blockedbyclient")
        else:
            await route.continue_()

    async def _websocket(self, websocket: Any) -> None:
        await websocket.close()

    def _schedule(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _close_extra(self, page: Any) -> None:
        with contextlib.suppress(Exception):
            await page.close()

    async def _dismiss(self, dialog: Any) -> None:
        with contextlib.suppress(Exception):
            await dialog.dismiss()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Browser session is closed; create a new isolated session")
        if self._browser is not None:
            return
        try:
            api = importlib.import_module("playwright.async_api")
        except ImportError:
            raise RuntimeError(BROWSER_NEEDS.instruction) from None
        try:
            await self.proxy.start()
            self._playwright = await api.async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                chromium_sandbox=True,
                timeout=self.options.timeout_s * 1000,
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key in {"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "SYSTEMROOT"}
                },
                proxy={
                    "server": self.proxy.address,
                    "username": self.proxy.username,
                    "password": self.proxy.password,
                    "bypass": "<-loopback>",
                },
                args=[
                    "--proxy-bypass-list=<-loopback>",
                    "--disable-quic",
                    "--disable-background-networking",
                    "--dns-prefetch-disable",
                    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                    "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
                ],
            )
            self._context = await self._browser.new_context(
                viewport={"width": self.options.viewport_width, "height": self.options.viewport_height},
                service_workers="block",
                accept_downloads=False,
                permissions=[],
            )
            await self._context.add_init_script("""
                for (const name of ['RTCPeerConnection', 'webkitRTCPeerConnection', 'WebTransport']) {
                    Object.defineProperty(globalThis, name, {value: undefined, writable: false, configurable: false});
                }
            """)
            await self._context.route("**/*", self._route)
            await self._context.route_web_socket("**/*", self._websocket)
            self._page = await self._context.new_page()
            self._context.on("page", lambda page: self._schedule(self._close_extra(page)))
            self._page.on("dialog", lambda dialog: self._schedule(self._dismiss(dialog)))
            self._page.set_default_timeout(self.options.timeout_s * 1000)
        except BaseException:
            await self.aclose()
            raise

    async def perform(
        self, action: BrowserAction, *, max_chars: int = 4000
    ) -> tuple[BrowserObservation, ImagePart | None]:
        async with self._lock, asyncio.timeout(self.options.timeout_s):
            await self.start()
            before = self.proxy.denied_connections
            if action.action == "navigate":
                url = checked_url(action.url)
                fragment = urlsplit(action.url).fragment
                if fragment:
                    url += "#" + quote(fragment, safe="/%:@!$&'()*+,;=-._~?")
                response = await self._page.goto(url, wait_until="domcontentloaded")
                if response is not None and response.status >= 400:
                    raise ValueError("Navigation was refused or the source returned an HTTP error")
            elif action.action in {"click", "fill"}:
                if not action.selector:
                    raise ValueError("Provide an element selector")
                locator = self._page.locator(action.selector)
                if action.action == "click":
                    await locator.click()
                else:
                    await locator.fill(action.text)
            image = None
            resource = None
            if action.action == "screenshot":
                data = await self._page.screenshot(type="png", full_page=False)
                if len(data) > self.options.max_screenshot_bytes:
                    raise ValueError("Screenshot exceeds the configured byte limit")
                path = await asyncio.to_thread(
                    publish_private_file, self.artifact_dir, f"browser-{uuid.uuid4().hex}.png", data
                )
                resource = await asyncio.to_thread(file_resource, path, data)
                image = ImagePart(
                    path=str(path.resolve()),
                    mime="image/png",
                    width=self.options.viewport_width,
                    height=self.options.viewport_height,
                    source=self._page.url,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
            text = await self._page.evaluate(
                "limit => (document.body?.innerText || '').slice(0, limit)", self.options.max_dom_chars + 1
            )
            text = str(text)
            bounded = text[: self.options.max_dom_chars]
            if action.offset > len(bounded):
                raise ValueError("DOM text offset is past the retained content")
            end = min(len(bounded), action.offset + max(1, max_chars))
            warnings = []
            if len(text) > len(bounded):
                warnings.append(
                    "DOM text exceeds the retained text limit; narrow the page or inspect a specific section."
                )
            if self.proxy.denied_connections > before:
                warnings.append("Some page connections were blocked by the browser egress policy.")
            observation = BrowserObservation(
                url=self._page.url,
                title=(await self._page.title())[:1024],
                text=bounded[action.offset : end],
                next_offset=end if end < len(bounded) else None,
                truncated=end < len(text),
                warnings=warnings,
                screenshot=resource,
            )
            return observation, image

    async def aclose(self) -> None:
        self._closed = True
        try:
            if self._context is not None:
                with contextlib.suppress(Exception):
                    await self._context.close()
            if self._browser is not None:
                with contextlib.suppress(Exception):
                    await self._browser.close()
        finally:
            if self._playwright is not None:
                with contextlib.suppress(Exception):
                    await self._playwright.stop()
            await self.proxy.aclose()
            tasks = list(self._background)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._context = self._browser = self._playwright = self._page = None


def browser_tool(session: BrowserSession) -> Tool:
    class BrowserTool(Tool):
        name = "browser"
        description = (
            "Navigate HTTP(S) pages, read DOM text, take screenshots, click or fill selectors in this isolated session."
        )
        Args = BrowserAction
        effects: ClassVar[frozenset[Effect]] = frozenset({"read", "write", "network"})

        async def run(self, args: BrowserAction, ctx: ToolContext) -> ToolOutput:
            if ctx.abort.is_set():
                return fail("The run was aborted.", "Stop.").model_copy(update={"status": "refused"})
            action = asyncio.create_task(session.perform(args, max_chars=ctx.max_output_chars))
            aborted = asyncio.create_task(ctx.abort.wait())
            interrupted = False
            try:
                done, _ = await asyncio.wait({action, aborted}, return_when=asyncio.FIRST_COMPLETED)
                if action not in done:
                    interrupted = True
                    raise RuntimeError("Browser action interrupted")
                observation, image = action.result()
                return ToolOutput(
                    content=observation.model_dump_json(),
                    resources=[observation.screenshot] if observation.screenshot else [],
                    images=[image] if image is not None and ctx.image_dir is not None else [],
                )
            except Exception:
                output = fail(
                    "Browser action failed, timed out or was blocked by policy.",
                    "Inspect the page state before repeating an action; check browser setup and URL policy.",
                )
                assert output.error is not None
                output.error.retry_safe = False
                return output
            finally:
                try:
                    await join_tasks(action, aborted, cancel=True)
                finally:
                    if interrupted:
                        await join_tasks(asyncio.create_task(session.aclose()))

    return BrowserTool()
