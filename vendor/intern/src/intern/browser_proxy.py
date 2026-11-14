"""Authenticated loopback forward proxy with socket-level destination policy.

Chromium must use this fixed proxy without bypasses. Every upstream connection is
made through PublicNetworkBackend, so validating DNS does not race another resolver.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hmac
import secrets
from typing import Any
from urllib.parse import urlsplit

import httpcore

from intern.web.reader import PublicNetworkBackend, checked_url


class BrowserProxy:
    def __init__(
        self,
        *,
        allow_private: bool = False,
        timeout_s: float = 30,
        backend: httpcore.AsyncNetworkBackend | None = None,
        max_connections: int = 16,
        max_total_connections: int = 512,
        max_stream_bytes: int = 32_000_000,
    ) -> None:
        self.backend = backend or PublicNetworkBackend(allow_private=allow_private)
        self.allow_private, self.timeout_s = allow_private, timeout_s
        self.max_connections, self.max_total_connections = max_connections, max_total_connections
        self.max_stream_bytes = max_stream_bytes
        self.username, self.password = secrets.token_hex(16), secrets.token_hex(24)
        self._authorization = "Basic " + base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
        self._server: asyncio.Server | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self.total_connections, self.denied_connections = 0, 0
        self.address = ""

    async def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("Browser proxy is already started")
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0, limit=32_768)
        self.address = f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}"

    @staticmethod
    async def _error(writer: asyncio.StreamWriter, status: str, *, authenticate: bool = False) -> None:
        challenge = 'Proxy-Authenticate: Basic realm="harness-browser"\r\n' if authenticate else ""
        writer.write(f"HTTP/1.1 {status}\r\n{challenge}Content-Length: 0\r\nConnection: close\r\n\r\n".encode())
        await writer.drain()

    async def _upload(self, reader: asyncio.StreamReader, upstream: httpcore.AsyncNetworkStream) -> None:
        count = 0
        while chunk := await reader.read(65_536):
            count += len(chunk)
            if count > self.max_stream_bytes:
                raise ValueError("Browser connection exceeded the upload limit")
            await upstream.write(chunk, timeout=self.timeout_s)

    async def _download(self, upstream: httpcore.AsyncNetworkStream, writer: asyncio.StreamWriter) -> None:
        count = 0
        while chunk := await upstream.read(65_536, timeout=self.timeout_s):
            count += len(chunk)
            if count > self.max_stream_bytes:
                raise ValueError("Browser connection exceeded the download limit")
            writer.write(chunk)
            await writer.drain()

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        assert task is not None
        if len(self._tasks) >= self.max_connections or self.total_connections >= self.max_total_connections:
            try:
                await self._error(writer, "503 Capacity exhausted")
            finally:
                writer.close()
            return
        self._tasks.add(task)
        self.total_connections += 1
        upstream = None
        pumps: list[asyncio.Task[None]] = []
        tunnel_started = False
        try:
            async with asyncio.timeout(self.timeout_s):
                header = await reader.readuntil(b"\r\n\r\n")
                if len(header) > 32_768:
                    raise ValueError("Oversized proxy request")
                lines = header.decode("latin1").split("\r\n")
                method, target, version = lines[0].split(" ")
                if version not in {"HTTP/1.0", "HTTP/1.1"}:
                    raise ValueError("Unsupported proxy protocol")
                headers: dict[str, str] = {}
                for line in lines[1:]:
                    if not line:
                        continue
                    key, value = line.split(":", 1)
                    key = key.lower()
                    if key in headers or not key or any(c.isspace() for c in key):
                        raise ValueError("Ambiguous proxy headers")
                    headers[key] = value.strip()
                if not hmac.compare_digest(headers.get("proxy-authorization", ""), self._authorization):
                    await self._error(writer, "407 Proxy Authentication Required", authenticate=True)
                    return
                connect = method == "CONNECT"
                if not connect and method not in {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"}:
                    raise ValueError("Unsupported request method")
                if "upgrade" in headers or headers.get("connection", "").lower() == "upgrade":
                    raise ValueError("WebSocket upgrades are disabled")
                parsed = urlsplit(checked_url("https://" + target if connect else target))
                if connect and (parsed.path != "/" or parsed.query):
                    raise ValueError("CONNECT requires an authority only")
                host = parsed.hostname
                assert host is not None
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                if not self.allow_private and port not in {80, 443}:
                    raise ValueError("Only public HTTP(S) destination ports are enabled")
                upstream = await self.backend.connect_tcp(host, port, timeout=self.timeout_s)
                if connect:
                    writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    await writer.drain()
                else:
                    if parsed.scheme != "http":
                        raise ValueError("HTTPS requires CONNECT")
                    path = parsed.path + ("?" + parsed.query if parsed.query else "")
                    origin_headers = {
                        k: v
                        for k, v in headers.items()
                        if k not in {"proxy-authorization", "proxy-connection", "connection", "host"}
                    }
                    origin_headers.update({"host": parsed.netloc, "connection": "close"})
                    request = f"{method} {path} HTTP/1.1\r\n" + "".join(
                        f"{k}: {v}\r\n" for k, v in origin_headers.items()
                    )
                    await upstream.write((request + "\r\n").encode("latin1"), timeout=self.timeout_s)
                tunnel_started = True
                pumps = [
                    asyncio.create_task(self._upload(reader, upstream)),
                    asyncio.create_task(self._download(upstream, writer)),
                ]
                done, _ = await asyncio.wait(pumps, return_when=asyncio.FIRST_COMPLETED)
                for finished in done:
                    finished.result()
        except Exception:
            self.denied_connections += 1
            if not tunnel_started:
                with contextlib.suppress(Exception):
                    await self._error(writer, "502 Browser egress refused")
        finally:
            for pump in pumps:
                if not pump.done():
                    pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            if upstream is not None:
                with contextlib.suppress(Exception):
                    await upstream.aclose()
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            self._tasks.discard(task)

    async def aclose(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
