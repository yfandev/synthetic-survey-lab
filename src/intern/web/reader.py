"""Bounded web reading with destination policy enforced when each socket connects.

Using HTTPCore's network backend preserves HTTP Host/TLS SNI while connecting to an already
validated IP. A preflight DNS check followed by an ordinary hostname connection would be racy.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import re
import socket
import zlib
from collections.abc import Awaitable, Callable, Iterable
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any, ClassVar, Protocol, cast
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import httpcore
from pydantic import BaseModel, Field

from intern import names

Resolver = Callable[[str, int], Awaitable[list[str]]]


class WebReadError(Exception):
    pass


class ReadSettings(BaseModel):
    timeout_s: float = Field(default=30, gt=0, le=180)
    max_bytes: int = Field(default=4_000_000, ge=1, le=64_000_000)
    max_redirects: int = Field(default=5, ge=0, le=10)
    allow_private: bool = False


class PageDocument(BaseModel):
    source_id: str
    requested_url: str
    final_url: str
    fetched_at: datetime
    mime: str
    title: str
    text: str
    sha256: str
    warnings: list[str] = Field(default_factory=list)


def checked_url(value: str) -> str:
    if len(value) > 8_192 or any(ord(c) < 33 for c in value) or "\\" in value:
        raise WebReadError("URL contains unsupported whitespace, control characters or escapes.")
    try:
        parsed = urlsplit(value)
        port = parsed.port
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError
        if port is not None and not 1 <= port <= 65535:
            raise ValueError
        host = parsed.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        if "%" in host:
            raise ValueError
    except (ValueError, UnicodeError):
        raise WebReadError("Use an HTTP(S) URL without embedded credentials or an invalid host/port.") from None
    authority = f"[{host}]" if ":" in host else host
    if port is not None:
        authority += f":{port}"
    return urlunsplit(
        (
            parsed.scheme,
            authority,
            quote(parsed.path or "/", safe="/%:@!$&'()*+,;=-._~"),
            quote(parsed.query, safe="%:@!$&'()*+,;=/?-._~"),
            "",
        )
    )


async def resolve_addresses(host: str, port: int) -> list[str]:
    try:
        info = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError:
        raise WebReadError("The source hostname could not be resolved.") from None
    return list(dict.fromkeys(str(item[4][0]) for item in info))


def _public_destination(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if not ip.is_global or ip.is_multicast or ip.is_unspecified:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        # RFC 8215 local-use translation is not public destination space.
        if ip in ipaddress.IPv6Network("64:ff9b:1::/48"):
            return False
        embedded: tuple[ipaddress.IPv4Address, ...] = ()
        if ip in ipaddress.IPv6Network("64:ff9b::/96"):
            # RFC 6052 NAT64: IPv6 .is_global does not classify the IPv4 target.
            embedded = (ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF),)
        elif ip.ipv4_mapped is not None:
            embedded = (ip.ipv4_mapped,)
        elif ip.sixtofour is not None:
            embedded = (ip.sixtofour,)
        elif ip.teredo is not None:
            embedded = ip.teredo
        return all(_public_destination(address) for address in embedded)
    return True


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(
        self,
        *,
        allow_private: bool = False,
        resolver: Resolver = resolve_addresses,
        backend: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self.allow_private = allow_private
        self.resolver = resolver
        self.backend: httpcore.AsyncNetworkBackend = backend or cast(
            httpcore.AsyncNetworkBackend, httpcore.AnyIOBackend()
        )

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 — HTTPCore's required interface
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await self.resolver(host, port)
        if not addresses:
            raise WebReadError("The source hostname has no usable address.")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                raise WebReadError("The source resolved to an invalid address.") from None
            if not self.allow_private and not _public_destination(ip):
                raise WebReadError("Private, local and non-public source addresses are disabled.")
        # Pass a validated numeric address to the underlying connector: never resolve host again.
        last_error: Exception | None = None
        for address in addresses:
            try:
                return await self.backend.connect_tcp(address, port, timeout, local_address, socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as error:
                last_error = error
        raise httpcore.ConnectError("No validated address accepted the connection.") from last_error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 — HTTPCore's required interface
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise WebReadError("Unix socket sources are unsupported.")

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class TextExtractor(Protocol):
    def extract(self, source: str, mime: str) -> tuple[str, str]: ...


class _HTMLText(HTMLParser):
    ignored: ClassVar[frozenset[str]] = frozenset({"script", "style", "noscript", "template", "svg"})
    blocks: ClassVar[frozenset[str]] = frozenset(
        {"p", "div", "article", "section", "li", "tr", "td", "th", "br", "h1", "h2", "h3", "h4", "h5", "h6"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.skip = 0
        self.in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.ignored:
            self.skip += 1
        if tag == "title":
            self.in_title = True
        if not self.skip and tag in self.blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.ignored:
            self.skip = max(0, self.skip - 1)
        if tag == "title":
            self.in_title = False
        if not self.skip and tag in self.blocks:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.in_title:
            self.title_parts.append(data)
        elif not self.skip:
            self.parts.append(data)


class DeterministicExtractor:
    def extract(self, source: str, mime: str) -> tuple[str, str]:
        if mime in {"text/html", "application/xhtml+xml"}:
            parser = _HTMLText()
            parser.feed(source)
            title = " ".join(" ".join(parser.title_parts).split())
            text = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
            return title, text
        return "", source


class WebReader:
    def __init__(
        self,
        settings: ReadSettings | None = None,
        *,
        extractor: TextExtractor | None = None,
        network_backend: httpcore.AsyncNetworkBackend | None = None,
    ) -> None:
        self.settings = settings or ReadSettings()
        self.extractor = extractor or DeterministicExtractor()
        self.network_backend = network_backend or PublicNetworkBackend(allow_private=self.settings.allow_private)

    async def _read(self, requested: str) -> PageDocument:
        settings = self.settings
        url = checked_url(requested)
        async with httpcore.AsyncConnectionPool(
            network_backend=self.network_backend, max_connections=1, max_keepalive_connections=0, retries=0
        ) as pool:
            for hop in range(settings.max_redirects + 1):
                async with pool.stream(
                    "GET",
                    url,
                    headers={"User-Agent": f"{names.USER_AGENT} web-reader", "Accept-Encoding": "gzip, deflate"},
                    extensions={"timeout": {"connect": settings.timeout_s, "read": settings.timeout_s}},
                ) as response:
                    headers = {k.decode("ascii").lower(): v.decode("latin1") for k, v in response.headers}
                    if response.status in {301, 302, 303, 307, 308}:
                        if hop == settings.max_redirects:
                            raise WebReadError("The source exceeded the redirect limit.")
                        if "location" not in headers:
                            raise WebReadError("The source returned a redirect without a destination.")
                        url = checked_url(urljoin(url, headers["location"]))
                        continue
                    if response.status != 200:
                        raise WebReadError(f"The source returned HTTP {response.status}.")
                    content_type = headers.get("content-type", "application/octet-stream")
                    mime = content_type.split(";", 1)[0].strip().lower()
                    if not (mime.startswith("text/") or mime in {"application/xhtml+xml", "application/json"}):
                        raise WebReadError(f"Unsupported source type {mime}; use a document or binary reader.")
                    encoding = headers.get("content-encoding", "identity").strip().lower()
                    if encoding not in {"identity", "gzip", "deflate"}:
                        raise WebReadError("The source uses unsupported content compression.")
                    decoder = (
                        zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
                        if encoding != "identity"
                        else None
                    )
                    raw_size = 0
                    body = bytearray()
                    async for chunk in response.aiter_stream():
                        raw_size += len(chunk)
                        if raw_size > settings.max_bytes:
                            raise WebReadError("The source exceeded the download size limit.")
                        remaining = settings.max_bytes + 1 - len(body)
                        body.extend(decoder.decompress(chunk, remaining) if decoder else chunk)
                        if len(body) > settings.max_bytes or (decoder is not None and decoder.unconsumed_tail):
                            raise WebReadError("The extracted source exceeded the decompressed size limit.")
                    if decoder is not None and not decoder.eof:
                        raise WebReadError("The source compression stream was incomplete.")
                    if decoder is not None and decoder.unused_data:
                        raise WebReadError("The source has unsupported trailing compression data.")
                    charset_match = re.search(r"charset\s*=\s*[\"']?([\w.-]+)", content_type, re.I)
                    charset = charset_match.group(1) if charset_match else "utf-8"
                    try:
                        source = bytes(body).decode(charset)
                    except (LookupError, UnicodeDecodeError):
                        raise WebReadError("The source cannot be decoded using its declared text encoding.") from None
                    title, text = self.extractor.extract(source, mime)
                    warnings = [] if text.strip() else ["No readable text; the source may require rendering or OCR."]
                    return PageDocument(
                        source_id="web-" + hashlib.sha256(url.encode()).hexdigest()[:16],
                        requested_url=checked_url(requested),
                        final_url=url,
                        fetched_at=datetime.now(UTC),
                        mime=mime,
                        title=title,
                        text=text,
                        sha256=hashlib.sha256(text.encode()).hexdigest(),
                        warnings=warnings,
                    )
        raise WebReadError("No readable source response.")

    async def read(self, url: str, abort: asyncio.Event) -> PageDocument:
        if abort.is_set():
            raise WebReadError("URL read cancelled.")

        async def perform() -> PageDocument:
            try:
                async with asyncio.timeout(self.settings.timeout_s):
                    return await self._read(url)
            except (TimeoutError, httpcore.TimeoutException):
                raise WebReadError("URL read timed out.") from None
            except (httpcore.NetworkError, httpcore.ProtocolError, OSError, zlib.error):
                raise WebReadError(
                    "The source could not be read safely; check its URL or use another source."
                ) from None

        pending = asyncio.create_task(perform())
        cancelled = asyncio.create_task(abort.wait())
        try:
            done, _ = await asyncio.wait({pending, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done:
                raise WebReadError("URL read cancelled.")
            return await pending
        finally:
            for task in (pending, cancelled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(pending, cancelled, return_exceptions=True)
