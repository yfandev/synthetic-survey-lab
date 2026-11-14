from __future__ import annotations

import asyncio
import gzip
import hashlib
from pathlib import Path

import httpcore
import pytest

from intern.messages import ToolResultMessage
from intern.providers.fake import Turn
from intern.tools.web_read import WebReadTool
from intern.web.reader import PublicNetworkBackend, ReadSettings, WebReader, WebReadError, checked_url


def response(body: bytes, *, mime="text/html", extra="", status="200 OK") -> bytes:
    return f"HTTP/1.1 {status}\r\nContent-Type: {mime}\r\nContent-Length: {len(body)}\r\n{extra}\r\n".encode() + body


def mock_reader(body: bytes, *, mime="text/html", extra="", settings=None) -> WebReader:
    return WebReader(settings, network_backend=httpcore.AsyncMockBackend([response(body, mime=mime, extra=extra)]))


async def test_html_extraction_has_provenance_and_ignores_executable_content():
    body = b"<title>Document</title><style>bad</style><h1>A &amp; B</h1><script>steal()</script><p>Evidence</p>"
    page = await mock_reader(body).read("https://example.org/doc#section", asyncio.Event())
    assert page.final_url == "https://example.org/doc"
    assert page.title == "Document"
    assert page.text == "A & B\nEvidence"
    assert page.sha256 == hashlib.sha256(page.text.encode()).hexdigest()
    assert page.fetched_at.tzinfo is not None
    assert page.source_id.startswith("web-")


async def test_connection_policy_pins_numeric_address_and_rejects_mixed_private_dns():
    calls, connects = [], []

    async def resolver(host, port):
        calls.append((host, port))
        return ["93.184.216.34"] if len(calls) == 1 else ["127.0.0.1"]

    class Spy(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
            connects.append((host, port))
            inner: httpcore.AsyncNetworkBackend = httpcore.AsyncMockBackend([response(b"text", mime="text/plain")])
            return await inner.connect_tcp(host, port, timeout, local_address, socket_options)

    backend = PublicNetworkBackend(resolver=resolver, backend=Spy())
    page = await WebReader(network_backend=backend).read("https://example.org", asyncio.Event())
    assert page.text == "text"
    assert calls == [("example.org", 443)]
    assert connects == [("93.184.216.34", 443)]

    async def mixed(host, port):
        return ["93.184.216.34", "169.254.169.254"]

    with pytest.raises(WebReadError, match="non-public"):
        await PublicNetworkBackend(resolver=mixed, backend=Spy()).connect_tcp("example.org", 443)
    assert len(connects) == 1


async def test_redirect_is_rechecked_at_the_connection_boundary():
    connected = []

    async def resolver(host, port):
        return ["127.0.0.1"] if host == "internal.example" else ["93.184.216.34"]

    class Redirect(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
            connected.append(host)
            inner: httpcore.AsyncNetworkBackend = httpcore.AsyncMockBackend(
                [response(b"", status="302 Found", extra="Location: http://internal.example/metadata\r\n")]
            )
            return await inner.connect_tcp(host, port, timeout, local_address, socket_options)

    reader = WebReader(network_backend=PublicNetworkBackend(resolver=resolver, backend=Redirect()))
    with pytest.raises(WebReadError, match="non-public"):
        await reader.read("https://example.org", asyncio.Event())
    assert connected == ["93.184.216.34"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "https://user:pass@example.org",
        "https://example.org/\nX:yes",
        "http://[::1%25lo0]",
        "http://example.org:bad",
        "http://example.org\\@evil.org",
    ],
)
def test_unsafe_url_forms_are_rejected(url):
    with pytest.raises(WebReadError):
        checked_url(url)


async def test_download_decompression_and_mime_limits():
    with pytest.raises(WebReadError, match="download size"):
        await mock_reader(b"x" * 101, settings=ReadSettings(max_bytes=100)).read("https://example.org", asyncio.Event())
    with pytest.raises(WebReadError, match="decompressed size"):
        await mock_reader(
            gzip.compress(b"x" * 10_000), extra="Content-Encoding: gzip\r\n", settings=ReadSettings(max_bytes=100)
        ).read("https://example.org", asyncio.Event())
    with pytest.raises(WebReadError, match="Unsupported source type"):
        await mock_reader(b"%PDF", mime="application/pdf").read("https://example.org", asyncio.Event())
    with pytest.raises(WebReadError, match="compression stream was incomplete"):
        await mock_reader(gzip.compress(b"abc")[:-5], extra="Content-Encoding: gzip\r\n").read(
            "https://example.org", asyncio.Event()
        )
    assert (
        await mock_reader(gzip.compress(b"hello"), mime="text/plain", extra="Content-Encoding: gzip\r\n").read(
            "https://example.org", asyncio.Event()
        )
    ).text == "hello"


async def test_real_local_http_requires_explicit_operator_opt_in_and_cancels():
    started = asyncio.Event()

    async def serve(reader, writer):
        try:
            request = await reader.readuntil(b"\r\n\r\n")
            if b"GET /slow " in request:
                started.set()
                await reader.read()
            else:
                writer.write(response(b"local text", mime="text/plain"))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    try:
        with pytest.raises(WebReadError, match="non-public"):
            await WebReader().read(url, asyncio.Event())
        reader = WebReader(ReadSettings(allow_private=True))
        assert (await reader.read(url, asyncio.Event())).text == "local text"
        abort = asyncio.Event()
        task = asyncio.create_task(reader.read(url + "/slow", abort))
        await asyncio.wait_for(started.wait(), 1)
        abort.set()
        with pytest.raises(WebReadError, match="cancelled"):
            await task
        with pytest.raises(WebReadError, match="timed out"):
            await WebReader(ReadSettings(allow_private=True, timeout_s=0.01)).read(url + "/slow", asyncio.Event())
    finally:
        server.close()
        await server.wait_closed()


async def test_tool_retains_full_source_through_real_engine(make_harness, tmp_path: Path):
    body = ("evidence\n" * 10_000).encode()
    tool = WebReadTool(mock_reader(body, mime="text/plain"), storage_dir=tmp_path / "sources")
    harness = make_harness([Turn(tool_calls=[("web_read", {"url": "https://example.org"})]), Turn(text="Done")])
    harness.deps.tools.append(tool)
    await harness.run()
    results = [m for m in harness.provider.requests[-1].messages if isinstance(m, ToolResultMessage)]
    result = results[0].result
    assert result.status == "ok" and "TRUNCATED" in result.content
    assert len(result.resources) == 1
    path = next((tmp_path / "sources").glob("*.txt"))
    assert path.read_bytes() == body
    assert result.resources[0].sha256 == hashlib.sha256(body).hexdigest()


@pytest.mark.parametrize(
    "address",
    [
        "64:ff9b::7f00:1",
        "64:ff9b::a00:1",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b::c0a8:1",
        "64:ff9b:1::808:808",
        "::ffff:127.0.0.1",
        "2002:7f00:1::",
        "2001::ffff:80ff:fffe",
    ],
)
async def test_translation_destinations_cannot_hide_private_ipv4(address):
    connected = []

    async def resolver(host, port):
        return [address]

    class Spy(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
            connected.append(host)
            raise AssertionError("non-public translated address reached the connector")

    with pytest.raises(WebReadError, match="non-public"):
        await PublicNetworkBackend(resolver=resolver, backend=Spy()).connect_tcp("example.org", 443)
    assert connected == []


async def test_public_nat64_still_connects_to_the_validated_numeric_address():
    address = "64:ff9b::808:808"
    connected = []

    async def resolver(host, port):
        return [address]

    class Spy(httpcore.AsyncNetworkBackend):
        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):  # noqa: ASYNC109
            connected.append(host)
            inner: httpcore.AsyncNetworkBackend = httpcore.AsyncMockBackend([response(b"ok", mime="text/plain")])
            return await inner.connect_tcp(host, port, timeout, local_address, socket_options)

    page = await WebReader(network_backend=PublicNetworkBackend(resolver=resolver, backend=Spy())).read(
        "https://example.org", asyncio.Event()
    )
    assert page.text == "ok"
    assert connected == [address]


@pytest.mark.parametrize("unsafe", ["broad", "symlink"])
def test_retained_store_requires_a_private_owner_directory(tmp_path, unsafe):
    from intern.tools.web_read import _retain
    from intern.web.storage import PrivateStorageError

    folder = tmp_path / "store"
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    if unsafe == "broad":
        folder.mkdir(mode=0o755)
    else:
        folder.symlink_to(target, target_is_directory=True)
    with pytest.raises(PrivateStorageError, match=r"private|symlink"):
        _retain("source", hashlib.sha256(b"source").hexdigest(), folder)
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("unsafe", ["oversized", "symlink", "public_file", "wrong_bytes"])
def test_existing_retained_content_is_bounded_regular_and_private(tmp_path, unsafe):
    from intern.tools.web_read import _retain

    text = "source"
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = tmp_path / f"{digest}.txt"
    if unsafe == "symlink":
        actual = tmp_path / "actual"
        actual.write_text(text)
        actual.chmod(0o600)
        path.symlink_to(actual)
    else:
        path.touch(mode=0o600)
        with path.open("wb") as stream:
            if unsafe == "oversized":
                stream.truncate(64_000_000)
            else:
                stream.write(b"xxxxxx" if unsafe == "wrong_bytes" else text.encode())
        if unsafe == "public_file":
            path.chmod(0o644)
    with pytest.raises(OSError):
        _retain(text, digest, tmp_path)
    assert not list(tmp_path.glob(".pending-*"))


def test_retained_fifo_is_refused_without_blocking_a_worker(tmp_path):
    import os
    import subprocess
    import sys

    text = "source"
    digest = hashlib.sha256(text.encode()).hexdigest()
    os.mkfifo(tmp_path / f"{digest}.txt")
    script = """
import sys
from pathlib import Path
from intern.tools.web_read import _retain
from intern.web.storage import PrivateStorageError
try:
    _retain('source', sys.argv[2], Path(sys.argv[1]))
except PrivateStorageError:
    pass
else:
    raise AssertionError('FIFO was accepted')
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path), digest], check=True, timeout=2)
    assert not list(tmp_path.glob(".pending-*"))


def test_concurrent_duplicate_publication_allows_a_publishers_staging_link(tmp_path):
    import os
    import stat
    from concurrent.futures import ThreadPoolExecutor

    from intern.tools.web_read import _retain

    text = "shared evidence" * 1000
    data = text.encode()
    digest = hashlib.sha256(data).hexdigest()
    seed = tmp_path / "private-staging"
    seed.write_bytes(data)
    seed.chmod(0o600)
    target = tmp_path / f"{digest}.txt"
    os.link(seed, target)
    assert target.stat().st_nlink == 2
    assert _retain(text, digest, tmp_path) == target
    seed.unlink()
    target.unlink()
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda _: _retain(text, digest, tmp_path), range(24)))
    assert set(paths) == {target}
    assert target.read_bytes() == data
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".pending-*"))
