"""MCP OAuth sign-in for HTTP entries, built on the MCP SDK's OAuth client provider.

`intern mcp login <namespace>` signs in once, in a browser, and saves the tokens. Sessions load them,
refresh them when they expire and never start a sign-in themselves: a session whose tokens are missing or can
no longer be refreshed fails with a message that names the login command.

Tokens live in `<config home>/mcp-oauth/<namespace>.json`, mode 0600, which the guard's credential rules keep
the agent's own tools from reading or writing. The MCP SDK is an optional extra, so it is imported on use.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import importlib
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import httpx
from pydantic import SecretStr

from intern.capability_settings import McpConfig, McpOAuthConfig

if TYPE_CHECKING:
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken

STORE_DIR = "mcp-oauth"
LOGIN_TIMEOUT_S = 300.0


class OAuthSignInRequired(Exception):
    """A session needs a sign-in it may not start itself."""

    def __init__(self, namespace: str) -> None:
        super().__init__(f"MCP {namespace} needs sign-in: run `intern mcp login {namespace}`")


def _auth_models() -> Any:
    try:
        return importlib.import_module("mcp.shared.auth")
    except ImportError:
        raise RuntimeError("MCP support requires the optional extra: uv sync --extra mcp") from None


def store_path(home: Path, namespace: str) -> Path:
    return home / STORE_DIR / f"{namespace}.json"


class OAuthStore:
    """The MCP SDK's `TokenStorage`, kept in one private JSON file per MCP entry.

    It also keeps when the access token expires: the SDK loads stored tokens without an expiry, so it would
    never refresh them before a 401, and on a 401 it starts a new sign-in instead of refreshing.
    Saved tokens belong to one server URL; a changed URL starts from nothing.
    """

    def __init__(self, path: Path, server_url: str, preregistered: OAuthClientInformationFull | None = None) -> None:
        self.path = path
        self.server_url = server_url
        self._preregistered = preregistered

    def __repr__(self) -> str:
        return f"OAuthStore({str(self.path)!r})"

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("server_url") != self.server_url:
            return {}
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path.parent.chmod(0o700)
        temporary = self.path.with_suffix(".tmp")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump({**data, "server_url": self.server_url}, handle)
        temporary.replace(self.path)

    def signed_in(self) -> bool:
        tokens = self._load().get("tokens")
        return isinstance(tokens, dict) and bool(tokens.get("access_token"))

    def forget(self) -> bool:
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        return True

    async def get_tokens(self) -> OAuthToken | None:
        data = self._load()
        if not data.get("tokens"):
            return None
        tokens = _auth_models().OAuthToken.model_validate(data["tokens"])
        expires_at = data.get("expires_at")
        expires_in = max(0, int(expires_at - time.time())) if isinstance(expires_at, int | float) else None
        return tokens.model_copy(update={"expires_in": expires_in})

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._load()
        data["tokens"] = tokens.model_dump(mode="json")
        data["expires_at"] = time.time() + tokens.expires_in if tokens.expires_in is not None else None
        self._save(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        info = self._load().get("client_info")
        if info:
            return _auth_models().OAuthClientInformationFull.model_validate(info)
        return self._preregistered

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._load()
        data["client_info"] = client_info.model_dump(mode="json")
        self._save(data)


def redirect_uri(config: McpOAuthConfig) -> str:
    return f"http://127.0.0.1:{config.callback_port}/callback"


def _client(
    config: McpOAuthConfig, secret: SecretStr | None
) -> tuple[OAuthClientMetadata, OAuthClientInformationFull | None]:
    models = _auth_models()
    method = "client_secret_post" if secret is not None else "none"
    metadata = models.OAuthClientMetadata(
        client_name="intern",
        redirect_uris=[redirect_uri(config)],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method=method,
        scope=config.scope,
    )
    if config.client_id is None:
        return metadata, None
    preregistered = models.OAuthClientInformationFull(
        **metadata.model_dump(),
        client_id=config.client_id,
        client_secret=secret.get_secret_value() if secret is not None else None,
    )
    return metadata, preregistered


def oauth_store(entry: McpConfig, home: Path, secret: SecretStr | None) -> OAuthStore:
    assert entry.oauth is not None and entry.url is not None
    _, preregistered = _client(entry.oauth, secret)
    return OAuthStore(store_path(home, entry.namespace), entry.url, preregistered)


@functools.cache
def _provider_class() -> type[Any]:
    base = importlib.import_module("mcp.client.auth").OAuthClientProvider

    class StoredExpiryProvider(base):  # type: ignore[valid-type, misc]
        """The SDK provider, with the expiry of stored tokens restored on load so they refresh before a 401."""

        async def _initialize(self) -> None:
            await super()._initialize()
            if self.context.current_tokens is not None:
                self.context.update_token_expiry(self.context.current_tokens)

    return StoredExpiryProvider


def oauth_auth(
    entry: McpConfig,
    store: OAuthStore,
    secret: SecretStr | None,
    *,
    redirect_handler: Callable[[str], Awaitable[None]] | None = None,
    callback_handler: Callable[[], Awaitable[tuple[str, str | None]]] | None = None,
) -> httpx.Auth:
    """The `httpx.Auth` for an OAuth entry. Without handlers it refuses to start a sign-in, as sessions must."""
    assert entry.oauth is not None and entry.url is not None
    metadata, _ = _client(entry.oauth, secret)

    async def refuse_redirect(_: str) -> None:
        raise OAuthSignInRequired(entry.namespace)

    async def refuse_callback() -> tuple[str, str | None]:
        raise OAuthSignInRequired(entry.namespace)

    return _provider_class()(
        server_url=entry.url,
        client_metadata=metadata,
        storage=store,
        redirect_handler=redirect_handler or refuse_redirect,
        callback_handler=callback_handler or refuse_callback,
        timeout=LOGIN_TIMEOUT_S,
    )


_DONE_PAGE = b"<!doctype html><title>intern</title><p>Signed in. You can close this tab.</p>"
_FAILED_PAGE = b"<!doctype html><title>intern</title><p>Sign-in failed. See the terminal.</p>"


class _Callback:
    """A one-shot local HTTP listener for the browser's return to `/callback?code=...&state=...`."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.result: asyncio.Future[tuple[str, str | None]] = asyncio.get_running_loop().create_future()
        self._server: asyncio.Server | None = None

    async def __aenter__(self) -> _Callback:
        try:
            self._server = await asyncio.start_server(self._handle, "127.0.0.1", self.port)
        except OSError as error:
            raise RuntimeError(f"Port {self.port} is busy; set [capabilities.mcp.oauth] callback_port") from error
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = (await asyncio.wait_for(reader.readline(), 10)).decode("latin-1")
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(10):
                    while (await reader.readline()) not in (b"\r\n", b"\n", b""):
                        pass
            parts = line.split()
            target = urlsplit(parts[1]) if len(parts) >= 2 else None
            ok = False
            if target is not None and target.path == "/callback" and not self.result.done():
                query = parse_qs(target.query)
                if "code" in query:
                    self.result.set_result((query["code"][0], query.get("state", [None])[0]))
                    ok = True
                else:
                    reason = query.get("error", ["no authorization code"])[0]
                    self.result.set_exception(RuntimeError(f"Sign-in failed: {reason}"))
            page = _DONE_PAGE if ok else _FAILED_PAGE
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(page)}\r\nConnection: close\r\n\r\n".encode()
                + page
            )
            await writer.drain()
        except (OSError, TimeoutError, UnicodeDecodeError):
            pass
        finally:
            writer.close()
            with contextlib.suppress(OSError):
                await writer.wait_closed()


async def login(
    entry: McpConfig,
    home: Path,
    headers: dict[str, SecretStr],
    secret: SecretStr | None,
    *,
    open_url: Callable[[str], Awaitable[None]],
) -> Path:
    """Sign in to an OAuth entry in the browser and save its tokens; returns where they were saved.

    `open_url` shows the authorization URL to the person signing in (the CLI prints it and opens a browser).
    """
    assert entry.oauth is not None and entry.url is not None
    from intern.mcp_client import http_connection

    store = oauth_store(entry, home, secret)
    async with _Callback(entry.oauth.callback_port) as callback:

        async def wait_for_callback() -> tuple[str, str | None]:
            return await asyncio.wait_for(asyncio.shield(callback.result), LOGIN_TIMEOUT_S)

        auth = oauth_auth(entry, store, secret, redirect_handler=open_url, callback_handler=wait_for_callback)
        async with (
            httpx.AsyncClient(
                headers={name: value.get_secret_value() for name, value in headers.items()},
                auth=auth,
                timeout=entry.timeout_s,
                follow_redirects=False,
                trust_env=False,
            ) as client,
            http_connection(entry.url, entry.namespace, timeout_s=LOGIN_TIMEOUT_S, http_client=client),
        ):
            pass
    if not store.signed_in():
        raise RuntimeError(f"MCP {entry.namespace} sign-in finished without tokens")
    return store.path
