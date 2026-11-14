"""`intern mcp login|logout <namespace>`: sign in to, or forget, an OAuth MCP entry from config.toml."""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from collections.abc import Awaitable, Callable, Mapping

from pydantic import SecretStr

from intern import names
from intern.capability_settings import McpConfig
from intern.config import home_dir, load_user_config, process_env
from intern.mcp_oauth import login, oauth_store
from intern.models.registry import ConfigError
from intern.session_setup import oauth_client_secret

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


async def _show(url: str) -> None:
    print(f"Open this address to sign in:\n\n  {url}\n", file=sys.stderr)
    webbrowser.open(url)


def _entry(namespace: str, env: Mapping[str, str]) -> McpConfig:
    entries = {entry.namespace: entry for entry in load_user_config(home_dir(env)).capabilities.mcp}
    entry = entries.get(namespace)
    if entry is None:
        raise ConfigError(f"No [[capabilities.mcp]] entry has namespace {namespace!r}")
    if entry.oauth is None:
        raise ConfigError(f"MCP {namespace} has no [capabilities.mcp.oauth] table, so it does not sign in")
    return entry


def mcp_main(
    argv: list[str],
    env: Mapping[str, str] | None = None,
    open_url: Callable[[str], Awaitable[None]] = _show,
) -> int:
    parser = argparse.ArgumentParser(prog=f"{names.NAME} mcp", description="Sign in to MCP servers that use OAuth.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("login", "Sign in in the browser and save the tokens for later sessions."),
        ("logout", "Delete the saved tokens."),
    ):
        commands.add_parser(name, help=help_text).add_argument("namespace")
    args = parser.parse_args(argv)
    env = process_env() if env is None else env
    try:
        entry = _entry(args.namespace, env)
        if args.command == "logout":
            store = oauth_store(entry, home_dir(env), None)
            print(f"Signed out of MCP {entry.namespace}" if store.forget() else "Nothing was saved", file=sys.stderr)
            return EXIT_OK
        headers: dict[str, SecretStr] = {}
        for header, source in entry.header_env.items():
            value = env.get(source, "")
            if not value.strip():
                raise ConfigError(f"MCP {entry.namespace} needs the environment variable {source}")
            headers[header] = SecretStr(value)
        path = asyncio.run(login(entry, home_dir(env), headers, oauth_client_secret(entry, env), open_url=open_url))
    except ConfigError as error:
        print(f"{names.NAME}: {error}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as error:  # every sign-in failure ends the command with its reason
        print(f"{names.NAME}: MCP {args.namespace} sign-in failed: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_FAILED
    print(f"Signed in to MCP {entry.namespace}; tokens saved in {path}", file=sys.stderr)
    return EXIT_OK
