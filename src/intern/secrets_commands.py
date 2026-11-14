"""`intern secrets set|list|test|remove`: manage the `[secrets]` store from a terminal, the TUI or a GUI.

Every command takes `--json`, so a GUI can spawn it and read one JSON document from stdout. No output, JSON or
text, ever holds a value. `set` reads the value from the terminal without echo, or from stdin with `--stdin`.
"""

from __future__ import annotations

import argparse
import getpass
import json
import shutil
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import tomlkit
from pydantic import BaseModel

from intern import names
from intern.config import CONFIG_FILE, harness_keys, home_dir, load_user_config, process_env
from intern.models.registry import ConfigError
from intern.secret_store import (
    ENV_NAME,
    KEYCHAIN,
    Reason,
    SecretFailure,
    StoreEntry,
    keychain_delete,
    keychain_write,
    read_entry,
)

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2


class SecretEntry(BaseModel):
    name: str
    source: Literal["keychain", "command"]
    command: list[str] | None = None
    """For a command entry, its words as configured."""
    harness_only: bool = False
    """A key the harness reads itself (a model's, search's, MCP's): the agent's commands don't get it."""


class SecretStatus(BaseModel):
    name: str
    source: Literal["keychain", "command"]
    ok: bool
    harness_only: bool = False
    reason: Reason | None = None
    message: str = ""
    """Why it could not be read, and what to do; empty when `ok`."""
    path: str | None = None
    """The program a command entry runs, as found on PATH."""
    duration_ms: int = 0


def _entry(name: str, entry: StoreEntry, harness: frozenset[str] = frozenset()) -> SecretEntry:
    if entry == KEYCHAIN:
        return SecretEntry(name=name, source="keychain", harness_only=name in harness)
    return SecretEntry(name=name, source="command", command=list(entry), harness_only=name in harness)


def entries(home: Path) -> list[SecretEntry]:
    """What `[secrets]` names, in config order. Reads nothing from the keychain or any command."""
    config = load_user_config(home)
    harness = harness_keys(config)
    return [_entry(name, entry, harness) for name, entry in config.secrets.items()]


def check(home: Path, env: Mapping[str, str], names: list[str] | None = None) -> list[SecretStatus]:
    """Read each entry now, the way a session would, and report how it went."""
    config = load_user_config(home)
    configured, harness = config.secrets, harness_keys(config)
    unknown = [name for name in names or [] if name not in configured]
    if unknown:
        raise ConfigError(f"[secrets] has no {', '.join(unknown)}")
    statuses = []
    for name, entry in configured.items():
        if names and name not in names:
            continue
        info = _entry(name, entry, harness)
        path = None if entry == KEYCHAIN else shutil.which(entry[0], path=env.get("PATH"))
        started = time.monotonic()
        try:
            read_entry(name, entry, env, cwd=home if home.is_dir() else None)
            status = SecretStatus(name=name, source=info.source, ok=True, path=path, harness_only=info.harness_only)
        except SecretFailure as failure:
            status = SecretStatus(
                name=name,
                source=info.source,
                ok=False,
                reason=failure.reason,
                message=str(failure),
                path=path,
                harness_only=info.harness_only,
            )
        statuses.append(status.model_copy(update={"duration_ms": int((time.monotonic() - started) * 1000)}))
    return statuses


def _config(home: Path) -> tuple[Path, tomlkit.TOMLDocument]:
    path = home / CONFIG_FILE
    return path, tomlkit.parse(path.read_text()) if path.is_file() else tomlkit.document()


def store(name: str, value: str, home: Path, env: Mapping[str, str]) -> None:
    """Put `value` in the keychain and `NAME = "keychain"` in config.toml, keeping the file's comments and layout."""
    if not ENV_NAME.fullmatch(name):
        raise ConfigError(f"{name!r} is not an environment variable name")
    current = load_user_config(home).secrets.get(name)
    if current is not None and current != KEYCHAIN:
        raise ConfigError(f"{name} comes from a command in config.toml; remove that line to keep it in the keychain")
    keychain_write(name, value, env)
    if current is None:
        path, document = _config(home)
        table = document.get("secrets")
        if table is None:
            table = tomlkit.table()
            document["secrets"] = table
        table[name] = KEYCHAIN
        home.mkdir(parents=True, exist_ok=True)
        path.write_text(tomlkit.dumps(document))


def remove(name: str, home: Path, env: Mapping[str, str]) -> None:
    """Delete the keychain item (for a keychain entry) and the entry's line in config.toml."""
    current = load_user_config(home).secrets.get(name)
    if current is None:
        raise ConfigError(f"[secrets] has no {name}")
    if current == KEYCHAIN:
        keychain_delete(name, env)
    path, document = _config(home)
    del document["secrets"][name]  # type: ignore[union-attr]
    path.write_text(tomlkit.dumps(document))


def _read_value(from_stdin: bool) -> str:
    if from_stdin:
        value = sys.stdin.read()
    elif sys.stdin.isatty():
        value = getpass.getpass("Value (not shown): ")
    else:
        raise ConfigError("no terminal to ask for the value; pass it on stdin with --stdin")
    for newline in ("\r\n", "\n"):
        if value.endswith(newline):
            return value[: -len(newline)]
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{names.NAME} secrets", description="Keep keys in the OS keychain instead of plain-text files."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    put = commands.add_parser("set", help="Store a value in the keychain and name it in config.toml.")
    put.add_argument("name", help="The environment variable commands will see, e.g. GITHUB_TOKEN.")
    put.add_argument("--stdin", action="store_true", help="Read the value from stdin instead of asking for it.")
    commands.add_parser("list", help="Show each entry and where it comes from; reads no values.")
    test = commands.add_parser("test", help="Read each entry now and say whether it works.")
    test.add_argument("names", nargs="*")
    commands.add_parser("remove", help="Delete an entry and its keychain item.").add_argument("name")
    for command in commands.choices.values():
        command.add_argument("--json", action="store_true", help="Print one JSON document to stdout.")
    return parser


def secrets_main(argv: list[str], env: Mapping[str, str] | None = None) -> int:
    args = _parser().parse_args(argv)
    env = process_env() if env is None else env
    home = home_dir(env)
    try:
        if args.command == "list":
            listed = entries(home)
            if args.json:
                print(_dump([entry.model_dump(exclude_none=True) for entry in listed]))
            for entry in [] if args.json else listed:
                where = " ".join(entry.command) if entry.command else entry.source
                print(f"{entry.name}  {where}" + ("  (harness only)" if entry.harness_only else ""))
            return EXIT_OK
        if args.command == "test":
            statuses = check(home, env, args.names)
            if args.json:
                print(_dump([status.model_dump(exclude_none=True) for status in statuses]))
            for status in [] if args.json else statuses:
                print(f"{status.name}  ok" if status.ok else status.message)
            return EXIT_OK if all(status.ok for status in statuses) else EXIT_FAILED
        if args.command == "set":
            store(args.name, _read_value(args.stdin), home, env)
            done = f"Stored {args.name} in the keychain"
        else:
            remove(args.name, home, env)
            done = f"Removed {args.name}"
    except (ConfigError, ValueError) as error:  # SecretFailure is a ConfigError
        if args.json:
            print(_dump({"ok": False, "error": str(error)}))
        else:
            print(f"{names.NAME}: {error}", file=sys.stderr)
        return EXIT_FAILED if isinstance(error, SecretFailure) else EXIT_USAGE
    if args.json:
        print(_dump({"ok": True, "name": args.name}))
    else:
        print(done, file=sys.stderr)
    return EXIT_OK


def _dump(data: object) -> str:
    return json.dumps(data)
