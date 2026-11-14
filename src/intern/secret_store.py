"""The secrets store: `[secrets]` in config.toml names each value's source, never the value itself.

    [secrets]
    GITHUB_TOKEN = "keychain"                                   # the OS keychain, service "intern"
    OPENAI_API_KEY = ["op", "read", "op://Private/OpenAI/key"]  # any command that prints the value

Each entry is read when a session is set up (a command entry only once per process). Its value reaches the
harness's own key lookups and the agent's commands under the entry's name, and is redacted from what those
commands print. An entry that can't be read is reported, and the session runs without it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from intern import names
from intern.models.registry import ConfigError
from intern.secrets import MAX_COMMAND_SECRET_BYTES

KEYCHAIN = "keychain"
KEYCHAIN_SERVICE = names.NAME
MIN_SECRET_CHARS = 8
"""Shorter values can't be redacted without mangling ordinary output."""
TIMEOUT_S = 30.0

StoreEntry = Literal["keychain"] | list[str]
Reason = Literal["missing", "no_keychain", "command_failed", "timeout", "too_short", "too_large", "invalid"]

ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAC_ITEM_NOT_FOUND = 44
_HINTS = {
    "op": "Try: op signin",
    "security": "Try: security unlock-keychain",
    "vault": "Try: vault login",
    "aws": "Try: aws sso login",
    "gcloud": "Try: gcloud auth login",
}


class SecretFailure(ConfigError):
    """A `[secrets]` entry that could not be read. The message names the entry and what to do; it never holds the
    value or what the command printed."""

    def __init__(self, name: str, reason: Reason, detail: str) -> None:
        super().__init__(f"{name}: {detail}")
        self.name = name
        self.reason: Reason = reason


def check_entries(entries: object) -> object:
    """The `[secrets]` table's shape, with a message that shows both forms."""
    if not isinstance(entries, dict):
        raise ValueError('[secrets] must be a table of NAME = "keychain" or NAME = ["command", "arg", ...]')
    for name, entry in entries.items():
        if not isinstance(name, str) or not ENV_NAME.fullmatch(name):
            raise ValueError(f"[secrets] {name!r} is not an environment variable name")
        command = isinstance(entry, list) and entry and all(isinstance(word, str) and word for word in entry)
        if entry != KEYCHAIN and not command:
            raise ValueError(f'[secrets] {name} must be "keychain" or a command list like ["op", "read", "op://..."]')
    return entries


def _keychain_tool(name: str, env: Mapping[str, str], platform: str) -> str:
    """`security` on macOS, `secret-tool` on Linux in a desktop session; a SecretFailure elsewhere."""
    path = env.get("PATH")
    if platform == "darwin" and shutil.which("security", path=path):
        return "security"
    if platform.startswith("linux") and shutil.which("secret-tool", path=path) and env.get("DBUS_SESSION_BUS_ADDRESS"):
        return "secret-tool"
    raise SecretFailure(
        name, "no_keychain", "no keychain on this host. Use a command entry, or keep it in an environment variable"
    )


def keychain_command(
    name: str, env: Mapping[str, str], platform: str | None = None, service: str = KEYCHAIN_SERVICE
) -> list[str]:
    """The system command that prints the keychain item for `name`; a SecretFailure where there is none."""
    platform = platform or sys.platform
    if _keychain_tool(name, env, platform) == "security":
        return ["security", "find-generic-password", "-s", service, "-a", name, "-w"]
    return ["secret-tool", "lookup", "service", service, "account", name]


def keychain_write(name: str, value: str, env: Mapping[str, str], platform: str | None = None) -> None:
    """Store `value` as the keychain item for `name`, replacing one already there. The value goes to the tool on
    stdin, never in its arguments, where other processes could read it."""
    platform = platform or sys.platform
    check_value(name, value.encode())
    if not (value.isascii() and value.isprintable()):
        # macOS `security -w` prints any other value as hex, so it would not read back as stored.
        raise SecretFailure(
            name, "invalid", "a keychain value must be one line of plain ASCII; use a command entry for anything else"
        )
    tool = _keychain_tool(name, env, platform)
    if tool == "security":
        # `security -i` reads its command from stdin; hex (-X) needs no quoting, whatever the value holds.
        argv = ["security", "-i"]
        stdin = f"add-generic-password -U -s {KEYCHAIN_SERVICE} -a {name} -X {value.encode().hex()}\n"
    else:
        label = f"--label={KEYCHAIN_SERVICE} {name}"
        argv = ["secret-tool", "store", label, "service", KEYCHAIN_SERVICE, "account", name]
        stdin = value
    _run_keychain(name, argv, stdin, env)


def keychain_delete(
    name: str, env: Mapping[str, str], platform: str | None = None, service: str = KEYCHAIN_SERVICE
) -> None:
    """Delete the keychain item for `name`; nothing to delete is not an error."""
    platform = platform or sys.platform
    if _keychain_tool(name, env, platform) == "security":
        argv = ["security", "delete-generic-password", "-s", service, "-a", name]
    else:
        argv = ["secret-tool", "clear", "service", service, "account", name]
    _run_keychain(name, argv, "", env, missing_ok=True)


def _run_keychain(name: str, argv: list[str], stdin: str, env: Mapping[str, str], *, missing_ok: bool = False) -> None:
    executable = shutil.which(argv[0], path=env.get("PATH")) or argv[0]
    try:
        done = subprocess.run(
            [executable, *argv[1:]], input=stdin.encode(), capture_output=True, env=dict(env), timeout=TIMEOUT_S
        )
    except subprocess.TimeoutExpired:
        raise SecretFailure(name, "timeout", f"{argv[0]} did not answer within {TIMEOUT_S:g} s") from None
    except OSError as error:
        raise SecretFailure(name, "command_failed", f"{argv[0]} could not start ({error.strerror})") from None
    if done.returncode == 0 or (missing_ok and done.returncode in (1, _MAC_ITEM_NOT_FOUND)):
        return
    hint = _HINTS.get(argv[0])
    raise SecretFailure(name, "command_failed", f"{argv[0]} exited {done.returncode}" + (f". {hint}" if hint else ""))


def store_hint(name: str) -> str:
    return f"Store it with: {names.NAME} secrets set {name}"


def read_entry(
    name: str,
    entry: StoreEntry,
    env: Mapping[str, str],
    *,
    cwd: Path | None = None,
    timeout_s: float = TIMEOUT_S,
    platform: str | None = None,
) -> SecretStr:
    """Run the entry's command with the caller's environment (so `op`, `vault` and `secret-tool` find their sessions),
    closed stdin and a timeout, and check what it printed."""
    platform = platform or sys.platform
    keychain = entry == KEYCHAIN
    argv = keychain_command(name, env, platform) if keychain else list(entry)
    tool = Path(argv[0]).name
    executable = argv[0] if "/" in argv[0] else shutil.which(argv[0], path=env.get("PATH"))
    if executable is None:
        raise SecretFailure(name, "command_failed", f"{argv[0]} is not on PATH")
    try:
        done = subprocess.run(
            [executable, *argv[1:]],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=dict(env),
            cwd=cwd,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise SecretFailure(
            name, "timeout", f"{tool} did not answer within {timeout_s:g} s; it may be waiting for an unlock"
        ) from None
    except OSError as error:
        raise SecretFailure(name, "command_failed", f"{tool} could not start ({error.strerror})") from None
    missing = keychain and (
        done.returncode == _MAC_ITEM_NOT_FOUND if platform == "darwin" else done.returncode == 1 and not done.stdout
    )
    if missing:
        raise SecretFailure(name, "missing", f"not in the keychain. {store_hint(name)}")
    if done.returncode != 0:
        hint = _HINTS.get(tool)
        raise SecretFailure(name, "command_failed", f"{tool} exited {done.returncode}" + (f". {hint}" if hint else ""))
    return check_value(name, done.stdout)


def check_value(name: str, stdout: bytes) -> SecretStr:
    """A value as the store accepts it: UTF-8, one trailing newline removed, 8 characters to 64 KiB."""
    if len(stdout) > MAX_COMMAND_SECRET_BYTES:
        raise SecretFailure(name, "too_large", f"value is over {MAX_COMMAND_SECRET_BYTES // 1024} KiB")
    try:
        text = stdout.decode()
    except UnicodeDecodeError:
        raise SecretFailure(name, "invalid", "value is not UTF-8 text") from None
    for newline in ("\r\n", "\n"):
        if text.endswith(newline):
            text = text[: -len(newline)]
            break
    if not text:
        raise SecretFailure(name, "missing", "the command printed nothing")
    if len(text) < MIN_SECRET_CHARS:
        raise SecretFailure(
            name, "too_short", f"value is {len(text)} characters; the store needs at least {MIN_SECRET_CHARS}"
        )
    return SecretStr(text)


_read: dict[tuple[str, tuple[str, ...]], SecretStr] = {}
"""Command entries already read in this process, so a new session (an editor opens many) runs no password-manager
command and asks for no unlock. Keychain entries are read fresh every session: that is silent and fast, and a value
changed in the keychain applies to the next session. A failure is not kept: the next session tries again."""


def read_store(
    entries: Mapping[str, StoreEntry], env: Mapping[str, str], *, cwd: Path | None = None
) -> tuple[dict[str, SecretStr], list[SecretFailure]]:
    """Every entry that could be read, and why each other one could not. One broken entry never stops the others."""
    values: dict[str, SecretStr] = {}
    failures: list[SecretFailure] = []
    for name, entry in entries.items():
        key = (name, (entry,) if isinstance(entry, str) else tuple(entry))
        if entry == KEYCHAIN or key not in _read:
            try:
                _read[key] = read_entry(name, entry, env, cwd=cwd)
            except SecretFailure as failure:
                failures.append(failure)
                continue
        values[name] = _read[key]
    return values, failures
