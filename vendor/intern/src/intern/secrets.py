"""Scoped secret references, resolved only at an authorized execution boundary."""

from __future__ import annotations

import asyncio
import inspect
import os
import re
import shlex
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field, SecretStr

from intern.executor import CapturingExecutor, ExecResult, Executor, WorkspaceMount, mount_of, run_command
from intern.mcp_credentials import CredentialCommand, run_credential_command

#: The most a secret command may print: room for a PEM key or a service-account JSON.
MAX_COMMAND_SECRET_BYTES = 64 * 1024


class SecretAccessDenied(Exception):
    """A deliberately value-free setup or authorization failure."""


class SecretRef(BaseModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")


class EnvironmentBinding(BaseModel):
    reference: SecretRef
    source_env: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    scopes: frozenset[str] = Field(min_length=1)


class SecretResolver(Protocol):
    async def resolve(self, reference: SecretRef, *, scope: str) -> SecretStr: ...


class EnvironmentSecrets:
    """A snapshot of explicitly named environment credentials; no ambient lookup."""

    def __init__(self, bindings: list[EnvironmentBinding], environ: Mapping[str, str] | None = None) -> None:
        names = [b.reference.name for b in bindings]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate secret reference")
        self._bindings = {b.reference.name: b.model_copy(deep=True) for b in bindings}
        source = os.environ if environ is None else environ
        self._values = {b.reference.name: SecretStr(source.get(b.source_env, "")) for b in bindings}

    async def resolve(self, reference: SecretRef, *, scope: str) -> SecretStr:
        binding = self._bindings.get(reference.name)
        if binding is None or scope not in binding.scopes:
            raise SecretAccessDenied("Secret reference is unavailable in this execution scope")
        value = self._values[reference.name]
        if not value.get_secret_value():
            raise SecretAccessDenied("A credential required by this scope is not configured")
        return value


@dataclass(frozen=True)
class CommandBinding:
    reference: SecretRef
    command: CredentialCommand
    """Run once; its environment holds only what the binding names, never the session's."""
    scopes: frozenset[str]

    def __post_init__(self) -> None:
        if not self.scopes:
            raise ValueError("A secret binding needs at least one scope")


def command_secret(stdout: bytes) -> str | None:
    """What a secret command printed, less one trailing newline: non-empty UTF-8, possibly several lines."""
    if len(stdout) > MAX_COMMAND_SECRET_BYTES:
        return None
    try:
        text = stdout.decode()
    except UnicodeDecodeError:
        return None
    for newline in ("\r\n", "\n"):
        if text.endswith(newline):
            text = text[: -len(newline)]
            break
    return text or None


class CommandSecrets:
    """Values a host command prints (`op read`, `security find-generic-password -w`, `pass show`, ...), each run once
    on first use and kept for the life of this resolver."""

    def __init__(self, bindings: list[CommandBinding]) -> None:
        names = [b.reference.name for b in bindings]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate secret reference")
        self._bindings = {b.reference.name: b for b in bindings}
        self._values: dict[str, SecretStr] = {}
        self._lock = asyncio.Lock()

    async def resolve(self, reference: SecretRef, *, scope: str) -> SecretStr:
        binding = self._bindings.get(reference.name)
        if binding is None or scope not in binding.scopes:
            raise SecretAccessDenied("Secret reference is unavailable in this execution scope")
        async with self._lock:
            if reference.name not in self._values:
                self._values[reference.name] = await run_credential_command(binding.command, value=command_secret)
        return self._values[reference.name]


class RoutedSecrets:
    """One resolver over several: each reference name is served by the resolver it is configured in."""

    def __init__(self, routes: Mapping[str, SecretResolver]) -> None:
        self._routes = dict(routes)

    async def resolve(self, reference: SecretRef, *, scope: str) -> SecretStr:
        resolver = self._routes.get(reference.name)
        if resolver is None:
            raise SecretAccessDenied("Secret reference is unavailable in this execution scope")
        return await resolver.resolve(reference, scope=scope)


class SecretRedactor:
    """Literal redaction for ordinary output, not a general information-flow guard."""

    def __init__(self, values: list[SecretStr]) -> None:
        self._values = sorted({v.get_secret_value() for v in values if v.get_secret_value()}, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for value in self._values:
            text = text.replace(value, "[REDACTED]")
        return text

    def stream(self) -> StreamRedactor:
        """A redactor for output that arrives in chunks, such as a background job's log."""
        return StreamRedactor([value.encode() for value in self._values])

    def redact_file(self, source: Path, target: Path) -> None:
        """Publish a bounded-memory redacted copy, including credentials split across read chunks."""
        stream = self.stream()
        with (
            source.open("rb") as src,
            os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as dst,
        ):
            while chunk := src.read(64 * 1024):
                dst.write(stream.feed(chunk))
            dst.write(stream.flush())


class StreamRedactor:
    """Literal redaction of a byte stream: holds back the last few bytes of each chunk until the next one (or
    `flush`), so a value split across two chunks is still caught."""

    def __init__(self, values: list[bytes]) -> None:
        values = [value for value in values if value]
        self._pattern = re.compile(b"|".join(re.escape(value) for value in values)) if values else None
        self._overlap = max(map(len, values), default=1) - 1
        self._pending = b""

    def feed(self, chunk: bytes) -> bytes:
        if self._pattern is None:
            return chunk
        pending = self._pending + chunk
        cut = max(0, len(pending) - self._overlap)
        for match in self._pattern.finditer(pending):
            if match.start() < cut < match.end():
                cut = match.start()
                break
        ready, self._pending = pending[:cut], pending[cut:]
        return self._pattern.sub(b"[REDACTED]", ready)

    def flush(self) -> bytes:
        pending, self._pending = self._pending, b""
        return self._pattern.sub(b"[REDACTED]", pending) if self._pattern is not None else pending


@dataclass(frozen=True, repr=False)
class PreparedCommand:
    """A command wrapped to run with exactly `env` (the injected secrets included), and the redactor for its output."""

    command: str
    env: Mapping[str, str]
    redactor: SecretRedactor

    def __repr__(self) -> str:
        return "PreparedCommand(...)"


class SecretExecutor:
    """Inject scoped references into a scrubbed environment for an existing Executor.

    The configured scope and bindings are application-owned, never tool arguments.
    Executed code can access injected values; it must therefore be trusted or
    contained by an appropriate executor and egress policy.
    """

    def __init__(
        self,
        executor: Executor,
        resolver: SecretResolver,
        *,
        scope: str,
        inject: Mapping[str, SecretRef],
        base_env: Mapping[str, str] | None = None,
        allowed_env: frozenset[str] = frozenset({"PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TEMP", "SYSTEMROOT"}),
    ) -> None:
        if not scope or any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in [*inject, *allowed_env]):
            raise ValueError("A fixed scope and valid target environment names are required")
        if allowed_env.intersection(inject):
            raise ValueError("Injected credential names cannot also be caller-overridable environment names")
        if not inspect.iscoroutinefunction(resolver.resolve):
            raise TypeError(
                f"{type(resolver).__name__}.resolve must be async since 0.20.0; "
                "wrap blocking work in `await asyncio.to_thread(...)`"
            )
        self.executor, self.resolver, self.scope = executor, resolver, scope
        self.inject = {name: ref.model_copy(deep=True) for name, ref in inject.items()}
        self.allowed_env = allowed_env
        source = os.environ if base_env is None else base_env
        self.base_env = {name: value for name, value in source.items() if name in allowed_env}

    @property
    def output_capture(self) -> bool:
        return isinstance(self.executor, CapturingExecutor) and self.executor.output_capture

    @property
    def mount(self) -> WorkspaceMount | None:
        return mount_of(self.executor)

    async def prepare(self, command: str, *, env: Mapping[str, str] | None = None) -> PreparedCommand:
        """Resolve the injected references and wrap `command` to run with exactly the allowed environment plus them,
        for the wrapped executor to run (a command, or a background job through the process manager)."""
        if env is not None and not env.keys() <= self.allowed_env:
            raise SecretAccessDenied("Caller environment contains names outside the configured allowlist")
        resolved: dict[str, SecretStr] = {}
        try:
            for target, reference in self.inject.items():
                resolved[target] = await self.resolver.resolve(reference, scope=self.scope)
        except Exception:
            raise SecretAccessDenied("Credential resolution refused; execution was not started") from None
        environment = {
            **self.base_env,
            **(env or {}),
            **{name: value.get_secret_value() for name, value in resolved.items()},
        }
        # LocalExecutor merges ambient os.environ; use an exact inner environment.
        assignments = " ".join(f'{name}="${{{name}}}"' for name in environment)
        isolated = f"exec /usr/bin/env -i {assignments} /bin/bash --noprofile --norc -c {shlex.quote(command)}"
        return PreparedCommand(
            command=isolated,
            env={**environment, "BASH_ENV": "/dev/null", "ENV": "/dev/null"},
            redactor=SecretRedactor(list(resolved.values())),
        )

    async def run(
        self,
        command: str,
        *,
        cwd: Path,
        timeout_s: float,
        max_output_bytes: int,
        abort=None,
        env: Mapping[str, str] | None = None,
        capture_path: Path | None = None,
    ) -> ExecResult:
        prepared = await self.prepare(command, env=env)
        redactor = prepared.redactor

        async def execute_capture(path: Path | None) -> ExecResult:
            return await run_command(
                self.executor,
                prepared.command,
                cwd=cwd,
                timeout_s=timeout_s,
                max_output_bytes=max_output_bytes,
                abort=abort,
                env=prepared.env,
                capture_path=path,
            )

        try:
            if capture_path is None:
                result = await execute_capture(None)
            else:
                # Raw credential-bearing bytes never appear at the published artifact path.
                with tempfile.TemporaryDirectory(prefix="harness-private-output-") as private:
                    raw = Path(private) / "stdout"
                    result = await execute_capture(raw)
                    redactor.redact_file(raw, capture_path)
        except Exception:
            # A transport exception can embed env, request headers or subprocess text.
            raise SecretAccessDenied("Authorized execution failed; credential-bearing details were withheld") from None
        return result.model_copy(update={"output": redactor.redact(result.output)})
