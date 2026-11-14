from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import SecretStr

from intern.executor import LocalExecutor
from intern.mcp_credentials import CredentialCommand, CredentialCommandError
from intern.secrets import (
    CommandBinding,
    CommandSecrets,
    EnvironmentBinding,
    EnvironmentSecrets,
    RoutedSecrets,
    SecretAccessDenied,
    SecretExecutor,
    SecretRedactor,
    SecretRef,
    command_secret,
)


def resolver() -> EnvironmentSecrets:
    return EnvironmentSecrets(
        [EnvironmentBinding(reference=SecretRef(name="search"), source_env="SEARCH_KEY", scopes=frozenset({"web"}))],
        {"SEARCH_KEY": "credential-value"},
    )


async def test_real_scoped_injection_redaction_and_ambient_scrubbing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AMBIENT_SECRET", "do-not-inherit")
    executor = SecretExecutor(
        LocalExecutor(),
        resolver(),
        scope="web",
        inject={"API_KEY": SecretRef(name="search")},
        base_env={"PATH": "/usr/bin:/bin", "AMBIENT_SECRET": "do-not-inherit"},
    )
    result = await executor.run(
        'test "$API_KEY" = credential-value; printf "%s|%s" "$API_KEY" "${AMBIENT_SECRET-unset}"',
        cwd=tmp_path,
        timeout_s=5,
        max_output_bytes=500,
    )
    assert result.exit_code == 0 and result.output == "[REDACTED]|unset"


async def test_captured_secret_output_is_redacted_before_publication(tmp_path):
    executor = SecretExecutor(LocalExecutor(), resolver(), scope="web", inject={"API_KEY": SecretRef(name="search")})
    capture = tmp_path / "published.txt"
    result = await executor.run(
        'printf "%s" "$API_KEY"',
        cwd=tmp_path,
        timeout_s=5,
        max_output_bytes=100,
        capture_path=capture,
    )
    assert result.exit_code == 0 and capture.read_text() == "[REDACTED]"
    assert capture.stat().st_mode & 0o777 == 0o600


def test_capture_redaction_handles_credentials_across_chunk_boundaries(tmp_path):
    source, target = tmp_path / "raw", tmp_path / "redacted"
    secret = "credential-value"
    data = b"x" * (65536 - 3) + secret.encode() + b"y" * 65536 + secret.encode()
    source.write_bytes(data)
    SecretRedactor([SecretStr(secret)]).redact_file(source, target)
    assert target.read_bytes() == data.replace(secret.encode(), b"[REDACTED]")


async def test_denied_scope_missing_secret_and_override_never_start(tmp_path: Path):
    for scope, env in [("wrong", None), ("web", {"API_KEY": "override"})]:
        executor = SecretExecutor(
            LocalExecutor(), resolver(), scope=scope, inject={"API_KEY": SecretRef(name="search")}
        )
        with pytest.raises(SecretAccessDenied):
            await executor.run("touch should-not-exist", cwd=tmp_path, timeout_s=5, max_output_bytes=100, env=env)
    assert not (tmp_path / "should-not-exist").exists()


async def test_secret_values_hidden_in_repr_and_unknown_scope_errors():
    value = await resolver().resolve(SecretRef(name="search"), scope="web")
    assert "credential-value" not in repr(value)
    with pytest.raises(SecretAccessDenied, match="unavailable"):
        await resolver().resolve(SecretRef(name="unknown"), scope="web")
    assert SecretRedactor([SecretStr("abc"), SecretStr("abcdef")]).redact("abcdef abc") == "[REDACTED] [REDACTED]"


async def test_abort_cancellation_propagates(tmp_path: Path):
    executor = SecretExecutor(LocalExecutor(), resolver(), scope="web", inject={"API_KEY": SecretRef(name="search")})
    abort = asyncio.Event()
    abort.set()
    result = await executor.run("sleep 30", cwd=tmp_path, timeout_s=5, max_output_bytes=100, abort=abort)
    assert result.aborted


def test_duplicate_refs_and_overridable_bindings_rejected():
    binding = EnvironmentBinding(reference=SecretRef(name="x"), source_env="X", scopes=frozenset({"scope"}))
    with pytest.raises(ValueError, match="Duplicate"):
        EnvironmentSecrets([binding, binding])
    with pytest.raises(ValueError, match="caller-overridable"):
        SecretExecutor(LocalExecutor(), resolver(), scope="web", inject={"PATH": SecretRef(name="search")})


def script(tmp_path: Path, name: str, body: str) -> str:
    path = tmp_path / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return str(path)


def command_binding(command: str, *, env: dict[str, str] | None = None) -> CommandBinding:
    spec = CredentialCommand(
        command=command,
        args=(),
        env={name: SecretStr(value) for name, value in (env or {"PATH": "/usr/bin:/bin"}).items()},
        cwd=Path("/"),
        timeout_s=5,
    )
    return CommandBinding(reference=SecretRef(name="vault"), command=spec, scopes=frozenset({"web"}))


async def test_command_secret_runs_once_keeps_lines_and_injects_into_commands(tmp_path: Path):
    count = tmp_path / "runs"
    printer = script(tmp_path, "op", f'echo run >> {count}; printf -- "-----BEGIN KEY-----\\nline-$TOKEN\\n"')
    resolver = CommandSecrets([command_binding(printer, env={"PATH": "/usr/bin:/bin", "TOKEN": "t1"})])
    executor = SecretExecutor(LocalExecutor(), resolver, scope="web", inject={"PEM": SecretRef(name="vault")})
    first = await executor.run(
        'printf "%s" "$PEM" | wc -l; printf "%s" "$PEM"', cwd=tmp_path, timeout_s=5, max_output_bytes=500
    )
    await executor.run("true", cwd=tmp_path, timeout_s=5, max_output_bytes=500)
    assert first.output.split()[0] == "1"  # two lines: the trailing newline only was dropped
    assert "line-t1" not in first.output and "[REDACTED]" in first.output
    assert count.read_text() == "run\n"


async def test_command_secret_refuses_other_scopes_and_reports_failures(tmp_path: Path):
    resolver = CommandSecrets([command_binding(script(tmp_path, "ok", "printf value"))])
    with pytest.raises(SecretAccessDenied):
        await resolver.resolve(SecretRef(name="vault"), scope="other")
    failing = CommandSecrets([command_binding(script(tmp_path, "bad", "echo locked >&2; exit 3"))])
    with pytest.raises(CredentialCommandError, match="exit 3") as error:
        await failing.resolve(SecretRef(name="vault"), scope="web")
    assert "locked" in error.value.stderr_tail
    empty = CommandSecrets([command_binding(script(tmp_path, "empty", "printf '\\n'"))])
    with pytest.raises(CredentialCommandError, match="invalid value"):
        await empty.resolve(SecretRef(name="vault"), scope="web")


async def test_routed_secrets_serve_each_name_from_its_own_resolver(tmp_path: Path):
    routed = RoutedSecrets(
        {
            "search": resolver(),
            "vault": CommandSecrets([command_binding(script(tmp_path, "ok", "printf from-command"))]),
        }
    )
    assert (await routed.resolve(SecretRef(name="search"), scope="web")).get_secret_value() == "credential-value"
    assert (await routed.resolve(SecretRef(name="vault"), scope="web")).get_secret_value() == "from-command"
    with pytest.raises(SecretAccessDenied):
        await routed.resolve(SecretRef(name="nothing"), scope="web")


def test_command_secret_value_rules():
    assert command_secret(b"a\nb\n") == "a\nb"
    assert command_secret(b"value\r\n") == "value"
    assert command_secret(b"\n") is None
    assert command_secret(b"\xff") is None
    assert command_secret(b"x" * (64 * 1024 + 1)) is None


def test_a_resolver_written_before_resolve_became_async_is_refused_with_the_fix():
    class SyncSecrets:
        def resolve(self, ref: SecretRef, scope: str) -> SecretStr:
            return SecretStr("value")

    with pytest.raises(TypeError, match=r"SyncSecrets\.resolve must be async since 0\.20\.0"):
        SecretExecutor(LocalExecutor(), SyncSecrets(), scope="web", inject={"API_KEY": SecretRef(name="search")})  # type: ignore[arg-type]


def test_stream_redaction_catches_values_split_across_any_chunk_boundary():
    redactor = SecretRedactor([SecretStr("abcdef"), SecretStr("xyz")])
    text = b"start abcdef middle xyz abc end abcdef"
    for size in range(1, len(text) + 1):
        stream = redactor.stream()
        out = b"".join(stream.feed(text[i : i + size]) for i in range(0, len(text), size)) + stream.flush()
        assert out == b"start [REDACTED] middle [REDACTED] abc end [REDACTED]", size
    plain = SecretRedactor([]).stream()
    assert plain.feed(b"as is") == b"as is" and plain.flush() == b""


async def test_prepared_job_has_only_the_allowed_environment_and_the_secret(tmp_path: Path):
    executor = SecretExecutor(
        LocalExecutor(),
        resolver(),
        scope="web",
        inject={"API_KEY": SecretRef(name="search")},
        base_env={"PATH": "/usr/bin:/bin", "AMBIENT_SECRET": "do-not-inherit"},
    )
    prepared = await executor.prepare('printf "%s|%s" "$API_KEY" "${AMBIENT_SECRET-unset}"')
    assert "credential-value" not in repr(prepared) and "credential-value" not in prepared.command
    process = await asyncio.create_subprocess_exec(
        "bash", "-c", prepared.command, env=dict(prepared.env), stdout=asyncio.subprocess.PIPE
    )
    out, _ = await process.communicate()
    stream = prepared.redactor.stream()
    assert stream.feed(out) + stream.flush() == b"[REDACTED]|unset"
