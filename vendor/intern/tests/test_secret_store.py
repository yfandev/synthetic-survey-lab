from __future__ import annotations

from pathlib import Path

import pytest

from intern.config import Bootstrap, Options, UserConfig
from intern.events import SecretUnavailable
from intern.protocol.acp_mapping import MappingContext, event_updates
from intern.secret_store import SecretFailure, read_entry


def tool(bin_dir: Path, name: str, body: str) -> Path:
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"TOKEN": "vault"}, 'TOKEN must be "keychain" or a command list'),
        ({"TOKEN": []}, 'TOKEN must be "keychain" or a command list'),
        ({"TOKEN": ["op", ""]}, 'TOKEN must be "keychain" or a command list'),
        ({"op:token": "keychain"}, "'op:token' is not an environment variable name"),
    ],
)
def test_secrets_table_accepts_only_keychain_or_a_command(entries, message):
    with pytest.raises(ValueError, match=message):
        UserConfig.model_validate({"secrets": entries})
    config = UserConfig.model_validate({"secrets": {"A_TOKEN": "keychain", "B_KEY": ["op", "read", "op://x"]}})
    assert config.secrets == {"A_TOKEN": "keychain", "B_KEY": ["op", "read", "op://x"]}


def test_a_command_entry_runs_with_the_callers_environment_and_one_newline_removed(tmp_path):
    tool(tmp_path / "bin", "op", 'printf "%s-value\\n" "$OP_SESSION"')
    env = {"PATH": str(tmp_path / "bin"), "OP_SESSION": "session"}
    assert read_entry("TOKEN", ["op", "read", "x"], env).get_secret_value() == "session-value"


@pytest.mark.parametrize(
    ("body", "reason", "message"),
    [
        ("exit 1", "command_failed", "TOKEN: op exited 1. Try: op signin"),
        ("printf short", "too_short", "TOKEN: value is 5 characters; the store needs at least 8"),
        ("true", "missing", "TOKEN: the command printed nothing"),
        ("exec /bin/sleep 5", "timeout", "TOKEN: op did not answer within 0.3 s; it may be waiting for an unlock"),
    ],
)
def test_a_failing_entry_names_itself_and_what_to_do_never_the_output(tmp_path, body, reason, message):
    tool(tmp_path / "bin", "op", f"echo leaked-detail >&2; {body}")
    with pytest.raises(SecretFailure) as error:
        read_entry("TOKEN", ["op"], {"PATH": str(tmp_path / "bin")}, timeout_s=0.3)
    assert error.value.reason == reason and str(error.value) == message
    with pytest.raises(SecretFailure, match="TOKEN: no-such-tool is not on PATH"):
        read_entry("TOKEN", ["no-such-tool"], {"PATH": str(tmp_path / "bin")})


def test_keychain_entries_use_the_system_tool_and_say_how_to_store_a_missing_item(tmp_path):
    bin_dir = tmp_path / "bin"
    tool(bin_dir, "security", '[ "$5" = GITHUB_TOKEN ] && printf "%s:%s" "$3" "keychain-value" && exit 0; exit 44')
    env = {"PATH": str(bin_dir)}
    assert read_entry("GITHUB_TOKEN", "keychain", env, platform="darwin").get_secret_value() == (
        "intern:keychain-value"
    )
    with pytest.raises(SecretFailure, match=r"OTHER: not in the keychain\. Store it with: intern secrets set"):
        read_entry("OTHER", "keychain", env, platform="darwin")

    tool(bin_dir, "secret-tool", "exit 1")
    with pytest.raises(SecretFailure, match="no keychain on this host") as error:
        read_entry("OTHER", "keychain", env, platform="linux")  # no D-Bus session
    assert error.value.reason == "no_keychain"
    with pytest.raises(SecretFailure, match="Store it with: intern secrets set OTHER"):
        read_entry("OTHER", "keychain", {**env, "DBUS_SESSION_BUS_ADDRESS": "unix:path=/x"}, platform="linux")
    with pytest.raises(SecretFailure, match="no keychain on this host"):
        read_entry("OTHER", "keychain", env, platform="win32")


def test_stored_values_reach_the_harness_key_lookups_over_an_exported_variable(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    tool(tmp_path / "bin", "fetch", "printf stored-key-value")
    (home / "config.toml").write_text('[secrets]\nOPENAI_API_KEY = ["fetch"]\n')
    env = {"INTERN_HOME": str(home), "PATH": str(tmp_path / "bin"), "OPENAI_API_KEY": "exported"}
    bootstrap = Bootstrap(Options(cwd=tmp_path), env)
    assert bootstrap.env["OPENAI_API_KEY"] == "stored-key-value"
    assert "stored-key-value" not in repr(bootstrap.store)


def test_entries_are_read_once_per_process_and_a_broken_one_is_reported_not_fatal(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    runs = tmp_path / "runs"
    tool(tmp_path / "bin", "counted", f'echo run >> "{runs}"; printf counted-secret-value')
    tool(tmp_path / "bin", "locked", "exit 1")
    (home / "config.toml").write_text('[secrets]\nGOOD_TOKEN = ["counted"]\nNPM_TOKEN = ["locked"]\n')
    env = {"INTERN_HOME": str(home), "PATH": str(tmp_path / "bin")}
    first, second = Bootstrap(Options(cwd=tmp_path), env), Bootstrap(Options(cwd=tmp_path), env)
    assert first.env["GOOD_TOKEN"] == second.env["GOOD_TOKEN"] == "counted-secret-value"
    assert runs.read_text().count("run") == 1
    assert [(f.name, f.reason) for f in second.store_failures] == [("NPM_TOKEN", "command_failed")]
    assert "NPM_TOKEN" not in second.env


def test_an_unloaded_secret_reaches_every_client_as_a_named_notice():
    event = SecretUnavailable(name="NPM_TOKEN", reason="command_failed", message="NPM_TOKEN: locked exited 1")
    ctx = MappingContext(cwd=Path("/w"))
    [update] = event_updates(event, ctx)
    assert "Secret not loaded: NPM_TOKEN: locked exited 1" in update.model_dump_json()


def test_keychain_entries_are_read_fresh_each_session_so_a_changed_value_applies(monkeypatch):
    from pydantic import SecretStr

    from intern import secret_store

    values = iter(["first-keychain-value", "second-keychain-value"])
    monkeypatch.setattr(secret_store, "read_entry", lambda name, entry, env, cwd=None: SecretStr(next(values)))
    read = [secret_store.read_store({"FRESH_TOKEN": "keychain"}, {})[0]["FRESH_TOKEN"] for _ in range(2)]
    assert [value.get_secret_value() for value in read] == ["first-keychain-value", "second-keychain-value"]


def test_harness_keys_are_the_variables_the_config_reads_keys_from():
    from intern.config import harness_keys

    config = UserConfig.model_validate(
        {
            "providers": {"mine": {"base_url": "http://x", "api_key_env": "MY_PROVIDER_KEY"}},
            "capabilities": {
                "mcp": [
                    {
                        "namespace": "gh",
                        "transport": "http",
                        "url": "https://x/mcp",
                        "header_env": {"Authorization": "GH_MCP_TOKEN"},
                        "remote_tools": {"lookup": ["read"]},
                    }
                ]
            },
            "secrets": {"GITHUB_TOKEN": "keychain"},
        }
    )
    keys = harness_keys(config)
    assert {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MY_PROVIDER_KEY", "GH_MCP_TOKEN", "DREX_API_KEY"} <= keys
    assert "GITHUB_TOKEN" not in keys and "OPENAI_BASE_URL" not in keys
