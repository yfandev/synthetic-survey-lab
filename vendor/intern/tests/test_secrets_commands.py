from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import pytest

from intern.config import Bootstrap, Options
from intern.secrets_commands import secrets_main

FAKE_SECURITY = r"""#!/bin/sh
store="$FAKE_KEYCHAIN"
case "$1" in
  -i) read line; set -- $line; printf "%s" "$8" > "$store/$6" ;;
  find-generic-password) [ -f "$store/$5" ] || exit 44
    python3 -c "import sys; print(bytes.fromhex(open(sys.argv[1]).read()).decode())" "$store/$5" ;;
  delete-generic-password) [ -f "$store/$5" ] || exit 44; rm "$store/$5" ;;
esac
"""


@pytest.fixture
def mac(tmp_path, monkeypatch) -> dict[str, str]:
    """A macOS-like host whose `security` keeps items as files, and an env for the harness home."""
    monkeypatch.setattr(sys, "platform", "darwin")
    bin_dir, keychain, home = tmp_path / "bin", tmp_path / "keychain", tmp_path / "home"
    for folder in (bin_dir, keychain, home):
        folder.mkdir()
    (bin_dir / "security").write_text(FAKE_SECURITY)
    (bin_dir / "security").chmod(0o755)
    return {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "FAKE_KEYCHAIN": str(keychain),
        "INTERN_HOME": str(home),
    }


def run(argv: list[str], env: dict[str, str], monkeypatch, capsys, stdin: str = "") -> tuple[int, str]:
    monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = secrets_main(argv, env)
    return code, capsys.readouterr().out


def test_set_list_test_remove_round_trip_keeps_the_config_file_as_written(mac, monkeypatch, capsys):
    home = Path(mac["INTERN_HOME"])
    (home / "config.toml").write_text('# my settings\nsystem2 = "openai/gpt-5"\n')
    code, out = run(["set", "GITHUB_TOKEN", "--stdin", "--json"], mac, monkeypatch, capsys, "ghp_a'b\"c$d\n")
    assert (code, json.loads(out)) == (0, {"ok": True, "name": "GITHUB_TOKEN"})
    config = (home / "config.toml").read_text()
    assert config.startswith('# my settings\nsystem2 = "openai/gpt-5"\n') and 'GITHUB_TOKEN = "keychain"' in config
    assert "ghp_" not in config

    assert json.loads(run(["list", "--json"], mac, monkeypatch, capsys)[1]) == [
        {"name": "GITHUB_TOKEN", "source": "keychain", "harness_only": False}
    ]
    run(["set", "OPENAI_API_KEY", "--stdin"], mac, monkeypatch, capsys, "sk-model-key-value")
    listed = {e["name"]: e["harness_only"] for e in json.loads(run(["list", "--json"], mac, monkeypatch, capsys)[1])}
    assert listed == {"GITHUB_TOKEN": False, "OPENAI_API_KEY": True}  # a model key is the harness's alone
    run(["remove", "OPENAI_API_KEY"], mac, monkeypatch, capsys)
    code, out = run(["test", "--json"], mac, monkeypatch, capsys)
    assert code == 0 and json.loads(out)[0]["ok"] is True and "ghp_" not in out
    assert Bootstrap(Options(cwd=home), mac).env["GITHUB_TOKEN"] == "ghp_a'b\"c$d"

    code, out = run(["remove", "GITHUB_TOKEN", "--json"], mac, monkeypatch, capsys)
    assert code == 0 and "GITHUB_TOKEN" not in (home / "config.toml").read_text()
    assert not list(Path(mac["FAKE_KEYCHAIN"]).iterdir())


def test_set_refuses_what_the_store_cannot_keep_and_says_why(mac, monkeypatch, capsys):
    code, out = run(["set", "PIN", "--stdin", "--json"], mac, monkeypatch, capsys, "1234")
    assert code == 1 and "PIN: value is 4 characters" in json.loads(out)["error"]
    code, out = run(["set", "ACCENT", "--stdin", "--json"], mac, monkeypatch, capsys, "unicodé-value-123")
    assert code == 1 and "one line of plain ASCII" in json.loads(out)["error"]
    home = Path(mac["INTERN_HOME"])
    (home / "config.toml").write_text('[secrets]\nNPM_TOKEN = ["pass", "show", "npm"]\n')
    code, out = run(["set", "NPM_TOKEN", "--stdin", "--json"], mac, monkeypatch, capsys, "long-enough-value")
    assert code == 2 and "comes from a command" in json.loads(out)["error"]
    monkeypatch.setattr(sys, "platform", "linux")
    code, out = run(["set", "OTHER_TOKEN", "--stdin", "--json"], mac, monkeypatch, capsys, "long-enough-value")
    assert code == 1 and "no keychain on this host" in json.loads(out)["error"]


def test_test_reports_each_failure_by_name_without_the_value(mac, monkeypatch, capsys):
    home = Path(mac["INTERN_HOME"])
    (home / "config.toml").write_text('[secrets]\nMISSING_TOKEN = "keychain"\nBROKEN = ["sh", "-c", "exit 3"]\n')
    code, out = run(["test", "--json"], mac, monkeypatch, capsys)
    assert code == 1
    by_name = {status["name"]: status for status in json.loads(out)}
    assert by_name["MISSING_TOKEN"]["reason"] == "missing"
    assert "intern secrets set MISSING_TOKEN" in by_name["MISSING_TOKEN"]["message"]
    assert by_name["BROKEN"]["reason"] == "command_failed" and by_name["BROKEN"]["path"].endswith("/sh")
