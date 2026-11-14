"""Provider keys from the harness home's `.env`, for those the shell does not export."""

from __future__ import annotations

from pathlib import Path

import pytest

from intern.config import ENV_FILE, HOME_ENV, Bootstrap, Options, process_env, read_env_file
from intern.models.registry import resolve_endpoint


def test_the_home_env_file_fills_in_keys_the_shell_does_not_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    home.mkdir()
    (home / ENV_FILE).write_text(
        "# keys\n"
        'export ANTHROPIC_API_KEY="from-file"\n'
        "OPENROUTER_API_KEY=file-router\n"
        "QUOTED='x = y'\n"
        "not a line\n"
        "=novalue\n"
        "1BAD=x\n"
    )
    monkeypatch.setenv(HOME_ENV, str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-shell")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = process_env()
    assert env["ANTHROPIC_API_KEY"] == "from-file" and env["QUOTED"] == "x = y"
    assert env["OPENROUTER_API_KEY"] == "from-shell"  # the shell wins over the file
    assert "not a line" not in env and "" not in env and "1BAD" not in env
    # Bootstrap without an environment of its own takes the process environment, the file behind it.
    bootstrap = Bootstrap(Options(cwd=tmp_path))
    endpoint = resolve_endpoint("anthropic/claude-haiku-5-5", bootstrap.env, bootstrap.config.providers)
    assert endpoint.api_key is not None and endpoint.api_key.get_secret_value() == "from-file"
    # An environment a caller passes is taken as is: no file behind it, so tests and hosts stay in control.
    assert "ANTHROPIC_API_KEY" not in Bootstrap(Options(cwd=tmp_path), {HOME_ENV: str(home)}).env


def test_a_missing_env_file_is_empty(tmp_path: Path):
    assert read_env_file(tmp_path / ENV_FILE) == {}
