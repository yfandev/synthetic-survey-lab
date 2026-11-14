"""What a session opens with, as the TUI's capability switch reads and sets it: the effective capability
settings, and the memory drive, which the options can name or turn off."""

from __future__ import annotations

from pathlib import Path

from intern.capability_settings import CapabilitySettings
from intern.config import Bootstrap, Options


def home_with(tmp_path: Path, text: str) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.toml").write_text(text)
    return {"INTERN_HOME": str(home)}


def test_capability_settings_are_the_options_whole_object_else_the_saved_ones(tmp_path: Path):
    env = home_with(tmp_path, '[capabilities]\ntool_packs = ["base", "web-read"]\n')
    saved = Bootstrap(Options(cwd=tmp_path), env).capability_settings()
    assert saved.tool_packs == ["base", "web-read"] and not saved.browser.enabled
    given = CapabilitySettings(tool_packs=["base"], browser=saved.browser.model_copy(update={"enabled": True}))
    assert Bootstrap(Options(cwd=tmp_path, capabilities=given), env).capability_settings() is given


def test_the_memory_drive_comes_from_the_options_the_config_or_the_environment_and_memory_off_wins(tmp_path: Path):
    bare = home_with(tmp_path, "")
    assert Bootstrap(Options(cwd=tmp_path), bare).memory_path() == tmp_path / "home" / "memory"
    assert Bootstrap(Options(cwd=tmp_path), {**bare, "INTERN_MEMORY": "/drives/env"}).memory_path() == Path(
        "/drives/env"
    )
    saved = home_with(tmp_path, '[memory]\ndrive = "/drives/saved"\n')
    assert Bootstrap(Options(cwd=tmp_path), {**saved, "INTERN_MEMORY": "/drives/env"}).memory_path() == Path(
        "/drives/saved"
    )
    named = Options(cwd=tmp_path, memory=Path("/drives/named"))
    assert Bootstrap(named, saved).memory_path() == Path("/drives/named")
    assert Bootstrap(named.model_copy(update={"memory_off": True}), saved).memory_path() is None
    assert Bootstrap(Options(cwd=tmp_path, memory_off=True), saved).memory() is None
