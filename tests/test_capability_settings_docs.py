"""docs/capability-settings.md lists every top-level `[capabilities]` key with its kind, so a new key cannot land
undocumented."""

import re
from pathlib import Path

from intern.capability_settings import CapabilitySettings

DOC = Path(__file__).resolve().parent.parent / "docs" / "capability-settings.md"
KINDS = {"switch setting", "capability option", "session setting"}


def documented() -> dict[str, str]:
    section = DOC.read_text().split("## What each setting is", 1)[1].split("\n## ", 1)[0]
    return dict(re.findall(r"^\| `(\w+)` \| ([a-z ]+) \|", section, flags=re.MULTILINE))


def test_every_capability_setting_is_documented_with_its_kind():
    rows = documented()
    assert set(rows) == set(CapabilitySettings.model_fields)
    assert set(rows.values()) <= KINDS


def test_the_switches_documented_on_by_default_are_on_by_default():
    settings = CapabilitySettings()
    lines = DOC.read_text().splitlines()
    for key in ("browser", "code_search", "workspace", "memory"):
        row = next(line for line in lines if line.startswith(f"| `{key}` |"))
        assert row.endswith("| on |") == getattr(settings, key).enabled, row
