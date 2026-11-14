"""Every extension the harness ships is one of two kinds, and naming one the wrong way is a clear error.

A fixed extension is chosen for the whole session by a profile's `extensions` list or `--extension`, and is in
`REGISTRY`. An extension behind a capability is turned on by that capability's switch, and is in
`CAPABILITY_EXTENSIONS`, which refuses it when it is named as an extension."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import intern
from intern.capability_settings import CapabilitySettings
from intern.capability_toggles import toggles
from intern.config import CAPABILITY_EXTENSIONS, _extension
from intern.extensions import REGISTRY
from intern.hooks import Extension, load_extension
from intern.locate.store import Locate
from intern.models.registry import ConfigError

BASE_CLASSES: dict[type[Extension], str] = {
    Locate: "the base of code search and the workspace, and of third-party search extensions; never loaded itself",
}


def shipped_extensions() -> set[type[Extension]]:
    for module in pkgutil.walk_packages(intern.__path__, f"{intern.__name__}."):
        if module.name.endswith("__main__"):
            continue
        importlib.import_module(module.name)
    found: set[type[Extension]] = set()
    pending = [Extension]
    while pending:
        for cls in pending.pop().__subclasses__():
            if cls not in found:
                found.add(cls)
                pending.append(cls)
    return {cls for cls in found if cls.__module__.startswith(f"{intern.__name__}.")}


def test_every_shipped_extension_is_exactly_one_kind():
    fixed = set(REGISTRY.values())
    for cls in shipped_extensions():
        kinds = [
            cls in fixed,
            any(issubclass(cls, entry.cls) for entry in CAPABILITY_EXTENSIONS.values()),
            cls in BASE_CLASSES,
        ]
        assert sum(kinds) == 1, f"{cls.__module__}.{cls.__qualname__}: fixed, behind a capability, base = {kinds}"


def test_every_capability_extension_names_a_real_switch():
    settings = CapabilitySettings()
    switches = {toggle.id for toggle in toggles(settings, settings, ["base"])}
    assert {entry.toggle for entry in CAPABILITY_EXTENSIONS.values()} <= switches


@pytest.mark.parametrize("name", sorted(CAPABILITY_EXTENSIONS))
def test_an_extension_behind_a_capability_is_refused_by_name_and_by_class(name):
    entry = CAPABILITY_EXTENSIONS[name]
    with pytest.raises(ConfigError) as by_name:
        _extension(name)
    with pytest.raises(ConfigError) as by_class:
        _extension(f"{entry.cls.__module__}:{entry.cls.__qualname__}")
    assert str(by_name.value) == str(by_class.value) == entry.message


def test_memory_is_refused_before_it_is_built():
    """`Memory` needs its drive, so building it first would fail with a `TypeError` instead of this message."""
    with pytest.raises(ConfigError, match="memory is a capability, on by default"):
        _extension("intern.extensions.memory:Memory")


class Broken(Extension):
    name = "broken"

    def __init__(self) -> None:
        raise RuntimeError("no config here")


def test_an_extension_that_cannot_be_built_is_a_config_error_with_its_cause():
    with pytest.raises(ConfigError, match=r"cannot build extension .*Broken.*RuntimeError: no config here") as raised:
        load_extension(f"{__name__}:Broken", {})
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_a_base_class_named_as_an_extension_is_a_config_error():
    with pytest.raises(ConfigError, match="cannot build extension"):
        _extension("intern.locate.store:Locate")
