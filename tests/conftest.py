from __future__ import annotations

from pathlib import Path

import pytest

from intern import decisions
from intern.decisions import DecisionBackend, DecisionClient
from intern.executor import LocalExecutor
from intern.hooks import Extension, Hooks
from intern.instructions import FileInstructions
from intern.journal import Journal
from intern.loop import EngineDeps
from intern.models.profile import Profile
from intern.providers.fake import FakeProvider, Script
from intern.tools import load_packs, with_extension_tools
from intern.tools import process as process_tool

from .support import Filled, Harness, Quiet, Recorder, binding


@pytest.fixture(autouse=True)
def quick_system1_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """A System 1 that fails in a test gives up in a fraction of a second, not after the two minutes a live one gets."""
    monkeypatch.setattr(decisions, "RETRY_WINDOW_S", 0.2)
    monkeypatch.setattr(decisions, "RETRY_FIRST_S", 0.01)


@pytest.fixture(autouse=True)
def jobs_start_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """`process start` returns its handle at once in tests about other things; a quick job's result in the start is
    tested where it is set (`QUICK_S`)."""
    monkeypatch.setattr(process_tool, "QUICK_S", 0.0)


@pytest.fixture
def make_harness(tmp_path: Path):
    def make(
        script: Script,
        *,
        window: int = 32_768,
        profile: Profile | None = None,
        extensions: list[Extension] | None = None,
        system1: DecisionBackend | None = None,
    ) -> Harness:
        cwd = tmp_path / "work"
        cwd.mkdir(exist_ok=True)
        recorder = Recorder()
        provider = FakeProvider(script)
        deps = EngineDeps(
            system2=binding(provider, window, profile),
            tools=with_extension_tools(load_packs(["base"]), extensions or []),
            hooks=Hooks(extensions or [], recorder),
            journal=Journal(cwd=str(cwd)),
            instructions=FileInstructions(tmp_path / "home", context_files=False).load(cwd),
            decisions=DecisionClient(Filled(system1) if system1 else Quiet(), recorder),
            executor=LocalExecutor(),
            cwd=cwd,
            spill_dir=tmp_path / "spill",
            emit=recorder,
        )
        return Harness(deps=deps, recorder=recorder, provider=provider, cwd=cwd)

    return make
