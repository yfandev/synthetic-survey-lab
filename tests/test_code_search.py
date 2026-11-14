"""The code-search extension: `decision_code` over a repository, the index built before the run and refreshed after an
edit, the pinned note and map before the task. The engine's own tests are tests/test_locate.py."""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest

from intern.capability_settings import CapabilitySettings
from intern.decisions import Answers, Question
from intern.events import Indexing, RunEnd, ToolCallEnd
from intern.extensions.code_search import CodeSearch
from intern.locate import index as wi
from intern.models.registry import ConfigError
from intern.providers.fake import Turn
from intern.session_setup import capability_extensions

from .support import Needle

pytest.importorskip("networkx", reason="needs the code-search extra: uv sync --all-extras")


def test_the_capability_setting_turns_code_search_on_and_a_missing_extra_fails_setup(monkeypatch: pytest.MonkeyPatch):
    on = CapabilitySettings.model_validate({"code_search": {"enabled": True}})
    assert capability_extensions(CapabilitySettings()) == []
    assert [type(e) for e in capability_extensions(on)] == [CodeSearch]
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "networkx" else find_spec(name))
    with pytest.raises(ConfigError, match=r"install intern-agent\[code-search\]"):
        capability_extensions(on)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The working directory (make_harness's), a git repository: where the index is built before the first turn."""
    work = tmp_path / "work"
    work.mkdir()
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    return work


@pytest.mark.usefixtures("repo")
async def test_the_index_is_built_before_the_run_and_refreshed_after_an_edit(make_harness):
    harness = make_harness(
        [
            Turn(tool_calls=[("decision_code", {"question": "where is the alpha function"})]),
            Turn(tool_calls=[("write", {"path": "pkg/b.py", "content": "def beta_feature():\n    return 2\n"})]),
            Turn(tool_calls=[("decision_code", {"question": "where is the beta feature"})]),
            Turn(text="done"),
        ],
        extensions=[CodeSearch()],
        system1=Needle("alpha", "beta"),
    )
    (harness.cwd / "pkg").mkdir()
    (harness.cwd / "pkg" / "a.py").write_text('"""Alpha."""\n\ndef alpha():\n    return 1\n')
    assert not (harness.cwd / ".intern").exists()
    await harness.run("find the alpha and beta functions")
    # Each build (before the run, and the refresh after the write) reports itself on the run's event stream:
    # progress a few times a second, then once finished with its duration; nothing goes to stderr.
    builds = harness.recorder.of(Indexing)
    assert builds and all(b.name == "code-search" and b.root == str(harness.cwd.resolve()) for b in builds)
    finished = [b for b in builds if b.finished]
    assert len(finished) >= 2 and all(b.done == b.total and b.duration_ms is not None for b in finished)
    assert all(not b.finished for b in builds[: builds.index(finished[0])])
    first, written, second = [e for e in harness.recorder.of(ToolCallEnd)][:3]
    assert first.content.splitlines()[1:3] == ["1. pkg/a.py  def alpha() (3-4)  [1.00]", "   read pkg/a.py"]
    assert written.status == "ok"
    assert second.content.splitlines()[1] == "1. pkg/b.py  def beta_feature() (1-2)  [1.00]"
    index = wi.load(harness.cwd / ".intern")
    assert index and set(index.files) == {"pkg/a.py", "pkg/b.py"}
    assert [t.name for t in harness.provider.requests[0].tools] == [
        "read",
        "write",
        "edit",
        "bash",
        "grep",
        "find",
        "process",
        "decision_code",
    ]
    heads = [r.messages[1].content for r in harness.provider.requests]
    assert all("`decision_code` finds the places" in h and "group .  (1 files)" in h for h in heads)
    assert all(
        h.index("group .  (") < h.index("find the alpha and beta functions") for h in heads
    )  # the task stays last
    assert all("read long files in pages" in r.messages[0].content for r in harness.provider.requests)
    assert len({h.split("group .", 1)[1] for h in heads}) == 1  # the map is fixed for the run; b.py does not redraw it


@pytest.fixture
def other(tmp_path: Path) -> Path:
    """A repository beside the working directory."""
    other = tmp_path / "other"
    (other / "lib").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(other)], check=True)
    (other / "lib" / "gamma.py").write_text('"""Gamma."""\n\ndef gamma():\n    return 3\n')
    (other / "notes.md").write_text("# Gamma notes\n\nAbout gamma.\n")
    return other.resolve()  # as the agent's read opens it from the working directory


@pytest.mark.usefixtures("repo")
async def test_a_scope_outside_the_working_directory_searches_that_repository_and_keeps_its_index(
    make_harness, other: Path
):
    harness = make_harness(
        [
            Turn(tool_calls=[("decision_code", {"question": "where is the gamma function", "scope": "../other/lib"})]),
            Turn(tool_calls=[("decision_code", {"question": "where is gamma", "scope": str(other / "gone")})]),
            Turn(text="done"),
        ],
        extensions=[CodeSearch()],
        system1=Needle("gamma"),
    )
    (harness.cwd / "a.py").write_text("def alpha():\n    return 1\n")
    await harness.run("find gamma in the other repository")
    found, gone = harness.recorder.of(ToolCallEnd)[:2]
    lib = f"{other}/lib/gamma.py"
    assert found.content.splitlines()[1:3] == [f"1. {lib}  def gamma() (3-4)  [1.00]", f"   read {lib}"]
    assert "notes.md" not in found.content  # the scope holds inside the other repository
    assert gone.status == "error" and "does not exist" in gone.content
    kept = wi.load(other / ".intern")
    assert kept and set(kept.files) == {"lib/gamma.py", "notes.md"}  # the whole repository, kept for next time
    own = wi.load(harness.cwd / ".intern")
    assert own and set(own.files) == {"a.py"}


async def test_an_empty_search_says_how_far_it_got_and_an_empty_or_missing_scope_is_named(make_harness):
    harness = make_harness(
        [
            Turn(tool_calls=[("decision_code", {"question": "where is zeta"})]),
            Turn(tool_calls=[("decision_code", {"question": "where is zeta", "scope": "assets"})]),
            Turn(tool_calls=[("decision_code", {"question": "where is zeta", "scope": "nowhere"})]),
            Turn(text="done"),
        ],
        extensions=[CodeSearch()],
        system1=Needle("zeta"),  # no card names zeta: every judgment is 0
    )
    (harness.cwd / "a.py").write_text("def alpha():\n    return 1\n")
    (harness.cwd / "assets").mkdir()
    (harness.cwd / "assets" / "empty.txt").write_text("")  # an empty file is not indexed
    await harness.run("find zeta")
    walked, empty, missing = harness.recorder.of(ToolCallEnd)[:3]
    assert walked.status == "ok" and "Nothing" in walked.content and re.search(r"\b1 file\b", walked.content)
    assert empty.status == "ok" and "holds no file" in empty.content
    assert missing.status == "error" and "does not exist" in missing.content


class Broken:
    """A System 1 backend whose every ask fails, as a wrong model name or a dead endpoint does."""

    name = "broken"

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        raise RuntimeError("HTTP 400 Bad Request: model not found")


async def test_a_failing_system_1_stops_the_run_instead_of_reading_as_nothing_found(make_harness):
    harness = make_harness(
        [Turn(tool_calls=[("decision_code", {"question": "where is the alpha function"})]), Turn(text="done")],
        extensions=[CodeSearch()],
        system1=Broken(),
    )
    (harness.cwd / "a.py").write_text("def alpha():\n    return 1\n")
    outcome = await harness.run("find alpha")
    assert outcome.reason == "system1_unavailable" and "400 Bad Request" in (outcome.detail or "")
    assert not harness.recorder.of(ToolCallEnd)
    end = harness.recorder.of(RunEnd)[0]
    assert end.system1.asks == 0 and end.system1.failed == 1


@pytest.mark.usefixtures("repo")
async def test_a_folder_past_max_files_is_not_reported_as_indexed_nor_walked_again_each_run(
    make_harness, monkeypatch: pytest.MonkeyPatch
):
    # A repository too large to index: no "Indexed 0 files" line, and not seconds of walking before every run.
    monkeypatch.setattr(wi, "MAX_FILES", 2)
    builds: list[Path] = []
    build = wi.build
    monkeypatch.setattr(wi, "build", lambda root, *a, **k: builds.append(root) or build(root, *a, **k))
    harness = make_harness([Turn(text="hi"), Turn(text="hi again")], extensions=[CodeSearch()])
    for name in ("a.py", "b.py", "c.py"):
        (harness.cwd / name).write_text("x = 1\n")
    await harness.run("hello")
    await harness.run("hello again")
    assert not harness.recorder.of(Indexing)
    assert len(builds) == 1


async def test_outside_a_git_repository_nothing_is_indexed_until_the_agent_searches(make_harness):
    # A home folder or Downloads: a message that needs no search walks and writes nothing there.
    harness = make_harness(
        [
            Turn(text="hi"),
            Turn(tool_calls=[("decision_code", {"question": "where is the alpha function"})]),
            Turn(text="done"),
        ],
        extensions=[CodeSearch()],
        system1=Needle("alpha"),
    )
    (harness.cwd / "a.py").write_text("def alpha():\n    return 1\n")
    await harness.run("hello")
    assert not harness.recorder.of(Indexing) and not (harness.cwd / ".intern").exists()
    await harness.run("find alpha")
    found = harness.recorder.of(ToolCallEnd)[0]
    assert found.content.splitlines()[1:3] == ["1. a.py  def alpha() (1-2)  [1.00]", "   read a.py"]
    assert harness.recorder.of(Indexing)[-1].finished


async def test_a_folder_of_documents_outside_git_is_not_indexed_and_the_agent_is_told_why(make_harness):
    # A client's folder (no git): what the code index holds of it is its one text file, not the PDFs and workbooks.
    harness = make_harness(
        [
            Turn(tool_calls=[("decision_code", {"question": "what was the 2024 revenue"})]),
            Turn(tool_calls=[("decision_code", {"question": "where is the alpha function", "scope": "scripts"})]),
            Turn(text="done"),
        ],
        extensions=[CodeSearch()],
        system1=Needle("alpha"),
    )
    (harness.cwd / "2024").mkdir()
    for name in ("report.pdf", "ledger.xlsx", "minutes.docx"):
        (harness.cwd / "2024" / name).write_bytes(b"\x00binary")
    (harness.cwd / "scripts").mkdir()
    (harness.cwd / "scripts" / "a.py").write_text("def alpha():\n    return 1\n")
    await harness.run("what was the 2024 revenue")
    head = harness.provider.requests[0].messages[1].content
    assert "cannot read" in head and "group ." not in head  # no map of an index without the documents
    refused, scoped = harness.recorder.of(ToolCallEnd)[:2]
    assert refused.status == "error" and "cannot read" in refused.content
    events = harness.recorder.events
    assert all(events.index(b) > events.index(refused) for b in harness.recorder.of(Indexing))  # none before the run
    assert scoped.status == "ok" and "scripts/a.py" in scoped.content  # a folder of code in it is still searched
