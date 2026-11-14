"""The search engine: processors, the index and its tree under `.intern/`, and the cascade (a scripted System 1
that knows the answers proves the plumbing; the 15-question gold set is tests/data/workspace_questions.json)."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

pytest.importorskip("networkx", reason="needs the code-search extra: uv sync --all-extras")

from intern.cli import main
from intern.decisions import (
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    NoulAnswer,
    NoulQuestion,
    Question,
    System1Unavailable,
)
from intern.events import Decision, ToolCallEnd
from intern.locate import index as wi
from intern.locate import tree
from intern.locate.lexical import Lexical
from intern.locate.processors import Derivative, KeyedProcessor, MarkdownProcessor, PythonProcessor
from intern.locate.search import NONE, TRACE_RANKED, Left, Questions, search, survivors
from intern.locate.trace import PHASES, SearchNode, SearchTrace

REPO = Path(__file__).resolve().parents[1]

#: Any wording does here: the scripted System 1 answers by file membership.
ASKS = Questions(
    node="Is what the question asks about in this folder or group of files?",
    file="Is what the question asks about in this file?",
    chunk="Is what the question asks about in the part shown as [{id}]?",
    leaf="Is this text what the question asks about?",
    compare="Which one of these files is what the question asks about?",
)
QUESTIONS = json.loads((Path(__file__).parent / "data" / "workspace_questions.json").read_text())


def build(root: Path) -> wi.Index:
    return wi.build(root, report=lambda line: None)


def test_python_files_chunk_by_symbol_with_methods_under_their_class(tmp_path: Path):
    source = textwrap.dedent('''
        """Widgets."""

        @decorated
        class Widget(Base):
            """A widget."""

            def spin(self, turns: int) -> None:
                """Spin it."""
                self.turns += turns

        async def make() -> Widget:
            return Widget()
        ''').lstrip()
    (tmp_path / "w.py").write_text(source)
    d = PythonProcessor().process(tmp_path / "w.py", tmp_path / ".intern")
    assert d.summary == "Widgets."
    assert [(c.key, c.kind, c.parent, c.start, c.end) for c in d.chunks] == [
        ("Widget", "class", None, 3, 9),
        ("Widget.spin", "method", "Widget", 7, 9),
        ("make", "function", None, 11, 12),
    ]
    assert d.chunks[0].title == "class Widget(Base)" and d.chunks[0].summary == "A widget."
    assert d.chunks[1].title == "def Widget.spin(self, turns: int) -> None"
    assert d.chunks[2].title == "async def make() -> Widget"


def test_markdown_sections_nest_by_level_and_ignore_fenced_headings(tmp_path: Path):
    (tmp_path / "d.md").write_text("# Top\n\nIntro.\n\n```\n# not a heading\n```\n\n## A\n\ntext\n\n## B\n\nmore\n")
    d = MarkdownProcessor().process(tmp_path / "d.md", tmp_path / ".intern")
    assert d.summary == "Intro."
    assert [(c.key, c.parent, c.start, c.end) for c in d.chunks] == [
        ("Top", None, 1, 15),
        ("A", "Top", 9, 12),
        ("B", "Top", 13, 15),
    ]


def test_rst_titles_are_sections_too(tmp_path: Path):
    (tmp_path / "d.rst").write_text("Title\n=====\n\nBody.\n\nPart\n----\n\nmore\n")
    d = MarkdownProcessor().process(tmp_path / "d.rst", tmp_path / ".intern")
    assert [(c.key, c.parent) for c in d.chunks] == [("Title", None), ("Part", "Title")]
    assert d.summary == "Body."


def test_config_files_chunk_by_top_level_key(tmp_path: Path):
    (tmp_path / "c.toml").write_text("[project]\nname = 'x'\n\n[tool.ruff]\nline-length = 120\n")
    (tmp_path / "c.yaml").write_text("name: ci\non:\n  push:\njobs:\n  test:\n    runs-on: ubuntu\n")
    assert [c.key for c in KeyedProcessor().process(tmp_path / "c.toml", tmp_path).chunks] == ["project", "tool.ruff"]
    assert [c.key for c in KeyedProcessor().process(tmp_path / "c.yaml", tmp_path).chunks] == ["name", "on", "jobs"]


def test_build_writes_the_index_and_refreshes_only_what_changed(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text('"""Alpha."""\n\ndef alpha():\n    return 1\n')
    (tmp_path / "pkg" / "b.py").write_text('"""Beta."""\n\ndef beta():\n    return 2\n')
    (tmp_path / "README.md").write_text("# Demo\n\nA demo.\n")
    (tmp_path / ".ruff_cache").mkdir()
    (tmp_path / ".ruff_cache" / "CACHEDIR.TAG").write_text("Signature: 8a477f597d28d172789f06886806bc55")
    (tmp_path / ".ruff_cache" / "junk.py").write_text("x = 1\n")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")

    index = build(tmp_path)
    assert set(index.files) == {"pkg/a.py", "pkg/b.py", "README.md"}
    assert (tmp_path / ".intern" / "index.json").exists()
    loaded = wi.load(tmp_path / ".intern")
    assert loaded is not None and loaded == index and loaded.home == tmp_path / ".intern"
    root = index.nodes["."]
    assert root.kind == "leaf" and sorted(root.files) == sorted(index.files)  # one group: every file on its card
    assert "  pkg/a.py | Alpha. | [alpha]" in root.card
    alpha = index.chunk("pkg/a.py", "alpha")
    assert alpha and alpha.card == "def alpha()  in pkg/a.py lines 3-4\n  | def alpha():\n  |     return 1"
    assert not wi.stale(tmp_path, index)

    (tmp_path / "pkg" / "b.py").write_text('"""Beta, changed."""\n\ndef gamma():\n    return 3\n')
    (tmp_path / "pkg" / "a.py").unlink()
    (tmp_path / "pkg" / "c.py").write_text("def c(): ...\n")
    assert wi.stale(tmp_path, index)
    again = build(tmp_path)
    assert set(again.files) == {"pkg/b.py", "pkg/c.py", "README.md"}
    assert again.files["README.md"] == index.files["README.md"]
    assert [c.key for c in again.files["pkg/b.py"].chunks] == ["gamma"]
    assert again.files["pkg/b.py"].summary == "Beta, changed."
    assert not wi.stale(tmp_path, again)


def test_in_a_git_work_tree_what_git_ignores_is_not_indexed(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("workspaces/\n*.log\n")
    (tmp_path / "app.py").write_text('"""App."""\n')
    (tmp_path / "notes.md").write_text("# Notes\n")  # untracked, not ignored
    (tmp_path / "run.log").write_text("noise\n")
    (tmp_path / "workspaces" / "client").mkdir(parents=True)
    (tmp_path / "workspaces" / "client" / "dump.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "app.py"], check=True)

    index = build(tmp_path)
    assert set(index.files) == {"app.py", "notes.md"}
    (tmp_path / "workspaces" / "client" / "more.py").write_text("y = 2\n")
    assert not wi.stale(tmp_path, index)  # output a command leaves where git ignores it costs no rebuild


def test_a_folder_with_more_files_than_an_index_holds_fails_the_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wi, "MAX_FILES", 2)
    for name in ("a.py", "b.py", "c.py"):
        (tmp_path / name).write_text("x = 1\n")
    with pytest.raises(wi.TooManyFiles):
        build(tmp_path)


class Pages:
    """A slow processor run in a build's worker processes: one page naming the file; `crash` kills its worker the
    way a native segfault would."""

    name: ClassVar[str] = "pages"
    suffixes: ClassVar[frozenset[str]] = frozenset({".pg"})
    parallel: ClassVar[bool] = True

    def process(self, path: Path, workspace: Path) -> Derivative:
        text = path.read_text()
        if "crash" in text:
            os._exit(11)
        return Derivative(unit="page", units=[f"{path.name}: {text}"], derived=True)


class Stops:
    """Processes every file once, recording which, and stops the build (a ^C) at the file named `stop`."""

    name: ClassVar[str] = "stops"
    suffixes: ClassVar[frozenset[str]] = frozenset({".txt"})
    seen: ClassVar[list[str]] = []

    def process(self, path: Path, workspace: Path) -> Derivative:
        if path.name == "stop.txt" and path.read_text() == "stop":
            raise KeyboardInterrupt
        Stops.seen.append(path.name)
        return Derivative(unit="line", units=[path.read_text()])


def test_an_interrupted_build_resumes_from_what_it_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(wi, "CHECKPOINT_S", 0.0)
    for name in ["a.txt", "b.txt", "stop.txt", "z.txt"]:
        (tmp_path / name).write_text("stop" if name == "stop.txt" else name)
    home = tmp_path / ".intern"
    Stops.seen = []
    with pytest.raises(KeyboardInterrupt):
        wi.build(tmp_path, home, processors=lambda root, home: [Stops()], report=lambda line: None)
    assert Stops.seen == ["a.txt", "b.txt"] and wi.load(home) is None
    (tmp_path / "stop.txt").write_text("go on")
    Stops.seen = []
    index = wi.build(tmp_path, home, processors=lambda root, home: [Stops()], report=lambda line: None)
    assert Stops.seen == ["stop.txt", "z.txt"]  # a and b came from the saved part
    assert list(index.files) == ["a.txt", "b.txt", "stop.txt", "z.txt"]
    assert not (home / wi.PARTIAL_FILE).exists()


def test_a_worker_crash_costs_only_the_file_that_crashed_it(tmp_path: Path):
    for n in range(5):
        (tmp_path / f"p{n}.pg").write_text("crash" if n == 2 else f"page {n}")
    lines: list[str] = []
    index = wi.build(tmp_path, processors=lambda root, home: [Pages()], report=lines.append)
    assert list(index.files) == ["p0.pg", "p1.pg", "p3.pg", "p4.pg"]
    assert index.files["p3.pg"].unit == "page"
    assert (tmp_path / ".intern" / "derived" / "p4.pg.md").read_text() == "--- Page 1 ---\np4.pg: page 4\n"
    assert [line for line in lines if line.startswith("skip")] == [
        "skip p2.pg: RuntimeError: its processing crashed the worker process twice"
    ]


def test_a_refresh_reads_the_imports_of_changed_files_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("from pkg import b\n")
    (tmp_path / "pkg" / "b.py").write_text("x = 1\n")
    index = build(tmp_path)
    assert index.files["pkg/a.py"].imports == ["pkg.b"] and index.files["pkg/b.py"].imports == []

    read: list[str] = []
    import_names = tree.import_names
    monkeypatch.setattr(tree, "import_names", lambda root, path: read.append(path) or import_names(root, path))
    (tmp_path / "pkg" / "b.py").write_text("if True:\n    from . import a\n")
    again = build(tmp_path)
    assert read == ["pkg/b.py"]
    names = {p: e.imports or [] for p, e in again.files.items()}
    assert tree.import_targets(list(again.files), names) == {"pkg/a.py": {"pkg/b.py"}, "pkg/b.py": {"pkg/a.py"}}


def test_empty_files_are_not_indexed(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "placeholder.docx").write_bytes(b"")
    (tmp_path / "placeholder.xlsx").write_bytes(b"")
    assert set(build(tmp_path).files) == {"a.py"}


def test_the_output_keeps_itself_out_of_git_and_leaves_the_rest_of_dot_intern_alone(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / ".intern").mkdir()
    (tmp_path / ".intern" / "settings.toml").write_text("kept = true\n")  # the user's, stays visible to git
    build(tmp_path)
    build(tmp_path)
    assert (tmp_path / ".intern" / ".gitignore").read_text() == ".gitignore\nindex.json\n"
    wi.build(tmp_path, tmp_path / ".intern" / "workspace", report=lambda line: None)  # a second home shares .intern
    assert (tmp_path / ".intern" / ".gitignore").read_text() == ".gitignore\nindex.json\nworkspace/\n"
    status = subprocess.run(["git", "-C", str(tmp_path), "status", "--short", "-uall"], capture_output=True, text=True)
    assert status.stdout == "?? .intern/settings.toml\n?? a.py\n"  # the index and its ignore file are silent


def test_the_index_skips_its_own_output(tmp_path: Path):
    (tmp_path / "x.py").write_text("x = 1\n")
    build(tmp_path)
    (tmp_path / ".intern" / "stray.md").write_text("# not indexed\n")
    assert set(build(tmp_path).files) == {"x.py"}


def test_a_build_reports_progress_on_a_terminal_only():
    tty = io.StringIO()
    tty.isatty = lambda: True  # type: ignore[method-assign]
    bar = wi.Progress("Indexing", stream=tty)
    bar(73, 541)
    bar(541, 541)
    bar.close()
    frames = tty.getvalue().split("\r")[1:]
    assert frames[0].startswith("Indexing:  13%|█████") and "| 73/541 [00:00<" in frames[0] and "file/s]" in frames[0]
    assert frames[1].startswith("Indexing: 100%|" + "█" * 40 + "| 541/541 [") and frames[1].endswith("\n")
    quiet = io.StringIO()
    bar = wi.Progress(stream=quiet)
    bar(1, 2)
    bar.close()
    assert quiet.getvalue() == ""


def test_the_tree_follows_folders_and_merges_small_ones_into_a_sibling(tmp_path: Path):
    for folder, n in [("src", 18), ("src/sub", 5), ("tests", 6), ("docs", 2)]:
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (tmp_path / folder / f"f{i}.py").write_text(f'"""Module {folder} {i}."""\n')
    (tmp_path / "README.md").write_text("# Top\n")
    index = build(tmp_path)
    root = index.nodes["."]
    assert root.kind == "inner" and set(root.children) == {". · [(files), docs, tests]", "src"}
    assert index.nodes["src"].kind == "inner" and set(index.nodes["src"].children) == {"src/sub", "src · [(files)]"}
    small = index.nodes[". · [(files), docs, tests]"]  # README, docs (2) and tests (6) are too small to stand alone
    assert small.kind == "leaf" and len(small.files) == 9
    assert all(len(n.files) <= tree.LEAF_FILES for n in index.leaves)


def test_near_identical_files_are_cut_into_balanced_groups_not_singletons(tmp_path: Path):
    (tmp_path / "data").mkdir()
    for i in range(60):
        (tmp_path / "data" / f"evaluation_{i}.txt").write_text(
            f"Product Lifecycle Assessment\nProduct ID: PLM-{i:04d}\nStage: growth\n"
        )
    index = build(tmp_path)
    sizes = sorted(len(n.files) for n in index.leaves)
    assert sizes[0] >= tree.LEAF_MIN and sizes[-1] <= tree.LEAF_FILES and sum(sizes) == 60, sizes


def test_a_folder_of_numbered_files_is_cut_in_number_order_with_spans_for_labels():
    months = [f"bulletin_{y}_{m:02d}.pdf" for y in range(1939, 1951) for m in range(1, 13)]  # 144 files
    nodes = tree.build([*reversed(months), "notes.md"], {})
    root = nodes["."]
    leaves = [n for n in nodes.values() if n.kind == "leaf" and "notes.md" not in n.files]
    assert all(len(n.files) <= tree.LEAF_FILES for n in leaves)
    runs = sorted(sorted(n.files) for n in leaves)
    assert [f for run in runs for f in run] == months  # each leaf a contiguous run of months
    series_child = next(nodes[k] for k in root.children if "bulletin" in nodes[k].label)
    assert series_child.label == ". · [(bulletin_1939_01 … 1950_12)]"
    first = nodes[series_child.children[0]]
    assert first.label.startswith(". · [(bulletin_1939_01 … ") and len(series_child.children) <= tree.FAN_MAX


def test_files_alike_only_in_kind_are_not_a_series():
    names = [f"{n:04d}_{w}.py" for n, w in enumerate(["initial", "add_user", "rename", "drop_x"] * 8)]
    members, rest = tree.series(names)
    assert members == [] and rest == names


def test_inner_nodes_get_cards_naming_their_subgroups_and_central_files(tmp_path: Path):
    for folder, words in [
        ("app/models", "model field query"),
        ("app/views", "view request response"),
        ("app/forms", "form widget input"),
        ("docs", "guide"),
    ]:
        (tmp_path / folder).mkdir(parents=True)
        for i in range(8):
            (tmp_path / folder / f"f{i}.py").write_text(
                f'"""{words} {i}."""\n\ndef {words.split()[0]}_{i}():\n    pass\n'
            )
    (tmp_path / "setup.py").write_text('"""Setup."""\n')
    index = build(tmp_path)
    assert set(index.nodes) == {".", "app", "app/models", "app/views", "app/forms", ". · [(files), docs]"}
    app = index.nodes["app"]
    assert app.kind == "inner" and len(app.files) == 24 and app.children == ["app/forms", "app/models", "app/views"]
    lines = app.card.splitlines()
    assert lines[0] == "folder app  (24 files in 3 subgroups)" and lines[1].startswith("keywords: ")
    assert set(lines[1].removeprefix("keywords: ").split(", ")) >= {"model", "view", "form"}  # balanced over children
    assert lines[2] == "subgroups:"
    assert (
        lines[3].startswith("  - app/forms (8 files): ")
        and " | f0.py (form_0); f1.py (form_1); f2.py (form_2)" in lines[3]
    )
    assert set(lines[3].split(": ", 1)[1].split(" | ")[0].split(", ")) >= {"form", "widget", "input"}  # its own words
    assert lines[6] == "most central files:" and lines[7].startswith("  ") and " | [" in lines[7]  # a member line
    assert "  - app (24 files): " in index.nodes["."].card


def test_the_root_card_says_what_the_folders_under_it_are_for(tmp_path: Path):
    # audit-app: the map pinned for the agent listed `str, async, worker` and NDI tests; nothing said audit.
    for name in ("audit", "billing", "web"):
        (tmp_path / "services" / name).mkdir(parents=True)
    for i in range(12):
        (tmp_path / "services" / "audit" / f"m{i}.py").write_text(f"def handle_audit_{i}():\n    pass\n")
        (tmp_path / "services" / "billing" / f"m{i}.py").write_text(f"def handle_billing_{i}():\n    pass\n")
        page = f"import {{ api }} from './api';\nexport const p{i} = 1;\n"  # a first line, not a description
        (tmp_path / "services" / "web" / f"page{i}.ts").write_text(page)
    front = "---\ntitle: audit (NAVI)\nsummary: The financial audit app: task graph and working papers.\n---\n"
    (tmp_path / "services" / "audit" / "README.md").write_text(front + "\n# Audit\n\nLong text.\n")
    (tmp_path / "services" / "billing" / "pyproject.toml").write_text(
        '[project]\nname = "billing"\ndescription = "Invoice billing from contracts"\n'
    )
    index = build(tmp_path)
    root = index.nodes["."]  # services/, with no description of its own: the described folders under it
    assert root.label == "services" and (purpose := root.card.splitlines()[1]).startswith("purpose: ")
    assert "audit — The financial audit app: task graph and working papers." in purpose  # its README's front matter
    assert "billing — Invoice billing from contracts" in purpose  # its package's description
    assert "import" not in root.card.split("most central files:")[0]
    judged = [n.card for k, n in index.nodes.items() if k != "."]  # the cards System 1 judges stay as they were
    assert judged and not any("purpose:" in card for card in judged)


def test_the_root_card_does_not_take_a_doc_page_s_first_paragraph_for_a_purpose(tmp_path: Path):
    # docs/ref/models read "Model API reference" while django/db/models had no docstrings.
    for i in range(12):
        (tmp_path / "docs").mkdir(exist_ok=True)
        (tmp_path / "docs" / f"page{i}.md").write_text(f"# Page {i}\n\nModel field reference, part {i}.\n")
        (tmp_path / "models").mkdir(exist_ok=True)
        (tmp_path / "models" / f"m{i}.py").write_text(f"def field_{i}():\n    pass\n")
    card = build(tmp_path).nodes["."].card.split("most central files:")[0]
    assert "Model field reference" not in card


def test_a_flat_folder_too_large_for_one_card_is_cut_where_files_are_alike(tmp_path: Path):
    topics = {
        "invoice": "invoice total due vat customer payment",
        "contract": "contract party clause term signature",
        "report": "report quarter revenue growth margin",
    }
    for topic, words in topics.items():
        for i in range(20):
            (tmp_path / f"{topic}-{i}.txt").write_text(f"{words} {words} item {i}\n" * 3)
    index = build(tmp_path)
    assert all(len(n.files) <= tree.LEAF_FILES for n in index.leaves)
    assert len(index.leaves) >= 3
    for leaf in index.leaves:
        kinds = {m.split("-")[0] for m in leaf.files}
        assert len(kinds) == 1, (leaf.key, kinds)
        assert leaf.key.startswith(". · (files") or leaf.key.startswith(". · [(files")


def test_index_command_builds_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    (tmp_path / "a.py").write_text("def a(): ...\n")
    assert main(["index", str(tmp_path)]) == 0
    assert capsys.readouterr().out.startswith("1 files, 1 chunks, 0 tables, 1 groups -> ")
    assert (tmp_path / ".intern" / "index.json").exists()
    assert main(["index", str(tmp_path / "missing")]) == 2


class Gold:
    """A System 1 that knows the answer. A node's card scores high when the node holds a gold file (looked up in
    the index, as a judge would read it off the card), a file card when it is a gold file, a part when its label
    names the gold symbol or a class holding it, leaf text when it names the symbol, and the side-by-side choice
    picks a gold file. What this proves is the cascade: that a right answer at every level reaches the result,
    through the tree the index built."""

    name = "gold"

    def __init__(self, index: wi.Index, files: list[str], symbols: list[str]) -> None:
        self.index = index
        self.files = files
        self.symbols = symbols
        self.calls = 0

    @staticmethod
    def _hit(marks: list[str], text: str) -> bool:
        return any(re.search(rf"\b{re.escape(mark)}\b", text) for mark in marks)

    def _option(self, label: str) -> bool:
        """The option is the gold symbol or a class holding it."""
        m = re.match(r"(?:async )?(?:def|class) ([\w.]+)", label)
        if not m:
            return self._hit([s.split(".")[-1] for s in self.symbols], label)
        return any(s == m.group(1) or s.startswith(m.group(1) + ".") for s in self.symbols)

    def node_of(self, card: str) -> tree.Node | None:
        """The node a folder or group card is about, by the label on its first line."""
        label = re.sub(r"  \(\d+ files.*$", "", card.splitlines()[0].split(" ", 1)[1])
        return next((n for n in self.index.nodes.values() if n.label == label), None)

    def _node_holds_gold(self, card: str) -> bool:
        node = self.node_of(card)
        return node is not None and any(f in node.files for f in self.files)

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        self.calls += 1
        answers: dict = {}
        what, body = state.split("\n\n", 1)[1].split(":\n", 1)
        for name, q in questions.items():
            if isinstance(q, NoulQuestion):
                if what.startswith("Candidate folder"):
                    hit = self._node_holds_gold(body)
                elif what.startswith("Candidate file"):
                    hit = self._hit(self.files, body.splitlines()[0])
                elif what.startswith("Parts of"):
                    card = body.split(f"[{name}] ", 1)[1].split("\n\n[", 1)[0]
                    hit = self._option(card.splitlines()[0])
                else:  # leaf text: the header names the chunk
                    hit = self._option(what.split(", ", 1)[1]) if ", " in what else False
                answers[name] = NoulAnswer(noul=0.9 if hit else 0.05)
            elif isinstance(q, ChoiceQuestion):
                hits = [i for i, label in q.criteria.items() if i != NONE and label in self.files]
                probabilities = {i: (1.0 / len(hits) if i in hits else 0.0) for i in q.criteria}
                if not hits:
                    probabilities[NONE] = 1.0
                best = max(probabilities, key=lambda i: probabilities[i])
                answers[name] = ChoiceAnswer(choice=best, probabilities=probabilities)
        return Answers(answers=answers, cost_usd=0.0001)


@pytest.fixture(scope="module")
def repo_index(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, wi.Index]:
    root = tmp_path_factory.mktemp("repo")
    shutil.copytree(REPO / "src" / "intern", root / "src" / "intern")
    return root, wi.build(root, report=lambda line: None)


@pytest.mark.parametrize("case", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
async def test_the_cascade_carries_a_right_answer_at_every_level_to_the_result(repo_index, case: dict):
    root, index = repo_index
    gold = Gold(index, case["files"], case["symbols"])
    result = await search(index, root, case["question"], DecisionClient(gold, lambda e: None), ASKS)
    files = [p for p, s in result.files if s > 0][:3]
    assert any(f in files for f in case["files"]), (files, case["files"])
    keys = [p.key for p in result.places if p.score > 0][:5]
    assert any(k in case["symbols"] for k in keys), (keys, case["symbols"])
    assert gold.calls <= 90  # one card a state: the budgets bound it (System 1 measured 45 asks a question here)


async def test_scope_limits_the_search_to_a_folder(repo_index):
    root, index = repo_index
    gold = Gold(index, ["src/intern/tools/base.py", "src/intern/providers/repair.py"], [])
    client = DecisionClient(gold, lambda e: None)
    result = await search(index, root, "where are tool calls repaired", client, ASKS, scope="src/intern/tools")
    assert result.files and all(p.startswith("src/intern/tools/") for p, _ in result.files)
    assert (await search(index, root, "anything", client, ASKS, scope="nowhere")).files == []


async def test_an_ask_that_keeps_failing_stops_the_search_instead_of_reading_as_a_no(repo_index):
    root, index = repo_index

    class Down:
        name = "down"

        async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
            raise ConnectionError("refused")

    events: list = []
    with pytest.raises(System1Unavailable, match="refused"):
        await search(index, root, "where are tool calls repaired", DecisionClient(Down(), events.append), ASKS)
    assert events and all(isinstance(e, Decision) and not e.ok for e in events)


async def test_a_large_workspace_is_entered_by_its_folders_and_the_wrong_ones_are_never_opened(tmp_path: Path):
    for top in "abc":
        for n in range(12):
            (tmp_path / top / f"part{n}").mkdir(parents=True)
            for i in range(5):
                (tmp_path / top / f"part{n}" / f"m{i}.py").write_text(f'"""Topic {top} part {n} item {i}."""\n')
    (tmp_path / "b" / "part7" / "m2.py").write_text('"""Topic b part 7."""\n\ndef needle():\n    return 1\n')
    index = wi.build(tmp_path, report=lambda line: None)
    assert index.nodes["."].kind == "inner" and set(index.nodes["."].children) == {"a", "b", "c"}
    states: list[str] = []

    class Knows(Gold):
        async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
            states.append(state)
            return await super().decide(state, questions)

    gold = Knows(index, ["b/part7/m2.py"], ["needle"])
    result = await search(index, tmp_path, "where is the needle", DecisionClient(gold, lambda e: None), ASKS)
    assert result.files[0][0] == "b/part7/m2.py" and result.places[0].key == "needle"
    opened = {s.split("\n\n", 1)[1].splitlines()[1].split("  (")[0] for s in states if "Candidate folder" in s}
    assert {"folder a", "folder b", "folder c"} <= opened  # the root offered every top folder
    assert not any(o.startswith("group c/") for o in opened)  # c, the least likely, was never entered
    assert len(states) <= 80  # 3 top folders, a and b entered (KEEP_MIN), b's parts, files, chunks, leaf, compare


def three_tops(root: Path) -> wi.Index:
    """Three top folders of twelve parts of five files; the needle is in b/part7/m2.py."""
    for top in "abc":
        for n in range(12):
            (root / top / f"part{n}").mkdir(parents=True)
            for i in range(5):
                (root / top / f"part{n}" / f"m{i}.py").write_text(f'"""Topic {top} part {n} item {i}."""\n')
    (root / "b" / "part7" / "m2.py").write_text('"""Topic b part 7."""\n\ndef needle():\n    return 1\n')
    return wi.build(root, report=lambda line: None)


class Folders(Gold):
    """Gold on files and parts; a folder or group card is scored by ``folder(node, gold_answer)`` instead, as a
    judge does when a card does not name what the question is about. Records the folders it was asked about."""

    def __init__(
        self, index: wi.Index, folder: Callable[[tree.Node, float], float], other: float | None = None
    ) -> None:
        super().__init__(index, ["b/part7/m2.py"], ["needle"])
        self.folder = folder
        self.other = other
        """Every other judgment (files, parts, text), when set."""
        self.asked: list[tree.Node] = []

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        answers = await super().decide(state, questions)
        what, body = state.split("\n\n", 1)[1].split(":\n", 1)
        node = self.node_of(body) if what.startswith("Candidate folder") else None
        if node is not None:
            self.asked.append(node)
            gold = answers.answers["q"]
            assert isinstance(gold, NoulAnswer)
            return Answers(answers={"q": NoulAnswer(noul=self.folder(node, gold.noul))}, cost_usd=answers.cost_usd)
        if self.other is None:
            return answers
        other = {n: NoulAnswer(noul=self.other) if isinstance(a, NoulAnswer) else a for n, a in answers.answers.items()}
        return Answers(answers=other, cost_usd=answers.cost_usd)


async def test_a_scope_starts_the_walk_at_its_folder_whatever_the_folders_above_it_score(tmp_path: Path):
    # audit-app: `. · [services, shared]` scored 0.03 for a revenue question scoped to services/main_service, and
    # the search ended after that one ask, never judging anything in scope.
    index = three_tops(tmp_path)
    scope = {f for f in index.files if f.startswith("b/")}
    judge = Folders(index, lambda node, gold: 0.01 if scope <= set(node.files) else gold)
    client = DecisionClient(judge, lambda e: None)
    result = await search(index, tmp_path, "where is the needle", client, ASKS, scope="b")
    assert result.files[0][0] == "b/part7/m2.py" and result.places[0].key == "needle"
    assert judge.asked and not [n.label for n in judge.asked if scope <= set(n.files)]  # nothing above b judged
    assert result.walk is not None and result.walk.start == "b"


async def test_when_no_folder_reaches_the_floor_the_best_are_entered_and_files_judged(tmp_path: Path):
    index = three_tops(tmp_path)
    judge = Folders(index, lambda node, gold: 0.03 if gold > 0.5 else 0.02)  # every card under MIN_PROB
    result = await search(index, tmp_path, "where is the needle", DecisionClient(judge, lambda e: None), ASKS)
    assert result.files[0][0] == "b/part7/m2.py" and result.places[0].key == "needle"
    assert result.walk is not None and result.walk.forced


async def test_an_empty_search_reports_how_far_its_walk_got(tmp_path: Path):
    index = three_tops(tmp_path)
    folders = {"c": 0.4, "c/part3": 0.4, "c/part5": 0.3, "c/part8": 0.09}  # c/part8 falls under the gap rule
    judge = Folders(index, lambda node, gold: folders.get(node.label, 0.0), other=0.0)  # every file says no
    client = DecisionClient(judge, lambda e: None)
    result = await search(index, tmp_path, "where is the needle", client, ASKS)
    assert not [p for p, s in result.files if s > 0]
    walk = result.walk
    assert walk is not None and walk.start == "." and walk.in_scope == len(index.files) and not walk.forced
    assert walk.files == 10 and walk.depth == 2  # c entered, and in it c/part3 and c/part5 down to their files
    assert walk.left == Left(label="c/part8", score=0.09)  # the best folder the walk did not enter
    nowhere = await search(index, tmp_path, "where is the needle", client, ASKS, scope="d")
    assert nowhere.walk is not None and nowhere.walk.in_scope == 0


async def test_the_question_s_words_steer_the_walk_to_a_file_no_card_on_the_way_points_to(tmp_path: Path):
    # audit-app: the revenue templates hold the question's words, but every card above them scored under the others.
    index = three_tops(tmp_path)
    above = {n.key for n in index.nodes.values() if "b/part7/m2.py" in n.files}
    judge = Folders(index, lambda node, gold: 0.01 if node.key in above else 0.03)  # the cards lead away from b
    plain = await search(index, tmp_path, "where is the needle", DecisionClient(judge, lambda e: None), ASKS)
    assert "b/part7/m2.py" not in [p for p, s in plain.files if s > 0]
    steered = await search(
        index, tmp_path, "where is the needle", DecisionClient(judge, lambda e: None), ASKS, lexical=True
    )
    assert steered.files[0][0] == "b/part7/m2.py" and steered.places[0].key == "needle"  # judged, then ranked
    assert steered.walk is not None and steered.walk.seeded >= 1


def test_the_lexical_table_follows_the_files_and_keeps_out_of_git(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("def alpha():\n    return 1\n")
    (tmp_path / "b.py").write_text("def beta():\n    return 2\n")
    words = Lexical(build(tmp_path), tmp_path)
    words.update()
    assert [h.path for h in words.hits("where is alpha", None, 5)] == ["a.py"]
    status = subprocess.run(["git", "-C", str(tmp_path), "status", "--short", "-uall"], capture_output=True, text=True)
    assert status.stdout == "?? a.py\n?? b.py\n"  # neither the table nor the index shows
    (tmp_path / "a.py").write_text("def gammaRay():\n    return 3\n")  # edited: its old word gone, a new one in
    (tmp_path / "b.py").unlink()
    words = Lexical(build(tmp_path), tmp_path)
    words.update()
    assert words.hits("where is alpha", None, 5) == [] and words.hits("where is beta", None, 5) == []
    assert [h.path for h in words.hits("where is the gamma ray", None, 5)] == ["a.py"]  # split from gammaRay


def test_a_scoped_lexical_search_holds_its_folder_not_a_pattern_of_it(tmp_path: Path):
    """`_` in a folder's name is a character, not SQL LIKE's any-one-character: `lib_a` holds none of `libXa`."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "lib_a").mkdir()
    (tmp_path / "lib_a" / "ledger.py").write_text("def reconcile():\n    return 'ledger'\n")
    (tmp_path / "libXa").mkdir()
    for n in range(3):
        (tmp_path / "libXa" / f"ledger{n}.py").write_text(f"def reconcile_ledger_{n}():\n    return 'ledger'\n")
    words = Lexical(build(tmp_path), tmp_path)
    words.update()
    assert [h.path for h in words.hits("reconcile the ledger", "lib_a", 2)] == ["lib_a/ledger.py"]
    assert len(words.hits("reconcile the ledger", None, 10)) == 4


def test_survivors_follow_the_gap_rule():
    assert survivors([("a", 0.8), ("b", 0.3), ("c", 0.1)], 0.25, 8) == [("a", 0.8), ("b", 0.3)]
    assert survivors([("a", 0.8), ("b", 0.1)], 0.25, 8) == [("a", 0.8), ("b", 0.1)]  # KEEP_MIN
    assert survivors([("a", 0.8), ("b", 0.7), ("c", 0.6)], 0.25, 2) == [("a", 0.8), ("b", 0.7)]  # the cap
    assert survivors([], 0.25, 8) == []


async def test_an_extension_whose_index_the_agent_builds_waits_for_the_index_tool(tmp_path: Path, make_harness):
    from intern.locate.store import IndexTool, Locate
    from intern.providers.fake import Turn
    from intern.tools.base import Tool

    from .support import Needle

    class Docs(Locate):
        name = "docs"
        folder = ".intern/docs"
        auto = False
        processors = staticmethod(wi.default_processors)
        asks = ASKS
        note = "docs are searchable once indexed"

        def tools(self) -> list[Tool]:
            return [IndexTool(self.store), *super().tools()]

    harness = make_harness(
        [
            Turn(tool_calls=[("decision_search", {"question": "where is the needle"})]),
            Turn(tool_calls=[("index", {"path": "papers"})]),
            Turn(tool_calls=[("decision_search", {"question": "where is the needle"})]),
            Turn(text="done"),
        ],
        extensions=[Docs()],
        system1=Needle("needle"),
    )
    (harness.cwd / "papers").mkdir()
    (harness.cwd / "papers" / "a.md").write_text("# A\n\nThe needle is here.\n")
    (harness.cwd / "code.py").write_text("x = 1\n")
    await harness.run("find the needle")
    before, built, after = [e for e in harness.recorder.of(ToolCallEnd)][:3]
    assert before.status == "error" and "no index under ." in before.content and "index tool" in before.content
    assert built.content == "Indexed papers: 1 files, 1 chunks, 0 tables -> papers/.intern/docs"
    assert after.content.splitlines()[1:3] == ["1. papers/a.md  A (1-3)  [1.00]", "   read papers/a.md"]
    assert (harness.cwd / "papers" / ".intern" / "docs" / "index.json").exists()
    assert not (harness.cwd / ".intern").exists()  # nothing was built at the root on its own
    heads = [r.messages[1].content for r in harness.provider.requests]
    assert "docs are searchable once indexed" in heads[0] and "group ." not in heads[0]  # a note, no map yet


async def test_a_traced_search_reports_each_node_after_its_parent_and_ranks_exactly_as_without(repo_index):
    root, index = repo_index
    case = QUESTIONS[0]
    plain = await search(
        index, root, case["question"], DecisionClient(Gold(index, case["files"], case["symbols"]), lambda e: None), ASKS
    )
    traces: list[SearchTrace] = []
    gold = Gold(index, case["files"], case["symbols"])
    traced = await search(
        index, root, case["question"], DecisionClient(gold, lambda e: None), ASKS, trace=traces.append
    )
    assert traced == plain  # the trace only watches
    # One search, its phases in order, the last trace the only finished one, with the ranking the tool returns.
    assert len({t.search for t in traces}) == 1 and traces[-1].finished and not any(t.finished for t in traces[:-1])
    phases = [PHASES.index(t.phase) for t in traces]
    assert phases == sorted(phases) and traces[-1].phase == "done"
    assert [r.path for r in traces[-1].ranking] == [p for p, s in traced.files if s > 0][:TRACE_RANKED]
    # Every node arrives after its parent, or with it; the latest state of each is a settled one.
    seen: dict[str, SearchNode] = {}
    for t in traces:
        for node in t.nodes:
            assert node.parent is None or node.parent in seen or node.parent in {n.id for n in t.nodes}
            seen[node.id] = node
    assert not [n for n in seen.values() if n.state in ("asking", "reading")]
    assert [n.kind for n in seen.values() if n.parent is None] == ["root"]
    files = {n.label: n for n in seen.values() if n.kind == "file"}
    assert set(files) == {p for p, _ in traced.files}  # every file judged, nothing else
    assert files[traced.files[0][0]].state == "chosen"
    assert any(n.kind == "part" and n.read is not None for n in seen.values())
    # The counts are this search's: each state System 1 was sent (a repeat answered from the cache included).
    last = traces[-1]
    assert last.asks >= gold.calls > 0 and last.judged >= last.asks
    assert last.files == len(files) <= last.files_budget and len(last.votes) == 3


async def test_a_trace_sink_that_raises_leaves_the_search_and_its_result_alone(repo_index):
    root, index = repo_index
    case = QUESTIONS[0]
    plain = await search(
        index, root, case["question"], DecisionClient(Gold(index, case["files"], case["symbols"]), lambda e: None), ASKS
    )
    calls: list[SearchTrace] = []

    def broken(trace: SearchTrace) -> None:
        calls.append(trace)
        raise RuntimeError("the view is gone")

    gold = Gold(index, case["files"], case["symbols"])
    traced = await search(index, root, case["question"], DecisionClient(gold, lambda e: None), ASKS, trace=broken)
    assert traced == plain and len(calls) == 1  # tried once, then no more traces
