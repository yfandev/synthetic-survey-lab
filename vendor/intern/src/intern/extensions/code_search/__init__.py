"""Code search: the working directory as a tree System 1 can search, for code. A capability, turned on only by
`[capabilities.code_search] enabled = true` (`session_setup.capability_extensions`), not by extension name.

`decision_code` walks the index (symbols of Python files, sections of Markdown and reStructuredText, keys of config
files, windows of the rest; folders first, then communities of files that import each other), steered also to the
groups whose chunks hold the question's words (`locate/lexical.py`), and returns the places where the question's
subject is defined or would be changed, each with the `read` call that opens it. The
index lives in `.intern/index.json`, built before the first turn when missing or stale in a git work tree, elsewhere
(a home folder, Downloads) only on `decision_code`'s first call, and refreshed after `write`, `edit` or `bash`; it
never calls a model. Needs `intern-agent[code-search]` (numpy, scipy, networkx).

In the terminal UI each `decision_code` call draws its search live right under the call (`view.py`, this
extension's `tui_view`): the tree walked top-down, System 1's score for every node, what the survivor rule kept
and pruned, the parts read, the side-by-side compare and the ranking. The search sends it as `search_trace`
extension events; the result the agent gets is the same.

Where documents (PDFs, Word, spreadsheets, Parquet) outnumber code and text, as in a plain folder of a client's
files, the index would hold none of them: the working directory is not indexed, and the note and `decision_code`
say why instead.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import ClassVar

from intern.locate.index import DOCUMENT_SUFFIXES, default_processors, sample
from intern.locate.search import Questions
from intern.locate.store import DecisionSearch, Locate, SearchArgs
from intern.needs import Needs
from intern.tools.base import ToolContext, ToolOutput, fail
from intern.tools.images import IMAGE_SUFFIXES

DOCUMENTS = "documents (PDFs, Word, spreadsheets, Parquet)"

#: The questions ask for the place where the subject lives (defined, done, or would be changed), not for files
#: that would help: with "would someone need to open this file?", the tests that exercise a feature scored 0.8-0.9
#: and crowded the implementation out. No folder is excluded and nothing ranks last.
CODE = Questions(
    node=(
        "Is the place the question asks about (where its subject is defined, done, or would be changed) inside "
        "this folder or group of files?"
    ),
    file=(
        "Is this file where the question's subject lives (defined, done, or would be changed), rather than a file "
        "that merely uses, references or tests it?"
    ),
    chunk="Is the part shown as [{id}] where the question's subject lives (defined, done, or would be changed)?",
    leaf="Is this text the place the question asks about (where its subject is defined, done, or would be changed)?",
    compare=(
        "Which one of these files is where the question's subject lives (defined, done, or would be changed), "
        "rather than one that merely uses, references or tests it?"
    ),
)


class DecisionCode(DecisionSearch):
    name = "decision_code"
    trace = True
    lexical = True
    description = (
        "Find the places in a repository or folder of code and text that answer a question, ranked, each with the read "
        f"call that opens it; it cannot read {DOCUMENTS}. Use it when you do not know where to look or the exact "
        "words to grep for; grep for exact words you know."
    )

    async def run(self, args: SearchArgs, ctx: ToolContext) -> ToolOutput:
        where = ctx.cwd / args.scope if args.scope else self.store.root or ctx.cwd
        if await asyncio.to_thread(mostly_documents, where):
            return fail(
                f"{args.scope or 'the working directory'} holds mostly {DOCUMENTS}, which {self.name} cannot read",
                f"{self.name} searches code and text only: use a tool that reads documents, or a scope with code.",
            )
        return await super().run(args, ctx)


def mostly_documents(path: Path) -> bool:
    """Whether ``path`` is a document, or a folder whose files (`sample`) are more documents than code and text; an
    image counts as neither."""
    if path.is_file():
        return path.suffix.lower() in DOCUMENT_SUFFIXES
    suffixes = [Path(rel).suffix.lower() for rel in sample(path)]
    documents = sum(s in DOCUMENT_SUFFIXES for s in suffixes)
    return documents > sum(s not in DOCUMENT_SUFFIXES and s not in IMAGE_SUFFIXES for s in suffixes)


def in_git_work_tree(path: Path) -> bool:
    try:
        done = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=path, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0 and done.stdout.strip() == b"true"


class CodeSearch(Locate):
    name: ClassVar[str] = "code-search"
    tui_view: ClassVar[str | None] = "intern.extensions.code_search.view:DecisionCodeView"
    needs: ClassVar[Needs | None] = Needs(
        title="Code search", extra="code-search", modules=("numpy", "networkx", "scipy")
    )
    search_tool = DecisionCode
    processors = staticmethod(default_processors)
    asks: ClassVar[Questions] = CODE
    note: ClassVar[str] = (
        "The working directory is indexed. `decision_code` finds the places where a question's subject is defined "
        "or would be changed (symbols, sections, config keys) when you do not know where to look or the exact "
        "words to grep for, each with the read call that opens it."
    )

    async def pin(self, cwd: Path) -> str | None:
        if await asyncio.to_thread(mostly_documents, cwd):
            return f"The working directory holds mostly {DOCUMENTS}, which `decision_code` cannot read: not indexed."
        if not await asyncio.to_thread(in_git_work_tree, cwd):
            return None  # not a project: nothing walked or written before the agent searches
        return await super().pin(cwd)
