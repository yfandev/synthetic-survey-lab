from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path

from intern import names
from intern.config import Bootstrap, Options
from intern.decisions import DecisionClient
from intern.events import Decision, Event
from intern.memory.drive import MAIN, STEWARD_LOCK, TREE_LOCK, GitError, create, locked
from intern.memory.lint import fix, lint, working_files
from intern.memory.spend import add_spend
from intern.memory.steward.git import publish, recover
from intern.memory.steward.run import Deferred, Skipped, commit, propose
from intern.memory.steward.write import StewardError
from intern.memory.steward.writer import writer_from_ref
from intern.messages import Usage
from intern.models.registry import ConfigError

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _init(root: Path) -> int:
    if root.exists() and any(root.iterdir()):
        print(f"{names.NAME}: {root} is not empty", file=sys.stderr)
        return EXIT_USAGE
    try:
        create(root)
    except GitError as e:
        print(f"{names.NAME}: {e}", file=sys.stderr)
        return EXIT_FAILED
    print(f"created a memory drive at {root}")
    return EXIT_OK


def _lint(root: Path, apply_fix: bool) -> int:
    if not root.is_dir():
        print(f"{names.NAME}: {root} is not a directory", file=sys.stderr)
        return EXIT_USAGE
    files = working_files(root)
    if apply_fix:
        for name, text in fix(files).items():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            (root / name).write_text(text)
            print(f"rewrote {name}")
        files = working_files(root)
    problems = lint(files)
    for problem in problems:
        print(problem)
    print(f"{len(problems)} problems in {len(files)} files", file=sys.stderr)
    return EXIT_FAILED if problems else EXIT_OK


EXIT_DEFERRED = 3
DEFERRED_FILE = "nace-steward-deferred"


def _decision_sink(event: Event) -> None:
    if isinstance(event, Decision) and not event.ok and event.error:
        print(f"{names.NAME}: {event.purpose}: {event.error}", file=sys.stderr)


def _deferred_file(drive: Path) -> Path | None:
    # Journals a run deferred, retried by the next run on this checkout; kept in git's own directory, so it is
    # never part of the drive.
    git = drive / ".git"
    return git / DEFERRED_FILE if git.is_dir() else None


async def _steward(
    journals: list[Path],
    drive: Path,
    system1: str | None,
    writer_ref: str | None,
    world: str = "",
    merge: bool = False,
) -> int:
    with locked(drive, STEWARD_LOCK):
        return await _steward_locked(journals, drive, system1, writer_ref, world, merge)


async def _steward_locked(
    journals: list[Path], drive: Path, system1: str | None, writer_ref: str | None, world: str, merge: bool
) -> int:
    bootstrap = Bootstrap(Options(cwd=Path.cwd(), system1=system1))
    writer = writer_from_ref(writer_ref, bootstrap.env) if writer_ref else None
    try:
        backend = bootstrap._system1()
    except ConfigError as e:
        print(f"{names.NAME}: system1 unavailable: {e}", file=sys.stderr)
        return EXIT_FAILED
    cfg = bootstrap.config.system1
    decisions = DecisionClient(
        backend,
        _decision_sink,
        timeout_s=cfg.timeout_s,
        max_state_tokens=cfg.max_state_tokens,
        max_concurrency=cfg.max_concurrency,
    )
    if merge:
        try:
            with locked(drive, TREE_LOCK):
                recover(drive)
        except GitError as e:
            print(f"{names.NAME}: {e}", file=sys.stderr)
            return EXIT_FAILED
    queue = _deferred_file(drive)
    given = [j.resolve() for j in journals]
    waiting = [Path(line) for line in queue.read_text().splitlines() if line] if queue and queue.exists() else []
    retried = [j for j in dict.fromkeys(waiting) if j not in given and j.exists()]
    for journal in retried:
        print(f"retrying deferred {journal}")
    still: list[Path] = []
    deferred = failed = False
    for journal in [*retried, *given]:
        before = decisions.usage.model_copy()
        written = writer.usage if writer is not None else Usage()
        try:
            proposed = await propose(journal, drive, decisions, writer, world)
            with locked(drive, TREE_LOCK):
                proposal = commit(drive, proposed)
                if merge:
                    publish(drive, proposal.branch)
        except Deferred as e:
            print(f"deferred {journal}: {e}")
            still.append(journal)
            deferred = deferred or journal in given
        except Skipped as e:
            print(f"skipped {journal}: {e}")
        except (StewardError, GitError) as e:
            print(f"{names.NAME}: {journal}: {e}", file=sys.stderr)
            failed = True
        else:
            where = MAIN if merge else proposal.branch
            print(f"{where}: {len(proposal.files)} files at {proposal.commit[:12]}")
            for name in sorted(proposal.files):
                print(f"  {name}")
            for gap in proposal.gaps:
                print(f"  not kept [{gap.pointer}]: {gap.text}")
        finally:
            with contextlib.suppress(OSError):
                after = writer.usage if writer is not None else Usage()
                add_spend(bootstrap.home, journal, decisions.usage.since(before), after - written)
    if queue is not None:
        if still:
            queue.write_text("".join(f"{j}\n" for j in still))
        else:
            queue.unlink(missing_ok=True)
    if failed:
        return EXIT_FAILED
    return EXIT_DEFERRED if deferred else EXIT_OK


def steward_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{names.NAME} steward",
        description="Propose memory pages from finished session journals, one proposed/<session> branch each. "
        "System 1 only: a session System 1 could not answer for is deferred (exit 3) and retried by the "
        "next steward run on the same drive checkout.",
    )
    parser.add_argument("journals", nargs="+", type=Path, help="Session journal files (JSONL).")
    parser.add_argument("--drive", type=Path, required=True, help="The memory drive: a clean git checkout of main.")
    parser.add_argument("--system1", help="decisions:<model>[,...] or drex:<model>[,...]; default from config.toml.")
    parser.add_argument(
        "--writer",
        help="provider/model of an OpenAI-compatible LLM that phrases notes, or replay:<dir> of saved drafts; "
        "System 1 still decides.",
    )
    parser.add_argument(
        "--world",
        default="",
        help="Label of the world the sessions ran in, recorded once in .wiki/world.md. A drive holds one world: a "
        "session of another world is refused and needs its own drive.",
    )
    parser.add_argument(
        "--merge", action="store_true", help="Fast-forward main to each proposal and delete its branch."
    )
    args = parser.parse_args(argv)
    try:
        drive = args.drive.expanduser().resolve()
        return asyncio.run(_steward(args.journals, drive, args.system1, args.writer, args.world, args.merge))
    except ConfigError as e:
        print(f"{names.NAME}: {e}", file=sys.stderr)
        return EXIT_USAGE


def memory_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog=f"{names.NAME} memory", description="Create and check memory drives.")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create a drive from the template, as a git repository on main.")
    init.add_argument("drive", type=Path)
    check = commands.add_parser("lint", help="Check a drive's working tree against the format.")
    check.add_argument("drive", type=Path)
    check.add_argument("--fix", action="store_true", help="Rewrite Contents token counts and tables of contents.")
    args = parser.parse_args(argv)
    root = args.drive.expanduser().resolve()
    if args.command == "init":
        return _init(root)
    return _lint(root, args.fix)
