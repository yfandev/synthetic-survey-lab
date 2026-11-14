"""The command line.

    intern [--system2 provider/model] ["task"]              the terminal UI: the default on a terminal
    intern -p [--json] --system2 provider/model -- "task"   run one task headlessly and exit
    intern rpc --system2 provider/model                     serve JSON-RPC over stdio
    intern acp [--system2 provider/model]                   serve the Agent Client Protocol over stdio
    intern tui [--system2 provider/model] ["task"]          the terminal UI, explicitly
    intern schema                                           print the event JSON Schema
    intern models [--json]                                  list the System 2 and System 1 models to pick
    intern index [path] [--documents] [--clean]             build .intern/ index (--documents: .decision_workspace/)
    intern memory init|lint DRIVE                           create or check a memory drive
    intern steward JOURNAL... --drive DRIVE                 propose memory pages from sessions

Without -p or --json, and with stdin and stdout on a terminal, the command opens the terminal UI (the task, if
given, is its first prompt), the way pi, opencode and Claude Code open theirs. Piped, or with -p, it runs the
task headlessly and prints the reply. With --json every event is one JSON object per stdout line; this is what
eval adapters read.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import shutil
import signal
import sys
from pathlib import Path

from intern import __version__, names
from intern.config import SYSTEM1_HELP, Bootstrap, Options
from intern.events import (
    Event,
    Indexing,
    RunEnd,
    SecretUnavailable,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    event_json_schema,
)
from intern.memory.commands import memory_main, steward_main
from intern.models.registry import ConfigError
from intern.protocol.rpc import serve

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_ABORTED = 130
ARG_PREVIEW_CHARS = 120


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=names.NAME, description="A general-purpose agent harness for small models.")
    parser.add_argument("--version", action="version", version=f"{names.NAME} {__version__}")
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Run the prompt headlessly and exit; the default when stdin or stdout is not a terminal.",
    )
    parser.add_argument("--json", action="store_true", help="Write every event as one JSON line to stdout.")
    # -m/--model and --decider are the old names, kept so existing scripts and eval configs still run.
    parser.add_argument(
        "--system2", "-m", "--model", help="Agent model as provider/model, e.g. openrouter/qwen/qwen3.8-27b."
    )
    parser.add_argument("--system1", "--decider", help=f"System 1: {SYSTEM1_HELP}.")
    parser.add_argument("--window", type=int, help="Cap the working window, in tokens.")
    parser.add_argument("--max-turns", type=int, help="Stop after this many turns.")
    parser.add_argument("--profile", type=Path, help="Profile TOML applied over the matched one.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Working directory for tools.")
    parser.add_argument(
        "--no-context-files", action="store_true", help="Ignore AGENTS.md, SYSTEM2.md and SYSTEM1.md files."
    )
    parser.add_argument(
        "--no-extensions",
        action="store_true",
        help="Load none of the profile's extensions; each --extension still adds one. Capabilities, including those"
        " backed by an extension (code search, workspace, memory), are not affected.",
    )
    parser.add_argument(
        "--extension", action="append", default=[], help="Extra extension: a name or module:Class. Repeatable."
    )
    parser.add_argument("--no-session", action="store_true", help="Keep the journal in memory only.")
    parser.add_argument("--session", type=Path, help="Journal file to create or resume.")
    parser.add_argument("--memory", type=Path, help="Memory drive to recall from: a clean git checkout of main.")
    parser.add_argument("--no-memory", action="store_true", help="Mount no memory drive.")
    parser.add_argument("prompt", nargs="*", help="The task. Read from stdin when omitted.")
    return parser


def _options(args: argparse.Namespace) -> Options:
    return Options(
        cwd=args.cwd.resolve(),
        system2=args.system2,
        system1=args.system1,
        window_cap=args.window,
        max_turns=args.max_turns,
        profile=args.profile,
        context_files=not args.no_context_files,
        extensions=not args.no_extensions,
        extra_extensions=args.extension,
        session_file=args.session,
        persist=not args.no_session,
        memory=args.memory,
        memory_off=args.no_memory,
    )


def _json_sink(event: Event) -> None:
    sys.stdout.write(event.model_dump_json() + "\n")
    sys.stdout.flush()


def _text_sink(event: Event) -> None:
    match event:
        case TextDelta(text=text):
            sys.stdout.write(text)
            sys.stdout.flush()
        case ToolCallStart(name=name, arguments=arguments):
            preview = json.dumps(arguments)
            preview = preview if len(preview) <= ARG_PREVIEW_CHARS else preview[:ARG_PREVIEW_CHARS] + "..."
            print(f"\n> {name} {preview}", file=sys.stderr)
        case SecretUnavailable(message=message):
            print(f"[secrets: {message}]", file=sys.stderr)
        case ToolCallEnd(status=status, content=content) if status != "ok":
            print(f"  {status}: {content.strip().splitlines()[-1] if content.strip() else ''}", file=sys.stderr)
        case Indexing(name=name, done=done, total=total, finished=finished, duration_ms=duration_ms):
            # The build's own stderr bar is off when it reports as events; this is the same bar, from them.
            if finished:
                print(f"\r[{name}: indexed {total} files in {(duration_ms or 0) / 1000:.1f}s]", file=sys.stderr)
            elif sys.stderr.isatty():
                share = done / total if total else 0.0
                sys.stderr.write(f"\r{name}: indexing {share:4.0%} {done}/{total}")
                sys.stderr.flush()
        case RunEnd(reason=reason, detail=detail, turns=turns, usage=usage, system1=system1):
            print(file=sys.stdout, flush=True)
            if reason != "completed":
                print(f"[stopped: {reason}{f': {detail}' if detail else ''}]", file=sys.stderr)
            line = f"{turns} turns, {usage.input_tokens} in / {usage.output_tokens} out tokens"
            if system1.asks or system1.failed:
                line += f"; System 1: {system1.asks} asks, {system1.input_tokens} in / {system1.output_tokens} out"
                if system1.failed:
                    line += f", {system1.failed} failed"
                if system1.cost_usd >= 0.0001:
                    line += f", ${system1.cost_usd:.4f}"
            print(f"[{line}]", file=sys.stderr)
        case _:
            pass


# A kill or a closed terminal. Each command runs in a process group of its own, so that a timeout can stop all of it;
# had the process just died, the command and what it started would have kept running, with no run_end.
ENDING_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _exit_code(reason: str) -> int:
    if reason == "completed":
        return EXIT_OK
    return EXIT_ABORTED if reason == "aborted" else EXIT_FAILED


async def _run_prompt(options: Options, prompt: str, as_json: bool) -> int:
    session = await Bootstrap(options).session(_json_sink if as_json else _text_sink)
    loop = asyncio.get_running_loop()
    main = asyncio.current_task()
    ended_by: list[int] = []

    def end(sig: int) -> None:
        """The run ends as an abort ends it: its command stopped with its group, the session closed, run_end
        written. A second signal, or one with no run going, stops waiting for that."""
        ended_by.append(sig)
        if session.running and len(ended_by) == 1:
            session.abort()
        elif main is not None:
            main.cancel()

    with contextlib.suppress(NotImplementedError):
        loop.add_signal_handler(signal.SIGINT, lambda: session.abort() if session.running else None)
        for sig in ENDING_SIGNALS:
            loop.add_signal_handler(sig, end, sig)
    try:
        async with session:
            outcome = await session.run(prompt)
        code = _exit_code(outcome.reason)
    except asyncio.CancelledError:
        if not ended_by:
            raise
        code = EXIT_ABORTED
    finally:
        with contextlib.suppress(NotImplementedError):
            for sig in (signal.SIGINT, *ENDING_SIGNALS):
                loop.remove_signal_handler(sig)
    return 128 + ended_by[0] if ended_by else code


def _index(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{names.NAME} index",
        description="Build or refresh the code-search index (.intern/index.json) of a directory.",
    )
    parser.add_argument("path", nargs="?", type=Path, default=Path.cwd(), help="The directory to index (default: cwd).")
    parser.add_argument("--clean", action="store_true", help="Delete the index first and build from nothing.")
    parser.add_argument(
        "--documents",
        action="store_true",
        help="The workspace extension's index (.decision_workspace/): PDFs, Word documents, workbooks and tables too.",
    )
    args = parser.parse_args(argv)
    from intern.locate import index as locate_index

    root = args.path.resolve()
    if not root.is_dir():
        print(f"{names.NAME}: {args.path} is not a directory", file=sys.stderr)
        return EXIT_USAGE
    home = root / names.PROJECT_DIR
    processors = locate_index.default_processors
    if args.documents:
        from intern.extensions.workspace.documents import office_processors
        from intern.extensions.workspace.extension import Workspace
        from intern.locate.store import moved_home

        home, processors = moved_home(root, Workspace.folder, Workspace.moved_from), office_processors
    if args.clean:
        shutil.rmtree(home, ignore_errors=True) if args.documents else (home / locate_index.INDEX_FILE).unlink(
            missing_ok=True
        )
    progress = locate_index.Progress()
    try:
        built = locate_index.build(root, home, processors=processors, report=lambda line: None, progress=progress)
    except ImportError as e:
        extra = "workspace" if args.documents else "code-search"
        print(f"{names.NAME}: {e}; install {names.DISTRIBUTION}[{extra}]", file=sys.stderr)
        return EXIT_USAGE
    except locate_index.TooManyFiles as e:
        print(f"{names.NAME}: {e}; index a subfolder, or git-ignore what is not worth searching", file=sys.stderr)
        return EXIT_USAGE
    finally:
        progress.close()
    chunks = sum(len(f.chunks) for f in built.files.values())
    tables = sum(len(f.tables) for f in built.files.values())
    print(f"{len(built.files)} files, {chunks} chunks, {tables} tables, {len(built.leaves)} groups -> {home}")
    return EXIT_OK


def _models(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=f"{names.NAME} models", description="List the System 2 and System 1 models clients can pick."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print {system2: [{id, name, default?, reasoning?, window?}], system1: [...], capabilities: [...]}.",
    )
    args = parser.parse_args(argv)
    try:
        catalog = asyncio.run(Bootstrap(Options(cwd=Path.cwd())).catalog())
    except ConfigError as e:
        print(f"{names.NAME}: {e}", file=sys.stderr)
        return EXIT_USAGE
    if args.json:
        print(catalog.model_dump_json(exclude_defaults=True))
        return EXIT_OK
    for role, models in (("System 2", catalog.system2), ("System 1", catalog.system1)):
        print(f"{role}:")
        for model in models:
            print(f"  {model.id}" + (f"  ({model.name})" if model.name != model.id else ""))
    return EXIT_OK


def _acp(options: Options) -> int:
    try:
        from intern.protocol import acp
    except ImportError as e:
        print(f"{names.NAME}: {e}; install {names.DISTRIBUTION}[acp]", file=sys.stderr)
        return EXIT_USAGE
    acp.main(options)
    return EXIT_OK


def _tui(options: Options, prompt: str | None) -> int:
    try:
        from intern.tui import app
    except ImportError as e:
        hint = f"reinstall {names.DISTRIBUTION}, or run with -p for the headless mode"
        print(f"{names.NAME}: {e}; {hint}", file=sys.stderr)
        return EXIT_USAGE
    app.main(options, prompt=prompt)
    return EXIT_OK


def interactive(args: argparse.Namespace) -> bool:
    """Whether a plain `intern [task]` opens the terminal UI: no -p or --json, and a terminal on both ends."""
    return not args.print and not args.json and sys.stdin.isatty() and sys.stdout.isatty()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["schema"]:
        print(json.dumps(event_json_schema(), indent=2))
        return EXIT_OK
    if argv[:1] == ["index"]:
        return _index(argv[1:])
    if argv[:1] == ["models"]:
        return _models(argv[1:])
    if argv[:1] == ["memory"]:
        return memory_main(argv[1:])
    if argv[:1] == ["mcp"]:
        from intern.mcp_commands import mcp_main

        return mcp_main(argv[1:])
    if argv[:1] == ["secrets"]:
        from intern.secrets_commands import secrets_main

        return secrets_main(argv[1:])
    if argv[:1] == ["steward"]:
        return steward_main(argv[1:])
    serving = argv[:1] == ["rpc"]
    serving_acp = argv[:1] == ["acp"]
    serving_tui = argv[:1] == ["tui"]
    args = _parser().parse_args(argv[1:] if serving or serving_acp or serving_tui else argv)
    try:
        options = _options(args)
        if serving_tui:
            return _tui(options, " ".join(args.prompt).strip() or None)
        if serving_acp:
            return _acp(options)
        if serving:
            ended_by = asyncio.run(serve(Bootstrap(options).session))
            return 128 + ended_by if ended_by else EXIT_OK
        prompt = " ".join(args.prompt).strip()
        if interactive(args):
            return _tui(options, prompt or None)
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            print(f"{names.NAME}: no prompt given", file=sys.stderr)
            return EXIT_USAGE
        return asyncio.run(_run_prompt(options, prompt, args.json))
    except ConfigError as e:
        print(f"{names.NAME}: {e}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
