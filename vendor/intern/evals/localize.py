"""Localization eval for the workspace index: a question, the files that answer it, and where `decision_search` ranks
them. Cheap and sensitive where agent evals are not: a few hundred System 1 asks, minutes, no agent in the loop.

Gold sources:

    uv run --with pyarrow evals/localize.py issues --data <issue parquet dir> --repos <git mirror dir> --tasks <ids...>
        The issue text is the question, the files the gold patch touches are the answer; the repository is
        checked out at the base commit (git clone --shared) and indexed.

    uv run evals/localize.py office --tasks <office task dir> --personas <persona workspaces dir>
        The task text is the question, its data_manifest files are the answer (any copy of a file name counts);
        each persona workspace is indexed once.

    uv run evals/localize.py questions --root <dir> --file questions.json
        Hand-written questions over one tree: [{"id", "question", "files": [...], "symbols": [...]}]; the gold
        symbols (chunk keys such as "Run._wrap_up") add symbol@1/@5 to the report. tests/data/workspace_questions.json
        is the 15-question set over src/intern.

Reports hit@1 (the top file is gold), any@5 / any@10 (a gold file in the top k) and recall@10 (share of the
gold files in the top ten), with asks, cost and seconds a question. The
index is built with the processors of one extension and searched with its questions: `--extension` names it,
by default code-search, or workspace for the office source.
`--out results.json` keeps per-question rows for a before/after diff; `--system1` is the Decisions API spec.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pydantic import SecretStr

from intern.config import CAPABILITY_EXTENSIONS
from intern.decisions import DecisionClient, DecisionsApiBackend
from intern.events import Decision
from intern.extensions import REGISTRY
from intern.locate import index as wi
from intern.locate.search import search
from intern.locate.store import Locate

PERSONA_DIRS = {
    "Operations Manager": "OperationsManager_Workdir",
    "Logistics Manager": "LogisticsManager_Workdir",
    "Product Manager": "ProductManager_Workdir",
    "Backend Developer": "BackendDeveloper_Workdir",
    "Researcher": "Research_Workdir",
}
QUESTION_CHARS = 4000


class Question:
    def __init__(self, qid: str, root: Path, text: str, gold: list[set[str]], symbols: list[str] | None = None) -> None:
        self.id, self.root, self.text, self.gold, self.symbols = qid, root, text, gold, symbols or []


def issue_questions(args: argparse.Namespace) -> list[Question]:
    import pyarrow.parquet as pq  # pyright: ignore[reportMissingImports]  (uv run --with pyarrow)

    rows = {r["instance_id"]: r for f in sorted(args.data.glob("*.parquet")) for r in pq.read_table(f).to_pylist()}
    out = []
    for tid in args.tasks:
        inst = rows[tid]
        dest = args.work / tid
        if not dest.exists():
            mirror = args.repos / (inst["repo"].replace("/", "__") + ".git")
            subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(mirror), str(dest)], check=True)
            subprocess.run(["git", "-C", str(dest), "checkout", "-q", inst["base_commit"]], check=True)
        gold = [
            {ln[len("diff --git a/") :].split(" b/")[0]}
            for ln in inst["patch"].splitlines()
            if ln.startswith("diff --git a/")
        ]
        out.append(Question(tid, dest, inst["problem_statement"], gold))
    return out


def office_questions(args: argparse.Namespace) -> list[Question]:
    out = []
    for mp in sorted(args.tasks_dir.glob("*/metadata.json"), key=lambda p: int(p.parent.name)):
        meta = json.loads(mp.read_text(encoding="utf-8"))
        persona = PERSONA_DIRS.get(meta["persona"])
        if (
            persona is None
            or (args.personas / persona).is_dir() is False
            or (args.only and mp.parent.name not in args.only)
        ):
            continue
        names = [e["filename"] for e in meta.get("data_manifest") or []]
        if names:
            out.append(
                Question(mp.parent.name, args.personas / persona, str(meta["task"]).strip(), [{n} for n in names])
            )
    return out


def hand_questions(args: argparse.Namespace) -> list[Question]:
    return [
        Question(str(q["id"]), args.root, q["question"], [{f} for f in q["files"]], q.get("symbols"))
        for q in json.loads(args.file.read_text())
    ]


def resolve(index: wi.Index, gold: list[set[str]]) -> list[set[str]]:
    """Gold given as file names (office) becomes every indexed path with that name; paths stay as they are."""
    by_name = collections.defaultdict(set)
    for p in index.files:
        by_name[p.rsplit("/", 1)[-1]].add(p)
    return [set().union(*(by_name.get(g, {g}) if "/" not in g else {g} for g in entry)) for entry in gold]


def score(ranked: list[str], gold: list[set[str]]) -> dict[str, float]:
    found = lambda k: [any(p in g for p in ranked[:k]) for g in gold]  # noqa: E731
    return {
        "hit1": float(bool(ranked) and any(ranked[0] in g for g in gold)),
        "any5": float(any(found(5))),
        "any10": float(any(found(10))),
        "recall10": sum(found(10)) / len(gold),
    }


def extension(name: str) -> type[Locate]:
    """The extension whose processors build the index and whose questions System 1 is asked."""
    searches = {**{n: e.cls for n, e in CAPABILITY_EXTENSIONS.items()}, **REGISTRY}  # capabilities aren't in it
    ext = searches.get(name)
    if ext is None or not issubclass(ext, Locate):
        known = [n for n, e in searches.items() if issubclass(e, Locate)]
        raise SystemExit(f"no search extension named {name!r}; the loaded ones: {known}")
    return ext


async def run(questions: list[Question], args: argparse.Namespace) -> list[dict]:
    ext = extension(args.extension or ("workspace" if args.source == "office" else "code-search"))
    models = args.system1.removeprefix("decisions:").split(",")
    backend = DecisionsApiBackend(args.base_url, SecretStr(os.environ[args.api_key_env]), models, timeout_s=30)
    events: list = []
    client = DecisionClient(backend, events.append)
    indexes: dict[Path, wi.Index] = {}
    rows = []
    for q in questions:
        if q.root not in indexes:
            t = time.perf_counter()
            home = q.root / ext.folder
            indexes[q.root] = wi.load(home) or wi.build(
                q.root, home, processors=ext.processors, report=lambda line: None
            )
            print(
                f"index {q.root.name}: {len(indexes[q.root].files)} files, {time.perf_counter() - t:.0f}s",
                file=sys.stderr,
            )
        index = indexes[q.root]
        gold = resolve(index, q.gold)
        n0, t = len(events), time.perf_counter()
        result = await search(index, q.root, q.text[:QUESTION_CHARS], client, ext.asks, lexical=args.lexical)
        secs = time.perf_counter() - t
        asks = [e for e in events[n0:] if isinstance(e, Decision)]
        ranked = [p for p, s in result.files if s > 0]
        row = {
            "id": q.id,
            "root": q.root.name,
            "gold": len(gold),
            "top": ranked[:5],
            "system1": score(ranked, gold),
            "asks": len(asks),
            "usd": round(sum(e.cost_usd or 0 for e in asks), 4),
            "secs": round(secs, 1),
        }
        if q.symbols:
            keys = [pl.key for pl in result.places if pl.key]
            row["symbols"] = {
                "hit1": float(bool(keys) and keys[0] in q.symbols),
                "any5": float(any(k in q.symbols for k in keys[:5])),
            }
        rows.append(row)
        s = row["system1"]
        line = f"hit@1 {s['hit1']:.0f} any@10 {s['any10']:.0f} recall@10 {s['recall10']:.2f}"
        if "symbols" in row:
            line += f" | symbol@1 {row['symbols']['hit1']:.0f} @5 {row['symbols']['any5']:.0f}"
        print(
            f"{q.id:>28} gold {len(gold):3d} | {line} | {len(asks):3d} asks ${row['usd']:.4f} {secs:.0f}s", flush=True
        )
    return rows


def report(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n{n} questions")
    print("| arm | hit@1 | any@5 | any@10 | recall@10 | asks / q | $ / q | s / q |")
    print("|---|---|---|---|---|---|---|---|")
    m = {k: sum(r["system1"][k] for r in rows) / n for k in ("hit1", "any5", "any10", "recall10")}
    asks, usd, secs = (sum(r[k] for r in rows) / n for k in ("asks", "usd", "secs"))
    print(
        f"| system1 | {m['hit1']:.2f} | {m['any5']:.2f} | {m['any10']:.2f} | {m['recall10']:.2f} "
        f"| {asks:.0f} | ${usd:.4f} | {secs:.1f} |"
    )
    if all("symbols" in r for r in rows):
        hit1, any5 = (sum(r["symbols"][k] for r in rows) / n for k in ("hit1", "any5"))
        print(f"| symbols (system1) | {hit1:.2f} | – | {any5:.2f} at 5 | – | | | |")
    for root in sorted({r["root"] for r in rows}):
        rs = [r for r in rows if r["root"] == root]
        if len({r["root"] for r in rows}) > 1:
            hit1, any10 = (sum(r["system1"][k] for r in rs) / len(rs) for k in ("hit1", "any10"))
            print(f"  {root}: n={len(rs)} hit@1 {hit1:.2f} any@10 {any10:.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--system1",
        default="decisions:nace-ai/drex-max,~typesafe/jev-latest",
        help="decisions:<model>[,<fallback>...]",
    )
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--out", type=Path, help="Write per-question rows here as JSON.")
    parser.add_argument(
        "--lexical", action="store_true", help="Steer the walk by the question's words, as decision_code does."
    )
    parser.add_argument(
        "--extension", help="Whose index and questions: code-search (default; office: workspace) or another name."
    )
    sub = parser.add_subparsers(dest="source", required=True)
    issues = sub.add_parser("issues")
    issues.add_argument("--data", type=Path, required=True, help="Directory of the dataset's parquet files.")
    issues.add_argument("--repos", type=Path, required=True, help="Directory of <owner>__<repo>.git mirrors.")
    issues.add_argument("--work", type=Path, default=Path("/tmp/localize-issues"), help="Where checkouts go.")
    issues.add_argument("--tasks", nargs="+", required=True, help="Instance ids.")
    office = sub.add_parser("office")
    office.add_argument(
        "--tasks",
        dest="tasks_dir",
        type=Path,
        required=True,
        help="Office task dir (one folder per task with metadata.json).",
    )
    office.add_argument("--personas", type=Path, required=True, help="The extracted persona workspaces.")
    office.add_argument("--only", nargs="*", help="Task ids to keep.")
    hand = sub.add_parser("questions")
    hand.add_argument("--root", type=Path, required=True, help="The directory the questions are about.")
    hand.add_argument("--file", type=Path, required=True, help="JSON: id, question, files, symbols (optional).")
    args = parser.parse_args(argv)
    sources = {"issues": issue_questions, "office": office_questions, "questions": hand_questions}
    questions = sources[args.source](args)
    if not questions:
        print("no questions", file=sys.stderr)
        return 2
    rows = asyncio.run(run(questions, args))
    report(rows)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
