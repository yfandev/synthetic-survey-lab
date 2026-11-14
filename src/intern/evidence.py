"""Evidence preservation: receipts for big test/build output, and the code the step works on kept through compaction.

After SoL-Pi's Evidence-Preserving Reducer (arXiv 2609.20519): a diagnostic command's output over MIN_CHARS
is replaced by its evidence (fatal signals, failures, failing targets, the summary) quoted byte-for-byte, the
exit status, and the path of the archived full output for readback; any doubt leaves the output alone. Where
the paper asks a small generative model for the quotes and verifies them, this extracts them by rule: the
quotes are verbatim by construction, and the same checks stand (a failing log must carry failure evidence,
a log without a test framework's own signal is left alone, the receipt must be smaller, no secrets).

The second half is ours. What compaction loses in our traces is not test output but code; prose in the State's
facts cannot stand in for code the model reasons over. At each compaction System 1 judges every file read about
to be cut (a `read` slice, a bash cat/sed/head of a file), and every read kept so far, against the State: does
the current step work on it? `working_on` reads stay verbatim after the State, within a budget; the rest go.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from intern.context import bound_output
from intern.decisions import ChoiceAnswer, ChoiceQuestion, DecisionClient
from intern.events import EventSink, Reduced
from intern.journal import Kept
from intern.messages import AssistantMessage, Message, ToolCall, ToolResult, ToolResultMessage

# -- receipts for test/build output ---------------------------------------------------------------

MIN_CHARS = 4_000  # the paper's 4 KiB
MAX_CHARS = 600_000
MAX_ITEMS = 12
MAX_QUOTE_CHARS = 600
MAX_ASSERTION_BLOCKS = 3  # pytest `E` blocks, one per failing test, first ones first
MAX_TARGETS = 4
MAX_WARNINGS = 2

DIAGNOSTIC_COMMAND = re.compile(
    r"(?:^|[;&|()\s'\"])(?:lake\s+build|lake\s+env\s+lean|lean|coq|cargo(?:\s+(?:build|test|check))?|zig\s+build"
    r"|pytest|python(?:3)?\s+-m\s+(?:pytest|unittest|py_compile)|python(?:3)?\s+(?:runtests|manage)\.py(?:\s+test)?"
    r"|tox|ctest|cmake\s+--build|ninja|make|npm\s+test|pnpm\s+test|yarn\s+test|go\s+test|bazel\s+test)(?:\s|$)",
    re.IGNORECASE,
)
FAILURE_SIGNAL = re.compile(
    r"error|failed|failure|fatal|exception|panic|timeout|unsolved|type mismatch|assert", re.IGNORECASE
)
LIKELY_SECRET = re.compile(
    r"(?:api[_-]?key|authorization|bearer|access[_-]?token|secret)[^\n]{0,32}[=:][^\n]+", re.IGNORECASE
)

VERDICT = re.compile(r"^(?:FAILED|ERROR)\s+\S")  # pytest short summary, unittest `ERROR: test_x`, `FAIL: ` below
UNITTEST_VERDICT = re.compile(r"^(?:FAIL|ERROR): \S")
ASSERTION = re.compile(r"^E\s{3}")
EXCEPTION = re.compile(r"^(?:[A-Za-z_][\w.]*(?:Error|Exception|Warning|Exit|Interrupt)\b.*|AssertionError.*)$")
FATAL = re.compile(
    r"^(?:(?:ModuleNotFoundError|ImportError|SyntaxError|IndentationError|FileNotFoundError|RecursionError)\b.*"
    r"|.*(?:error: could not compile|error\[E\d+\]|fatal error:|fatal:|panic:|Segmentation fault|Killed|OutOfMemory).*"
    r"|make: \*\*\* .*|ERROR: .*(?:not found|No such file).*)$"
)
TARGET = re.compile(r"^(?:_{3,} (?P<pytest>.+?) _{3,}|--- FAIL: (?P<go>\S+).*|---- (?P<cargo>.+?) stdout ----)$")
SUMMARY = re.compile(
    r"^(?:={3,} .*?(?:passed|failed|error|skipped|no tests ran).*? ={3,}"  # pytest
    r"|Ran \d+ tests? in .*|OK(?: \(.*\))?|FAILED \(.*\)"  # unittest
    r"|test result: .*|(?:ok|FAIL)\s+\S+\s+[\d.]+s"  # cargo, go
    r"|\[exit code 0\]|Error: the command (?:exited with code \d+|timed out.*))$"  # the bash tool's status line
)
WARNING = re.compile(r"warning:|DeprecationWarning|UserWarning", re.I)
EXIT_CODE = re.compile(r"^Error: the command exited with code (-?\d+)$", re.M)

# The engine's error footer (`Error: ...` / `Next: ...`) stays verbatim at the end of a receipt.
ERROR_FOOTER = re.compile(r"\n\nError: [^\n]*\nNext: [^\n]*$")


@dataclass(frozen=True)
class Evidence:
    kind: str  # fatal, failure, target, summary, warning
    line: int  # 1-based
    quote: str


def _quote(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end])[:MAX_QUOTE_CHARS]


def extract(body: str) -> list[Evidence]:
    """The evidence of a test/build log, by rule: at most MAX_ITEMS verbatim quotes, each at most
    MAX_QUOTE_CHARS. Fatal signals first, then failures in order, failing targets, the summary, warnings."""
    lines = body.splitlines()
    fatal: list[Evidence] = []
    failures: list[Evidence] = []
    targets: list[Evidence] = []
    summary: list[Evidence] = []
    warnings: list[Evidence] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if FATAL.match(line) and len(fatal) < 3:
            fatal.append(Evidence("fatal", i + 1, line[:MAX_QUOTE_CHARS]))
        elif SUMMARY.match(line):
            summary.append(Evidence("summary", i + 1, line[:MAX_QUOTE_CHARS]))
        elif (VERDICT.match(line) or UNITTEST_VERDICT.match(line)) and len(failures) < MAX_ITEMS:
            failures.append(Evidence("failure", i + 1, line[:MAX_QUOTE_CHARS]))
        elif ASSERTION.match(line) and sum(e.quote.startswith("E ") for e in failures) < MAX_ASSERTION_BLOCKS:
            j = i
            while j < len(lines) and ASSERTION.match(lines[j]):
                j += 1
            failures.append(Evidence("failure", i + 1, _quote(lines, i, j)))
            i = j
            continue
        elif line.startswith("Traceback (most recent call last)"):
            j = i + 1
            while j < len(lines) and (lines[j].startswith("  ") or not lines[j].strip()):
                j += 1
            if j < len(lines) and EXCEPTION.match(lines[j]):
                frame = next((k for k in range(j - 1, i, -1) if lines[k].startswith('  File "')), None)
                start = frame if frame is not None else j
                kind = "fatal" if FATAL.match(lines[j]) and len(fatal) < 3 else "failure"
                (fatal if kind == "fatal" else failures).append(Evidence(kind, start + 1, _quote(lines, start, j + 1)))
                i = j + 1
                continue
        elif TARGET.match(line) and len(targets) < MAX_TARGETS:
            targets.append(Evidence("target", i + 1, line[:MAX_QUOTE_CHARS]))
        elif WARNING.search(line) and len(warnings) < MAX_WARNINGS:
            warnings.append(Evidence("warning", i + 1, line[:MAX_QUOTE_CHARS]))
        i += 1
    summary = summary[-3:]  # the last lines decide: the final tally and the exit status
    out: list[Evidence] = []
    seen: set[str] = set()
    for item in fatal + failures + targets + summary + warnings:
        if item.quote not in seen:
            seen.add(item.quote)
            out.append(item)
    # Keep the summary within the cap: it is the one thing a reader must not lose.
    head = [e for e in out if e.kind != "summary"][: MAX_ITEMS - len(summary)]
    return head + [e for e in out if e.kind == "summary"]


def receipt(body: str, items: list[Evidence], status: str, archive: str) -> str:
    code = EXIT_CODE.search(body)
    outcome = f"failed (exit code {code.group(1)})" if code else "failed" if status == "error" else "succeeded"
    head = (
        f"[Test/build output reduced to its evidence: {len(body):,} chars, {body.count(chr(10)) + 1:,} lines; "
        f"the command {outcome}. Every line below is quoted verbatim. The full output is at {archive}; "
        f"read it with offset/limit or grep when you need the context around a quote.]"
    )
    quotes = [f"{e.kind} L{e.line}: {e.quote}" for e in items] or ["(no verdict, failure or summary line found)"]
    footer = ERROR_FOOTER.search(body)
    return "\n".join([head, *quotes]) + (footer.group(0) if footer else "")


STATUS_LINE = re.compile(r"^(?:\[exit code 0\]|Error: the command .*)$")


def verify(body: str, items: list[Evidence], status: str, text: str) -> str | None:
    """The paper's verifier, on our own extraction: a reason to fall back, or None."""
    if any(e.quote not in body for e in items):
        return "unverifiable-quote"
    if not any(
        e.kind in ("fatal", "failure", "target") or (e.kind == "summary" and not STATUS_LINE.match(e.quote))
        for e in items
    ):
        return "no-test-signal"  # a repro's printout, a diff: not a test log, whatever the command was
    if status == "error" and FAILURE_SIGNAL.search(body) and not any(e.kind in ("fatal", "failure") for e in items):
        return "missing-failure-evidence"
    if len(text) >= len(body):
        return "receipt-not-smaller"
    return None


# -- keeping the code the step works on -------------------------------------------------------------

KEPT_SHARE = 0.10  # of the working window, in chars at ~4 per token, for all kept reads together
KEPT_ITEM_CHARS = 6_000  # one read, head and tail
SHOWN_CHARS = 2_500  # of a read, for System 1's judgement

WORKING = ChoiceQuestion(
    instructions="A file read is about to be cut from the agent's context. Judge it against the session: what the "
    "agent is working on now and what it plans next.",
    criteria={
        "working_on": "the current step or the next planned one changes, tests or traces this code; the agent "
        "will need its exact text again",
        "reference": "it was read to understand something; what mattered is in the agent's own words since",
        "done_with": "a finished step read it; nothing planned touches it",
    },
)


BASH_READ = re.compile(
    r"^\s*(?:cd\s+\S+\s*&&\s*)?(?:cat|sed\s+-n\s+\S+|head(?:\s+-n?\s*\d+)?|tail(?:\s+-n?\s*\d+)?|awk\s+\S+)"
    r"\s+(?P<path>[\w./-]+\.\w+)\s*$"
)


def read_label(name: str, arguments: dict) -> str | None:
    """What a call merely shows: a `read` as `path (lines a-b)`, a bash cat/sed/head/tail of one file as its
    path; None for anything else. The label names the slice, so two slices of one file are judged apart."""
    if name == "read" and isinstance(path := arguments.get("path"), str):
        offset, limit = arguments.get("offset"), arguments.get("limit")
        if isinstance(limit, int):
            start = offset if isinstance(offset, int) else 1
            return f"{path} (lines {start}-{start + limit - 1})"
        return f"{path} (from line {offset})" if isinstance(offset, int) else path
    if name == "bash" and isinstance(command := arguments.get("command"), str):
        inner = re.sub(r"^\s*\S*-exec\s+['\"]?", "", command.strip()).rstrip("'\"")
        if m := BASH_READ.match(inner):
            return m.group("path")
    return None


def reads_in(cut: list[Message]) -> list[Kept]:
    """The file reads among the cut messages (read slices, bash cat/sed/head/tail), latest per slice."""
    labels: dict[str, str] = {}
    latest: dict[str, Kept] = {}
    for message in cut:
        if isinstance(message, AssistantMessage):
            for call in message.tool_calls:
                if label := read_label(call.name, call.arguments):
                    labels[call.id] = label
        elif (
            isinstance(message, ToolResultMessage)
            and message.result.call_id in labels
            and message.result.status == "ok"
        ):
            label = labels[message.result.call_id]
            latest[label] = Kept(path=label, content=message.result.content)
    return list(latest.values())


def reduce_output(call: ToolCall, result: ToolResult, spill_dir: Path, turn: int, emit: EventSink) -> ToolResult | None:
    """The receipt for a big diagnostic output, or None to leave the result alone."""
    command = call.arguments.get("command")
    if result.omitted_bytes:
        # An executor excerpt cannot prove that the missing middle had no failure signals. Keep the
        # capture receipt and its honest truncation metadata instead of claiming a complete reduction.
        return None
    if call.name != "bash" or not isinstance(command, str) or not DIAGNOSTIC_COMMAND.search(command):
        return None
    body = result.content
    if len(body) < MIN_CHARS:
        return None

    def report(text: str | None, items: int, reason: str | None) -> None:
        emit(
            Reduced(
                turn=turn,
                call_id=call.id,
                source_chars=len(body),
                receipt_chars=len(text or body),
                items=items,
                applied=text is not None,
                reason=reason,
            )
        )

    if len(body) > MAX_CHARS:
        report(None, 0, "source-over-max-chars")
        return None
    if LIKELY_SECRET.search(body):
        report(None, 0, "likely-secret")
        return None
    digest = hashlib.sha256(body.encode()).hexdigest()
    archive = spill_dir / "evidence" / f"{digest[:16]}.txt"
    try:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            archive.write_text(body)
            archive.chmod(0o600)
    except OSError:
        report(None, 0, "archive-failed")
        return None
    items = extract(body)
    text = receipt(body, items, result.status, str(archive))
    reason = verify(body, items, result.status, text)
    if reason is not None:
        report(None, len(items), reason)
        return None
    report(text, len(items), None)
    return result.model_copy(update={"content": text})


async def keep_reads(
    cut: list[Message],
    kept_before: list[Kept],
    state: str,
    decisions: DecisionClient,
    working_window: int,
    *,
    turn: int | None = None,
) -> list[Kept]:
    """The reads among the cut, and those kept so far, that System 1 says the current step works on.

    Asked at compaction, concurrently, and awaited: the cut set exists only then, and the next request
    follows at once, so there is no model call to hide the asks behind. Measured live: 0.4 s per compaction,
    ~4% of a turn at 16k (a compaction every ~3 turns), ~1% at 32k."""
    fresh = reads_in(cut)
    paths = {k.path for k in fresh}
    candidates = [k for k in kept_before if k.path not in paths] + fresh
    if not candidates:
        return []
    answers = await asyncio.gather(
        *(
            decisions.ask(
                "keep",
                f"{state}\n\nThe file read about to be cut: {k.path}\n{shown}",
                {"w": WORKING},
                turn=turn,
                subject=k.path,
            )
            for k in candidates
            for shown in (bound_output(k.content, SHOWN_CHARS)[0],)
        )
    )
    budget = int(working_window * 4 * KEPT_SHARE)
    kept: list[Kept] = []
    for candidate, answer in zip(candidates, answers, strict=True):
        verdict = answer["w"]
        if not (isinstance(verdict, ChoiceAnswer) and verdict.choice == "working_on"):
            continue
        content = bound_output(candidate.content, KEPT_ITEM_CHARS)[0]
        if sum(len(k.content) for k in kept) + len(content) > budget:
            break
        kept.append(Kept(path=candidate.path, content=content))
    return kept
