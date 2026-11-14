"""Cards: the small text views of groups, files and chunks the cascade shows System 1 at each level.

A card is built from a Derivative and nothing else: a file's outline (its symbols, headings, tabs or sections), its
first lines, the years it mentions and the words that set it apart from the rest of the workspace. No model writes
them, so an index is cheap and offline; a summarizer is a later, pluggable step.
"""

from __future__ import annotations

import re

from intern.locate.processors import Chunk, Derivative

#: Characters of a file card; outline lines past it are counted, not shown.
FILE_CARD_CHARS = 2500
#: Characters of the opening text a document's card shows.
HEAD_CHARS = 500
#: Lines of body a chunk card previews (code) or characters (documents).
PREVIEW_LINES = 15
PREVIEW_CHARS = 1200
#: Characters of member lines a group card carries; members past it are counted, not shown.
MEMBER_LINES_CHARS = 6000
MEMBER_SUMMARY_CHARS = 140
KEYWORDS = 10
#: Keywords a folder card gives for each folder it holds.
CHILD_KEYWORDS = 4
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def file_card(path: str, derivative: Derivative, keywords: list[str]) -> str:
    """Header, summary, outline of top-level chunks (sub-chunks named on their parent's line), head text for a
    document, tables, years, keywords."""
    count = f"{len(derivative.units)} {derivative.unit}{'s' if len(derivative.units) != 1 else ''}"
    out = [f"file {path}  ({count}{f'; {derivative.note}' if derivative.note else ''})"]
    if derivative.summary:
        out.append(f"  doc: {derivative.summary}")
    for chunk in (c for c in derivative.chunks if c.parent is None and c.kind != "page"):
        line = f"  {chunk.title}" + (f" — {chunk.summary}" if chunk.summary else "")
        children = [c for c in derivative.chunks if c.parent == chunk.key and c.kind != "page"]
        if children:
            line += f"\n    {children[0].kind}s: " + ", ".join(c.key.rsplit(".", 1)[-1] for c in children)
        out.append(line)
    if derivative.unit == "page" and derivative.units and not derivative.tables:
        head = " ".join(derivative.units[0].split())[:HEAD_CHARS]
        if head and head != derivative.summary:
            out.append(f"  head: {head}")
    years = sorted({y for u in derivative.units[:50] for y in _YEAR.findall(u)})
    if years and derivative.unit == "page":
        out.append(f"  years: {', '.join(years[:12])}")
    if keywords:
        out.append(f"  keywords: {', '.join(keywords)}")
    return _bounded(out, FILE_CARD_CHARS)


def chunk_card(path: str, chunk: Chunk, derivative: Derivative) -> str:
    """Title, summary, where it is, and a preview of its text (its own, not its sub-chunks')."""
    where = (
        f"{derivative.unit}s {chunk.start}-{chunk.end}"
        if chunk.end > chunk.start
        else f"{derivative.unit} {chunk.start}"
    )
    out = [f"{chunk.title}  in {path} {where}"]
    if chunk.summary:
        out.append(f"  {chunk.summary}")
    children = [c for c in derivative.chunks if c.parent == chunk.key]
    if children:
        out.append(f"  {children[0].kind}s: " + ", ".join(c.title for c in children)[:600])
    body = derivative.units[chunk.start - 1 : chunk.end]
    if derivative.unit == "line":
        shown = body[: PREVIEW_LINES + 1]
        if chunk.kind in {"class", "function", "method"}:
            shown = _past_docstring(shown)[:PREVIEW_LINES]
        out.extend("  | " + line for line in shown)
        if len(body) > len(shown):
            out.append(f"  | ... {len(body) - len(shown)} more lines")
    elif body and chunk.kind == "page":
        out.append("  | " + " ".join(body[0].split())[:PREVIEW_CHARS])
    elif body and chunk.kind == "table":
        out.extend("  | " + line for line in body[0].splitlines()[1 : PREVIEW_LINES + 10] if line.strip())
    return "\n".join(out)


def _past_docstring(lines: list[str]) -> list[str]:
    """The lines after a leading docstring, so a preview shows code rather than what the summary line already says."""
    for i, line in enumerate(lines[1:], 1):
        stripped = line.strip()
        if stripped.startswith(('"""', "'''")):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                return lines[:1] + lines[i + 1 :]
            close = next((j for j in range(i + 1, len(lines)) if quote in lines[j]), None)
            return lines[:1] + lines[close + 1 :] if close is not None else lines
        if stripped:
            break
    return lines


def member_line(derivative: Derivative) -> str:
    """What tells one member from its siblings beside its name: summary and top-level names."""
    tops = [c.key for c in derivative.chunks if c.parent is None and c.kind not in {"page", "pages", "window"}][:8]
    summary = derivative.summary[:MEMBER_SUMMARY_CHARS]
    return " | ".join(p for p in [summary, f"[{', '.join(tops)}]" if tops else ""] if p)


def _bounded(lines: list[str], limit: int) -> str:
    out: list[str] = []
    used = 0
    for n, line in enumerate(lines):
        if used + len(line) > limit:
            out.append(f"  ... {len(lines) - n} more lines")
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)
