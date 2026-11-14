"""The State as sections of items, so a writer can update it with edits instead of writing all of it again.

A full write re-emits the whole State: 2-4K output tokens, 16-35 s a write on Opus, while the median write in
a session covered two turns and changed a step or two. Rewriting also rewords: consecutive States
were 35-84% alike, and details an earlier State held went missing without anyone dropping them. Every model edits
after its first State (measured on Opus, GPT and Qwen3.8 27B).

Here the State is parsed into its sections (`## Plan`, `## Now`, ...), each a list of items: a line that starts
at the margin, with the indented lines under it. The writer sees the items numbered within their section and
replies with edits under the headings of the sections that change:

    ## Plan
    set 3: [done] the parser fixed; 5 tests pass
    add: [pending] write the report
    ## Facts
    drop: 2
    ## Now
    replace: the report, from the figures in Facts

Items are named by number, not by their text, so no edit depends on copying a line exactly. An edit that names
no such item, a line that is no edit, or a reply in no section is refused as a whole (`EditError`); the caller
then asks for the whole State instead. Code applies the edits; nothing else of the State changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Mark = Literal["number", "bullet", "plain"]

#: The sections the writer is asked for, in this order (`STATE_PROMPT`).
SECTIONS = ("Plan", "Done when", "Now", "Artifacts", "Facts")
NOW = "Now"
#: How a section's new items are marked when it has none to follow.
DEFAULT_MARKS: dict[str, Mark] = {"plan": "number", "artifacts": "bullet", "facts": "bullet"}
#: The size every edit request asks the State to stay under, giving its current size: edits add more readily than
#: they drop, and the write prompt bounds the State at about 1,000 words. An edit past it is trimmed by edits in the
#: same fork, not written whole (18 of 134 Opus edit requests went past it and cost a whole rewrite).
MAX_EDITED_WORDS = 1_500
#: A State is edited only when it has at least this many of the sections: anything else (a failed write's list
#: of dropped calls, a summary from before the State) is written whole.
MIN_SECTIONS = 2

_HEADING = re.compile(r"^#{1,3}\s+(?P<title>.+?)\s*#*\s*$")
_NUMBER = re.compile(r"^\d+[.)]\s+")
_BULLET = re.compile(r"^[-*+]\s+")
#: `replace 9: …` names an item, so it sets it, and `set 8. …` is `set 8: …` (Qwen3.8 wrote each in a replay).
_SET = re.compile(r"^(?:set|update|change|replace)\s+(?:item\s+)?#?(?P<n>\d+)\s*[:=.-]\s?(?P<text>.*)$", re.IGNORECASE)
_ADD = re.compile(r"^(?:add|append|new)(?:\s+#?\d+)?\s*:\s?(?P<text>.*)$", re.IGNORECASE)  # `add 10: …` too
_N = r"#?\d+(?:\s*[-\u2013]\s*\d+)?"  # an item, or a range of them
_DROP = re.compile(rf"^(?:drop|remove|delete)\s*:?\s*(?P<ns>{_N}(?:\s*(?:,|and)\s*{_N})*)\s*\.?$", re.IGNORECASE)
_LIKE_OP = re.compile(r"^[A-Za-z]+\s+#?\d+\s*:")  # `modify 3: ...`: an edit this format does not know
#: A line that starts like an edit but is none (`replace all 8:`, `set: 4. …`): refused, not taken as an item's
#: text, where it landed garbled in a replay.
_VERB = re.compile(r"^(?:set|add|append|drop|remove|delete|replace|rewrite)\b[^:\n]{0,12}:", re.IGNORECASE)
_REPLACE = re.compile(r"^(?:replace|rewrite)(?:\s+(?:all|section))?\s*:\s?(?P<text>.*)$", re.IGNORECASE)
_NO_CHANGE = re.compile(r"^\W*no\s+changes?\W*$", re.IGNORECASE)
#: What a section written out as `(unchanged)` says: nothing to do there.
_UNCHANGED = re.compile(r"^\W*(?:no\s+changes?|unchanged|nothing\s+changed|same(?:\s+as\s+before)?)\W*$", re.I)
_OP_BULLET = re.compile(r"^[-*+]\s+(?=(?:set|update|change|add|append|new|drop|remove|delete|replace|rewrite)\b)", re.I)


class EditError(ValueError):
    """The edits cannot be applied as given; the message says why."""


@dataclass
class Item:
    mark: Mark
    text: str
    """The first line without its marker, then the lines under it as written."""


@dataclass
class Section:
    title: str
    items: list[Item] = field(default_factory=list)

    @property
    def key(self) -> str:
        return self.title.casefold()

    def mark(self) -> Mark:
        """How a new item here is marked: like the first item, else the section's default."""
        return self.items[0].mark if self.items else DEFAULT_MARKS.get(self.key, "plain")


@dataclass
class StateDoc:
    sections: list[Section]
    preamble: str = ""
    """Text before the first heading, kept as written."""

    def section(self, title: str) -> Section | None:
        return next((s for s in self.sections if s.key == title.casefold()), None)

    def is_state(self) -> bool:
        """Whether this is a State at all: at least one of its sections. Writers have replied with the task's work
        instead, and on main that landed as the State: Qwen3.8 thought through its budget and wrote "The state was
        written as requested — no tools called."; Claude, asked about State writes, answered the user under
        headings of its own ("## Why it's slow", "## Options")."""
        known = {name.casefold() for name in SECTIONS}
        return any(s.key in known for s in self.sections)

    def editable(self) -> bool:
        known = {name.casefold() for name in SECTIONS}
        return sum(s.key in known for s in self.sections) >= MIN_SECTIONS


def heading(line: str) -> str | None:
    """The title of the section a line heads, or None when it heads none."""
    match = _HEADING.match(line.strip())
    return match.group("title").strip() if match else None


def _items(lines: list[str]) -> list[Item]:
    items: list[Item] = []
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line[0].isspace() and items:  # under the item above
            items[-1].text += "\n" + line
            continue
        line = line.strip()
        if number := _NUMBER.match(line):
            items.append(Item("number", line[number.end() :]))
        elif bullet := _BULLET.match(line):
            items.append(Item("bullet", line[bullet.end() :]))
        else:
            items.append(Item("plain", line))
    return items


def parse(text: str) -> StateDoc:
    """The State's sections in the order written, each a list of items."""
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if heading := _HEADING.match(line.strip()):
            sections.append((heading.group("title").strip(), []))
        elif sections:
            sections[-1][1].append(line)
        else:
            preamble.append(line)
    return StateDoc(
        sections=[Section(title, _items(lines)) for title, lines in sections], preamble="\n".join(preamble).strip()
    )


def render(doc: StateDoc, numbered: bool = False) -> str:
    """The State as markdown; ``numbered`` numbers every item within its section, as the writer sees it."""
    parts = [doc.preamble] if doc.preamble else []
    for section in doc.sections:
        lines = [f"## {section.title}"]
        count = 0
        for item in section.items:
            first, _, rest = item.text.partition("\n")
            if numbered or item.mark == "number":
                count += 1
                head = f"{count}. {first}"
            elif item.mark == "bullet":
                head = f"- {first}"
            else:
                head = first
            lines.append(f"{head}\n{rest}" if rest else head)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


@dataclass
class _Ops:
    sets: dict[int, str] = field(default_factory=dict)
    drops: set[int] = field(default_factory=set)
    adds: list[str] = field(default_factory=list)
    replace: str | None = None


def is_no_change(reply: str) -> bool:
    lines = [line for line in _unfenced(reply).splitlines() if line.strip()]
    return len(lines) == 1 and bool(_NO_CHANGE.match(lines[0]))


def is_whole(reply: str) -> bool:
    """Whether a reply is a whole State rather than edits: most of the sections, and no edit in them."""
    doc = parse(_unfenced(reply))
    if not doc.editable() or len(doc.sections) < len(SECTIONS) - 1:
        return False
    return not any(_op(item.text.partition("\n")[0]) for s in doc.sections for item in s.items)


def _op(line: str) -> bool:
    line = _OP_BULLET.sub("", line.strip())
    return any(p.match(line) for p in (_SET, _ADD, _DROP, _REPLACE))


def _unfenced(reply: str) -> str:
    """The reply without a code fence around it, which some models add."""
    lines = reply.strip().splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines)


def _strip_mark(text: str) -> str:
    """An item's text without a marker the writer copied from the numbered view or added."""
    first, sep, rest = text.partition("\n")
    first = first.strip()
    first = first[m.end() :] if (m := _NUMBER.match(first)) else first
    first = first[m.end() :] if (m := _BULLET.match(first)) else first
    return first + sep + rest


def apply_edits(doc: StateDoc, reply: str) -> StateDoc:
    """``doc`` with the reply's edits applied; raises EditError when any edit cannot be. Item numbers are those of
    ``doc`` as the writer saw it, whatever else the reply changes."""
    ops: dict[str, _Ops] = {}
    titles: dict[str, str] = {}
    current: str | None = None
    open_text: tuple[str, int | None] | None = None  # the edit the next indented or plain line continues

    def extend(line: str) -> None:
        assert current is not None and open_text is not None
        kind, n = open_text
        section = ops[current]
        if kind == "set" and n is not None:
            section.sets[n] += "\n" + line
        elif kind == "add":
            section.adds[-1] += "\n" + line
        elif kind == "replace":
            section.replace = f"{section.replace}\n{line}" if section.replace else line

    lines = _unfenced(reply).splitlines()
    has_now = any(
        (h := _HEADING.match(x.strip())) and h.group("title").strip().casefold() == NOW.casefold() for x in lines
    )
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if heading := _HEADING.match(line.strip()):
            title = heading.group("title").strip()
            known = doc.section(title) or next((s for s in SECTIONS if s.casefold() == title.casefold()), None)
            if known is None:
                raise EditError(f"no section `{title}` in the state")
            key = title.casefold()
            titles.setdefault(key, known.title if isinstance(known, Section) else known)
            ops.setdefault(key, _Ops())
            current = key
            open_text = None
            continue
        stripped = _OP_BULLET.sub("", line.strip())
        # `replace all` without a heading of its own, before any or after another section's item edits, is Now's
        # when the reply has no Now: the prompt asks it for Now, and Qwen3.8 left Now's heading out so.
        if (
            _REPLACE.match(stripped)
            and not has_now
            and NOW.casefold() not in ops
            and (current is None or ops[current].sets or ops[current].drops or ops[current].adds)
        ):
            now = doc.section(NOW)
            current = NOW.casefold()
            titles.setdefault(current, now.title if now is not None else NOW)
            ops[current] = _Ops()
        if current is None:
            if _op(stripped):
                raise EditError(f"an edit before any section heading: `{stripped[:80]}`")
            continue  # a preamble ("Here are the edits:") changes nothing
        section = ops[current]
        if m := _SET.match(stripped):
            n = int(m.group("n"))
            if n in section.sets or n in section.drops:
                raise EditError(f"item {n} of {titles[current]} edited twice")
            section.sets[n] = m.group("text")
            open_text = ("set", n)
        elif m := _DROP.match(stripped):
            for n in _numbers(m.group("ns")):
                if n in section.sets or n in section.drops:
                    raise EditError(f"item {n} of {titles[current]} edited twice")
                section.drops.add(n)
            open_text = None
        elif m := _ADD.match(stripped):
            section.adds.append(m.group("text"))
            open_text = ("add", None)
        elif m := _REPLACE.match(stripped):
            if section.replace is not None:
                raise EditError(f"{titles[current]} replaced twice")
            section.replace = m.group("text")
            open_text = ("replace", None)
        elif _LIKE_OP.match(stripped) or _VERB.match(stripped):
            raise EditError(f"not an edit this format knows: `{stripped[:80]}` (set, add, drop or replace)")
        elif open_text is not None:
            extend(line)  # the edit's text goes on, under it or not
        elif not (section.sets or section.drops or section.adds or section.replace is not None):
            # A section written out without edits is its new text (Qwen3.8 wrote `## Now` so in a replay).
            section.replace = line
            open_text = ("replace", None)
        else:
            raise EditError(f"a line under {titles[current]} that is no edit: `{stripped[:80]}`")

    if not ops:
        raise EditError("no section heading, so no edit")
    out = StateDoc(sections=[Section(s.title, list(s.items)) for s in doc.sections], preamble=doc.preamble)
    for key, edit in ops.items():
        if edit.replace is not None and _UNCHANGED.match(edit.replace.strip()):
            if edit.sets or edit.drops or edit.adds:
                raise EditError(f"{titles[key]} said unchanged and edited at once")
            continue
        section = out.section(key)
        if section is None:
            section = Section(titles[key])
            out.sections.append(section)
            out.sections.sort(key=_order)
        if edit.replace is not None:
            if edit.sets or edit.drops or edit.adds:
                raise EditError(f"{section.title} replaced and edited at once")
            section.items = _items(edit.replace.splitlines())
            continue
        for n in [*edit.sets, *edit.drops]:
            if not 1 <= n <= len(section.items):
                raise EditError(f"{section.title} has no item {n} (it has {len(section.items)})")
        mark = section.mark()
        kept: list[Item] = []
        for n, item in enumerate(section.items, 1):
            if n in edit.drops:
                continue
            if n in edit.sets:
                text = _strip_mark(edit.sets[n]).strip("\n")
                if not text.strip():
                    raise EditError(f"item {n} of {section.title} set to nothing; drop it instead")
                kept.append(Item(item.mark, text))
            else:
                kept.append(item)
        for text in edit.adds:
            if not text.strip():
                raise EditError(f"an empty item added to {section.title}")
            if text.startswith("\n"):  # `add:` alone, the items on the lines under it
                kept += [Item(mark if i.mark == "plain" else i.mark, i.text) for i in _items(text.splitlines())]
            else:
                kept.append(Item(mark, _strip_mark(text).strip("\n")))
        section.items = kept
    return out


def _numbers(spec: str) -> list[int]:
    """`2, 4-6 and 9` as [2, 4, 5, 6, 9]."""
    out: list[int] = []
    for start, end in re.findall(r"(\d+)(?:\s*[-\u2013]\s*(\d+))?", spec):
        out += list(range(int(start), int(end or start) + 1))
    return out


def _order(section: Section) -> int:
    """Known sections in their order; others after them, in the order met."""
    names = [s.casefold() for s in SECTIONS]
    return names.index(section.key) if section.key in names else len(names)


def words(text: str) -> int:
    return len(text.split())
