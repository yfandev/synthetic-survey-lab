"""The State as a structure: its sections, and the Plan's steps with their status.

The State is Markdown the agent wrote to the sections STATE_PROMPT asks for (Plan, Done when, Now, Artifacts,
Facts). The TUI shows it section by section, the Plan as a checklist, instead of as one blob. Parsing goes by
heading and by the words the prompt asks for ("done", "in progress", "pending"), so a State written in another
shape still shows, as lines, and a step whose words say nothing is `unknown` rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["done", "in_progress", "pending", "unknown"]

PLAN = "Plan"
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*:?\s*#*\s*$")
LIST_ITEM = re.compile(r"^(?P<indent>\s*)(?:[-*+•]|\d+[.)])\s+(?P<text>\S.*)$")
QUOTE = re.compile(r"^\s{0,3}> ?(?P<text>.*)$")
CHECKBOX = re.compile(r"^\[[ xX~-]\]\s*")
EMPHASIS = re.compile(r"\*\*|__")
# Checked in this order: a step "done, pending review" is pending, and "not started" is not started.
PENDING = re.compile(r"\[ \]|\bpending\b|\bto ?do\b|\bnot (?:yet )?(?:started|done|begun)\b|\bblocked\b", re.I)
IN_PROGRESS = re.compile(r"\[[~-]\]|\bin[ -]progress\b|\bongoing\b|\bunderway\b|\bwip\b|\bstarted\b", re.I)
DONE = re.compile(r"\[[xX]\]|[✓✔]|\bdone\b|\bcompleted?\b|\bfinished\b", re.I)


@dataclass(frozen=True)
class Step:
    text: str
    status: Status
    depth: int = 0
    """Nesting: 0 for a top-level item, 1 for a sub-item."""


@dataclass
class Section:
    title: str
    lines: list[str] = field(default_factory=list)
    """The body's lines as written, with the blank lines between them (one per run) but none around them."""
    steps: list[Step] = field(default_factory=list)
    """For the Plan: one per list item, with the status its words give."""


@dataclass
class Outline:
    sections: list[Section]

    def section(self, title: str) -> Section | None:
        key = title.casefold()
        return next((s for s in self.sections if s.title.casefold() == key), None)

    @property
    def plan(self) -> list[Step]:
        section = self.section(PLAN)
        return section.steps if section else []

    def counts(self) -> dict[Status, int]:
        counts: dict[Status, int] = {"done": 0, "in_progress": 0, "pending": 0, "unknown": 0}
        for step in self.plan:
            counts[step.status] += 1
        return counts


def plain(text: str) -> str:
    """A line without its Markdown emphasis markers; backticks stay, they read fine in a terminal."""
    return EMPHASIS.sub("", text).strip()


def step_status(text: str) -> Status:
    if PENDING.search(text):
        return "pending"
    if IN_PROGRESS.search(text):
        return "in_progress"
    if DONE.search(text):
        return "done"
    return "unknown"


def plan_steps(lines: list[str]) -> list[Step]:
    """One step per list item; a line that is not an item continues the item before it. A Plan with no list
    items is one step per line."""
    items: list[tuple[int, str]] = []
    listed = any(LIST_ITEM.match(line) for line in lines)
    for line in lines:
        if match := LIST_ITEM.match(line):
            indent = len(match.group("indent").expandtabs(4))
            items.append((1 if indent >= 2 else 0, match.group("text")))
        elif listed and items:
            depth, text = items[-1]
            items[-1] = (depth, f"{text} {line.strip()}")
        else:
            items.append((0, line.strip()))
    steps: list[Step] = []
    for depth, text in items:
        text = plain(CHECKBOX.sub("", text.strip()))
        if text:
            steps.append(Step(text=text, status=step_status(text), depth=depth))
    return steps


def parse_state(text: str) -> Outline:
    """The State's sections in the order written; text before the first heading is a section without a title."""
    sections: list[Section] = []
    current = Section(title="")
    for raw in text.splitlines():
        if match := HEADING.match(raw):
            if current.lines:
                sections.append(current)
            current = Section(title=match.group("title").strip())
        elif raw.strip():
            current.lines.append(raw.rstrip())
        elif current.lines and current.lines[-1]:
            current.lines.append("")
    if current.lines or current.title:
        sections.append(current)
    for section in sections:
        if section.lines and not section.lines[-1]:
            section.lines.pop()
        if section.title.casefold() == PLAN.casefold():
            section.steps = plan_steps(section.lines)
    return Outline(sections=sections)
