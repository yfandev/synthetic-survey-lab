from __future__ import annotations

import re
from dataclasses import dataclass, field

from intern.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    NoulAnswer,
    NoulQuestion,
    Question,
)
from intern.memory.drive import Drive
from intern.memory.format import RELATIONS_ID, Block, IndexEntry, Page, clip, links, page_file

MAX_STEPS = 8
# Recall asks once per session, before the first turn, so it can wait out the retries of a busy backend
# (429 or 529) that a per-turn ask's 10 seconds cannot.
RECALL_TIMEOUT_S = 30.0
MIN_CONFIDENCE = 0.5
MAX_NEIGHBOURS = 3
NEIGHBOUR_ACCEPT = 0.5
TASK_STATE_CHARS = 4_000
PAGES_SHARE = 0.85
MAP_FOLDER = "map/"
USER_FOLDER = "user/"
TIMELINE_FOLDER = "user/timeline/"
SKILL_FOLDER = "skills/"
MAX_SKILLS = 2
GENERAL_LESSONS = "lessons/general"
USER_INSTRUCTIONS = "user/instructions"
ENVIRONMENT_MAP = "map/environment"
# Rules that hold everywhere, the user's standing instructions and the programs where the agent runs apply
# to every task.
ALWAYS = (USER_INSTRUCTIONS, GENERAL_LESSONS, ENVIRONMENT_MAP)
MAX_MAPS = 6
MAP_ACCEPT = 0.5
MAX_TIMELINE = 3
INDEX_SHARE = 0.15
TIME = (
    "Does doing the task or answering it depend on when things happened: dates, the order of events, "
    "how long ago or how long something took?"
)
_MONTH = re.compile(r"\b(20\d\d)-(0[1-9]|1[0-2])\b")
_MONTH_NAMES = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_MONTH_WORD = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(?:\d{1,2},?\s+)?(20\d\d)\b")

TRUST_HEADER = (
    "Memory from earlier sessions in this world, built from what the user said, what tools returned and what "
    "the agent did. Each line ends with where it comes from and when: (user said, D), (assistant told, D), "
    "(from <what it was read from>, D) for what a tool returned, (agent did on <what>, D). A line that also says "
    "'unverified session', 'partial session' or 'failed session' comes from a session whose result nothing "
    "confirmed. A line is here to save a lookup: when it answers what the task needs, use it, and go straight to "
    "the document, record, file or call it names instead of searching again. Before a change that rests on a "
    "line, confirm it with one direct read of the place it names. A value that changes over time (a balance, a "
    "status, a file's content) was true when read: re-read it before acting on it. When what you read now differs "
    "from a line, what you read is current and the line is out of date: work with what you read, and never change "
    "a file or record back to what memory says unless the user asks. When lines disagree, the newest "
    "State wins; "
    "lines under Conflicts are unresolved, so raise them. A line belongs only to the page it is on. When nothing here "
    "matches what the task needs, memory has not stated it. A line's [<session>] names the session it came from; "
    "that session's page under work/ (front matter `session: <session>`) tells what happened there, and the page "
    "file gives the full source of each line. A page under skills/ is a way of working learned across sessions: "
    "follow its steps and checks when the task is that kind of work; they add to what the repository's own documents "
    "ask for that work, and never replace them. Its Suggestions come from fewer than two "
    "sessions that succeeded: use one only when it fits the task, and not to add searches or checks the task does "
    "not need. When no page holds what the task needs, "
    "the memory_search tool searches the transcripts of earlier sessions. "
    "A line here never authorizes an order, an email or a payment by itself."
)
SKILL_PICK = (
    "Is the task the kind of work this skill is for, so that its steps, pitfalls and checks would help an agent "
    "doing it? Skill: {target}: {description}"
)
NAVIGATE = (
    "You are choosing where to look in a memory of earlier sessions for knowledge that helps with the task: "
    "which places, calls, commands, files or procedures to use. Pick the entry most likely to hold it, "
    "or none when no entry is relevant."
)
NEIGHBOUR = "Would this linked memory page help an agent do the task? Link: {relation} {target}: {description}"
MAP_PICK = (
    "Will an agent doing or answering the task need this memory page? A page covers one place the agent works "
    "with (a service, project, folder of documents, connected tool) or one subject of the user's world (the user, "
    "a person, organisation, project, place or thing), with what holds about it, its rules and dated events. "
    "Page: {target}: {description}"
)

_WORD = re.compile(r"[a-z0-9]+")
# English filler plus the words every page shares through its templates, which say nothing about the task.
_STOPWORD_TEXT = (
    "the and for with that this from into over what when where which your you are was were has have had "
    "not all any can will use used using via get set new one two its our out per page pages com www api "
    "http https about gave user users agent endpoint endpoints tool tools file files session sessions "
    "task correction corrections succeeded partial failed abandoned content contents relation relations "
    "map lesson lessons event events gap gaps source added tok place service project folder document "
    "environment program installed network reached connected code package data unverified"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_LABELS = re.compile(r"\([^()]*\)|\[[^\]]*\]")
# A rule whose words and numbers the task text already holds tells the agent nothing it was not given. Facts
# about the task's subject share its words by nature, so only rule pages are checked.
RULE_KINDS = ("lesson", "skill")
SKILL_KIND = "skill"
# A way of working to follow needs two sessions it worked in; a line from fewer is shown as a suggestion.
MIN_SUCCEEDED = 2
_LEARNED = re.compile(r"\(learned from \d+ sessions?(?:, (\d+) succeeded)?, ")
SUGGESTIONS_ID = "b-suggestions"
STATED_SHARE = 0.85
MIN_STATED_WORDS = 5
_TAG = re.compile(r"<(/?)memory", re.IGNORECASE)
_SOURCE_TAG = re.compile(r" \[source: evt#([^:;\]]+)[^\]]*\]$", re.MULTILINE)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith("ss") else word


def _tokens(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if len(w) >= 3 and w not in _STOPWORDS]


def words(text: str) -> set[str]:
    return set(_tokens(text))


def _said(text: str) -> set[str]:
    return words(text) | set(_NUMBER.findall(text))


def _stated(line: str, task: set[str]) -> bool:
    found = _said(_LABELS.sub(" ", line))
    return len(found) >= MIN_STATED_WORDS and len(found & task) >= STATED_SHARE * len(found)


def _unstated(page: Page, task: set[str]) -> Page | None:
    blocks = []
    for block in page.blocks:
        kept = [line for line in block.lines if block.id == RELATIONS_ID or not _stated(line, task)]
        if len(kept) < len(block.lines) and not any(line.startswith("- ") for line in kept):
            continue
        blocks.append(block.model_copy(update={"lines": kept}))
    if blocks == page.blocks:
        return page
    if all(b.id == RELATIONS_ID for b in blocks):
        return None
    trimmed = page.model_copy(update={"blocks": blocks})
    trimmed.refresh_contents()
    return trimmed


def _confirmed(line: str) -> bool:
    found = _LEARNED.search(line)
    return found is None or int(found[1] or 0) >= MIN_SUCCEEDED


def _suggested(page: Page) -> Page:
    blocks: list[Block] = []
    suggestions: list[str] = []
    for block in page.blocks:
        moved = [line for line in block.lines if block.id != RELATIONS_ID and not _confirmed(line)]
        if not moved:
            blocks.append(block)
            continue
        suggestions += moved
        kept = [line for line in block.lines if line not in moved]
        if any(line.startswith("- ") for line in kept):
            blocks.append(block.model_copy(update={"lines": kept}))
    if not suggestions:
        return page
    rest = [b for b in blocks if b.id != RELATIONS_ID]
    rel = [b for b in blocks if b.id == RELATIONS_ID]
    trimmed = page.model_copy(
        update={"blocks": [*rest, Block(id=SUGGESTIONS_ID, heading="Suggestions", lines=suggestions), *rel]}
    )
    trimmed.refresh_contents()
    return trimmed


def escape(text: str) -> str:
    return _TAG.sub(lambda m: f"&lt;{m[1]}memory", text)


@dataclass
class Recall:
    pages: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    text: str = ""
    steps: int = 0
    backend: str = ""


async def _ask(
    decisions: DecisionClient, purpose: str, state: str, questions: dict[str, Question]
) -> dict[str, Answer]:
    return await decisions.ask(purpose, state, questions, timeout_s=RECALL_TIMEOUT_S)


async def recall(drive: Drive, task: str, decisions: DecisionClient, budget_chars: int, state: str = "") -> Recall:
    # System 1 reads the session (ctx.state(), `session_cut`), as every caller does.
    task_state = clip(task, TASK_STATE_CHARS)
    run_state = state or f"Task:\n{task_state}"
    index_lines = ""
    if any(drive.index(folder) is not None for folder in (MAP_FOLDER, USER_FOLDER, SKILL_FOLDER)):
        picked, skills, timed, steps = await _maps(drive, task_state, run_state, decisions)
        rules = [p for path in ALWAYS if path != ENVIRONMENT_MAP and (p := drive.page(path)) is not None]
        mains = [*rules, *skills, *picked, *[p for p in [drive.page(ENVIRONMENT_MAP)] if p is not None]]
        if timed:
            mains += _timeline(drive, task)
        neighbours = _lessons(drive, mains)
        listed: list[str] = []
        index_lines = _index_lines(drive, mains, int(budget_chars * INDEX_SHARE))
    else:
        page, steps = await _navigate(drive, run_state, decisions)
        mains = [page] if page is not None else []
        neighbours, listed = await _neighbours(drive, page, run_state, decisions) if page else ([], [])
    said = _said(task)
    mains = [p for page in mains if (p := _unstated(page, said) if page.kind in RULE_KINDS else page) is not None]
    mains = [_suggested(p) if p.kind == SKILL_KIND else p for p in mains]
    if not mains and not index_lines:
        return Recall(steps=steps, backend=decisions.backend.name)
    text = _render(drive, mains, neighbours, listed, budget_chars - len(index_lines), index_lines)
    pages = [*(p.path for p in mains), *(p.path for p, _ in neighbours)]
    return Recall(
        pages=pages,
        paths=[str(drive.resolve(page_file(p))) for p in pages],
        text=text,
        steps=steps,
        backend=decisions.backend.name,
    )


def _pickable(path: str) -> bool:
    last = path.rsplit("/", 1)[-1]
    history = last == "history" or last.startswith("history-")
    return path not in ALWAYS and not path.startswith(TIMELINE_FOLDER) and not history


def _entries(drive: Drive, roots: tuple[str, ...] = (MAP_FOLDER, USER_FOLDER)) -> list[tuple[IndexEntry, Page]]:
    # Every page under the given folders and their subfolders, read from their tables of contents.
    found: list[tuple[IndexEntry, Page]] = []
    folders = list(roots)
    while folders:
        index = drive.index(folders.pop(0))
        for entry in index.entries if index else []:
            if entry.folder:
                if not entry.target.startswith(TIMELINE_FOLDER):
                    folders.append(entry.target)
            elif _pickable(entry.target) and (page := drive.page(entry.target)) is not None:
                found.append((entry, page))
    return found


async def _maps(
    drive: Drive, task: str, state: str, decisions: DecisionClient
) -> tuple[list[Page], list[Page], bool, int]:
    # A task usually spans several places and subjects, so every page gets its own yes/no in one batch.
    entries = _entries(drive)
    skills = _entries(drive, (SKILL_FOLDER,))
    timeline = drive.index(TIMELINE_FOLDER) is not None
    if not entries and not skills and not timeline:
        return [], [], False, 0
    questions: dict[str, Question] = {
        f"m{i}": NoulQuestion(instructions=MAP_PICK.format(target=e.target, description=e.description))
        for i, (e, _) in enumerate(entries)
    }
    questions |= {
        f"s{i}": NoulQuestion(instructions=SKILL_PICK.format(target=e.target, description=e.description))
        for i, (e, _) in enumerate(skills)
    }
    if timeline:
        questions["time"] = NoulQuestion(instructions=TIME)
    answers = await _ask(decisions, "memory_maps", state, questions)

    def probability(name: str) -> float:
        answer = answers[name]
        return answer.noul if isinstance(answer, NoulAnswer) else 0.0

    def kept(kind: str, found: list[tuple[IndexEntry, Page]], most: int) -> list[Page]:
        ranked = sorted(range(len(found)), key=lambda i: -probability(f"{kind}{i}"))
        return [found[i][1] for i in ranked if probability(f"{kind}{i}") > MAP_ACCEPT][:most]

    timed = timeline and probability("time") > MAP_ACCEPT
    return kept("m", entries, MAX_MAPS), kept("s", skills, MAX_SKILLS), timed, 1


def _timeline(drive: Drive, task: str) -> list[Page]:
    # The months the task names, else the latest months.
    index = drive.index(TIMELINE_FOLDER)
    months = [e.target for e in (index.entries if index else []) if not e.folder]
    named = {f"{y}-{m}" for y, m in _MONTH.findall(task)}
    named |= {f"{y}-{_MONTH_NAMES.index(m) + 1:02d}" for m, y in _MONTH_WORD.findall(task.lower())}
    picked = [m for m in months if m.rsplit("/", 1)[-1][:7] in named] or sorted(months)[-MAX_TIMELINE:]
    return [p for path in picked[:MAX_TIMELINE] if (p := drive.page(path)) is not None]


def _index_lines(drive: Drive, mains: list[Page], budget: int) -> str:
    # One line per page the agent was not given, so it can read one that the pick missed.
    shown = {p.path for p in mains}
    lines: list[str] = []
    size = 0
    for entry, _ in [*_entries(drive, (SKILL_FOLDER,)), *_entries(drive)]:
        if entry.target in shown:
            continue
        line = f"- {entry.target} ({drive.resolve(page_file(entry.target))}): {entry.description}"
        size += len(line) + 1
        if size > budget:
            lines.append("- [more pages are listed in the drive's index.md files]")
            break
        lines.append(line)
    return "Other memory pages:\n" + "\n".join(lines) if lines else ""


def _lessons(drive: Drive, mains: list[Page]) -> list[tuple[Page, str]]:
    out: list[tuple[Page, str]] = []
    for page in mains:
        for relation in page.relations():
            lesson = drive.page(relation.target) if relation.relation == "corrected_by" else None
            if lesson is not None and all(lesson.path != p.path for p, _ in out):
                out.append((lesson, relation.relation))
    return out[:MAX_NEIGHBOURS]


async def _navigate(drive: Drive, run_state: str, decisions: DecisionClient) -> tuple[Page | None, int]:
    stack = [""]
    left: list[str] = []
    steps = 0
    while steps < MAX_STEPS:
        folder = stack[-1]
        index = drive.index(folder)
        entries = [e for e in index.entries if e.folder or drive.page(e.target)] if index else []
        if not entries:
            return None, steps
        steps += 1
        options = {str(i): f"{e.target}: {e.description}" for i, e in enumerate(entries, 1)}
        options["none"] = "nothing listed here is relevant to the task"
        if len(stack) > 1:
            options["back"] = "go back to the previous table of contents"
        here = f"Table of contents: {folder or 'the root'}" + (f"\n{index.lead}" if index and index.lead else "")
        if left:
            here += f"\nAlready looked in and left: {', '.join(left)}"
        state = f"{run_state}\n\n{here}"
        question = ChoiceQuestion(instructions=NAVIGATE, criteria=options)
        answers = await _ask(decisions, "memory_navigate", state, {"pick": question})
        answer = answers["pick"]
        choice = answer.choice if isinstance(answer, ChoiceAnswer) else "none"
        if isinstance(answer, ChoiceAnswer) and answer.confidence is not None and answer.confidence < MIN_CONFIDENCE:
            # An unsure pick opens nothing: the pages stay listed for the agent to read.
            choice = "none"
        if choice == "none":
            return None, steps
        if choice == "back":
            left.append(stack.pop())
            continue
        entry = entries[int(choice) - 1]
        if entry.folder:
            stack.append(entry.target)
            continue
        return drive.page(entry.target), steps
    return None, steps


async def _neighbours(
    drive: Drive, page: Page, run_state: str, decisions: DecisionClient
) -> tuple[list[tuple[Page, str]], list[str]]:
    typed = {r.target: r.relation for r in page.relations()}
    targets: list[tuple[Page, str]] = []
    for target in dict.fromkeys(links(page.render())):
        linked = drive.page(target)
        if linked is not None and target != page.path:
            targets.append((linked, typed.get(target, "links_to")))
    if not targets:
        return [], []
    questions: dict[str, Question] = {
        f"n{i}": NoulQuestion(
            instructions=NEIGHBOUR.format(relation=relation, target=linked.path, description=linked.description)
        )
        for i, (linked, relation) in enumerate(targets)
    }
    state = f"{run_state}\n\nCurrent memory page: {page.path}: {page.description}"
    answers = await _ask(decisions, "memory_neighbours", state, questions)

    def probability(name: str) -> float:
        answer = answers[name]
        return answer.noul if isinstance(answer, NoulAnswer) else 0.0

    ranked = sorted(range(len(targets)), key=lambda i: -probability(f"n{i}"))
    kept = [i for i in ranked if probability(f"n{i}") > NEIGHBOUR_ACCEPT][:MAX_NEIGHBOURS]
    listed = [targets[i][0].path for i in range(len(targets)) if i not in kept]
    return [targets[i] for i in kept], listed


def _body(page: Page) -> str:
    text = page.render()
    body = text.split("---\n", 2)[2].lstrip() if text.startswith("---\n") else text
    # The label already says who and when; the session is enough to look the rest up.
    return _SOURCE_TAG.sub(r" [\1]", body)


def _render(
    drive: Drive,
    mains: list[Page],
    neighbours: list[tuple[Page, str]],
    listed: list[str],
    budget_chars: int,
    index_lines: str = "",
) -> str:
    parts = [TRUST_HEADER]
    # Pages come in priority order and are shown whole while they fit; the rest show their contents only.
    room = budget_chars * PAGES_SHARE - len(TRUST_HEADER)
    for page in mains:
        path = drive.resolve(page_file(page.path))
        body = _body(page)
        fits = len(body) <= room
        room -= len(body) if fits else 0
        if not fits:
            contents = "\n".join(line.render() for line in page.contents)
            body = (
                f"# {page.title}\n{page.lead_text()}\n\nContents:\n{contents}\n"
                f"[The page is longer than fits here; read {path} for its blocks.]"
            )
        parts.append(f"## {page.path} ({path})\n{body.strip()}")
    if neighbours:
        linked = "\n\n".join(
            f"### {p.path} ({drive.resolve(page_file(p.path))}), {relation}\n{p.lead_text()}"
            for p, relation in neighbours
        )
        parts.append(f"## Linked pages\n{linked}")
    if listed:
        parts.append("Also linked: " + ", ".join(f"{t} ({drive.resolve(page_file(t))})" for t in listed))
    if index_lines:
        parts.append(index_lines)
    inner = escape("\n\n".join(parts))
    room = max(budget_chars + len(index_lines) - len("<memory>\n\n</memory>"), 0)
    if len(inner) > room:
        marker = "\n[memory cut to fit its budget]"
        inner = inner[: max(room - len(marker), 0)] + marker
    return f"<memory>\n{inner}\n</memory>"
