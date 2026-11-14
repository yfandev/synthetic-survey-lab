from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

import httpx

from intern.context import INITIAL_CHARS_PER_TOKEN
from intern.decisions import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    NoulAnswer,
    NoulQuestion,
    Question,
    System1Unavailable,
)
from intern.journal import Journal
from intern.memory.drive import main_commit, tracked_files
from intern.memory.format import (
    FOLDER_INDEX,
    RELATIONS_ID,
    SCHEMA_FILE,
    UNREAD_FOLDERS,
    WORLD_FILE,
    Block,
    FormatError,
    Page,
    build_indexes,
    clip,
    drive_world,
    inert,
    is_page_file,
    page_file,
    parse_page,
    parse_schema,
    source_of,
    strip_source,
    world_file,
)
from intern.memory.lint import lint
from intern.memory.steward.extract import (
    INJECTION,
    QUOTE_CHARS,
    Extract,
    Observation,
    UserNote,
    describe,
    extract,
    file_pages,
)
from intern.memory.steward.git import branch_exists, commit_proposal
from intern.memory.steward.write import (
    BLOCKS,
    GENERAL_LESSONS,
    LEAD_BLOCKS,
    OTHER_WORK,
    SESSIONS_FOLDER,
    SKILL_FOLDER,
    USAGE_FILE,
    USER_FOLDER,
    USER_INSTRUCTIONS,
    WORK_FOLDER,
    Gap,
    Placed,
    StewardError,
    archive,
    forget,
    is_history,
    item_body,
    read_usage,
    skill_path,
    stale,
    stewarded,
    stewarded_turns,
    subject_path,
    tally,
    topic_of,
    user_turns,
    with_vocabulary,
    with_work_kind,
    write,
)
from intern.memory.steward.writer import (
    GENERAL,
    INSTRUCTIONS,
    ME,
    NEW,
    Draft,
    Item,
    Learned,
    Past,
    ReplayWriter,
    Skill,
    Story,
    Writer,
    condense,
    draft,
    learn,
    numbers,
    similar,
    trace,
)

STEWARD_VERSION = "47"
KEEP = 0.7
NOTE_KEEP = 0.5
MAYBE_CORRECTION = 0.3
MAX_STATE_CHARS = int(32_768 * INITIAL_CHARS_PER_TOKEN)
TASK_STATE_CHARS = 2_000
DESCRIPTION_CHARS = 90

SUCCEEDED = "succeeded"
UNVERIFIED = "unverified"
OUTCOMES = {
    SUCCEEDED: "the task was done as asked",
    UNVERIFIED: "the agent says it finished, but nothing in the session checks it",
    "partial": "some of the task was done",
    "failed": "the task was attempted and not done",
    "abandoned": "the session stopped before the task was really attempted",
}
OUTCOME = (
    "How did this session end? Judge from the task, the agent's final reply and whether its calls returned errors."
)
VERIFY = (
    "Does step {name} check that the work's result is right, for example by reading back what was changed, "
    "running a check or test on it, or comparing it with what was asked? Only a step that worked counts."
)
CONFIRM = (
    "Does user message {name} confirm that the result is right, by saying it got or checked what it asked for? "
    "Thanks or a goodbye alone does not confirm it."
)
MAX_VERIFY_STEPS = 12
CHANGED = (
    "Did step {name} change something outside the session, such as creating, updating or deleting a record, "
    "submitting a request or application, or sending something on to another person or system, rather than only "
    "reading, searching or listing? A reply to the user is not a change. Judge from the call, its result and any "
    "description of the tool the session printed."
)
REPLY = (
    "Is the result of step {name} a message written by the person the agent is helping, such as their answer to "
    "something the agent sent them, rather than output of a program or service?"
)
MAX_CHANGE_STEPS = 48
_EXIT_CODE = re.compile(r"\s*\[exit code -?\d+\]\s*$")
WORTH = (
    "Is action {name} worth remembering for future tasks in this workspace: a reusable fact about where something "
    "is or how a service, project, folder of documents, tool or the environment is used, not a one-off detail?"
)
CORRECTION = (
    "Is user message {name} a correction of how the agent worked, such as the right place, command, file, call "
    "or procedure to use?"
)
LASTING = (
    "Will item {name} still be true and useful in a later session in the same world, for a different task or "
    "question? Yes for where things are, how they work, what fails, facts about the user and the people and things "
    "they deal with, events, standing rules, preferences and what the assistant told the user. No for what only "
    "served this one task, such as a draft, a lookup or an intermediate result, and no when the session does not "
    "support it."
)
TASK_ONLY = (
    "Does item {name} hold only for this session's task: an instruction, scope, source or setting given for this "
    "one piece of work, in the task or in a message about it, which a later task in the same world would not "
    "share? For example which copy or file to use this time, not to edit or save anything this time, or the method, "
    "grouping or rounding asked for this request. A rule for every case of a kind (every POST handler, any change "
    "under a folder, whenever staging is checked) holds for later tasks too, even when it came in a message about "
    "this work."
)
# A method one request asked for reads like how something is done.
TASK_ONLY_KINDS = ("state", "rule", "preference", "how")
MAYBE_TASK_ONLY = 0.3
_PATH_WORD = re.compile(r"[a-z0-9]+")
RESULT = (
    "Is item {name} a value or conclusion this session worked out for its task, such as a total, average, count, "
    "ranking, percentage, chosen option or final answer, rather than something it found as it was in a file, "
    "a service or a message? What it read back from a file it wrote itself (listed under Written) is its own result. "
    "How to work such a value out is not a result."
)
RESULT_KINDS = ("state", "where", "event", "told")
ABOUT_DATA = (
    "Is item {name} about what particular files hold (their values, columns, codes or formats) rather than a "
    "way of working that helps with any task?"
)
COMPARED_KINDS = ("state", "preference", "rule", "where", "how", "fails")
LEARNED = (
    "Do the cited story lines of at least two sessions support lesson {name}, and will it help later tasks in this "
    "world? No for a lesson that only restates one task or its results, or that only names steps any capable agent "
    "takes anyway, such as editing the file the task names, adding tests or running the project's checks."
)
STALE_LESSON = (
    "Lesson {name} states a value that memory has since replaced. Does lesson {name} still hold now? It does not "
    "when it gives the replaced value as current."
)
SAME_LESSON = "Does lesson {name} say what one of the lessons memory already holds says?"
SKILL_FITS = (
    "Does lesson {name} help with the kind of work its skill names, and is the skill a kind of work later "
    "tasks would share rather than one task?"
)
SAME_SKILL = "Which skill already in memory covers the kind of work lesson {name} is for, under another name?"
MAX_SKILL_LINES = 12
MERGED = (
    "Are the lines merged line {name} replaces about the same thing, and does it keep every fact, value, name, "
    "negation and limit of them, and add nothing they do not say?"
)
DROPPED = "Does another line on the page say what line {name} says, or show that it no longer holds?"
MIN_FREED = 400
FREED_SHARE = 4
# Merging frees at most a third of the text it may rewrite.
FREED_MAX_SHARE = 3
INVENTORY_BLOCKS = ("b-files", "b-programs", "b-calls")
_LABEL_TAIL = re.compile(r" (\([^()]*\d{4}-\d{2}-\d{2}[^()]*\))$")
_ADDED = re.compile(r"; added: (\d{4}-\d{2}-\d{2})")
_POINTER = re.compile(r"\[source: (evt#[^;\]]+)")
MAX_FILE_PAGES = 20
MAX_PAST = 30
MAX_PAST_CHARS = 24_000
_OUTCOME_LEAD = re.compile(r"`([a-z]+)` \(session `")
CHANGE = "How does new item {name} relate to the lines memory already holds?"
STILL_HOLDS = (
    "Line {name} states a value that a new item replaced. Does line {name} still hold now? It does not when it "
    "gives the replaced value as current."
)
# Other lines that state a replaced value, wherever they are; work pages are history and keep theirs.
STATED_FOLDERS = ("map/", "user/", "lessons/", "skills/")
STATED_BLOCKS = (*LEAD_BLOCKS, "b-steps", "b-checks", "b-pitfalls", "b-told")
MAX_RETIRED = 8
# A one-digit number (a count, a step) is too common to tie two lines to one value.
MIN_STATED_DIGITS = 2
SAME_SUBJECT = "Is the subject of item {name} one of these existing subjects?"
MAX_KNOWN_CHARS = 8_000
MAX_SUBJECTS = 40
MAX_COMPARED = 5
CITED_CHARS = 300
_NAME_WORD = re.compile(r"[a-z0-9]{3,}")
WORK = (
    "Which kind of work did this session do? Pick the kind the user would file this task under: one that names the"
    " same work even when the details differ, not a catch-all that would also fit unrelated work."
)
CONTENTS = "Pick the description that best tells a reader what this block of the page holds."
BLOCK_NOUNS = {
    "b-where": "where things are and what the agent used here",
    "b-how": "how things are done here",
    "b-fails": "what does not work here",
    "b-state": "what holds now",
    "b-rules": "standing rules and what they apply to",
    "b-events": "dated events",
    "b-earlier": "what held before, and until when",
    "b-conflicts": "values that contradict each other",
    "b-told": "what the assistant told the user",
    "b-corr": "corrections users gave",
    "b-calls": "what the session used",
    "b-gaps": "what the steward left out, for review",
    "b-notes": "notes on where things are, what fails and how to work",
    "b-story": "what was asked, what the agent did and how it ended",
    "b-files": "the files and folders sessions used here",
    "b-programs": "the programs sessions ran here",
    "b-task": "what applied only to this session's task",
    "b-results": "what this session worked out for its task",
    "b-steps": "the steps that worked",
    "b-pitfalls": "what went wrong or wasted steps",
    "b-checks": "checks to always run",
}


class Deferred(Exception):
    pass


class Skipped(Exception):
    pass


@dataclass
class Proposal:
    session_id: str
    branch: str
    files: dict[str, str]
    commit: str = ""
    # What the steward left out and why: for the operator's log, never a page an agent loads.
    gaps: list[Gap] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


class _Decision:
    def __init__(self, decisions: DecisionClient, run_state: str = "") -> None:
        self.decisions = decisions
        self.run_state = run_state

    async def ask(self, purpose: str, state: str, questions: dict[str, Question]) -> dict[str, Answer]:
        # Every ask reads the session as System 1 saw it during the run (ctx.state()), then what this ask is about.
        if self.run_state:
            state = f"The session:\n{self.run_state}\n\n{state}"
        if len(state) > MAX_STATE_CHARS:
            raise Deferred(f"the {purpose} state is {len(state)} characters, over System 1's limit")
        try:
            return await self.decisions.ask(purpose, state, questions)
        except System1Unavailable as e:
            raise Deferred(f"{purpose}: {e}; the steward writes only on System 1 answers") from e


def _task(task: str) -> str:
    return clip(task, TASK_STATE_CHARS)


def _state(ex: Extract) -> str:
    lines = [
        f"Workspace: {Path(ex.cwd).name}",
        f"Task: {_task(ex.task)}",
        f"Tool calls: {ex.calls}, of which {ex.failed} failed",
        f"Last stop: {ex.stop}",
        f"Final reply: {ex.final or '(none)'}",
    ]
    if ex.observations:
        lines.append("Observations:")
        lines += [
            f"o{i}. {describe(o)} -> {o.place} ({o.place_kind}) [{'ok' if o.ok else 'error'}]"
            for i, o in enumerate(ex.observations)
        ]
    notes = [n for n in ex.notes if not n.injected]
    if notes:
        lines.append("User messages after the task:")
        lines += [f'u{i}. "{n.text}"' for i, n in enumerate(notes)]
    return "\n".join(lines)


def _noul(answer: Answer | None) -> float:
    return answer.noul if isinstance(answer, NoulAnswer) else 0.0


async def _triage(decision: _Decision, ex: Extract) -> tuple[str, list[Observation], list[UserNote], list[Gap]]:
    state = _state(ex)
    gaps: list[Gap] = [
        Gap("a user message matched a prompt-injection pattern; it is not quoted", n.pointer)
        for n in ex.notes
        if n.injected
    ]
    # What worked is a fact about its place; whether a failure is worth remembering is a judgment.
    kept = [o for o in ex.observations if o.ok]
    asked = [(i, o) for i, o in enumerate(ex.observations) if not o.ok]
    if asked:
        questions: dict[str, Question] = {
            f"o{i}": NoulQuestion(instructions=WORTH.format(name=f"o{i}")) for i, _ in asked
        }
        answers = await decision.ask("steward_triage", state, questions)
        for i, observation in asked:
            p = _noul(answers.get(f"o{i}"))
            if p >= KEEP:
                kept.append(observation)
            else:
                gaps.append(Gap(f"not kept, worth remembering p={p:.2f}: {describe(observation)}", observation.pointer))
    kept.sort(key=ex.observations.index)

    outcome_q = ChoiceQuestion(instructions=OUTCOME, criteria=OUTCOMES)
    answers = await decision.ask("steward_outcome", state, {"outcome": outcome_q})
    outcome = answer.choice if isinstance(answer := answers.get("outcome"), ChoiceAnswer) else "failed"
    notes = [n for n in ex.notes if not n.injected]
    if outcome == SUCCEEDED and not await _verified(decision, ex, notes):
        outcome = UNVERIFIED

    lessons: list[UserNote] = []
    if notes:
        questions = {f"u{i}": NoulQuestion(instructions=CORRECTION.format(name=f"u{i}")) for i in range(len(notes))}
        answers = await decision.ask("steward_correction", state, questions)
        for i, note in enumerate(notes):
            p = _noul(answers.get(f"u{i}"))
            if p >= KEEP:
                lessons.append(note)
            elif p >= MAYBE_CORRECTION:
                gaps.append(Gap(f"a user message that may be a correction, p={p:.2f}; not quoted", note.pointer))
    return outcome, kept, await _place_lessons(decision, state, notes, lessons, kept), gaps


async def _step_changes(decision: _Decision, ex: Extract) -> None:
    # A command-line or custom tool does not show whether it reads or writes; what it was called with and what it
    # returned usually do ("Transfer successful", "Application submitted"). Such a tool may also relay the reply of
    # the person the agent is helping, which then counts as a user message.
    unknown = [(i, s) for i, s in enumerate(ex.steps) if s.change is None and s.ok][:MAX_CHANGE_STEPS]
    if not unknown:
        return
    lines = [f"Task: {_task(ex.task)}", "Steps:"]
    lines += [f"s{i}. $ {s.call}\n    -> {s.result}" for i, s in unknown]
    questions: dict[str, Question] = {}
    for i, _ in unknown:
        questions[f"s{i}"] = NoulQuestion(instructions=CHANGED.format(name=f"s{i}"))
        questions[f"r{i}"] = NoulQuestion(instructions=REPLY.format(name=f"s{i}"))
    answers = await decision.ask("steward_step_change", "\n".join(lines), questions)
    for i, step in unknown:
        step.change = _noul(answers.get(f"s{i}")) >= KEEP
        if _noul(answers.get(f"r{i}")) >= KEEP and (text := _EXIT_CODE.sub("", " ".join(step.output.split()))):
            ex.notes.append(
                UserNote(
                    text=text[:QUOTE_CHARS], pointer=step.pointer, place=None, injected=bool(INJECTION.search(text))
                )
            )


async def _verified(decision: _Decision, ex: Extract, notes: list[UserNote]) -> bool:
    # Success needs evidence: a step that checked a change the session is known to have made, or a user
    # confirming the result. A read with no known change before it checks only the sources, not the result.
    first = min((i for i, s in enumerate(ex.steps) if s.change), default=None)
    after = (
        []
        if first is None
        else [(i, s) for i, s in enumerate(ex.steps) if i > first and s.ok and not s.change][-MAX_VERIFY_STEPS:]
    )
    if not after and not notes:
        return False
    lines = [f"Task: {_task(ex.task)}", "Steps after the agent's first change:"]
    lines += [f"s{i}. $ {s.call}\n    -> {s.result}" for i, s in after] or ["(none)"]
    if notes:
        lines.append("User messages after the task:")
        lines += [f'u{i}. "{n.text}"' for i, n in enumerate(notes)]
    questions: dict[str, Question] = {f"s{i}": NoulQuestion(instructions=VERIFY.format(name=f"s{i}")) for i, _ in after}
    questions |= {f"u{i}": NoulQuestion(instructions=CONFIRM.format(name=f"u{i}")) for i in range(len(notes))}
    answers = await decision.ask("steward_verify", "\n".join(lines), questions)
    return any(_noul(answers.get(name)) >= KEEP for name in questions)


async def _place_lessons(
    decision: _Decision, state: str, notes: list[UserNote], lessons: list[UserNote], kept: list[Observation]
) -> list[UserNote]:
    places = list(dict.fromkeys(o.place for o in kept))
    if not lessons or len(places) <= 1:
        return [n.model_copy(update={"place": places[0] if places else None}) for n in lessons]
    criteria = {str(i): place for i, place in enumerate(places, 1)} | {"none": "none of these pages"}
    questions: dict[str, Question] = {
        f"l{i}": ChoiceQuestion(
            instructions=f"Which memory page does the correction in user message u{notes.index(note)} concern?",
            criteria=criteria,
        )
        for i, note in enumerate(lessons)
    }
    answers = await decision.ask("steward_place", state, questions)
    placed: list[UserNote] = []
    for i, note in enumerate(lessons):
        answer = answers.get(f"l{i}")
        choice = answer.choice if isinstance(answer, ChoiceAnswer) else "none"
        placed.append(note.model_copy(update={"place": places[int(choice) - 1] if choice.isdigit() else None}))
    return placed


def _parsed(files: dict[str, str], path: str) -> Page | None:
    text = files.get(page_file(path))
    try:
        return parse_page(path, text) if text else None
    except FormatError:
        return None


def _subjects(files: dict[str, str]) -> dict[str, Page]:
    found: dict[str, Page] = {}
    for name in files:
        path = name[:-3]
        if not name.startswith(USER_FOLDER) or not name.endswith(".md") or is_history(path):
            continue
        if (page := _parsed(files, path)) is not None and page.kind == "subject":
            found[path] = page
    return found


def _names(page: Page) -> list[str]:
    aliases = [a.strip() for a in page.meta.get("aliases", "").split(",") if a.strip()]
    return [n for n in [page.meta.get("name", ""), *aliases] if n]


def _session_text(ex: Extract) -> str:
    return " ".join([*(t.text for t in ex.turns), *(f"{s.call} {s.output}" for s in ex.steps)])


def _mentioned(subjects: dict[str, Page], text: str) -> list[str]:
    # A subject is mentioned by one of its names, or by a capitalised word of a name ("Priya" for "Priya Shah").
    lowered = text.lower()
    found: list[str] = []
    for path, page in subjects.items():
        names = [n.lower() for n in _names(page) if len(n) >= 3]
        words = {w for n in _names(page) for w in n.split() if len(w) >= 3 and w[0].isupper()}
        if (
            path == ME
            or any(re.search(rf"(?<![\w]){re.escape(n)}(?![\w])", lowered) for n in names)
            or any(re.search(rf"(?<![\w]){re.escape(w)}(?![\w])", text) for w in words)
        ):
            found.append(path)
    return found[:MAX_SUBJECTS]


def _lines(page: Page | None, base_ids: tuple[str, ...] = ()) -> list[str]:
    if page is None:
        return []
    return [
        line
        for block in page.blocks
        if block.id != RELATIONS_ID and (not base_ids or "-".join(block.id.split("-")[:2]) in base_ids)
        for line in block.bullets
    ]


def _parts(files: dict[str, str], path: str) -> list[str]:
    # The part pages a full page moved its topics to, which hold what still holds about it.
    prefix = f"{path}/"
    found: list[str] = []
    for name in files:
        child = name[: -len(".md")] if is_page_file(name) else ""
        if child.startswith(prefix) and "/" not in child[len(prefix) :] and not is_history(child):
            page = _parsed(files, child)
            if page is not None and page.meta.get("part"):
                found.append(child)
    return sorted(found)


def _known(files: dict[str, str], paths: list[str]) -> list[str]:
    lines: list[str] = []
    size = 0
    with_parts = [q for path in paths for q in (path, *_parts(files, path))]
    for path in dict.fromkeys([*with_parts, GENERAL_LESSONS, USER_INSTRUCTIONS, ME]):
        page = _parsed(files, path)
        for block in page.blocks if page else []:
            if block.id == RELATIONS_ID:
                continue
            for line in block.bullets:
                topic = topic_of(block)
                text = f"[{path}{', topic ' + topic if topic else ''}] {strip_source(line)[2:]}"
                size += len(text)
                if size > MAX_KNOWN_CHARS:
                    return lines
                lines.append(text)
    return lines


def _target(item: Item) -> str:
    if item.page == GENERAL:
        return GENERAL_LESSONS
    if item.page == INSTRUCTIONS:
        return USER_INSTRUCTIONS
    return item.page


async def _match_subjects(
    decision: _Decision, ex: Extract, items: list[Item], subjects: dict[str, Page]
) -> dict[int, str]:
    # A new subject is an existing one when a name matches, or when System 1 says so among subjects sharing a word.
    paths: dict[int, str] = {}
    by_name = {n.lower(): path for path, page in subjects.items() for n in _names(page)}
    questions: dict[str, Question] = {}
    options: dict[str, list[str]] = {}
    for i, item in enumerate(items):
        if item.page != NEW:
            continue
        exact = next((by_name[n.lower()] for n in [item.subject, *item.aliases] if n.lower() in by_name), None)
        if exact is not None:
            paths[i] = exact
            continue
        words = set(_NAME_WORD.findall(" ".join([item.subject, *item.aliases]).lower()))
        near = [p for p, page in subjects.items() if words & set(_NAME_WORD.findall(" ".join(_names(page)).lower()))]
        if near:
            near = near[:MAX_COMPARED]
            options[f"i{i}"] = near
            criteria = {str(k): f"{p}: {subjects[p].description}" for k, p in enumerate(near, 1)}
            questions[f"i{i}"] = ChoiceQuestion(
                instructions=SAME_SUBJECT.format(name=f"i{i}"),
                criteria=criteria | {"none": "a different subject from all of these"},
            )
        paths[i] = subject_path(item.group, item.subject)
    if questions:
        listed = "\n".join(
            f"i{i}. {item.subject} ({', '.join(item.aliases) or 'no other names'}): {item.text}"
            for i, item in enumerate(items)
            if f"i{i}" in questions
        )
        state = f"Task: {_task(ex.task)}\nNew subjects:\n{listed}"
        answers = await decision.ask("steward_subject", state, questions)
        for name, near in options.items():
            answer = answers.get(name)
            choice = answer.choice if isinstance(answer, ChoiceAnswer) else "none"
            if choice.isdigit() and 0 < int(choice) <= len(near):
                paths[int(name[1:])] = near[int(choice) - 1]
    return paths


def _cited(ex: Extract, item: Item) -> str:
    texts = [t.text for t in ex.turns if t.pointer == item.pointer] + [
        f"{s.call} -> {s.result}" for s in ex.steps if s.pointer == item.pointer
    ]
    return " ".join(" ".join(texts).split())[:CITED_CHARS]


def _data_home(ex: Extract, p: Placed) -> str:
    # The file page of the one file the cited step names, or the folder page all the files it names share.
    calls = " ".join(s.call for s in ex.steps if s.pointer == p.item.pointer)
    named = [path for path, key in file_pages(ex.observations).items() if PurePosixPath(key).name in calls]
    if len(named) == 1:
        return named[0]
    folders = {path.rsplit("/", 1)[0] for path in named}
    return folders.pop() if len(folders) == 1 else ""


async def _lasting(decision: _Decision, ex: Extract, placed: list[Placed]) -> tuple[list[Placed], list[Gap]]:
    if not placed:
        return [], []
    listed = "\n".join(
        f"i{i}. [{p.item.kind} on {p.path}] {item_body(p.item)}\n    from: {_cited(ex, p.item)}"
        for i, p in enumerate(placed)
    )
    written = sorted({o.key for o in ex.observations if o.kind == "file" and o.change})
    made = {PurePosixPath(o.key).name for o in ex.observations if o.kind == "file" and o.made}
    state = (
        f"Session date: {ex.date}\nTask: {_task(ex.task)}\nWritten: {', '.join(written) or '(no files)'}\n"
        f"Proposed memory items:\n{listed}"
    )
    questions: dict[str, Question] = {
        f"i{i}": NoulQuestion(instructions=LASTING.format(name=f"i{i}")) for i in range(len(placed))
    }
    questions |= {
        f"t{i}": NoulQuestion(instructions=TASK_ONLY.format(name=f"i{i}"))
        for i, p in enumerate(placed)
        if p.item.kind in TASK_ONLY_KINDS
    }
    questions |= {
        f"r{i}": NoulQuestion(instructions=RESULT.format(name=f"i{i}"))
        for i, p in enumerate(placed)
        if p.item.kind in RESULT_KINDS
    }
    # A general lesson loads in every task, so one about what the files its step read hold goes to their page.
    homes = {i: home for i, p in enumerate(placed) if p.path == GENERAL_LESSONS and (home := _data_home(ex, p))}
    questions |= {f"g{i}": NoulQuestion(instructions=ABOUT_DATA.format(name=f"i{i}")) for i in homes}
    answers = await decision.ask("steward_lasting", state, questions)
    kept: list[Placed] = []
    gaps: list[Gap] = []
    for i, p in enumerate(placed):
        noul = _noul(answers.get(f"i{i}"))
        if noul >= NOTE_KEEP:
            kept.append(p)
        else:
            gaps.append(Gap(f"item not kept, lasting p={noul:.2f}: {inert(p.item.text)}", p.item.pointer))
            continue
        if (result := _noul(answers.get(f"r{i}"))) >= NOTE_KEEP:
            p.result = True
            gaps.append(Gap(f"a result, kept on the work page, p={result:.2f}: {inert(p.item.text)}", p.item.pointer))
            continue
        # What a file the session made holds is its own output, which is wiped with the workspace.
        if p.item.kind in RESULT_KINDS and any(name in f"{p.item.text} {p.item.origin}" for name in made):
            p.result = True
            gaps.append(Gap(f"about a file the session made, on the work page: {inert(p.item.text)}", p.item.pointer))
            continue
        if i in homes and (about := _noul(answers.get(f"g{i}"))) >= NOTE_KEEP:
            gaps.append(Gap(f"a lesson about [[{homes[i]}]], p={about:.2f}: {inert(p.item.text)}", p.item.pointer))
            p.path = homes[i]
        task = _noul(answers.get(f"t{i}"))
        if task >= NOTE_KEEP:
            p.task_only = True
            gaps.append(
                Gap(f"kept on the work page only, task-only p={task:.2f}: {inert(p.item.text)}", p.item.pointer)
            )
        elif task >= MAYBE_TASK_ONLY:
            gaps.append(Gap(f"kept as lasting, task-only p={task:.2f}: {inert(p.item.text)}", p.item.pointer))
    return kept, gaps


def _overlap(a: str, b: str) -> float:
    x, y = set(_NAME_WORD.findall(a.lower())), set(_NAME_WORD.findall(b.lower()))
    return len(x & y) / len(x | y) if x and y else 0.0


async def _changes(decision: _Decision, files: dict[str, str], placed: list[Placed], mentioned: list[str]) -> None:
    # A new state, preference or rule may restate, replace or contradict what memory holds; System 1 decides which.
    # A rule is compared with the rules of every page the session is about, since a user's new rule can
    # replace one kept on another page.
    questions: dict[str, Question] = {}
    compared: dict[str, list[tuple[str, str]]] = {}
    for i, p in enumerate(placed):
        if p.change != "new" or p.item.kind not in COMPARED_KINDS:
            continue
        # A fact may already be on the page in another kind of block (a layout under Where and again under State),
        # or on a part page the page moved its topic to.
        bases = (BLOCKS["rule"][0],) if p.item.kind == "rule" else LEAD_BLOCKS
        paths = [p.path, *_parts(files, p.path)]
        if p.item.kind == "rule":
            paths = list(dict.fromkeys([*paths, USER_INSTRUCTIONS, *mentioned]))
        existing = [(path, line) for path in paths for line in _lines(_parsed(files, path), bases)]
        if not existing:
            continue
        body = item_body(p.item)
        ranked = sorted(existing, key=lambda found: -_overlap(body, strip_source(found[1])))[:MAX_COMPARED]
        compared[f"i{i}"] = ranked
        criteria = {"new": "it adds something the page does not hold yet"}
        for k in range(1, len(ranked) + 1):
            criteria[f"same{k}"] = f"it says the same as line {k}, which still holds"
            criteria[f"replaces{k}"] = f"it updates line {k}: what line {k} says no longer holds"
            criteria[f"conflicts{k}"] = f"it contradicts line {k}, with nothing saying which is newer or right"
        questions[f"i{i}"] = ChoiceQuestion(instructions=CHANGE.format(name=f"i{i}"), criteria=criteria)
    if not questions:
        return
    parts: list[str] = []
    for name, ranked in compared.items():
        p = placed[int(name[1:])]
        lines = "\n".join(f"    {k}. [{path}] {strip_source(line)[2:]}" for k, (path, line) in enumerate(ranked, 1))
        parts.append(f"{name}. [{p.item.kind} on {p.path}] {item_body(p.item)}\n  Lines memory holds:\n{lines}")
    answers = await decision.ask("steward_change", "New items:\n" + "\n".join(parts), questions)
    for name, ranked in compared.items():
        answer = answers.get(name)
        choice = answer.choice if isinstance(answer, ChoiceAnswer) else "new"
        found = re.fullmatch(r"(same|replaces|conflicts)(\d+)", choice)
        if found and 0 < int(found[2]) <= len(ranked):
            p = placed[int(name[1:])]
            p.change = found[1]
            p.old_path, p.old = ranked[int(found[2]) - 1]


async def _retired(decision: _Decision, files: dict[str, str], placed: list[Placed]) -> list[Placed]:
    # A value one line held may be stated on other pages too; each line that states it is asked whether it still
    # holds, and one that does not is retired with the line its item replaced.
    found: list[tuple[Placed, str, str]] = []
    for p in placed:
        old, new = numbers(strip_source(p.old)), numbers(item_body(p.item))
        # An item may name the value it replaced as history ("moved from 8099"); its own new value marks the old one.
        replaced = (old - new or (old if new - old else set())) if p.change == "replaces" else set()
        gone = {n for n in replaced if len(n) >= MIN_STATED_DIGITS or "." in n}
        for name in files if gone else ():
            if not is_page_file(name) or not name.startswith(STATED_FOLDERS):
                continue
            for line in _lines(_parsed(files, name[:-3]), STATED_BLOCKS):
                if line != p.old and "(until " not in line and gone & numbers(strip_source(line)):
                    found.append((p, name[:-3], line))
    found = found[:MAX_RETIRED]
    if not found:
        return []
    listed = "\n".join(
        f"c{k}. [{path}] {strip_source(line)[2:]}\n    replaced: {strip_source(p.old)[2:]}\n    by: {item_body(p.item)}"
        for k, (p, path, line) in enumerate(found)
    )
    questions: dict[str, Question] = {
        f"c{k}": NoulQuestion(instructions=STILL_HOLDS.format(name=f"c{k}")) for k in range(len(found))
    }
    answers = await decision.ask("steward_still_holds", "Lines memory holds:\n" + listed, questions)
    return [
        Placed(item=p.item, path=path, change="retires", old=line, old_path=path)
        for k, (p, path, line) in enumerate(found)
        if _noul(answers.get(f"c{k}")) < NOTE_KEEP
    ]


async def _items(
    decision: _Decision,
    writer: Writer | ReplayWriter,
    ex: Extract,
    outcome: str,
    files: dict[str, str],
    kinds: list[str],
) -> tuple[Story | None, list[Placed], list[Gap]]:
    places = list(dict.fromkeys(o.place for o in ex.observations))
    places += list(file_pages(ex.observations))[:MAX_FILE_PAGES]
    subjects = _subjects(files)
    mentioned = _mentioned(subjects, _session_text(ex))
    listed = [f"{p} ({'; '.join(_names(subjects[p]))})" for p in mentioned]
    known = _known(files, [*places, *mentioned])
    try:
        story, items, rejected = await draft(writer, ex, outcome, places, listed, known, kinds)
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        raise Deferred(f"the writer {writer.name} failed: {e}") from e
    gaps = [Gap(f"not kept, {inert(r.reason)}", r.pointer) for r in rejected]
    if outcome != SUCCEEDED:
        # What the assistant told the user in a session nothing confirmed must not read as an answer later.
        gaps += [
            Gap(f"told in a {outcome} session, not kept: {inert(i.text)}", i.pointer) for i in items if i.kind == "told"
        ]
        items = [i for i in items if i.kind != "told"]
    matched = await _match_subjects(decision, ex, items, subjects)
    placed: list[Placed] = []
    for i, item in enumerate(items):
        path = matched.get(i) or _target(item)
        if item.attribute and not path.startswith(USER_FOLDER):
            # A place's lines are grouped by topic; the attribute names a fact about a subject.
            item = replace(item, attribute="")
        own = set(_PATH_WORD.findall(path.rsplit("/", 1)[-1]))
        if item.topic and set(_PATH_WORD.findall(item.topic.lower())) <= own:
            # A topic that only names the page itself ("mail" on a mailbox) is about the whole page.
            item = replace(item, topic="")
        if any(p.path == path and similar(p.item.text, item.text) for p in placed):
            continue
        base = BLOCKS[item.kind][0]
        body = item_body(item)
        same = next((line for line in _lines(_parsed(files, path), (base,)) if similar(strip_source(line), body)), None)
        placed.append(Placed(item=item, path=path, change="same" if same else "new", old=same or ""))
    repeats = [p for p in placed if p.change == "same"]
    kept, lasting_gaps = await _lasting(decision, ex, [p for p in placed if p.change == "new"])
    gaps += lasting_gaps
    await _changes(decision, files, kept, mentioned)
    retired = await _retired(decision, files, kept)
    return story, sorted([*repeats, *kept], key=lambda p: p.item.order) + retired, gaps


def _past(files: dict[str, str], ex: Extract, outcome: str, story: Story | None) -> list[Past]:
    # Every session's story as its work page tells it, newest last, with this session's own story at the end.
    found: list[tuple[str, Past]] = []
    for name in files:
        if not name.startswith(WORK_FOLDER) or not name.endswith(".md"):
            continue
        page = _parsed(files, name[:-3])
        if page is None or page.kind != "event" or not page.meta.get("session"):
            continue
        lead = " ".join(page.lead)
        status = m[1] if (m := _OUTCOME_LEAD.search(lead)) else ""
        story_block = page.block("b-story")
        lines = [
            Draft(page="", pointer=pointer.removeprefix(" [source: ").split(";", 1)[0], text=strip_source(b)[2:])
            for b in (story_block.bullets if story_block else [])
            if (pointer := source_of(b))
        ]
        if lines:
            found.append((page.meta.get("date", ""), Past(page.meta["session"], page.title, status, lines)))
    found.sort(key=lambda f: f[0])
    past = [p for _, p in found][-(MAX_PAST - 1) :]
    if story is not None and story.lines:
        past.append(Past(ex.session_id, story.title or "this session", outcome, story.lines))
    while sum(len(line.text) for p in past for line in p.lines) > MAX_PAST_CHARS and len(past) > 2:
        past.pop(0)
    return past


def _skills(files: dict[str, str]) -> dict[str, Page]:
    found: dict[str, Page] = {}
    for name in files:
        if name.startswith(SKILL_FOLDER) and is_page_file(name):
            page = _parsed(files, name[:-3])
            if page is not None and page.kind == "skill":
                found[page.path] = page
    return found


async def _same_skill(decision: _Decision, learned: list[Learned], held: list[Skill], skills: dict[str, Page]) -> None:
    # A lesson the writer files under a new skill may belong to a skill memory already holds under another name.
    # Two lessons of one session may also name one new skill twice; a lesson is offered only earlier new names.
    fresh = [i for i, lesson in enumerate(learned) if skill_path(lesson.skill) not in skills]
    options: dict[int, list[tuple[str, str]]] = {}
    for n, i in enumerate(fresh):
        earlier = list(dict.fromkeys(learned[j].skill for j in fresh[:n] if learned[j].skill != learned[i].skill))
        options[i] = [(s.name, f"{s.name}: {s.when}") for s in held] + [(name, name) for name in earlier]
    asked = [i for i in fresh if options[i]]
    if not asked:
        return
    questions: dict[str, Question] = {}
    for i in asked:
        criteria = {str(k): text for k, (_, text) in enumerate(options[i], 1)}
        criteria["new"] = "a kind of work none of these skills covers"
        questions[f"w{i}"] = ChoiceQuestion(instructions=SAME_SKILL.format(name=f"l{i}"), criteria=criteria)
    state = "Lessons filed under new skills:\n" + "\n".join(
        f"l{i}. [new skill: {learned[i].skill}] {learned[i].text}" for i in asked
    )
    answers = await decision.ask("steward_skill_same", state, questions)
    renamed: dict[str, str] = {}
    for i in asked:
        answer = answers.get(f"w{i}")
        choice = answer.choice if isinstance(answer, ChoiceAnswer) else "new"
        if choice.isdigit() and 0 < int(choice) <= len(options[i]):
            target = options[i][int(choice) - 1][0]
            renamed[learned[i].skill] = renamed.get(target, target)
    for lesson in learned:
        lesson.skill = renamed.get(lesson.skill, lesson.skill)


async def _lessons(
    decision: _Decision, writer: Writer | ReplayWriter, files: dict[str, str], ex: Extract, past: list[Past]
) -> tuple[list[Learned], list[Gap]]:
    skills = _skills(files)
    held_skills = [
        Skill(
            page.meta.get("name", page.title),
            page.description.split(": ", 1)[-1],
            [strip_source(line)[2:] for line in _lines(page)][:MAX_SKILL_LINES],
        )
        for page in skills.values()
    ]
    try:
        learned = await learn(writer, past, held_skills, f"{ex.session_id}.lessons")
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        raise Deferred(f"the writer {writer.name} failed on lessons: {e}") from e
    if not learned:
        return [], []
    await _same_skill(decision, learned, held_skills, skills)
    text_of = {line.pointer: line.text for p in past for line in p.lines}
    listed = "\n".join(
        f"l{i}. [{lesson.kind}, skill: {lesson.skill}"
        + (f", for {lesson.when}" if lesson.when else "")
        + f"] {lesson.text}\n"
        + "\n".join(f"    cited: {text_of.get(ptr, '')}" for ptr in lesson.pointers)
        for i, lesson in enumerate(learned)
    )
    questions: dict[str, Question] = {
        f"l{i}": NoulQuestion(instructions=LEARNED.format(name=f"l{i}")) for i in range(len(learned))
    }
    questions |= {f"k{i}": NoulQuestion(instructions=SKILL_FITS.format(name=f"l{i}")) for i in range(len(learned))}
    # A lesson worded anew may say what its skill or the general lessons already hold; System 1 compares it with the
    # closest lines.
    general = _lines(_parsed(files, GENERAL_LESSONS), ("b-how", "b-fails", "b-rules"))
    known_of = {i: [*_lines(skills.get(skill_path(lesson.skill))), *general] for i, lesson in enumerate(learned)}
    ranked = {
        i: sorted(known_of[i], key=lambda line: -_overlap(lesson.text, strip_source(line)))[:MAX_COMPARED]
        for i, lesson in enumerate(learned)
    }
    held = ""
    for i, lines in ranked.items():
        if not lines:
            continue
        criteria = {"new": "it adds a way of working memory does not hold yet"}
        criteria |= {f"same{k}": f"it says what held lesson {k} says" for k in range(1, len(lines) + 1)}
        questions[f"s{i}"] = ChoiceQuestion(instructions=SAME_LESSON.format(name=f"l{i}"), criteria=criteria)
        held += f"\nHeld lessons for l{i}:\n" + "\n".join(
            f"    {k}. {strip_source(line)[2:]}" for k, line in enumerate(lines, 1)
        )
    # Lessons draw on older sessions' stories, which may still give a value memory has since replaced.
    replaced = [
        strip_source(line)[2:]
        for name in files
        if is_page_file(name) and name.startswith(STATED_FOLDERS)
        for line in _lines(_parsed(files, name[:-3]))
        if "(until " in line
    ]
    for i, lesson in enumerate(learned):
        stated = {n for n in numbers(lesson.text) if len(n) >= MIN_STATED_DIGITS or "." in n}
        if found := [line for line in replaced if stated & numbers(line)][:MAX_COMPARED]:
            questions[f"r{i}"] = NoulQuestion(instructions=STALE_LESSON.format(name=f"l{i}"))
            held += f"\nReplaced lines for l{i}:\n" + "\n".join(f"    {line}" for line in found)
    state = f"Lessons drawn across sessions:\n{listed}{held}"
    answers = await decision.ask("steward_lessons", state, questions)
    kept: list[Learned] = []
    gaps: list[Gap] = []
    for i, lesson in enumerate(learned):
        p = _noul(answers.get(f"l{i}"))
        if p < KEEP:
            gaps.append(Gap(f"lesson not kept, p={p:.2f}: {inert(lesson.text)}", lesson.pointers[-1]))
            continue
        fits = _noul(answers.get(f"k{i}"))
        if fits < KEEP:
            gaps.append(Gap(f"lesson does not fit skill {inert(lesson.skill)}, p={fits:.2f}", lesson.pointers[-1]))
            continue
        if f"r{i}" in questions and (holds := _noul(answers.get(f"r{i}"))) < NOTE_KEEP:
            gaps.append(Gap(f"lesson gives a replaced value, p={holds:.2f}: {inert(lesson.text)}", lesson.pointers[-1]))
            continue
        answer = answers.get(f"s{i}")
        choice = answer.choice if isinstance(answer, ChoiceAnswer) else "new"
        found = re.fullmatch(r"same(\d+)", choice)
        if found and 0 < int(found[1]) <= len(ranked[i]):
            lesson.old = ranked[i][int(found[1]) - 1]
        else:
            lesson.old = next((line for line in known_of[i] if similar(strip_source(line), lesson.text)), "")
        if any(k.old and k.old == lesson.old for k in kept):
            continue
        if lesson.old in general:
            lesson.old_page = GENERAL_LESSONS
        kept.append(lesson)
    return kept, gaps


async def _condense(
    decision: _Decision, writer: Writer | ReplayWriter, files: dict[str, str], paths: set[str], ex: Extract
) -> tuple[dict[str, str], list[Gap]]:
    # A full page is merged before any line moves to history; what still does not fit spills after.
    out: dict[str, str] = {}
    gaps: list[Gap] = []
    for path in sorted(paths):
        page = _parsed(files, path)
        if page is None:
            continue
        sections: dict[str, list[tuple[str, str]]] = {}
        where: dict[str, tuple[Block, str, str]] = {}
        for block in page.blocks:
            # A list of files, programs or calls has one line per thing; nothing in it says the same twice.
            if block.id == RELATIONS_ID or block.id.startswith(INVENTORY_BLOCKS):
                continue
            for line in block.bullets:
                if not source_of(line):
                    continue
                body = strip_source(line)[2:]
                label = m[1] if (m := _LABEL_TAIL.search(body)) else ""
                ident = str(len(where) + 1)
                where[ident] = (block, line, label)
                sections.setdefault(f"{block.heading} ({block.id})", []).append(
                    (ident, body.removesuffix(f" {label}") if label else body)
                )
        mergeable = sum(len(text) for lines in sections.values() for _, text in lines)
        if len(where) < 2 or mergeable < MIN_FREED:
            continue
        need = min(max(MIN_FREED, len(page.render()) // FREED_SHARE), mergeable // FREED_MAX_SHARE)
        try:
            merges, dropped = await condense(writer, sections, need, f"{ex.session_id}.condense.{path}")
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            # Merging only saves room: without it the page spills to history as before.
            gaps.append(Gap(f"[[{path}]] not merged: the writer failed: {inert(str(e))[:200]}", ""))
            continue
        # Lines merge only when they share a label: who said a fact stays part of it.
        merges = [m for m in merges if len({where[i][2] for i in m.ids}) == 1]
        if not merges and not dropped:
            continue
        text_of = {ident: text for lines in sections.values() for ident, text in lines}
        listed = "\n\n".join(
            f"{name}:\n" + "\n".join(f"{i}. {t}" for i, t in lines) for name, lines in sections.items()
        )
        shown = "\n".join(
            f"m{k}. {m.text}\n" + "\n".join(f"    replaces {i}. {text_of[i]}" for i in m.ids)
            for k, m in enumerate(merges)
        )
        shown += "".join(f"\nd{k}. drop {i}. {text_of[i]}" for k, i in enumerate(dropped))
        questions: dict[str, Question] = {
            f"m{k}": NoulQuestion(instructions=MERGED.format(name=f"m{k}")) for k in range(len(merges))
        }
        questions |= {f"d{k}": NoulQuestion(instructions=DROPPED.format(name=f"d{k}")) for k in range(len(dropped))}
        state = f"Page {path}: {page.description}\n\n{listed}\n\nProposed:\n{shown}"
        answers = await decision.ask("steward_condense", state, questions)
        replace: dict[tuple[str, str], str | None] = {}
        for k, merge in enumerate(merges):
            if (p := _noul(answers.get(f"m{k}"))) < KEEP:
                gaps.append(Gap(f"merge on [[{path}]] not kept, p={p:.2f}: {inert(merge.text)}", ""))
                continue
            olds = [where[i] for i in merge.ids]
            added = min((m[1] for _, line, _ in olds if (m := _ADDED.search(line))), default=ex.date)
            newest = max(olds, key=lambda o: m[1] if (m := _ADDED.search(o[1])) else "")
            pointer = m[1] if (m := _POINTER.search(newest[1])) else f"evt#{ex.session_id}:merge"
            line = f"- {inert(merge.text)} {olds[0][2]} [source: {pointer}; added: {added}]"
            replace[(olds[0][0].id, olds[0][1])] = line
            replace |= {(block.id, old): None for block, old, _ in olds[1:]}
        for k, ident in enumerate(dropped):
            if (p := _noul(answers.get(f"d{k}"))) < KEEP:
                continue
            block, old, _ = where[ident]
            replace.setdefault((block.id, old), None)
        if not replace:
            continue
        for block in page.blocks:
            block.lines = [new for line in block.lines if (new := replace.get((block.id, line), line)) is not None]
        page.blocks = [b for b in page.blocks if b.lines or b.id == RELATIONS_ID]
        page.refresh_contents()
        out[page_file(path)] = page.render()
    return out, gaps


def _transcript(ex: Extract, outcome: str) -> str:
    # The whole session as the steward saw it, for the agent's memory search when no page holds what it needs.
    task = " ".join(ex.task.split())
    head = [f"session: {ex.session_id}", f"date: {ex.date}", f"outcome: {outcome}", f"task: {task}"]
    return "\n".join(head) + "\n\n" + trace(ex, outcome) + "\n"


def _usage(root: Path, files: dict[str, str], ex: Extract, written: list[str]) -> tuple[str, dict[str, str], list[str]]:
    # Each session counts which pages it used; a page unused and unwritten for long moves to graveyard/.
    known = root.expanduser().resolve() / USAGE_FILE
    usage = read_usage(known.read_text() if known.is_file() else None)
    live = [name[:-3] for name in files if is_page_file(name)]
    usage = tally(usage, live, ex.used, written)
    gone = [p for p in stale(usage) if p not in written]
    if not gone:
        return usage.model_dump_json(indent=1), {}, []
    archived, removed = archive(files, gone)
    for path in gone:
        usage.pages.pop(path, None)
    return usage.model_dump_json(indent=1), archived, removed


async def _work(decision: _Decision, ex: Extract, story: Story | None, kinds: list[str]) -> str:
    proposed = [story.work] if story is not None and story.work else []
    options: list[str] = list(dict.fromkeys([*kinds, *proposed]))
    if not options:
        return OTHER_WORK
    criteria = {str(i): kind for i, kind in enumerate(options, 1)} | {"none": "none of these kinds of work"}
    state = _state(ex) + (f"\nSession title: {story.title}" if story is not None and story.title else "")
    question = ChoiceQuestion(instructions=WORK, criteria=criteria)
    answers = await decision.ask("steward_work", state, {"work": question})
    choice = answer.choice if isinstance(answer := answers.get("work"), ChoiceAnswer) else "none"
    return options[int(choice) - 1] if choice.isdigit() and 0 < int(choice) <= len(options) else OTHER_WORK


def _candidates(page: Page, block: Block) -> list[str]:
    items = [strip_source(b)[2:] for b in block.bullets]
    keys = [item.split("`")[1] if item.count("`") >= 2 else item for item in items]
    if block.id == RELATIONS_ID:
        targets = [r.target for r in page.relations()]
        found = [
            "related lessons, maps and events",
            f"links to {', '.join(targets[:3])}" + (" and more" if len(targets) > 3 else ""),
            f"{len(targets)} {'link' if len(targets) == 1 else 'links'} to other pages",
        ]
    else:
        noun = BLOCK_NOUNS.get("-".join(block.id.split("-")[:2]), block.heading.lower())
        if topic := topic_of(block):
            noun = f"{noun}: {topic}"
        found = [noun, f"{noun} ({len(items)})"]
        if keys:
            found.append(f"{noun}, starting with {keys[0]}")
        if len(keys) > 1:
            found.append(f"{noun}: {keys[0]} to {keys[-1]}")
    unique: list[str] = []
    for text in found:
        cleaned = " ".join(text.replace("·", "-").split())[:DESCRIPTION_CHARS]
        if cleaned and cleaned not in unique:
            unique.append(cleaned)
    while len(unique) < 2:
        unique.append(f"block {block.id} of {page.title}")
    return unique


async def _contents(decision: _Decision, page: Page) -> dict[str, str]:
    described = {line.block_id for line in page.contents}
    options = {block.id: _candidates(page, block) for block in page.blocks if block.id not in described}
    if not options:
        return {}
    questions: dict[str, Question] = {
        block_id: ChoiceQuestion(instructions=CONTENTS, criteria={str(i): c for i, c in enumerate(cands, 1)})
        for block_id, cands in options.items()
    }
    state = f"Page {page.path}: {page.description}\n\n" + "\n\n".join(b.render() for b in page.blocks)
    answers = await decision.ask("steward_contents", state, questions)
    picked: dict[str, str] = {}
    for block_id, cands in options.items():
        answer = answers.get(block_id)
        index = int(answer.choice) - 1 if isinstance(answer, ChoiceAnswer) and answer.choice.isdigit() else 0
        picked[block_id] = cands[min(max(index, 0), len(cands) - 1)]
    return picked


_USER_TURN = re.compile(r"^u\d+ \[user", re.MULTILINE)


def _continued(files: dict[str, str], ex: Extract) -> bool:
    """Whether the journal has user turns its work page did not count: the session was resumed after its steward."""
    if (counted := stewarded_turns(files, ex.session_id)) is not None:
        return user_turns(ex) > counted
    # A work page from before the count: both transcripts are cut, so a long session may look unchanged.
    old = files.get(f"{SESSIONS_FOLDER}{ex.session_id}.txt", "")
    return len(_USER_TURN.findall(trace(ex, ""))) > len(_USER_TURN.findall(old))


def _prepare(
    journal_path: Path, drive_root: Path, world: str = ""
) -> tuple[dict[str, str], Extract, tuple[dict[str, str], list[str]]]:
    if not journal_path.is_file():
        raise StewardError(f"{journal_path} is not a journal file")
    root = drive_root.expanduser().resolve()
    files = tracked_files(root, main_commit(root))
    # A drive holds one world: an agent reading it must never see another world's lines.
    held = drive_world(files)
    if held is None and any(name.startswith(WORK_FOLDER) and "\nsession: " in text for name, text in files.items()):
        held = ""
    if held is not None and held != world:
        raise StewardError(
            f"the drive holds {'world ' + held if held else 'an unlabelled world'}; a session of "
            f"{'world ' + world if world else 'an unlabelled world'} needs its own drive"
        )
    ex = extract(Journal(cwd="", path=journal_path), root, world)
    forgotten: tuple[dict[str, str], list[str]] = ({}, [])
    if stewarded(files, ex.session_id):
        if not _continued(files, ex):
            raise Skipped(f"session {ex.session_id} is already in the drive")
        files, changed, removed = forget(files, ex.session_id)
        forgotten = (changed, removed)
    if branch_exists(root, f"proposed/{ex.session_id}"):
        raise Skipped(f"proposed/{ex.session_id} already exists")
    return files, ex, forgotten


async def propose(
    journal_path: Path,
    drive_root: Path,
    decisions: DecisionClient,
    writer: Writer | ReplayWriter | None = None,
    world: str = "",
) -> Proposal:
    files, ex, (forgot, gone) = _prepare(journal_path, drive_root, world)

    decision = _Decision(decisions, ex.run_state)
    await _step_changes(decision, ex)
    outcome, kept, lessons, gaps = await _triage(decision, ex)
    items: list[Placed] = []
    story: Story | None = None
    kinds = parse_schema(files[SCHEMA_FILE]).work_kinds if SCHEMA_FILE in files else []
    learned: list[Learned] = []
    if writer is not None:
        story, items, item_gaps = await _items(decision, writer, ex, outcome, files, kinds)
        gaps += item_gaps
        learned, lesson_gaps = await _lessons(decision, writer, files, ex, _past(files, ex, outcome, story))
        gaps += lesson_gaps
    work = await _work(decision, ex, story, kinds)
    written = write(files, ex, outcome, kept, lessons, items, story, work, learned, spill=False)
    condensed: dict[str, str] = {}
    if written.full and writer is not None:
        condensed, merge_gaps = await _condense(decision, writer, files, written.full, ex)
        gaps += merge_gaps
    if written.full:
        written = write(files | condensed, ex, outcome, kept, lessons, items, story, work, learned)
    for page in written.pages.values():
        page.refresh_contents(await _contents(decision, page))

    changed = condensed | {page_file(path): page.render() for path, page in written.pages.items()}
    if SCHEMA_FILE in files:
        schema = with_vocabulary(with_work_kind(files[SCHEMA_FILE], work))
        if schema != files[SCHEMA_FILE]:
            changed[SCHEMA_FILE] = schema
    if ex.world and drive_world(files) is None:
        changed[WORLD_FILE] = world_file(ex.world)
    merged = files | changed
    touched = [name[:-3] for name in changed if is_page_file(name)]
    usage, archived, removed = _usage(drive_root, merged, ex, touched)
    changed |= archived
    merged = {name: text for name, text in (merged | archived).items() if name not in removed}
    built = build_indexes(merged)
    # A folder whose pages all moved to graveyard/ loses its table of contents too.
    removed += [
        name
        for name in merged
        if name.endswith(f"/{FOLDER_INDEX}") and not name.startswith(UNREAD_FOLDERS) and name not in built
    ]
    merged = {name: text for name, text in merged.items() if name not in removed}
    changed = {name: text for name, text in changed.items() if name not in removed}
    changed |= {name: text for name, text in built.items() if files.get(name) != text}
    merged |= changed
    if problems := lint(merged):
        raise StewardError("the proposal does not pass lint:\n" + "\n".join(problems))
    changed[USAGE_FILE] = usage
    changed[f"{SESSIONS_FOLDER}{ex.session_id}.txt"] = _transcript(ex, outcome)
    changed = {name: text for name, text in forgot.items() if name not in removed} | changed
    return Proposal(
        session_id=ex.session_id,
        branch=f"proposed/{ex.session_id}",
        files=changed,
        gaps=gaps + written.gaps,
        removed=[name for name in removed if name in files] + [name for name in gone if name not in changed],
    )


def commit(drive_root: Path, proposal: Proposal) -> Proposal:
    trailers = {"Intern-Session": proposal.session_id, "Intern-Steward-Version": STEWARD_VERSION}
    message = f"Steward: session {proposal.session_id}"
    proposal.commit = commit_proposal(
        drive_root.expanduser().resolve(), proposal.branch, proposal.files, message, trailers, proposal.removed
    )
    return proposal
