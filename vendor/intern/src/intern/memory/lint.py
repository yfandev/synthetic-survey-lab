from __future__ import annotations

from pathlib import Path

from intern.memory.format import (
    BLOCK_TOKENS,
    FOLDER_INDEX,
    MAX_TOC_ENTRIES,
    PAGE_TOKENS,
    RELATIONS_ID,
    ROOT_INDEX,
    SCHEMA_FILE,
    UNREAD_FOLDERS,
    FormatError,
    Page,
    Schema,
    build_indexes,
    has_source,
    is_page_file,
    links,
    page_file,
    parse_index,
    parse_page,
    parse_schema,
    tokens,
)


def working_files(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): p.read_text()
        for p in sorted(root.rglob("*.md"))
        if ".git" not in p.relative_to(root).parts
    }


def _page_problems(page: Page, text: str, schema: Schema, known: set[str]) -> list[str]:
    name = page_file(page.path)
    problems: list[str] = []
    if page.meta.get("id") != page.path:
        problems.append(f"{name}: front matter id is {page.meta.get('id')!r}, expected {page.path!r}")
    if page.kind not in schema.kinds:
        problems.append(f"{name}: kind {page.kind!r} is not in the schema ({', '.join(schema.kinds)})")
    parts = page.path.split("/")
    if parts[0] == "work" and len(parts) > 2 and parts[1] not in schema.work_kinds:
        problems.append(f"{name}: kind of work {parts[1]!r} is not in the schema's work kinds")
    listed = [(line.block_id, line.tokens) for line in page.contents]
    actual = [(b.id, tokens(b.render())) for b in page.blocks]
    if listed != actual:
        problems.append(f"{name}: Contents is out of date (expected {actual}, found {listed})")
    if len(page.contents) > MAX_TOC_ENTRIES:
        problems.append(f"{name}: {len(page.contents)} Contents lines, more than {MAX_TOC_ENTRIES}")
    for block in page.blocks:
        size = tokens(block.render())
        if size > BLOCK_TOKENS:
            problems.append(f"{name}: block {block.id} is {size} tokens, more than {BLOCK_TOKENS}")
        if block.id == RELATIONS_ID:
            continue
        problems += [
            f"{name}: bullet without a source in {block.id}: {b[:80]!r}" for b in block.bullets if not has_source(b)
        ]
    relations_block = page.block(RELATIONS_ID)
    typed = page.relations()
    if relations_block and len(typed) != len(relations_block.bullets):
        problems.append(f"{name}: a Relations line is not `- <relation> [[path]]`")
    problems += [
        f"{name}: relation {r.relation!r} is not in the schema" for r in typed if r.relation not in schema.relations
    ]
    problems += [f"{name}: link [[{target}]] does not resolve" for target in links(text) if target not in known]
    size = tokens(text)
    if size > PAGE_TOKENS:
        problems.append(f"{name}: {size} tokens, more than {PAGE_TOKENS}")
    return problems


def lint(files: dict[str, str]) -> list[str]:
    problems: list[str] = []
    schema = Schema(kinds=[], relations=[])
    if SCHEMA_FILE not in files:
        problems.append(f"{SCHEMA_FILE} is missing")
    else:
        try:
            schema = parse_schema(files[SCHEMA_FILE])
        except FormatError as e:
            problems.append(str(e))
    if ROOT_INDEX not in files:
        problems.append(f"{ROOT_INDEX} is missing")

    known = {name[:-3] for name in files if is_page_file(name)}
    for name, text in sorted(files.items()):
        if not is_page_file(name):
            continue
        try:
            page = parse_page(name[:-3], text)
        except FormatError as e:
            problems.append(str(e))
            continue
        problems += _page_problems(page, text, schema, known)

    expected = build_indexes(files)
    for name, text in sorted(expected.items()):
        if files.get(name) != text:
            problems.append(f"{name}: table of contents is out of date; run `intern memory lint --fix`")
        entries = parse_index(name, text).entries
        if len(entries) > MAX_TOC_ENTRIES:
            problems.append(f"{name}: {len(entries)} entries, more than {MAX_TOC_ENTRIES}")
    for name in sorted(files):
        stale = name.rsplit("/", 1)[-1] == FOLDER_INDEX and not name.startswith(UNREAD_FOLDERS)
        if stale and name not in expected:
            problems.append(f"{name}: table of contents for a folder with no pages")
    return problems


def fix(files: dict[str, str]) -> dict[str, str]:
    changed: dict[str, str] = {}
    current = dict(files)
    for name, text in files.items():
        if not is_page_file(name):
            continue
        try:
            page = parse_page(name[:-3], text)
        except FormatError:
            continue
        page.refresh_contents()
        if page.render() != text:
            changed[name] = current[name] = page.render()
    for name, text in build_indexes(current).items():
        if current.get(name) != text:
            changed[name] = text
    return changed
