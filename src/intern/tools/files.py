"""File tools: read, write, edit."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from intern.messages import ResourceRef
from intern.tools import images
from intern.tools.base import (
    MAX_DIFF_CHARS,
    Effect,
    Tool,
    ToolContext,
    ToolOutput,
    fail,
    file_diff,
    file_resource,
)

BINARY_SNIFF_BYTES = 8192
MAX_DIR_ENTRIES = 500
# Room kept for the paging footer, so a full page is never cut by the loop's output bound.
FOOTER_RESERVE_CHARS = 120


class ReadArgs(BaseModel):
    path: str = Field(description="File or directory, relative to the working directory or absolute.")
    offset: int = Field(default=1, ge=1, description="First line to show, 1-based.")
    limit: int | None = Field(default=None, ge=1, description="Number of lines to show; omit for as many as fit.")


class ReadTool(Tool):
    """Set ``attach_images=False`` when the host owns image admission and viewing."""

    name = "read"
    description = (
        "Read a text file, or list a directory. Long files are shown in pages; "
        "the footer says which offset to read next."
    )
    Args = ReadArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read"})
    # Existing subclasses may have constructors that do not call super().__init__.
    attach_images: bool = True

    def __init__(self, *, attach_images: bool = True) -> None:
        self.attach_images = attach_images

    async def run(self, args: ReadArgs, ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args.path)
        if not path.exists():
            return fail(
                f"{args.path} does not exist",
                "Check the name with find, or read the parent directory to list it.",
                argument="path",
            )
        if path.is_dir():
            entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            names = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:MAX_DIR_ENTRIES]]
            more = len(entries) - len(names)
            listing = "\n".join(names) or "(empty directory)"
            return ToolOutput(content=listing + (f"\n[{more} more entries not shown]" if more > 0 else ""))

        if path.suffix.lower() in images.IMAGE_SUFFIXES:
            if not self.attach_images:
                return ToolOutput(
                    content=f"{args.path} is an image file. Automatic image attachment in read is disabled; "
                    "use the host's image-viewing tool to view it."
                )
            if ctx.image_dir is not None:
                return await self._image(path, args, ctx.image_dir)
        data = path.read_bytes()
        if b"\x00" in data[:BINARY_SNIFF_BYTES]:
            return fail(
                f"{args.path} is a binary file ({len(data)} bytes)",
                "Inspect it with bash using a tool that understands the format (e.g. file, xxd | head).",
                argument="path",
            )
        lines = data.decode(errors="replace").splitlines()
        return page_lines(args.path, lines, args, ctx.max_output_chars, resources=[file_resource(path, data)])

    async def _image(self, path: Path, args: ReadArgs, store: Path) -> ToolOutput:
        try:
            image = await asyncio.to_thread(images.load_image, path, store)
        except images.ImageError as e:
            return fail(str(e), "Check the file type with: file <path>.", argument="path")
        shown = f"{args.path} is an image ({image.width}x{image.height} as shown); it is attached below."
        return ToolOutput(content=shown, images=[image])


def page_lines(
    shown_as: str, lines: list[str], args: ReadArgs, max_output_chars: int, *, resources: list[ResourceRef]
) -> ToolOutput:
    """``lines`` from ``args.offset``, at most ``args.limit`` and what fits the budget, with a footer saying how to
    continue. ``shown_as`` names the text in messages."""
    if args.offset > max(len(lines), 1):
        return fail(
            f"offset {args.offset} is past the end of {shown_as} ({len(lines)} lines)",
            f"Read again with an offset between 1 and {max(len(lines), 1)}.",
            argument="offset",
        )
    start = args.offset - 1
    stop = len(lines) if args.limit is None else min(len(lines), start + args.limit)
    shown: list[str] = []
    size = 0
    budget = max(max_output_chars - FOOTER_RESERVE_CHARS, FOOTER_RESERVE_CHARS)
    for line in lines[start:stop]:
        if size + len(line) + 1 > budget and shown:
            break
        shown.append(line)
        size += len(line) + 1
    end = start + len(shown)
    body = "\n".join(shown)
    if end < len(lines):
        body += f"\n[Lines {start + 1}-{end} of {len(lines)}. Continue with offset={end + 1}.]"
    return ToolOutput(content=body or "(empty file)", resources=resources)


class WriteArgs(BaseModel):
    path: str = Field(description="File to create or overwrite.")
    content: str = Field(description="The complete new contents of the file.")


class WriteTool(Tool):
    name = "write"
    description = "Create a file, or replace a file's whole contents. To change part of a file, use edit."
    Args = WriteArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"write"})

    async def run(self, args: WriteArgs, ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args.path)
        if path.is_dir():
            return fail(f"{args.path} is a directory", "Write to a file path inside it instead.", argument="path")
        existed = path.exists()
        # The old text, for the diff, unless it is too big to carry anyway.
        before = path.read_bytes() if existed and path.stat().st_size <= MAX_DIFF_CHARS else None
        path.parent.mkdir(parents=True, exist_ok=True)
        data = args.content.encode()
        path.write_bytes(data)
        verb = "Overwrote" if existed else "Created"
        lines = args.content.count("\n") + (0 if args.content.endswith("\n") or not args.content else 1)
        return ToolOutput(
            content=f"{verb} {args.path} ({lines} lines).",
            resources=[file_resource(path, data)],
            diff=file_diff(path, before, data) if before is not None or not existed else None,
        )


class EditArgs(BaseModel):
    path: str = Field(description="File to change.")
    old_string: str = Field(description="Exact text to replace, copied from the file with its whitespace.")
    new_string: str = Field(description="Text to put in its place.")
    replace_all: bool = Field(default=False, description="Replace every occurrence instead of exactly one.")


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


class EditTool(Tool):
    name = "edit"
    description = (
        "Replace exact text in a file. old_string must match the file exactly and, unless replace_all "
        "is set, exactly once; include a few surrounding lines to make it unique."
    )
    Args = EditArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read", "write"})

    async def run(self, args: EditArgs, ctx: ToolContext) -> ToolOutput:
        path = ctx.resolve(args.path)
        if not path.is_file():
            return fail(
                f"{args.path} does not exist", "Create it with write, or check the path with find.", argument="path"
            )
        if not args.old_string:
            return fail(
                "old_string is empty", "Copy the text to replace into old_string, or use write.", argument="old_string"
            )
        if args.old_string == args.new_string:
            return fail(
                "old_string and new_string are identical",
                "Put the changed text in new_string.",
                argument="new_string",
            )
        before = path.read_bytes()
        text = before.decode("utf-8", errors="surrogateescape")
        old, new = args.old_string, args.new_string
        if "\r\n" in text and old not in text and "\r" not in old:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        count = text.count(old)
        if count == 0:
            hint = self._near_miss(text, old)
            return fail(
                f"old_string was not found in {args.path}" + (f"; {hint}" if hint else ""),
                "Read the file again and copy old_string exactly, including whitespace and indentation.",
                argument="old_string",
            )
        if count > 1 and not args.replace_all:
            lines: list[int] = []
            index = text.find(old)
            while index >= 0 and len(lines) < 10:
                lines.append(_line_of(text, index))
                index = text.find(old, index + 1)
            return fail(
                f"old_string occurs {count} times in {args.path} (lines {', '.join(map(str, lines))})",
                "Add surrounding lines to old_string so it matches once, or set replace_all to true.",
                argument="old_string",
            )
        first_line = _line_of(text, text.find(old))
        updated = text.replace(old, new) if args.replace_all else text.replace(old, new, 1)
        data = updated.encode("utf-8", errors="surrogateescape")
        path.write_bytes(data)
        replaced = count if args.replace_all else 1
        noun = "occurrence" if replaced == 1 else "occurrences"
        return ToolOutput(
            content=f"Replaced {replaced} {noun} in {args.path} (first at line {first_line}).",
            resources=[file_resource(path, data)],
            diff=file_diff(path, before, data),
        )

    @staticmethod
    def _near_miss(text: str, old: str) -> str | None:
        """Point at a match that differs only in indentation or trailing spaces."""
        wanted = [line.strip() for line in old.strip("\n").splitlines()]
        if not wanted or not wanted[0]:
            return None
        lines = text.splitlines()
        for i in range(len(lines) - len(wanted) + 1):
            if [line.strip() for line in lines[i : i + len(wanted)]] == wanted:
                return f"lines {i + 1}-{i + len(wanted)} match if whitespace is ignored"
        return None
