"""The optional web-read pack, with recoverable retained text rather than silent truncation."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail, file_resource
from intern.web.reader import WebReader, WebReadError
from intern.web.storage import PrivateStorageError, publish_private_file


class WebReadArgs(BaseModel):
    url: str = Field(min_length=1, max_length=8_192)


def _retain(text: str, digest: str, storage_dir: Path | None) -> Path:
    from intern.config import home_dir

    return publish_private_file(storage_dir or home_dir(os.environ) / "web-content", f"{digest}.txt", text.encode())


class WebReadTool(Tool):
    name = "web_read"
    description = "Read an HTTP(S) URL as untrusted source text. Returns source metadata and a retained full-text file."
    Args = WebReadArgs
    effects: ClassVar[frozenset[Effect]] = frozenset({"read", "write", "network"})

    def __init__(
        self,
        reader: WebReader | None = None,
        *,
        storage_dir: Path | None = None,
        display_dir: Path | None = None,
    ) -> None:
        self.reader = reader or WebReader()
        self.storage_dir = storage_dir
        # An application may map retained host files into an executor workspace.
        # Resource references retain the actual host path for the calling client.
        self.display_dir = display_dir

    async def run(self, args: WebReadArgs, ctx: ToolContext) -> ToolOutput:
        try:
            page = await self.reader.read(args.url, ctx.abort)
            path = await asyncio.to_thread(_retain, page.text, page.sha256, self.storage_dir)
            display_path = self.display_dir / path.name if self.display_dir is not None else path
            data = page.text.encode()
            preview_chars = max(1, ctx.max_output_chars // 2)
            metadata = page.model_dump(exclude={"text"}, mode="json")
            # This mapping is local serialization; typed PageDocument is the module boundary.
            content = json.dumps(metadata, ensure_ascii=False) + "\n\n" + page.text[:preview_chars]
            if len(page.text) > preview_chars:
                content += f"\n[TRUNCATED. Full UTF-8 source: {display_path}; read with offset/limit.]"
            return ToolOutput(content=content, resources=[file_resource(path, data)])
        except (WebReadError, OSError) as error:
            detail = (
                str(error)
                if isinstance(error, WebReadError | PrivateStorageError)
                else "Could not retain the extracted source."
            )
            return fail(detail, "Check the URL, use another source, or configure a supported reader.")
