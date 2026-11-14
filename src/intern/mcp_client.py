"""Lazy optional MCP connections adapted to the existing typed Tool seam."""

from __future__ import annotations

import asyncio
import importlib
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, RootModel, SecretStr

from intern.async_tasks import join_tasks
from intern.mcp_credentials import CommandCredential as CommandCredential
from intern.mcp_credentials import CredentialCommand as CredentialCommand
from intern.messages import JsonObject
from intern.needs import Needs
from intern.providers.base import ToolSpec
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail


class StdioConfig(BaseModel):
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, SecretStr] = Field(default_factory=dict)
    cwd: Path | None = None


class _Description(BaseModel):
    name: str
    description: str = "External MCP tool"
    inputSchema: JsonObject


class _ToolPage(BaseModel):
    tools: list[_Description]
    nextCursor: str | None = None


class _Content(BaseModel):
    type: str
    text: str | None = None


class _CallResult(BaseModel):
    content: list[_Content] = Field(default_factory=list)
    isError: bool = False


class _Arguments(RootModel[JsonObject]):
    pass


MCP_NEEDS = Needs(title="MCP", extra="mcp", modules=("mcp", "jsonschema"))
"""The MCP SDK and the schema checker, for every configured MCP connection."""


def _dependencies():
    try:
        return importlib.import_module("mcp"), importlib.import_module("jsonschema")
    except ImportError:
        raise RuntimeError("MCP support requires the optional extra: uv sync --extra mcp") from None


def _check_references(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"$ref", "$dynamicRef"} and (not isinstance(child, str) or not child.startswith("#")):
                raise ValueError("External schema references are not supported; provide a self-contained input schema")
            _check_references(child)
    elif isinstance(value, list):
        for child in value:
            _check_references(child)


class McpConnection:
    """A live SDK session; close it in the same task that opened its context."""

    def __init__(self, session: Any, namespace: str, *, timeout_s: float = 30) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,23}", namespace) or not 0 < timeout_s <= 600:
            raise ValueError("Use a short lowercase namespace and a timeout in (0, 600]")
        self.session, self.namespace, self.timeout_s = session, namespace, timeout_s
        self.closed = False

    async def tools(self, effects: Mapping[str, frozenset[Effect]]) -> list[Tool]:
        """Select explicitly authorized remote names; annotations never grant scope."""
        if self.closed:
            raise RuntimeError("MCP connection is closed")
        _, schema = _dependencies()
        descriptions: dict[str, _Description] = {}
        cursor = None
        for _ in range(16):
            async with asyncio.timeout(self.timeout_s):
                raw = await self.session.list_tools(cursor=cursor)
            page = _ToolPage.model_validate(raw.model_dump(mode="json"))
            for description in page.tools:
                if description.name in descriptions:
                    raise ValueError("Duplicate remote tool name")
                descriptions[description.name] = description
            if len(descriptions) > 512:
                raise ValueError("MCP tool catalog exceeds the configured client capacity")
            cursor = page.nextCursor
            if cursor is None:
                break
        else:
            raise ValueError("MCP tool pagination exceeded 16 pages")
        if not effects.keys() <= descriptions.keys():
            raise ValueError("An explicitly selected tool is absent from the server catalog")
        names: set[str] = set()
        tools: list[Tool] = []
        for remote_name, declared in effects.items():
            descriptor = descriptions[remote_name]
            _check_references(descriptor.inputSchema)
            schema.Draft202012Validator.check_schema(descriptor.inputSchema)
            validator = schema.Draft202012Validator(descriptor.inputSchema)
            name = f"mcp__{self.namespace}__{re.sub(r'[^a-zA-Z0-9_]', '_', remote_name)}"
            if len(name) > 64 or name in names:
                raise ValueError("Remote tool names collide or exceed the model's 64-character tool-name limit")
            names.add(name)
            tools.append(self._tool(descriptor, name, declared, validator))
        return tools

    def _tool(self, descriptor: _Description, local_name: str, declared: frozenset[Effect], validator: Any) -> Tool:
        connection = self

        class McpTool(Tool):
            name: ClassVar[str] = local_name
            description: ClassVar[str] = descriptor.description
            Args: ClassVar[type[BaseModel]] = _Arguments
            effects: ClassVar[frozenset[Effect]] = declared

            def spec(self) -> ToolSpec:
                return ToolSpec(name=self.name, description=self.description, parameters=descriptor.inputSchema)

            async def run(self, args: Any, ctx: ToolContext) -> ToolOutput:
                if connection.closed:
                    return fail("The MCP connection is closed.", "Reconnect before using this tool.")
                if ctx.abort.is_set():
                    return fail("The run was aborted.", "Stop.").model_copy(update={"status": "refused"})
                try:
                    arguments = _Arguments.model_validate(args).root
                    validator.validate(arguments)
                except Exception:
                    return fail("Arguments do not match the remote tool schema.", "Use the advertised input schema.")
                call = asyncio.create_task(
                    connection.session.call_tool(
                        descriptor.name,
                        arguments=arguments,
                        read_timeout_seconds=timedelta(seconds=connection.timeout_s),
                    )
                )
                aborted = asyncio.create_task(ctx.abort.wait())
                try:
                    done, _ = await asyncio.wait(
                        {call, aborted}, timeout=connection.timeout_s, return_when=asyncio.FIRST_COMPLETED
                    )
                    if call not in done:
                        return fail(
                            "MCP call was interrupted or timed out.",
                            "Check the remote action before retrying.",
                            retry_safe=False,
                        )
                    result = _CallResult.model_validate(call.result().model_dump(mode="json"))
                    parts = [
                        item.text
                        if item.type == "text" and item.text is not None
                        else f"[MCP {item.type} content omitted; text adapter only]"
                        for item in result.content
                    ]
                    content = "\n".join(parts)
                    cap = max(ctx.max_output_chars * 8, 16_384)
                    if len(content) > cap:
                        content = content[:cap] + "\n[MCP result truncated by adapter; request a narrower result.]"
                    if result.isError:
                        return fail(
                            "The remote tool reported an error.",
                            "Check its effects before retrying.",
                            content=content,
                            retry_safe=False,
                        )
                    return ToolOutput(content=content)
                except Exception:
                    return fail(
                        "MCP transport or server call failed.",
                        "Check the remote outcome before retrying.",
                        retry_safe=False,
                    )
                finally:
                    await join_tasks(call, aborted, cancel=True)

        return McpTool()


@asynccontextmanager
async def stdio_connection(
    config: StdioConfig, namespace: str, *, timeout_s: float = 30
) -> AsyncIterator[McpConnection]:
    sdk, _ = _dependencies()
    stdio = importlib.import_module("mcp.client.stdio")
    parameters = sdk.StdioServerParameters(
        command=config.command,
        args=config.args,
        cwd=config.cwd,
        env={key: value.get_secret_value() for key, value in config.env.items()},
    )
    async with (
        stdio.stdio_client(parameters) as (read, write),
        sdk.ClientSession(read, write, read_timeout_seconds=timedelta(seconds=timeout_s)) as session,
    ):
        connection = McpConnection(session, namespace, timeout_s=timeout_s)
        try:
            await session.initialize()
            yield connection
        finally:
            connection.closed = True


@asynccontextmanager
async def http_connection(
    url: str, namespace: str, *, timeout_s: float = 30, http_client: Any = None
) -> AsyncIterator[McpConnection]:
    """Connect to an explicit endpoint; caller may supply an authenticated httpx client."""
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ValueError("Provide an http(s) MCP URL without embedded credentials or fragment")
    sdk, _ = _dependencies()
    transport = importlib.import_module("mcp.client.streamable_http")
    async with (
        transport.streamable_http_client(url, http_client=http_client) as (read, write, _),
        sdk.ClientSession(read, write, read_timeout_seconds=timedelta(seconds=timeout_s)) as session,
    ):
        connection = McpConnection(session, namespace, timeout_s=timeout_s)
        try:
            await session.initialize()
            yield connection
        finally:
            connection.closed = True
