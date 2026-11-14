"""Adapter for Anthropic's native Messages API, with prompt caching.

Each request marks three cache breakpoints: the system prompt (which, in Anthropic's order, also covers
the tool definitions), the end of the previous request and the end of this one. A turn then reads
everything the previous turn sent from the cache and pays full price only for what is new.

Signed thinking replays to the model that wrote it. A thinking block is bound to the conversation before
it; when the harness changes that history (compaction, shortened tool output), the request asks the API
to drop the stale blocks (`drop_block`) instead of rejecting the request.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from intern import names
from intern.messages import (
    AssistantMessage,
    ImagePart,
    InvalidToolCall,
    Message,
    StopKind,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from intern.models.prices import priced
from intern.models.profile import EFFORT_LEVELS
from intern.models.registry import Endpoint
from intern.providers.base import (
    TRANSIENT_ERRORS,
    Completed,
    ProviderError,
    ProviderRequest,
    ReasoningChunk,
    StreamItem,
    TextChunk,
    image_base64,
    missing_image_note,
    sse_lines,
)
from intern.providers.repair import parse_arguments

API_VERSION = "2023-06-01"
# Lets the request set `thinking.block_binding`.
THINKING_BINDING_BETA = "thinking-binding-controls-2026-08-01"
DROP_STALE_THINKING = {"prefix_mismatch_behavior": "drop_block"}
RETRY_WAITS_S = (2.0, 8.0, 20.0)
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
RETRYABLE_ERRORS = frozenset({"overloaded_error", "api_error", "rate_limit_error"})
CACHE = {"type": "ephemeral"}
_TOOL_ID = re.compile(r"[^A-Za-z0-9_-]")
_STOP_KINDS: dict[str, StopKind] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "error",
}

Block = dict[str, Any]
WireMessage = dict[str, Any]


def _tool_id(call_id: str) -> str:
    """Anthropic accepts only [A-Za-z0-9_-] in tool-use ids; other providers' ids may carry more."""
    return _TOOL_ID.sub("_", call_id)


def tool_result_content(content: str, images: list[ImagePart]) -> str | list[Block]:
    """A tool result's text, followed by the images it showed: the Messages API takes both inside `tool_result`."""
    text = content or "(no output)"
    if not images:
        return text
    blocks: list[Block] = [{"type": "text", "text": text}]
    for image in images:
        data = image_base64(image)
        if data is None:
            blocks.append({"type": "text", "text": missing_image_note(image)})
        else:
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": image.mime, "data": data}})
    return blocks


def wire_messages(messages: list[Message], model_ref: str | None = None) -> tuple[list[Block], list[WireMessage]]:
    """Split out the system prompt and turn the rest into alternating user and assistant turns.

    Tool results become `tool_result` blocks of a user turn, and adjacent turns of one role are merged,
    as the API requires. Empty text blocks are rejected by the API, so they are left out. Thinking blocks go
    back only to the model that wrote them (`model_ref`): their signatures are valid for it alone, and with
    them the model continues its reasoning across tool calls instead of starting over each turn.
    """
    system: list[Block] = []
    wire: list[WireMessage] = []

    def add(role: str, blocks: list[Block]) -> None:
        if not blocks:
            return
        if wire and wire[-1]["role"] == role:
            wire[-1]["content"] += blocks
        else:
            wire.append({"role": role, "content": blocks})

    for message in messages:
        match message:
            case SystemMessage(content=text):
                if text.strip():
                    system.append({"type": "text", "text": text})
            case UserMessage(content=text):
                add("user", [{"type": "text", "text": text}] if text.strip() else [])
            case AssistantMessage():
                blocks: list[Block] = []
                if model_ref is not None and message.model == model_ref:
                    blocks += [dict(b) for b in message.reasoning_blocks]
                blocks += [{"type": "text", "text": message.text}] if message.text.strip() else []
                blocks += [
                    {"type": "tool_use", "id": _tool_id(call.id), "name": call.name, "input": call.arguments}
                    for call in message.tool_calls
                ]
                add("assistant", blocks)
            case ToolResultMessage(result=result):
                add(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": _tool_id(result.call_id),
                            "content": tool_result_content(result.content, result.images),
                            "is_error": result.status != "ok",
                        }
                    ],
                )
    return system, wire


def mark_cache(system: list[Block], wire: list[WireMessage]) -> None:
    if system:
        system[-1]["cache_control"] = CACHE
    ends = {len(wire) - 1}
    # The previous request ended just before the latest assistant reply; marking it again lets this
    # request find that cache entry even when the latest turn added many blocks.
    last_reply = max((i for i, m in enumerate(wire) if m["role"] == "assistant"), default=None)
    if last_reply:
        ends.add(last_reply - 1)
    for index in ends:
        if index >= 0:
            wire[index]["content"][-1]["cache_control"] = CACHE


def _usage(raw: dict[str, Any]) -> Usage:
    uncached = raw.get("input_tokens") or 0
    read = raw.get("cache_read_input_tokens") or 0
    write = raw.get("cache_creation_input_tokens") or 0
    return Usage(
        input_tokens=uncached + read + write,
        output_tokens=raw.get("output_tokens") or 0,
        cache_read_tokens=read,
        cache_write_tokens=write,
    )


@dataclass
class _Block:
    type: str
    id: str = ""
    name: str = ""
    parts: list[str] = field(default_factory=list)
    signature: str = ""
    data: str = ""


@dataclass
class _Reply:
    blocks: dict[int, _Block] = field(default_factory=dict)
    stop: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    done: bool = False


class AnthropicProvider:
    effort_levels = EFFORT_LEVELS
    """`output_config.effort` is taken for every current Claude model, profile or not: low to max on Opus 4.7 and
    later, Sonnet 5 and later, Haiku 5.5 and Fable. The API refuses what an older model lacks (xhigh before Opus
    4.7, max before 4.6) and says so."""

    def __init__(self, endpoint: Endpoint, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.endpoint = endpoint
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "User-Agent": names.USER_AGENT,
            "Content-Type": "application/json",
            "anthropic-version": API_VERSION,
            "anthropic-beta": THINKING_BINDING_BETA,
        }
        if self.endpoint.api_key:
            headers["x-api-key"] = self.endpoint.api_key.get_secret_value()
        return headers

    def body(self, request: ProviderRequest) -> dict[str, Any]:
        compat = request.compat
        system, messages = wire_messages(request.messages, self.endpoint.ref)
        mark_cache(system, messages)
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "max_tokens": request.max_output_tokens,
            "messages": messages,
            # Long outputs must stream: the API refuses large max_tokens on non-streaming requests.
            "stream": True,
        }
        if system:
            body["system"] = system
        if request.tools:
            body["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in request.tools
            ]
            if compat.send_tool_choice or compat.parallel_tool_calls is False:
                choice: dict[str, Any] = {"type": "auto"}
                if compat.parallel_tool_calls is False:
                    choice["disable_parallel_tool_use"] = True
                body["tool_choice"] = choice
        if request.sampling.temperature is not None:
            body["temperature"] = request.sampling.temperature
        if request.sampling.top_p is not None:
            body["top_p"] = request.sampling.top_p
        if request.reasoning_effort:
            # Anthropic's effort level sets how much the model thinks (and how many tokens it spends).
            body["output_config"] = {"effort": request.reasoning_effort}
        for key, value in compat.extra_body.items():
            # One level deep, so a profile that sets `thinking.display` keeps the block binding below.
            current = body.get(key)
            body[key] = {**current, **value} if isinstance(current, dict) and isinstance(value, dict) else value
        thinking = body.get("thinking")
        if thinking is None or (isinstance(thinking, dict) and thinking.get("type") != "disabled"):
            body["thinking"] = {"type": "adaptive", "block_binding": DROP_STALE_THINKING, **(thinking or {})}
        return body

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        body = self.body(request)
        url = f"{self.endpoint.base_url}/messages"
        timeout = httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0)
        started = time.monotonic()
        emitted = False
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
            for attempt in range(len(RETRY_WAITS_S) + 1):
                try:
                    async with client.stream("POST", url, json=body, headers=self._headers()) as response:
                        if response.status_code >= 400:
                            detail = (await response.aread()).decode(errors="replace")[:500]
                            raise ProviderError(
                                f"HTTP {response.status_code} from {self.endpoint.provider}: {detail}",
                                status=response.status_code,
                                retryable=response.status_code in RETRYABLE_STATUS,
                            ) from None
                        reply = _Reply()
                        async for item in self._read(response, reply):
                            emitted = True
                            yield item
                        duration_ms = int((time.monotonic() - started) * 1000)
                        yield Completed(message=self._finish(reply, duration_ms))
                        return
                except (ProviderError, *TRANSIENT_ERRORS) as error:
                    retryable = not isinstance(error, ProviderError) or error.retryable
                    if emitted or not retryable or attempt == len(RETRY_WAITS_S):
                        if isinstance(error, ProviderError):
                            raise
                        raise ProviderError(
                            f"{type(error).__name__} from {self.endpoint.provider}: {error}", retryable=True
                        ) from error
                    await asyncio.sleep(RETRY_WAITS_S[attempt])

    async def _read(self, response: httpx.Response, reply: _Reply) -> AsyncIterator[StreamItem]:
        async for line in sse_lines(response):
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError as e:
                raise ProviderError(f"undecodable stream data from {self.endpoint.provider}: {e}") from e
            match event.get("type"):
                case "message_start":
                    reply.usage.update((event.get("message") or {}).get("usage") or {})
                case "content_block_start":
                    block = event.get("content_block") or {}
                    started = _Block(type=block.get("type", ""), id=block.get("id", ""), name=block.get("name", ""))
                    if started.type == "tool_use" and block.get("input"):
                        started.parts.append(json.dumps(block["input"]))
                    elif started.type == "text" and block.get("text"):
                        started.parts.append(block["text"])
                    elif started.type == "thinking":
                        started.parts.append(block.get("thinking") or "")
                        started.signature = block.get("signature") or ""
                    elif started.type == "redacted_thinking":
                        started.data = block.get("data") or ""
                    reply.blocks[event.get("index", len(reply.blocks))] = started
                case "content_block_delta":
                    item = self._delta(reply, event)
                    if item is not None:
                        yield item
                case "message_delta":
                    reply.stop = (event.get("delta") or {}).get("stop_reason") or reply.stop
                    reply.usage.update({k: v for k, v in (event.get("usage") or {}).items() if v is not None})
                case "message_stop":
                    reply.done = True
                    break
                case "error":
                    error = event.get("error") or {}
                    raise ProviderError(
                        f"stream error from {self.endpoint.provider}: {error.get('message') or error}",
                        retryable=error.get("type") in RETRYABLE_ERRORS,
                    )
                case _:
                    pass
        if not reply.done:
            raise ProviderError(
                f"the stream from {self.endpoint.provider} ended before the reply finished", retryable=True
            )

    @staticmethod
    def _delta(reply: _Reply, event: dict[str, Any]) -> StreamItem | None:
        block = reply.blocks.setdefault(event.get("index", 0), _Block(type="text"))
        delta = event.get("delta") or {}
        match delta.get("type"):
            case "text_delta":
                block.parts.append(delta.get("text", ""))
                return TextChunk(text=delta.get("text", ""))
            case "thinking_delta":
                block.parts.append(delta.get("thinking", ""))
                return ReasoningChunk(text=delta.get("thinking", ""))
            case "input_json_delta":
                block.parts.append(delta.get("partial_json", ""))
            case "signature_delta":
                block.signature += delta.get("signature", "")
            case _:
                pass
        return None

    def _finish(self, reply: _Reply, duration_ms: int) -> AssistantMessage:
        text: list[str] = []
        reasoning: list[str] = []
        signed: list[Block] = []
        calls: list[ToolCall] = []
        invalid: list[InvalidToolCall] = []
        for _, block in sorted(reply.blocks.items()):
            joined = "".join(block.parts)
            if block.type == "text":
                text.append(joined)
            elif block.type == "thinking":
                reasoning.append(joined)
                if block.signature:
                    signed.append({"type": "thinking", "thinking": joined, "signature": block.signature})
            elif block.type == "redacted_thinking" and block.data:
                signed.append({"type": "redacted_thinking", "data": block.data})
            elif block.type == "tool_use":
                arguments, error = parse_arguments(joined)
                if not block.name:
                    invalid.append(InvalidToolCall(id=block.id, name="", raw_arguments=joined, error="no tool name"))
                elif arguments is None:
                    invalid.append(
                        InvalidToolCall(id=block.id, name=block.name, raw_arguments=joined, error=error or "")
                    )
                else:
                    calls.append(ToolCall(id=block.id, name=block.name, arguments=arguments))

        stop = _STOP_KINDS.get(reply.stop or "end_turn", "stop")
        if calls and stop == "stop":
            stop = "tool_calls"
        return AssistantMessage(
            text="".join(text).strip(),
            reasoning="".join(reasoning).strip() or None,
            reasoning_blocks=signed,
            tool_calls=calls,
            invalid_tool_calls=invalid,
            model=self.endpoint.ref,
            stop=stop,
            usage=priced(_usage(reply.usage), self.endpoint.ref),  # the API reports tokens, not dollars
            duration_ms=duration_ms,
            error="the model refused to answer" if reply.stop == "refusal" else None,
        )
