"""Adapter for any OpenAI-compatible chat completions endpoint.

Covers OpenRouter, Fireworks, Together, DeepInfra, Groq, vLLM, SGLang, llama.cpp, Ollama, LM Studio
and MLX servers. Differences between them are `Compat` flags on the model profile.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from intern import names
from intern.journal import new_id
from intern.messages import (
    AssistantMessage,
    ImagePart,
    InvalidToolCall,
    StopKind,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from intern.models.prices import priced
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
from intern.providers.repair import (
    ThinkSplitter,
    extract_text_tool_calls,
    parse_arguments,
    split_unopened_think,
)

RETRY_WAITS_S = (2.0, 8.0, 20.0)
MAX_RETRY_AFTER_S = 60.0
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
_STOP_KINDS: dict[str, StopKind] = {
    "stop": "stop",
    "end_turn": "stop",
    "length": "length",
    "max_tokens": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "error",
}


def reply_usage(raw: dict[str, Any] | None, ref: str) -> Usage:
    if not raw:
        return Usage()
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    usage = Usage(
        input_tokens=raw.get("prompt_tokens") or 0,
        output_tokens=raw.get("completion_tokens") or 0,
        cache_read_tokens=prompt_details.get("cached_tokens") or 0,
        cache_write_tokens=prompt_details.get("cache_write_tokens") or 0,
        reasoning_tokens=completion_details.get("reasoning_tokens") or 0,
        cost_usd=float(raw.get("cost") or 0.0),
    )
    # OpenRouter reports each call's cost; Fireworks and the rest report tokens only.
    return usage if usage.cost_usd else priced(usage, ref)


@dataclass
class _PendingCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class _Reply:
    """Accumulates one reply, streamed or not."""

    text: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    calls: dict[int, _PendingCall] = field(default_factory=dict)
    finish: str | None = None
    usage: dict[str, Any] | None = None

    def add_call_delta(self, delta: dict[str, Any]) -> None:
        index = delta.get("index")
        if not isinstance(index, int):
            # Without an index, a delta carrying an id starts a call and the rest continue the last one.
            index = len(self.calls) if delta.get("id") or not self.calls else max(self.calls)
        pending = self.calls.setdefault(index, _PendingCall())
        pending.id = delta.get("id") or pending.id
        function = delta.get("function") or {}
        pending.name = function.get("name") or pending.name
        pending.arguments += function.get("arguments") or ""


def _image_message(images: list[ImagePart]) -> dict[str, Any]:
    """The images a turn's tool calls showed, as the user message after their results: chat completions
    take images only in user messages."""
    parts: list[dict[str, Any]] = [{"type": "text", "text": "Images from the tool results above:"}]
    for image in images:
        data = image_base64(image)
        if data is None:
            parts.append({"type": "text", "text": missing_image_note(image)})
        else:
            parts.append({"type": "text", "text": f"[{image.source}]"})
            parts.append({"type": "image_url", "image_url": {"url": f"data:{image.mime};base64,{data}"}})
    return {"role": "user", "content": parts}


class OpenAICompatProvider:
    def __init__(self, endpoint: Endpoint, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.endpoint = endpoint
        self._transport = transport

    def _headers(self, request: ProviderRequest) -> dict[str, str]:
        headers = {"User-Agent": names.USER_AGENT, "Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key.get_secret_value()}"
        if self.endpoint.session_header and request.session_id:
            headers[self.endpoint.session_header] = request.session_id
        return headers

    def body(self, request: ProviderRequest) -> dict[str, Any]:
        compat = request.compat
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": self._messages(request),
            "stream": compat.stream,
            compat.max_tokens_field: request.max_output_tokens,
        }
        if compat.stream and compat.stream_usage:
            body["stream_options"] = {"include_usage": True}
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in request.tools
            ]
            if compat.send_tool_choice:
                body["tool_choice"] = "auto"
            if compat.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = compat.parallel_tool_calls
        if request.sampling.temperature is not None:
            body["temperature"] = request.sampling.temperature
        if request.sampling.top_p is not None:
            body["top_p"] = request.sampling.top_p
        if request.reasoning_effort:
            body["reasoning_effort"] = request.reasoning_effort
        body.update(compat.extra_body)
        return body

    def _messages(self, request: ProviderRequest) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        images: list[ImagePart] = []
        for message in request.messages:
            if images and not isinstance(message, ToolResultMessage):
                wire.append(_image_message(images))
                images = []
            match message:
                case SystemMessage(content=content):
                    wire.append({"role": "system", "content": content})
                case UserMessage(content=content):
                    wire.append({"role": "user", "content": content})
                case AssistantMessage():
                    item: dict[str, Any] = {"role": "assistant", "content": message.text}
                    if message.tool_calls:
                        item["content"] = message.text or None
                        item["tool_calls"] = [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
                            }
                            for call in message.tool_calls
                        ]
                    if request.reasoning_replay and message.reasoning and message.model == self.endpoint.ref:
                        item[self.endpoint.reasoning_field] = message.reasoning
                    wire.append(item)
                case ToolResultMessage(result=result):
                    wire.append({"role": "tool", "tool_call_id": result.call_id, "content": result.content})
                    images += result.images
        if images:
            wire.append(_image_message(images))
        return wire

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        body = self.body(request)
        url = f"{self.endpoint.base_url}/chat/completions"
        timeout = httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0)
        started = time.monotonic()
        emitted = False
        async with httpx.AsyncClient(timeout=timeout, transport=self._transport) as client:
            for attempt in range(len(RETRY_WAITS_S) + 1):
                try:
                    async with client.stream("POST", url, json=body, headers=self._headers(request)) as response:
                        if response.status_code >= 400:
                            detail = (await response.aread()).decode(errors="replace")[:500]
                            raise ProviderError(
                                f"HTTP {response.status_code} from {self.endpoint.provider}: {detail}",
                                status=response.status_code,
                                retryable=response.status_code in RETRYABLE_STATUS,
                            ) from None
                        reply = _Reply()
                        splitter = ThinkSplitter() if request.compat.think_tags else None
                        if request.compat.stream:
                            async for item in self._read_sse(response, reply, splitter):
                                emitted = True
                                yield item
                        else:
                            for item in self._read_json(json.loads(await response.aread()), reply, splitter):
                                emitted = True
                                yield item
                        for item in self._flush(splitter, reply):
                            yield item
                        duration_ms = int((time.monotonic() - started) * 1000)
                        yield Completed(message=self._finish(reply, request, duration_ms))
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

    async def _read_sse(
        self, response: httpx.Response, reply: _Reply, splitter: ThinkSplitter | None
    ) -> AsyncIterator[StreamItem]:
        done = False
        async for line in sse_lines(response):
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                done = True
                break
            if not data:
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError as e:
                raise ProviderError(f"undecodable stream data from {self.endpoint.provider}: {e}") from e
            if chunk.get("error"):
                raise ProviderError(f"stream error from {self.endpoint.provider}: {chunk['error']}")
            if chunk.get("usage"):
                reply.usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                for item in self._content(delta, reply, splitter):
                    yield item
                for call in delta.get("tool_calls") or []:
                    reply.add_call_delta(call)
                if choice.get("finish_reason"):
                    reply.finish = choice["finish_reason"]
        if not done and reply.finish is None:
            raise ProviderError(
                f"the stream from {self.endpoint.provider} ended before the reply finished", retryable=True
            )

    def _read_json(self, data: dict[str, Any], reply: _Reply, splitter: ThinkSplitter | None) -> list[StreamItem]:
        if data.get("error"):
            raise ProviderError(f"error from {self.endpoint.provider}: {data['error']}")
        reply.usage = data.get("usage")
        items: list[StreamItem] = []
        for choice in data.get("choices") or []:
            message = choice.get("message") or {}
            items += self._content(message, reply, splitter)
            for position, call in enumerate(message.get("tool_calls") or []):
                reply.add_call_delta({**call, "index": position})
            reply.finish = choice.get("finish_reason")
        return items

    def _content(self, delta: dict[str, Any], reply: _Reply, splitter: ThinkSplitter | None) -> list[StreamItem]:
        items: list[StreamItem] = []
        reasoning = delta.get("reasoning_content") or delta.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            reply.reasoning.append(reasoning)
            items.append(ReasoningChunk(text=reasoning))
        content = delta.get("content")
        if isinstance(content, str) and content:
            pieces = splitter.feed(content) if splitter else [(False, content)]
            items += self._route(pieces, reply)
        return items

    def _flush(self, splitter: ThinkSplitter | None, reply: _Reply) -> list[StreamItem]:
        return self._route(splitter.finish(), reply) if splitter else []

    @staticmethod
    def _route(pieces: list[tuple[bool, str]], reply: _Reply) -> list[StreamItem]:
        items: list[StreamItem] = []
        for is_reasoning, text in pieces:
            if is_reasoning:
                reply.reasoning.append(text)
                items.append(ReasoningChunk(text=text))
            else:
                reply.text.append(text)
                items.append(TextChunk(text=text))
        return items

    def _finish(self, reply: _Reply, request: ProviderRequest, duration_ms: int) -> AssistantMessage:
        text = "".join(reply.text)
        reasoning = "".join(reply.reasoning) or None
        if request.compat.think_tags and reasoning is None:
            reasoning, text = split_unopened_think(text)

        calls: list[ToolCall] = []
        invalid: list[InvalidToolCall] = []
        for _, pending in sorted(reply.calls.items()):
            call_id = pending.id or f"call_{new_id()}"
            arguments, error = parse_arguments(pending.arguments)
            if not pending.name:
                invalid.append(
                    InvalidToolCall(id=call_id, name="", raw_arguments=pending.arguments, error="no tool name")
                )
            elif arguments is None:
                invalid.append(
                    InvalidToolCall(id=call_id, name=pending.name, raw_arguments=pending.arguments, error=error or "")
                )
            else:
                calls.append(ToolCall(id=call_id, name=pending.name, arguments=arguments))

        if not reply.calls and request.compat.text_tool_calls:
            text, found = extract_text_tool_calls(text)
            for call in found:
                call_id = f"call_{new_id()}"
                if call.arguments is None or not call.name:
                    invalid.append(
                        InvalidToolCall(id=call_id, name=call.name, raw_arguments=call.raw, error=call.error or "")
                    )
                else:
                    calls.append(ToolCall(id=call_id, name=call.name, arguments=call.arguments))

        stop = _STOP_KINDS.get(reply.finish or "stop", "stop")
        if calls and stop == "stop":
            stop = "tool_calls"
        return AssistantMessage(
            text=text.strip(),
            reasoning=reasoning.strip() if reasoning else None,
            tool_calls=calls,
            invalid_tool_calls=invalid,
            model=self.endpoint.ref,
            stop=stop,
            usage=reply_usage(reply.usage, self.endpoint.ref),
            duration_ms=duration_ms,
        )
