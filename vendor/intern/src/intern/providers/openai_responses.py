"""Adapter for OpenAI's native Responses API.

OpenAI's chat completions endpoint refuses function tools for GPT-5.6 and later ("Function tools with
reasoning_effort are not supported"), so OpenAI models go through `/v1/responses` instead.

Requests are stateless (`store: false`): the whole conversation is sent each time, as with every other
provider. Reasoning comes back encrypted and replays to the model that wrote it, so the model continues
its reasoning across tool calls; the readable summary is what the journal shows.
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

RETRY_WAITS_S = (2.0, 8.0, 20.0)
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
RETRYABLE_ERRORS = frozenset({"server_error", "rate_limit_exceeded"})
_INCOMPLETE_STOPS: dict[str, StopKind] = {"max_output_tokens": "length", "content_filter": "error"}

Item = dict[str, Any]


def _image_message(images: list[ImagePart]) -> Item:
    """The images a turn's tool calls showed, as the user message after their results."""
    parts: list[Item] = [{"type": "input_text", "text": "Images from the tool results above:"}]
    for image in images:
        data = image_base64(image)
        if data is None:
            parts.append({"type": "input_text", "text": missing_image_note(image)})
        else:
            parts.append({"type": "input_text", "text": f"[{image.source}]"})
            parts.append({"type": "input_image", "image_url": f"data:{image.mime};base64,{data}"})
    return {"role": "user", "content": parts}


def wire_input(messages: list[Message], model_ref: str | None = None) -> tuple[str, list[Item]]:
    """Split out the instructions and turn the rest into input items.

    Reasoning items go back only to the model that wrote them (`model_ref`): their encrypted content is
    valid for it alone. Item ids are left out, since nothing is stored server-side to refer to.
    """
    instructions: list[str] = []
    items: list[Item] = []
    images: list[ImagePart] = []
    for message in messages:
        if images and not isinstance(message, ToolResultMessage):
            items.append(_image_message(images))
            images = []
        match message:
            case SystemMessage(content=text):
                if text.strip():
                    instructions.append(text)
            case UserMessage(content=text):
                items.append({"role": "user", "content": text})
            case AssistantMessage():
                if model_ref is not None and message.model == model_ref:
                    items += [dict(b) for b in message.reasoning_blocks if b.get("type") == "reasoning"]
                if message.text.strip():
                    items.append({"role": "assistant", "content": message.text})
                items += [
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                    for call in message.tool_calls
                ]
            case ToolResultMessage(result=result):
                items.append({"type": "function_call_output", "call_id": result.call_id, "output": result.content})
                images += result.images
    if images:
        items.append(_image_message(images))
    return "\n\n".join(instructions), items


def _usage(raw: dict[str, Any] | None) -> Usage:
    if not raw:
        return Usage()
    input_details = raw.get("input_tokens_details") or {}
    output_details = raw.get("output_tokens_details") or {}
    return Usage(
        input_tokens=raw.get("input_tokens") or 0,
        output_tokens=raw.get("output_tokens") or 0,
        cache_read_tokens=input_details.get("cached_tokens") or 0,
        cache_write_tokens=input_details.get("cache_write_tokens") or 0,
        reasoning_tokens=output_details.get("reasoning_tokens") or 0,
    )


@dataclass
class _Reply:
    items: list[Item] = field(default_factory=list)
    response: dict[str, Any] | None = None


class OpenAIResponsesProvider:
    effort_levels = ("low", "medium", "high", "xhigh")
    """`reasoning.effort` is taken for every current GPT model, profile or not: this run on all of them; `none` and
    `minimal` on some only, so a profile names those."""

    def __init__(self, endpoint: Endpoint, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.endpoint = endpoint
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": names.USER_AGENT, "Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key.get_secret_value()}"
        return headers

    def body(self, request: ProviderRequest) -> dict[str, Any]:
        compat = request.compat
        instructions, items = wire_input(request.messages, self.endpoint.ref if request.reasoning_replay else None)
        reasoning: dict[str, Any] = {"summary": "auto"}
        if request.reasoning_effort:
            reasoning["effort"] = request.reasoning_effort
        body: dict[str, Any] = {
            "model": self.endpoint.model,
            "input": items,
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": request.max_output_tokens,
            "reasoning": reasoning,
        }
        if instructions:
            body["instructions"] = instructions
        if request.tools:
            body["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
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
        body.update(compat.extra_body)
        return body

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamItem]:
        body = self.body(request)
        url = f"{self.endpoint.base_url}/responses"
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
                case "response.output_text.delta":
                    if event.get("delta"):
                        yield TextChunk(text=event["delta"])
                case "response.reasoning_summary_text.delta" | "response.reasoning_text.delta":
                    if event.get("delta"):
                        yield ReasoningChunk(text=event["delta"])
                case "response.output_item.done":
                    reply.items.append(event.get("item") or {})
                case "response.completed" | "response.incomplete":
                    reply.response = event.get("response") or {}
                    break
                case "response.failed":
                    error = (event.get("response") or {}).get("error") or {}
                    raise ProviderError(
                        f"stream error from {self.endpoint.provider}: {error.get('message') or error}",
                        retryable=error.get("code") in RETRYABLE_ERRORS,
                    )
                case "error":
                    raise ProviderError(
                        f"stream error from {self.endpoint.provider}: {event.get('message') or event}",
                        retryable=event.get("code") in RETRYABLE_ERRORS,
                    )
                case _:
                    pass
        if reply.response is None:
            raise ProviderError(
                f"the stream from {self.endpoint.provider} ended before the reply finished", retryable=True
            )

    def _finish(self, reply: _Reply, duration_ms: int) -> AssistantMessage:
        text: list[str] = []
        reasoning: list[str] = []
        replayable: list[Item] = []
        refusals: list[str] = []
        calls: list[ToolCall] = []
        invalid: list[InvalidToolCall] = []
        for item in reply.items:
            match item.get("type"):
                case "message":
                    for part in item.get("content") or []:
                        if part.get("type") == "output_text":
                            text.append(part.get("text") or "")
                        elif part.get("type") == "refusal":
                            refusals.append(part.get("refusal") or "")
                case "reasoning":
                    reasoning += [s.get("text") or "" for s in item.get("summary") or []]
                    if item.get("encrypted_content"):
                        replayable.append(
                            {
                                "type": "reasoning",
                                "summary": item.get("summary") or [],
                                "encrypted_content": item["encrypted_content"],
                            }
                        )
                case "function_call":
                    call_id, name, raw = item.get("call_id") or "", item.get("name") or "", item.get("arguments") or ""
                    arguments, error = parse_arguments(raw)
                    if not name:
                        invalid.append(InvalidToolCall(id=call_id, name="", raw_arguments=raw, error="no tool name"))
                    elif arguments is None:
                        invalid.append(InvalidToolCall(id=call_id, name=name, raw_arguments=raw, error=error or ""))
                    else:
                        calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
                case _:
                    pass

        response = reply.response or {}
        stop: StopKind = "stop"
        if response.get("status") == "incomplete":
            reason = (response.get("incomplete_details") or {}).get("reason") or ""
            stop = _INCOMPLETE_STOPS.get(reason, "length")
        elif calls:
            stop = "tool_calls"
        return AssistantMessage(
            text="".join(text).strip(),
            reasoning="\n\n".join(r for r in reasoning if r).strip() or None,
            reasoning_blocks=replayable,
            tool_calls=calls,
            invalid_tool_calls=invalid,
            model=self.endpoint.ref,
            stop=stop,
            usage=priced(_usage(response.get("usage")), self.endpoint.ref),  # the API reports tokens, not dollars
            duration_ms=duration_ms,
            error=f"the model refused to answer: {' '.join(refusals)}" if refusals else None,
        )
