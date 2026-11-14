"""The native OpenAI Responses provider against recorded wire shapes."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from intern.messages import AssistantMessage, SystemMessage, ToolCall, ToolResult, ToolResultMessage, UserMessage
from intern.models.registry import resolve_endpoint
from intern.providers import openai_responses
from intern.providers.base import Completed, ProviderError, ProviderRequest, ReasoningChunk, ToolSpec
from intern.providers.openai_responses import OpenAIResponsesProvider, wire_input

REF = "openai/gpt-6.1-sol"
READ = ToolSpec(name="read", description="Read a file", parameters={"type": "object", "properties": {}})


def sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def provider(handler: Any = None) -> OpenAIResponsesProvider:
    endpoint = resolve_endpoint(REF, {"OPENAI_API_KEY": "k"})
    return OpenAIResponsesProvider(endpoint, transport=httpx.MockTransport(handler or (lambda r: httpx.Response(500))))


def request(**overrides: Any) -> ProviderRequest:
    fields: dict[str, Any] = {"model": "gpt-6.1-sol", "messages": [UserMessage(content="hi")], "max_output_tokens": 256}
    return ProviderRequest(**{**fields, **overrides})


async def collect(p: OpenAIResponsesProvider, req: ProviderRequest) -> tuple[list[Any], AssistantMessage]:
    items = [item async for item in p.stream(req)]
    assert isinstance(items[-1], Completed)
    return items[:-1], items[-1].message


def completed(status: str = "completed", **response: Any) -> dict[str, Any]:
    return {"type": f"response.{status}", "response": {"status": status, **response}}


REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [{"type": "summary_text", "text": "Need the file."}],
    "encrypted_content": "opaque",
}
CALL = {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "read", "arguments": '{"path":"a.py"}'}


def test_an_openai_model_is_sent_to_the_responses_api_with_its_output_cap_and_flat_tools():
    messages = [
        SystemMessage(content="Be brief."),
        UserMessage(content="read a.py"),
        AssistantMessage(tool_calls=[ToolCall(id="call_1", name="read", arguments={"path": "a.py"})], model=REF),
        ToolResultMessage(result=ToolResult(call_id="call_1", tool_name="read", status="ok", content="print(1)")),
    ]
    body = provider().body(request(messages=messages, tools=[READ]))
    assert provider().endpoint.api == "responses"
    assert body["instructions"] == "Be brief."
    assert body["max_output_tokens"] == 256 and "max_tokens" not in body and "max_completion_tokens" not in body
    assert (body["store"], body["stream"], body["reasoning"]) == (False, True, {"summary": "auto"})
    assert body["tools"] == [{"type": "function", **READ.model_dump()}]
    assert body["input"][1:] == [
        {"type": "function_call", "call_id": "call_1", "name": "read", "arguments": '{"path": "a.py"}'},
        {"type": "function_call_output", "call_id": "call_1", "output": "print(1)"},
    ]
    assert provider().body(request(reasoning_effort="low"))["reasoning"] == {"summary": "auto", "effort": "low"}


async def test_streamed_reasoning_and_tool_calls_replay_only_to_the_model_that_wrote_them():
    body = sse(
        {"type": "response.output_item.added", "item": {**REASONING, "summary": []}},
        {"type": "response.reasoning_summary_text.delta", "delta": "Need the file."},
        {"type": "response.output_item.done", "item": REASONING},
        {"type": "response.function_call_arguments.delta", "delta": '{"path":"a.py"}'},
        {"type": "response.output_item.done", "item": CALL},
        completed(
            usage={
                "input_tokens": 120,
                "output_tokens": 30,
                "input_tokens_details": {"cached_tokens": 100},
                "output_tokens_details": {"reasoning_tokens": 5},
            }
        ),
    )
    chunks, message = await collect(provider(lambda r: httpx.Response(200, content=body)), request(tools=[READ]))
    assert chunks == [ReasoningChunk(text="Need the file.")]
    assert message.reasoning == "Need the file."
    assert message.tool_calls == [ToolCall(id="call_1", name="read", arguments={"path": "a.py"})]
    assert message.stop == "tool_calls"
    usage = message.usage
    assert (usage.input_tokens, usage.cache_read_tokens, usage.reasoning_tokens) == (120, 100, 5)
    assert usage.cost_usd == pytest.approx((20 * 2.0 + 100 * 0.10 + 30 * 10.0) / 1e6)  # GPT-6.1 Sol's price

    _, replayed = wire_input([UserMessage(content="go"), message], REF)
    assert replayed[1] == {"type": "reasoning", "summary": REASONING["summary"], "encrypted_content": "opaque"}
    assert all("id" not in item for item in replayed)
    _, elsewhere = wire_input([UserMessage(content="go"), message], "openai/other")
    assert [item.get("type") for item in elsewhere[1:]] == ["function_call"]


async def test_a_reply_cut_at_the_output_cap_is_a_length_stop():
    message_item = {"type": "message", "content": [{"type": "output_text", "text": "half"}]}
    body = sse(
        {"type": "response.output_text.delta", "delta": "half"},
        {"type": "response.output_item.done", "item": message_item},
        completed("incomplete", incomplete_details={"reason": "max_output_tokens"}),
    )
    _, message = await collect(provider(lambda r: httpx.Response(200, content=body)), request())
    assert (message.text, message.stop) == ("half", "length")


async def test_failed_and_cut_streams_are_errors(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_responses, "RETRY_WAITS_S", ())
    failed = sse(completed("failed", error={"code": "invalid_prompt", "message": "bad input"}))
    with pytest.raises(ProviderError, match="bad input"):
        await collect(provider(lambda r: httpx.Response(200, content=failed)), request())
    cut = sse({"type": "response.output_text.delta", "delta": "half"})
    with pytest.raises(ProviderError, match="ended before the reply finished"):
        await collect(provider(lambda r: httpx.Response(200, content=cut)), request())
