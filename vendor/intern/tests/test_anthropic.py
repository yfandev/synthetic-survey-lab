"""The native Anthropic provider: request body, replayed thinking and the streamed reply."""

from __future__ import annotations

import json
import ssl
from typing import Any

import httpx
import pytest

from intern.messages import AssistantMessage, UserMessage
from intern.models.profile import Compat
from intern.models.registry import Endpoint
from intern.providers import anthropic
from intern.providers.anthropic import AnthropicProvider, mark_cache, wire_messages
from intern.providers.base import Completed, ProviderError, ProviderRequest

REF = "anthropic/m"


def sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)


def provider(handler: Any = None) -> AnthropicProvider:
    endpoint = Endpoint(provider="anthropic", model="m", base_url="http://x", api="anthropic")
    return AnthropicProvider(endpoint, transport=httpx.MockTransport(handler or (lambda r: httpx.Response(500))))


def request(**overrides: Any) -> ProviderRequest:
    return ProviderRequest(model="m", messages=[UserMessage(content="hi")], max_output_tokens=10, **overrides)


OK_REPLY = sse(
    {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
    {"type": "message_stop"},
)


async def collect(p: AnthropicProvider, req: ProviderRequest) -> AssistantMessage:
    items = [item async for item in p.stream(req)]
    assert isinstance(items[-1], Completed)
    return items[-1].message


def test_reasoning_effort_is_sent_as_the_effort_level_and_extra_body_still_wins():
    provider = AnthropicProvider(Endpoint(provider="anthropic", model="m", base_url="http://x", api="anthropic"))
    request = ProviderRequest(model="m", messages=[UserMessage(content="hi")], max_output_tokens=10)
    assert "output_config" not in provider.body(request)
    request.reasoning_effort = "low"
    assert provider.body(request)["output_config"] == {"effort": "low"}
    request.compat = Compat(extra_body={"output_config": {"effort": "high"}})
    assert provider.body(request)["output_config"] == {"effort": "high"}


def test_thinking_blocks_are_kept_with_signatures_and_replayed_only_to_their_model():
    from intern.messages import AssistantMessage, ToolCall
    from intern.providers.anthropic import _Block, _Reply, wire_messages

    provider = AnthropicProvider(Endpoint(provider="anthropic", model="m", base_url="http://x", api="anthropic"))
    reply = _Reply(stop="tool_use")
    reply.blocks[0] = _Block(type="thinking", parts=["plan: list files"], signature="sig")
    reply.blocks[1] = _Block(type="redacted_thinking", data="opaque")
    reply.blocks[2] = _Block(type="tool_use", id="t1", name="bash", parts=['{"command": "ls"}'])
    message = provider._finish(reply, 5)  # pyright: ignore[reportPrivateUsage]
    assert message.reasoning == "plan: list files"
    assert message.reasoning_blocks == [
        {"type": "thinking", "thinking": "plan: list files", "signature": "sig"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]
    assert message.tool_calls == [ToolCall(id="t1", name="bash", arguments={"command": "ls"})]

    _, wire = wire_messages([UserMessage(content="go"), message], provider.endpoint.ref)
    assert [b["type"] for b in wire[1]["content"]] == ["thinking", "redacted_thinking", "tool_use"]
    other = AssistantMessage(text="hi", reasoning_blocks=message.reasoning_blocks, model="anthropic/other")
    _, wire = wire_messages([UserMessage(content="go"), other], provider.endpoint.ref)
    assert [b["type"] for b in wire[1]["content"]] == ["text"]


def test_stale_thinking_is_dropped_instead_of_failing_and_profile_thinking_settings_are_kept():
    p = provider()
    assert p.body(request())["thinking"] == {"type": "adaptive", "block_binding": anthropic.DROP_STALE_THINKING}
    profile = Compat(extra_body={"thinking": {"type": "adaptive", "display": "summarized"}})
    assert p.body(request(compat=profile))["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
        "block_binding": anthropic.DROP_STALE_THINKING,
    }
    off = Compat(extra_body={"thinking": {"type": "disabled"}})
    assert p.body(request(compat=off))["thinking"] == {"type": "disabled"}
    assert p._headers()["anthropic-beta"] == anthropic.THINKING_BINDING_BETA  # pyright: ignore[reportPrivateUsage]


def test_extra_body_merges_one_level_deep():
    req = request(reasoning_effort="low", compat=Compat(extra_body={"output_config": {"format": "x"}, "top_k": 5}))
    body = provider().body(req)
    assert body["output_config"] == {"effort": "low", "format": "x"} and body["top_k"] == 5


def test_cache_marks_never_reach_the_stored_reply():
    reply = AssistantMessage(reasoning_blocks=[{"type": "thinking", "thinking": "t", "signature": "s"}], model=REF)
    system, wire = wire_messages([UserMessage(content="go"), reply], REF)
    mark_cache(system, wire)
    assert wire[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in reply.reasoning_blocks[0]


async def test_an_overloaded_stream_is_retried_before_any_output(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(anthropic, "RETRY_WAITS_S", (0.0,))
    replies = [sse({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}), OK_REPLY]
    message = await collect(provider(lambda r: httpx.Response(200, content=replies.pop(0))), request())
    assert message.text == "ok" and message.stop == "stop"


async def test_a_client_error_is_not_retried():
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, json={"error": {"message": "bad"}})

    with pytest.raises(ProviderError) as raised:
        await collect(provider(handler), request())
    assert raised.value.status == 400 and calls == 1


async def test_a_reply_is_priced_from_the_models_published_price_since_the_api_reports_only_tokens():
    usage = {"input_tokens": 100, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 10}
    body = sse(
        {"type": "message_start", "message": {"usage": usage}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 50}},
        {"type": "message_stop"},
    )
    endpoint = Endpoint(provider="anthropic", model="claude-opus-5-5", base_url="http://x", api="anthropic")
    opus = AnthropicProvider(endpoint, transport=httpx.MockTransport(lambda r: httpx.Response(200, content=body)))
    message = await collect(opus, request())
    assert message.usage.cost_usd == pytest.approx((100 * 4 + 1000 * 0.20 + 10 * 5 + 50 * 20) / 1e6)
    unpriced = await collect(provider(lambda r: httpx.Response(200, content=body)), request())
    assert unpriced.usage.cost_usd == 0  # a model with no published price here: not invented


async def test_a_refusal_ends_the_reply_as_an_error():
    body = sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "message_delta", "delta": {"stop_reason": "refusal"}, "usage": {"output_tokens": 0}},
        {"type": "message_stop"},
    )
    message = await collect(provider(lambda r: httpx.Response(200, content=body)), request())
    assert message.stop == "error" and message.error


async def test_a_tls_error_before_any_output_is_retried(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(anthropic, "RETRY_WAITS_S", (0.0,))
    calls = 0

    def handler(req: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ssl.SSLError("bad record mac")
        return httpx.Response(200, content=OK_REPLY)

    message = await collect(provider(handler), request())
    assert calls == 2 and message.text == "ok"
