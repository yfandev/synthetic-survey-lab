"""Search adapters and the actual engine tool path, without paid provider calls."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from intern.events import ToolCallEnd
from intern.messages import ToolResultMessage
from intern.providers.fake import Turn
from intern.tools import load_packs
from intern.tools.web_search import WebSearchTool
from intern.web.search import HTTPSearchProvider, SearchError, SearchRequest, SearchSettings


@pytest.mark.parametrize(
    ("backend", "reply", "host", "credential_header"),
    [
        (
            "brave",
            {
                "grounding": {
                    "generic": [
                        {"url": "https://docs.example.org/a#heading", "title": "A", "snippets": ["source evidence"]}
                    ]
                }
            },
            "api.search.brave.com",
            "X-Subscription-Token",
        ),
        (
            "tavily",
            {
                "results": [
                    {
                        "url": "https://docs.example.org/a#heading",
                        "title": "A",
                        "content": "source evidence",
                        "published_date": "2026-11-01",
                    }
                ]
            },
            "api.tavily.com",
            "Authorization",
        ),
        (
            "exa",
            {
                "results": [
                    {"url": "https://docs.example.org/a#heading", "title": "A", "highlights": ["source evidence"]}
                ]
            },
            "api.exa.ai",
            "x-api-key",
        ),
        (
            "searxng",
            {"results": [{"url": "https://docs.example.org/a#heading", "title": "A", "content": "source evidence"}]},
            "search.example.org",
            None,
        ),
    ],
)
async def test_adapters_return_source_evidence_and_keep_credentials_out_of_results(
    backend, reply, host, credential_header
):
    seen = []

    def handler(request: httpx.Request):
        seen.append(request)
        return httpx.Response(200, json=reply)

    provider = HTTPSearchProvider(
        SearchSettings(backend=backend, api_key=SecretStr("private-key"), searxng_url="https://search.example.org"),
        transport=httpx.MockTransport(handler),
    )
    result = await provider.search(SearchRequest(query="verify this", domains=["example.org"]), asyncio.Event())
    assert seen[0].url.host == host
    assert seen[0].headers["Accept-Encoding"] == "identity"
    if credential_header:
        assert "private-key" in seen[0].headers[credential_header]
    assert result.provider == backend
    assert result.hits[0].url == "https://docs.example.org/a"
    assert result.hits[0].excerpts == ["source evidence"]
    assert result.hits[0].source_id.startswith("web-")
    assert result.fetched_at.tzinfo is not None
    assert "private-key" not in result.model_dump_json()
    if backend == "tavily":
        assert json.loads(seen[0].content)["include_answer"] is False


async def test_sources_are_filtered_deduplicated_and_keep_original_rank():
    reply = {
        "results": [
            {"url": "https://evil-example.org/a", "content": "wrong domain"},
            {"url": "file:///etc/passwd"},
            {"url": "https://user:password@example.org/a"},
            {"url": "https://example.org/a#one", "content": "right"},
            {"url": "https://example.org/a#two", "content": "duplicate"},
            {"url": "https://sub.example.org/b", "content": "second"},
        ]
    }
    provider = HTTPSearchProvider(
        SearchSettings(backend="tavily"), transport=httpx.MockTransport(lambda _: httpx.Response(200, json=reply))
    )
    result = await provider.search(SearchRequest(query="q", domains=["example.org"], limit=2), asyncio.Event())
    assert [h.provider_rank for h in result.hits] == [4, 6]
    assert [h.excerpts for h in result.hits] == [["right"], ["second"]]


async def test_transient_status_retries_but_auth_errors_do_not_leak_body_or_keys():
    calls = 0

    def transient(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"results": []})

    settings = SearchSettings(backend="tavily", api_key=SecretStr("secret"))
    provider = HTTPSearchProvider(settings, transport=httpx.MockTransport(transient))
    assert (await provider.search(SearchRequest(query="q"), asyncio.Event())).hits == []
    assert calls == 2
    provider = HTTPSearchProvider(settings, transport=httpx.MockTransport(lambda _: httpx.Response(401, text="secret")))
    with pytest.raises(SearchError, match="credentials") as error:
        await provider.search(SearchRequest(query="q"), asyncio.Event())
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"results": [{"title":"no url"}]}'])
async def test_invalid_provider_responses_are_sanitized(body):
    provider = HTTPSearchProvider(
        SearchSettings(backend="tavily"), transport=httpx.MockTransport(lambda _: httpx.Response(200, content=body))
    )
    with pytest.raises(SearchError, match="invalid search response"):
        await provider.search(SearchRequest(query="q"), asyncio.Event())


async def test_response_and_total_time_are_bounded():
    provider = HTTPSearchProvider(
        SearchSettings(backend="tavily"),
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 2_000_001)),
    )
    with pytest.raises(SearchError, match="download limit"):
        await provider.search(SearchRequest(query="q"), asyncio.Event())

    async def slow(_):
        await asyncio.sleep(10)
        return httpx.Response(200, json={"results": []})

    provider = HTTPSearchProvider(SearchSettings(backend="tavily", timeout_s=0.01), transport=httpx.MockTransport(slow))
    with pytest.raises(SearchError, match="timed out"):
        await provider.search(SearchRequest(query="q"), asyncio.Event())


async def test_cancellation_stops_the_inflight_request():
    started, stopped, abort = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked(_):
        started.set()
        try:
            await asyncio.sleep(10)
        finally:
            stopped.set()
        return httpx.Response(200, json={"results": []})

    provider = HTTPSearchProvider(SearchSettings(backend="tavily"), transport=httpx.MockTransport(blocked))
    task = asyncio.create_task(provider.search(SearchRequest(query="q"), abort))
    await asyncio.wait_for(started.wait(), 1)
    abort.set()
    with pytest.raises(SearchError, match="cancelled"):
        await task
    assert stopped.is_set()


def test_selection_is_explicit_and_domain_arguments_cannot_inject_operators():
    with pytest.raises(SearchError, match="INTERN_SEARCH_PROVIDER"):
        SearchSettings.from_env({"TAVILY_API_KEY": "secret"})
    with pytest.raises(SearchError, match="BRAVE_SEARCH_API_KEY"):
        SearchSettings.from_env({"INTERN_SEARCH_PROVIDER": "brave"})
    with pytest.raises(SearchError, match="SEARXNG_URL"):
        SearchSettings.from_env({"INTERN_SEARCH_PROVIDER": "searxng", "SEARXNG_URL": "file:///tmp/search"})
    for domain in ["example.org OR site:evil.org", "https://example.org", "a..b", "example.org:8080", "*.example.org"]:
        with pytest.raises(ValidationError):
            SearchRequest(query="q", domains=[domain])
    assert SearchRequest(query=" q ", domains=["Example.ORG.", "example.org"]).domains == ["example.org"]


async def test_search_runs_through_real_engine_and_is_absent_from_base(make_harness):
    provider = HTTPSearchProvider(
        SearchSettings(backend="tavily"),
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"results": [{"url": "https://example.org/fact", "content": "verified fact"}]}
            )
        ),
    )
    harness = make_harness([Turn(tool_calls=[("web_search", {"query": "a fact"})]), Turn(text="Done")])
    harness.deps.tools.append(WebSearchTool(provider))
    outcome = await harness.run("Research a fact")
    assert outcome.reason == "completed"
    assert harness.recorder.of(ToolCallEnd)[0].status == "ok"
    results = [m for m in harness.provider.requests[-1].messages if isinstance(m, ToolResultMessage)]
    assert "verified fact" in results[0].result.content
    assert all(t.name != "web_search" for t in load_packs(["base"]))
    assert [t.name for t in load_packs(["web-search"])] == ["web_search"]


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "zstd", "gzip, identity"])
async def test_unexpected_compression_is_refused_before_body_is_consumed(encoding):
    consumed, closed = asyncio.Event(), asyncio.Event()

    class UnboundedCompressedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            consumed.set()
            raise AssertionError("compressed response body must not be consumed or decompressed")
            yield b""  # pragma: no cover - async iterator marker

        async def aclose(self):
            closed.set()

    def handler(request):
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(200, headers={"Content-Encoding": encoding}, stream=UnboundedCompressedBody())

    provider = HTTPSearchProvider(SearchSettings(backend="tavily"), transport=httpx.MockTransport(handler))
    with pytest.raises(SearchError, match="unexpected compression"):
        await provider.search(SearchRequest(query="q"), asyncio.Event())
    assert not consumed.is_set()
    assert closed.is_set()
