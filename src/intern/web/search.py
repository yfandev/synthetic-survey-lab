"""One small, typed search contract over explicitly selected providers.

Provider SDKs are deliberately unnecessary. Credentials and endpoints are operator configuration,
never model arguments. Search returns source evidence, not a provider-generated answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError, field_validator

from intern import names

SearchBackend = Literal["brave", "tavily", "exa", "searxng"]
SEARCH_PROVIDER_ENV = f"{names.ENV_PREFIX}SEARCH_PROVIDER"
MAX_RESPONSE_BYTES = 2_000_000
MAX_CANDIDATES = 100


class SearchError(Exception):
    """A sanitized provider failure with an actionable, credential-free message."""


def canonical_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("an HTTP(S) source URL without credentials is required")
    # Do not rewrite queries: they can be necessary to identify the original source.
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=5, ge=1, le=10)
    domains: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value.strip()

    @field_validator("domains")
    @classmethod
    def domain_names(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            value = value.strip().lower().rstrip(".").encode("idna").decode("ascii")
            if len(value) > 253 or not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value):
                raise ValueError("domains must be hostnames, without schemes, paths, ports or wildcards")
            if any(
                not label or len(label) > 63 or label.startswith("-") or label.endswith("-")
                for label in value.split(".")
            ):
                raise ValueError("invalid domain hostname")
            if value not in result:
                result.append(value)
        return result


class SearchHit(BaseModel):
    source_id: str
    title: str
    url: str
    excerpts: list[str] = Field(default_factory=list)
    published_at: str | None = None
    provider_rank: int


class SearchResponse(BaseModel):
    provider: SearchBackend
    query: str
    fetched_at: datetime
    hits: list[SearchHit]
    warnings: list[str] = Field(default_factory=list)


class SearchProvider(Protocol):
    async def search(self, request: SearchRequest, abort: asyncio.Event) -> SearchResponse: ...


class SearchSettings(BaseModel):
    backend: SearchBackend
    api_key: SecretStr | None = None
    searxng_url: str | None = None
    timeout_s: float = Field(default=20, gt=0, le=120)
    max_attempts: int = Field(default=2, ge=1, le=3)

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SearchSettings:
        backend = env.get(SEARCH_PROVIDER_ENV, "").strip().lower()
        keys = {"brave": "BRAVE_SEARCH_API_KEY", "tavily": "TAVILY_API_KEY", "exa": "EXA_API_KEY"}
        if backend not in {*keys, "searxng"}:
            raise SearchError(f"Set {SEARCH_PROVIDER_ENV} to brave, tavily, exa or searxng.")
        if backend == "searxng":
            url = env.get("SEARXNG_URL", "").strip()
            try:
                canonical_url(url)
            except ValueError:
                raise SearchError("Set SEARXNG_URL to your configured HTTP(S) search instance.") from None
            return cls(backend="searxng", searxng_url=url.rstrip("/"))
        key = env.get(keys[backend], "").strip()
        if not key:
            raise SearchError(f"The selected {backend} search provider needs {keys[backend]}.")
        # Validation keeps this operator-controlled selection separate from model arguments.
        return cls.model_validate({"backend": backend, "api_key": SecretStr(key)})


class _BravePage(BaseModel):
    url: str
    title: str = ""
    snippets: list[str] = Field(default_factory=list)


class _BraveGrounding(BaseModel):
    generic: list[_BravePage] = Field(default_factory=list)


class _BraveReply(BaseModel):
    grounding: _BraveGrounding


class _TavilyPage(BaseModel):
    title: str = ""
    url: str
    content: str = ""
    published_date: str | None = None


class _TavilyReply(BaseModel):
    results: list[_TavilyPage]


class _ExaPage(BaseModel):
    title: str | None = None
    url: str
    highlights: list[str] = Field(default_factory=list)
    text: str | None = None
    publishedDate: str | None = None


class _ExaReply(BaseModel):
    results: list[_ExaPage]


class _SearxPage(BaseModel):
    title: str = ""
    url: str
    content: str = ""
    publishedDate: str | None = None


class _SearxReply(BaseModel):
    results: list[_SearxPage]


class HTTPSearchProvider:
    def __init__(self, settings: SearchSettings, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self._transport = transport

    def _request(self, query: SearchRequest) -> httpx.Request:
        settings = self.settings
        key = settings.api_key.get_secret_value() if settings.api_key else ""
        scoped_query = query.query
        if query.domains:
            scoped_query += " (" + " OR ".join(f"site:{d}" for d in query.domains) + ")"
        if settings.backend == "brave":
            return httpx.Request(
                "POST",
                "https://api.search.brave.com/res/v1/llm/context",
                headers={"X-Subscription-Token": key},
                json={
                    "q": scoped_query,
                    "count": query.limit,
                    "maximum_number_of_urls": query.limit,
                    "operators_fallback": False,
                },
            )
        if settings.backend == "tavily":
            return httpx.Request(
                "POST",
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "query": query.query,
                    "max_results": query.limit,
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_domains": query.domains,
                },
            )
        if settings.backend == "exa":
            return httpx.Request(
                "POST",
                "https://api.exa.ai/search",
                headers={"x-api-key": key},
                json={
                    "query": query.query,
                    "numResults": query.limit,
                    "type": "auto",
                    "includeDomains": query.domains,
                    "contents": {"highlights": True},
                },
            )
        if not settings.searxng_url:
            raise SearchError("The searxng backend requires a configured searxng_url.")
        return httpx.Request("GET", f"{settings.searxng_url}/search", params={"q": scoped_query, "format": "json"})

    def _decode(self, data: bytes, request: SearchRequest) -> SearchResponse:
        backend = self.settings.backend
        candidates: list[tuple[str, str, list[str], str | None]]
        try:
            if backend == "brave":
                candidates = [
                    (p.url, p.title, p.snippets, None) for p in _BraveReply.model_validate_json(data).grounding.generic
                ]
            elif backend == "tavily":
                candidates = [
                    (p.url, p.title, [p.content], p.published_date)
                    for p in _TavilyReply.model_validate_json(data).results
                ]
            elif backend == "exa":
                candidates = [
                    (p.url, p.title or "", p.highlights or [p.text or ""], p.publishedDate)
                    for p in _ExaReply.model_validate_json(data).results
                ]
            else:
                candidates = [
                    (p.url, p.title, [p.content], p.publishedDate)
                    for p in _SearxReply.model_validate_json(data).results
                ]
        except (ValidationError, ValueError):
            raise SearchError(
                f"{backend} returned an invalid search response; try again or select another provider."
            ) from None
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for rank, (url, title, excerpts, published) in enumerate(candidates[:MAX_CANDIDATES], 1):
            try:
                url = canonical_url(url)
            except ValueError:
                continue
            host = urlsplit(url).hostname or ""
            if request.domains and not any(host == d or host.endswith("." + d) for d in request.domains):
                continue
            if url in seen:
                continue
            seen.add(url)
            hits.append(
                SearchHit(
                    source_id="web-" + hashlib.sha256(url.encode()).hexdigest()[:16],
                    title=title.strip() or url,
                    url=url,
                    excerpts=[e for e in excerpts if e.strip()],
                    published_at=published,
                    provider_rank=rank,
                )
            )
            if len(hits) == request.limit:
                break
        warnings = []
        if request.domains:
            warnings.append(
                "Domain filtering is enforced on returned sources; providers may return fewer matching hits."
            )
        return SearchResponse(
            provider=backend, query=request.query, fetched_at=datetime.now(UTC), hits=hits, warnings=warnings
        )

    async def _fetch(self, client: httpx.AsyncClient, request: SearchRequest) -> bytes:
        for attempt in range(self.settings.max_attempts):
            wire = self._request(request)
            async with client.stream(wire.method, wire.url, headers=wire.headers, content=wire.content) as response:
                if response.status_code in {429, 500, 502, 503, 504} and attempt + 1 < self.settings.max_attempts:
                    # Keep retry waiting inside the total deadline. Never sleep an unbounded Retry-After.
                    try:
                        delay = min(2.0, max(0.0, float(response.headers.get("Retry-After", "0.25"))))
                    except ValueError:
                        delay = 0.25
                    await asyncio.sleep(delay)
                    continue
                if response.status_code in {401, 403}:
                    raise SearchError(f"{self.settings.backend} rejected its credentials; check provider setup.")
                if response.status_code >= 300:
                    raise SearchError(f"{self.settings.backend} search failed with HTTP {response.status_code}.")
                # Bound network bytes before any decompression can allocate output.
                # Providers are explicitly asked for identity encoding below.
                if response.headers.get("Content-Encoding", "identity").strip().lower() not in {"", "identity"}:
                    raise SearchError("Search provider returned unexpected compression; use an identity response.")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > MAX_RESPONSE_BYTES:
                        raise SearchError("Search response exceeded the download limit; narrow the query.")
                return bytes(body)
        raise SearchError("Search retry budget exhausted.")

    async def search(self, request: SearchRequest, abort: asyncio.Event) -> SearchResponse:
        if abort.is_set():
            raise SearchError("Search was cancelled.")

        async def perform() -> SearchResponse:
            try:
                async with asyncio.timeout(self.settings.timeout_s):
                    async with httpx.AsyncClient(
                        transport=self._transport,
                        headers={"Accept-Encoding": "identity"},
                        follow_redirects=False,
                        trust_env=False,
                        timeout=self.settings.timeout_s,
                    ) as client:
                        return self._decode(await self._fetch(client, request), request)
            except (TimeoutError, httpx.TimeoutException):
                raise SearchError("Search timed out; narrow the query or try again.") from None
            except httpx.HTTPError:
                raise SearchError(
                    "Search provider could not be reached; check its configuration and network."
                ) from None

        pending = asyncio.create_task(perform())
        cancelled = asyncio.create_task(abort.wait())
        try:
            done, _ = await asyncio.wait({pending, cancelled}, return_when=asyncio.FIRST_COMPLETED)
            if cancelled in done:
                raise SearchError("Search was cancelled.")
            return await pending
        finally:
            for task in (pending, cancelled):
                if not task.done():
                    task.cancel()
            await asyncio.gather(pending, cancelled, return_exceptions=True)


def render_search(response: SearchResponse) -> str:
    """A readable envelope preserving provenance; long results use the engine's existing spill path."""
    return json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2)
