# Web search (opt-in)

Add `web-search` alongside `base` in a model profile's `tool_packs` to expose `web_search`.
No shipped profile changes. Importing or loading the pack performs no network requests and requires
no provider SDK. Choose one backend explicitly:

```sh
export INTERN_SEARCH_PROVIDER=brave
export BRAVE_SEARCH_API_KEY=...
```

Other selections are `tavily` (`TAVILY_API_KEY`), `exa` (`EXA_API_KEY`), or `searxng`
(`SEARXNG_URL`, with JSON search enabled on the operator's instance). There is no anonymous gateway,
automatic provider switching, or key guessing. A self-hosted search service still contacts its
configured upstream engines.

The model supplies `query`, `limit` (1–10) and optional `domains`. Returned evidence includes stable
URL-derived source IDs, original provider rank, canonical HTTP(S) URLs, excerpts, fetch time and
publication dates when the provider supplies them. Domain restrictions are enforced on returned
hostnames; a provider may return fewer matching hits. Search excerpts are untrusted data, not full
page verification. Unknown publication dates remain unknown. Provider-generated answers are disabled.

Programmatic users may inject a `SearchProvider` into `WebSearchTool` or use `HTTPSearchProvider`
with typed `SearchSettings`. Configuration and credentials never enter the model-facing schema.
Requests have a total deadline, cancellation, a bounded response and retry budget, and sanitized
errors. Only transient failures retry; there is no persistent cache or cost accounting for search
provider credits in this first capability.

This implementation uses the existing Tool/ToolContext seam rather than a new lifecycle hook:
search is a callable capability, not a change to the agent's reasoning policy. The engine's normal
result bounding, spill, journal and tool events apply. The new tool declares read and network
effects, so it does not masquerade as a local read-only tool.

## Validation and promotion

Contract tests exercise all adapters using mock HTTP, error/cancellation boundaries and the real
scripted engine path. A live provider smoke check and paired eval research tasks (same
models/tasks, at least two repeats; accuracy, citations, tokens, cost and wall time) are required
before recommending a provider or enabling this pack by default. No measured provider winner is
claimed here.

Primary implementation references:
[Hermes provider contract](https://github.com/NousResearch/hermes-agent/blob/e26e0e4b67bfae31eddaaefee87c062efd67429b/website/docs/developer-guide/web-search-provider-plugin.md),
[Brave LLM Context](https://api-dashboard.search.brave.com/documentation/services/llm-context),
[Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search),
[Exa Search](https://exa.ai/docs/reference/search),
[SearXNG Search API](https://docs.searxng.org/dev/search_api.html).
