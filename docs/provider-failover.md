# Explicit provider fallback and invocation budgets

`FallbackProvider([FallbackTarget(TargetConfig(...), provider), ...])` implements the
existing Provider protocol. The caller selects and configures every target explicitly.
It advances only for retryable ProviderErrors, rewrites the wire model/output cap and
removes signed reasoning from messages written by another model. The completed response
records its actual model, and typed attempt metadata reports failures and unknown usage.
No model selection or fallback is activated by default.

Replies are buffered until a complete response exists. This intentionally trades live
token streaming for a safe fallback boundary: partial output from a failed provider is
never concatenated with a backup answer. Cancellation still propagates immediately.
The buffer has a byte cap. Configure compatible providers, context windows, tools and
sampling/effort through the normal per-model assembly; this wrapper does not silently
truncate history or guess a backup model's profile. An incompatible request fails.

`BudgetProvider(provider, RunBudget(BudgetPolicy(...)))` enforces attempted-call counts,
reserved completed-output budgets and a wall-clock deadline. Multiple providers can
share one RunBudget. Place it around individual fallback targets when each attempted
backend must count separately. `snapshot()` at a settled boundary returns typed portable
state for a durable runner to persist and restore. It never snapshots in-flight reserves.
Reservations are immutable `BudgetReservation` tickets, not reusable token integers;
settling an unknown, changed or already settled ticket is an explicit error. Invalid
provider usage becomes an unknown-usage call rather than poisoning the ledger.
An expired budget deadline stops the fallback chain. A backend's own transport
timeout remains a recoverable transport failure; it is not relabelled as a budget
deadline. System 2 reports it through a retryable `ProviderError`, while System 1
retains the `TimeoutError` category with a sanitized diagnostic for the explicitly
configured recovery types.

System 1 already has model-list fallback through `System1Config.fallback_models` and
`DecisionsApiBackend`. That existing API remains unchanged. The optional
`FallbackDecisionBackend` additionally composes different typed `DecisionBackend`s
(including native Decisions API instances), and `BudgetDecisionBackend` shares the
same call/cost/deadline ledger with System 2:

```python
from pydantic import SecretStr
from intern.decisions import DecisionsApiBackend
from intern.decision_failover import BudgetDecisionBackend, FallbackDecisionBackend
from intern.providers.failover import BudgetPolicy, BudgetProvider, RunBudget

budget = RunBudget(BudgetPolicy(max_calls=100, max_observed_cost_usd=0.25))
# Assemble while idle. The existing DecisionClient, its cache, retries and
# decision events stay in control.
session.deps.decisions.backend = FallbackDecisionBackend(
    [
        BudgetDecisionBackend(DecisionsApiBackend(url, SecretStr(key), [model]), budget)
        for model in [primary_decision_model, backup_decision_model]
    ]
)
session.deps.system2.provider = BudgetProvider(session.deps.system2.provider, budget)
```

Wrap individual System 2 fallback targets with `BudgetProvider` when each target
should consume a separate call. Avoid wrapping the same dispatch twice. These are
logical backend dispatches; backend-internal retries and batched HTTP requests
retain their existing semantics. No new HTTP client or System 1 decision path is
introduced: extensions still call `ctx.decisions.ask()`.

The decision adapter's recoverable exception types are explicit. Its defaults include
`RuntimeError` because the native `DecisionsApiBackend` normalizes transport errors
and malformed response/answer validation into that type; either can therefore advance
to the next configured backend. A directly returned invalid typed result or a
nonfinite/negative cost stops the outer chain. Native internal model lists and retries
still run inside one backend dispatch. Use one model per native backend when the
shared ledger must count each configured fallback target separately.

System 1's `Answers` reports cost but no token usage. These successful calls increment
`unknown_output_calls`; token counts are not fabricated. A shared budget with an
output-token cap refuses such calls before dispatch. Use a shared call/cost/deadline
budget and a separate System 2 token budget when both are needed. In-flight cost-only
calls also prevent snapshots; cancellation settles them as unknown usage. A shared
ledger is an aggregate view, not an additional charge: do not add its cost again to
the individual provider/DecisionClient costs. Changing models/clients later requires
the host to apply its wrappers to those newly selected instances as well.

Cost limits are explicitly **observed-cost thresholds**, not a promise that paid APIs
cannot overshoot: price is learned after completion (the provider's reported cost, or on Anthropic's
and OpenAI's own APIs and Fireworks the published price of the reply's tokens, `models/prices.toml`), simultaneous in-flight calls may
finish, and a failed stream may have unknown billed usage. Those calls are counted and
flagged, but their unknown cost/output cannot be invented. A host requiring a monetary
hard cap must also reserve known per-target pricing and enforce downstream provider
limits. Provider adapters may retry internally; wrap the network-attempt layer if every
such retry must have an independent reservation. Paired evals are pending before
default activation; no core loop, configuration or profile behavior changes.
