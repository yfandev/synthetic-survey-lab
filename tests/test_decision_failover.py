from __future__ import annotations

import asyncio
import json
import math
import traceback

import httpx
import pytest
from pydantic import SecretStr

from intern.decision_failover import BudgetDecisionBackend, FallbackDecisionBackend
from intern.decisions import (
    Answers,
    DecisionClient,
    DecisionsApiBackend,
    NoulAnswer,
    NoulQuestion,
    Question,
    System1Unavailable,
)
from intern.events import Decision
from intern.messages import Usage
from intern.providers.base import Completed, ProviderError, ProviderRequest
from intern.providers.failover import BudgetPolicy, BudgetProvider, RunBudget
from intern.providers.fake import FakeProvider, Turn

QUESTIONS: dict[str, Question] = {"ready": NoulQuestion(instructions="The work is ready.")}


async def test_real_native_decision_transport_failover_shared_with_system2_and_client_cache():
    called = []

    def respond(request):
        body = json.loads(request.content)
        called.append(body)
        if body["model"] == "primary":
            return httpx.Response(503, text="credential-bearing diagnostic")
        return httpx.Response(200, json={"answers": {"ready": {"type": "noul", "noul": 0.9}}, "usage": {"cost": 0.02}})

    budget = RunBudget(BudgetPolicy(max_calls=3, max_observed_cost_usd=0.1))
    backend = FallbackDecisionBackend(
        [
            BudgetDecisionBackend(
                DecisionsApiBackend(
                    "https://fixture.invalid/v1",
                    SecretStr("synthetic"),
                    [model],
                    transport=httpx.MockTransport(respond),
                ),
                budget,
            )
            for model in ["primary", "backup"]
        ]
    )
    events = []
    client = DecisionClient(backend, events.append)
    answer = await client.ask("review", "Shared host State", QUESTIONS)
    assert answer["ready"] == NoulAnswer(noul=0.9)
    assert await client.ask("review", "Shared host State", QUESTIONS) == answer
    assert len(called) == 2 and all(call["state"] == "Shared host State" for call in called)

    class CostProvider(FakeProvider):
        async def stream(self, request):
            async for item in super().stream(request):
                if isinstance(item, Completed):
                    item = item.model_copy(
                        update={"message": item.message.model_copy(update={"usage": Usage(cost_usd=0.03)})}
                    )
                yield item

    provider = BudgetProvider(CostProvider([Turn(text="done")]), budget)
    _ = [item async for item in provider.stream(ProviderRequest(model="system2", max_output_tokens=100, messages=[]))]
    state = await budget.snapshot()
    assert state.calls == 3 and state.completed_calls == 2 and state.unknown_usage_calls == 1
    assert state.unknown_output_calls == 2 and state.usage.cost_usd == pytest.approx(0.05)
    assert client.usage.cost_usd == 0.02
    assert isinstance(events[-1], Decision) and events[-1].cached
    with pytest.raises(System1Unavailable):
        await client.ask("review", "new state", QUESTIONS)
    assert len(called) == 2  # exhausted shared budget cannot trigger another paid call


async def test_a_broken_backend_stops_the_ask_without_leaking_its_diagnostic():
    class Broken:
        name = "broken"

        async def decide(self, state, questions):
            raise RuntimeError("private backend diagnostic")

    budget = RunBudget(BudgetPolicy(max_calls=20))
    events = []
    client = DecisionClient(FallbackDecisionBackend([BudgetDecisionBackend(Broken(), budget)]), events.append)
    with pytest.raises(System1Unavailable) as stopped:
        await client.ask("review", "state", QUESTIONS)
    assert (await budget.snapshot()).calls == len(events) >= 2
    assert "private" not in str(events) and "private" not in str(stopped.value)


async def test_cost_only_reservation_blocks_snapshot_and_cancellation_settles_unknown_usage():
    started = asyncio.Event()

    class Waiting:
        name = "waiting"

        async def decide(self, state, questions):
            started.set()
            await asyncio.Event().wait()
            return Answers(answers={})

    budget = RunBudget(BudgetPolicy(max_calls=2))
    task = asyncio.create_task(BudgetDecisionBackend(Waiting(), budget).decide("state", QUESTIONS))
    await started.wait()
    with pytest.raises(ValueError, match="settled"):
        await budget.snapshot()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = await budget.snapshot()
    assert state.calls == state.unknown_usage_calls == state.unknown_output_calls == 1


async def test_decision_output_token_cap_is_refused_before_transport():
    class Never:
        name = "never"

        async def decide(self, state, questions):
            raise AssertionError("must not dispatch an unmetered output")

    budget = RunBudget(BudgetPolicy(max_completed_output_tokens=10))
    client = DecisionClient(BudgetDecisionBackend(Never(), budget), lambda _: None)
    with pytest.raises(System1Unavailable):
        await client.ask("review", "state", QUESTIONS)
    assert (await budget.snapshot()).calls == 0


async def test_shared_deadline_cancels_decision_and_does_not_retry():
    class Waiting:
        name = "waiting"

        async def decide(self, state, questions):
            await asyncio.sleep(10)
            return Answers(answers={})

    budget = RunBudget(BudgetPolicy(deadline_unix=10.01), clock=lambda: 10)
    with pytest.raises(ProviderError, match="deadline"):
        await BudgetDecisionBackend(Waiting(), budget).decide("state", QUESTIONS)
    assert (await budget.snapshot()).unknown_usage_calls == 1


async def test_active_system1_call_reserves_shared_capacity_before_system2_can_dispatch():
    started, release = asyncio.Event(), asyncio.Event()

    class Active:
        name = "active"

        async def decide(self, state, questions):
            started.set()
            await release.wait()
            return Answers(answers={"ready": NoulAnswer(noul=1)})

    budget = RunBudget(BudgetPolicy(max_calls=1))
    decision = asyncio.create_task(BudgetDecisionBackend(Active(), budget).decide("state", QUESTIONS))
    await started.wait()
    live = FakeProvider([Turn(text="must not run")])
    with pytest.raises(ProviderError, match="budget"):
        _ = [
            item
            async for item in BudgetProvider(live, budget).stream(
                ProviderRequest(model="system2", max_output_tokens=100, messages=[])
            )
        ]
    assert not live.requests
    release.set()
    assert (await decision).answers["ready"] == NoulAnswer(noul=1)
    assert (await budget.snapshot()).calls == 1


@pytest.mark.parametrize("cost", [math.nan, math.inf, -0.01])
async def test_standalone_fallback_refuses_invalid_cost_without_poisoning_client(cost):
    called = []

    class BadCost:
        name = "bad-cost"

        async def decide(self, state, questions):
            called.append("primary")
            return Answers(answers={"ready": NoulAnswer(noul=1)}, cost_usd=cost)

    class Backup:
        name = "backup"

        async def decide(self, state, questions):
            called.append("backup")
            return Answers(answers={"ready": NoulAnswer(noul=1)})

    client = DecisionClient(FallbackDecisionBackend([BadCost(), Backup()], recoverable=(Exception,)), lambda _: None)
    with pytest.raises(System1Unavailable):
        await client.ask("review", "State", QUESTIONS)
    assert client.usage.cost_usd == 0
    assert set(called) == {"primary"}


@pytest.mark.parametrize("deadline", [None, 11])
async def test_budget_does_not_relabel_backend_timeout_as_expired_deadline(deadline):
    called = []

    class TransportTimeout:
        name = "timeout"

        async def decide(self, state, questions):
            called.append("primary")
            raise TimeoutError("private transport diagnostic")

    class Backup:
        name = "backup"

        async def decide(self, state, questions):
            called.append("backup")
            return Answers(answers={"ready": NoulAnswer(noul=1)}, cost_usd=0.01)

    budget = RunBudget(BudgetPolicy(max_calls=2, deadline_unix=deadline), clock=lambda: 10)
    client = DecisionClient(
        FallbackDecisionBackend(
            [BudgetDecisionBackend(TransportTimeout(), budget), BudgetDecisionBackend(Backup(), budget)]
        ),
        lambda _: None,
    )
    assert (await client.ask("review", "State", QUESTIONS))["ready"] == NoulAnswer(noul=1)
    assert await client.ask("review", "State", QUESTIONS) == {"ready": NoulAnswer(noul=1)}
    assert called == ["primary", "backup"]
    state = await budget.snapshot()
    assert state.calls == 2 and state.completed_calls == state.unknown_usage_calls == 1
    assert state.usage.cost_usd == pytest.approx(0.01)


async def test_client_cancellation_settles_budget_without_retry():
    started = asyncio.Event()
    called = []

    class Active:
        name = "active"

        async def decide(self, state, questions):
            called.append("primary")
            started.set()
            await asyncio.Event().wait()
            return Answers(answers={})

    class Backup:
        name = "backup"

        async def decide(self, state, questions):
            called.append("backup")
            return Answers(answers={})

    budget = RunBudget(BudgetPolicy(max_calls=2))
    client = DecisionClient(
        FallbackDecisionBackend([BudgetDecisionBackend(Active(), budget), BudgetDecisionBackend(Backup(), budget)]),
        lambda _: None,
    )
    task = asyncio.create_task(client.ask("review", "State", QUESTIONS))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    state = await budget.snapshot()
    assert called == ["primary"]
    assert state.calls == state.unknown_usage_calls == state.unknown_output_calls == 1


async def test_standalone_decision_transport_timeout_withholds_details_from_client_events():
    class PrivateTimeout:
        name = "timeout"

        async def decide(self, state, questions):
            raise TimeoutError("credential-bearing-private-timeout")

    backend = BudgetDecisionBackend(PrivateTimeout(), RunBudget(BudgetPolicy(max_calls=2)))
    with pytest.raises(TimeoutError) as captured:
        await backend.decide("State", QUESTIONS)
    assert "credential-bearing-private" not in "".join(traceback.format_exception(captured.value))
    events = []
    client = DecisionClient(backend, events.append)
    with pytest.raises(System1Unavailable):
        await client.ask("review", "State", QUESTIONS)
    assert events[0].error == "TimeoutError: decision backend transport timed out"
    assert "credential-bearing-private" not in str(events)
