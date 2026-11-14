"""The decision client's cache and retries, and the two wire backends."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from intern import decisions
from intern.decisions import (
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionClient,
    DecisionsApiBackend,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    System1Unavailable,
)
from intern.events import Decision

QUESTIONS = {
    "keep": NoulQuestion(instructions="This output matters for the task."),
    "kind": ChoiceQuestion(instructions="What is it?", criteria={"log": "a log", "code": "source code"}),
}


def answered() -> dict:
    return {"keep": NoulAnswer(noul=0.5), "kind": ChoiceAnswer(choice="log")}


class Scripted:
    name = "scripted"

    def __init__(self, *results: Answers | Exception) -> None:
        self.results = list(results)
        self.calls = 0

    async def decide(self, state: str, questions: dict) -> Answers:
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def client(backend, events: list, **kw: Any) -> DecisionClient:
    return DecisionClient(backend, events.append, **kw)


async def test_an_ask_missing_an_answer_or_failing_is_retried_and_the_answer_is_cached():
    backend = Scripted(
        Answers(answers={"keep": NoulAnswer(noul=0.9)}, cost_usd=0.001),
        RuntimeError("503"),
        Answers(answers=answered()),
    )
    events: list = []
    c = client(backend, events, retry_window_s=5)
    first = await c.ask("triage", "state", QUESTIONS)
    second = await c.ask("triage", "state", QUESTIONS)
    assert first == second == answered()
    assert backend.calls == 3
    assert [(e.ok, e.cached) for e in events if isinstance(e, Decision)] == [
        (False, False),
        (False, False),
        (True, False),
        (True, True),
    ]


async def test_a_system1_that_stays_down_stops_the_ask_instead_of_answering_for_it():
    backend = Scripted(*[RuntimeError("down")] * 50)
    events: list = []
    c = client(backend, events, retry_window_s=0.1)
    with pytest.raises(System1Unavailable, match="down"):
        await c.ask("triage", "state", QUESTIONS)
    assert backend.calls >= 2 and c.usage.failed == 1 and c.usage.asks == 0
    assert all(not e.ok and e.answers is None for e in events)


async def test_decisions_api_fails_over_to_the_next_model():
    seen: list[str] = []

    def handler(r: httpx.Request) -> httpx.Response:
        body = json.loads(r.content)
        seen.append(body["model"])
        if body["model"] == "primary":
            return httpx.Response(503)
        assert body["questions"]["keep"] == {"type": "noul", "instructions": "This output matters for the task."}
        return httpx.Response(
            200,
            json={
                "answers": {
                    "keep": {"type": "noul", "noul": 0.2},
                    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.8, "log": 0.2}},
                },
                "usage": {"cost": 0.00002},
            },
        )

    backend = DecisionsApiBackend(
        "https://example/api/v1", SecretStr("k"), ["primary", "backup"], transport=httpx.MockTransport(handler)
    )
    result = await backend.decide("state", QUESTIONS)
    assert seen == ["primary", "backup"]
    assert result.answers["kind"] == ChoiceAnswer(choice="code", probabilities={"code": 0.8, "log": 0.2})


async def test_decisions_api_posts_to_systemone_rescales_scores_and_keeps_probabilities():
    def handler(r: httpx.Request) -> httpx.Response:
        assert r.url.path == "/v1/systemone"
        score = {"type": "score", "score": 1.5, "probabilities": {"2": 0.6, "0": 0.1, "1": 0.3}, "legend": {}}
        return httpx.Response(200, json={"answers": {"s": score}})

    backend = DecisionsApiBackend(
        "https://console.nace.ai/v1", SecretStr("k"), ["drex-max"], transport=httpx.MockTransport(handler)
    )
    questions: dict[str, decisions.Question] = {
        "s": ScoreQuestion(instructions="How risky?", criteria=["low", "medium", "high"])
    }
    result = await backend.decide("state", questions)
    assert result.answers["s"] == ScoreAnswer(
        score=0.75, probabilities=[0.1, 0.3, 0.6], legend=["low", "medium", "high"]
    )


async def test_decisions_api_splits_asks_over_the_question_cap_and_merges_the_answers():
    sizes: list[int] = []

    def handler(r: httpx.Request) -> httpx.Response:
        asked = json.loads(r.content)["questions"]
        sizes.append(len(asked))
        return httpx.Response(
            200, json={"answers": {k: {"type": "noul", "noul": 0.5} for k in asked}, "usage": {"cost": 0.001}}
        )

    backend = DecisionsApiBackend("https://x/v1", SecretStr("k"), ["decider"], transport=httpx.MockTransport(handler))
    questions: dict[str, Question] = {f"q{i}": NoulQuestion(instructions=f"claim {i}") for i in range(40)}
    result = await backend.decide("state", questions)
    assert sorted(sizes) == [8, 16, 16] and set(result.answers) == set(questions)
    assert abs(result.cost_usd - 0.003) < 1e-9


async def test_decisions_api_keeps_one_connection_across_asks():
    """A search makes hundreds of asks; none should pay for a new connection and TLS handshake."""
    peers: set[tuple[str, int]] = set()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            peers.add(self.client_address)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps({"answers": {"keep": {"type": "noul", "noul": 0.5}}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        backend = DecisionsApiBackend(f"http://127.0.0.1:{server.server_port}/v1", SecretStr("k"), ["decider"])
        for _ in range(5):
            result = await backend.decide("state", {"keep": QUESTIONS["keep"]})
            assert result.answers["keep"] == NoulAnswer(noul=0.5)
    finally:
        server.shutdown()
        server.server_close()
    assert len(peers) == 1


async def test_decisions_api_retries_once_after_a_rate_limit(monkeypatch):
    import intern.decisions as d

    monkeypatch.setattr(d, "RATE_LIMIT_RETRY_S", 0)
    calls: list[str] = []

    def handler(r: httpx.Request) -> httpx.Response:
        calls.append(json.loads(r.content)["model"])
        if len(calls) == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"answers": {"keep": {"type": "noul", "noul": 0.9}}, "usage": {"cost": 0}})

    backend = DecisionsApiBackend("https://x/v1", SecretStr("k"), ["decider"], transport=httpx.MockTransport(handler))
    result = await backend.decide("state", {"keep": QUESTIONS["keep"]})
    assert calls == ["decider", "decider"]
    assert result.answers["keep"] == NoulAnswer(noul=0.9)


async def test_the_event_carries_the_turn_the_subject_and_the_answers_in_compact_form():
    backend = Scripted(
        Answers(
            answers={
                "keep": NoulAnswer(noul=0.9),
                "kind": ChoiceAnswer(choice="log", probabilities={"log": 0.7, "code": 0.3}),
                "size": ScoreAnswer(score=0.5, probabilities=[0.2, 0.6, 0.2], legend=["s", "m", "l"]),
            },
            cost_usd=0.001,
            input_tokens=120,
            output_tokens=9,
        )
    )
    events: list = []
    c = client(backend, events)
    questions = QUESTIONS | {"size": ScoreQuestion(instructions="How big?", criteria=["s", "m", "l"])}
    await c.ask("keep", "state", questions, turn=7, subject="src/app.py")
    await c.ask("keep", "state", questions, turn=8, subject="src/app.py")  # cached
    first, second = [e for e in events if isinstance(e, Decision)]
    assert (first.turn, first.subject, first.cached) == (7, "src/app.py", False)
    assert first.answers == {
        "keep": {"noul": 0.9},
        "kind": {"choice": "log", "probabilities": {"log": 0.7, "code": 0.3}},
        "size": {"score": 0.5, "probabilities": [0.2, 0.6, 0.2]},
    }
    assert (second.turn, second.cached, second.answers) == (8, True, first.answers)
    assert c.usage.input_tokens == 120 and c.usage.output_tokens == 9 and c.usage.asks == 1


class SlowServer(httpx.AsyncBaseTransport):
    """Answers after ``delay`` seconds; like a real connection, it gives up first when the request's read timeout is
    shorter."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        read = (request.extensions.get("timeout") or {}).get("read")
        if read is not None and read < self.delay:
            await asyncio.sleep(read)
            raise httpx.ReadTimeout("timed out", request=request)
        await asyncio.sleep(self.delay)
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.9}}, "usage": {}})


async def test_an_ask_given_more_time_than_the_default_gets_it_from_the_http_request_too():
    backend = DecisionsApiBackend(
        "https://x/v1", SecretStr("k"), ["decider"], timeout_s=0.05, transport=SlowServer(0.2)
    )
    client = DecisionClient(backend, lambda e: None, timeout_s=0.05, retry_window_s=0)
    question: dict[str, Question] = {"q": NoulQuestion(instructions="Yes?")}
    slow = await client.ask("evidence", "state", question, timeout_s=1.0)
    assert slow == {"q": NoulAnswer(noul=0.9)}  # judged, not cut at the backend's 0.05 s
    with pytest.raises(System1Unavailable):  # the default per-attempt limit still holds
        await client.ask("triage", "another state", question)


class HungFirst(SlowServer):
    """The failover list's first model never answers in time; the others answer after ``delay``."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "hung":
            return await SlowServer(60).handle_async_request(request)
        return await super().handle_async_request(request)


async def test_a_model_that_hangs_fails_over_within_an_ask_given_more_time():
    backend = DecisionsApiBackend(
        "https://x/v1", SecretStr("k"), ["hung", "decider"], timeout_s=0.05, transport=HungFirst(0.2)
    )
    client = DecisionClient(backend, lambda e: None, timeout_s=0.05, retry_window_s=0)
    question: dict[str, Question] = {"q": NoulQuestion(instructions="Yes?")}
    # The first model is cut at the backend's 0.05 s, the last has the ask's 1 s: the next model answers.
    assert await client.ask("evidence", "state", question, timeout_s=1.0) == {"q": NoulAnswer(noul=0.9)}
