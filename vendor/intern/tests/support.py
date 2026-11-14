"""Test helpers shared by several test modules."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel

from intern import hooks as h
from intern.context import STATE_EDIT_PROMPT, STATE_PROMPT, TokenMeter
from intern.decisions import (
    Answer,
    Answers,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionBackend,
    NoulAnswer,
    Question,
    ScoreAnswer,
)
from intern.events import Event, RunEnd, WindowInfo
from intern.loop import EngineDeps, ModelBinding, Run, RunControl
from intern.messages import UserMessage
from intern.models.profile import Profile
from intern.providers.base import Provider, ProviderRequest
from intern.providers.fake import FakeProvider
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput
from intern.tools.files import ReadTool

from .fake_openai import QUIET

STATE_EDIT_HEAD = STATE_EDIT_PROMPT.split("{size}")[0]


def is_state_write(request: ProviderRequest) -> bool:
    """A State write: the agent's own request with the State asked for at the end, whole or as edits (and any
    follow-up of it)."""
    return any(
        isinstance(m, UserMessage) and (STATE_PROMPT in m.content or STATE_EDIT_HEAD in m.content)
        for m in request.messages
    )


def is_state_edit(request: ProviderRequest) -> bool:
    """A State write asking for edits on the State there is."""
    return any(isinstance(m, UserMessage) and STATE_EDIT_HEAD in m.content for m in request.messages)


@dataclass
class Recorder:
    events: list[Event] = field(default_factory=list)

    def __call__(self, event: Event) -> None:
        self.events.append(event)

    def of[T](self, kind: type[T]) -> list[T]:
        return [e for e in self.events if isinstance(e, kind)]

    @property
    def end(self) -> RunEnd:
        return self.of(RunEnd)[-1]


def binding(provider: Provider, window: int = 32_768, profile: Profile | None = None) -> ModelBinding:
    return ModelBinding(
        provider=provider,
        ref="fake/model",
        wire_model="model",
        profile=profile or Profile(),
        window=WindowInfo(
            advertised=window, advertised_source="profile", effective=None, user_cap=None, working=window
        ),
    )


@dataclass
class Harness:
    """A ready engine over a scripted provider, in a temporary working directory."""

    deps: EngineDeps
    recorder: Recorder
    provider: FakeProvider
    cwd: Path
    meter: TokenMeter = field(default_factory=TokenMeter)

    async def run(self, prompt: str = "do the task", control: RunControl | None = None):
        return await Run(self.deps, self.meter, control or RunControl()).execute(prompt)


class GreetArgs(BaseModel):
    who: str


class GreetTool(Tool):
    name = "greet"
    description = "Say hello."
    Args = GreetArgs
    effects: ClassVar[frozenset[Effect]] = frozenset()

    async def run(self, args: GreetArgs, ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content=f"hello {args.who}")


class ReadUpper(ReadTool):
    """A `read` that replaces the pack's, as a document-aware read would."""

    async def run(self, args, ctx: ToolContext) -> ToolOutput:
        out = await super().run(args, ctx)
        return out.model_copy(update={"content": out.content.upper()})


class WithTools(h.Extension):
    name = "with-tools"

    def tools(self) -> list[Tool]:
        return [GreetTool(), ReadUpper()]


class Quiet:
    """A System 1 that judges nothing worth acting on (`fake_openai.QUIET`): the default in tests about System 2."""

    name = "quiet"

    def __init__(self) -> None:
        self.asks: list[tuple[str, dict[str, Question]]] = []

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        self.asks.append((state, questions))
        answers: dict[str, Answer] = {}
        for name, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                pick = next((o for o in QUIET if o in q.criteria), next(iter(q.criteria)))
                answers[name] = ChoiceAnswer(choice=pick, probabilities={o: float(o == pick) for o in q.criteria})
            elif q.type == "score":
                answers[name] = ScoreAnswer(score=0.0)
            else:
                answers[name] = NoulAnswer(noul=0.0)
        return Answers(answers=answers)


class Filled:
    """A test's own System 1, with the questions it leaves out (another caller's: the guard's beside a supervisor
    script) answered quietly, so the test sees only what it scripted."""

    def __init__(self, backend: DecisionBackend) -> None:
        self.backend = backend
        self.name = backend.name

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        result = await self.backend.decide(state, questions)
        if missing := {n: q for n, q in questions.items() if n not in result.answers}:
            quiet = await Quiet().decide(state, missing)
            result = result.model_copy(update={"answers": {**quiet.answers, **result.answers}})
        return result


class Needle:
    """A System 1 for a search test: a question's needle is the first of its words found in the question (a word,
    or a group of words whose share in a card is the judgment); a card naming it scores 1, the rest 0. The state is
    the question and one card, or "[id] card" blocks; a choice picks the block holding the needle. A question that
    shows a passage (the workspace's union) is judged on that passage."""

    name = "needle"

    def __init__(self, *needles: str | tuple[str, ...]) -> None:
        self.groups = [(n,) if isinstance(n, str) else n for n in needles]

    async def decide(self, state: str, questions: dict[str, Question]) -> Answers:
        question, _, body = state.partition("\n\n")
        group = next((g for g in self.groups if g[0] in question), None)
        choices = [q for q in questions.values() if isinstance(q, ChoiceQuestion)]
        if group is None or any("none of these" not in q.criteria.values() for q in choices):
            return Answers(answers={})  # another caller's ask (the supervisor's, the guard's): `Filled` answers it
        parts = re.split(r"^\[(\w+)\] ", body, flags=re.M)
        blocks = {parts[i]: parts[i + 1] for i in range(1, len(parts) - 1, 2)}
        share = lambda text: sum(w in text for w in group) / len(group)  # noqa: E731
        answers: dict[str, Answer] = {}
        for name, q in questions.items():
            if isinstance(q, ChoiceQuestion):
                none = next(c for c in q.criteria if q.criteria[c] == "none of these")
                scored = {c: share(blocks.get(c, "")) for c in q.criteria if c != none}
                best = max(scored, key=lambda c: scored[c]) if any(scored.values()) else none
                answers[name] = ChoiceAnswer(choice=best, probabilities={c: float(c == best) for c in q.criteria})
            elif "\nPassage [" in q.instructions:  # the workspace's union puts each passage in its question
                answers[name] = NoulAnswer(noul=share(q.instructions.split("\nPassage [", 1)[1]))
            else:
                answers[name] = NoulAnswer(noul=share(blocks.get(name, body)))
        return Answers(answers=answers, cost_usd=0.0)
