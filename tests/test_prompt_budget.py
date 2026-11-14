"""The cold-start prompt (system prompt plus tool schemas) is paid on every request, so it is capped.

A 32k window is the smallest the harness promises to work in; the cap keeps the fixed cost a small
share of it. Raise it only with a measured reason.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from intern.context import INITIAL_CHARS_PER_TOKEN, tools_chars
from intern.hooks import Extension
from intern.instructions import FileInstructions
from intern.messages import UserMessage
from intern.models.profile import load_profile
from intern.providers.base import ToolSpec
from intern.providers.fake import Turn
from intern.tools import load_packs

# 1,500 until `process` joined the base tools: with its description cut to what a call needs,
# its schema is 940 characters, about 315 tokens, on a prompt of 4,428 of the 4,500 characters 1,500 tokens allowed
# (macOS's temporary path, which the prompt names). 1,789 measured with it; 1,850 leaves room for a longer path,
# and is 5.6% of a 32k window. Telling the agent it hears of a job's end made the schema 1,085: 1,838.
COLD_START_BUDGET_TOKENS = 1_850


def test_cold_start_prompt_fits_its_budget(tmp_path: Path):
    instructions = FileInstructions(tmp_path, context_files=False).load(tmp_path)
    # Installed optional packs are not model-visible until a profile selects them.
    specs = [tool.spec() for tool in load_packs(load_profile("unknown/model").tool_packs)]
    chars = len(instructions.system) + tools_chars(specs)
    assert chars / INITIAL_CHARS_PER_TOKEN <= COLD_START_BUDGET_TOKENS


@pytest.mark.parametrize("source", ["task", "context", "request", "tools"])
async def test_final_request_overflow_is_rejected_before_provider_dispatch(make_harness, source):
    big = "immutable constraint " * 10_000

    class Expand(Extension):
        async def context(self, view, ctx):
            if source == "context":
                view.tail.append(big)

        async def before_provider_request(self, request, ctx):
            if source == "request":
                return request.model_copy(update={"messages": [UserMessage(content=big)]})
            if source == "tools":
                request.tools.append(ToolSpec(name="big", description=big, parameters={}))
            return request

    harness = make_harness([Turn(text="must not be called")], window=4096, extensions=[Expand()])
    outcome = await harness.run(big if source == "task" else "small task")
    assert outcome.reason == "provider_error"
    assert "working input budget" in (outcome.detail or "")
    assert not harness.provider.requests


async def test_final_request_hook_updates_output_room_and_prompt_calibration(make_harness):
    class Replace(Extension):
        async def before_provider_request(self, request, ctx):
            return request.model_copy(
                update={
                    "messages": [UserMessage(content="x" * 6000)],
                    "tools": [],
                    "max_output_tokens": 99999,
                }
            )

    harness = make_harness([Turn(text="done", input_tokens=2005)], window=4096, extensions=[Replace()])
    outcome = await harness.run()
    assert outcome.reason == "completed"
    request = harness.provider.requests[0]
    assert 1 <= request.max_output_tokens <= 4096 - 2004
    assert abs(harness.meter.chars_per_token - 3) < 0.01
