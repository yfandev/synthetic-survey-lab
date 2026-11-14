from __future__ import annotations

from pathlib import Path

from intern.decisions import Answers, ChoiceAnswer, NoulAnswer
from intern.events import Compaction, Decision, Reduced, ToolCallEnd
from intern.evidence import MIN_CHARS, extract, receipt, verify
from intern.messages import UserMessage
from intern.models.profile import Profile
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn

from .support import is_state_write

PYTEST_LOG = (
    "============================= test session starts ==============================\n"
    "collected 120 items\n\n"
    "tests/test_card.py "
    + "." * 70
    + "F.\n"
    + "\n".join(f"tests/test_other_{i}.py ...... [{i * 2}%]" for i in range(120))
    + "\n\n=================================== FAILURES ===================================\n"
    "_________________________ test_double_single_quotes ____________________________\n\n"
    "    def test_double_single_quotes():\n"
    "        card = Card('CONFIG', \"x'''\")\n"
    ">       assert card.value == \"x'''\"\n"
    "E       AssertionError: assert \"x''\" == \"x'''\"\n"
    "E         - x'''\n"
    "E         + x''\n\n"
    "tests/test_card.py:42: AssertionError\n"
    "=========================== short test summary info ============================\n"
    "FAILED tests/test_card.py::test_double_single_quotes - AssertionError: assert \"x''\" == \"x'''\"\n"
    "========================= 1 failed, 119 passed in 2.31s ========================\n\n"
    "Error: the command exited with code 1\n"
    "Next: Read the output above and fix the cause before running it again."
)


def test_a_pytest_log_reduces_to_its_verbatim_evidence_and_keeps_the_error_footer():
    assert len(PYTEST_LOG) >= MIN_CHARS
    items = extract(PYTEST_LOG)
    kinds = [e.kind for e in items]
    assert kinds.count("failure") == 2 and "target" in kinds and kinds[-1] == "summary"
    assert all(e.quote in PYTEST_LOG for e in items)
    assert any(e.quote.startswith("E       AssertionError") and "+ x''" in e.quote for e in items), "the whole E block"
    text = receipt(PYTEST_LOG, items, "error", "/spill/evidence/abc.txt")
    assert verify(PYTEST_LOG, items, "error", text) is None
    assert (
        text.startswith("[Test/build output reduced")
        and "failed (exit code 1)" in text
        and "/spill/evidence/abc.txt" in text
    )
    assert "1 failed, 119 passed" in text and "test_double_single_quotes" in text
    assert text.endswith("Next: Read the output above and fix the cause before running it again.")
    assert len(text) < len(PYTEST_LOG) / 3


def test_the_verifier_refuses_a_failing_log_without_failure_evidence_and_a_log_that_is_not_a_test_log():
    prose = (
        "the build failed somewhere in the middle of a long story " * 80
    ) + "\n\nError: the command exited with code 1\nNext: fix it"
    items = extract(prose)
    assert verify(prose, items, "error", receipt(prose, items, "error", "/a")) in (
        "missing-failure-evidence",
        "no-test-signal",
    )
    printout = "\n".join(f"value[{i}] = {i * 7}" for i in range(400)) + "\n[exit code 0]"
    items = extract(printout)
    assert verify(printout, items, "ok", receipt(printout, items, "ok", "/a")) == "no-test-signal"


def runs_tests(request: ProviderRequest) -> Turn:
    if is_state_write(request):
        return Turn(text="## Plan\n- fix\n## Now\nfix the quotes")
    if len(request.messages) < 3:
        return Turn(tool_calls=[("bash", {"command": "cat log.txt && make test"})])  # `make test` names it diagnostic
    return Turn(text="done")


async def test_at_arrival_a_big_test_log_becomes_a_receipt_and_the_full_output_is_archived(
    make_harness, tmp_path: Path
):
    harness = make_harness(runs_tests)
    (harness.cwd / "log.txt").write_text(PYTEST_LOG)
    await harness.run("fix the quotes")
    end = harness.recorder.of(ToolCallEnd)[0]
    assert end.content.startswith("[Test/build output reduced") and "1 failed, 119 passed" in end.content
    reduced = harness.recorder.of(Reduced)[0]
    assert reduced.applied and reduced.items >= 3 and reduced.receipt_chars < reduced.source_chars
    archive = next((tmp_path / "spill" / "evidence").iterdir())
    assert archive.read_text().startswith("=====") and "1 failed, 119 passed" in archive.read_text()
    # The model saw the receipt, not the log.
    seen = harness.provider.requests[1].messages[-1]
    assert "[Test/build output reduced" in getattr(seen, "result", seen).content if hasattr(seen, "result") else True


class Judge:
    """System 1 that marks one file as worked on and the rest as done with; records the files it was shown."""

    name = "judge"

    def __init__(self, working: str) -> None:
        self.working = working
        self.shown: list[str] = []

    async def decide(self, state: str, questions: dict) -> Answers:
        if "w" not in questions:  # another System 1 consumer (the supervisor and its wrap-up): no opinion
            neutral = {"mode": "advancing", "wrap": "working"}
            return Answers(
                answers={
                    k: ChoiceAnswer(choice=neutral[k]) if k in neutral else NoulAnswer(noul=0.0) for k in questions
                }
            )
        line = next(line for line in state.splitlines() if line.startswith("The file read about to be cut: "))
        path = line.removeprefix("The file read about to be cut: ")
        self.shown.append(path)
        return Answers(answers={"w": ChoiceAnswer(choice="working_on" if self.working in path else "done_with")})


def reads_then_churns(request: ProviderRequest) -> Turn:
    if is_state_write(request):
        return Turn(text="## Plan\n- edit card.py\n## Now\nchange _split in card.py")
    n = len(request.messages)
    if n < 3:
        return Turn(tool_calls=[("read", {"path": "card.py", "offset": 1, "limit": 30})])
    if n < 5:
        return Turn(tool_calls=[("bash", {"command": "cat util.py"})])
    return Turn(tool_calls=[("bash", {"command": f"printf '{'x' * 2500}'"})])


async def test_at_compaction_system_1_keeps_the_file_the_step_works_on_and_judges_it_again_next_time(make_harness):
    judge = Judge("card.py")
    harness = make_harness(reads_then_churns, window=12_000, profile=Profile(max_turns=30), system1=judge)
    (harness.cwd / "card.py").write_text('def _split(value):\n    return value.replace("\'\'", "\'")\n' * 20)
    (harness.cwd / "util.py").write_text("def _words_group(s):\n    return s.split()\n" * 20)
    await harness.run("fix the quotes")
    compactions = harness.recorder.of(Compaction)
    assert compactions and compactions[0].kept == 1
    assert judge.shown[:2] == ["card.py (lines 1-30)", "util.py"]  # a read slice and a bash cat, both judged
    assert all(e.ok for e in harness.recorder.of(Decision) if e.purpose == "keep")
    # After the first compaction the model sees card.py verbatim after the State, and not util.py.
    after = next(
        r
        for r in harness.provider.requests
        if not is_state_write(r) and any("kept verbatim" in m.content for m in r.messages if isinstance(m, UserMessage))
    )
    head = next(m.content for m in after.messages if isinstance(m, UserMessage) and "kept verbatim" in m.content)
    assert "card.py (lines 1-30):\ndef _split(value):" in head and "util.py" not in head
    # The kept read is judged again at the next compaction, among the new cut (from the client's cache when the
    # picture is the same, as it is here: the State is written at each compaction and the fake writes it alike).
    keeps = [d for d in harness.recorder.of(Decision) if d.purpose == "keep" and d.subject == "card.py (lines 1-30)"]
    assert len(keeps) >= 2 and len({d.turn for d in keeps}) >= 2
