"""The base tool pack against a real temporary directory."""

from __future__ import annotations

import asyncio
import hashlib
import shlex
import sys
from pathlib import Path

import pytest

import intern.executor as executor_module
from intern.executor import LocalExecutor
from intern.tools import search
from intern.tools.base import MAX_DIFF_CHARS, ToolContext, ToolOutput
from intern.tools.files import EditTool, ReadTool, WriteTool
from intern.tools.search import FindTool, GrepTool
from intern.tools.shell import BashTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, executor=LocalExecutor(), abort=asyncio.Event(), max_output_chars=10_000)


async def call(tool, ctx: ToolContext, **arguments) -> ToolOutput:
    parsed = tool.parse(arguments)
    if isinstance(parsed, ToolOutput):
        return parsed
    return await tool.run(parsed, ctx)


async def test_read_pages_long_files_and_says_where_to_continue(ctx: ToolContext):
    (ctx.cwd / "long.txt").write_text("\n".join(f"line {i}" for i in range(1, 5001)))
    ctx.max_output_chars = 1_000
    first = await call(ReadTool(), ctx, path="long.txt")
    assert first.content.startswith("line 1\n")
    assert "Continue with offset=" in first.content
    offset = int(first.content.rsplit("offset=", 1)[1].rstrip(".]"))
    second = await call(ReadTool(), ctx, path="long.txt", offset=offset, limit=2)
    assert second.content.splitlines()[:2] == [f"line {offset}", f"line {offset + 1}"]


async def test_read_lists_directories_and_rejects_binary_files(ctx: ToolContext):
    (ctx.cwd / "sub").mkdir()
    (ctx.cwd / "a.bin").write_bytes(b"\x00\x01\x02")
    listing = await call(ReadTool(), ctx, path=".")
    assert listing.content.splitlines() == ["sub/", "a.bin"]
    binary = await call(ReadTool(), ctx, path="a.bin")
    assert binary.status == "error" and "binary" in binary.content


async def test_missing_files_fail_with_a_next_action(ctx: ToolContext):
    result = await call(ReadTool(), ctx, path="nope.txt")
    assert result.status == "error"
    assert result.error is not None and result.error.argument == "path"
    assert "Next:" in result.content


async def test_write_then_edit(ctx: ToolContext):
    await call(WriteTool(), ctx, path="pkg/mod.py", content="def f():\n    return 1\n")
    edited = await call(EditTool(), ctx, path="pkg/mod.py", old_string="return 1", new_string="return 2")
    assert edited.status == "ok" and "line 2" in edited.content
    assert (ctx.cwd / "pkg/mod.py").read_text() == "def f():\n    return 2\n"


async def test_write_and_edit_report_the_change_as_a_diff(ctx: ToolContext):
    created = await call(WriteTool(), ctx, path="a.txt", content="one\n")
    target = str((ctx.cwd / "a.txt").resolve())
    assert created.diff is not None and (created.diff.path, created.diff.old_text) == (target, None)
    assert created.diff.new_text == "one\n"
    overwritten = await call(WriteTool(), ctx, path="a.txt", content="two\n")
    assert overwritten.diff is not None and (overwritten.diff.old_text, overwritten.diff.new_text) == ("one\n", "two\n")
    edited = await call(EditTool(), ctx, path="a.txt", old_string="two", new_string="three")
    assert edited.diff is not None and (edited.diff.old_text, edited.diff.new_text) == ("two\n", "three\n")
    failed = await call(EditTool(), ctx, path="a.txt", old_string="absent", new_string="x")
    assert failed.status == "error" and failed.diff is None


async def test_diffs_are_left_out_for_binary_and_huge_files(ctx: ToolContext):
    (ctx.cwd / "blob.bin").write_bytes(b"\xff\xfe old")
    assert (await call(WriteTool(), ctx, path="blob.bin", content="text")).diff is None
    (ctx.cwd / "big.txt").write_text("x" * (MAX_DIFF_CHARS + 1))
    assert (await call(WriteTool(), ctx, path="big.txt", content="small")).diff is None
    assert (await call(EditTool(), ctx, path="big.txt", old_string="small", new_string="tiny")).diff is not None


async def test_edit_refuses_ambiguous_matches_unless_asked(ctx: ToolContext):
    (ctx.cwd / "a.txt").write_text("x = 1\nx = 1\n")
    ambiguous = await call(EditTool(), ctx, path="a.txt", old_string="x = 1", new_string="x = 2")
    assert ambiguous.status == "error" and "lines 1, 2" in ambiguous.content
    every = await call(EditTool(), ctx, path="a.txt", old_string="x = 1", new_string="x = 2", replace_all=True)
    assert every.status == "ok"
    assert (ctx.cwd / "a.txt").read_text() == "x = 2\nx = 2\n"


async def test_edit_points_at_a_whitespace_near_miss(ctx: ToolContext):
    (ctx.cwd / "a.py").write_text("def f():\n    if x:\n        return 1\n")
    result = await call(EditTool(), ctx, path="a.py", old_string="if x:\n    return 1", new_string="pass")
    assert result.status == "error"
    assert "lines 2-3 match if whitespace is ignored" in result.content


async def test_bash_reports_exit_codes_and_timeouts(ctx: ToolContext):
    ok = await call(BashTool(), ctx, command="echo hello")
    assert ok.status == "ok" and ok.content == "hello\n[exit code 0]"
    failed = await call(BashTool(), ctx, command="echo oops >&2; exit 3")
    assert failed.status == "error" and "oops" in failed.content and "code 3" in failed.content
    assert failed.error is not None and not failed.error.retry_safe
    slow = await call(BashTool(), ctx, command="sleep 5", timeout_s=1)
    assert slow.status == "error" and "timed out" in slow.content
    assert slow.error is not None and not slow.error.retry_safe


async def test_bash_stops_a_command_past_its_memory_budget_and_the_next_one_runs(tmp_path: Path):
    budget = LocalExecutor(memory_limit=200 * 2**20)
    ctx = ToolContext(cwd=tmp_path, executor=budget, abort=asyncio.Event(), max_output_chars=10_000)
    hog = f"{shlex.quote(sys.executable)} -c \"import time; x = b'1' * (1500 * 2**20); time.sleep(30)\""
    started = asyncio.get_running_loop().time()
    stopped = await call(BashTool(), ctx, command=hog, timeout_s=60)
    assert stopped.status == "error" and "was stopped at" in stopped.content and "of memory" in stopped.content
    assert "over the 0.2 GB a command may hold" in stopped.content and "Read less at once" in stopped.content
    assert asyncio.get_running_loop().time() - started < 20  # stopped by memory, not by the 60 s timeout
    small = await call(BashTool(), ctx, command=f"{shlex.quote(sys.executable)} -c \"print(len(b'1' * 2**20))\"")
    assert small.status == "ok" and "1048576" in small.content


async def test_a_process_a_command_leaves_in_the_background_is_watched_too(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(executor_module, "BACKGROUND_POLL_S", 0.1)
    ctx = ToolContext(
        cwd=tmp_path, executor=LocalExecutor(memory_limit=200 * 2**20), abort=asyncio.Event(), max_output_chars=10_000
    )
    hog = f"{shlex.quote(sys.executable)} -c \"import time; x = b'1' * (1500 * 2**20); time.sleep(30)\""
    quiet = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(1)" > /dev/null 2>&1 &'
    assert (await call(BashTool(), ctx, command=f"{hog} > /dev/null 2>&1 &")).status == "ok"  # returns at once
    assert (await call(BashTool(), ctx, command=quiet)).status == "ok"  # stays under the budget, then exits
    for _ in range(100):
        await asyncio.sleep(0.1)
        if ctx.executor._notices:  # pyright: ignore[reportAttributeAccessIssue]
            break
    told = await call(BashTool(), ctx, command="echo next")
    assert told.content.startswith("[A process the command `") and "left running in the background was stopped" in (
        told.content
    )
    assert told.content.count("left running in the background") == 1  # the quiet one is not reported
    assert "left running" not in (await call(BashTool(), ctx, command="echo again")).content  # said once


async def test_memory_that_cannot_be_read_stops_the_watch_not_the_command(tmp_path: Path, monkeypatch):
    real_exec = asyncio.create_subprocess_exec

    async def no_ps(program, *args, **kwargs):  # as in a sandbox that may not run ps
        if program == "ps":
            raise PermissionError(1, "Operation not permitted", "ps")
        return await real_exec(program, *args, **kwargs)

    monkeypatch.setattr(executor_module.asyncio, "create_subprocess_exec", no_ps)
    monkeypatch.setattr(executor_module, "BACKGROUND_POLL_S", 0.1)
    ctx = ToolContext(
        cwd=tmp_path, executor=LocalExecutor(memory_limit=200 * 2**20), abort=asyncio.Event(), max_output_chars=10_000
    )
    slow = await call(BashTool(), ctx, command="sleep 1; echo done")
    assert slow.status == "ok" and "done" in slow.content
    assert (await call(BashTool(), ctx, command="sleep 0.5 > /dev/null 2>&1 &")).status == "ok"
    await asyncio.sleep(0.5)
    assert "left running" not in (await call(BashTool(), ctx, command="echo next")).content


def test_the_config_sets_a_commands_memory_share(tmp_path: Path):
    from intern.config import UserConfig, _command_memory
    from intern.executor import physical_memory

    config = UserConfig.model_validate({"commands": {"memory_share": 0.25}})
    memory = physical_memory()
    assert memory is not None and _command_memory(config.commands.memory_share) == int(memory * 0.25)
    assert UserConfig().commands.memory_share == 0.5


async def test_interrupted_shell_effect_is_not_advertised_as_safe_to_repeat(ctx):
    result = await call(BashTool(), ctx, command="echo effect >> effects; sleep 30", timeout_s=1)
    assert (ctx.cwd / "effects").read_text() == "effect\n"
    assert result.error is not None and not result.error.retry_safe
    invalid = BashTool().parse({"timeout_s": 1})
    assert isinstance(invalid, ToolOutput) and invalid.error is not None and invalid.error.retry_safe


@pytest.mark.parametrize("limited", [False, True])
async def test_bash_exposes_a_readable_capture_for_the_missing_middle(ctx, monkeypatch, limited):
    if limited:
        monkeypatch.setattr(executor_module, "MAX_CAPTURE_BYTES", 1000)
    script = 'import sys; sys.stdout.write("A"*800000+"\\nMIDDLE_EVIDENCE\\n"+"B"*800000)'
    result = await call(BashTool(), ctx, command=f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}")
    assert result.status == "ok" and result.omitted_bytes > 0
    assert "MIDDLE_EVIDENCE" not in result.content
    capture = next((ctx.cwd / ".intern" / "command-output").glob("*.txt"))
    data = capture.read_bytes()
    assert result.resources[0].sha256 == hashlib.sha256(data).hexdigest()
    if limited:
        assert "Partial command output" in result.content and "Complete command output" not in result.content
        assert result.capture_omitted_bytes > 0 and len(data) == 1000
    else:
        assert "Complete command output" in result.content
        read = await call(ReadTool(), ctx, path=str(capture.relative_to(ctx.cwd)), offset=2, limit=1)
        assert "MIDDLE_EVIDENCE" in read.content


async def test_small_command_output_does_not_leave_an_archive(ctx):
    assert (await call(BashTool(), ctx, command="printf small")).status == "ok"
    assert not list((ctx.cwd / ".intern" / "command-output").glob("*.txt"))


async def test_legacy_custom_executor_keeps_its_original_call_signature(ctx):
    class Legacy:
        async def run(self, command, *, cwd, timeout_s, max_output_bytes, abort=None, env=None):
            return await LocalExecutor().run(
                command,
                cwd=cwd,
                timeout_s=timeout_s,
                max_output_bytes=100,
                abort=abort,
                env=env,
            )

    ctx.executor = Legacy()
    result = await call(BashTool(), ctx, command="seq 1 10000")
    assert result.status == "ok" and result.omitted_bytes > 0
    assert "does not support full output capture" in result.content and not result.resources


@pytest.mark.parametrize("ripgrep", [True, False])
async def test_grep_and_find_with_and_without_ripgrep(ctx: ToolContext, monkeypatch, ripgrep: bool):
    if not ripgrep:
        monkeypatch.setattr(search, "_ripgrep", lambda: None)
    elif search._ripgrep() is None:
        pytest.skip("ripgrep is not installed")
    (ctx.cwd / "src").mkdir()
    (ctx.cwd / "src" / "app.py").write_text("import os\n\ndef main():\n    return os.getcwd()\n")
    (ctx.cwd / "README.md").write_text("main entry point\n")
    (ctx.cwd / ".decision_workspace").mkdir()  # the workspace index, with a copy of every text: never searched
    (ctx.cwd / ".decision_workspace" / "copy.py").write_text("def main():\n")

    found = await call(GrepTool(), ctx, pattern=r"def \w+", path=".")
    assert found.content == "src/app.py:3:def main():"
    only_md = await call(GrepTool(), ctx, pattern="main", glob="*.md")
    assert only_md.content == "README.md:1:main entry point"
    none = await call(GrepTool(), ctx, pattern="nothing_here")
    assert none.content.startswith("No matches")
    files = await call(FindTool(), ctx, pattern="*.py")
    assert files.content == "src/app.py"


async def test_grep_rejects_bad_patterns(ctx: ToolContext):
    result = await call(GrepTool(), ctx, pattern="(unclosed")
    assert result.status == "error" and result.error is not None and result.error.argument == "pattern"


async def test_read_pages_fit_the_budget_so_no_line_is_skipped(ctx: ToolContext):
    (ctx.cwd / "short.txt").write_text("\n".join(f"{i:09d}" for i in range(1, 20_001)) + "\n")
    ctx.max_output_chars = 2_000
    offset, seen = 1, []
    while offset:
        page = await call(ReadTool(), ctx, path="short.txt", offset=offset)
        assert len(page.content) <= ctx.max_output_chars
        body, _, footer = page.content.partition("\n[Lines")
        seen += [int(line) for line in body.splitlines()]
        offset = int(footer.rsplit("offset=", 1)[1].rstrip(".]")) if footer else 0
        if len(seen) > 600:
            break
    assert seen == list(range(1, len(seen) + 1))


async def test_edit_keeps_crlf_line_endings_and_undecodable_bytes(ctx: ToolContext):
    path = ctx.cwd / "win.txt"
    path.write_bytes(b"one\r\ncaf\xe9\r\nthree\r\n")
    result = await call(EditTool(), ctx, path="win.txt", old_string="one\ncaf", new_string="1\ncaf")
    assert result.status == "ok", result.content
    assert path.read_bytes() == b"1\r\ncaf\xe9\r\nthree\r\n"


async def test_bash_returns_promptly_when_the_command_leaves_a_background_process(ctx: ToolContext):
    started = asyncio.get_running_loop().time()
    result = await call(BashTool(), ctx, command="sleep 4 & echo started", timeout_s=30)
    assert result.status == "ok" and "started" in result.content
    assert asyncio.get_running_loop().time() - started < 3


def test_tool_schemas_carry_no_titles():
    spec = BashTool().spec()
    assert "title" not in str(spec.parameters)
    assert spec.parameters["required"] == ["command"]
