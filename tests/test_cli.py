"""The headless CLI and the JSON-RPC server, as separate processes against a local fake model server."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from intern.config import load_user_config
from intern.events import event_adapter
from intern.journal import Journal

from .fake_openai import MODEL, FakeOpenAI, Reply


def env_for(server: FakeOpenAI, home: Path) -> dict[str, str]:
    return {**os.environ, "VLLM_BASE_URL": server.url, "INTERN_HOME": str(home), **server.system1_env}


def cli(*args: str, env: dict[str, str], cwd: Path, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "intern", *args],
        env=env,
        cwd=cwd,
        input=stdin,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def work(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


def test_headless_json_run_does_the_task_and_streams_typed_events(tmp_path: Path, work: Path):
    replies = [
        Reply(tool_calls=[("write", {"path": "hello.txt", "content": "hi\n"})]),
        Reply(text="Created hello.txt."),
    ]
    with FakeOpenAI(replies) as server:
        result = cli(
            "-p", "--json", "--no-session", "--no-context-files", "--no-extensions", "--no-memory",
            "--system2", f"vllm/{MODEL}", "--", "create hello.txt",
            env=env_for(server, tmp_path / "home"), cwd=work,
        )  # fmt: skip
    assert result.returncode == 0, result.stderr
    assert (work / "hello.txt").read_text() == "hi\n"
    events = [event_adapter.validate_json(line) for line in result.stdout.splitlines()]
    start, end = events[0], events[-1]
    assert start.type == "run_start" and end.type == "run_end"
    assert (start.window.working, start.window.advertised_source) == (40_960, "provider")
    assert end.reason == "completed" and end.final_text == "Created hello.txt."
    assert end.usage.input_tokens > 0
    roles = [e.message.role for e in events if e.type == "message"]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert not (tmp_path / "home" / "sessions").exists()


def test_text_mode_prints_the_answer_and_saves_a_resumable_session(tmp_path: Path, work: Path):
    home = tmp_path / "home"
    with FakeOpenAI([Reply(text="Four.")]) as server:
        result = cli("--system2", f"vllm/{MODEL}", "what is 2+2", env=env_for(server, home), cwd=work)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Four."
    sessions = list((home / "sessions").glob("*.jsonl"))
    assert len(sessions) == 1
    assert len(Journal(cwd=str(work), path=sessions[0]).entries()) >= 3


def test_an_extension_loaded_by_flag_brings_its_tools(tmp_path: Path, work: Path):
    (work / "a.txt").write_text("quiet\n")
    replies = [Reply(tool_calls=[("greet", {"who": "bob"}), ("read", {"path": "a.txt"})]), Reply(text="ok")]
    repo = Path(__file__).resolve().parent.parent  # so `tests.support` imports in the subprocess
    with FakeOpenAI(replies) as server:
        env = {**env_for(server, tmp_path / "home"), "PYTHONPATH": str(repo)}
        result = cli(
            "-p", "--json", "--no-session", "--no-context-files", "--extension", "tests.support:WithTools",
            "--system2", f"vllm/{MODEL}", "--", "greet bob, then read a.txt",
            env=env, cwd=work,
        )  # fmt: skip
    assert result.returncode == 0, result.stderr
    events = [event_adapter.validate_json(line) for line in result.stdout.splitlines()]
    tools = [e for e in events if e.type == "tool_call_end"]
    assert [(e.name, e.content) for e in tools] == [("greet", "hello bob"), ("read", "QUIET")]


def test_code_search_is_turned_on_by_its_capability_setting(tmp_path: Path, work: Path):
    pytest.importorskip("networkx", reason="needs the code-search extra")
    (work / "app.py").write_text("def greet(who):\n    return f'hello {who}'\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text("[capabilities.code_search]\nenabled = true\n")
    replies = [Reply(tool_calls=[("decision_code", {"question": "where is the greeting made"})]), Reply(text="ok")]
    with FakeOpenAI(replies) as server:
        result = cli(
            "-p", "--json", "--no-session", "--no-context-files", "--no-extensions",
            "--system2", f"vllm/{MODEL}", "--", "find the greeting",
            env=env_for(server, home), cwd=work,
        )  # fmt: skip
    assert result.returncode == 0, result.stderr
    events = [event_adapter.validate_json(line) for line in result.stdout.splitlines()]
    found = next(e for e in events if e.type == "tool_call_end")
    assert found.status == "ok", (found.content, result.stderr)
    assert (work / ".intern").exists()
    assert "decision_code" in [t["function"]["name"] for t in server.requests[0]["tools"]]


@pytest.mark.parametrize("how", ["--extension code-search", "--extension module:Class", "profile"])
def test_code_search_cannot_be_turned_on_as_an_extension(tmp_path: Path, work: Path, how: str):
    (tmp_path / "p.toml").write_text('extensions = ["code-search"]\n')
    args = {
        "--extension code-search": ["--extension", "code-search"],
        "--extension module:Class": ["--extension", "intern.extensions.code_search:CodeSearch"],
        "profile": ["--profile", str(tmp_path / "p.toml")],
    }[how]
    with FakeOpenAI([Reply(text="ok")]) as server:
        result = cli(
            "-p", "--no-session", *args, "--system2", f"vllm/{MODEL}", "--", "task",
            env=env_for(server, tmp_path / "home"), cwd=work,
        )  # fmt: skip
        assert not server.requests
    assert result.returncode == 2
    assert "set [capabilities.code_search] enabled = true" in result.stderr


def test_the_workspace_is_turned_on_by_its_capability_setting_beside_code_search(tmp_path: Path, work: Path):
    pytest.importorskip("duckdb", reason="needs the workspace extra")
    (work / "notes.md").write_text("# Notes\n\nThe total due is 1,250 EUR.\n")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text(
        "[capabilities.code_search]\nenabled = true\n[capabilities.workspace]\nenabled = true\n"
    )
    replies = [Reply(tool_calls=[("decision_files", {"question": "the total due"})]), Reply(text="ok")]
    with FakeOpenAI(replies) as server:
        result = cli(
            "-p", "--json", "--no-session", "--no-context-files", "--no-extensions",
            "--system2", f"vllm/{MODEL}", "--", "find the total",
            env=env_for(server, home), cwd=work,
        )  # fmt: skip
    assert result.returncode == 0, result.stderr
    names = [t["function"]["name"] for t in server.requests[0]["tools"]]
    assert {"decision_code", "index_files", "decision_files", "sql", "read"} <= set(names) and names.count("read") == 1
    events = [event_adapter.validate_json(line) for line in result.stdout.splitlines()]
    found = next(e for e in events if e.type == "tool_call_end")
    assert found.status == "error" and "no index under ." in found.content, found.content  # the agent builds it


@pytest.mark.parametrize("how", ["--extension workspace", "--extension module:Class", "profile"])
def test_the_workspace_cannot_be_turned_on_as_an_extension(tmp_path: Path, work: Path, how: str):
    (tmp_path / "p.toml").write_text('extensions = ["workspace"]\n')
    args = {
        "--extension workspace": ["--extension", "workspace"],
        "--extension module:Class": ["--extension", "intern.extensions.workspace.extension:Workspace"],
        "profile": ["--profile", str(tmp_path / "p.toml")],
    }[how]
    with FakeOpenAI([Reply(text="ok")]) as server:
        result = cli(
            "-p", "--no-session", *args, "--system2", f"vllm/{MODEL}", "--", "task",
            env=env_for(server, tmp_path / "home"), cwd=work,
        )  # fmt: skip
        assert not server.requests
    assert result.returncode == 2
    assert "set [capabilities.workspace] enabled = true" in result.stderr


def test_prompt_can_come_from_stdin(tmp_path: Path, work: Path):
    with FakeOpenAI([Reply(text="ok")]) as server:
        result = cli(
            "--no-session", "--model", f"vllm/{MODEL}",
            env=env_for(server, tmp_path / "h"), cwd=work, stdin="hi",
        )  # fmt: skip
        assert server.requests[0]["messages"][-1]["content"] == "hi"
    assert result.returncode == 0


def test_configuration_errors_exit_with_a_usage_code(tmp_path: Path, work: Path):
    env = {**os.environ, "INTERN_HOME": str(tmp_path / "h")}
    env.pop("INTERN_SYSTEM2", None)
    result = cli("-p", "--", "task", env=env, cwd=work)
    assert result.returncode == 2
    assert "no model given" in result.stderr


def test_a_run_without_system_1_is_refused_not_run_on_heuristics(tmp_path: Path, work: Path):
    """System 1 is required: without the console key or a configured backend the CLI says what to set and exits 2
    before any model call; the old `heuristic` opt-out is refused the same way."""
    home = tmp_path / "home"
    with FakeOpenAI([Reply(text="Four."), Reply(text="Four.")]) as server:
        keys = ("NACE_API_KEY", "DREX_API_KEY", "DREX_BASE_URL", "OPENROUTER_API_KEY")
        env = {k: v for k, v in env_for(server, home).items() if k not in keys}
        run = ("-p", "--no-session", "--no-memory", "--system2", f"vllm/{MODEL}", "--", "2+2")
        refused = cli(*run, env=env, cwd=work)
        heuristic = cli("--system1", "heuristic", *run, env=env, cwd=work)
        assert server.requests == []
        # With the console's key and address the default, drex-latest, is set up and the run goes.
        keyed = cli(*run, env={**env, **server.system1_env}, cwd=work)
    assert refused.returncode == 2, refused.stderr
    assert "drex-latest" in refused.stderr and "NACE_API_KEY" in refused.stderr and "--system1" in refused.stderr
    assert heuristic.returncode == 2 and "System 1 is required" in heuristic.stderr, heuristic.stderr
    assert keyed.returncode == 0 and keyed.stdout.strip() == "Four.", keyed.stderr


@pytest.mark.parametrize(
    "toml",
    [
        'system2 = "vllm/a"\n[system1]\nbackend = "decisions-api"\nmodel = "decider"\n',
        'model = "vllm/a"\n[decider]\nbackend = "decisions-api"\nmodel = "decider"\n',
    ],
)
def test_config_reads_system2_and_system1_under_new_and_old_keys(tmp_path: Path, toml: str):
    (tmp_path / "config.toml").write_text(toml)
    config = load_user_config(tmp_path)
    assert (config.system2, config.system1.backend, config.system1.model) == ("vllm/a", "decisions-api", "decider")


def test_schema_command_prints_the_event_schema(tmp_path: Path, work: Path):
    result = cli("schema", env={**os.environ, "INTERN_HOME": str(tmp_path / "h")}, cwd=work)
    schema = json.loads(result.stdout)
    assert "run_end" in json.dumps(schema)
    defs = schema["$defs"]
    # The fields that let a listener attribute System 1's extra System 2 work are all optional: old streams parse.
    # A compaction's own State write is a `state` event, so every write's usage is on one event type.
    for event, fields in {
        "State": ["dropped", "usage", "duration_ms", "model"],
        "Decision": ["turn", "subject", "answers"],
        "RunEnd": ["agent_usage", "state_usage", "system1"],
    }.items():
        for field in fields:
            assert field in defs[event]["properties"] and field not in defs[event].get("required", []), (event, field)


def test_rpc_server_runs_prompts_and_answers_commands(tmp_path: Path, work: Path):
    (work / "notes.txt").write_text("a needle here\n")
    replies = [
        Reply(tool_calls=[("grep", {"pattern": "needle"})]),
        Reply(tool_calls=[("bash", {"command": "echo rpc-ok"})]),
        Reply(text="ran it"),
    ]
    with FakeOpenAI(replies) as server:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "intern",
                "rpc",
                "--no-session",
                "--system2",
                f"vllm/{MODEL}",
            ],
            env=env_for(server, tmp_path / "home"),
            cwd=work,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        stdin, stdout = process.stdin, process.stdout
        assert stdin is not None and stdout is not None

        def send(payload: dict[str, Any]) -> None:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()

        def read_until(predicate) -> list[dict[str, Any]]:
            lines = []
            while True:
                line = json.loads(stdout.readline())
                lines.append(line)
                if predicate(line):
                    return lines

        send({"jsonrpc": "2.0", "id": 1, "method": "get_state"})
        state = read_until(lambda m: m.get("id") == 1)[-1]["result"]
        assert state["running"] is False and state["window"]["working"] == 40_960

        send({"jsonrpc": "2.0", "id": 2, "method": "prompt", "params": {"text": "run echo"}})
        lines = read_until(lambda m: m.get("method") == "event" and m["params"]["type"] == "run_end")
        assert {"jsonrpc": "2.0", "id": 2, "result": {"started": True}} in lines
        tool_ends = [m["params"] for m in lines if m.get("params", {}).get("type") == "tool_call_end"]
        # stdin is an open pipe here, which ripgrep would read instead of the working directory.
        assert tool_ends[0]["content"] == "notes.txt:1:a needle here"
        assert "rpc-ok" in tool_ends[1]["content"]

        send({"jsonrpc": "2.0", "id": 3, "method": "steer", "params": {"text": "late"}})
        assert read_until(lambda m: m.get("id") == 3)[-1]["error"]["code"] == -32001

        send({"jsonrpc": "2.0", "id": 4, "method": "nope"})
        assert read_until(lambda m: m.get("id") == 4)[-1]["error"]["code"] == -32601

        send({"jsonrpc": "2.0", "id": 5, "method": "get_entries"})
        entries = read_until(lambda m: m.get("id") == 5)[-1]["result"]["entries"]
        assert [e["kind"] for e in entries][:3] == ["session", "system_prompt", "message"]

        stdin.close()
        assert process.wait(timeout=20) == 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(os.name != "posix", reason="POSIX signals")
@pytest.mark.parametrize("mode", ["-p", "rpc"])
def test_sigterm_ends_the_run_and_its_command_with_it(tmp_path: Path, work: Path, mode: str):
    # The command runs in a process group of its own; killing the harness used to leave it and its child running,
    # with no run_end.
    long_call = "echo $$ > shell.pid; sleep 300 & echo $! > child.pid; wait"
    replies = [Reply(tool_calls=[("bash", {"command": long_call, "timeout_s": 600})]), Reply(text="done")]
    with FakeOpenAI(replies) as server:
        common = ["--no-session", "--system2", f"vllm/{MODEL}"]
        argv = ["-p", "--json", *common, "--", "go"] if mode == "-p" else ["rpc", *common]
        process = subprocess.Popen(
            [sys.executable, "-m", "intern", *argv],
            env=env_for(server, tmp_path / "home"),
            cwd=work,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdin is not None
        if mode == "rpc":
            process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "prompt", "params": {"text": "go"}}))
            process.stdin.write("\n")
            process.stdin.flush()
        pids = work / "child.pid"
        deadline = time.monotonic() + 30
        while not (pids.exists() and pids.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.05)
        shell, child = int((work / "shell.pid").read_text()), int(pids.read_text())
        try:
            process.send_signal(signal.SIGTERM)
            out, _ = process.communicate(timeout=30)
            deadline = time.monotonic() + 5
            while (_alive(shell) or _alive(child)) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not _alive(shell) and not _alive(child)
            assert process.returncode == 128 + signal.SIGTERM
            lines = [json.loads(line) for line in out.splitlines() if '"run_end"' in line]
            assert [line.get("params", line)["reason"] for line in lines] == ["aborted"]
        finally:
            for pid in (shell, child):
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)
