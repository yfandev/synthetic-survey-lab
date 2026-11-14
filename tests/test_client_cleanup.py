"""CLI/RPC teardown through real session assembly and owned local processes."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

import intern.cli as cli
from intern.config import Bootstrap, Options
from intern.events import Event, EventSink, RunEnd, ToolCallEnd
from intern.protocol.rpc import serve
from intern.session import Session
from intern.tools.process import ProcessTool

from .fake_openai import MODEL, FakeOpenAI, Reply


@pytest.fixture
def client_paths(tmp_path: Path) -> tuple[Path, Path]:
    home, work = tmp_path / "home", tmp_path / "work"
    home.mkdir()
    work.mkdir()
    (home / "config.toml").write_text('[capabilities]\ntool_packs = ["process"]\n')
    return home, work


def _options(work: Path) -> Options:
    return Options(cwd=work, system2=f"vllm/{MODEL}", persist=False, context_files=False, extensions=False)


def _replies() -> list[Reply]:
    command = shlex.join([sys.executable, "-c", "import time; time.sleep(60)"])
    return [Reply(tool_calls=[("process", {"action": "start", "command": command})]), Reply(text="Task finished.")]


def _process_tool(session: Session) -> ProcessTool:
    return next(tool for tool in session.deps.tools if isinstance(tool, ProcessTool))


async def _assert_closed_processes(session: Session) -> None:
    assert session.closed and not session.running
    tool = _process_tool(session)
    jobs = tool.manager.list()
    assert len(jobs) == 1
    assert jobs[0].state != "running" and jobs[0].exit_code is not None
    with pytest.raises(RuntimeError, match="closed"):
        await tool.manager.start("echo must-not-start", cwd=session.deps.cwd)


@pytest.mark.parametrize("disconnect", [False, True], ids=["success", "output-error"])
async def test_cli_closes_real_bootstrap_process_on_success_or_output_error(
    client_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, disconnect: bool
):
    home, work = client_paths
    sessions: list[Session] = []
    events: list[Event] = []
    output_failed = False

    class CapturingBootstrap(Bootstrap):
        async def session(self, emit: EventSink) -> Session:
            session = await super().session(emit)
            sessions.append(session)
            return session

    def sink(event: Event) -> None:
        nonlocal output_failed
        events.append(event)
        if disconnect and isinstance(event, ToolCallEnd) and event.name == "process":
            assert event.status == "ok" and json.loads(event.content)["state"] == "running"
            output_failed = True
        if output_failed:
            raise BrokenPipeError("client output disconnected")

    monkeypatch.setattr(cli, "Bootstrap", CapturingBootstrap)
    monkeypatch.setattr(cli, "_json_sink", sink)
    monkeypatch.setenv("INTERN_HOME", str(home))
    with FakeOpenAI(_replies()) as server:
        monkeypatch.setenv("VLLM_BASE_URL", server.url)
        for name, value in server.system1_env.items():
            monkeypatch.setenv(name, value)
        try:
            if disconnect:
                with pytest.raises(BrokenPipeError, match="client output disconnected"):
                    await cli._run_prompt(_options(work), "start a background task", as_json=True)
            else:
                assert await cli._run_prompt(_options(work), "start a background task", as_json=True) == cli.EXIT_OK
                assert any(isinstance(event, RunEnd) and event.reason == "completed" for event in events)
            assert len(sessions) == 1
            await _assert_closed_processes(sessions[0])
            assert any(isinstance(event, ToolCallEnd) and event.name == "process" for event in events)
            assert server.requests
        finally:
            # A cleanup regression must fail without leaving the test's child alive.
            for session in sessions:
                await session.aclose()


async def test_rpc_eof_drains_real_run_before_closing_owned_processes(client_paths: tuple[Path, Path], capsys):
    home, work = client_paths
    events: list[Event] = []
    sessions: list[Session] = []
    request = {"jsonrpc": "2.0", "id": 1, "method": "prompt", "params": {"text": "start a background task"}}
    stream = io.StringIO(json.dumps(request) + "\n")
    with FakeOpenAI(_replies()) as server:
        bootstrap = Bootstrap(
            _options(work), {"INTERN_HOME": str(home), "VLLM_BASE_URL": server.url, **server.system1_env}
        )

        async def make_session(emit: EventSink) -> Session:
            def sink(event: Event) -> None:
                events.append(event)
                emit(event)

            session = await bootstrap.session(sink)
            sessions.append(session)
            return session

        try:
            await asyncio.wait_for(serve(make_session, stream), 10)
            assert len(sessions) == 1
            outcome = sessions[0].last_outcome
            assert outcome is not None and outcome.reason == "completed" and outcome.final_text == "Task finished."
            assert any(isinstance(event, ToolCallEnd) and event.name == "process" for event in events)
            await _assert_closed_processes(sessions[0])
            assert not stream.closed
            replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
            assert {"jsonrpc": "2.0", "id": 1, "result": {"started": True}} in replies
            assert any(message.get("params", {}).get("type") == "run_end" for message in replies)
        finally:
            for session in sessions:
                await session.aclose()


@pytest.mark.skipif(os.name != "posix", reason="POSIX cancellable RPC stdin")
def test_rpc_cancel_closes_session_and_process_exits_with_stdin_still_open(client_paths: tuple[Path, Path]):
    home, work = client_paths
    script = """
import asyncio
import json
from pathlib import Path
from intern.config import Bootstrap, Options
from intern.events import ToolCallEnd
from intern.protocol.rpc import serve
from intern.tools.process import ProcessTool

async def main():
    started = asyncio.Event()
    sessions = []
    bootstrap = Bootstrap(Options(
        cwd=Path.cwd(), system2='vllm/test-model', persist=False, context_files=False,
        extensions=False,
    ))

    async def make_session(emit):
        def sink(event):
            emit(event)
            if isinstance(event, ToolCallEnd) and event.name == 'process':
                assert event.status == 'ok' and json.loads(event.content)['state'] == 'running'
                started.set()
        session = await bootstrap.session(sink)
        sessions.append(session)
        return session

    task = asyncio.create_task(serve(make_session))
    try:
        await asyncio.wait_for(started.wait(), 10)
        assert sessions[0].running, 'cancellation must exercise an active run'
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError('serve swallowed cancellation')
        session = sessions[0]
        assert session.closed and not session.running
        tool = next(tool for tool in session.deps.tools if isinstance(tool, ProcessTool))
        jobs = tool.manager.list()
        assert len(jobs) == 1 and jobs[0].state != 'running' and jobs[0].exit_code is not None
        print('RPC-CLEANUP-VERIFIED', flush=True)
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for session in sessions:
            await session.aclose()

asyncio.run(main())
print('RPC-PROCESS-EXITED', flush=True)
"""
    request = {"jsonrpc": "2.0", "id": 1, "method": "prompt", "params": {"text": "start a background task"}}
    with FakeOpenAI(_replies()) as server:
        env = {**os.environ, "INTERN_HOME": str(home), "VLLM_BASE_URL": server.url, **server.system1_env}
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=work,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            code = process.wait(timeout=20)
            output, errors = process.stdout.read(), process.stderr.read()
            assert code == 0, errors
            assert not process.stdin.closed, "the parent must keep its pipe open until the child exits"
            assert "RPC-CLEANUP-VERIFIED\nRPC-PROCESS-EXITED\n" in output
            assert server.requests
        finally:
            if process.stdin is not None:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()
