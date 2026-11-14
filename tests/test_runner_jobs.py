"""The process tool where the session's commands run: jobs an executor starts, with the local jobs' rules."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from intern.executor import ExecResult
from intern.processes import ProcessManager
from intern.runners.docker import DockerExecutor, DockerOptions
from intern.tools.base import ToolContext
from intern.tools.process import ProcessArgs, ProcessTool

from .fake_openai import Reply
from .test_session_setup import configured, outputs


class FakeJob:
    def __init__(self, *, takes_input: bool = True) -> None:
        self.output: asyncio.Queue[bytes] = asyncio.Queue()
        self.exit: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        self.written: list[tuple[bytes, bool]] = []
        self.kills = 0
        self.takes_input = takes_input

    async def read(self) -> bytes:
        return await self.output.get()

    async def write(self, data: bytes, *, eof: bool = False) -> None:
        if not self.takes_input:
            raise NotImplementedError
        self.written.append((data, eof))
        self.output.put_nowait(b"got:" + data)

    async def wait(self) -> int | None:
        return await asyncio.shield(self.exit)

    async def kill(self) -> None:
        self.kills += 1
        if not self.exit.done():
            self.exit.set_result(137)
        self.output.put_nowait(b"")

    def end(self, code: int) -> None:
        self.exit.set_result(code)
        self.output.put_nowait(b"")


class FakeJobExecutor:
    """Runs no command itself: what `start_job` was asked, and the jobs it handed out."""

    def __init__(self, **job: bool) -> None:
        self.started: list[tuple[str, Path]] = []
        self.jobs: list[FakeJob] = []
        self.job = job

    async def run(self, command: str, **kwargs: object) -> ExecResult:
        raise AssertionError("jobs never go through run")

    async def start_job(self, command: str, *, cwd: Path, env: Mapping[str, str] | None = None) -> FakeJob:
        self.started.append((command, cwd))
        self.jobs.append(FakeJob(**self.job))
        self.jobs[-1].output.put_nowait(b"started\n")
        return self.jobs[-1]


async def test_an_executor_job_is_polled_fed_and_ended_like_a_local_one(tmp_path):
    executor = FakeJobExecutor()
    async with ProcessManager() as jobs:
        started = await jobs.start("serve", cwd=tmp_path, executor=executor, stdin=True)
        assert executor.started == [("serve", tmp_path)] and started.state == "running"
        await jobs.write(started.handle, "hello\n")
        polled = await jobs.poll(started.handle, wait_s=0.1)
        assert polled.output == "started\ngot:hello\n" and polled.state == "running"
        executor.jobs[0].end(3)
        ended = await jobs.poll(started.handle, cursor=polled.next_cursor, wait_s=2)
        assert (ended.state, ended.exit_code, ended.output) == ("exited", 3, "")
        with pytest.raises(ValueError, match="no longer accepts input"):
            await jobs.write(started.handle, "late")
    assert executor.jobs[0].kills >= 1  # stopped once its record completes, as a local group is


@pytest.mark.parametrize("ending", ["terminate", "timeout", "abort", "close"])
async def test_every_way_a_job_ends_stops_it_on_its_executor(tmp_path, ending):
    executor, abort = FakeJobExecutor(), asyncio.Event()
    jobs = ProcessManager()
    started = await jobs.start("sleep", cwd=tmp_path, executor=executor, timeout_s=0.2, abort=abort)
    if ending == "terminate":
        snapshot = await jobs.terminate(started.handle)
    elif ending == "abort":
        abort.set()
        snapshot = await jobs.poll(started.handle, wait_s=2)
    elif ending == "timeout":
        snapshot = await jobs.poll(started.handle, wait_s=2)
    else:
        await jobs.aclose()
        snapshot = await jobs.poll(started.handle)
    expected = {"terminate": "terminated", "timeout": "timed_out", "abort": "aborted", "close": "terminated"}
    assert snapshot.state == expected[ending] and executor.jobs[0].kills >= 1
    await jobs.aclose()


async def test_a_job_that_takes_no_input_says_so(tmp_path):
    async with ProcessManager() as jobs:
        started = await jobs.start("run", cwd=tmp_path, executor=FakeJobExecutor(takes_input=False), stdin=True)
        with pytest.raises(ValueError, match="take no input"):
            await jobs.write(started.handle, "x")


async def test_the_tool_uses_the_sessions_executor_and_never_escapes_to_the_host(tmp_path):
    def context(executor) -> ToolContext:
        return ToolContext(cwd=tmp_path, executor=executor, abort=asyncio.Event(), max_output_chars=4000)

    tool = ProcessTool()
    sandbox = FakeJobExecutor()
    result = await tool.run(ProcessArgs(action="start", command="serve"), context(sandbox))
    assert result.status == "ok" and sandbox.started == [("serve", tmp_path)]

    class NoJobs:
        async def run(self, command: str, **kwargs: object) -> ExecResult:
            raise AssertionError

    refused = await tool.run(ProcessArgs(action="start", command="serve"), context(NoJobs()))
    assert refused.status == "error" and "cannot run background jobs" in refused.content
    await tool.aclose()


docker = pytest.mark.skipif(not os.environ.get("INTERN_DOCKER_IMAGE"), reason="explicit prepared local Docker image")


def containers(name: str) -> str:
    listed = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", f"name={name}"], capture_output=True, text=True, check=True
    )
    return listed.stdout.strip()


@docker
async def test_a_real_container_job_takes_input_streams_output_and_is_removed(tmp_path):
    executor = DockerExecutor(tmp_path, DockerOptions(image=os.environ["INTERN_DOCKER_IMAGE"]))
    job = await executor.start_job('read line; echo "got:$line in $(pwd)"; sleep 60', cwd=tmp_path)
    await job.write(b"hello\n")
    seen = b""
    async with asyncio.timeout(30):
        while b"got:hello in /workspace" not in seen:
            seen += await job.read()
    assert containers(job.name)
    await job.kill()
    await job.kill()  # safe again
    assert not containers(job.name)


@docker
async def test_a_docker_session_runs_the_process_tool_in_its_container(tmp_path):
    toml = f"""[capabilities]
tool_packs = ["base", "process"]
[capabilities.runner]
name = "docker"
[capabilities.runner.options]
image = {json.dumps(os.environ["INTERN_DOCKER_IMAGE"])}
"""
    replies = [
        Reply(tool_calls=[("process", {"action": "start", "command": "pwd; touch made-in-container; sleep 30"})]),
        Reply(text="started"),
    ]
    before = set(containers("harness-").split())
    async with configured(tmp_path, replies, toml) as (session, model, _, work):
        await session.run("start it")
        handle = json.loads(outputs(session)[0])["handle"]
        model.replies.extend(
            [Reply(tool_calls=[("process", {"action": "poll", "handle": handle, "wait_s": 2})]), Reply(text="ok")]
        )
        await session.run("poll it")
        assert "/workspace" in outputs(session)[1]
        assert (work / "made-in-container").exists()  # in the mounted workspace, so on the host too
    # The session closed: its job's container is gone.
    assert set(containers("harness-").split()) <= before


async def test_a_backend_whose_jobs_flag_disagrees_with_its_executor_fails_setup(monkeypatch, tmp_path):
    import intern.runners as registry
    from intern.models.registry import ConfigError

    from .test_runners import CopyBackend, EntryPoint

    class Claims(CopyBackend):
        name = "claims"
        jobs = True  # but CopySandbox has no start_job

    monkeypatch.setattr(registry, "entry_points", lambda group: [EntryPoint("claims", Claims)])
    registry.registered.cache_clear()
    (tmp_path / "remote").mkdir()
    toml = f"""[capabilities.runner]
name = "claims"
[capabilities.runner.options]
root = "{tmp_path / "remote"}"
"""
    try:
        with pytest.raises(ConfigError, match="says jobs = True but its executor has no start_job"):
            async with configured(tmp_path, [], toml):
                pytest.fail("setup must fail before the session opens")
    finally:
        registry.registered.cache_clear()
