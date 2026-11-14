from __future__ import annotations

import asyncio
import os
import shlex
from pathlib import Path

import pytest

from intern.executor import ExecResult
from intern.runners.docker import DockerExecutor, DockerOptions
from intern.tools.base import ToolContext, ToolOutput
from intern.workspace_tools import workspace_tools


class Launcher:
    output_capture = True

    def __init__(self):
        self.commands: list[str] = []
        self.captures: list[Path | None] = []

    async def run(self, command: str, **kwargs) -> ExecResult:
        self.commands.append(command)
        self.captures.append(kwargs.get("capture_path"))
        return ExecResult(exit_code=0, output="ok", omitted_bytes=0, timed_out=False, aborted=False, duration_ms=1)


async def test_container_argv_quotes_inputs_and_always_removes_owned_container(tmp_path: Path):
    launcher = Launcher()
    executor = DockerExecutor(tmp_path, launcher=launcher)
    await executor.run(
        "echo 'x; false'",
        cwd=tmp_path,
        timeout_s=1,
        max_output_bytes=100,
        env={"TOKEN": "secret-env-value"},
        capture_path=tmp_path / "capture.txt",
    )
    argv = shlex.split(launcher.commands[0])
    assert argv[-1] == "echo 'x; false'" and "secret-env-value" not in launcher.commands[0]
    assert argv[argv.index("--network") + 1] == "none" and "--read-only" in argv
    name = argv[argv.index("--name") + 1]
    assert shlex.split(launcher.commands[1]) == ["docker", "rm", "-f", name]
    assert launcher.captures == [tmp_path / "capture.txt", None]
    with pytest.raises(ValueError):
        await executor.run("true", cwd=tmp_path.parent, timeout_s=1, max_output_bytes=100)


@pytest.mark.skipif(not os.environ.get("INTERN_DOCKER_IMAGE"), reason="explicit prepared local Docker image required")
async def test_real_docker_shell_and_every_filesystem_tool_share_workspace(tmp_path: Path):
    executor = DockerExecutor(tmp_path, DockerOptions(image=os.environ["INTERN_DOCKER_IMAGE"]))
    ctx = ToolContext(cwd=tmp_path, executor=executor, abort=asyncio.Event(), max_output_chars=5000)
    tools = {t.name: t for t in workspace_tools()}

    async def call(name, **arguments):
        tool = tools[name]
        args = tool.parse(arguments)
        assert not isinstance(args, ToolOutput)
        result = await tool.run(args, ctx)
        assert result.status == "ok", result.content
        return result.content

    await call("write", path="sub/a.txt", content="hello\nworld\n")
    assert "world" in await call("read", path="sub/a.txt")
    await call("edit", path="sub/a.txt", old_string="world", new_string="changed")
    assert "sub/a.txt:2:changed" in await call("grep", pattern="changed")
    assert "sub/a.txt" in await call("find", pattern="*.txt")
    assert "changed" in await call("bash", command="cat sub/a.txt")
    denied = tools["read"].parse({"path": str(tmp_path.parent / "host-only")})
    assert not isinstance(denied, ToolOutput)
    assert (await tools["read"].run(denied, ctx)).status == "error"
    assert (tmp_path / "sub/a.txt").read_text() == "hello\nchanged\n"
