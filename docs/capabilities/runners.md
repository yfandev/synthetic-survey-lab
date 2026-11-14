# Runners

A **runner** decides where the agent's commands run. A session runs on exactly one:

| Runner | Where commands run | What persists between commands |
|---|---|---|
| `local` (default) | this machine, as the user who started the harness | everything; it is the host |
| `docker` | a fresh locked-down container per command ([Docker workspace](docker-workspace.md)) | only the mounted workspace |
| `opensandbox` | one sandbox per session, on a local or remote OpenSandbox server ([OpenSandbox](opensandbox.md)) | the whole sandbox, until the session closes |

A package can add another runner without changing the harness.

Two words, two things: a *runner* is the plugin you pick by name. The *executor* is the object it builds for a
session, which runs each command (`intern.executor.Executor`).

## Configuration

```toml
[capabilities.runner]
name = "docker"             # local (default), docker, opensandbox, a registered name, or module:Class
workspace = "."             # isolated runners only; relative to the session's working directory, and must contain it

[capabilities.runner.options]
image = "python:3.12.11-slim-bookworm"   # the runner's own options
```

`name` takes:

- `local`, `docker` or `opensandbox`, which ship with the harness (`opensandbox` needs its extra);
- the name a package registers under the `intern.runners` entry-point group;
- `module:Class`, the same form `--extension` takes for [extensions](../architecture.md).

Two moments matter:

- **Config load.** This is when `config.toml` is read. The runner's module is imported, and the runner checks its own
  `options`. A misspelt option fails here, and the error names the field without its value, which may be a secret.
  A runner this install cannot load (its package missing) does not stop the load. Its options stay unchecked, and a
  session that uses it fails at setup and says why.
- **Setup.** This is when each session is opened, and again when picking another runner reopens the session. The
  executor is built here, but nothing starts: a runner that owns a service, such as a sandbox, starts it on the
  first command.

A runner is **isolated** when commands and files live away from the host, as with `docker` and `opensandbox`.
`local` is the only runner that is not.

## Picking the runner per session

The runner is one choice per session, made without editing `config.toml`:

- **T3 Code and other ACP clients:** the `runner` option, a select. See [ACP](../acp.md#config-options).
- **Terminal UI:** `/runner` opens a picker; `/runner docker` sets it directly.

The session reopens on its journal on the new runner, at the next prompt in ACP and at once in the terminal UI.
The pick is journaled, so a loaded session continues on it.

The list holds every runner: the built-ins, each one an installed package registers, and the configured one when
it is a `module:Class`. A runner that cannot start here shows why beside its name, before anyone picks it, and
picking it is refused with that reason. The reasons are:

- it cannot load;
- its extra is missing or its key is unset (its `problem()`);
- it cannot run without options that only `config.toml` can give.

The configured runner runs with its configured options. Any other runner picked for a session runs on its own
default options, never another runner's.

## What follows the runner

- **Shell and file tools.** For an isolated runner, setup replaces `read`, `write`, `edit`, `grep` and `find` with
  versions that go through the executor, next to `bash`. If an extension already replaced one of those tools,
  setup fails instead of overriding it: use that extension's own adapter, or the `local` runner.
- **Background jobs.** The `process` tool runs its jobs through the executor when the runner runs jobs (Docker runs
  one container per job; OpenSandbox runs them in the session's sandbox). With a runner that runs none, the
  `process` tool is refused rather than started on the host while commands run in the sandbox.
- **The tool-call guard.** The guard refuses destructive or exfiltrating calls by fixed rules, and System 1 judges
  the risky rest. It runs before every tool call, whatever the runner, and reads paths as the tools see them
  (`/workspace/...` under Docker) as well as host paths. Its uncommitted-work check runs `git status` through the
  executor, so the image needs `git` for that check to apply.
- **The web reader.** It saves the pages it fetches (its retained sources) inside the workspace, under
  `.intern-resources/`, where the file tools can read them.
- **Session close.** An executor that owns a service is closed with the session's other resources, last, after
  anything that could still run a command in it.

## Writing a runner

```python
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from intern.executor import Executor
from intern.runners import Runner


class SandboxOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)  # a misspelt option fails, values unshown
    template: str = "base"
    api_key_env: str = "SANDBOX_API_KEY"


class SandboxRunner(Runner):
    name = "sandbox"
    title = "My sandbox"  # what /runner and T3 Code show
    note = "commands run in a remote sandbox"
    Options = SandboxOptions
    jobs = True  # its executor has start_job; the default is False
    # isolated = True is the default

    def problem(self, options: BaseModel) -> str | None:
        return None  # e.g. "needs $SANDBOX_API_KEY"; shown beside its name, and setup fails with it

    def create(self, options: BaseModel, *, workspace: Path, cwd: Path) -> Executor:
        assert isinstance(options, SandboxOptions)
        return SandboxExecutor(workspace, options)  # the class below; start nothing here
```

Register it in the package's `pyproject.toml`:

```toml
[project.entry-points."intern.runners"]
sandbox = "my_package:SandboxRunner"
```

The runner's class attributes:

- `isolated` (default `True`)
- `jobs` (default `False`): set it to `True` exactly when the executor has `start_job`. Config load checks this flag,
  because no executor exists yet then. Setup then fails when the flag and the executor disagree.
- `problem(options)`: optional. A cheap check that starts nothing; return the reason it cannot run here.
- `create`: raises `ValueError` with a message free of secrets and option values when the workspace or options do
  not fit. Setup reports it as a configuration error.

The executor (`SandboxExecutor` above) **must** have:

- `async run(command, *, cwd, timeout_s, max_output_bytes, abort=None, env=None) -> ExecResult`; see
  `intern.executor.Executor`. `cwd` arrives as a host path. Run the command at `self.mount.to_inner(cwd)`.
- When isolated, a `mount` property returning `WorkspaceMount(host, inner)`: the host workspace folder and the path
  the tools see it at. During a session the executor's side is the source of truth. The host must have every change
  back after each command and at close, by a bind mount or by copying back. Files the harness writes into
  `.intern-resources/` on the host must reach the executor's side before the next command.

It **may** have:

- `async start_job(command, *, cwd, env=None) -> Job`, for the `process` tool, with `jobs = True`. A `Job` has:
  - `async read() -> bytes`: output so far, then `b""` at the end;
  - `async write(data: bytes, *, eof: bool = False)`: raise `NotImplementedError` when jobs take no input;
  - `async wait() -> int | None`: the exit code;
  - `async kill()`: stops the job and what it started; safe to call again.

  The process manager applies its deadline, abort and output bound on top.
- `async aclose()`, when it owns a service: stop the sandbox and close its client connections. Start the service on
  the first `run`, not in `create`.

The image needs bash, Python 3 (the file tools run a small Python program in it) and `git` (for the guard's check).
Each file-tool call sends at most about 100 KB and has 60 s.

`tests/test_runners.py` has a worked example: a copy-based sandbox registered the way a package registers one.
