# OpenSandbox

[OpenSandbox](https://github.com/opensandbox-group/OpenSandbox) runs sandboxes on self-hosted Docker or Kubernetes.
As a [runner](runners.md), it runs a session's shell commands, file tools and background jobs
in one sandbox, which lives as long as the session. Use it over Docker to keep state between commands (installed
packages, running services), or to run the agent's work on another machine.

## Setup

1. Install the extra: `uv sync --extra opensandbox`, or install `intern-agent[opensandbox]`.
2. Run an OpenSandbox server (its `opensandbox-server`, with the Docker or Kubernetes runtime).
3. Configure the backend:

```toml
[capabilities.runner]
name = "opensandbox"

[capabilities.runner.options]
domain = "sandbox.internal:8080"      # unset: $OPEN_SANDBOX_DOMAIN, else the OpenSandbox SDK's localhost:8080
protocol = "https"                    # default "http": the API key then crosses the network in the clear
api_key_env = "OPEN_SANDBOX_API_KEY"   # "" for a server without a key
image = "python:3.12-slim"            # needs bash, tar, Python 3, and git for the guard's check
workspace_mode = "copy"               # or "mount"
timeout_s = 3600                      # the sandbox's lifetime on the server
# cpu = "1", memory = "2Gi", sandbox_path = "/workspace", scratch_path = "/tmp",
# max_upload_mb = 256, request_timeout_s = 60
```

The runner ships with the harness, so it is always among the runners (ACP `runner`, `/runner opensandbox`)
listed. Until the extra is installed and the key is set, it shows which one is missing, and a session
configured with it fails at setup and says so, starting nothing.

## How the workspace reaches the sandbox

- **`copy`** (default; works with any server, local or remote).
  - On the first command, the workspace is uploaded as one archive, up to `max_upload_mb`. `.venv`, `node_modules`
    and `__pycache__` are left out, since they are built for the host.
  - From then on, the sandbox is the source of truth. After each command and at close, the files that changed
    there are copied back, renamed ones included.
  - Files the sandbox deletes stay on the host.
  - Host edits made after the upload are not seen in the sandbox. The exception is what the harness itself writes
    into `.intern-resources/` (the web reader's saved pages), which is sent before each command.
  - Each command costs a round trip to list and fetch its changes.
- **`mount`.** The host folder is bind-mounted, so both sides see one set of files, and nothing is copied. This
  needs a server on this machine whose `[storage] allowed_host_paths` include the workspace.

## Network and credentials

An agent that reads web pages can be steered by what it reads (prompt injection). Two options limit what a
sandbox can reach and what it can leak:

```toml
[capabilities.runner.options]
egress_default = "deny"                         # what happens to a connection no rule matches
egress = [                                       # rules, matched in order
  { action = "allow", target = "pypi.org" },
  { action = "allow", target = "*.github.com" },
  { action = "allow", target = "10.0.0.0/8", ports = [5432] },
]

[capabilities.runner.options.credentials.github]
env = "GITHUB_TOKEN"            # read on this machine when the sandbox starts; never written in config.toml
hosts = ["api.github.com"]      # https requests to these hosts get it
auth = "bearer"                 # or "basic" (the value is user:password), or "api_key" with header = "X-Api-Key"
```

- **`egress`.** These are the only places the sandbox can connect to; a blocked domain does not even resolve.
  - A rule's `target` is a domain (`example.com`, `*.example.com`), an IP or a CIDR.
  - Add `ports` to limit an IP or CIDR rule to those TCP ports. A domain rule takes no ports.
  - Leaving `egress` unset leaves the server's network as it is.
- **`credentials`.** Each one goes into the sandbox's credential vault, which only the server's proxy can read.
  - The proxy adds the secret to https requests for its `hosts`. Code in the sandbox makes the call without a
    token, and the request arrives signed.
  - The value is not in the sandbox's environment or files, so a compromised agent cannot print it or send it
    anywhere else.
  - Setup checks that every credential's hosts are allowed by the `egress` rules.
  - The runner shows as unavailable while any `env` variable is unset.

What the OpenSandbox server needs for these:

- `egress`: its egress sidecar (the `[egress]` section, with its image pulled) and Docker `bridge` networking.
- `credentials`: also `[egress] mode = "dns+nft"`. Otherwise sandbox creation fails with that message.

Both were tested against a live server: an allowed site loads, others don't resolve, and a credentialed request
arrives with `Authorization: Bearer <value>` while the sandbox's environment holds no copy of the value.

## Background jobs

The `process` tool's jobs run in the session's sandbox. They take no input: `process write` returns an error
saying so. `terminate` interrupts the job on the server.

## Limits and risks

- **Writes back into the workspace.** Sandboxed code can write any file in the workspace, and copy-back or the
  mount brings it to the host. That includes files the host runs later: `.git/hooks/*`, `.envrc`, editor and task
  settings. Archive extraction refuses paths that escape the workspace, but not these. Review such files before
  you use the workspace on the host.
- **The API key.** It goes to whatever `domain` names, over plain `http` unless `protocol = "https"`.
- **Open network by default.** Without `egress`, the sandbox can reach whatever the server's network allows. For an
  agent that browses or installs packages, set `egress_default = "deny"` and list what it needs.
- **Credentials and their hosts.** A vault credential is safe from the sandbox, but anything the sandbox sends to
  its `hosts` is sent with it. Give it only the hosts and token scope the task needs.
- **Lifetime.** The server ends the sandbox after `timeout_s`. A longer session then fails its next command;
  changes not yet copied back are lost.
- **Close.** The last copy-back runs before the sandbox is killed. If it fails, session close reports it.
- **Upload size.** A workspace over `max_upload_mb` fails the first command, not setup.
- **Stopping jobs.** `terminate` relies on the server's interrupt to stop what the job started.
