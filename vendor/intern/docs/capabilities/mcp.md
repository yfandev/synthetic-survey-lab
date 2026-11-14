# Optional MCP client

Install `uv sync --extra mcp`. Importing the harness or this module does not start
servers, connect, install packages or import the optional SDK. Connections use
the official Python MCP SDK (tested with 1.30) and are explicit async contexts.

```python
from intern.mcp_client import StdioConfig, stdio_connection

async with stdio_connection(StdioConfig(command="my-server", args=[]), "docs") as connection:
    remote_tools = await connection.tools({"search": frozenset({"network"})})
    session.deps.tools += remote_tools
    await session.run("Find the relevant documentation")
```

The application supplies effects for each selected remote tool. Server
annotations do not grant permissions. Each tool has its actual namespaced
identity, original JSON input schema and standard `Tool.run` execution boundary,
so policy and receipt wrappers can apply normally. JSON schema validation rejects
bad calls before dispatch; external schema references are refused. Colliding or
overlong names and unbounded catalog pagination fail setup explicitly.

`http_connection(url, namespace, http_client=...)` supports Streamable HTTP; supply
an authenticated httpx client for bearer/OAuth refresh. URLs cannot embed secrets.
Credential refresh comes from that client: your own `httpx.Auth`, or the
`CommandCredential` below. Nothing is copied from browser subscriptions. No
redirects or endpoints are inferred.

## Credential command

Some servers take a token that expires: Pipedream's MCP server takes an access
token from its client-credentials endpoint, and cloud CLIs print short-lived
tokens. An HTTP entry can name a command that prints the header value:

```toml
[capabilities.mcp.credential_command]
command = "/usr/local/bin/pd-token"     # absolute path; not a shell string
args = []
env_refs = { CLIENT_ID = "PIPEDREAM_CLIENT_ID", CLIENT_SECRET = "PIPEDREAM_CLIENT_SECRET" }
header = "Authorization"                 # the default
```

- **Contract.** The command exits 0 and prints the whole header value, such as
  `Bearer eyJ...`. One trailing newline is removed. The value must be 1 to 8192
  printable ASCII characters. Anything else fails: a non-zero exit, a timeout,
  more than 64 KiB of output, or an invalid value.
- **When it runs.** Once when the session is set up, before any MCP server starts;
  a failure stops setup with the exit status and the end of stderr, with the
  `env_refs` values redacted. Then again only when the server rejects the value:
  a 401, or any status whose `WWW-Authenticate` header says
  `error="invalid_token"` (GitHub answers a malformed token with 400 that way). The
  request is retried once with the new value, and requests that fail together
  share one run. It does not run on a timer, so each token lifetime costs one
  rejected request. It also runs again when a session reopens.
- **How it runs.** Without a shell, in its own process group, with stdin closed,
  in the harness config home, with only `PATH` (without the harness's own
  virtualenv), `HOME`, `LANG`, `LC_ALL`, `TMPDIR`, `TEMP`, `SYSTEMROOT` and the
  `env_refs` (`{name the command sees: host variable}`). It may take up to 30 s,
  and less than the entry's `timeout_s`. Its process group is stopped when it
  finishes, times out or is cancelled.
- **Trust.** It is operator configuration, like a stdio server: it runs with your
  user's rights. The model cannot set it, and it does not run in the session's
  working directory. The header value is not written to model context, events or
  the journal. The command's `env_refs` come from the host environment, as for
  stdio servers; with the local executor the agent's shell sees that environment
  too, so use the `docker` runner or `[capabilities.secrets]` to isolate it.
- **Limits.** A server that rejects an expired token with 403 or 400 and no
  `error="invalid_token"` is not recognized as an expired token. Each
  entry runs its own command; values are not shared across entries or processes.
  If the command fails mid-session, the MCP transport stops: the tool call fails
  and the connection is unusable until the session reopens.

For a token that another program keeps in a file, use
`command = "/bin/cat"` with `args = ["/run/secrets/mcp-token"]`. For a CLI that
prints a bare token, wrap it:
`command = "/bin/sh"` with `args = ["-c", "printf 'Bearer %s' \"$(gcloud auth print-access-token)\""]`.

A Pipedream token command, using only the standard library (check Pipedream's
Connect docs for the exact token request):

```python
#!/usr/bin/env python3
import json
import os
import urllib.request

body = json.dumps(
    {
        "grant_type": "client_credentials",
        "client_id": os.environ["CLIENT_ID"],
        "client_secret": os.environ["CLIENT_SECRET"],
    }
).encode()
request = urllib.request.Request(
    "https://api.pipedream.com/v1/oauth/token", data=body, headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(request, timeout=10) as response:
    print("Bearer " + json.load(response)["access_token"])
```

Pipedream's server also needs `x-pd-project-id`, `x-pd-environment`,
`x-pd-external-user-id` and `x-pd-app-slug`, which go in `header_env`.

Python callers can use `CommandCredential` (exported from
`intern.mcp_client`) as the `auth=` of their own `httpx.AsyncClient`.

## OAuth sign-in

Hosted servers that follow MCP's authorization spec ask you to sign in with your
own account. Add an `oauth` table to the HTTP entry and sign in once:

```toml
[[capabilities.mcp]]
namespace = "linear"
transport = "http"
url = "https://mcp.example.com/mcp"
remote_tools = { list_issues = ["read", "network"], create_issue = ["write", "network"] }

[capabilities.mcp.oauth]                 # an empty table is enough for most servers
# client_id = "my-app"                   # only if the server needs a pre-registered app
# client_secret_env = "MY_APP_SECRET"    # its secret, read from the environment
# scope = "read write"
# callback_port = 33418                  # the browser returns to http://127.0.0.1:<port>/callback
```

```sh
intern mcp login linear     # opens the browser; prints the address too
intern mcp logout linear    # deletes the saved tokens
```

- **Sign-in.** `mcp login` finds the server's authorization server, registers
  the harness as a client when no `client_id` is given, and opens the sign-in
  page. The flow uses PKCE. It waits up to 5 minutes for the browser to return
  to the local callback address. This all comes from the official MCP SDK's
  OAuth client; the harness adds the login command and the token store.
- **Sessions.** A session uses the saved access token and refreshes it with the
  refresh token when it expires. A session never starts a sign-in: if no tokens
  are saved, setup stops with `MCP <namespace> needs sign-in: run intern
  mcp login <namespace>`. If the server rejects the saved tokens and the refresh,
  setup or the tool call fails; run `mcp login` again.
- **Storage.** Tokens are saved in `<config home>/mcp-oauth/<namespace>.json`,
  mode 0600 in a 0700 folder, together with the server URL. Tokens saved for
  another URL are ignored. In the default config home (`~/.intern`), the
  guard treats that folder as credentials: the agent's `read`, `write` and shell
  tools are refused there. With `$INTERN_HOME` elsewhere, the guard does
  not know the folder.
- **Limits.** The OS keychain is not used. Two processes refreshing the same
  entry at once can invalidate each other's refresh token; sign in again if that
  happens. Servers that do not report how long a token lives are only refreshed
  through a new `mcp login`. Sign-in needs a browser on the same machine, because
  the callback address is on `127.0.0.1`.

Close connections in the task that opened them. Timeout/cancellation does not
prove a remote side effect failed; check the remote outcome before retrying an
unsafe operation. Tools become unavailable after context closure. This adapter
returns text; nontext blocks are explicitly reported as omitted. Very large text
is bounded with an explicit truncation marker; ask for narrower results. Which
servers to connect, and which tools and effects to expose, is operator
configuration. What a server returns is not trusted, and neither are a hosted
server's tool descriptions, which its operator can change: see
[treat what servers return as untrusted](connections.md#treat-what-servers-return-as-untrusted).
Default tools/profiles remain unchanged; paired evals are pending.

To choose a server and a sign-in method for a service, see
[connecting outside services](connections.md).
