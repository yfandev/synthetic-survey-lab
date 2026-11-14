# Connecting outside services

Intern has no Gmail, Slack or GitHub code. It reaches outside services through
MCP servers: the vendor's own server where one exists, a server you run yourself,
or an aggregator for the rest. Each server is one `[[capabilities.mcp]]` entry in
`~/.intern/config.toml` ([MCP settings](../capability-settings.md#mcp)).

## Pick how the entry signs in

| The server… | Use | Example |
|---|---|---|
| runs on your machine | `transport = "stdio"`, secrets in `env_refs` | a local filesystem or database server |
| takes a fixed token or API key | `header_env = { Authorization = "MY_TOKEN" }` | a GitHub personal access token |
| takes a token that expires, from a command or token endpoint | [`credential_command`](mcp.md#credential-command) | Pipedream, `gcloud auth print-access-token` |
| signs you in with your own account (MCP OAuth) | [`[capabilities.mcp.oauth]`](mcp.md#oauth-sign-in), then `intern mcp login <namespace>` | Linear, Notion |

Every entry lists the exact remote tools it exposes in `remote_tools`, each with
its effects. Setup fails if a listed tool does not exist, so take the names from
the vendor's tool list.

## Declare effects honestly

Effects decide what the [guard](../architecture.md) asks System 1 about.
Tools declared only `read` are not judged.

- A tool that sends, posts, creates, edits, shares or deletes is **never** `read`.
  Declare it `["write", "network"]`.
- A tool that only fetches is `["read", "network"]`.
- When unsure, declare `write`. Declaring a write as `read` lets it run without a
  check.

```toml
remote_tools = { search_issues = ["read", "network"], create_issue = ["write", "network"] }
```

Start with the read tools you need and add write tools one by one.

## Treat what servers return as untrusted

Gmail messages, Slack threads, Notion pages and web pages are written by other
people, and they can contain instructions aimed at the agent.

- The guard judges a tool call from your messages and the call itself; it never
  reads tool results, so a planted instruction cannot argue a call past it. It can
  still steer which calls the model makes.
- Be careful when one session has both a private-data server (mail, files, chat)
  and a way to send data out (a web tool, a send or post tool). Make sure every
  outgoing tool is declared `write`, so each call is judged.
- A hosted server's tool descriptions and schemas come from its operator and can
  change. Keep `remote_tools` to the tools you have read about.

## Servers we list

A server is listed here when its vendor documents a remote MCP endpoint, it uses
one of the sign-in methods above, and there is a free way to use it with your own
account. Each entry says whether we have tested it with Intern. Endpoints
and auth details change; check the vendor's page before relying on them.

Checked against vendor and integration docs on 2026-11-11. "Tested" means a
harness build signed in or authenticated, listed the server's tools and
called a read tool on that date.

| Service | Endpoint | Sign-in | Tested | Notes |
|---|---|---|---|---|
| GitHub | `https://api.githubcopilot.com/mcp/` | a token in `header_env`, or `credential_command` running `gh auth token` | yes, read-only endpoint, token from `gh auth token` | OAuth there needs a GitHub app registered for the client. `/mcp/readonly` serves read tools only. A malformed token gets 400 with `error="invalid_token"`, which the credential command treats as rejected. [GitHub docs](https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp-in-your-ide/set-up-the-github-mcp-server) |
| Linear | `https://mcp.linear.app/mcp` | OAuth (`oauth = {}`) | yes, sign-in, refresh, a harness session | Registers the client itself; 78 tools. `/mcp/readonly` serves read tools only. [Linear MCP](https://linear.app/docs/mcp) |
| Notion | `https://mcp.notion.com/mcp` | OAuth (`oauth = {}`) | yes, sign-in and refresh | Registers the client itself; 77 tools, names start with `notion-`. [Notion MCP](https://developers.notion.com/docs/get-started-with-mcp) |
| Slack | `https://mcp.slack.com/mcp` | OAuth with a Slack app: `client_id`, `client_secret_env` | no | A workspace admin must approve the app, and Slack limits the server to internal or directory-published apps. [Slack MCP server](https://docs.slack.dev/ai/slack-mcp-server/) |
| Google Drive | `https://drivemcp.googleapis.com/mcp/v1` | OAuth with a Google Cloud OAuth client: `client_id`, `client_secret_env` | no | Developer Preview; enable the Drive MCP service in your Google Cloud project first. Google also previews a Gmail server (drafts and labels, no send). [Workspace announcement](https://workspaceupdates.googleblog.com/2026/05/agent-tools-and-security-updates-for-workspace-developers.html) |

For services without their own server, an aggregator can cover many apps behind
one sign-in:

| Service | Endpoint | Sign-in | Cost |
|---|---|---|---|
| Pipedream (2,500+ apps) | `https://remote.mcp.pipedream.net/v3` | `credential_command` that mints a Connect token, plus `x-pd-project-id`, `x-pd-environment`, `x-pd-external-user-id`, `x-pd-app-slug` in `header_env` | Free in development; production is billed per end user. Not tested yet. [Pipedream MCP for developers](https://pipedream.com/docs/connect/mcp/developers) |

Not listed: paid per-call data catalogs (Treg, Monid), which are not account
connections, and Nango, whose MCP server is not in the free self-hosted edition.
They work like any other HTTP entry if you choose to use them.

## Examples

A Linear entry that signs in with your account and exposes two tools:

```toml
[[capabilities.mcp]]
namespace = "linear"
transport = "http"
url = "https://mcp.linear.app/mcp"
remote_tools = { list_issues = ["read", "network"], save_issue = ["write", "network"] }

[capabilities.mcp.oauth]
```

```sh
intern mcp login linear
```

A GitHub entry, read tools only, that takes its token from the GitHub CLI each
time GitHub rejects the current one:

```toml
[[capabilities.mcp]]
namespace = "github"
transport = "http"
url = "https://api.githubcopilot.com/mcp/readonly"
remote_tools = { search_issues = ["read", "network"], get_me = ["read", "network"] }

[capabilities.mcp.credential_command]
command = "/bin/sh"
args = ["-c", "printf 'Bearer %s' \"$(gh auth token)\""]
```

These tool names were on each server's list on 2026-11-11; servers add and rename
tools, so check the current list.

## Adding a server to this page

Open a pull request that adds a row with the endpoint, the sign-in method, a link
to the vendor's own page, the date you checked it, and whether you tested it with
`intern`. Mark a row tested only after a session called one of its tools.
