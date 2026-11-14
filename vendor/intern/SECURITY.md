# Security

## Reporting a vulnerability

Report it privately through GitHub: the repository's **Security** tab, then **Report a vulnerability**.
Do not open a public issue for it.

Say what you found, how to reproduce it (the command, the task and the models), and the version
(`intern --version`). We keep you posted until it is fixed, and credit you in the release notes unless you ask us not to.

## Scope

In scope:

- **The tool-call guard** (`src/intern/guard.py`): a command its fixed rules should refuse but let through,
  or a way to hide a dangerous call from the rules or from System 1's judgement. The rules and what each one
  refuses are listed in [docs/architecture.md](docs/architecture.md#system-2-and-system-1).
- **Secrets**: a key from the environment, `~/.intern/.env` or the [secrets store](docs/secrets.md) that
  reaches the model, a tool's output or a log unredacted.
- **Network capabilities**: the browser's or the web reader's URL policy letting a request reach a host it
  should refuse, or an MCP server's token readable by the agent.
- The ACP and JSON-RPC servers accepting input that runs code it should not.

Out of scope:

- What the agent does with the permissions you gave it. Intern is not a sandbox: commands run as your
  user, and System 1's judgement of a call is a model's judgement. Run untrusted work in the
  [Docker workspace](docs/capabilities/docker-workspace.md) or another container.
- A model provider's or the Nace console's own service.

## Supported versions

Fixes land on `main` and ship in the next release. Older versions are not patched.
