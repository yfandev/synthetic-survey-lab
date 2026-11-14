# Secrets store

Keep a key in the OS keychain or your password manager instead of a plain-text file. You name where it comes from
in `[secrets]`, and the harness does the rest:

- it reads the value when a session starts. A `"keychain"` entry is read fresh each session, which is silent (no
  Touch ID, no password), so a changed value applies to the next session. A command entry runs once per harness
  process and is kept in memory, so a password manager asks to unlock at most once;
- the harness's own key lookups use it: model providers, search, MCP `header_env` and `env_refs`. So it can
  replace a line in `~/.intern/.env`, and it wins over an exported variable with the same name;
- the agent's commands and background jobs get it as an environment variable with the same name, alongside
  your normal environment. Keys the harness uses itself stay out of them: a model provider's `api_key_env`
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...), and any variable the config names in a `*_env`, `header_env` or
  `env_refs` setting (search, System 1, MCP). The agent can't spend or leak your model key. If a task needs that
  key too (say, a script that calls the OpenAI API), export it in your shell as before;
- it is replaced with `[REDACTED]` in what those commands print.

Anything you don't put in the store works as before. `gh`, `npm` and `aws` keep their own logins, and exported
variables still reach the agent.

## One secret

```bash
intern secrets set GITHUB_TOKEN     # asks for the value without showing it
```

That stores the value in the OS keychain (service `intern`, the name as the account) and adds
`GITHUB_TOKEN = "keychain"` to `[secrets]` in `~/.intern/config.toml`, keeping the rest of the file as you
wrote it. Running it again replaces the value. The next session uses it, with no restart.

| Command | What it does |
|---|---|
| `intern secrets set NAME` | Store a value in the keychain and name it in config. `--stdin` reads the value from stdin. |
| `intern secrets list` | Show each entry, where it comes from, and whether it is harness only. Reads no values. |
| `intern secrets test [NAME...]` | Read each entry now and say whether it works; exits 1 if one doesn't. |
| `intern secrets remove NAME` | Delete the entry and its keychain item. |

Every command takes `--json` and then prints one JSON document to stdout, for a GUI or script. `set --stdin` is
how a GUI passes a value: on stdin, never as an argument. No output ever contains a value. A keychain value is
one line of plain ASCII, as API keys are; for a multi-line key (PEM, JSON) or other text, use a command entry.

## From a password manager

Any command that prints the value works. It runs with your environment, so the CLI finds its own session:

```toml
[secrets]
OPENAI_API_KEY = ["op", "read", "op://Private/OpenAI/key"]   # 1Password
NPM_TOKEN = ["pass", "show", "npm/token"]                     # pass
DB_PASSWORD = ["vault", "kv", "get", "-field=password", "secret/db"]
```

A bare command name is looked up on the `PATH` the harness started with.

## Rules

- Each name on the left is the environment variable that commands see. The value must be `"keychain"` or a
  command list.
- A value must be at least 8 characters, at most 64 KiB of UTF-8, and is used without its trailing newline.
  Shorter values can't be redacted without mangling ordinary output.
- If an entry can't be read, the session starts without it and shows a notice in the terminal, the TUI or your
  editor. The notice names the entry and what to do, for example `GITHUB_TOKEN: not in the keychain. Store it
  with: intern secrets set GITHUB_TOKEN` or `NPM_TOKEN: op exited 1. Try: op signin`. It never contains
  the value or what the command printed.
  The next session tries again. A missing model key still stops the session, with its own message.
- A password manager may ask for its own unlock: `op` asks for Touch ID the first time a process uses it. The
  harness asks once per process, not once per session. After you change a value there, restart the harness (or
  your editor's agent) to pick it up. Keychain values need no restart.
- The keychain works on macOS, and on Linux in a desktop session (Secret Service over D-Bus). CI, containers
  and E2B usually have no keychain. There, use a command with a service-account token, or an environment
  variable.
- `[secrets]` and `[capabilities.secrets] inject` can't both be set. The second is for hosts that need
  scopes; see [scoped secret references](capabilities/secrets.md).
- With the `local` runner, commands see your whole environment plus the stored values. Other runners keep their
  short allowlist (`PATH`, `HOME`, `LANG`, ...).

## What this protects, and what it doesn't

It keeps keys out of config and `.env` files, and out of everything the harness records. A stored value is
replaced with `[REDACTED]` in the output of every tool (commands, background jobs, `read`, `grep`, MCP, the
browser, extensions) before the model, the session journal, your editor or a spill file sees it. Spill and
evidence files are readable only by you (mode 0600).

Files the agent's own commands write are theirs: if a command writes the value into `leak.txt`, the file holds
it, but reading it back through any tool shows `[REDACTED]`.

It does not stop an agent that sets out to misuse a key. The agent's commands run as you, so they can use the
value, send it somewhere, or print it encoded (base64 isn't caught). For that, run commands in a sandboxed runner
with egress rules.
