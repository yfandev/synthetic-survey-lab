# Agent instructions

Read `CONTRIBUTING.md` before changing this repository. These rules apply to every
agent working here, including documentation, configuration, fixes and features.

## Release what clients see

T3 Code and other ACP clients install a tagged version (`@vX.Y.Z`), so a change reaches them only
once it is released.

1. **Release a change clients can see**: anything [docs/acp.md](docs/acp.md) describes (config
   options, `models --json`, errors, updates), or a fix users are waiting on. Internal changes ship
   with the next release.
2. **Bump the version in that PR**: `pyproject.toml` and `src/intern/__init__.py`, then
   `uv lock`. Patch for fixes, minor for additions. A breaking change also raises **Minimum version
   for clients** in docs/acp.md.
3. **Add a `### X.Y.Z` section** at the end of docs/acp.md: what changed, and whether older clients
   keep working.
4. **Say it in the PR body**: "Releases X.Y.Z", and the client change that needs it, if any.
5. **Do not tag by hand.** CI tags `vX.Y.Z` on the commit where the bump lands on `main`. If another
   PR released that version first, rebase and take the next one.

## Add only tests that check something real

A new test must be able to fail for a reason a user cares about: behaviour a change could break,
not that a widget got a CSS class or a helper returns what it was just written to return. Don't add a
test to show a change was made; check a look by eye, or with a throwaway script you delete. When a
change breaks an existing test that only pinned a format, update the test; don't add another.

## Keep SYSTEM1.md in step with this file

Intern's System 1, which judges an agent's tool calls, does not read this file; it reads the user's
`~/.intern/SYSTEM1.md`. When you add or change a step here that acts outside the repository (GitHub,
an issue tracker, a chat channel), make the same change there or tell the user to, or Intern's guard will block it.
