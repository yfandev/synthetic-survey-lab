"""What a capability needs installed beyond the base install, and installing it.

Each capability declares its own `Needs`: an extension as `Extension.needs` (code search, the workspace), the browser
and MCP beside their code. A `Needs` names the `intern` extra, the modules that show the extra is there, and the
commands that finish the setup once it is (the browser's Chromium download). Nothing here knows any one capability.

`install` puts a missing extra's packages into the environment this Python runs in, as the installed `intern`
lists them (the extra's requirements in its metadata), and makes them importable in this process; the terminal UI's
`/caps` and the ACP server's `cap.*` toggles call it before they switch a capability on.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.metadata
import importlib.util
import re
import shlex
import shutil
import sys
from collections.abc import Awaitable, Callable, Sequence

from pydantic import BaseModel, ConfigDict

from intern import names

DISTRIBUTION = names.DISTRIBUTION
QUIET_S = 1.0
"""How long a setup command may run before `install` says it is running: one that has nothing to do is quiet."""
OUTPUT_LINES = 6
"""The last lines of a failed command's output shown with its failure."""

Say = Callable[[str], Awaitable[None]]
"""Where `install` reports what it is doing: a line for the user."""


class InstallFailed(Exception):
    """What could not be installed and why, with the command to run by hand."""


class Needs(BaseModel):
    """What a capability needs installed beyond the base install, as the capability declares it."""

    model_config = ConfigDict(frozen=True)

    title: str
    """The capability, as messages start: `Browser`, `Code search`, `The workspace`."""
    extra: str
    """Its `intern` extra: `pip install 'intern-agent[<extra>]'`."""
    modules: tuple[str, ...]
    """Modules the extra brings; one that cannot be found means the extra is not installed."""
    setup: tuple[tuple[str, ...], ...] = ()
    """Commands run with this Python once the packages are there (`-m playwright install chromium`), each a no-op
    when its work is done already, so every switch-on runs them."""

    @property
    def instruction(self) -> str:
        """What the capability needs, said the way a user would install it by hand."""
        steps = ", then ".join(_shown(command) for command in self.setup)
        then = f", then {steps}" if steps else ""
        return f"{self.title} needs the {self.extra} extra: install {DISTRIBUTION}[{self.extra}]{then}"

    def missing(self) -> list[str]:
        """The modules this environment lacks; empty when the extra is installed."""
        return [name for name in self.modules if importlib.util.find_spec(name) is None]

    def problem(self) -> str | None:
        """Why the capability cannot switch on in this environment, else None."""
        return self.instruction if self.missing() else None

    def installable(self) -> bool:
        """Whether `install` can give this environment what the capability needs: nothing is missing, or the installed
        `intern` lists the extra's packages and there is uv or pip to install them with."""
        if not self.missing():
            return True
        requirements = self.requirements()
        return bool(requirements) and installer(requirements) is not None

    def requirements(self) -> list[str]:
        """The extra's packages as the installed `intern` lists them (`playwright<2,>=1.58`): the versions this
        harness was built with, whatever installed it."""
        try:
            listed = importlib.metadata.requires(DISTRIBUTION) or []
        except importlib.metadata.PackageNotFoundError:
            return []
        marker = re.compile(rf"""^\s*extra\s*==\s*["']{re.escape(self.extra)}["']\s*$""")
        found = []
        for line in listed:
            requirement, _, condition = line.partition(";")
            if marker.match(condition):
                found.append(requirement.strip())
        return found


def _shown(command: Sequence[str]) -> str:
    """A setup command as a user types it: `playwright install chromium` for `-m playwright install chromium`."""
    return shlex.join(command[1:] if command[:1] == ("-m",) else command)


def installer(requirements: Sequence[str]) -> list[str] | None:
    """The command that installs `requirements` into this Python's environment: uv's pip when uv is on the PATH (a
    uv tool or project environment has no pip), else this Python's pip; None when there is neither."""
    if (uv := shutil.which("uv")) is not None:
        return [uv, "pip", "install", "--python", sys.executable, *requirements]
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "install", *requirements]
    return None


async def install(needs: Needs, say: Say) -> None:
    """Install what `needs` lacks into this Python's environment, then run its setup commands. Says what it does
    through `say`; raises InstallFailed with the reason and the command to run by hand."""
    if missing := needs.missing():
        requirements = needs.requirements()
        if not requirements:
            raise InstallFailed(f"{needs.instruction}. This install does not list the {needs.extra} extra's packages.")
        command = installer(requirements)
        if command is None:
            raise InstallFailed(f"{needs.instruction}. There is no uv or pip here to install it with.")
        await say(
            f"{needs.title} needs the {needs.extra} extra ({', '.join(missing)} missing): installing "
            f"{' '.join(requirements)} into {sys.prefix}…"
        )
        code, output = await _run(command)
        if code != 0:
            by_hand = shlex.join(command)
            raise InstallFailed(f"Installing the {needs.extra} extra failed{_tail(output)}. By hand: {by_hand}")
        importlib.invalidate_caches()  # the new packages' folders were not there when this process looked
        if still := needs.missing():
            raise InstallFailed(
                f"The {needs.extra} extra installed, but {', '.join(still)} still cannot be imported here; "
                f"/reload restarts on it"
            )
        await say(f"Installed the {needs.extra} extra.")
    for step in needs.setup:
        command = [sys.executable, *step]
        code, output = await _run(command, say, f"{needs.title}: {_shown(step)} (the first time downloads it)…")
        if code != 0:
            raise InstallFailed(f"{_shown(step)} failed{_tail(output)}. By hand: {shlex.join(command)}")


async def _run(command: Sequence[str], say: Say | None = None, note: str = "") -> tuple[int, str]:
    """Run `command` to its end and return its exit code and output. With `say`, says `note` once it has run for
    QUIET_S. A cancelled run (the TUI quit meanwhile) kills the command."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except OSError as error:
        raise InstallFailed(f"{shlex.join(command)} could not start: {error}") from None
    finished = asyncio.ensure_future(process.communicate())
    try:
        done, _ = await asyncio.wait({finished}, timeout=QUIET_S if say is not None else None)
        if not done and say is not None:
            await say(note)
        output, _ = await finished
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            finished.cancel()
    code = process.returncode
    return (code if code is not None else -1), output.decode(errors="replace")


def _tail(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return f": {' | '.join(lines[-OUTPUT_LINES:])}" if lines else ""
