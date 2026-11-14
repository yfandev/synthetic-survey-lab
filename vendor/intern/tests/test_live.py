"""One real task against a real model. Skipped unless INTERN_LIVE_MODEL names one, e.g.

    INTERN_LIVE_MODEL=fireworks/accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b \\
    FIREWORKS_API_KEY=... uv run pytest tests/test_live.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from intern.events import event_adapter

LIVE_MODEL = os.environ.get("INTERN_LIVE_MODEL")
BUGGY = '''def average(values):
    """Mean of a non-empty list."""
    return sum(values) / len(values) + 1
'''
CHECK = "from stats import average; assert average([2, 4, 6]) == 4; print('ok')"


@pytest.mark.live
@pytest.mark.skipif(not LIVE_MODEL, reason="set INTERN_LIVE_MODEL to run against a real model")
def test_a_real_model_fixes_a_bug_and_reports_it(tmp_path: Path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "stats.py").write_text(BUGGY)
    result = subprocess.run(
        [
            sys.executable, "-m", "intern", "-p", "--json", "--no-session", "--no-context-files",
            "--window", "32768", "--system2", str(LIVE_MODEL), "--",
            f"stats.py has a bug: average([2, 4, 6]) should be 4. Fix it, then run: python -c \"{CHECK}\"",
        ],
        env={**os.environ, "INTERN_HOME": str(tmp_path / "home")},
        cwd=work,
        capture_output=True,
        text=True,
        timeout=600,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-2000:]
    end = event_adapter.validate_json(result.stdout.splitlines()[-1])
    assert end.type == "run_end" and end.reason == "completed"
    check = subprocess.run([sys.executable, "-c", CHECK], cwd=work, capture_output=True, text=True)
    assert check.stdout.strip() == "ok", check.stderr
