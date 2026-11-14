"""Acceptance test: total variation distance.

The spec requires the toy case [0.5, 0.5] vs [0.7, 0.3] to give 0.2. The worker's
implementation is the one under test, so this shells out to it rather than
re-implementing the maths in Python.

    python tests/test_distance.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    ("spec toy case", [0.5, 0.5], [0.7, 0.3], 0.2),
    ("identical", [0.5, 0.5], [0.5, 0.5], 0.0),
    ("disjoint", [1.0, 0.0], [0.0, 1.0], 1.0),
    ("three options", [0.2, 0.3, 0.5], [0.4, 0.3, 0.3], 0.2),
]


def main() -> int:
    failures = 0
    for name, a, b, expected in CASES:
        script = (
            "import('./worker/gss.js').then(m => {"
            f"const a = {json.dumps([{'option': 'o' + str(i), 'p': p} for i, p in enumerate(a)])};"
            f"const b = {json.dumps([{'option': 'o' + str(i), 'p': p} for i, p in enumerate(b)])};"
            "console.log(m.tvd(a, b));});"
        )
        out = subprocess.run(["node", "-e", script], cwd=ROOT, capture_output=True, text=True)
        if out.returncode != 0:
            print(f"FAIL {name}: node error {out.stderr[:200]}")
            failures += 1
            continue
        actual = float(out.stdout.strip())
        ok = abs(actual - expected) < 1e-9
        print(f"{'PASS' if ok else 'FAIL'} {name}: tvd={actual:.10f} expected={expected}")
        failures += 0 if ok else 1
    print("all passed" if not failures else f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
