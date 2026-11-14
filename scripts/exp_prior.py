"""Offline experiment: does anchoring Drex to the national split help?

For each scored benchmark pair, ask Drex twice:
  A  the prompt the app uses today
  B  the same prompt plus the known whole-country split, with instructions to
     start there and shift only where the evidence for the group is strong

Compares both with the real GSS answer. Nothing here is deployed; it only reads
data/scoreboard.json and data/gss_2024.json and calls the Nace console.

    python scripts/exp_prior.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
SCORES = ROOT / "data" / "scoreboard.json"
GSS = ROOT / "data" / "gss_2024.json"
DEMO = ROOT / "public" / "data" / "demo_questions.json"
BASE = "https://console.nace.ai/v1/systemone"


def load_env() -> dict[str, str]:
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def tvd(a: dict[str, float], b: dict[str, float]) -> float:
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)) / 2


def national(gss: dict, variable: str) -> dict[str, float]:
    var = next(v for v in gss["variables"] if v["name"].lower() == variable.lower())
    totals = [0.0] * len(var["options"])
    weight = 0.0
    for row, value in zip(gss["rows"], var["values"]):
        if value is None or value == 255 or value >= len(totals):
            continue
        w = row[4] or 0.0
        totals[value] += w
        weight += w
    return {opt: totals[i] / weight for i, opt in enumerate(var["options"])}


def ask(client: httpx.Client, key: str, state: str, instructions: str, options: list[str]) -> dict[str, float]:
    criteria = {f"o{i}": opt for i, opt in enumerate(options)}
    res = client.post(
        BASE,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": "drex-latest",
            "state": state,
            "questions": {
                "share": {"type": "choice", "instructions": instructions, "criteria": criteria},
            },
        },
    )
    res.raise_for_status()
    probs = (res.json().get("answers", {}).get("share", {}) or {}).get("probabilities", {}) or {}
    raw = [float(probs.get(f"o{i}", 0.0)) for i in range(len(options))]
    total = sum(raw) or 1.0
    return {opt: raw[i] / total for i, opt in enumerate(options)}


def main() -> None:
    env = load_env()
    key = env["NACE_API_KEY"]
    rows = [r for r in json.loads(SCORES.read_text())["rows"] if r.get("status") == "scored"]
    gss = json.loads(GSS.read_text())
    demo = {q["id"]: q for q in json.loads(DEMO.read_text())["questions"]}

    out = []
    with httpx.Client(timeout=120) as client:
        for r in rows:
            text = demo[r["qid"]]["text"]
            options = [s["option"] for s in r["real"]]
            real = {s["option"]: s["p"] for s in r["real"]}
            prior = national(gss, r["variable"])
            prior_txt = ", ".join(f"{k} {v * 100:.0f}%" for k, v in prior.items())
            persona = r["persona"]

            base_q = (
                f'{persona} is asked: "{text}" '
                "Which answer does this person give? Answer as the share of people like this, not one individual."
            )
            anchored_q = (
                f'{persona} is asked: "{text}" '
                f"The whole country answers this question {prior_txt}. "
                "Start from that national split and adjust it for this group only where you have strong evidence; "
                "answer as the share of people like this."
            )

            a = ask(client, key, persona, base_q, options)
            b = ask(client, key, persona, anchored_q, options)
            out.append(
                {
                    "qid": r["qid"],
                    "group": r["group"],
                    "variable": r["variable"],
                    "recorded": r["distance"]["drex"],
                    "gemini": r.get("gemini_distance"),
                    "A": tvd(a, real),
                    "B": tvd(b, real),
                    "too_few": r.get("too_few", False),
                }
            )
            print(
                f'{r["qid"]:3} {r["group"]:20} recorded {r["distance"]["drex"]:.3f}  '
                f"A {out[-1]['A']:.3f}  B {out[-1]['B']:.3f}  gemini {(r.get('gemini_distance') or 0):.3f}"
            )

    scored = [o for o in out if not o["too_few"]]
    def mean(key: str) -> float:
        return sum(o[key] for o in scored) / len(scored)

    print()
    print(f"{len(scored)} pairs")
    print(f"  recorded app prompt : {mean('recorded'):.3f}")
    print(f"  A rerun (same prompt): {mean('A'):.3f}")
    print(f"  B national anchor    : {mean('B'):.3f}")
    print(f"  Gemini recorded      : {mean('gemini'):.3f}")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "exp-prior.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
