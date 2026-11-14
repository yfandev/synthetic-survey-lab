"""Run every verified question x group pair once and save the scoreboard.

    python scripts/score_all.py            # against the local dev server
    python scripts/score_all.py --url https://...freebuff.page

Writes data/scoreboard.json and prints the aggregate the deck quotes.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "public" / "data" / "demo_questions.json"
OUT = ROOT / "data" / "scoreboard.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:3000")
    opts = ap.parse_args()

    demo = json.loads(DEMO.read_text())
    rows = []

    with httpx.Client(timeout=120) as client:
        for q in demo["questions"]:
            match = client.post(f"{opts.url}/api/match", json={"text": q["text"], "options": q["options"]}).json()
            if match.get("error"):
                print(f"{q['id']}: match error {match['error']}")
                continue
            if not match["accepted"]:
                best = (match.get("candidates") or [{}])[0]
                print(f"{q['id']}: UNVERIFIED (best {best.get('label')} @ {best.get('same_thing')})")
                rows.append({"qid": q["id"], "status": "unverified", "best": best})
                continue

            var = match["accepted"]
            for group in demo["groups"]:
                p = client.post(
                    f"{opts.url}/api/predict",
                    json={"variable": var["variable"], "text": q["text"], "selection": group["selection"]},
                ).json()
                if p.get("error"):
                    print(f"{q['id']} / {group['label']}: {p['error']}")
                    continue
                g = p.get("gemini") or {}
                rows.append(
                    {
                        "qid": q["id"],
                        "status": "scored",
                        "group": group["label"],
                        "variable": p["variable"],
                        "match": round(var["same_thing"], 3),
                        "persona": p["persona"],
                        "n": p["real"]["n"],
                        "too_few": p["too_few"],
                        "distance": p["distance"],
                        "gemini_distance": g.get("distance_to_real"),
                        "gemini_model": g.get("model"),
                        "gemini_latency_ms": g.get("latency_ms"),
                        "drex_ms": p["drex"]["wall_ms"],
                        "drex_tokens": p["drex"]["input_tokens"],
                        "drex_cost_usd": p["drex"]["cost_usd"],
                        "drex_probs": p["drex"]["probabilities"],
                        "real": p["real"]["shares"],
                    }
                )
                print(
                    f"{q['id']:3} {group['label']:18} {p['variable']:9} n={p['real']['n']:4} "
                    f"drex={p['distance']['drex']:.3f} gemini={g.get('distance_to_real') and round(g['distance_to_real'], 3)} "
                    f"whole={p['distance']['whole_sample']:.3f} uniform={p['distance']['uniform']:.3f} "
                    f"{p['drex']['wall_ms']}ms"
                )

    scored = [r for r in rows if r["status"] == "scored" and not r["too_few"]]
    def mean(key):
        vals = [r[key] for r in scored if isinstance(r.get(key), (int, float))]
        return round(statistics.mean(vals), 4) if vals else None

    summary = {
        "pairs_scored": len(scored),
        "unverified": sum(1 for r in rows if r["status"] == "unverified"),
        "drex_mean_distance": mean("distance") if False else round(statistics.mean([r["distance"]["drex"] for r in scored]), 4) if scored else None,
        "gemini_mean_distance": round(statistics.mean([r["gemini_distance"] for r in scored if r["gemini_distance"] is not None]), 4) if any(r["gemini_distance"] is not None for r in scored) else None,
        "whole_mean_distance": round(statistics.mean([r["distance"]["whole_sample"] for r in scored]), 4) if scored else None,
        "uniform_mean_distance": round(statistics.mean([r["distance"]["uniform"] for r in scored]), 4) if scored else None,
        "drex_mean_ms": round(statistics.mean([r["drex_ms"] for r in scored]), 1) if scored else None,
        "gemini_mean_ms": round(statistics.mean([r["gemini_latency_ms"] for r in scored if r["gemini_latency_ms"]]), 1) if scored else None,
        "drex_mean_tokens": round(statistics.mean([r["drex_tokens"] for r in scored]), 1) if scored else None,
        "drex_cost_per_1k": round(statistics.mean([r["drex_cost_usd"] for r in scored]) * 1000, 4) if scored else None,
        "biggest_misses": sorted(scored, key=lambda r: -r["distance"]["drex"])[:5],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n== summary ==")
    for k, v in summary.items():
        if k != "biggest_misses":
            print(f"  {k}: {v}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
