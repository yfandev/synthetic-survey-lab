"""Record the Gemini baseline once, sequentially, and keep it on disk.

The app never calls Gemini. It reads public/data/gemini_cache.json, produced
here, and labels every Gemini number on screen "recorded".

    python scripts/record_gemini.py                     # record what is missing
    python scripts/record_gemini.py --only Q2           # one question
    python scripts/record_gemini.py --dry-run           # list pairs

Resumable: a pair already in the cache is never called again.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "public" / "data" / "demo_questions.json"
CACHE = ROOT / "public" / "data" / "gemini_cache.json"
APP = "http://127.0.0.1:3000"

# Both models the brief named are retired for new API keys (the API answers 404),
# so this is the live free-tier flash-lite chain, in preference order.
MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.8-flash"]

GAP_SECONDS = 6.0
MAX_RETRIES = 5


def load_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("GEMINI_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("GEMINI_API_KEY missing from .env")


PROMPT = (
    "You are simulating survey respondents. {persona}\n\n"
    "Question: \"{question}\"\n"
    "Answer options, in order: {options}\n\n"
    "Estimate the share of people like this who choose each option. "
    "Reply with JSON only: {{\"probabilities\": [{{\"option\": \"<exact option text>\", \"p\": <number 0-1>}}, ...]}} "
    "with one entry per option, in the order given, and the p values summing to 1."
)


def call_gemini(key: str, prompt: str) -> dict:
    """One call, with exponential backoff on 429/5xx. Returns the parsed answer."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        for model in MODELS:
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=90) as client:
                    resp = client.post(
                        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                        headers={"x-goog-api-key": key, "content-type": "application/json"},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
                        },
                    )
            except Exception as exc:  # network
                last_error = f"{type(exc).__name__}: {exc}"
                continue
            latency_ms = int((time.perf_counter() - started) * 1000)

            if resp.status_code == 200:
                body = resp.json()
                text = body["candidates"][0]["content"]["parts"][0]["text"]
                usage = body.get("usageMetadata", {})
                return {
                    "ok": True,
                    "model": model,
                    "latency_ms": latency_ms,
                    "input_tokens": usage.get("promptTokenCount"),
                    "output_tokens": usage.get("candidatesTokenCount"),
                    "cost_usd": 0.0,  # free tier
                    "raw": text[:600],
                    "parsed": json.loads(text),
                }

            last_error = f"{resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (429, 500, 503):
                time.sleep(GAP_SECONDS * (2**attempt))
                continue
            time.sleep(2)

    return {"ok": False, "error": last_error}


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--only", default=None, help="question id, e.g. Q2")
    args.add_argument("--dry-run", action="store_true")
    args.add_argument("--limit", type=int, default=0)
    opts = args.parse_args()

    key = load_key()
    demo = json.loads(DEMO.read_text())
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

    questions = [q for q in demo["questions"] if not opts.only or q["id"] == opts.only]
    pairs = []

    with httpx.Client(timeout=120) as client:
        for q in questions:
            match = client.post(f"{APP}/api/match", json={"text": q["text"], "options": q["options"]}).json()
            accepted = match.get("accepted")
            if not accepted:
                print(f"{q['id']}: no GSS match (best {match['candidates'][0]['same_thing'] if match.get('candidates') else None})")
                continue
            print(f"{q['id']}: {accepted['variable']} @ {accepted['same_thing']:.3f}")
            for group in demo["groups"]:
                pred = client.post(
                    f"{APP}/api/predict",
                    json={"variable": accepted["variable"], "text": q["text"], "selection": group["selection"]},
                ).json()
                if pred.get("error"):
                    print("   predict error:", pred["error"])
                    continue
                pairs.append({"qid": q["id"], "group": group["label"], "pred": pred})

    print(f"\n{len(pairs)} pairs, {sum(1 for p in pairs if p['pred']['gemini_key'] in cache)} already recorded")
    if opts.dry_run:
        for p in pairs:
            print("  ", p["pred"]["gemini_key"], "recorded" if p["pred"]["gemini_key"] in cache else "missing")
        return

    todo = [p for p in pairs if p["pred"]["gemini_key"] not in cache]
    if opts.limit:
        todo = todo[: opts.limit]

    for i, pair in enumerate(todo, start=1):
        pred = pair["pred"]
        options = [o["option"] for o in pred["drex"]["probabilities"]]
        verbose = ", ".join(f'"{o}"' for o in options)
        result = call_gemini(key, PROMPT.format(persona=pred["persona"], question=pair["qid"] + ". " + pred["question"], options=verbose))

        if not result["ok"]:
            print(f"[{i}/{len(todo)}] {pred['gemini_key']} FAILED {result['error']}")
            time.sleep(GAP_SECONDS)
            continue

        entries = result["parsed"].get("probabilities", [])
        by_option = {}
        for e in entries:
            if isinstance(e, dict) and "option" in e:
                by_option[str(e["option"]).strip().lower()] = float(e.get("p", 0) or 0)
        probs = []
        for opt in options:
            probs.append({"option": opt, "p": by_option.get(opt.strip().lower(), 0.0)})
        total = sum(p["p"] for p in probs)
        if total <= 0:
            print(f"[{i}/{len(todo)}] {pred['gemini_key']} unparsed: {result['raw'][:120]}")
            time.sleep(GAP_SECONDS)
            continue
        for p in probs:
            p["p"] = p["p"] / total

        cache[pred["gemini_key"]] = {
            "probabilities": probs,
            "model": result["model"],
            "latency_ms": result["latency_ms"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": 0.0,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "qid": pair["qid"],
            "group": pair["group"],
        }
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache, indent=1))
        print(f"[{i}/{len(todo)}] {pair['qid']} {pair['group']} -> {result['model']} {result['latency_ms']} ms")
        time.sleep(GAP_SECONDS)

    print(f"\ncache: {len(cache)} pairs -> {CACHE}")


if __name__ == "__main__":
    main()
