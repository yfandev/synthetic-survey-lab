"""Bake the recorded results into the home page as plain HTML.

Everything here is written into public/index.html at build time, so a reader with
scripts blocked still sees the numbers. Two blocks:

    BAKE   the three headline tiles
    BAKE2  worked example, the eleven checks and the timing note

    python scripts/bake_home.py
"""

from __future__ import annotations

import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "public" / "index.html"
SCORES = ROOT / "data" / "scoreboard.json"
GEMINI_CACHE = ROOT / "public" / "data" / "gemini_cache.json"

# Gemini 3.1 Flash-Lite, paid Standard tier, from Google's published pricing page:
# $0.25 per 1M input tokens, $1.50 per 1M output tokens.
GEMINI_IN = 0.25 / 1e6
GEMINI_OUT = 1.5 / 1e6


def gemini_cost_per_1k(rows: list[dict]) -> float | None:
    """Price the recorded Gemini calls at Google's published paid rate."""
    cache = json.loads(GEMINI_CACHE.read_text())
    costs = []
    for r in rows:
        hit = None
        for key, entry in cache.items():
            if key.startswith(str(r.get("variable")) + "|") and entry.get("qid") == r.get("qid"):
                hit = entry
                break
        if not hit:
            continue
        costs.append(hit["input_tokens"] * GEMINI_IN + hit["output_tokens"] * GEMINI_OUT)
    return (sum(costs) / len(costs)) * 1000 if costs else None

BLOCKS = {
    "BAKE": ("    <!--BAKE:START-->", "    <!--BAKE:END-->"),
    "BAKE2": ("    <!--BAKE2:START-->", "    <!--BAKE2:END-->"),
}

LABELS = {
    "Q1": "Spending on education",
    "Q2": "Happiness",
    "Q4": "Trust in people",
    "Q5": "Government's role",
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def whole_country(variable: str) -> dict[str, float]:
    """The same survey question answered by every respondent, weighted."""
    gss = json.loads((ROOT / "public" / "data" / "gss_2024.json").read_text())
    var = next(v for v in gss["variables"] if v["name"] == variable)
    totals: dict[str, float] = {}
    weight_sum = 0.0
    for i, row in enumerate(gss["rows"]):
        code = var["values"][i]
        if code is None or code == 255 or code >= len(var["options"]):
            continue
        w = row[4] or 0
        totals[var["options"][code]] = totals.get(var["options"][code], 0.0) + w
        weight_sum += w
    if not weight_sum:
        return {}
    # keep the option order the question uses, not the order rows happened to appear
    return {opt: totals[opt] / weight_sum * 100 for opt in var["options"] if opt in totals}


WHOLE_HAPPY = whole_country("happy")


def load():
    data = json.loads(SCORES.read_text())
    rows = [
        r
        for r in data["rows"]
        if r.get("distance")
        and r["distance"].get("drex") is not None
        and not r.get("unverified")
        and not r.get("too_few")
    ]
    return data["summary"], rows


def tiles(m: dict, rows: list[dict]) -> list[str]:
    gem_cost = gemini_cost_per_1k(rows) or 0.0
    drex_med = statistics.median(r["drex_ms"] for r in rows)
    gem_med = statistics.median(r["gemini_latency_ms"] for r in rows)
    speed = gem_med / drex_med          # median ratio: the headline
    mean_speed = m["gemini_mean_ms"] / m["drex_mean_ms"]
    return [
        '    <div class="stats3">',
        '      <div class="stat"><div class="k">Faster</div>'
        f'<div class="v good">{speed:.1f}×</div>'
        f'<div class="num">{drex_med:.0f} ms vs {gem_med:.0f} ms per answer, median<br>'
        f"{mean_speed:.1f}× on averages; Gemini ran on its free tier, whose throttled calls inflate its mean</div></div>",
        '      <div class="stat"><div class="k">Cost</div>'
        f'<div class="v good">${m["drex_cost_per_1k"]:.4f}</div>'
        f"<div class=\"num\">Drex per 1,000 answers, against ${gem_cost:.4f} for Gemini at Google's "
        f"published paid rate ($0.25 in / $1.50 out per 1M tokens): {m['drex_cost_per_1k'] and (gem_cost / m['drex_cost_per_1k']):.0f}× cheaper.</div></div>",
        '      <div class="stat"><div class="k">Accuracy</div>'
        f'<div class="v {"good" if m["drex_mean_distance"] <= m["gemini_mean_distance"] else "bad"}">'
        f'{m["drex_mean_distance"]:.3f}</div>'
        f'<div class="num">Across {m["pairs_scored"]} pairs with a real answer: '
        f'{m["drex_mean_distance"] * 100:.0f}% of answers on the wrong option, against '
        f"{m['gemini_mean_distance'] * 100:.0f}% for Gemini and {m['whole_mean_distance'] * 100:.0f}% for a "
        "straight whole-country split.</div></div>",
        "    </div>",
    ]


def detail(m: dict, rows: list[dict]) -> list[str]:
    out: list[str] = []
    ex = next((r for r in rows if r["variable"] == "happy" and r["group"] == "Democrats"), None)
    if ex:
        drex_txt = " · ".join(f"{esc(p['option'])} {p['p'] * 100:.0f}%" for p in ex["drex_probs"])
        real_txt = " · ".join(f"{esc(o['option'])} {o['p'] * 100:.0f}%" for o in ex["real"])
        out.append('    <div class="ex">')
        out.append(
            '      <div class="xlbl">Worked example · Democrats, '
            "“taken all together, how happy are you these days?”</div>"
        )
        out.append(f'      <div class="xl"><b>Drex said</b> {drex_txt} <span class="num">(group answer, blended 90/10 with the country)</span></div>')
        out.append(f'      <div class="xl"><b>Real answer</b> {real_txt}</div>')
        out.append(
            '      <div class="xl" style="color:var(--warn)">'
            f"Drex puts {ex['distance']['drex'] * 100:.0f}% of the answers on the wrong option "
            f"(raw group answer, before the blend: {ex['distance']['drex_raw'] * 100:.0f}%) · "
            f"Gemini {ex['gemini_distance'] * 100:.0f}% · national average "
            f"{ex['distance']['whole_sample'] * 100:.0f}%</div>"
        )
        out.append("    </div>")

    whole = WHOLE_HAPPY
    if ex and whole:
        r = {o["option"]: o["p"] * 100 for o in ex["real"]}
        w = whole
        fmt = lambda d: " · ".join(f"{esc(k)} {v:.0f}%" for k, v in d.items())
        out.append(
            '    <p class="num" style="margin-top:12px">Worked through: the Democrats\' real answer is '
            f"{fmt(r)}. The whole-country answer to the same survey question, over all 3,986 respondents, is "
            f"{fmt(w)}; using that for Democrats is the whole-country guess column. The real answer is what "
            "both predictions are scored against.</p>"
        )
    out.append('    <div class="tablewrap">')
    out.append(
        '    <table><thead><tr><th>Question</th><th>Group</th><th>Drex</th><th>Gemini</th>'
        "<th>Whole-country guess</th></tr></thead><tbody>"
    )
    for r in rows:
        out.append(
            "      <tr>"
            f'<td>{esc(LABELS.get(r["qid"], r["qid"]))} ({esc(r["qid"])})</td>'
            f'<td>{esc(r["group"])}</td>'
            f'<td>{r["distance"]["drex"]:.3f}</td><td>{r["gemini_distance"]:.3f}</td>'
            f'<td>{r["distance"]["whole_sample"]:.3f}</td></tr>'
        )
    out.append("    </tbody></table>")
    out.append("    </div>")

    out.append(
        '    <p class="num" style="margin-top:14px">'
        f"{m['pairs_scored']} pairs have a real survey answer to score against. The other "
        f"{m['unverified']} are new questions: the survey has not asked them yet, so we mark them new and "
        "leave them unscored. Error is the share of answers sitting on the wrong option: 0 means every "
        "answer matches the real survey, 0.27 means 27% of them are misplaced. The national average column "
        "is the same survey asked of all 3,986 respondents, weighted, then used as the answer for every "
        "group.</p>"
    )
    out.append(
        '    <p class="num">Four survey questions — spending on education (Q1), happiness (Q2), trust in '
        "people (Q4), and whether government should do more (Q5) — asked of three groups: 18–29 with a "
        "bachelor's degree, people 65 and over, and Democrats. That is 12 combinations; the trust question "
        "for 18–29 is dropped because only 14 respondents fall in that cell, which is too few to call a "
        "real answer.</p>"
    )
    out.append(
        '    <p class="num">Each call was timed end to end on the same machine, with Gemini on its free '
        "tier: a real measurement of both models side by side. The Gemini answers come from one pass, one "
        "question at a time, saved as they arrived.</p>"
    )
    return out


def inject(page: str, key: str, body: list[str]) -> str:
    start, end = BLOCKS[key]
    if start not in page or end not in page:
        raise SystemExit(f"{key} markers missing from public/index.html")
    head, rest = page.split(start, 1)
    _, tail = rest.split(end, 1)
    return head + start + "\n" + "\n".join(body) + "\n" + end + tail


def main() -> None:
    m, rows = load()
    page = PAGE.read_text()
    page = inject(page, "BAKE", tiles(m, rows))
    page = inject(page, "BAKE2", detail(m, rows))
    PAGE.write_text(page)
    print(f"baked {len(rows)} rows into {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
