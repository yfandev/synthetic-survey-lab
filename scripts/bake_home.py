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
    speed = m["gemini_mean_ms"] / m["drex_mean_ms"]
    drex_med = statistics.median(r["drex_ms"] for r in rows)
    gem_med = statistics.median(r["gemini_latency_ms"] for r in rows)
    return [
        '    <div class="stats3">',
        '      <div class="stat"><div class="k">Faster</div>'
        f'<div class="v good">{speed:.1f}×</div>'
        f'<div class="num">{m["drex_mean_ms"]} ms vs {round(m["gemini_mean_ms"])} ms per answer<br>'
        f"median {drex_med:.0f} vs {gem_med:.0f} ms</div></div>",
        '      <div class="stat"><div class="k">Cost</div>'
        f'<div class="v good">${m["drex_cost_per_1k"]:.4f}</div>'
        "<div class=\"num\">per 1,000 answers at Nace's list price. Gemini ran on a free tier, "
        "so there is no frontier price to compare.</div></div>",
        '      <div class="stat"><div class="k">Accuracy</div>'
        f'<div class="v bad">{m["drex_mean_distance"]:.3f}</div>'
        f'<div class="num">{m["drex_mean_distance"] * 100:.0f}% of answers on the wrong option, against '
        f"{m['gemini_mean_distance'] * 100:.0f}% for Gemini and {m['whole_mean_distance'] * 100:.0f}% for the "
        "national average.</div></div>",
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
        out.append(f'      <div class="xl"><b>Drex said</b> {drex_txt}</div>')
        out.append(f'      <div class="xl"><b>Real answer</b> {real_txt}</div>')
        out.append(
            '      <div class="xl" style="color:var(--warn)">'
            f"Drex puts {ex['distance']['drex'] * 100:.0f}% of the answers on the wrong option · "
            f"Gemini {ex['gemini_distance'] * 100:.0f}% · national average "
            f"{ex['distance']['whole_sample'] * 100:.0f}%</div>"
        )
        out.append("    </div>")

    out.append(
        '    <p class="num" style="margin-top:12px">Three ways to answer the same question, scored by the same '
        "ruler: Drex's prediction, Gemini's prediction, and the all-respondents answer. The real answer for "
        "that group is the referee.</p>"
    )
    out.append('    <div class="tablewrap">')
    out.append(
        '    <table><thead><tr><th>Question</th><th>Group</th><th>Drex</th><th>Gemini</th>'
        "<th>National avg</th></tr></thead><tbody>"
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
        "is the same question answered by all 3,986 respondents together, weighted, and applied to every "
        "group as one answer.</p>"
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
