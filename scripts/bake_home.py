"""Bake the recorded results into the home page as plain HTML.

The first version drew the headline stats, the worked example and the per-question
table with JavaScript from data/scoreboard.json. A reader with scripts blocked — a
crawler, a locked-down phone, a reviewer fetching the page — saw an empty box. This
script writes the same markup into public/index.html between the BAKE markers so
the numbers are in the HTML a judge actually receives.

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
START = "    <!--BAKE:START-->"
END = "    <!--BAKE:END-->"


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def main() -> None:
    data = json.loads(SCORES.read_text())
    m = data["summary"]
    rows = [
        r
        for r in data["rows"]
        if r.get("distance")
        and r["distance"].get("drex") is not None
        and not r.get("unverified")
        and not r.get("too_few")
    ]
    speed = m["gemini_mean_ms"] / m["drex_mean_ms"]
    drex_med = statistics.median(r["drex_ms"] for r in rows)
    gem_med = statistics.median(r["gemini_latency_ms"] for r in rows)

    parts: list[str] = []
    parts.append(START)
    parts.append('    <div class="stats3">')
    parts.append(
        '      <div class="stat"><div class="k">Faster</div>'
        f'<div class="v good">{speed:.1f}×</div>'
        f'<div class="num">{m["drex_mean_ms"]} ms vs {round(m["gemini_mean_ms"])} ms mean per answer'
        f" · median {drex_med:.0f} ms vs {gem_med:.0f} ms</div></div>"
    )
    parts.append(
        '      <div class="stat"><div class="k">Cost</div>'
        f'<div class="v good">${m["drex_cost_per_1k"]:.4f}</div>'
        "<div class=\"num\">per 1,000 answers at Nace's list price. Gemini ran on a free tier, "
        "so we have no frontier price to compare.</div></div>"
    )
    parts.append(
        '      <div class="stat"><div class="k">Accuracy — Drex loses</div>'
        f'<div class="v bad">{m["drex_mean_distance"]:.3f}</div>'
        f'<div class="num">Gemini {m["gemini_mean_distance"]:.3f} (saved answers) '
        f'· national average {m["whole_mean_distance"]:.3f}</div></div>'
    )
    parts.append("    </div>")

    parts.append(
        '    <p class="lead" style="margin-top:12px;font-size:22px">'
        f"Drex is about {speed:.1f}× faster than the frontier model. It is also <b>less accurate</b> than "
        "that model and than the simple national average, and we are showing that instead of hiding it.</p>"
    )

    ex = next((r for r in rows if r["variable"] == "happy" and r["group"] == "Democrats"), None)
    if ex:
        drex_txt = " · ".join(f"{esc(p['option'])} {p['p'] * 100:.0f}%" for p in ex["drex_probs"])
        real_txt = " · ".join(f"{esc(o['option'])} {o['p'] * 100:.0f}%" for o in ex["real"])
        parts.append('    <div style="margin-top:14px;border:1px solid var(--line);border-radius:10px;padding:14px 16px">')
        parts.append(
            '      <div style="font-size:17px;color:var(--muted)">Worked example · Democrats, '
            "“taken all together, how happy are you these days?”</div>"
        )
        parts.append(f'      <div style="font-size:20px;margin-top:8px"><b>Drex said</b> {drex_txt}</div>')
        parts.append(f'      <div style="font-size:20px"><b>Real answer</b> {real_txt}</div>')
        parts.append(
            '      <div style="font-size:20px;color:var(--warn);margin-top:6px">'
            f"Error {ex['distance']['drex']:.3f} · Gemini {ex['gemini_distance']:.3f} "
            f"· national average {ex['distance']['whole_sample']:.3f}</div>"
        )
        parts.append("    </div>")

    parts.append(
        '    <p class="num" style="margin-top:12px">'
        f"We could check {m['pairs_scored']} questions against real answers. The other {m['unverified']} "
        "could not be checked, because the survey never asked them, so they say unverified rather than guess. "
        "The Gemini answers were collected in one pass, one question at a time, and were never re-asked when "
        "they came out badly.</p>"
    )

    parts.append('    <div style="margin-top:12px">')
    parts.append(
        '      <div style="font-size:17px;color:var(--muted);margin-bottom:6px">Every question we could check. '
        "Error is how far the predicted shares sit from the real ones: the gaps added up, then halved. "
        "0 is a perfect match.</div>"
    )
    parts.append('      <table style="width:100%;border-collapse:collapse;font-size:19px">')
    parts.append(
        '        <thead><tr style="text-align:left;color:var(--muted);font-size:15px">'
        '<th style="padding:4px 8px">Question</th><th>Group</th><th>Drex</th><th>Gemini</th>'
        "<th>National avg</th></tr></thead>"
    )
    parts.append("        <tbody>")
    for r in rows:
        parts.append(
            '          <tr style="border-top:1px solid var(--line)">'
            f'<td style="padding:4px 8px">{esc(r["qid"])}</td><td>{esc(r["group"])}</td>'
            f'<td>{r["distance"]["drex"]:.3f}</td><td>{r["gemini_distance"]:.3f}</td>'
            f'<td>{r["distance"]["whole_sample"]:.3f}</td></tr>'
        )
    parts.append("        </tbody>")
    parts.append("      </table>")
    parts.append("    </div>")

    parts.append(
        '    <p class="num" style="margin-top:12px">Timing was taken per call, end to end, on the same machine, '
        "with Gemini on its free tier — a real measurement, not a benchmark harness.</p>"
    )
    parts.append(END)

    page = PAGE.read_text()
    if START not in page or END not in page:
        raise SystemExit("BAKE markers missing from public/index.html")
    head, rest = page.split(START, 1)
    _, tail = rest.split(END, 1)
    PAGE.write_text(head + "\n".join(parts) + tail)
    print(f"baked {len(rows)} rows into {PAGE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
