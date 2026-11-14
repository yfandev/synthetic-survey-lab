"""Convert the 2024 GSS Stata file into the compact JSON the app reads.

Input : data/gss/2024/GSS2024.dta  (from the official 2024_stata.zip)
Output: data/gss_2024.json

Decisions, all traceable to the GSS documentation:
  * weight      WTSSNRPS_AS — the nonresponse-adjusted post-stratification weight
                created for the 2024 single-year file so all 3,986 cases,
                AmeriSpeak included, can be analysed together ("What's New R3a").
  * missing     any answer label that is a non-answer (don't know, refused,
                no answer, not applicable, can't choose). Never imputed.
  * variables   every attitude item with 2-9 answer options and at least 200
                valid answers, excluding the demographic and design fields.
  * ordinal     flagged when the option labels form a recognisable ordered scale.

    python scripts/prep_gss.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent.parent
DTA = ROOT / "data" / "gss" / "2024" / "GSS2024.dta"
OUT = ROOT / "data" / "gss_2024.json"

WEIGHT = "WTSSNRPS_AS"
FALLBACK_WEIGHT = "WTSSPS"

NON_ANSWER = re.compile(
    r"(don'?t know|do not know|refused|no answer|not applicable|can'?t choose|cannot choose|"
    r"no opinion|skipped|not asked|missing|iap|dk$|^na$|^n/a$)",
    re.I,
)

# Fields that describe the survey or the respondent, never an attitude.
EXCLUDE_PREFIX = ("wt", "vstrat", "vpsu", "ballot", "version", "id", "year", "mode", "sample")
EXCLUDE_EXACT = {
    "age", "educ", "sex", "partyid", "region", "race", "hispanic", "income", "income16",
    "marital", "wrkstat", "occ", "prestg10", "prestg80", "spdeg", "spwrksta", "spocc",
    "agekdbrn", "cohort", "birthyr", "born", "parborn", "granborn", "sibs", "childs",
    "hompop", "babies", "adults", "unrelat", "earnrs", "earnhus", "rincome", "rincom16",
    "finalter", "finrela", "fund", "natfund", "acquaint", "attnfix", "weight",
}

# Survey design and administration, never an attitude item.
ADMIN_LABEL = re.compile(
    r"(sample indicator|eligible|split questionnaire|is r a visitor|interviews conducted|"
    r"baseline|version|release|asked only|asked of|not asked|experiment|random assignment|"
    r"respondent id|interviewer|weight|mode of|questionnaire form|language of interview)",
    re.I,
)

ORDER_WORDS = [
    "strongly", "very", "extremely", "somewhat", "slightly", "moderately",
    "agree", "disagree", "always", "often", "sometimes", "never", "hardly",
    "too much", "too little", "about right", "satisfied", "dissatisfied",
    "better", "worse", "more", "less", "increase", "decrease", "same",
    "excellent", "good", "fair", "poor", "none", "a great deal", "only some",
    "most", "all", "few", "many", "high", "low", "trust",
]


def _clean(label: str) -> str:
    return re.sub(r"\s+", " ", str(label)).strip().rstrip(".")


def build_groups(df) -> tuple[list[list], dict]:
    lookup = {c.lower(): c for c in df.columns}

    def resolve(name):
        return lookup.get(name.lower())

    def num(name):
        actual = resolve(name)
        if actual is None:
            raise KeyError(f"no column {name}; weight columns present: "
                           f"{[c for c in df.columns if c.lower().startswith('wt')]}")
        return pd.to_numeric(df[actual], errors="coerce")

    age = num("age")
    educ = num("educ")
    party = num("partyid")
    sex = num("sex")
    weight = num(WEIGHT) if resolve(WEIGHT) else num(FALLBACK_WEIGHT)
    print(f"weight column: {resolve(WEIGHT) or resolve(FALLBACK_WEIGHT)}")

    def band_age(a):
        if a != a:
            return 255
        if a <= 29:
            return 0
        if a <= 44:
            return 1
        if a <= 64:
            return 2
        return 3

    def band_educ(e):
        if e != e:
            return 255
        if e < 12:
            return 0
        if e == 12:
            return 1
        if e < 16:
            return 2
        return 3

    def band_party(p):
        if p != p or p > 6:
            return 255
        if p <= 2:
            return 0
        if p <= 4:
            return 1
        return 2

    def band_sex(s):
        if s != s or s not in (1.0, 2.0):
            return 255
        return int(s) - 1

    rows = []
    for i in range(len(df)):
        rows.append(
            [
                band_age(age.iloc[i]),
                band_educ(educ.iloc[i]),
                band_party(party.iloc[i]),
                band_sex(sex.iloc[i]),
                round(float(weight.iloc[i]), 4) if weight.iloc[i] == weight.iloc[i] else 0.0,
            ]
        )

    groups = {
        "age_group": {
            "label": "Age",
            "options": ["18-29", "30-44", "45-64", "65+"],
            "field": 0,
        },
        "education": {
            "label": "Education",
            "options": ["less than high school", "high school", "some college", "bachelor's or higher"],
            "field": 1,
        },
        "party_id": {
            "label": "Party identification",
            "options": ["Democrat", "Independent", "Republican"],
            "field": 2,
        },
        "sex": {"label": "Sex", "options": ["male", "female"], "field": 3},
    }
    return rows, groups


def attitude_variables(df, meta) -> list[dict]:
    value_labels = meta.variable_value_labels or {}
    out: list[dict] = []

    for name in df.columns:
        lower = str(name).lower()
        if lower in EXCLUDE_EXACT or lower.startswith(EXCLUDE_PREFIX):
            continue
        labels = value_labels.get(name)
        if not labels:
            continue
        label_text = _clean(meta.column_names_to_labels.get(name, ""))
        if ADMIN_LABEL.search(label_text) or ADMIN_LABEL.search(name):
            continue
        options = [_clean(v) for k, v in _sorted_labels(labels) if not NON_ANSWER.search(_clean(v))]
        if not (2 <= len(options) <= 9):
            continue

        raw = pd.to_numeric(df[name], errors="coerce")
        if int(raw.notna().sum()) < 200:
            continue

        code_to_index = {}
        ordered = []
        for code, label in _sorted_labels(labels):
            text = _clean(label)
            if NON_ANSWER.search(text):
                continue
            try:
                idx = int(round(float(code)))
            except (TypeError, ValueError):
                continue  # lettered codes (d, i, j, m, n) are non-answers
            code_to_index[idx] = len(ordered)
            ordered.append(text)
        if len(ordered) < 2:
            continue

        # A code that appears in the data but carries no label (common when a
        # release ships a trimmed label set) cannot be placed on the answer
        # scale, and silently dropping those answers would distort every share.
        numeric_present = {}
        for v in raw:
            if v != v:
                continue
            f = float(v)
            if f == int(f):
                numeric_present[int(f)] = numeric_present.get(int(f), 0) + 1
        total_rows = int(raw.notna().sum()) or 1
        unlabeled_share = sum(c for code, c in numeric_present.items() if code not in code_to_index) / total_rows
        if unlabeled_share > 0.02:
            continue

        values = [
            code_to_index.get(int(round(float(v))), 255) if v == v else 255
            for v in raw
        ]

        ordinal = _looks_ordinal(ordered)
        out.append(
            {
                "name": name,
                "label": _clean(meta.column_names_to_labels.get(name, name)),
                "options": ordered,
                "ordinal": ordinal,
                "values": values,
                "n": int(sum(1 for v in values if v != 255)),
            }
        )

    out.sort(key=lambda v: -v["n"])
    return out


def _sorted_labels(labels: dict) -> list[tuple]:
    """pyreadstat may key labels by int or by string; sort numerically."""
    def key(kv):
        try:
            return (0, float(kv[0]))
        except (TypeError, ValueError):
            return (1, str(kv[0]))
    return sorted(labels.items(), key=key)


def _looks_ordinal(options: list[str]) -> bool:
    if len(options) < 3:
        return False
    hits = sum(1 for o in options if any(w in o.lower() for w in ORDER_WORDS))
    if hits >= max(2, len(options) - 1):
        return True
    # a bare numeric ladder, e.g. "1 Strong Democrat" ... "7 Strong Republican"
    return all(re.match(r"^\d+\b", o) for o in options)


def main() -> None:
    # codes, not formatted strings: "89 or older" is not a number
    df, meta = pyreadstat.read_dta(str(DTA), apply_value_formats=False)
    rows, groups = build_groups(df)
    variables = attitude_variables(df, meta)

    payload = {
        "source": "General Social Survey (GSS) 2024, NORC at the University of Chicago",
        "file": "GSS2024.dta (2024_stata.zip, R3a)",
        "weight": (meta.column_names[list(c.lower() for c in meta.column_names).index(WEIGHT.lower())]
                   if WEIGHT.lower() in [c.lower() for c in meta.column_names] else WEIGHT),
        "n_respondents": len(rows),
        "groups": groups,
        "rows": rows,
        "variables": variables,
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    size_mb = OUT.stat().st_size / 1e6
    print(f"respondents: {len(rows)} | variables: {len(variables)} | {size_mb:.2f} MB -> {OUT}")
    print(f"weight: {WEIGHT} (min {min(r[4] for r in rows):.3f}, max {max(r[4] for r in rows):.3f})")

    for g in groups:
        field = groups[g]["field"]
        counts = {}
        for r in rows:
            counts[r[field]] = counts.get(r[field], 0) + 1
        named = {groups[g]["options"][k] if k != 255 else "missing": v for k, v in sorted(counts.items())}
        print(f"  {g}: {named}")


if __name__ == "__main__":
    main()
