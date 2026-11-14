# Synthetic Survey Lab

A questionnaire PDF goes in. Nace Document Intelligence reads it, Drex predicts how a chosen group of
Americans answers each question, and the real General Social Survey 2024 result is revealed and scored.

Built at Nace Drexathon, **Apex Predator** track. Everything below is measured, and the measurement does
not all go our way — that is the point of the project.

## Live

<https://site-9744494842a94abab91693c343f61169.freebuff.page> — Freebuff Sites, run by the same worker
code in this repo. Keys live only on the server.

## What it does

1. **Read** — upload a questionnaire PDF (optionally a page range). NDI returns every question with its
   answer options and the page each was printed on. Hash-cached: the same file is parsed once.
2. **Match** — each question is matched to a GSS 2024 variable: five lexical candidates, then a Drex yes/no
   *"do these ask the same thing, with compatible options?"*, accepted at 0.80 or above.
3. **Predict** — Drex answers with a probability for every option. Two baselines are always shown beside it:
   the whole-sample split and a uniform split.
4. **Reveal** — the weighted real GSS shares for that group, when it holds at least 30 respondents.
5. **Scoreboard** — every verified question × group pair once, with mean total-variation distance and the
   five biggest misses.
6. **Panel run** — Drex answers as N sampled real GSS respondents, one call each, with live counters.
7. **Reword test** — Gemini writes four rewordings, Drex keeps only those it judges identical in meaning
   (≥ 0.90), and the splits are compared. Flagged unstable when any two differ by more than 0.10.

## Measured results

Eleven verified question × group pairs, compared with the weighted real GSS 2024 split.
Distance is total variation: 0 identical, 1 no overlap.

| | Mean distance | Mean latency |
|---|---|---|
| Drex | **0.267** | **275 ms** |
| Gemini 3.1 Flash-Lite (recorded) | 0.180 | 1,304 ms |
| Whole-sample baseline | **0.073** | — |
| Uniform baseline | 0.279 | — |

**Drex is 4.7× faster than the recorded frontier model and loses on accuracy to both it and the naive
national average.** We tested two rescues and are reporting both:

- *Microsimulation* (one Drex decision per real demographic cell, weighted): helped `nateduc`
  (0.342 → 0.263) and hurt `happy` (0.039 → 0.126); not enough to overtake the baselines.
- *Confidence as a trust signal*: Pearson(confidence, error) = **+0.27** over 11 pairs — the wrong
  direction, and the two most confident pairs were the two worst (0.490, 0.466).

Panel run: 60 Drex decisions in 2.66 s (22.5 calls/s at 6-way concurrency), 0 failures, $0.00026 at the
$0.03/M launch price. The deployed free plan rejects an invocation after roughly 15 Drex calls ("Too
many subrequests"), so the browser sends batches of 15 and sums the counts: a 100-person panel is
seven requests, not one. Reword test: 4 rewordings, 4 kept at 0.92–0.97, worst split movement
0.089 → stable.

## Where each API is called

| Call | Where | Live or recorded |
|---|---|---|
| NDI parse / extract | `worker/index.js` `/api/extract` | live |
| Drex matching, prediction, panel | `worker/index.js` `/api/match`, `/api/predict`, `/api/panel`, `/api/panel-run` | live |
| Drex same-meaning gate, reword splits | `/api/reword` | live |
| Gemini rewordings | `/api/reword` (free tier, cached per request) | live |
| Gemini baseline predictions | `scripts/record_gemini.py` → `public/data/gemini_cache.json` | **recorded**, labelled in the UI |

The Gemini baseline is never called live by the app. `scripts/record_gemini.py` makes each call once,
sequentially, six seconds apart, with exponential backoff, resumable, and writes the cache the app reads.

## Data

`data/gss/2024/GSS2024.dta` from the official 2024 release archive, converted by `scripts/prep_gss.py` into
`public/data/gss_2024.json`:

- weight **`wtssnrps_as`**, the nonresponse-adjusted post-stratification weight that makes all 3,986 cases
  (AmeriSpeak included) analysable together, per *GSS 2024 — What's New R3a*;
- 413 attitude items with 2–9 options and at least 200 valid answers;
- non-answers (don't know, refused, no answer, not applicable) are dropped, never imputed;
- 46 items flagged ordinal by a label-shape heuristic (see `_looks_ordinal`).

## Run it

```bash
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install -r requirements.txt
python scripts/make_sample_survey.py      # sample questionnaire PDF
python scripts/prep_gss.py                # needs data/gss/2024/GSS2024.dta
node server/dev.mjs                       # http://127.0.0.1:3000
```

`.env` (untracked) holds `NACE_API_KEY` and `GEMINI_API_KEY`. Without the Nace key every endpoint returns
the API's own error rather than fake data.

Deploy: `deploy.json` describes the worker and `public/`; secrets go in through Sites, never in the repo.

## Acceptance checks

- Predicted bars sum to 100% (normalised server-side after every Drex call).
- `tvd([0.5,0.5],[0.7,0.3]) = 0.2` (unit test in `tests/test_distance.py`).
- The recorder never repeats a cached call (`if key in cache: continue`).
- A 100-respondent panel run finishes and the counter equals the calls made (`counters.calls`, summed
  across batches) — never fewer.
- The unstable flag fires above 0.10 (`max_distance` computed over all variant pairs).
- No API key in the browser: every key is read from the server environment.
- A PDF question with no GSS match shows "Unverified prediction".

## Known limitations, stated plainly

- **Group-split prediction is a hard task and Drex does not win it.** Subgroup differences on most GSS items
  are small next to sampling noise, which is why "just use the national average" scores 0.073.
- Four of the eight sample questions had no GSS match: work stress, religious attendance, environment taxes,
  and the party-placement ladder (that last one is the group variable itself). They show as unverified.
- The panel run answers as sampled respondents but aggregates by majority choice, so it under-represents
  options that are nobody's first choice.
- The free Workers plan caps subrequests per invocation, so the panel is batched at 15 decisions per
  request; the browser stitches the batches together.
- GSS 2024 is a multi-mode study; mode effects are real and not modelled here.
- The sample questionnaire is generated, not a scanned paper form.
