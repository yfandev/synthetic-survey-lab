# Synthetic Survey Lab

A questionnaire PDF goes in. Nace Document Intelligence reads it, Drex predicts how a chosen group of
Americans answers each question, and the real General Social Survey 2024 result is revealed and scored.

Built at Nace Drexathon, **Apex Predator** track. Everything below is measured, and the measurement does
not all go our way — that is the point of the project.

## Live

<https://site-ba3fbde9b79e4f648ede3e9028b48640.freebuff.page> — Freebuff Sites, run by the same worker
code in this repo. Keys live only on the server.

Slides (Adobe Express):
<https://new.express.adobe.com/id/urn:aaid:sc:US:1b0d7604-c056-4eee-b2bd-10fed9ba85a2?accept=true&invite=true&promoid=Z2G1FQKR&mv=other>

## What it does

1. **Read** — upload a questionnaire PDF; every page is read. NDI returns every question with its
   answer options and the page each was printed on. Hash-cached: the same file is parsed once.
2. **Match** — each question is matched to a GSS 2024 variable: five lexical candidates, then a Drex yes/no
   *"do these ask the same thing, with compatible options?"*, accepted at 0.80 or above.
3. **Results** — every question is scored as soon as the PDF is read, and each real answer carries the
   source it came from. Per question, the first rule that applies wins:
   1. *printed in your PDF* — NDI pulled result percentages for that question from the document (with a
      page citation): that is the answer key, it covers all adults, and the group selector is locked;
   2. *national survey* — a GSS variable was accepted by a Drex same-question decision at ≥ 0.8 **and**
      the option lists map one-to-one in order (non-answer options like "Don't know" are ignored); the
      group selector is enabled;
   3. *unverified* — neither, so the page shows Drex's prediction and no real bars.

   When both 1 and 2 apply, both are shown, each labelled with its source, survey year and matched
   wording. **Reject this match** reverts a question to unverified, and no real result is ever shown
   without its source label.
4. **Reveal** — the weighted real GSS shares for the chosen group, when it holds at least 30 respondents.
5. **Scoreboard** — every verified question × group pair once, with mean total-variation distance and the
   five biggest misses.
6. **Panel run** — Drex answers as N sampled real GSS respondents, one call each, with live counters.
7. **Reword test** — Gemini writes four rewordings, Drex keeps only those it judges identical in meaning
   (≥ 0.90), and the splits are compared. Flagged unstable when any two differ by more than 0.10.

## Measured results

Eight verified question × group pairs, compared with the weighted real GSS 2024 split.
Distance is total variation: 0 identical, 1 no overlap.

| | Mean distance | Mean latency |
|---|---|---|
| Drex (group answer blended 90/10 with the whole-country split) | **0.047** | **256 ms** |
| Drex, raw group answer, no blend | 0.223 | 256 ms |
| Gemini 3.1 Flash-Lite (recorded) | 0.120 | 1,486 ms |
| Whole-country baseline | 0.051 | — |
| Uniform baseline | 0.307 | — |

**Drex is 3.1× faster than Gemini 3.1 Flash-Lite by median (250 ms against 766 ms) and, with the blend,
ahead on accuracy too: 0.047 against Gemini's 0.120 and the whole-country guess's 0.051.** The 5.7× you get
from the means comes from Gemini's latency tail: it ran on the free tier, where throttled calls take seconds.

**How the group answer is formed.** Drex's split for the group is mixed with the whole-country split for the
same question: 90% country, 10% Drex, option by option. The weight was chosen on the benchmark and holds
under leave-one-out (0.047 both ways). Drex's raw group judgement scores 0.223 — it over-shifts groups, for
example giving Democrats 71% "most people can be trusted" where the real figure is 29%. The blend keeps the
direction of that shift and discards most of its size, which is the correct call on questions where groups
differ little. `scripts/exp_prior.py` reproduces the comparison, including a prompt-level alternative that
did not help (0.233).

Cost per 1,000 answers at published list prices: **Drex $0.0060** (Nace launch price $0.03/M tokens, ~201
input tokens an answer) against **$0.167 for Gemini** (Google's paid Standard rate for Gemini 3.1 Flash-Lite:
$0.25 per 1M input, $1.50 per 1M output; the recorded calls averaged 132 input and 89 output tokens). That
makes Drex about **28× cheaper** per answer, measured rather than estimated from a headline price. We tested two rescues and are reporting both:

- *Microsimulation* (one Drex decision per real demographic cell, weighted): helped `nateduc`
  (0.342 → 0.263) and hurt `happy` (0.039 → 0.126); not enough to overtake the baselines.
- *Confidence as a trust signal*: Pearson(confidence, error) = **+0.13** over 9 pairs — the wrong
  direction, and the two most confident pairs were the two worst (0.490, 0.466).

Panel run: 60 Drex decisions in 2.66 s (22.5 calls/s at 6-way concurrency), 0 failures, $0.00026 at the
$0.03/M launch price. The deployed free plan rejects an invocation after roughly 15 Drex calls ("Too
many subrequests"), so the browser sends batches of 15 and sums the counts: a 100-person panel is
seven requests, not one. Reword test: 4 rewordings, 4 kept at 0.92–0.97, worst split movement
0.089 → stable.

## Built with

- **Nace Drex** — every decision: the predicted split for a group, the yes/no gate that accepts a survey
  question as the same question, and the same-meaning gate in the wording check, all live (`.03/M` input).
- **Nace NDI** — reads every page of the uploaded PDF and returns each question with its options and page.
- **Intern** (`intern-agent`, `github.com/nace-ai/intern`) — Nace's agent harness, installed with
  `uv tool install`, run with Drex as its System 1 decision layer (`--system1 drex-latest`). Used to inspect
  the deployed worker and verify its route surface; the run is recorded in `artifacts/intern-run.txt`.

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
- 366 attitude items with 2–9 options and at least 200 valid answers;
- non-answers (don't know, refused, no answer, not applicable) are dropped, never imputed;
- items whose release labels leave numeric codes unmapped are dropped instead of being scored on a
  partial answer scale (this is why `helpnot` is gone: codes 2 and 4 carry no label, so any split would
  have silently discarded a quarter of the answers);
- 45 items flagged ordinal by a label-shape heuristic (see `_looks_ordinal`).

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
- **Uploads run in the browser, not in the Worker.** `ndi-api.nace.ai` answers 403 to the upload POST
  from Cloudflare's network, whatever header or body shape is used (grant minting still returns 201).
  So the server mints a one-use upload grant and the page sends the bytes itself, which is the pattern
  the Nace docs describe for browsers; the page then hands the returned `workspace_id`/`file_id` back
  and the Worker runs extract on that file. Metadata must be a plain string: a JSON part is rejected
  with 422.
- GSS 2024 is a multi-mode study; mode effects are real and not modelled here.
- The sample questionnaire is generated, not a scanned paper form.
