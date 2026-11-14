/**
 * Synthetic Survey Lab — server.
 *
 * Runs unchanged in two places:
 *   - locally, through server/dev.mjs (Node http adapter)
 *   - on Freebuff Sites / Cloudflare Workers (env.ASSETS serves /public)
 *
 * Every Nace call happens here, so no key ever reaches the browser.
 */

import {
  loadGss,
  split,
  persona as personaOf,
  tvd,
  uniform,
  persona,
  candidates,
  optionsCompatible,
  describeSelection,
} from "./gss.js";

const NACE_BASE = "https://console.nace.ai";

// $0.03 per million input tokens, Drex Max launch price (docs.nace.ai pricing).
const DREX_USD_PER_TOKEN = 0.03 / 1e6;

const CORS = {
  "content-type": "application/json; charset=utf-8",
  "cache-control": "no-store",
};

const json = (data, status = 200) =>
  new Response(JSON.stringify(data, null, 2), { status, headers: CORS });

function naceHeaders(env) {
  if (!env.NACE_API_KEY) throw new Error("NACE_API_KEY is not set on the server");
  return {
    Authorization: `Bearer ${env.NACE_API_KEY}`,
    "Content-Type": "application/json",
  };
}

/* ---------------------------------------------------------------- Drex ---- */

async function drexDecide(env, state, questions, model = "drex-latest") {
  const started = Date.now();
  const res = await fetch(`${NACE_BASE}/v1/systemone`, {
    method: "POST",
    headers: naceHeaders(env),
    body: JSON.stringify({ model, state, questions }),
  });
  const text = await res.text();
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    body = { parse_error: true, raw: text.slice(0, 2000) };
  }
  return {
    http_status: res.status,
    wall_ms: Date.now() - started,
    request_id: res.headers.get("x-request-id"),
    response: body,
  };
}

/* ----------------------------------------------------------------- NDI ---- */

/** Upload one file: grant -> PUT bytes -> workspace_file source. */
async function ndiUpload(env, file, name, keyHint) {
  const path = `survey-lab/${keyHint || Date.now()}-${name}`;
  const grantRes = await fetch(`${NACE_BASE}/v1/documents/upload-grants`, {
    method: "POST",
    headers: naceHeaders(env),
    body: JSON.stringify({ path, ttl_seconds: 900 }),
  });
  const grantText = await grantRes.text();
  if (!grantRes.ok) throw new Error(`upload-grant ${grantRes.status}: ${grantText.slice(0, 400)}`);
  const grant = JSON.parse(grantText);

  // Built by hand instead of with FormData: the upload endpoint rejected the
  // multipart body workerd produced (403) while accepting the identical Node
  // request, and a hand-built body is byte-for-byte the same in both runtimes.
  const boundary = `----surveylab${Math.random().toString(16).slice(2)}`;
  const enc = new TextEncoder();
  const head = enc.encode(
    `--${boundary}\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n` +
      `${JSON.stringify({ path })}\r\n--${boundary}\r\n` +
      `Content-Disposition: form-data; name="file"; filename="${name}"\r\n` +
      `Content-Type: ${file.type || "application/pdf"}\r\n\r\n`
  );
  const tail = enc.encode(`\r\n--${boundary}--\r\n`);
  const bytes = new Uint8Array(await file.arrayBuffer());
  const body = new Uint8Array(head.length + bytes.length + tail.length);
  body.set(head, 0);
  body.set(bytes, head.length);
  body.set(tail, head.length + bytes.length);

  const up = await fetch(grant.upload_url, {
    method: "POST",
    headers: {
      "X-Upload-Token": grant.token,
      "Content-Type": `multipart/form-data; boundary=${boundary}`,
    },
    body: body.buffer,
  });
  const upText = await up.text();
  if (!up.ok) throw new Error(`upload ${up.status}: ${upText.slice(0, 400)}`);
  let job = JSON.parse(upText);

  if (job.status && job.status !== "succeeded") job = await ndiWait(env, job.job_id);

  const fileRef = (job.result && job.result.file) || {};
  if (!fileRef.file_id) throw new Error(`upload returned no file_id: ${upText.slice(0, 400)}`);
  const workspace_id = fileRef.workspace_id || grant.workspace_id;
  const file_id = fileRef.file_id;
  return {
    path,
    // exactly the three keys the API accepts; extra keys are refused with 422
    source: { type: "workspace_file", workspace_id, file_id },
  };
}

async function ndiWait(env, jobId, timeoutMs = 240000) {
  const deadline = Date.now() + timeoutMs;
  while (true) {
    const res = await fetch(`${NACE_BASE}/v1/documents/jobs/${jobId}`, {
      headers: { Authorization: naceHeaders(env).Authorization },
    });
    const job = await res.json();
    if (["succeeded", "failed", "cancelled"].includes(job.status)) {
      if (job.status !== "succeeded") throw new Error(`job ${job.status}: ${JSON.stringify(job.error).slice(0, 400)}`);
      return job;
    }
    if (Date.now() > deadline) throw new Error(`job ${jobId} still ${job.status} after ${timeoutMs}ms`);
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function ndiDocument(env, kind, source, options = {}) {
  const started = Date.now();
  const res = await fetch(`${NACE_BASE}/v1/documents/${kind}?wait_seconds=90`, {
    method: "POST",
    headers: naceHeaders(env),
    body: JSON.stringify({ source, ...options }),
  });
  const text = await res.text();
  let job;
  try {
    job = JSON.parse(text);
  } catch {
    return { http_status: res.status, wall_ms: Date.now() - started, response: { raw: text.slice(0, 2000) } };
  }
  if (!res.ok) return { http_status: res.status, wall_ms: Date.now() - started, response: job };
  if (job.status && !["succeeded", "failed", "cancelled"].includes(job.status)) {
    job = await ndiWait(env, job.job_id);
  }
  return { http_status: res.status, wall_ms: Date.now() - started, response: job };
}

/* ------------------------------------------- recorded Gemini baseline ---- */

// Gemini is never called live by the app. scripts/record_gemini.py writes this
// file once, and every Gemini number on screen is read from it and labelled
// "recorded".
let GEMINI_CACHE = null;

async function loadGeminiCache(request, env) {
  if (GEMINI_CACHE) return GEMINI_CACHE;
  try {
    const url = new URL("/data/gemini_cache.json", request.url).toString();
    const res = env.ASSETS ? await env.ASSETS.fetch(new Request(url)) : await fetch(url);
    GEMINI_CACHE = res.ok ? await res.json() : {};
  } catch {
    GEMINI_CACHE = {};
  }
  return GEMINI_CACHE;
}

function geminiKey(variableName, selection) {
  const parts = Object.keys(selection || {})
    .sort()
    .map((k) => `${k}=${selection[k]}`);
  return `${variableName}|${parts.join(",")}`;
}

/* ------------------------------------------------------- Phase 1: intake --- */

// One parse per file: keyed by content hash + page window. Lives for the life
// of the isolate, which covers a demo session; the deployed build adds D1 later.
const INTAKE_CACHE = new Map();

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

const QUESTION_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["questions"],
  properties: {
    questions: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["id", "text", "options"],
        properties: {
          id: { type: "string", description: "Short id printed in the questionnaire, else Q<number>" },
          text: { type: "string", description: "The question exactly as printed, without its number" },
          options: { type: "array", items: { type: "string" }, description: "Answer options in printed order" },
          reported_result: {
            type: "array",
            description:
              "Only when the document prints result percentages for this question (a results table, a previous survey's answers). One entry per option, in the same order. Leave empty when the document prints no percentages.",
            items: {
              type: "object",
              additionalProperties: false,
              required: ["option", "percent"],
              properties: {
                option: { type: "string" },
                percent: { type: "number", description: "Percent as printed, 0-100" },
              },
            },
          },
        },
      },
    },
  },
};

const QUESTION_INSTRUCTIONS =
  "This is a survey questionnaire. Extract every question that has explicit answer options. " +
  "Keep the question wording exactly as printed, without its number or its options. " +
  "Keep the options in printed order and do not merge or reword them. " +
  "Skip instructions, section headings and interviewer notes.";

function mapQuestions(job) {
  const result = (job && job.result) || {};
  const data = result.data || {};
  const list = Array.isArray(data.questions) ? data.questions : [];
  const pageFor = {};
  for (const field of result.fields || []) {
    const match = /^\/questions\/(\d+)/.exec(field.path || "");
    if (!match) continue;
    const cite = (field.citations || [])[0];
    if (!cite || !cite.location) continue;
    const idx = match[1];
    if (pageFor[idx] === undefined) pageFor[idx] = { page: cite.location.page ?? null, quote: cite.source_text || null };
  }
  return list.map((q, i) => ({
    id: q.id || `Q${i + 1}`,
    text: q.text || "",
    options: Array.isArray(q.options) ? q.options : [],
    page: pageFor[i] ? pageFor[i].page : null,
    quote: pageFor[i] ? pageFor[i].quote : null,
    reported_result: Array.isArray(q.reported_result) ? q.reported_result : [],
  }));
}

/* -------------------------------------------------------------- routes ---- */

const SAMPLE_QUESTIONS = {
  q_funding: {
    type: "choice",
    instructions:
      "A respondent is asked: 'Are we spending too much, too little, or about the right amount on education?' What share picks each option?",
    criteria: { too_little: "Spend too little", about_right: "About right", too_much: "Spend too much" },
  },
  q_happy: {
    type: "choice",
    instructions:
      "A respondent is asked: 'Taken all together, how would you say things are these days?' What share picks each option?",
    criteria: { very_happy: "Very happy", pretty_happy: "Pretty happy", not_too_happy: "Not too happy" },
  },
};

async function handle(request, env) {
  const url = new URL(request.url);
  const route = url.pathname;

  if (route === "/api/health") {
    return json({
      ok: true,
      nace_key: Boolean(env.NACE_API_KEY),
      gemini_key: Boolean(env.GEMINI_API_KEY),
      runtime: env.ASSETS ? "workers" : "node-dev",
    });
  }

  if (route === "/api/upload-grant" && request.method === "POST") {
    // The document service refuses uploads that come from a Worker's address, so the
    // browser uploads the bytes itself with a short-lived one-use token minted here.
    try {
      const body = await request.json().catch(() => ({}));
      const safe = String(body.name || "upload.pdf").replace(/[^\w.\-]/g, "_").slice(-80);
      const path = `survey-lab/${crypto.randomUUID()}-${safe}`;
      const r = await fetch(`${NACE_BASE}/v1/documents/upload-grants`, {
        method: "POST",
        headers: naceHeaders(env),
        body: JSON.stringify({ path, ttl_seconds: 900 }),
      });
      const text = await r.text();
      if (!r.ok) return json({ error: `upload-grant ${r.status}: ${text.slice(0, 300)}` }, 502);
      const g = JSON.parse(text);
      return json({ upload_url: g.upload_url, token: g.token, path, expires_at: g.expires_at });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/resolve" && request.method === "POST") {
    // Result source, per question, first rule that applies:
    //   1. percentages printed in the uploaded PDF (with the group selector locked)
    //   2. a GSS variable, checked by a Drex same-question decision at 0.8 plus
    //      one-to-one option mapping
    //   3. unverified
    try {
      const body = await request.json();
      const text = String(body.text || "");
      const options = Array.isArray(body.options) ? body.options : [];
      const printed = Array.isArray(body.reported_result) ? body.reported_result : [];

      let pdf = null;
      if (printed.length === options.length && options.length > 1) {
        const shares = printed.map((r, i) => ({ option: options[i], p: Number(r.percent) / 100 }));
        const total = shares.reduce((a, b) => a + b.p, 0);
        const sane = shares.every((s) => Number.isFinite(s.p) && s.p >= 0 && s.p <= 1.2);
        if (sane && total > 0.9 && total < 1.1) pdf = { shares, page: body.page ?? null, quote: body.quote ?? null };
      }

      const gss = await loadGss(request, env);
      const top = candidates(gss, text, options, 5);
      const printedName = gss.variables.find((v) => {
        const re = new RegExp(`\\b${v.name.replace(/[^a-z0-9_]/gi, "")}\\b`, "i");
        return re.test(text);
      });
      const pool = printedName ? [{ variable: printedName, score: 1, text: 1, opts: 1 }, ...top.filter((c) => c.variable.name !== printedName)] : top;
      const shortlist = pool.slice(0, 5);

      const questions = {};
      shortlist.forEach((c, i) => {
        questions[`q${i}`] = {
          type: "noul",
          instructions:
            "Do these two survey questions ask the same thing, with compatible answer options? " +
            `A: "${text}" with options ${JSON.stringify(options)}. ` +
            `B: "${c.variable.label}" (variable ${c.variable.name}) with options ${JSON.stringify(c.variable.options)}.`,
        };
      });
      const state =
        "Two questions ask the same thing when a respondent would give the same answer to both, " +
        "and the option sets line up one-to-one. Different wording of the same idea counts as the same. " +
        "A broader or narrower topic does not.";
      const out = await drexDecide(env, state, questions);
      const answers = (out.response && out.response.answers) || {};
      const ranked = shortlist.map((c, i) => {
        const noul = answers[`q${i}`] ? answers[`q${i}`].noul : null;
        return {
          variable: c.variable.name,
          label: c.variable.label,
          options: c.variable.options,
          same_thing: noul,
          options_compatible: optionsCompatible({ options }, c.variable),
          named_in_text: printedName ? c.variable.name === printedName.name : false,
        };
      });
      const accepted =
        ranked.find((r) => r.options_compatible && r.same_thing != null && r.same_thing >= 0.8) || null;

      return json({
        decision: pdf && accepted ? "both" : pdf ? "pdf" : accepted ? "gss" : "none",
        pdf,
        gss: accepted,
        candidates: ranked,
        drex: { wall_ms: out.wall_ms, request_id: out.request_id },
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/parse" && request.method === "POST") {
    // Parse a file the browser already uploaded, using the ids it handed back.
    try {
      const body = await request.json();
      if (!body.file_id || !body.workspace_id) {
        return json({ error: "send workspace_id and file_id from the upload" }, 400);
      }
      const source = { type: "workspace_file", workspace_id: body.workspace_id, file_id: body.file_id };
      const parsed = await ndiDocument(env, "parse", source, {});
      return json({ upload: { source }, ...parsed });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/extract" && request.method === "POST") {
    // Browser-uploaded file: the bytes already went up with a one-use grant, so this
    // only runs extract on the workspace file the browser handed back.
    if ((request.headers.get("content-type") || "").includes("application/json")) {
      let stage = "read body";
      try {
        const body = await request.json();
        if (!body.file_id || !body.workspace_id) {
          return json({ error: "send workspace_id and file_id from the upload" }, 400);
        }
        const from = Math.max(1, Number(body.from || 1));
        const to = Number(body.to || 0);
        const key = `ws:${body.workspace_id}:${body.file_id}:${from}-${to || "end"}`;
        if (INTAKE_CACHE.has(key)) return json({ cached: true, ...INTAKE_CACHE.get(key) });

        stage = "extract";
        const options = { schema: QUESTION_SCHEMA, instructions: QUESTION_INSTRUCTIONS };
        if (to >= from && to > 0) options.page_ranges = [{ start: from, end: to }];
        const source = { type: "workspace_file", workspace_id: body.workspace_id, file_id: body.file_id };
        const out = await ndiDocument(env, "extract", source, options);
        const job = out.response || {};
        if (job.status !== "succeeded") {
          return json(
            {
              error: `extract ${job.status || out.http_status}`,
              stage,
              job_error: job.error || null,
              http_status: out.http_status,
              request_id: job.request_id || null,
            },
            502,
          );
        }
        const payload = {
          hash: null,
          file: body.name || "questionnaire.pdf",
          page_range: options.page_ranges ? { from, to } : null,
          questions: mapQuestions(job),
          units: (job.result || {}).units ?? null,
          job_id: job.job_id,
          wall_ms: out.wall_ms,
          ndi_ms: (job.timing && job.timing.total_ms) ?? null,
          fields_without_citation: ((job.result || {}).fields || []).filter((f) => !(f.citations || []).length).length,
        };
        INTAKE_CACHE.set(key, payload);
        return json({ cached: false, ...payload });
      } catch (err) {
        return json({ error: String((err && err.message) || err), stage }, 502);
      }
    }

    let stage = "read form";
    try {
      const form = await request.formData();
      const file = form.get("file");
      if (!file || typeof file === "string") return json({ error: "send the PDF as the `file` field" }, 400);
      const name = file.name || "questionnaire.pdf";
      const from = Math.max(1, Number(form.get("from") || 1));
      const to = Number(form.get("to") || 0);
      const bytes = await file.arrayBuffer();
      const hash = await sha256Hex(bytes);
      const key = `${hash}:${from}-${to || "end"}`;

      if (INTAKE_CACHE.has(key)) {
        return json({ cached: true, ...INTAKE_CACHE.get(key) });
      }

      stage = "upload";
      const upload = await ndiUpload(env, new File([bytes], name, { type: "application/pdf" }), name, hash.slice(0, 12));

      stage = "extract";
      const options = {
        schema: QUESTION_SCHEMA,
        instructions: QUESTION_INSTRUCTIONS,
      };
      if (to >= from && to > 0) options.page_ranges = [{ start: from, end: to }];

      const out = await ndiDocument(env, "extract", upload.source, options);
      const job = out.response || {};
      if (job.status !== "succeeded") {
        return json(
          {
            error: `extract ${job.status || out.http_status}`,
            stage,
            job_error: job.error || null,
            http_status: out.http_status,
            wall_ms: out.wall_ms,
            request_id: job.request_id || null,
          },
          502,
        );
      }

      const payload = {
        hash,
        file: name,
        page_range: options.page_ranges ? { from, to } : null,
        questions: mapQuestions(job),
        units: (job.result || {}).units ?? null,
        job_id: job.job_id,
        wall_ms: out.wall_ms,
        ndi_ms: (job.timing && job.timing.total_ms) ?? null,
        fields_without_citation: ((job.result || {}).fields || []).filter((f) => !(f.citations || []).length).length,
      };
      INTAKE_CACHE.set(key, payload);
      return json({ cached: false, ...payload });
    } catch (err) {
      return json({ error: String((err && err.message) || err), stage }, 502);
    }
  }

  if (route === "/api/gss") {
    try {
      const gss = await loadGss(request, env);
      return json({
        source: gss.source,
        file: gss.file,
        weight: gss.weight,
        n_respondents: gss.n_respondents,
        groups: gss.groups,
        variables: gss.variables.map((v) => ({
          name: v.name,
          label: v.label,
          options: v.options,
          ordinal: v.ordinal,
          n: v.n,
        })),
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 500);
    }
  }

  if (route === "/api/match" && request.method === "POST") {
    try {
      const body = await request.json();
      const gss = await loadGss(request, env);
      const text = body.text || "";
      const options = Array.isArray(body.options) ? body.options : [];
      const top = candidates(gss, text, options, 6);

      const questions = {};
      top.forEach((c, i) => {
        questions[`q${i}`] = {
          type: "noul",
          instructions:
            `Do these two survey questions ask the same thing, with compatible answer options? ` +
            `A: "${text}" with options ${JSON.stringify(options)}. ` +
            `B: "${c.variable.label}" (variable ${c.variable.name}) with options ${JSON.stringify(c.variable.options)}.`,
        };
      });

      const state =
        "Two questions ask the same thing when a respondent would give the same answer to both, " +
        "and the option sets line up one-to-one. Different wording of the same idea counts as the same. " +
        "A broader or narrower topic does not.";

      const out = await drexDecide(env, state, questions);
      const answers = (out.response && out.response.answers) || {};
      const ranked = top.map((c, i) => {
        const noul = answers[`q${i}`] ? answers[`q${i}`].noul : null;
        return {
          variable: c.variable.name,
          label: c.variable.label,
          options: c.variable.options,
          ordinal: c.variable.ordinal,
          lexical: Number(c.score.toFixed(3)),
          option_overlap: Number((c.opts || 0).toFixed(2)),
          same_thing: noul,
          options_compatible: optionsCompatible({ options }, c.variable),
        };
      });
      const best = ranked.find((r) => r.same_thing != null && r.same_thing >= 0.8 && r.options_compatible) || null;

      return json({
        accepted: best,
        threshold: 0.8,
        candidates: ranked,
        drex: {
          http_status: out.http_status,
          wall_ms: out.wall_ms,
          request_id: out.request_id,
          input_tokens: (out.response && out.response.usage && out.response.usage.input_tokens) ?? null,
        },
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/panel" && request.method === "POST") {
    // Microsimulation: ask Drex to decide for one respondent per real
    // demographic cell in the group, then weight the answers by how many real
    // respondents that cell holds. Drex is used as a decider, not as a
    // distribution generator.
    try {
      const body = await request.json();
      const gss = await loadGss(request, env);
      const variable = gss.byName.get(String(body.variable || "").toLowerCase());
      if (!variable) return json({ error: `no GSS variable ${body.variable}` }, 404);
      const selection = body.selection || {};
      const text = body.text || variable.label;

      const cells = new Map();
      const values = variable.values;
      for (let i = 0; i < gss.rows.length; i++) {
        const row = gss.rows[i];
        let ok = true;
        for (const [k, v] of Object.entries(selection)) {
          const field = { age_group: 0, education: 1, party_id: 2, sex: 3 }[k];
          if (row[field] !== v) { ok = false; break; }
        }
        if (!ok || values[i] === 255) continue;
        const key = `${row[0]}-${row[1]}-${row[2]}-${row[3]}`;
        cells.set(key, (cells.get(key) || 0) + (row[4] || 0));
      }
      if (!cells.size) return json({ error: "no respondents in that group" }, 400);

      const top = [...cells.entries()].sort((a, b) => b[1] - a[1]).slice(0, Number(body.cells || 8));
      const totalWeight = top.reduce((a, [, w]) => a + w, 0);

      const results = [];
      const started = Date.now();
      let inputTokens = 0;
      for (const [key, weight] of top) {
        const [a, e, p, s] = key.split("-").map(Number);
        const cellSelection = { age_group: a, education: e, party_id: p, sex: s };
        for (const [k, v] of Object.entries(selection)) cellSelection[k] = v;
        const who = persona(cellSelection, gss.groups);
        const criteria = {};
        variable.options.forEach((opt, idx) => { criteria[`o${idx}`] = opt; });
        const out = await drexDecide(env, who, {
          answer: {
            type: "choice",
            instructions: `${who} is asked: "${text}" Which answer does this person give? Answer for one person.`,
            criteria,
          },
        });
        const ans = ((out.response || {}).answers || {}).answer || {};
        inputTokens += ((out.response || {}).usage || {}).input_tokens || 0;
        results.push({ persona: who, weight, choice: ans.choice, probabilities: ans.probabilities || {} });
      }

      const predicted = variable.options.map((opt, idx) => {
        let mass = 0;
        for (const r of results) mass += (r.probabilities[`o${idx}`] || 0) * r.weight;
        return { option: opt, p: mass / totalWeight };
      });
      const sum = predicted.reduce((a, b) => a + b.p, 0);
      if (sum > 0) predicted.forEach((d) => (d.p = d.p / sum));

      const real = split(gss, variable, selection);
      return json({
        method: "panel",
        variable: variable.name,
        cells: results.length,
        cell_breakdown: results.map((r) => ({ persona: r.persona, weight: Number(r.weight.toFixed(1)), choice: r.choice })),
        drex: {
          probabilities: predicted,
          calls: results.length,
          input_tokens: inputTokens,
          cost_usd: inputTokens * DREX_USD_PER_TOKEN,
          wall_ms: Date.now() - started,
        },
        real: { shares: real.shares, n: real.n },
        distance: { panel: real.shares.length ? tvd(predicted, real.shares) : null },
        persona: persona(selection, gss.groups),
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/panel-run" && request.method === "POST") {
    // Draw real GSS respondents, ask Drex to answer one question as each of
    // them, then compare the simulated split with the real one. One call per
    // respondent, so the counter and the API bill agree.
    try {
      const body = await request.json();
      const gss = await loadGss(request, env);
      const variable = gss.byName.get(String(body.variable || "").toLowerCase());
      if (!variable) return json({ error: `no GSS variable ${body.variable}` }, 404);
      const text = body.text || variable.label;
      // The free Workers plan allows ~50 subrequests per invocation, so one
      // request simulates at most 40 respondents; the client batches the rest
      // and sums the counts.
      const n = Math.min(Math.max(Number(body.n || 15), 5), 15);
      // workerd allows at most 6 simultaneous connections per request, so a
      // higher fan-out fails rather than going faster.
      const concurrency = Math.min(Math.max(Number(body.concurrency || 6), 1), 6);

      // deterministic sample of real rows, so a rerun gives the same panel
      const step = gss.rows.length / n;
      const picked = [];
      for (let k = 0; k < n; k++) picked.push(gss.rows[Math.floor(k * step)]);

      const criteria = {};
      variable.options.forEach((opt, i) => { criteria[`o${i}`] = opt; });
      const instructions = `You are this person answering a survey. Question: "${text}" Which answer do you give?`;

      const started = Date.now();
      let calls = 0;
      let inputTokens = 0;
      let failures = 0;
      let lastError = null;
      const answers = new Array(variable.options.length).fill(0);
      const simAnswers = new Array(picked.length).fill(-1);

      let cursor = 0;
      async function worker() {
        while (cursor < picked.length) {
          const idx = cursor++;
          const row = picked[idx];
          const who = persona(
            { age_group: row[0], education: row[1], party_id: row[2], sex: row[3] },
            gss.groups,
          );
          try {
            let out = null;
            for (let attempt = 0; attempt < 2; attempt++) {
              try {
                out = await drexDecide(env, who, {
                  answer: { type: "choice", instructions, criteria },
                });
                break;
              } catch (err) {
                lastError = String((err && err.message) || err);
                await new Promise((r) => setTimeout(r, 150 * (attempt + 1)));
              }
            }
            if (!out) throw new Error(lastError || "drex call failed");
            calls += 1;
            inputTokens += ((out.response || {}).usage || {}).input_tokens || 0;
            const ans = ((out.response || {}).answers || {}).answer || {};
            // the choice comes back as the criteria key (o0, o1, ...), not the label
            const key = String(ans.choice || "");
            const match = /^o(\d+)$/.exec(key);
            const optionIdx = match ? Number(match[1]) : variable.options.indexOf(key);
            if (optionIdx >= 0 && optionIdx < answers.length) {
              answers[optionIdx] += 1;
              simAnswers[idx] = optionIdx;
            }
          } catch {
            failures += 1;
          }
        }
      }
      await Promise.all(Array.from({ length: concurrency }, () => worker()));

      const total = answers.reduce((a, b) => a + b, 0) || 1;
      const simulated = variable.options.map((opt, i) => ({ option: opt, p: answers[i] / total }));
      const real = split(gss, variable, {});

      return json({
        method: "panel",
        variable: variable.name,
        question: text,
        n,
        concurrency,
        counters: {
          calls,
          failures,
          last_error: lastError,
          answers: total,
          elapsed_ms: Date.now() - started,
          calls_per_second: Number((calls / Math.max(0.001, (Date.now() - started) / 1000)).toFixed(2)),
          input_tokens: inputTokens,
          cost_usd: inputTokens * DREX_USD_PER_TOKEN,
        },
        simulated,
        counts: answers,
        real: { shares: real.shares, n: real.n },
        distance: real.shares.length ? tvd(simulated, real.shares) : null,
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/reword" && request.method === "POST") {
    // Phase 4: does the answer move when only the wording moves?
    try {
      if (!env.GEMINI_API_KEY) return json({ skipped: true, reason: "GEMINI_API_KEY not set on the server" }, 200);
      const body = await request.json();
      const gss = await loadGss(request, env);
      const variable = gss.byName.get(String(body.variable || "").toLowerCase());
      if (!variable) return json({ error: `no GSS variable ${body.variable}` }, 404);
      const text = body.text || variable.label;
      const selection = body.selection || {};
      const who = persona(selection, gss.groups);

      const prompt =
        `Rewrite this survey question four ways that keep exactly the same meaning and the same answer options. ` +
        `Question: "${text}" Options: ${JSON.stringify(variable.options)}. ` +
        `Reply with JSON only: {"rewordings": ["...", "...", "...", "..."]}`;

      const started = Date.now();
      const gRes = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent`,
        {
          method: "POST",
          headers: { "x-goog-api-key": env.GEMINI_API_KEY, "content-type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { responseMimeType: "application/json", temperature: 0.4 },
          }),
        },
      );
      if (!gRes.ok) return json({ skipped: true, reason: `gemini ${gRes.status}: ${(await gRes.text()).slice(0, 200)}` }, 200);
      const gBody = await gRes.json();
      const gText = gBody.candidates?.[0]?.content?.parts?.[0]?.text || "{}";
      let rewordings = [];
      try { rewordings = JSON.parse(gText).rewordings || []; } catch { rewordings = []; }
      rewordings = rewordings.filter((r) => typeof r === "string" && r.trim()).slice(0, 4);

      const criteria = {};
      variable.options.forEach((opt, i) => { criteria[`o${i}`] = opt; });

      // gate: does Drex say each rewording asks the same thing?
      const gateQuestions = {};
      rewordings.forEach((r, i) => {
        gateQuestions[`r${i}`] = {
          type: "noul",
          instructions: `Do these two questions ask the same thing, with the same answer options? A: "${text}" B: "${r}". Options: ${JSON.stringify(variable.options)}.`,
        };
      });
      let gate = { answers: {} };
      if (rewordings.length) {
        const g = await drexDecide(env, "Two questions ask the same thing when a respondent would answer both the same way.", gateQuestions);
        gate = g.response || { answers: {} };
      }

      const kept = rewordings
        .map((r, i) => ({ text: r, same: gate.answers?.[`r${i}`]?.noul ?? null }))
        .filter((r) => r.same != null && r.same >= 0.9);

      async function splitFor(question) {
        const out = await drexDecide(env, who, {
          share: {
            type: "choice",
            instructions: `${who} is asked: "${question}" Which answer does this person give? Answer as the share of people like this.`,
            criteria,
          },
        });
        const ans = ((out.response || {}).answers || {}).share || {};
        const probs = variable.options.map((opt, i) => ({ option: opt, p: Number((ans.probabilities || {})[`o${i}`] || 0) }));
        const s = probs.reduce((a, b) => a + b.p, 0);
        if (s > 0) probs.forEach((d) => (d.p = d.p / s));
        return probs;
      }

      const variants = [{ text, same: 1, original: true, probabilities: await splitFor(text) }];
      for (const k of kept) variants.push({ ...k, probabilities: await splitFor(k.text) });

      let worst = null;
      for (let i = 0; i < variants.length; i++) {
        for (let j = i + 1; j < variants.length; j++) {
          const d = tvd(variants[i].probabilities, variants[j].probabilities);
          if (!worst || d > worst.distance) worst = { a: i, b: j, distance: Number(d.toFixed(3)) };
        }
      }

      return json({
        question: text,
        persona: who,
        model: "gemini-3.1-flash-lite",
        proposed: rewordings.length,
        kept,
        dropped: rewordings.filter((r) => !kept.some((k) => k.text === r)).length,
        variants,
        max_distance: worst ? worst.distance : 0,
        unstable: Boolean(worst && worst.distance > 0.1),
        threshold: 0.1,
        gemini_ms: Date.now() - started,
        note: "Question text was sent to Gemini (free tier) to write rewordings.",
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/predict-options" && request.method === "POST") {
    // Drex answers a question exactly as the uploaded document words it, so a
    // question with no survey counterpart still gets a prediction.
    try {
      const body = await request.json();
      const text = String(body.text || "");
      const options = Array.isArray(body.options) ? body.options : [];
      if (!text || options.length < 2) return json({ error: "send text and at least two options" }, 400);
      const who = persona(body.selection || {}, (await loadGss(request, env)).groups);
      const criteria = {};
      options.forEach((opt, i) => (criteria[`o${i}`] = opt));
      const out = await drexDecide(env, who, {
        share: {
          type: "choice",
          instructions: `${who} is asked: "${text}" Which answer does this person give? Answer as the share of people like this, not one individual.`,
          criteria,
        },
      });
      const raw = ((out.response || {}).answers || {}).share || {};
      const probs = raw.probabilities || {};
      const shares = options.map((opt, i) => ({ option: opt, p: Number(probs[`o${i}`] || 0) }));
      const sum = shares.reduce((a, b) => a + b.p, 0);
      if (sum > 0) shares.forEach((d) => (d.p = d.p / sum));
      return json({ shares, wall_ms: out.wall_ms, request_id: out.request_id, input_tokens: ((out.response || {}).usage || {}).input_tokens ?? null });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/split" && request.method === "POST") {
    // The real survey answer for one group, plus the whole-country answer.
    try {
      const body = await request.json();
      const gss = await loadGss(request, env);
      const variable = gss.byName.get(String(body.variable || "").toLowerCase());
      if (!variable) return json({ error: `no GSS variable ${body.variable}` }, 404);
      const selection = body.selection || {};
      const real = split(gss, variable, selection);
      const whole = split(gss, variable, {});
      return json({
        variable: variable.name,
        label: variable.label,
        options: variable.options,
        real: { shares: real.shares, n: real.n },
        whole: { shares: whole.shares, n: whole.n },
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (route === "/api/predict" && request.method === "POST") {
    try {
      const body = await request.json();
      const gss = await loadGss(request, env);
      const variable = gss.byName.get(String(body.variable || "").toLowerCase());
      if (!variable) return json({ error: `no GSS variable ${body.variable}` }, 404);
      const selection = body.selection || {};
      const text = body.text || variable.label;
      const who = persona(selection, gss.groups);

      const criteria = {};
      variable.options.forEach((opt, i) => {
        criteria[`o${i}`] = opt;
      });

      const questions = {
        share: {
          type: "choice",
          instructions:
            `${who} is asked: "${text}" ` +
            `Which answer does this person give? Answer as the share of people like this, not one individual.`,
          criteria,
        },
        confidence: {
          type: "score",
          instructions: "How well documented is public opinion for this group on this question?",
          criteria: ["no evidence", "thin", "reasonable", "well documented"],
        },
      };

      const out = await drexDecide(env, who, questions);
      const response = out.response || {};
      const answers = response.answers || {};
      const share = answers.share || {};
      const raw = share.probabilities || {};

      const predicted = variable.options.map((opt, i) => ({
        option: opt,
        p: Number(raw[`o${i}`] || 0),
      }));
      const sum = predicted.reduce((a, b) => a + b.p, 0);
      if (sum > 0) predicted.forEach((d) => (d.p = d.p / sum));

      const real = split(gss, variable, selection);
      const whole = split(gss, variable, {});
      const inputTokens = (response.usage && response.usage.input_tokens) || 0;
      const key = geminiKey(variable.name, selection);
      const cache = await loadGeminiCache(request, env);
      const recorded = cache[key] || null;

      return json({
        variable: variable.name,
        label: variable.label,
        ordinal: variable.ordinal,
        question: text,
        persona: who,
        group: describeSelection(selection, gss.groups),
        selection,
        drex: {
          probabilities: predicted,
          confidence: share.confidence ?? null,
          evidence: answers.confidence ? answers.confidence.score : null,
          http_status: out.http_status,
          wall_ms: out.wall_ms,
          input_tokens: inputTokens,
          cost_usd: inputTokens * DREX_USD_PER_TOKEN,
          request_id: out.request_id,
        },
        real: { shares: real.shares, n: real.n, total_weight: Number(real.total_weight.toFixed(1)) },
        baselines: {
          whole_sample: { shares: whole.shares, n: whole.n },
          uniform: uniform(variable.options),
        },
        distance: {
          drex: real.shares.length ? tvd(predicted, real.shares) : null,
          whole_sample: real.shares.length ? tvd(whole.shares, real.shares) : null,
          uniform: real.shares.length ? tvd(uniform(variable.options), real.shares) : null,
        },
        gemini_key: key,
        gemini: recorded
          ? {
              label: "recorded",
              probabilities: recorded.probabilities || [],
              model: recorded.model || null,
              latency_ms: recorded.latency_ms ?? null,
              input_tokens: recorded.input_tokens ?? null,
              output_tokens: recorded.output_tokens ?? null,
              cost_usd: recorded.cost_usd ?? null,
              distance_to_real:
                recorded.probabilities && real.shares.length
                  ? tvd(recorded.probabilities, real.shares)
                  : null,
            }
          : null,
        too_few: real.n < 30,
      });
    } catch (err) {
      return json({ error: String(err.message || err) }, 502);
    }
  }

  if (env.ASSETS) return env.ASSETS.fetch(request);
  return json({ error: "not found", path: route }, 404);
}

export default {
  async fetch(request, env) {
    try {
      return await handle(request, env);
    } catch (err) {
      return json({ error: String((err && err.message) || err) }, 500);
    }
  },
};
