/**
 * General Social Survey layer: loading, weighted splits, personas, distance.
 * Pure functions over data/gss_2024.json, produced by scripts/prep_gss.py.
 */

let GSS = null;

export async function loadGss(request, env) {
  if (GSS) return GSS;
  const url = new URL("/data/gss_2024.json", request.url).toString();
  const res = env.ASSETS ? await env.ASSETS.fetch(new Request(url)) : await fetch(url);
  if (!res.ok) throw new Error(`cannot load GSS data (${res.status})`);
  GSS = await res.json();
  GSS.byName = new Map(GSS.variables.map((v) => [v.name.toLowerCase(), v]));
  return GSS;
}

/** Weighted share of each option among respondents matching `selection`. */
export function split(gss, variable, selection) {
  const values = variable.values;
  const counts = new Array(variable.options.length).fill(0);
  let totalWeight = 0;
  let n = 0;
  for (let i = 0; i < gss.rows.length; i++) {
    const row = gss.rows[i];
    if (!matches(row, selection)) continue;
    const v = values[i];
    if (v === 255 || v == null || v >= counts.length) continue;
    const w = row[4] || 0;
    counts[v] += w;
    totalWeight += w;
    n += 1;
  }
  if (!totalWeight) return { shares: [], total_weight: 0, n: 0 };
  return {
    shares: counts.map((c, i) => ({ option: variable.options[i], p: c / totalWeight, weight: c })),
    total_weight: totalWeight,
    n,
  };
}

function matches(row, selection) {
  for (const [key, value] of Object.entries(selection || {})) {
    const field = { age_group: 0, education: 1, party_id: 2, sex: 3 }[key];
    if (field === undefined) continue;
    if (row[field] !== value) return false;
  }
  return true;
}

export function selectionIsEmpty(selection) {
  return !selection || Object.keys(selection).length === 0;
}

/** Total variation distance: half the sum of absolute differences. */
export function tvd(a, b) {
  const byOption = new Map();
  for (const { option, p } of a) byOption.set(option, (byOption.get(option) || 0) + p);
  for (const { option, p } of b) byOption.set(option, (byOption.get(option) || 0) - p);
  let sum = 0;
  for (const d of byOption.values()) sum += Math.abs(d);
  return sum / 2;
}

export function uniform(options) {
  return options.map((option) => ({ option, p: 1 / options.length }));
}

/** "An American adult aged 18-29 with a bachelor's degree." — attributes only. */
export function persona(selection, groups) {
  const bits = [];
  if (selection.age_group != null) bits.push(`aged ${groups.age_group.options[selection.age_group]}`);
  const tail = [];
  if (selection.education != null) tail.push(`with ${educationPhrase(groups.education.options[selection.education])}`);
  if (selection.party_id != null) tail.push(`who identifies as ${partyPhrase(groups.party_id.options[selection.party_id])}`);
  if (selection.sex != null) tail.push(`who is ${groups.sex.options[selection.sex]}`);
  const head = bits.length ? `An American adult ${bits[0]}` : "An American adult";
  return `${head}${tail.length ? " " + tail.join(" and ") : ""}.`;
}

function educationPhrase(option) {
  return option === "bachelor's or higher" ? "a bachelor's degree or higher" : `a ${option} education`;
}

function partyPhrase(option) {
  return option === "Democrat" ? "a Democrat" : option === "Republican" ? "a Republican" : "an Independent";
}

export function describeSelection(selection, groups) {
  return Object.entries(selection || {})
    .filter(([, v]) => v != null)
    .map(([k, v]) => `${groups[k].label}: ${groups[k].options[v]}`)
    .join(" · ");
}

/* ------------------------------------------------------------ matching ---- */

const STOP = new Set(
  ("a an the of to in on for and or is are was were do does did you your we our i it this that these those " +
    "how what which would say think about please tell me us all together much many more less")
    .split(" "),
);

// Crude stemming: happy/happiness and trust/trusted must collide, so compare
// prefixes rather than whole words. Good enough to rank candidates for Drex.
function stem(token) {
  return token.length > 5 ? token.slice(0, 5) : token;
}

function tokens(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOP.has(t))
    .map(stem);
}

/** Cheap lexical similarity: token overlap with an option-text bonus. */
export function similarity(a, b) {
  const ta = new Set(tokens(a));
  const tb = new Set(tokens(b));
  if (!ta.size || !tb.size) return 0;
  let shared = 0;
  for (const t of ta) if (tb.has(t)) shared += 1;
  const jaccard = shared / (ta.size + tb.size - shared);
  const containment = shared / Math.min(ta.size, tb.size);
  return 0.5 * jaccard + 0.5 * containment;
}

/** Fraction of the question's options that look like the variable's options. */
export function optionOverlap(questionOptions, variableOptions) {
  const q = (questionOptions || []).map(tokens);
  const v = (variableOptions || []).map(tokens);
  if (!q.length || !v.length) return 0;
  let hits = 0;
  for (const qt of q) {
    const best = v.reduce((max, vt) => {
      const shared = qt.filter((t) => vt.includes(t)).length;
      return Math.max(max, shared / Math.max(1, Math.min(qt.length, vt.length)));
    }, 0);
    if (best >= 0.5) hits += 1;
  }
  return hits / q.length;
}

/** Top N GSS variables by text + option similarity, before any model check. */
export function candidates(gss, questionText, questionOptions = [], limit = 6) {
  const scored = gss.variables.map((v) => {
    const text = Math.max(similarity(questionText, v.label), similarity(questionText, v.name));
    const opts = optionOverlap(questionOptions, v.options);
    return { variable: v, score: 0.6 * text + 0.4 * opts, text, opts };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, limit);
}

/** Options count difference is a hard compatibility signal. */
const normOption = (s) =>
  String(s).toLowerCase().replace(/[^a-z0-9 ]/g, " ").replace(/\s+/g, " ").trim();

const words = (s) => new Set(normOption(s).split(" ").filter((w) => w.length > 2));

/** Same option, allowing small wording differences ("very happy" vs "pretty happy"). */
function sameOption(a, b) {
  if (a === b) return true;
  const A = words(a);
  const B = words(b);
  if (!A.size || !B.size) return false;
  let shared = 0;
  A.forEach((w) => {
    if (B.has(w)) shared += 1;
  });
  return shared / Math.max(A.size, B.size) >= 0.5;
}

const NON_ANSWER_OPTION = /\b(don'?t know|do not know|no opinion|not sure|refused|no answer)\b/i;

/** True when the question's options map one-to-one, in order, onto the variable's. */
export function optionsCompatible(question, variable) {
  const q = (question.options || []).filter((o) => !NON_ANSWER_OPTION.test(o)).map(normOption);
  const v = variable.options.map(normOption);
  if (q.length !== v.length) return false;
  return q.every((opt, i) => sameOption(opt, v[i]));
}
