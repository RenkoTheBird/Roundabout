/**
 * Exactitude scores: all in-browser — @xenova/transformers (token-classification NER)
 * plus compromise for noun phrases, modals, and term-level checks. Seven sub-metrics
 * (A–G); A–F each 0..2, G (personal relativity) 0 or negative (−1/−2). Total /12,
 * clamped to [0, 12] after summing (G subtracts from A–F).
 *
 * Exactitude component mapping used by the popup:
 *   A = quantification
 *   B = timeSpecificity
 *   C = locationScope
 *   D = definedTerms
 *   E = sourceClarity
 *   F = falsifiability (testable claims; equality/comparison or definitive universal
 *       phrasing counts toward high score)
 *   G = personalRelativity (first-person / inclusive + experiential phrasing; ≤0)
 */

import nlp from "compromise";

export const EXACTITUDE_THRESHOLD = 6;
export const EXACTITUDE_VERSION = "browser-xenova-compromise-1.1.5";

/** Small NER model (ONNX); first run downloads from Hugging Face CDN (no API key). */
const NER_MODEL_ID = "Xenova/bert-base-NER";

// Keep batches small so we yield control back to the UI thread regularly.
// Large batches can make the popup feel "unclickable" while ONNX/model work runs.
const BATCH_SIZE = 8;

let nerPipelinePromise = null;

async function getNerPipeline() {
  if (!nerPipelinePromise) {
    const { pipeline } = await import("@xenova/transformers");
    nerPipelinePromise = pipeline("token-classification", NER_MODEL_ID);
  }
  return nerPipelinePromise;
}

const _SUBJECTIVE_LEMMAS = new Set([
  "subjective",
  "opinion",
  "feel",
  "believe",
  "think",
  "seem",
  "appear",
  "probably",
  "maybe",
  "perhaps",
  "allegedly",
  "supposedly",
  "likely",
  "unlikely",
  "obvious",
  "clearly",
  "obviously",
]);

const _WEAK_TIME_ADV = new Set([
  "recently",
  "currently",
  "today",
  "yesterday",
  "tomorrow",
  "now",
  "soon",
]);

const _ATTRIBUTION_PHRASES = [
  "according to",
  "reported by",
  "data from",
  "published by",
  "study by",
  "survey by",
  "research from",
  "figures from",
  "cited by",
  "released by",
  "statement from",
];

/** Adjectives common in social greetings — exclude with generic "day" (Duration) for defined-terms. */
const _GREETING_ADJS = new Set([
  "peaceful",
  "nice",
  "great",
  "good",
  "wonderful",
  "lovely",
  "happy",
  "beautiful",
  "blessed",
  "amazing",
  "fantastic",
]);

/** NER labels that count as a proper / named anchor inside a noun phrase (excludes DATE/TIME-only noise). */
const _DEF_ENTITY_LABELS = new Set([
  "PER",
  "ORG",
  "GPE",
  "LOC",
  "MISC",
  "FAC",
]);

const _STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "and",
  "or",
  "but",
  "in",
  "on",
  "at",
  "to",
  "for",
  "of",
  "as",
  "by",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
  "have",
  "has",
  "had",
  "do",
  "does",
  "did",
  "will",
  "would",
  "could",
  "should",
  "may",
  "might",
  "must",
  "can",
  "that",
  "this",
  "these",
  "those",
  "it",
  "its",
]);

/** Spelled-out cardinals / scales for definition-clarity heuristic when compromise is unavailable. */
const _CARDINAL_WORDS = new Set([
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
  "twenty",
  "thirty",
  "forty",
  "fifty",
  "sixty",
  "seventy",
  "eighty",
  "ninety",
  "hundred",
  "thousand",
  "million",
  "billion",
  "trillion",
]);

function clamp(n, lo = 0, hi = 2) {
  return Math.max(lo, Math.min(hi, n));
}

function escapeRegExp(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function lemmaLike(w) {
  return w.replace(/[^a-z]/gi, "").toLowerCase();
}

/**
 * Quantity modifiers (often with units: dose, fold, blind) — quantification + defined-terms nummod cue.
 */
const _QUANTITY_MODIFIER_LEMMAS = new Set([
  "single",
  "double",
  "triple",
  "quadruple",
  "quintuple",
  "twice",
  "thrice",
  "once",
  "multiply",
  "multiplies",
  "multiplying",
  "multiplied",
  "multiple",
  "dual",
  "pair",
  "pairs",
]);

/** Definitive / universal claim adverbs — strengthen falsifiability (testable absolutes). */
const _DEFINITIVE_CLAIM_LEMMAS = new Set([
  "completely",
  "complete",
  "fully",
  "full",
  "wholly",
  "whole",
  "entirely",
  "entire",
  "totally",
  "total",
  "utterly",
  "always",
  "never",
  "invariably",
  "necessarily",
  "necessary",
  "exclusively",
  "exclusive",
  "solely",
  "sole",
  "absolutely",
  "absolute",
]);

const _QUANTITY_MODIFIER_RE =
  /\b(?:single|double|triple|quadruple|quintuple|twice|thrice|once|multiply|multiplies|multiplying|multiplied|multiple|dual|pair|pairs)\b/gi;

/**
 * @returns {{ matches: string[], hasAny: boolean }}
 */
function detectQuantityModifierLanguage(text) {
  const matches = [];
  const r = new RegExp(_QUANTITY_MODIFIER_RE.source, "gi");
  let m;
  while ((m = r.exec(text)) !== null) {
    matches.push(m[0]);
  }
  const uniq = [...new Set(matches.map((x) => x.toLowerCase()))];
  return { matches: uniq, hasAny: uniq.length > 0 };
}

/**
 * @returns {{ hits: string[], hasAny: boolean }}
 */
function detectDefinitiveClaimLanguage(text, doc) {
  const hits = [];
  const pushLemma = (raw) => {
    const lem = lemmaLike(raw);
    if (lem && _DEFINITIVE_CLAIM_LEMMAS.has(lem)) hits.push(lem);
  };
  if (doc) {
    for (const row of doc.terms().json()) {
      for (const t of row.terms ?? []) {
        pushLemma(t.normal || t.text || "");
      }
    }
  } else {
    for (const raw of text.split(/\s+/)) {
      pushLemma(raw);
    }
  }
  const uniq = [...new Set(hits)];
  return { hits: uniq, hasAny: uniq.length > 0 };
}

/**
 * Comparison / equality phrasing (quantification + falsifiability).
 * Strong patterns justify a full quantification point; any match counts as "comparison language".
 */
const _COMPARISON_STRONG_RES = [
  /\bexact\s+same\b/i,
  /\bgreater\s+than\b/i,
  /\bless\s+than\b/i,
  /\bmore\s+than\b/i,
  /\bfewer\s+than\b/i,
  /\bhigher\s+than\b/i,
  /\blower\s+than\b/i,
  /\bequal\s+to\b/i,
  /\bidentical\s+to\b/i,
  /\bas\s+(?:much|many|high|low)\s+as\b/i,
  /\bcompared\s+(?:to|with)\b/i,
  /\bversus\b|\bvs\.?\b/i,
  /\bdifference\s+between\b/i,
];

/** Weaker equality/comparison words (counts toward quantification / falsifiability). */
const _COMPARISON_WEAK_RES = [
  /\bthe\s+same\b/gi,
  /\bsame\b/gi,
  /\bequals?\b/gi,
  /\bequality\b/gi,
  /\bidentical\b/gi,
];

/**
 * @returns {{ matches: string[], hasStrong: boolean, hasAny: boolean }}
 */
function detectComparisonLanguage(text) {
  const matches = [];
  for (const re of _COMPARISON_STRONG_RES) {
    const r = new RegExp(re.source, re.flags.includes("g") ? re.flags : `${re.flags}g`);
    let m;
    while ((m = r.exec(text)) !== null) {
      matches.push(m[0]);
    }
  }
  for (const re of _COMPARISON_WEAK_RES) {
    const r = new RegExp(re.source, re.flags.includes("g") ? re.flags : `${re.flags}g`);
    let m;
    while ((m = r.exec(text)) !== null) {
      matches.push(m[0]);
    }
  }
  const uniq = [...new Set(matches)];
  const hasStrong = _COMPARISON_STRONG_RES.some((re) => {
    const r = new RegExp(re.source, re.flags);
    return r.test(text);
  });
  const hasWeak = _COMPARISON_WEAK_RES.some((re) => {
    const r = new RegExp(re.source, "i");
    return r.test(text);
  });
  const hasAny = hasStrong || hasWeak;
  return { matches: uniq, hasStrong, hasAny };
}

/** Canadian provinces/territories + US states + common country names (case-insensitive word match). */
const _KNOWN_GPE_NAMES = [
  "British Columbia",
  "Prince Edward Island",
  "Newfoundland and Labrador",
  "Northwest Territories",
  "New Brunswick",
  "Saskatchewan",
  "Pennsylvania",
  "Massachusetts",
  "Mississippi",
  "Connecticut",
  "North Carolina",
  "South Carolina",
  "North Dakota",
  "South Dakota",
  "West Virginia",
  "New Hampshire",
  "Rhode Island",
  "New Mexico",
  "New York",
  "New Jersey",
  "California",
  "Washington",
  "Wisconsin",
  "Minnesota",
  "Louisiana",
  "Tennessee",
  "Kentucky",
  "Maryland",
  "Oklahoma",
  "Missouri",
  "Colorado",
  "Nebraska",
  "Michigan",
  "Illinois",
  "Ohio",
  "Georgia",
  "Virginia",
  "Indiana",
  "Arizona",
  "Oregon",
  "Montana",
  "Wyoming",
  "Nevada",
  "Idaho",
  "Utah",
  "Texas",
  "Florida",
  "Alaska",
  "Hawaii",
  "Alabama",
  "Arkansas",
  "Delaware",
  "Iowa",
  "Kansas",
  "Maine",
  "Vermont",
  "Alberta",
  "Manitoba",
  "Nova Scotia",
  "Ontario",
  "Quebec",
  "Québec",
  "Nunavut",
  "Yukon",
  "Canada",
  "Mexico",
  "England",
  "Scotland",
  "Wales",
  "Ireland",
  "France",
  "Germany",
  "Japan",
  "China",
  "India",
  "Brazil",
  "Australia",
  "United States",
  "United Kingdom",
  "South Korea",
  "North Korea",
  "New Zealand",
].sort((a, b) => b.length - a.length);

/**
 * Case-insensitive geographic names when NER misses lowercase place names.
 * @returns {Array<{ text: string, label: string }>}
 */
function extractCaseInsensitiveGeoHints(text) {
  const found = [];
  const used = [];
  function overlaps(start, end) {
    return used.some(([a, b]) => !(end <= a || start >= b));
  }
  for (const name of _KNOWN_GPE_NAMES) {
    const re = new RegExp(`\\b${escapeRegExp(name)}\\b`, "gi");
    let m;
    while ((m = re.exec(text)) !== null) {
      const start = m.index;
      const end = start + m[0].length;
      if (overlaps(start, end)) continue;
      used.push([start, end]);
      found.push({ text: m[0], label: "GPE" });
    }
  }
  return found;
}

/**
 * Normalize Xenova / HF-style token-classification output to entity list.
 * @param {unknown} data
 * @returns {Array<{ text: string, label: string, start?: number, end?: number }>}
 */
export function normalizeNerResponse(data) {
  if (!data) return [];
  let rows = Array.isArray(data) ? data : [data];
  if (rows.length && Array.isArray(rows[0])) {
    rows = rows.flat();
  }
  const out = [];
  for (const item of rows) {
    if (!item || typeof item !== "object") continue;
    const g = item.entity_group ?? item.entity;
    const label = typeof g === "string" ? g.replace(/^[BI]-/, "") : "MISC";
    const word = item.word ?? item.text ?? "";
    if (!word && item.start == null) continue;
    out.push({
      text: String(word),
      label,
      start: item.start,
      end: item.end,
    });
  }
  return out;
}

function extractMoneyLikeEntities(text) {
  const s = text;
  const found = [];
  const patterns = [
    { re: /\$\s*\d+(?:,\d{3})*(?:\.\d+)?/g, label: "MONEY" },
    { re: /€\s*\d+(?:,\d{3})*(?:\.\d+)?/g, label: "MONEY" },
    { re: /£\s*\d+(?:,\d{3})*(?:\.\d+)?/g, label: "MONEY" },
    { re: /\d+(?:\.\d+)?\s*%/g, label: "PERCENT" },
    { re: /\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:percent|percentage)\b/gi, label: "PERCENT" },
    { re: /\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)\b/gi, label: "QUANTITY" },
  ];
  for (const { re, label } of patterns) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(s)) !== null) {
      found.push({ text: m[0], label });
    }
  }
  return found;
}

function countNumericTokens(text) {
  const re = /\b\d+(?:\.\d+)?\b/g;
  let n = 0;
  while (re.exec(text) !== null) n += 1;
  return n;
}

function hasPercentSymbol(text) {
  return text.includes("%");
}

function hasPercentWord(text) {
  return /\b(percent|percentage)\b/i.test(text);
}

function scoreQuantification(text, nerEnts, signals) {
  const comp = detectComparisonLanguage(text);
  const qtyMod = detectQuantityModifierLanguage(text);
  const moneyFromRegex = extractMoneyLikeEntities(text);
  const moneyLabels = new Set(["PERCENT", "MONEY", "QUANTITY", "CARDINAL"]);
  const ents = [
    ...nerEnts.filter((e) => moneyLabels.has(e.label)),
    ...moneyFromRegex,
  ];
  const entText = ents.map((e) => [e.text, e.label]);
  const numericTokens = countNumericTokens(text);
  const hp = hasPercentSymbol(text);
  const hpw = hasPercentWord(text);

  signals.quantification = {
    entities: entText,
    numericTokenCount: numericTokens,
    hasPercentSymbol: hp,
    hasPercentWord: hpw,
    comparisonMatches: comp.matches,
    comparisonStrong: comp.hasStrong,
    comparisonAny: comp.hasAny,
    quantityModifierMatches: qtyMod.matches,
    quantityModifierAny: qtyMod.hasAny,
  };

  const hasMoneyOrPercentEnt = ents.some((e) =>
    ["PERCENT", "MONEY"].includes(e.label)
  );
  const strong =
    ents.length >= 2 ||
    (hasMoneyOrPercentEnt && (numericTokens >= 1 || hp || hpw)) ||
    comp.hasStrong ||
    (qtyMod.hasAny &&
      (numericTokens >= 1 ||
        hp ||
        hpw ||
        ents.length >= 1 ||
        comp.hasAny ||
        comp.hasStrong));
  const medium =
    ents.length >= 1 ||
    numericTokens >= 1 ||
    hp ||
    hpw ||
    comp.hasAny ||
    qtyMod.hasAny;
  if (strong) return 2;
  if (medium) return 1;
  return 0;
}

const _MONTH_RE =
  /\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b/i;
const _DATE_LIKE =
  /\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|(?:19|20)\d{2})\b/;

/**
 * Scientific / superscript exponentials: 10^78, 10⁷⁸, 1.2e+34 (not plain "2024").
 * Used for time-scale specificity and defined-term numeric chunks.
 */
const _SCI_EXPONENT_OR_NOTATION_RE =
  /10(?:\^[-+]?\d+|[⁰¹²³⁴⁵⁶⁷⁸⁹]+)|(?:[+-]?(?:\d+\.?\d*|\.\d+))[eE][+-]?\d+/;

/** "10⁷⁸ years" / "10^78 years" / "1e78 years" — exponent + duration unit. */
const _SCI_YEARS_DURATION_RE = new RegExp(
  `(?:${_SCI_EXPONENT_OR_NOTATION_RE.source})\\s*,?\\s*years?`,
  "i",
);

/** Ordinary cardinal + years (e.g. "in 50 years") — explicit duration. */
const _PLAIN_NUMBER_YEARS_RE = /\b\d+(?:\.\d+)?\s+years?\b/i;

const _YEARS_WORD_RE = /\byears?\b/i;

/**
 * @returns {{ sciExponentMatch: boolean, sciYearsPhrase: boolean, plainNumberYears: boolean, yearsWord: boolean }}
 */
function detectTimeScaleSignals(text) {
  return {
    sciExponentMatch: _SCI_EXPONENT_OR_NOTATION_RE.test(text),
    sciYearsPhrase: _SCI_YEARS_DURATION_RE.test(text),
    plainNumberYears: _PLAIN_NUMBER_YEARS_RE.test(text),
    yearsWord: _YEARS_WORD_RE.test(text),
  };
}

function scoreTime(text, nerEnts, signals) {
  const dates = nerEnts.filter((e) => e.label === "DATE");
  const dateTexts = dates.map((e) => e.text);
  const regexDates = [];
  if (_DATE_LIKE.test(text) || _MONTH_RE.test(text)) {
    regexDates.push("regex-date");
  }
  const times = nerEnts.filter((e) => e.label === "TIME");
  const scale = detectTimeScaleSignals(text);
  signals.timeSpecificity = {
    dates: dateTexts.length ? dateTexts : regexDates,
    times: times.map((e) => e.text),
    sciExponentMatch: scale.sciExponentMatch,
    sciYearsPhrase: scale.sciYearsPhrase,
    plainNumberYears: scale.plainNumberYears,
    yearsWord: scale.yearsWord,
  };
  if (
    dates.length ||
    times.length ||
    regexDates.length ||
    scale.sciYearsPhrase ||
    scale.plainNumberYears
  ) {
    return 2;
  }
  const lower = text.toLowerCase();
  const words = lower.split(/\s+/);
  if (words.some((w) => _WEAK_TIME_ADV.has(w.replace(/[^a-z]/gi, "")))) {
    return 1;
  }
  if (scale.yearsWord || scale.sciExponentMatch) {
    return 1;
  }
  return 0;
}

function scoreLocationScope(text, nerEnts, signals) {
  const geoHints = extractCaseInsensitiveGeoHints(text);
  const seen = new Set(
    nerEnts.map((e) => `${e.label}:${(e.text || "").toLowerCase()}`)
  );
  const merged = [...nerEnts];
  for (const g of geoHints) {
    const key = `${g.label}:${(g.text || "").toLowerCase()}`;
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(g);
    }
  }
  const scopeLabels = new Set(["GPE", "LOC", "FAC", "ORG", "NORP"]);
  const hfMap = { LOC: "LOC", ORG: "ORG", PER: "GPE" };
  const scoped = merged.filter((e) => {
    if (scopeLabels.has(e.label)) return true;
    return hfMap[e.label] != null;
  });
  const labels = scoped.map((e) => hfMap[e.label] ?? e.label);
  const uniqueTypes = new Set(labels);
  signals.locationScope = {
    entities: scoped.map((e) => [e.text, hfMap[e.label] ?? e.label]),
    distinctTypeCount: uniqueTypes.size,
    caseInsensitiveGeoHints: geoHints.map((e) => e.text),
  };
  if (scoped.length >= 2 || uniqueTypes.size >= 2) return 2;
  if (scoped.length === 1) return 1;
  return 0;
}

function tokenizeWords(text) {
  return text
    .split(/\s+/)
    .map((w) => w.replace(/^[^\w]+|[^\w]+$/g, "").toLowerCase())
    .filter(Boolean);
}

/**
 * True when a named entity span (non-date) appears inside the phrase text.
 * @param {string} phraseText
 * @param {Array<{ text: string, label: string }>} nerEnts
 */
function phraseOverlapsDefiningEntity(phraseText, nerEnts) {
  if (!nerEnts?.length || !phraseText) return false;
  const lower = phraseText.toLowerCase();
  for (const e of nerEnts) {
    if (!_DEF_ENTITY_LABELS.has(e.label)) continue;
    const t = (e.text || "").trim();
    if (t.length < 2) continue;
    if (lower.includes(t.toLowerCase())) return true;
  }
  return false;
}

function isQualifyingAdjNounTerms(terms) {
  if (terms.length !== 2) return false;
  const [a, n] = terms;
  const aTags = a.tags || [];
  const nTags = n.tags || [];
  if (!aTags.includes("Adjective") || !nTags.includes("Noun")) return false;
  if (nTags.includes("WeekDay")) return false;
  if (
    nTags.includes("Duration") &&
    n.normal === "day" &&
    _GREETING_ADJS.has(a.normal)
  ) {
    return false;
  }
  return true;
}

function isQualifyingAdvNounTerms(terms) {
  if (terms.length !== 2) return false;
  const [a, n] = terms;
  const aTags = a.tags || [];
  const nTags = n.tags || [];
  return aTags.includes("Adverb") && nTags.includes("Noun");
}

function isQualifyingAdvAdjNounTerms(terms) {
  if (terms.length !== 3) return false;
  const [a, b, n] = terms;
  const aTags = a.tags || [];
  const bTags = b.tags || [];
  const nTags = n.tags || [];
  return (
    aTags.includes("Adverb") &&
    bTags.includes("Adjective") &&
    nTags.includes("Noun")
  );
}

/**
 * Leading token is a quantity (digits or number words), not a determiner mis-tagged as Value.
 * Optional leading term allows quantity-modifier lemmas (e.g. "single dose").
 * @param {string[]} tags
 * @param {{ normal?: string, text?: string } | undefined} leadingTerm
 */
function isNumberLikeLeadingTerm(tags, leadingTerm) {
  if (leadingTerm) {
    const lem = lemmaLike(leadingTerm.normal || leadingTerm.text || "");
    if (lem && _QUANTITY_MODIFIER_LEMMAS.has(lem)) return true;
  }
  if (!tags || tags.length === 0) return false;
  if (tags.includes("Determiner")) return false;
  return (
    tags.includes("NumericValue") ||
    tags.includes("TextValue") ||
    tags.includes("Multiple") ||
    tags.includes("Fraction")
  );
}

function isQualifyingValueNounTerms(terms) {
  if (terms.length < 2) return false;
  const firstTags = terms[0].tags || [];
  const lastTags = terms[terms.length - 1].tags || [];
  if (!lastTags.includes("Noun")) return false;
  return isNumberLikeLeadingTerm(firstTags, terms[0]);
}

function scoreChunkFromTerms(terms, kind) {
  const n = terms.length;
  const chunkText = terms.map((t) => t.text).join(" ");
  const tagStr = (terms.map((t) => (t.tags || []).join(" ")) || []).join(" ").toLowerCase();
  const hasAmod = tagStr.includes("adjective");
  const hasCompound = terms.length >= 2;
  const hasQtyLemma = terms.some((t) => {
    const lem = lemmaLike(t.normal || t.text || "");
    return lem && _QUANTITY_MODIFIER_LEMMAS.has(lem);
  });
  const hasSciNumericHint = _SCI_EXPONENT_OR_NOTATION_RE.test(chunkText);
  const hasNummod =
    hasQtyLemma ||
    hasSciNumericHint ||
    tagStr.includes("value") ||
    tagStr.includes("cardinal");
  let score = 0;
  if (n >= 3 || (n >= 2 && (hasAmod || hasCompound || hasNummod))) score = 2;
  else if (n >= 2) score = 1;
  return {
    text: chunkText,
    tokenCount: n,
    hasAmod,
    hasCompound,
    hasNummod,
    kind,
    score,
  };
}

/** Fallback when compromise doc unavailable */
function scoreDefinitionClarityHeuristic(text, signals, nerEnts) {
  const words = tokenizeWords(text);
  let best = 0;
  const chunkDetails = [];
  const sciYearsMatch = text.match(_SCI_YEARS_DURATION_RE);
  if (sciYearsMatch) {
    const span = sciYearsMatch[0].trim();
    chunkDetails.push({
      text: span,
      tokenCount: 2,
      hasAmod: false,
      hasCompound: true,
      hasNummod: true,
      hasQuantityModifier: false,
      kind: "sciExponentYears",
    });
    best = 2;
  }
  let i = 0;
  while (i < words.length) {
    if (_STOPWORDS.has(words[i])) {
      i += 1;
      continue;
    }
    let j = i;
    while (j < words.length && !_STOPWORDS.has(words[j])) j += 1;
    const run = words.slice(i, j);
    const n = run.length;
    const chunkText = run.join(" ");
    const hasDigit = run.some((w) => /\d/.test(w));
    const hasLeadingCardinalWord = n >= 2 && _CARDINAL_WORDS.has(run[0]);
    const hasQtyMod = run.some((w) => _QUANTITY_MODIFIER_LEMMAS.has(w));
    const hasNum = hasDigit || hasLeadingCardinalWord || hasQtyMod;
    const hasLong = run.some((w) => w.length >= 8);
    const entityOverlap =
      phraseOverlapsDefiningEntity(chunkText, nerEnts) && n >= 2;
    chunkDetails.push({
      text: chunkText,
      tokenCount: n,
      hasAmod: hasLong,
      hasCompound: n >= 2,
      hasNummod: hasNum,
      hasQuantityModifier: hasQtyMod,
    });
    let local = 0;
    if (n >= 2 && (hasNum || hasLong || entityOverlap)) local = 2;
    else if (n >= 3) local = 1;
    else if (n >= 2) local = 1;
    else local = 0;
    best = Math.max(best, local);
    i = j + 1;
  }
  if (!chunkDetails.length) {
    signals.definitionClarity = { chunks: [], bestChunkScore: 0 };
    return 0;
  }
  signals.definitionClarity = { chunks: chunkDetails, bestChunkScore: best };
  return clamp(best);
}

/**
 * Defined terms: noun phrases that look like definitional language — strict
 * adjective+noun, adverb+noun, adverb+adjective+noun, proper-noun+head, acronym+noun,
 * number-led noun phrases (e.g. "3 dogs", "69 million people", "four ships"),
 * or multi-noun phrase with a named-entity anchor (NER). Avoids loose #Adjective? #Noun+.
 */
function scoreDefinitionClarityCompromise(doc, signals, nerEnts, fullText) {
  const seen = new Set();
  const chunkDetails = [];
  let best = 0;
  const rawText =
    (typeof fullText === "string" && fullText) ||
    (typeof doc.text === "function" ? doc.text() : "") ||
    "";

  function pushPhrase(phrase, kind) {
    const terms = phrase.terms ?? [];
    if (terms.length === 0) return;
    const key = (phrase.text || terms.map((t) => t.text).join(" ")).trim();
    if (seen.has(key)) return;
    seen.add(key);
    const row = scoreChunkFromTerms(terms, kind);
    chunkDetails.push(row);
    best = Math.max(best, row.score);
  }

  for (const phrase of doc.match("#Adjective #Noun").json()) {
    const terms = phrase.terms ?? [];
    if (!isQualifyingAdjNounTerms(terms)) continue;
    pushPhrase(phrase, "adjNoun");
  }
  for (const phrase of doc.match("#Adverb #Noun").json()) {
    const terms = phrase.terms ?? [];
    if (!isQualifyingAdvNounTerms(terms)) continue;
    pushPhrase(phrase, "advNoun");
  }
  for (const phrase of doc.match("#Adverb #Adjective #Noun").json()) {
    const terms = phrase.terms ?? [];
    if (!isQualifyingAdvAdjNounTerms(terms)) continue;
    pushPhrase(phrase, "advAdjNoun");
  }
  for (const phrase of doc.match("#ProperNoun #Noun*").json()) {
    pushPhrase(phrase, "properNoun");
  }
  for (const phrase of doc.match("#Acronym #Noun").json()) {
    pushPhrase(phrase, "acronymNoun");
  }
  for (const phrase of doc.match("#NumericValue #Value* #Noun").json()) {
    pushPhrase(phrase, "numericNoun");
  }
  for (const phrase of doc.match("#Value #Noun").json()) {
    const terms = phrase.terms ?? [];
    if (!isQualifyingValueNounTerms(terms)) continue;
    pushPhrase(phrase, "valueNoun");
  }
  for (const phrase of doc.match("#Noun #Noun+").json()) {
    const t = phrase.text || "";
    if (!phraseOverlapsDefiningEntity(t, nerEnts)) continue;
    pushPhrase(phrase, "nerCompound");
  }

  const sciYearsMatch = rawText.match(_SCI_YEARS_DURATION_RE);
  if (
    sciYearsMatch &&
    !chunkDetails.some((c) => _SCI_YEARS_DURATION_RE.test((c.text || "").trim()))
  ) {
    const span = sciYearsMatch[0].trim();
    chunkDetails.push({
      text: span,
      tokenCount: 2,
      hasAmod: false,
      hasCompound: true,
      hasNummod: true,
      kind: "sciExponentYears",
      score: 2,
    });
    best = Math.max(best, 2);
  }

  if (!chunkDetails.length) {
    signals.definitionClarity = { chunks: [], bestChunkScore: 0, source: "compromise" };
    return 0;
  }
  signals.definitionClarity = {
    chunks: chunkDetails,
    bestChunkScore: best,
    source: "compromise",
  };
  return clamp(best);
}

function scoreDefinitionClarity(text, signals, doc, nerEnts) {
  if (doc) {
    return scoreDefinitionClarityCompromise(doc, signals, nerEnts, text);
  }
  return scoreDefinitionClarityHeuristic(text, signals, nerEnts);
}

function attributionSpanCount(lower) {
  let n = 0;
  for (const p of _ATTRIBUTION_PHRASES) {
    let idx = 0;
    while ((idx = lower.indexOf(p, idx)) !== -1) {
      n += 1;
      idx += p.length;
    }
  }
  return n;
}

function scoreSource(text, nerEnts, signals) {
  const orgs = nerEnts.filter((e) =>
    ["ORG", "WORK_OF_ART", "MISC"].includes(e.label)
  );
  const orgTexts = orgs.map((e) => e.text);
  const lower = text.toLowerCase();
  const attrCount = attributionSpanCount(lower);

  signals.sourceClarity = {
    attributionMatchCount: attrCount,
    organizations: orgTexts,
  };

  const hasAttr = attrCount > 0;
  const hasOrg = orgTexts.length > 0;
  if (hasAttr && hasOrg) return 2;
  if (hasAttr || hasOrg) return 1;
  return 0;
}

const _FINITE_VERB_RE =
  /\b(?:is|are|was|were|am|been|being|have|has|had|do|does|did|pay|pays|paid|rise|rose|risen|fall|fell|fallen|increase|decreased|say|says|said|show|shows|showed|found|find|report|reports|reported|remain|remains|stood|stand|goes|went|come|came|eradicate|eradicates|eradicated|eliminate|eliminates|eliminated)\b/i;

/**
 * Proper noun / acronym subject + will — testable prediction (not hedging "might");
 * excludes bare pronoun/expletive subjects (There will, It will, …).
 */
const _ACTOR_WILL_RE =
  /\b(?!(?:There|It|They|We|You|I|He|She|This|That|These|Those)\b)(?:[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+){0,4}|[A-Z]\.(?:[A-Z]\.)+)\s+will\b/;

/** Proper / acronym subject + past "did" — reportable act. */
const _ACTOR_DID_RE =
  /\b(?:[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+){0,4}|[A-Z]\.(?:[A-Z]\.)+)\s+did\b/;

/**
 * Quote / speech attribution: named actor + speech verb (provably sourced or not).
 * Optional leading "The" for titles (The White House says).
 */
const _SPEECH_ATTRIBUTION_RE =
  /\b(?:The\s+)?(?:[A-Z][a-zA-Z']+(?:\s+[A-Z][a-zA-Z']+){0,4}|[A-Z]\.(?:[A-Z]\.)+)\s+(?:says|said|saying|states|stated|announces?|told|writes|wrote|claims|reports|reported)\b/;

/**
 * @returns {{ hasActorWill: boolean, hasActorDid: boolean, hasSpeechAttribution: boolean }}
 */
function detectReportClaimStructures(text) {
  return {
    hasActorWill: _ACTOR_WILL_RE.test(text),
    hasActorDid: _ACTOR_DID_RE.test(text),
    hasSpeechAttribution: _SPEECH_ATTRIBUTION_RE.test(text),
  };
}

/**
 * "U.S. will …" is a falsifiable prediction, not epistemic hedging — drop will from modal tally.
 */
function filterModalsAllowingActorWill(text, modals) {
  if (!modals.includes("will")) return modals;
  if (_ACTOR_WILL_RE.test(text)) return modals.filter((m) => m !== "will");
  return modals;
}

function scoreFalsifiability(text, nerEnts, signals, doc) {
  const comp = detectComparisonLanguage(text);
  const definitive = detectDefinitiveClaimLanguage(text, doc);
  let modals = [];
  let subjectiveHits = [];

  if (doc) {
    doc.match("#Modal").forEach((m) => {
      const w = m.text();
      if (w) modals.push(w.toLowerCase());
    });
    for (const row of doc.terms().json()) {
      for (const t of row.terms ?? []) {
        const norm = t.normal;
        if (norm && _SUBJECTIVE_LEMMAS.has(norm)) {
          subjectiveHits.push(norm);
        }
      }
    }
  } else {
    const words = text.split(/\s+/);
    for (const raw of words) {
      const w = lemmaLike(raw);
      if (
        w &&
        [
          "can",
          "could",
          "should",
          "would",
          "may",
          "might",
          "must",
          "will",
          "shall",
        ].includes(w)
      ) {
        modals.push(w);
      }
    }
    for (const raw of words) {
      const lem = lemmaLike(raw);
      if (lem && _SUBJECTIVE_LEMMAS.has(lem)) subjectiveHits.push(lem);
    }
  }

  modals = filterModalsAllowingActorWill(text, modals);

  const reportClaim = detectReportClaimStructures(text);

  const anchorLabels = new Set([
    "DATE",
    "TIME",
    "CARDINAL",
    "PERCENT",
    "MONEY",
    "QUANTITY",
    "GPE",
    "LOC",
    "FAC",
    "ORG",
  ]);
  const hasAnchor =
    nerEnts.some((e) => anchorLabels.has(e.label)) ||
    _DATE_LIKE.test(text) ||
    /\b\d+\b/.test(text);

  let finiteVerbs = 0;
  if (_FINITE_VERB_RE.test(text)) finiteVerbs = 1;
  if (doc) {
    const v = doc.verbs().length;
    if (v > 0) finiteVerbs = Math.max(finiteVerbs, 1);
  }

  signals.falsifiability = {
    modalCount: modals.length,
    modalLemmas: modals,
    subjectiveTokenCount: subjectiveHits.length,
    hasConcreteEntityAnchor: hasAnchor,
    finiteVerbCount: finiteVerbs,
    equalityOrComparison: comp.hasAny,
    equalityOrComparisonStrong: comp.hasStrong,
    definitiveClaimHits: definitive.hits,
    definitiveClaimAny: definitive.hasAny,
    reportSpeechAttribution: reportClaim.hasSpeechAttribution,
    reportActorWill: reportClaim.hasActorWill,
    reportActorDid: reportClaim.hasActorDid,
  };

  const reportClaimSignal =
    reportClaim.hasSpeechAttribution ||
    reportClaim.hasActorWill ||
    reportClaim.hasActorDid;

  if (modals.length || subjectiveHits.length) return 0;
  if (
    finiteVerbs &&
    (hasAnchor || comp.hasAny || definitive.hasAny || reportClaimSignal)
  ) {
    return 2;
  }
  if (finiteVerbs) return 1;
  return 0;
}

/** Curly apostrophe → ASCII for consistent contraction matching. */
function normalizeApostrophesForPronouns(text) {
  return (text || "").replace(/\u2019/g, "'");
}

/**
 * First-person singular + inclusive plural only (not you/he/she/they).
 * Each pattern is tested on apostrophe-normalized text.
 */
const _FIRST_PERSON_PRONOUN_CHECKS = [
  { id: "I", re: /\bI\b/i },
  { id: "me", re: /\bme\b/i },
  { id: "my", re: /\bmy\b/i },
  { id: "mine", re: /\bmine\b/i },
  { id: "myself", re: /\bmyself\b/i },
  { id: "we", re: /\bwe\b/i },
  { id: "our", re: /\bour\b/i },
  { id: "ours", re: /\bours\b/i },
  { id: "ourselves", re: /\bourselves\b/i },
  { id: "I'm", re: /\bI['']m\b/i },
  { id: "I've", re: /\bI['']ve\b/i },
  { id: "I'll", re: /\bI['']ll\b/i },
  { id: "I'd", re: /\bI['']d\b/i },
  { id: "we're", re: /\bwe['']re\b/i },
  { id: "we've", re: /\bwe['']ve\b/i },
  { id: "we'll", re: /\bwe['']ll\b/i },
  { id: "we'd", re: /\bwe['']d\b/i },
];

/**
 * "us" pronoun vs acronym US (e.g. "US inflation") — /\bus\b/i falsely matches the latter.
 * @param {string} normalized apostrophe-normalized text
 */
function hasUsPronounNotAcronym(normalized) {
  const re = /\bus\b/gi;
  let m;
  while ((m = re.exec(normalized)) !== null) {
    if (m[0] !== "US") return true;
  }
  return false;
}

/** Lowercase phrases checked after apostrophe normalization (substring match). */
const _EXPERIENTIAL_PHRASES = [
  "my time",
  "i'm back",
  "i am back",
  "we're back",
  "we are back",
  "my last",
  "my trip",
  "my visit",
  "my experience",
  "i went",
  "i came",
  "i saw",
  "i remember",
];

/**
 * G: personal relativity — penalizes first-person / inclusive experiential wording (0, −1, or −2).
 * @param {string} text
 * @param {Record<string, unknown>} signals
 * @returns {0 | -1 | -2}
 */
function scorePersonalRelativity(text, signals) {
  const normalized = normalizeApostrophesForPronouns(text);
  const pronounHits = [];
  for (const { id, re } of _FIRST_PERSON_PRONOUN_CHECKS) {
    if (re.test(normalized)) pronounHits.push(id);
  }
  if (hasUsPronounNotAcronym(normalized)) pronounHits.push("us");
  const lower = normalized.toLowerCase();
  const experientialHits = [];
  for (const phrase of _EXPERIENTIAL_PHRASES) {
    if (lower.includes(phrase)) experientialHits.push(phrase);
  }
  let value = 0;
  if (pronounHits.length) {
    value = experientialHits.length ? -2 : -1;
  }
  signals.personalRelativity = {
    pronounHits,
    experientialHits,
    value,
  };
  return value;
}

/**
 * @param {string} text
 * @param {unknown} nerRaw
 * @param {ReturnType<typeof nlp>} [doc] compromise doc when available
 */
export function scoreSentenceFromNer(text, nerRaw, doc) {
  const nerEnts = normalizeNerResponse(nerRaw);
  const signals = {};
  // A–F component scores (each 0..2); G (personalRelativity) is 0, −1, or −2.
  // Total = sum(A..G) clamped to 0..12.
  // A: quantification  -> numbers, percentages, money, quantities
  // B: timeSpecificity -> explicit dates/times or weaker temporal wording
  // C: locationScope   -> geographic/organizational scope entities
  // D: definedTerms    -> concrete noun-phrase definition clarity
  // E: sourceClarity   -> attribution phrases and source/org mentions
  // F: falsifiability  -> testable phrasing; equality/comparison boosts score (vs modal/subjective)
  // G: personalRelativity -> first-person/inclusive + optional experiential phrasing (penalty)
  const breakdown = {
    quantification: clamp(scoreQuantification(text, nerEnts, signals)),
    timeSpecificity: clamp(scoreTime(text, nerEnts, signals)),
    locationScope: clamp(scoreLocationScope(text, nerEnts, signals)),
    definedTerms: clamp(scoreDefinitionClarity(text, signals, doc, nerEnts)),
    sourceClarity: clamp(scoreSource(text, nerEnts, signals)),
    falsifiability: clamp(scoreFalsifiability(text, nerEnts, signals, doc)),
    personalRelativity: scorePersonalRelativity(text, signals),
  };
  let total = Object.values(breakdown).reduce((a, b) => a + b, 0);
  total = Math.max(0, Math.min(12, total));
  return {
    total,
    breakdown,
    threshold: EXACTITUDE_THRESHOLD,
    passesThreshold: total >= EXACTITUDE_THRESHOLD,
    signals,
    /** Normalized NER spans (model entity labels) for debugging. */
    nerEntities: nerEnts.map((e) => ({
      text: e.text,
      label: e.label,
      ...(e.start != null && e.end != null ? { start: e.start, end: e.end } : {}),
    })),
  };
}

async function runNerOnSentence(ner, text) {
  const raw = await ner(text || "", { aggregation_strategy: "simple" });
  return raw;
}

/**
 * @param {string[]} sentences
 */
async function scoreExactitudeBatch(ner, sentences) {
  const results = [];
  for (const s of sentences) {
    const raw = await runNerOnSentence(ner, s);
    const doc = nlp(s || "");
    const scored = scoreSentenceFromNer(s || "", raw, doc);
    results.push({
      total: scored.total,
      breakdown: scored.breakdown,
      threshold: scored.threshold ?? EXACTITUDE_THRESHOLD,
      passesThreshold: scored.passesThreshold,
      signals: { ...scored.signals, exactitudeVersion: EXACTITUDE_VERSION },
      nerEntities: scored.nerEntities,
    });
  }
  return results;
}

/**
 * Score all sentences in-browser (no API key).
 * @param {string[]} sentences
 */
export async function fetchExactitudeScores(sentences) {
  if (!Array.isArray(sentences) || sentences.length === 0) {
    return [];
  }
  const ner = await getNerPipeline();
  const out = [];
  for (let i = 0; i < sentences.length; i += BATCH_SIZE) {
    const batch = sentences.slice(i, i + BATCH_SIZE);
    const part = await scoreExactitudeBatch(ner, batch);
    out.push(...part);
    // Yield to allow clicks/React events during longer scoring runs.
    await new Promise((r) => setTimeout(r, 0));
  }
  return out;
}
