// Given search results (normalized), rank source quality
/*
    Metrics:
            The formula:
            AdFontes credibility: 0 (lowest ~6) to 64 (highest ~50)
                     bias: -42 (lowest ~-30) to 42 (highest ~30)
            MBFC credibility: 0 (best) to 10 (worst)
                bias: -10 to 10 (0 best; take absolute value)


            step 1
            Normalize bias: ( (abs(AdFontes Bias) * 0.333) + abs(MBFC bias) ) / 2 = average bias
            if MBFC null, skip second addition & division
            bias then equals (abs(average bias - 10)), highest score becomes 10 (best)

            step 2:
            Normalize credibility: ((AdFontes * 0.20) + abs(MBFC - 10)) / 2 = average credibility
            highest score becomes 10 (best)

            step 3:
            bias + credibility * 2.5; highest score becomes 50!

*/

import { pipeline } from '@xenova/transformers';

let credibilityCache = null;
let academicCache = null;
let govCache = null;
let credibilityKeys = null;
let credibilityKeysLower = null;
let academicKeys = null;
let academicKeysLower = null;
let govKeys = null;
let govKeysLower = null;
let credibilityEntryCacheByUrl = new Map();
let academicEntryCacheByUrl = new Map();
let govEntryCacheByUrl = new Map();

function getCredibilityUrl() {
  return typeof chrome !== 'undefined' && chrome.runtime?.getURL
    ? chrome.runtime.getURL('datasets/credibility.json')
    : '/datasets/credibility.json';
}

async function loadCredibility() {
  if (credibilityCache !== null) return credibilityCache;
  const url = getCredibilityUrl();
  const r = await fetch(url);
  if (!r.ok) {
    credibilityCache = {};
    credibilityKeys = [];
    credibilityKeysLower = [];
    credibilityEntryCacheByUrl.clear();
    return credibilityCache;
  }
  credibilityCache = await r.json();
  credibilityKeys = Object.keys(credibilityCache || {});
  credibilityKeysLower = credibilityKeys.map((k) => (typeof k === 'string' ? k.toLowerCase() : String(k)));
  credibilityEntryCacheByUrl.clear();
  return credibilityCache;
}

function getAcademicUrl() {
  return typeof chrome !== 'undefined' && chrome.runtime?.getURL
    ? chrome.runtime.getURL('datasets/academicSources.json')
    : '/datasets/academicSources.json';
}

async function loadAcademic() {
  if (academicCache !== null) return academicCache;
  const url = getAcademicUrl();
  try {
    const r = await fetch(url);
    if (!r.ok) {
      academicCache = {};
      academicKeys = [];
      academicKeysLower = [];
      academicEntryCacheByUrl.clear();
      return academicCache;
    }
    academicCache = await r.json();
  } catch {
    academicCache = {};
  }
  academicKeys = Object.keys(academicCache || {});
  academicKeysLower = academicKeys.map((k) => (typeof k === 'string' ? k.toLowerCase() : String(k)));
  academicEntryCacheByUrl.clear();
  return academicCache;
}

function getGovUrl() {
  return typeof chrome !== 'undefined' && chrome.runtime?.getURL
    ? chrome.runtime.getURL('datasets/govSources.json')
    : '/datasets/govSources.json';
}

async function loadGov() {
  if (govCache !== null) return govCache;
  const url = getGovUrl();
  try {
    const r = await fetch(url);
    if (!r.ok) {
      govCache = {};
      govKeys = [];
      govKeysLower = [];
      govEntryCacheByUrl.clear();
      return govCache;
    }
    govCache = await r.json();
  } catch {
    govCache = {};
  }
  govKeys = Object.keys(govCache || {});
  govKeysLower = govKeys.map((k) => (typeof k === 'string' ? k.toLowerCase() : String(k)));
  govEntryCacheByUrl.clear();
  return govCache;
}

/** Extract hostname from URL for lookup in credibility.json (keys are domains, e.g. "theguardian.com"). */
function getHostname(url) {
  if (!url || typeof url !== 'string') return '';
  try {
    const u = new URL(url);
    return (u.hostname || '').toLowerCase();
  } catch {
    // Some inputs may omit scheme (e.g. "www.example.com/path").
    const cleaned = url.trim();
    const hasScheme = /^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//.test(cleaned);
    if (!hasScheme) {
      try {
        const u = new URL(`https://${cleaned}`);
        return (u.hostname || '').toLowerCase();
      } catch {
        return '';
      }
    }
    return '';
  }
}

function stripWww(hostname) {
  if (!hostname) return '';
  return hostname.replace(/^www\./i, '');
}

function normalizeUrlForLookup(url) {
  if (!url || typeof url !== 'string') return '';
  return url.trim().toLowerCase();
}

function normalizeLookupKey(key) {
  if (!key || typeof key !== 'string') return '';
  const lowered = key.trim().toLowerCase();
  return lowered.replace(/^[a-z][a-z0-9+.-]*:\/\//, '').replace(/^www\./, '');
}

function findEntryBySubstring(url, data, keys, keysLower, cache) {
  if (!data || typeof data !== 'object') return null;

  const urlLower = normalizeUrlForLookup(url);
  if (!urlLower) return null;

  if (cache.has(urlLower)) {
    return cache.get(urlLower) ?? null;
  }

  const hostnameRaw = getHostname(url);
  const hostname = stripWww(hostnameRaw);
  const normalizedUrl = normalizeLookupKey(urlLower);
  let found = null;

  for (let i = 0; i < (keysLower?.length ?? 0); i++) {
    const key = keys?.[i];
    const keyLower = keysLower[i];
    if (!keyLower || typeof key !== 'string') continue;

    const normalizedKey = normalizeLookupKey(keyLower);
    if (!normalizedKey) continue;

    if (urlLower.includes(keyLower) || normalizedUrl.includes(normalizedKey)) {
      found = data[key];
      break;
    }
    if (
      hostnameRaw === normalizedKey ||
      hostname === normalizedKey ||
      hostnameRaw.includes(normalizedKey) ||
      hostname.includes(normalizedKey)
    ) {
      found = data[key];
      break;
    }
  }

  cache.set(urlLower, found);
  return found;
}

async function findCredibilityEntry(url) {
  const data = await loadCredibility();
  if (!data || typeof data !== 'object') return null;

  const urlLower = normalizeUrlForLookup(url);
  if (!urlLower) return null;

  // Cache by normalized URL string; avoids scanning all credibility keys repeatedly.
  if (credibilityEntryCacheByUrl.has(urlLower)) {
    return credibilityEntryCacheByUrl.get(urlLower) ?? null;
  }

  const hostnameRaw = getHostname(url);
  const hostnameCandidates = [hostnameRaw, stripWww(hostnameRaw)].filter(Boolean);
  // Try exact hostname matches first (most common case).
  for (const host of hostnameCandidates) {
    const entry = data[host];
    if (entry && entry.credibility) {
      credibilityEntryCacheByUrl.set(urlLower, entry);
      return entry;
    }
  }

  // Fall back to substring match (requested behavior).
  // This handles cases like:
  // - "https://www.upi.com/..." matching key "upi.com"
  // - keys that include a path segment, like "afp.com/en"
  let found = null;
  for (let i = 0; i < (credibilityKeysLower?.length ?? 0); i++) {
    const keyLower = credibilityKeysLower[i];
    const key = credibilityKeys[i];
    if (!keyLower || typeof key !== 'string') continue;

    if (urlLower.includes(keyLower)) {
      found = data[key];
      break;
    }
    for (const host of hostnameCandidates) {
      if (host.includes(keyLower)) {
        found = data[key];
        break;
      }
    }
    if (found) break;
  }

  credibilityEntryCacheByUrl.set(urlLower, found);
  return found;
}

/**
 * Source quality from credibility data (0-65). Default 32 when URL not in dataset.
 * Pulls data directly from credibility.json: keys are domains; each entry has
 * credibility.AdFontes.{ bias, credibility } and credibility.MediaBiasFactCheck.{ bias, credibility }.
 * @param {{ url?: string }} source - source object with url
 * @returns {Promise<number>}
 */
export async function sourceQuality(source) {
  const url = source?.url;
  if (!url) return 32;

  const hostname = getHostname(url);
  if (!hostname) return 32;

  // Government sources use max credibility bucket (65).
  const gov = await loadGov();
  const govEntry = findEntryBySubstring(url, gov, govKeys, govKeysLower, govEntryCacheByUrl);
  if (govEntry) {
    return 65;
  }

  // Academic default credibility remains 58.
  const academic = await loadAcademic();
  const academicEntry = findEntryBySubstring(url, academic, academicKeys, academicKeysLower, academicEntryCacheByUrl);
  if (academicEntry) {
    return 58;
  }

  // If the source is a social platform (Reddit, X/Twitter, Instagram), its credibility is 0.
  const socialHosts = new Set([
    'reddit.com',
    'www.reddit.com',
    'old.reddit.com',
    'np.reddit.com',
    'twitter.com',
    'www.twitter.com',
    'mobile.twitter.com',
    'x.com',
    'www.x.com',
    't.co',
    'instagram.com',
    'www.instagram.com',
  ]);
  if (socialHosts.has(hostname) || socialHosts.has(stripWww(hostname))) {
    return 0;
  }

  const entry = await findCredibilityEntry(url);
  if (entry == null || !entry.credibility) return 32;

  const adf = entry.credibility.AdFontes;
  const mbfc = entry.credibility.MediaBiasFactCheck;
  const ADFbias = adf && typeof adf.bias === 'number' ? adf.bias : 0;
  const ADFcred = adf && typeof adf.credibility === 'number' ? adf.credibility : 0;
  const MBFCbias = mbfc && typeof mbfc.bias === 'number' ? mbfc.bias : null;
  const MBFCcred = mbfc && typeof mbfc.credibility === 'number' ? mbfc.credibility : null;

  let bias = Math.abs(ADFbias) * 0.333;
  if (MBFCbias !== null) {
    bias = (bias + Math.abs(MBFCbias)) / 2;
  }
  bias = Math.abs(bias - 10);

  let cred = ADFcred * 0.20;
  if (MBFCcred !== null) {
    cred = (cred + Math.abs(MBFCcred - 10)) / 2;
  }

  let quality = (bias + cred) * 3.25;
  quality = Math.max(0, quality);
  return quality;
}

let extractorPromise = null;

function getExtractor() {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return extractorPromise;
}

/** Cosine similarity between two normalized vectors (dot product). */
function cosineSimilarity(a, b) {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return 0;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return Math.max(0, Math.min(1, sum));
}

/**
 * Similarity score 0-25 based on claim vs source title embedding.
 * @param {{ title?: string }} source
 * @param {string} claim
 * @returns {Promise<number>}
 */
export async function sourceSimilarity(source, claim) {
  const title = source?.title ?? '';
  if (!claim?.trim() || !title.trim()) return 0;

  const extractor = await getExtractor();
  const output = await extractor([claim.trim(), title.trim()], { pooling: 'mean', normalize: true });

  let embeddings;
  if (typeof output.tolist === 'function') {
    embeddings = output.tolist();
  } else if (output.data && output.dims) {
    const [n, dim] = output.dims;
    embeddings = [];
    for (let i = 0; i < n; i++) {
      embeddings.push(Array.from(output.data.slice(i * dim, (i + 1) * dim)));
    }
  } else {
    embeddings = Array.from(output.data ?? output);
  }

  const embClaim = Array.isArray(embeddings[0]) ? embeddings[0] : embeddings.slice(0, embeddings.length / 2);
  const embTitle = Array.isArray(embeddings[1]) ? embeddings[1] : embeddings.slice(embeddings.length / 2);
  const sim = cosineSimilarity(embClaim, embTitle);
  return Math.min(25, Math.max(0, sim * 25));
}

/** Parse source date from Brave result (age: ISO 8601 string). Returns Date or null. */
export function getSourceDate(source) {
  const age = source?.age;
  if (age == null || typeof age !== 'string') return null;
  const d = new Date(age);
  return Number.isNaN(d.getTime()) ? null : d;
}

/**
 * Source display name from credibility.json (if present). Keys are domains; entry.name is the label.
 * @param {string} [url]
 * @returns {Promise<string | null>}
 */
export async function getSourceName(url) {
  if (!url || typeof url !== 'string') return null;
  const gov = await loadGov();
  const govEntry = findEntryBySubstring(url, gov, govKeys, govKeysLower, govEntryCacheByUrl);
  if (govEntry?.name) return govEntry.name;

  const academic = await loadAcademic();
  const academicEntry = findEntryBySubstring(url, academic, academicKeys, academicKeysLower, academicEntryCacheByUrl);
  if (academicEntry?.name) return academicEntry.name;

  const entry = await findCredibilityEntry(url);
  return entry?.name ?? null;
}

/** Parse post createdTimestamp (ISO string or ms) to Date or null. */
function getPostDate(post) {
  const ts = post?.createdTimestamp;
  if (ts == null) return null;
  if (typeof ts === 'number') {
    const d = new Date(ts);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  if (typeof ts === 'string') {
    const d = new Date(ts);
    return Number.isNaN(d.getTime()) ? null : d;
  }
  return null;
}

/**
 * Recency score 0-10. Uses max of score vs "now" and score vs "post date".
 * Curve: 0y→10, ~0.5y→9, 1y→8, 2y→6, 5y→0 (linear interpolation). No source date → 5.
 * @param {Date | null} sourceDate
 * @param {Date | null} postDate
 * @returns {number}
 */
function recencyScore(sourceDate, postDate) {
  if (!sourceDate) return 5;

  function scoreForYearsAway(yearsAway) {
    if (yearsAway <= 0) return 10;
    if (yearsAway >= 5) return 0;
    if (yearsAway <= 4) return 10 - 2 * yearsAway;
    return 5;
  }

  const now = Date.now();
  const refs = [now];
  if (postDate) refs.push(postDate.getTime());

  let best = 0;
  for (const refMs of refs) {
    const yearsAway = Math.abs(sourceDate.getTime() - refMs) / (365.25 * 24 * 60 * 60 * 1000);
    best = Math.max(best, scoreForYearsAway(yearsAway));
  }
  return Math.max(0, Math.min(10, best));
}

/**
 * Total source score 0-100 (65% credibility + 25% title similarity + 10% recency).
 * @param {string} claim
 * @param {{ title?: string; url?: string; age?: string }} source
 * @param {{ createdTimestamp?: string | number }?} post - optional; used for recency vs post date
 * @returns {Promise<number>}
 */
export async function sourceChecker(claim, source, post) {
  const quality = await sourceQuality(source);
  const similarity = await sourceSimilarity(source, claim);
  const sourceDate = getSourceDate(source);
  const postDate = getPostDate(post);
  const recPart = recencyScore(sourceDate, postDate);
  return quality + similarity + recPart;
}
