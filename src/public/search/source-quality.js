// Given search results (normalized), rank source quality
/*
    Metrics:
            The formula:
            AdFontes credibility: 0 (lowest ~6) to 64 (highest ~56)
                     bias: -42 (lowest ~-32) to 42 (highest ~32)
            MBFC credibility: 0 (best) to 10 (worst)
                bias: -10 to 10 (0 best; take absolute value)


            step 1
            Normalize bias: ( (abs(AdFontes Bias) * 0.238) + abs(MBFC bias) ) / 2 = average bias
            if MBFC null, skip second addition & division
            bias then equals (abs(average bias - 10)), highest score becomes 10 (best)

            step 2:
            Normalize credibility: ((AdFontes * 0.15625) + abs(MBFC - 10)) / 2 = average credibility
            highest score becomes 10 (best)

            step 3:
            bias + credibility * 2.5; highest score becomes 50!

*/

import { pipeline } from '@xenova/transformers';

let credibilityCache = null;
let academicCache = null;

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
    return credibilityCache;
  }
  credibilityCache = await r.json();
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
      return academicCache;
    }
    academicCache = await r.json();
  } catch {
    academicCache = {};
  }
  return academicCache;
}

/** Extract hostname from URL for lookup in credibility.json (keys are domains, e.g. "theguardian.com"). */
function getHostname(url) {
  if (!url || typeof url !== 'string') return '';
  try {
    const u = new URL(url);
    return u.hostname || '';
  } catch {
    return '';
  }
}

/**
 * Source quality from credibility data (0-50). Default 25 when URL not in dataset.
 * Pulls data directly from credibility.json: keys are domains; each entry has
 * credibility.AdFontes.{ bias, credibility } and credibility.MediaBiasFactCheck.{ bias, credibility }.
 * @param {{ url?: string }} source - source object with url
 * @returns {Promise<number>}
 */
export async function sourceQuality(source) {
  const url = source?.url;
  if (!url) return 25;

  const hostname = getHostname(url);
  if (!hostname) return 25;

  // If URL matches an academic source exactly (keys in academicSources.json), give it credibility 42.
  const academic = await loadAcademic();
  if (academic && Object.prototype.hasOwnProperty.call(academic, url)) {
    return 42;
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
  if (socialHosts.has(hostname)) {
    return 0;
  }

  const data = await loadCredibility();
  if (!data || typeof data !== 'object') return 25;

  const entry = data[hostname];
  if (entry == null || !entry.credibility) return 25;

  const adf = entry.credibility.AdFontes;
  const mbfc = entry.credibility.MediaBiasFactCheck;
  const ADFbias = adf && typeof adf.bias === 'number' ? adf.bias : 0;
  const ADFcred = adf && typeof adf.credibility === 'number' ? adf.credibility : 0;
  const MBFCbias = mbfc && typeof mbfc.bias === 'number' ? mbfc.bias : null;
  const MBFCcred = mbfc && typeof mbfc.credibility === 'number' ? mbfc.credibility : null;

  let bias = Math.abs(ADFbias) * 0.238;
  if (MBFCbias !== null) {
    bias = (bias + Math.abs(MBFCbias)) / 2;
  }
  bias = Math.abs(bias - 10);

  let cred = ADFcred * 0.15625;
  if (MBFCcred !== null) {
    cred = (cred + Math.abs(MBFCcred - 10)) / 2;
  }

  let quality = (bias + cred) * 2.5;
  quality = Math.min(50, Math.max(0, quality));
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
 * Similarity score 0-50 based on claim vs source title embedding.
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
  return Math.min(50, Math.max(0, sim * 50));
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
  const data = await loadCredibility();
  if (!data || typeof data !== 'object') return null;
  const hostname = getHostname(url);
  if (!hostname) return null;
  const entry = data[hostname];
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
  const credPart = (quality / 50) * 65;
  const simPart = (similarity / 50) * 25;
  return credPart + simPart + recPart;
}
