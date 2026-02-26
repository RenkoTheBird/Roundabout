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

  const data = await loadCredibility();
  if (!data || typeof data !== 'object') return 25;

  const hostname = getHostname(url);
  if (!hostname) return 25;

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

/**
 * Total source score 0-100 (quality + similarity).
 * @param {string} claim
 * @param {{ title?: string; url?: string }} source
 * @returns {Promise<number>}
 */
export async function sourceChecker(claim, source) {
  const quality = await sourceQuality(source);
  const similarity = await sourceSimilarity(source, claim);
  return quality + similarity;
}
