/**
 * Claim detection in browser: MiniLM encodings + LR weights (no server).
 * Weights from claim_lr_weights.json at extension root; must run lr_train.py and rebuild.
 */
import { pipeline } from '@xenova/transformers';

let extractorPromise = null;

async function getWeights() {
  const url = typeof chrome !== 'undefined' && chrome.runtime?.getURL
    ? chrome.runtime.getURL('claim_lr_weights.json')
    : '/claim_lr_weights.json';
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error('Weights file not found. Run lr_train.py and rebuild (npm run build).');
  }
  return r.json();
}

/**
 * @param {string[]} clauses
 * @returns {Promise<string[]>} clauses classified as claims
 */
export async function runClaimDetectionInBrowser(clauses) {
  if (!clauses || clauses.length === 0) return [];

  const weights = await getWeights();
  const coef = weights.coef?.[0];
  const intercept = weights.intercept?.[0];
  if (!coef || intercept === undefined) {
    throw new Error('Invalid weights file (missing coef/intercept).');
  }

  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  const extractor = await extractorPromise;
  const output = await extractor(clauses, { pooling: 'mean', normalize: true });

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

  const claims = [];
  const dim = coef.length;
  for (let i = 0; i < clauses.length; i++) {
    const emb = Array.isArray(embeddings[i]) ? embeddings[i] : embeddings.slice(i * dim, (i + 1) * dim);
    let score = intercept;
    for (let j = 0; j < coef.length; j++) {
      score += coef[j] * (emb[j] ?? 0);
    }
    if (score >= 0) claims.push(clauses[i]);
  }
  return claims;
}
