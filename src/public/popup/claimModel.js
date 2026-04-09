/**
 * Claim detection in browser: MiniLM encodings + LR weights (no server).
 * Weights from claim_lr_weights.json at extension root; must run lr_train.py and rebuild.
 */
import { pipeline } from '@xenova/transformers';
import { fetchExactitudeScores, EXACTITUDE_THRESHOLD } from '../search/exactitude.js';

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
 * @returns {Promise<Array<{
 *   clause: string,
 *   lrScore: number,
 *   lrIsClaim: boolean,
 *   exactitudeTotal: number,
 *   exactitudeBreakdown: {
 *     quantification: number,
 *     timeSpecificity: number,
 *     locationScope: number,
 *     definedTerms: number,
 *     sourceClarity: number,
 *     falsifiability: number,
 *     personalRelativity: number (0, -1, or -2)
 *   },
 *   exactitudeSignals: Record<string, unknown>,
 *   exactitudeThreshold: number,
 *   nerEntities: Array<{ text: string, label: string, start?: number, end?: number }>,
 *   isClaimFinal: boolean
 * }>}
 */
export async function runClaimDetectionInBrowser(clauses) {
  if (!clauses || clauses.length === 0) return [];

  const weights = await getWeights();
  const coef = weights.coef?.[0];
  const intercept = weights.intercept?.[0];
  const scalerMean = weights.scaler_mean;
  const scalerScale = weights.scaler_scale;
  if (!coef || intercept === undefined) {
    throw new Error('Invalid weights file (missing coef/intercept).');
  }

  const useScaling =
    Array.isArray(scalerMean) &&
    Array.isArray(scalerScale) &&
    scalerMean.length === coef.length &&
    scalerScale.length === coef.length;

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

  const dim = coef.length;
  const lrPart = [];
  for (let i = 0; i < clauses.length; i++) {
    const emb = Array.isArray(embeddings[i]) ? embeddings[i] : embeddings.slice(i * dim, (i + 1) * dim);
    let score = intercept;
    for (let j = 0; j < coef.length; j++) {
      let x = emb[j] ?? 0;
      if (useScaling) {
        x = (x - scalerMean[j]) / (scalerScale[j] ?? 1);
      }
      score += coef[j] * x;
    }
    lrPart.push({ score, lrIsClaim: score >= 0 });
  }

  const exactitudes = await fetchExactitudeScores(clauses);

  const clauseDecisions = [];
  for (let i = 0; i < clauses.length; i++) {
    const { score, lrIsClaim } = lrPart[i];
    const exactitude = exactitudes[i];
    clauseDecisions.push({
      clause: clauses[i],
      lrScore: score,
      lrIsClaim,
      exactitudeTotal: exactitude.total,
      exactitudeBreakdown: exactitude.breakdown,
      exactitudeSignals: exactitude.signals,
      exactitudeThreshold: EXACTITUDE_THRESHOLD,
      nerEntities: exactitude.nerEntities ?? [],
      isClaimFinal: lrIsClaim && exactitude.total >= EXACTITUDE_THRESHOLD,
    });
  }
  return clauseDecisions;
}
