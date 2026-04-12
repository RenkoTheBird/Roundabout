/**
 * Claim detection in the browser using Exactitude only (NER + compromise + heuristics).
 */
import { fetchExactitudeScores, EXACTITUDE_THRESHOLD } from '../search/exactitude.js';

/**
 * @param {string[]} clauses
 * @returns {Promise<Array<{
 *   clause: string,
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

  const exactitudes = await fetchExactitudeScores(clauses);

  const clauseDecisions = [];
  for (let i = 0; i < clauses.length; i++) {
    const exactitude = exactitudes[i];
    clauseDecisions.push({
      clause: clauses[i],
      exactitudeTotal: exactitude.total,
      exactitudeBreakdown: exactitude.breakdown,
      exactitudeSignals: exactitude.signals,
      exactitudeThreshold: EXACTITUDE_THRESHOLD,
      nerEntities: exactitude.nerEntities ?? [],
      isClaimFinal: exactitude.total >= EXACTITUDE_THRESHOLD,
    });
  }
  return clauseDecisions;
}
