/**
 * Exactitude self-check: in-browser NER (Xenova) + compromise (same as extension).
 *
 * First run downloads the NER ONNX model from Hugging Face CDN (no API key).
 *
 *   node scripts/exactitude-selfcheck.mjs
 */

const { fetchExactitudeScores } = await import(
  new URL("../src/public/search/exactitude.js", import.meta.url)
);

const fixtures = [
  { sentence: "Inflation is up.", maxTotal: 12 },
  { sentence: "US inflation rose 5% in 2025.", minTotal: 4 },
  {
    sentence:
      "According to the U.S. Bureau of Labor Statistics, CPI rose 5% in 2025 in the United States.",
    minTotal: 6,
  },
  { sentence: "Inflation might be rising.", maxTotal: 12 },
  {
    sentence:
      "people in alberta pay the exact same federal tax rates that people in quebec pay.",
    minTotal: 8,
  },
];

const data = await fetchExactitudeScores(fixtures.map((f) => f.sentence));

let failed = 0;
data.forEach((r, i) => {
  const f = fixtures[i];
  let ok = true;
  if (f.minTotal != null && r.total < f.minTotal) ok = false;
  if (f.maxTotal != null && r.total > f.maxTotal) ok = false;
  if (!ok) failed += 1;
  console.log(
    `${ok ? "PASS" : "FAIL"} | total=${r.total} | ${f.sentence.slice(0, 60)}...`
  );
});

if (failed > 0) {
  console.error(`\nSelf-check failed: ${failed} case(s).`);
  process.exit(1);
}
console.log("\nExactitude self-check passed.");
