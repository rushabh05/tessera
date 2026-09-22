// Parity test: does the hand-written JS forward pass reproduce the REAL trained
// PyTorch model's outputs? This is the demo's entire credibility claim - if this
// fails, nothing else about the page can be trusted, so it is deliberately strict
// rather than merely "close enough".
//
// Run: node web/parity.test.mjs
// (or via `npm test` inside web/, see package.json)

import { readFileSync } from "node:fs";
import { forward } from "./js/forward.js";

const weights = JSON.parse(readFileSync(new URL("./data/weights.json", import.meta.url)));
const golden = JSON.parse(readFileSync(new URL("./data/golden.json", import.meta.url))).vectors;

// A score tolerance this tight is achievable ONLY because both sides use the
// identical tanh-GELU formula and the identical float32-min mask sentinel -
// see forward.js's header comment. A looser tolerance here would be hiding a
// real, fixable mismatch rather than accounting for float64-vs-float32
// arithmetic noise (which this small a model does not meaningfully accumulate).
const SCORE_TOL = 1e-4;
const ATTRIBUTION_TOL = 1e-4;

let nPass = 0;
let maxScoreDelta = 0;
let maxAttributionDelta = 0;
const failures = [];

for (let i = 0; i < golden.length; i++) {
  const v = golden[i];
  const { score, attribution } = forward(v.input, v.availability, weights);

  const scoreDelta = Math.abs(score - v.expected_score);
  maxScoreDelta = Math.max(maxScoreDelta, scoreDelta);

  let attrDelta = 0;
  for (let m = 0; m < attribution.length; m++) {
    attrDelta = Math.max(attrDelta, Math.abs(attribution[m] - v.expected_attribution[m]));
  }
  maxAttributionDelta = Math.max(maxAttributionDelta, attrDelta);

  const ok = scoreDelta < SCORE_TOL && attrDelta < ATTRIBUTION_TOL;
  if (ok) {
    nPass++;
  } else {
    failures.push({ i, scoreDelta, attrDelta, got: score, expected: v.expected_score });
  }
}

console.log(`PARITY TEST: ${nPass}/${golden.length} vectors passed`);
console.log(`  max score delta:       ${maxScoreDelta.toExponential(3)} (tolerance ${SCORE_TOL})`);
console.log(`  max attribution delta: ${maxAttributionDelta.toExponential(3)} (tolerance ${ATTRIBUTION_TOL})`);

// Also check: at least one golden vector must exercise a MASKED (absent)
// modality, or the masking path (the trickiest part to port correctly) is
// never actually tested by this run.
const nWithAbsence = golden.filter((v) => v.availability.some((a) => a < 0.5)).length;
console.log(`  vectors exercising modality absence: ${nWithAbsence}/${golden.length}`);
if (nWithAbsence === 0) {
  console.error("FAIL: no golden vector has an absent modality - masking path untested");
  process.exit(1);
}

if (failures.length > 0) {
  console.error(`\nFAILURES (${failures.length}):`);
  for (const f of failures.slice(0, 5)) {
    console.error(
      `  vector ${f.i}: score ${f.got.toFixed(6)} vs expected ${f.expected.toFixed(6)} (delta ${f.scoreDelta.toExponential(3)})`
    );
  }
  process.exit(1);
}

console.log("\nPARITY CONFIRMED: the browser model matches the real trained PyTorch model.");
