// Pure helpers exported by the Training Lab UI (js/lab/lab-ui.js): the
// false-alert arithmetic behind the deployment sentence, the test-support pill
// (never a leakage statement) and the pooled-folds labelling of LORO results.
import test from "node:test";
import assert from "node:assert/strict";
import { falseAlertRate, alertCountText, supportInfo, pooledLabel, scoredRows } from "../js/lab/lab-ui.js";

test("falseAlertRate: the rate is fp / benign scaled to 1,440 windows a day", () => {
  const r = falseAlertRate(5, 832);
  assert.equal(r.measurable, true);
  assert.equal(r.benign, 832);
  assert.ok(Math.abs(r.rate - (5 / 832) * 1440) < 1e-12);
  assert.equal(r.upper, null, "no rule-of-three bound when something was flagged");
});

test("falseAlertRate: fp = 0 gives the 95% rule-of-three upper bound, not 'no false alerts'", () => {
  const r = falseAlertRate(0, 832);
  assert.equal(r.rate, 0);
  assert.ok(Math.abs(r.upper - (3 / 832) * 1440) < 1e-12);
  assert.equal(alertCountText(r.upper), "5.2");
});

test("falseAlertRate: no benign windows is not measurable", () => {
  const r = falseAlertRate(0, 0);
  assert.equal(r.measurable, false);
  assert.equal(r.rate, null);
  assert.equal(r.upper, null);
});

test("alertCountText: one decimal under 10, whole numbers above", () => {
  assert.equal(alertCountText(8.65), "8.7");
  assert.equal(alertCountText(1), "1");
  assert.equal(alertCountText(0.29), "0.3");
  assert.equal(alertCountText(81.3), "81");
  assert.equal(alertCountText(1234.4), "1,234");
  assert.equal(alertCountText(NaN), "—");
});

test("supportInfo: null when the test set has enough attacks", () => {
  assert.equal(supportInfo(null), null);
  assert.equal(supportInfo({ support: "ok", overall: "fail" }), null);
});

test("supportInfo: a single split without attacks is 'No attacks to score', a warning, never a leak", () => {
  const s0 = supportInfo({ support: "fail", overall: "ok", testPositives: 0 });
  assert.equal(s0.word, "No attacks to score");
  assert.equal(s0.cls, "warn");
  assert.doesNotMatch(s0.word, /leak/i);
  assert.equal(supportInfo({ support: "warn", overall: "ok", testPositives: 7 }).word, "Too few attacks to score");
});

test("supportInfo: a merged LORO certificate names the excluded folds", () => {
  const merged = { support: "warn", overall: "ok", excludedLowSupport: ["shaw"], perFold: [{ heldOut: "shaw" }] };
  const s0 = supportInfo(merged);
  assert.equal(s0.word, "1 fold has too few attacks to score");
  assert.match(s0.short, /^shaw: /);
  const two = supportInfo({ ...merged, excludedLowSupport: ["shaw", "fox"] });
  assert.equal(two.word, "2 folds have too few attacks to score");
  assert.equal(supportInfo({ ...merged, support: "fail" }).word, "No fold has enough attacks to score");
});

test("pooledLabel: only LORO results are labelled, with the fold and model counts", () => {
  assert.equal(pooledLabel({ mode: "replica", metricsScope: { kind: "single-split" } }), null);
  assert.equal(pooledLabel({ mode: "loro", metricsScope: { kind: "pooled-folds", nFolds: 7, nModels: 7 } }), "pooled over 7 scoreable folds (7 models)");
  assert.equal(pooledLabel({ mode: "loro", metricsScope: { kind: "pooled-folds", nFolds: 1, nModels: 1 } }), "pooled over 1 scoreable fold (1 model)");
  assert.equal(pooledLabel({ mode: "loro" }), "pooled over the scoreable folds");
});

test("scoredRows: a single split keeps every row; LORO keeps only test.included rows", () => {
  const y = Uint8Array.from([0, 1, 0, 1, 0]);
  const scores = Float32Array.from([0.1, 0.9, 0.2, 0.8, 0.3]);
  const single = scoredRows({ mode: "replica", test: { y, scores, n: 5 } });
  assert.equal(single.n, 5);
  assert.equal(single.y, y);
  const included = Uint8Array.from([1, 1, 0, 0, 1]);
  const loro = { mode: "loro", test: { y, scores, n: 5, included } };
  const rows = scoredRows(loro);
  assert.equal(rows.n, 3);
  assert.deepEqual(Array.from(rows.y), [0, 1, 0]);
  assert.deepEqual(Array.from(rows.scores), [scores[0], scores[1], scores[4]]);
  assert.equal(scoredRows(loro), rows, "cached per result");
  assert.equal(scoredRows({ mode: "loro" }), null);
});
