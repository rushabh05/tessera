// metrics.js must reproduce scikit-learn / tessera.eval exactly. The expected
// values in web/data/metrics_golden.json were computed by sklearn itself
// (`uv run python -m tessera.demo.golden_metrics`); every scalar and every curve
// element must agree to 1e-12.
//
// Run: node --test "web/tests/*.test.mjs"   (Node 24 does not accept a bare directory)

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  averagePrecision,
  rocAuc,
  prCurve,
  rocCurve,
  confusionAt,
  metricsAt,
  ece,
  classConditionalEce,
  histogram,
  summarise,
  evaluateAll,
  MIN_SUPPORT_FOR_RATES,
} from "../js/lab/metrics.js";

const TOL = 1e-12;
const golden = JSON.parse(readFileSync(new URL("../data/metrics_golden.json", import.meta.url)));

const num = (v) => (v === "inf" ? Infinity : v === "-inf" ? -Infinity : v);

function close(actual, expected, label) {
  expected = num(expected);
  if (expected === null || expected === undefined) {
    assert.equal(actual, null, `${label}: expected null, got ${actual}`);
    return;
  }
  assert.equal(typeof actual, "number", `${label}: expected a number, got ${actual}`);
  if (!Number.isFinite(expected)) {
    assert.equal(actual, expected, label);
    return;
  }
  assert.ok(Number.isFinite(actual), `${label}: got non-finite ${actual}`);
  assert.ok(
    Math.abs(actual - expected) <= TOL,
    `${label}: got ${actual}, expected ${expected} (|diff| ${Math.abs(actual - expected)})`,
  );
}

function closeArray(actual, expected, label) {
  assert.ok(actual, `${label}: missing`);
  assert.equal(actual.length, expected.length, `${label}: length ${actual.length} vs ${expected.length}`);
  for (let i = 0; i < expected.length; i++) close(actual[i], expected[i], `${label}[${i}]`);
}

function inputs(c, { typed = false } = {}) {
  const y = typed ? Uint8Array.from(c.y) : c.y.slice();
  const s = c.float32 ? Float32Array.from(c.s) : typed ? Float64Array.from(c.s) : c.s.slice();
  return { y, s };
}

test("golden file covers the required awkward cases", () => {
  const names = golden.cases.map((c) => c.name);
  for (const required of [
    "heavy_ties",
    "all_equal",
    "single_positive",
    "perfect_separation",
    "perfectly_inverted",
    "at_threshold",
    "tiny_n2",
    "large_float32",
  ]) {
    assert.ok(names.includes(required), `missing golden case ${required}`);
  }
  assert.ok(golden.cases.length >= 12);
  assert.ok(golden.cases.find((c) => c.name === "large_float32").n >= 3000);
});

for (const c of golden.cases) {
  for (const typed of [false, true]) {
    test(`${c.name} (${typed ? "typed arrays" : "plain arrays"}) matches sklearn`, () => {
      const { y, s } = inputs(c, { typed });
      close(averagePrecision(y, s), c.ap, "ap");
      close(rocAuc(y, s), c.roc_auc, "roc_auc");

      const pr = prCurve(y, s);
      closeArray(pr.precision, c.pr.precision, "pr.precision");
      closeArray(pr.recall, c.pr.recall, "pr.recall");
      closeArray(pr.thresholds, c.pr.thresholds, "pr.thresholds");

      const roc = rocCurve(y, s);
      closeArray(roc.fpr, c.roc.fpr, "roc.fpr");
      closeArray(roc.tpr, c.roc.tpr, "roc.tpr");
      closeArray(roc.thresholds, c.roc.thresholds, "roc.thresholds");

      for (const exp of c.at_threshold) {
        const t = exp.threshold;
        assert.deepEqual(confusionAt(y, s, t), { tp: exp.tp, fp: exp.fp, tn: exp.tn, fn: exp.fn }, `confusion@${t}`);
        const m = metricsAt(y, s, t);
        close(m.precision, exp.precision, `precision@${t}`);
        close(m.recall, exp.recall, `recall@${t}`);
        close(m.f1, exp.f1, `f1@${t}`);
        close(m.mcc, exp.mcc, `mcc@${t}`);
        close(m.accuracy, exp.accuracy, `accuracy@${t}`);
        close(m.balancedAccuracy, exp.balanced_accuracy, `balancedAccuracy@${t}`);
        close(m.specificity, exp.tn + exp.fp ? exp.tn / (exp.tn + exp.fp) : 0, `specificity@${t}`);
        close(m.fpr, exp.fp + exp.tn ? exp.fp / (exp.fp + exp.tn) : 0, `fpr@${t}`);
      }

      close(ece(y, s, c.ece.n_bins), c.ece.aggregate, "ece");
      const cc = classConditionalEce(y, s, c.ece.n_bins);
      close(cc.aggregate, c.ece.aggregate, "ece.aggregate");
      close(cc.positive, c.ece.positive, "ece.positive");
      close(cc.negative, c.ece.negative, "ece.negative");

      const h = histogram(y, s, c.histogram.n_bins);
      closeArray(h.edges, c.histogram.edges, "hist.edges");
      assert.deepEqual(h.benign, c.histogram.benign, "hist.benign");
      assert.deepEqual(h.attack, c.histogram.attack, "hist.attack");

      const all = evaluateAll(y, s, { threshold: 0.5 });
      close(all.ap, c.ap, "evaluateAll.ap");
      close(all.rocAuc, c.roc_auc, "evaluateAll.rocAuc");
      assert.equal(all.n, c.n);
      assert.equal(all.nPositive, c.n_positive);
      close(all.prevalence, c.n_positive / c.n, "evaluateAll.prevalence");
      close(all.atThreshold.mcc, c.at_threshold[0].mcc, "evaluateAll.mcc");
      closeArray(all.pr.precision, c.pr.precision, "evaluateAll.pr.precision");
      closeArray(all.roc.tpr, c.roc.tpr, "evaluateAll.roc.tpr");
      close(all.ece, c.ece.aggregate, "evaluateAll.ece");
      assert.equal(all.lowSupport, c.n_positive < MIN_SUPPORT_FOR_RATES);
    });
  }
}

for (const sc of golden.summaries) {
  test(`summarise(${sc.name}) matches tessera.eval.stats.summarise_seeds`, () => {
    const got = summarise(sc.values);
    assert.equal(got.n, sc.expected.n_seeds);
    close(got.mean, sc.expected.mean, "mean");
    close(got.std, sc.expected.std, "std");
    if ("min" in sc.expected) {
      close(got.min, sc.expected.min, "min");
      close(got.max, sc.expected.max, "max");
    } else {
      assert.equal(got.min, null);
      assert.equal(got.max, null);
    }
  });
}

test("single-class and empty inputs give null ranking metrics, never NaN or a throw", () => {
  const inputsList = [
    { y: [0, 0, 0, 0], s: [0.1, 0.9, 0.5, 0.5] },
    { y: Uint8Array.from([1, 1, 1]), s: Float32Array.from([0.2, 0.7, 0.7]) },
    { y: [], s: [] },
    { y: new Uint8Array(0), s: new Float64Array(0) },
  ];
  for (const { y, s } of inputsList) {
    assert.equal(averagePrecision(y, s), null);
    assert.equal(rocAuc(y, s), null);
    assert.equal(prCurve(y, s), null);
    assert.equal(rocCurve(y, s), null);
    const all = evaluateAll(y, s);
    assert.equal(all.ap, null);
    assert.equal(all.rocAuc, null);
    assert.equal(all.pr, null);
    assert.equal(all.roc, null);
    const walk = (v, path) => {
      if (typeof v === "number") assert.ok(!Number.isNaN(v), `NaN at ${path}`);
      else if (v && typeof v === "object") for (const [k, w] of Object.entries(v)) walk(w, `${path}.${k}`);
    };
    walk(all, "evaluateAll");
    assert.equal(all.lowSupport, true);
  }
});

test("single-class threshold metrics follow sklearn's zero-division=0 convention", () => {
  const m = metricsAt([0, 0, 0], [0.9, 0.1, 0.2], 0.5);
  assert.deepEqual(m.confusion, { tp: 0, fp: 1, tn: 2, fn: 0 });
  assert.equal(m.precision, 0);
  assert.equal(m.recall, 0);
  assert.equal(m.f1, 0);
  assert.equal(m.mcc, 0);
  // sklearn balanced_accuracy_score averages recall over the classes present in y
  close(m.balancedAccuracy, 2 / 3, "balancedAccuracy");
  assert.equal(summarise([]).mean, null);
  assert.equal(summarise([0.3]).std, 0);
});

test("mismatched lengths and non-finite scores are rejected loudly", () => {
  assert.throws(() => averagePrecision([0, 1], [0.5]), RangeError);
  assert.throws(() => averagePrecision([0, 1], [0.5, NaN]), RangeError);
});

test("evaluateAll on 50k float32 scores is fast (one sort)", () => {
  const n = 50000;
  const y = new Uint8Array(n);
  const s = new Float32Array(n);
  let state = 12345;
  const rnd = () => {
    state = (Math.imul(state, 1103515245) + 12345) >>> 0;
    return state / 4294967296;
  };
  for (let i = 0; i < n; i++) {
    y[i] = rnd() < 0.05 ? 1 : 0;
    s[i] = Math.round((rnd() * 0.6 + 0.4 * y[i]) * 500) / 500; // heavy ties
  }
  const t0 = performance.now();
  const all = evaluateAll(y, s);
  const ms = performance.now() - t0;
  assert.ok(all.ap > 0 && all.ap <= 1);
  assert.ok(all.rocAuc > 0.5 && all.rocAuc <= 1);
  assert.ok(ms < 1500, `evaluateAll took ${ms.toFixed(0)} ms`);
});
