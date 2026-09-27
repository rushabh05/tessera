// knn.js: the nearest-neighbour "memoriser" is exact (equals a naive
// sort-everything reference), deterministic, cancellable, and uses exactly the
// columns of its feature set.
//
// Run: node --test web/tests/knn.test.mjs

import test from "node:test";
import assert from "node:assert/strict";

import { trainKnn, predictKnn, knnWeights } from "../js/lab/knn.js";
import { featureColumns } from "../js/lab/splits.js";
import { makeRng } from "../js/lab/rng.js";

/** Small zero-inflated, heavy-tailed rows like the real features, with many exact ties. */
function rows(seed, n, { ties = true } = {}) {
  const rng = makeRng(seed, "knn-test");
  const X = new Float32Array(n * 42);
  const y = new Uint8Array(n);
  const avail = new Uint8Array(n * 4).fill(1);
  for (let r = 0; r < n; r++) {
    y[r] = rng.bernoulli(0.3) ? 1 : 0;
    for (let j = 0; j < 42; j++) {
      if (rng.bernoulli(0.6)) continue;
      // few distinct values -> exact distance ties are common
      X[r * 42 + j] = ties ? rng.int(3) * (1 + y[r]) : Math.exp(rng.normal() + y[r]);
    }
  }
  return { X, y, avail, n };
}

/** Naive reference: every distance, sort by (distance, index), weight 2^-(j-1). */
function naive(train, test, { k, featureSet }) {
  const cols = featureColumns(featureSet);
  const d = cols.length;
  const lg = (v) => Math.log1p(v > 0 ? v : 0);
  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  for (let r = 0; r < train.n; r++) for (let j = 0; j < d; j++) mean[j] += lg(train.X[r * 42 + cols[j]]);
  for (let j = 0; j < d; j++) mean[j] /= train.n;
  for (let r = 0; r < train.n; r++) {
    for (let j = 0; j < d; j++) {
      const dv = lg(train.X[r * 42 + cols[j]]) - mean[j];
      std[j] += dv * dv;
    }
  }
  for (let j = 0; j < d; j++) std[j] = Math.max(Math.sqrt(std[j] / train.n), 1e-6);
  const z = (X, r) => Array.from(cols, (c, j) => (lg(X[r * 42 + c]) - mean[j]) / std[j]);
  const Z = Array.from({ length: train.n }, (_, r) => z(train.X, r));
  const kk = Math.min(k, train.n);
  const out = new Float64Array(test.n);
  for (let r = 0; r < test.n; r++) {
    const q = z(test.X, r);
    const all = Z.map((row, i) => {
      let s = 0;
      for (let j = 0; j < d; j++) {
        const dv = row[j] - q[j];
        s += dv * dv;
      }
      return { s, i };
    });
    all.sort((a, b) => a.s - b.s || a.i - b.i);
    let num = 0;
    let den = 0;
    for (let j = 0; j < kk; j++) {
      const w = 2 ** -j;
      num += w * train.y[all[j].i];
      den += w;
    }
    out[r] = num / den;
  }
  return out;
}

test("weights halve from the nearest neighbour: 1, 1/2, 1/4, ...", () => {
  assert.deepEqual(Array.from(knnWeights(4)), [1, 0.5, 0.25, 0.125]);
});

for (const featureSet of ["all", "m2"]) {
  for (const k of [1, 3, 5]) {
    for (const ties of [true, false]) {
      test(`exact: equals the naive sort-everything reference (featureSet ${featureSet}, k ${k}, ${ties ? "many ties" : "continuous"})`, async () => {
        const train = rows(1, 180, { ties });
        const test_ = rows(2, 70, { ties });
        const model = trainKnn({ train, config: { k, featureSet } });
        const got = await predictKnn(model, test_);
        const want = naive(train, test_, { k, featureSet });
        assert.equal(got.length, test_.n);
        for (let r = 0; r < test_.n; r++) assert.equal(got[r], want[r], `row ${r}`);
        for (const v of got) assert.ok(v >= 0 && v <= 1);
      });
    }
  }
}

test("a test row that copies a training row scores that row's label under k = 1", async () => {
  const train = rows(3, 120, { ties: false });
  const model = trainKnn({ train, config: { k: 1 } });
  const s = await predictKnn(model, train);
  assert.deepEqual(Array.from(s), Array.from(train.y));
});

test("k larger than the training set uses every training row", async () => {
  const train = rows(4, 3);
  const test_ = rows(5, 10);
  const got = await predictKnn(trainKnn({ train, config: { k: 50 } }), test_);
  assert.deepEqual(Array.from(got), Array.from(naive(train, test_, { k: 50, featureSet: "all" })));
});

test("deterministic: same data, same scores; the stored transform uses training rows only", async () => {
  const train = rows(6, 150);
  const test_ = rows(7, 90);
  const a = await predictKnn(trainKnn({ train, config: { k: 5 } }), test_);
  const b = await predictKnn(trainKnn({ train, config: { k: 5 } }), test_);
  assert.deepEqual(Array.from(a), Array.from(b));
  const m = trainKnn({ train, config: { k: 5 } });
  assert.equal(m.kind, "knn");
  assert.equal(m.nStored, 150);
  for (const s of m.std) assert.ok(s >= 1e-6, "std floor");
});

test("featureSet 'm2' ignores every column outside the 24 network metrics; 'all' does not", async () => {
  const train = rows(8, 160, { ties: false });
  const test_ = rows(9, 60, { ties: false });
  const m2Cols = new Set(featureColumns("m2"));
  assert.equal(m2Cols.size, 24);
  const scramble = (d) => {
    const X = d.X.slice();
    const rng = makeRng(99, "scramble");
    for (let r = 0; r < d.n; r++) for (let j = 0; j < 42; j++) if (!m2Cols.has(j)) X[r * 42 + j] = rng.int(1000);
    return { ...d, X };
  };
  const m2a = await predictKnn(trainKnn({ train, config: { featureSet: "m2" } }), test_);
  const m2b = await predictKnn(trainKnn({ train: scramble(train), config: { featureSet: "m2" } }), scramble(test_));
  assert.deepEqual(Array.from(m2a), Array.from(m2b), "non-M2 columns have no effect under 'm2'");
  const model = trainKnn({ train, config: { featureSet: "m2" } });
  assert.equal(model.d, 24);
  assert.deepEqual(Array.from(model.cols), Array.from(featureColumns("m2")));
  const alla = await predictKnn(trainKnn({ train, config: { featureSet: "all" } }), test_);
  const allb = await predictKnn(trainKnn({ train: scramble(train), config: { featureSet: "all" } }), scramble(test_));
  assert.notDeepEqual(Array.from(alla), Array.from(allb), "'all' sees the other columns");
  assert.equal(trainKnn({ train, config: { featureSet: "all" } }).d, 42);
  assert.throws(() => trainKnn({ train, config: { featureSet: "m9" } }), /Unknown feature set/);
});

test("progress is reported about every 64 rows and ends at 1; shouldCancel stops with null", async () => {
  const train = rows(10, 100);
  const test_ = rows(11, 300);
  const model = trainKnn({ train, config: { k: 5 } });
  const fr = [];
  const s = await predictKnn(model, test_, { onProgress: (e) => fr.push(e.fraction) });
  assert.ok(s instanceof Float64Array);
  assert.deepEqual(fr, [64 / 300, 128 / 300, 192 / 300, 256 / 300, 1]);
  let calls = 0;
  const cancelled = await predictKnn(model, test_, { shouldCancel: () => ++calls > 2 });
  assert.equal(cancelled, null);
  assert.ok(calls <= 4, `checked ${calls} times before stopping`);
});

test("yields to the event loop while scoring (so cancel works inside a worker)", async () => {
  const train = rows(12, 200);
  const test_ = rows(13, 400);
  let ticks = 0;
  const timer = setInterval(() => ticks++, 0);
  await predictKnn(trainKnn({ train }), test_);
  clearInterval(timer);
  assert.ok(ticks > 0);
});
