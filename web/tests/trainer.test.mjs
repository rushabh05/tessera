// Proves web/js/lab/trainer.js (hand-written TESSERA-base training) against
// PyTorch autograd, and web/js/lab/logreg.js end to end.
//
// Golden: web/data/train_golden.json, produced by
//   uv run python -m tessera.demo.golden_train
// (the real _TesseraBaseNet in float64, on a synthetic batch - no real rows).
//
// Run: node --test web/tests/trainer.test.mjs

import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { forward } from "../js/forward.js";
import { makeRng } from "../js/lab/rng.js";
import {
  averagePrecisionScore,
  clipGradNorm,
  cosineLr,
  earlyStopMetricFor,
  countParams,
  createAdamW,
  flattenParams,
  initWeights,
  lossAndGrads,
  paramLayout,
  predict,
  trainTesseraBase,
} from "../js/lab/trainer.js";
import { predictLogReg, trainLogReg } from "../js/lab/logreg.js";

const readJson = (rel) => JSON.parse(readFileSync(new URL(rel, import.meta.url)));
const golden = readJson("../data/train_golden.json");
const pretrained = readJson("../data/weights.json");

const REL = 1e-9;
const ABS = 1e-12;

function b64ToF64(b64) {
  const buf = Buffer.from(b64, "base64");
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length);
  return new Float64Array(ab);
}

const clone = (o) => JSON.parse(JSON.stringify(o));

function tensorNameAt(i) {
  const layout = golden.param_layout;
  let name = "?";
  for (const t of layout) if (t.offset <= i) name = t.name;
  return name;
}

// Element-wise |a - e| <= max(ABS, REL * |e|); reports the worst element.
function assertClose(actual, expected, what, { rel = REL, abs = ABS, names = false } = {}) {
  assert.equal(actual.length, expected.length, `${what}: length`);
  let worst = -1;
  let worstRatio = 0;
  for (let i = 0; i < expected.length; i++) {
    const tol = Math.max(abs, rel * Math.abs(expected[i]));
    const ratio = Math.abs(actual[i] - expected[i]) / tol;
    if (!(ratio <= worstRatio)) {
      worstRatio = ratio;
      worst = i;
    }
  }
  if (worstRatio > 1 || Number.isNaN(worstRatio)) {
    const where = names ? ` (${tensorNameAt(worst)})` : "";
    assert.fail(
      `${what}: element ${worst}${where} actual ${actual[worst]} expected ${expected[worst]} ` +
        `(${worstRatio.toFixed(2)}x tolerance)`,
    );
  }
  return worstRatio;
}

function goldenBatch() {
  const b = golden.batch;
  return {
    X: Float64Array.from(b.X.flat()),
    y: Float64Array.from(b.y),
    avail: Float64Array.from(b.avail.flat()),
    b: b.n,
  };
}
const goldenMask = () => Float64Array.from(golden.dropout_mask.flat());

// ---------------------------------------------------------------------------
// 1. Golden parity against PyTorch autograd (float64)

test("golden: layout and parameter count match torch's net.parameters()", () => {
  const w = golden.weights;
  assert.equal(countParams(w), 5005);
  assert.equal(golden.n_params, 5005);
  const layout = paramLayout(w);
  assert.equal(layout.length, golden.param_layout.length);
  for (let i = 0; i < layout.length; i++) {
    assert.deepEqual(layout[i].shape, golden.param_layout[i].shape, golden.param_layout[i].name);
    assert.equal(layout[i].offset, golden.param_layout[i].offset, golden.param_layout[i].name);
  }
});

for (const mode of ["eval", "dropout"]) {
  test(`golden (${mode}): loss and every gradient match autograd`, (t) => {
    const g = golden[mode];
    const opts = { posWeight: golden.pos_weight, dropoutP: golden.dropout_p };
    if (mode === "dropout") opts.dropoutMask = goldenMask();
    const { loss, grads } = lossAndGrads(golden.weights, goldenBatch(), opts);
    assertClose([loss], [g.loss], `${mode} loss`);
    const r = assertClose(flattenParams(grads), b64ToF64(g.grads_b64), `${mode} grads`, { names: true });
    t.diagnostic(`${mode}: loss ${loss} vs ${g.loss}; worst grad error ${r.toExponential(2)}x tolerance`);
  });

  test(`golden (${mode}): grad norm and clipping match clip_grad_norm_`, () => {
    const g = golden[mode];
    const opts = { posWeight: golden.pos_weight, dropoutP: golden.dropout_p };
    if (mode === "dropout") opts.dropoutMask = goldenMask();
    const { grads } = lossAndGrads(golden.weights, goldenBatch(), opts);
    const norm = clipGradNorm(grads, g.max_norm);
    assertClose([norm], [g.grad_norm], `${mode} grad norm`);
    assert.equal(norm > g.max_norm, g.clip_active, "clip branch taken as in torch");
    assertClose(flattenParams(grads), b64ToF64(g.clipped_grads_b64), `${mode} clipped grads`, { names: true });
    // The flat-array form must agree with the nested form.
    const flat = flattenParams(lossAndGrads(golden.weights, goldenBatch(), opts).grads);
    const normFlat = clipGradNorm(flat, g.max_norm);
    assertClose([normFlat], [g.grad_norm], `${mode} flat grad norm`);
    assertClose(flat, b64ToF64(g.clipped_grads_b64), `${mode} flat clipped grads`, { names: true });
  });

  test(`golden (${mode}): 3 AdamW steps with clipping match torch.optim.AdamW`, () => {
    const g = golden[mode];
    const a = golden.adamw;
    const w = clone(golden.weights);
    const opt = createAdamW(w, { lr: a.lr, betas: a.betas, eps: a.eps, weightDecay: a.weight_decay });
    const opts = { posWeight: golden.pos_weight, dropoutP: golden.dropout_p };
    if (mode === "dropout") opts.dropoutMask = goldenMask();
    for (let s = 0; s < a.n_steps; s++) {
      const { loss, grads } = lossAndGrads(w, goldenBatch(), opts);
      assertClose([loss], [g.steps[s].loss], `${mode} step ${s} loss`);
      const norm = clipGradNorm(grads, g.max_norm);
      assertClose([norm], [g.steps[s].grad_norm], `${mode} step ${s} grad norm`);
      opt.step(grads, a.lr);
    }
    assertClose(flattenParams(w), b64ToF64(g.params_after_steps_b64), `${mode} params after steps`, {
      names: true,
    });
  });
}

test("golden: eval-mode scores and gates match torch", () => {
  const b = goldenBatch();
  const { scores, attribution } = predict(golden.weights, { X: b.X, avail: b.avail, n: b.b });
  assertClose(scores, golden.eval_outputs.scores, "scores");
  assertClose(attribution, golden.eval_outputs.gates.flat(), "gates");
  // Masked modalities get exactly zero weight.
  for (let r = 0; r < b.b; r++) {
    for (let m = 0; m < 4; m++) if (b.avail[r * 4 + m] < 0.5) assert.equal(attribution[r * 4 + m], 0);
  }
});

test("golden: cosineLr matches torch CosineAnnealingLR", () => {
  const c = golden.cosine;
  const ours = c.lrs.map((_, e) => cosineLr(c.base_lr, e, c.t_max));
  assertClose(ours, c.lrs, "cosine lr", { abs: 1e-15 });
});

// ---------------------------------------------------------------------------
// 2. predict() equals forward.js

function heavyTailedRows(rng, n) {
  const X = new Float64Array(n * 42);
  const avail = new Float64Array(n * 4);
  const slices = [
    [0, 8],
    [8, 32],
    [32, 34],
    [34, 42],
  ];
  for (let r = 0; r < n; r++) {
    for (let m = 0; m < 4; m++) {
      const present = m === 2 || rng.bernoulli(0.7);
      avail[r * 4 + m] = present ? 1 : 0;
      if (!present) continue;
      const [a, b] = slices[m];
      if (m === 2) {
        X[r * 42 + a] = [10, 34, 47][rng.int(3)];
        X[r * 42 + a + 1] = 1 + rng.int(3);
        continue;
      }
      for (let j = a; j < b; j++) if (rng.bernoulli(0.6)) X[r * 42 + j] = Math.exp(rng.uniform(0, 13) + rng.normal());
      X[r * 42 + a] ||= 1;
    }
  }
  return { X, avail, n };
}

function assertMatchesForwardJs(weights, rows, what) {
  const { scores, attribution } = predict(weights, rows);
  let worst = 0;
  for (let r = 0; r < rows.n; r++) {
    const x = Array.from(rows.X.subarray(r * 42, r * 42 + 42));
    const a = Array.from(rows.avail.subarray(r * 4, r * 4 + 4));
    const ref = forward(x, a, weights);
    worst = Math.max(worst, Math.abs(ref.score - scores[r]));
    for (let m = 0; m < 4; m++) worst = Math.max(worst, Math.abs(ref.attribution[m] - attribution[r * 4 + m]));
  }
  assert.ok(worst <= 1e-9, `${what}: max |predict - forward.js| = ${worst}`);
  return worst;
}

test("predict() equals forward.js on the pretrained weights.json", (t) => {
  const vectors = readJson("../data/golden.json").vectors;
  const demo = readJson("../data/demo_windows.json").windows;
  const inputs = [
    ...vectors.map((v) => [v.input, v.availability]),
    ...demo.map((w) => [w.features, ["m1_log", "m2_metrics", "m3_identity", "m4_graph"].map((k) => (w.availability[k] ? 1 : 0))]),
  ];
  const rows = {
    X: Float64Array.from(inputs.flatMap(([x]) => x)),
    avail: Float64Array.from(inputs.flatMap(([, a]) => a)),
    n: inputs.length,
  };
  const worst1 = assertMatchesForwardJs(pretrained, rows, "pretrained, recorded inputs");
  const worst2 = assertMatchesForwardJs(pretrained, heavyTailedRows(makeRng(7, "rows"), 300), "pretrained, random rows");
  t.diagnostic(`max abs difference: ${worst1} (recorded inputs), ${worst2} (random rows)`);
});

test("predict() equals forward.js on freshly initialised weights", () => {
  for (const seed of [0, 1, 2]) {
    const w = initWeights(makeRng(seed, "init"));
    assertMatchesForwardJs(w, heavyTailedRows(makeRng(seed, "rows"), 200), `fresh seed ${seed}`);
  }
});

// ---------------------------------------------------------------------------
// 3. Finite-difference gradient check

test("finite-difference gradient check on random parameters", (t) => {
  const rng = makeRng(3, "fd");
  const w = initWeights(makeRng(11, "init"));
  const rows = heavyTailedRows(makeRng(12, "rows"), 16);
  const y = new Float64Array(16);
  for (let i = 0; i < 16; i++) y[i] = i % 3 === 0 ? 1 : 0;
  const batch = { X: rows.X, y, avail: rows.avail, b: 16 };
  const mask = new Float64Array(16 * 32);
  for (let i = 0; i < mask.length; i++) mask[i] = rng.bernoulli(0.8) ? 1 : 0;
  const layout = paramLayout(w);
  const tensors = [];
  // walk into the nested structure by layout name: collect (container, index) per flat index
  const refs = [];
  const walk = (tensor) => {
    if (Array.isArray(tensor[0])) for (const row of tensor) for (let i = 0; i < row.length; i++) refs.push([row, i]);
    else for (let i = 0; i < tensor.length; i++) refs.push([tensor, i]);
  };
  for (const enc of w.encoders) {
    walk(enc.linear0.weight);
    walk(enc.linear0.bias);
    walk(enc.groupnorm.weight);
    walk(enc.groupnorm.bias);
    walk(enc.linear1.weight);
    walk(enc.linear1.bias);
  }
  for (const l of [w.fusion.gate_linear0, w.fusion.gate_linear1, w.head.linear0, w.head.linear1]) {
    walk(l.weight);
    walk(l.bias);
  }
  assert.equal(refs.length, 5005);
  for (const tl of layout) tensors.push(tl);

  for (const dropoutMask of [null, mask]) {
    const opts = { posWeight: 2.0, dropoutMask };
    const analytic = flattenParams(lossAndGrads(w, batch, opts).grads);
    let worst = 0;
    // two random parameters from every tensor
    for (const tl of tensors) {
      for (let k = 0; k < 2; k++) {
        const idx = tl.offset + rng.int(tl.size);
        const [arr, i] = refs[idx];
        const orig = arr[i];
        const h = 1e-6 * Math.max(1, Math.abs(orig));
        arr[i] = orig + h;
        const lp = lossAndGrads(w, batch, opts).loss;
        arr[i] = orig - h;
        const lm = lossAndGrads(w, batch, opts).loss;
        arr[i] = orig;
        const fd = (lp - lm) / (2 * h);
        const err = Math.abs(fd - analytic[idx]) / Math.max(1e-6, Math.abs(fd) + Math.abs(analytic[idx]));
        worst = Math.max(worst, err);
        assert.ok(err < 1e-4, `${tl.name}[${idx - tl.offset}]: fd ${fd} analytic ${analytic[idx]}`);
      }
    }
    t.diagnostic(`${dropoutMask ? "dropout" : "eval"}: worst relative FD error ${worst.toExponential(2)}`);
  }
});

// ---------------------------------------------------------------------------
// 4. initWeights

test("initWeights: deterministic per seed, 5005 params, PyTorch default bounds", () => {
  const a = initWeights(makeRng(5, "init"));
  const b = initWeights(makeRng(5, "init"));
  const c = initWeights(makeRng(6, "init"));
  assert.deepEqual(a, b);
  assert.notDeepEqual(flattenParams(a), flattenParams(c));
  assert.equal(countParams(a), 5005);
  assert.equal(countParams(pretrained), 5005);
  // same structure/metadata as the real exported weights.json
  for (const k of ["architecture", "gelu", "modality_slices", "modality_names", "n_features"]) {
    assert.deepEqual(a[k], pretrained[k], k);
  }
  assert.deepEqual(paramLayout(a), paramLayout(pretrained));
  const gn = a.encoders[0].groupnorm;
  assert.equal(gn.num_groups, 4);
  assert.equal(gn.eps, 1e-5);
  assert.ok(gn.weight.every((v) => v === 1) && gn.bias.every((v) => v === 0));
  const bound = 1 / Math.sqrt(68);
  const gw = a.fusion.gate_linear0.weight.flat();
  assert.ok(gw.every((v) => Math.abs(v) <= bound));
  assert.ok(Math.max(...gw.map(Math.abs)) > 0.9 * bound, "uses the full U(-1/sqrt(fan_in), +) range");
});

test("averagePrecisionScore: sklearn values incl. ties; null for one class", () => {
  // sklearn.metrics.average_precision_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.8333333333333333
  assert.ok(Math.abs(averagePrecisionScore([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.8333333333333333) < 1e-15);
  // ties: average_precision_score([1,0,1,0],[0.5,0.5,0.5,0.2]) == 0.6666666666666666
  assert.ok(Math.abs(averagePrecisionScore([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.2]) - 2 / 3) < 1e-15);
  assert.equal(averagePrecisionScore([0, 0, 0], [0.1, 0.2, 0.3]), null);
});

// ---------------------------------------------------------------------------
// 5. End-to-end training on a small synthetic separable dataset

function syntheticDataset(seed, n, prevalence = 0.2) {
  const rng = makeRng(seed, "smoke");
  const X = new Float32Array(n * 42);
  const y = new Uint8Array(n);
  const avail = new Uint8Array(n * 4);
  const muRng = makeRng(0, "smoke-mu"); // shared by train/val/test: one distribution
  const mu = Array.from({ length: 42 }, () => muRng.uniform(0, 10));
  const shift = Array.from({ length: 42 }, (_, j) => (j % 2 === 0 ? 2.0 : 0));
  const slices = [
    [0, 8],
    [8, 32],
    [32, 34],
    [34, 42],
  ];
  for (let r = 0; r < n; r++) {
    const t = rng.bernoulli(prevalence) ? 1 : 0;
    y[r] = t;
    let nSrc = 0;
    for (const m of [0, 1, 3]) {
      const p = t ? 0.98 : [0.2, 0.85, 0, 0.8][m];
      const present = rng.bernoulli(p);
      if (!present) continue;
      nSrc++;
      const [a, b] = slices[m];
      for (let j = a; j < b; j++) {
        if (rng.bernoulli(t ? 0.15 : 0.4)) continue;
        X[r * 42 + j] = Math.round(Math.exp(mu[j] + t * shift[j] + 1.0 * rng.normal())) || 1;
      }
      X[r * 42 + a] ||= 1;
    }
    X[r * 42 + 32] = t ? 10 : [10, 34, 47][rng.int(3)];
    X[r * 42 + 33] = Math.max(1, nSrc);
    for (const [m, [a, b]] of slices.entries()) {
      let any = 0;
      for (let j = a; j < b; j++) if (X[r * 42 + j] !== 0) any = 1;
      avail[r * 4 + m] = m === 2 ? 1 : any;
    }
  }
  return { X, y, avail, n };
}

const smokeTrain = syntheticDataset(1, 1500);
const smokeVal = syntheticDataset(2, 400);
const smokeTest = syntheticDataset(3, 600);

test("training: high AP, decreasing loss, best epoch restored, forward.js runs the result", async (t) => {
  const events = [];
  const res = await trainTesseraBase({
    train: smokeTrain,
    val: smokeVal,
    // batch 64 so 1,500 rows give 24 steps per epoch (the default 256 gives 6)
    config: { epochs: 15, patience: 4, batchSize: 64, seed: 0 },
    onProgress: (e) => events.push(e),
  });
  const h = res.history;
  assert.equal(res.cancelled, false);
  assert.equal(res.nParams, 5005);
  assert.equal(countParams(res.weights), 5005);
  assert.ok(h.trainLoss[h.trainLoss.length - 1] < h.trainLoss[0], `loss ${h.trainLoss[0]} -> ${h.trainLoss.at(-1)}`);
  // the returned weights are the best-validation-AP epoch's
  const best = h.valAp.indexOf(Math.max(...h.valAp));
  assert.equal(h.bestEpoch, best);
  const vs = predict(res.weights, smokeVal).scores;
  assert.ok(Math.abs(averagePrecisionScore(smokeVal.y, vs) - h.valAp[h.bestEpoch]) < 1e-12, "restored best epoch");
  const ts = predict(res.weights, smokeTest).scores;
  const ap = averagePrecisionScore(smokeTest.y, ts);
  assert.ok(ap > 0.9, `test AP ${ap}`);
  // lr follows the cosine schedule; progress fractions are monotone and end at 1
  h.lr.forEach((lr, e) => assert.ok(Math.abs(lr - cosineLr(1e-3, e, 15)) < 1e-18));
  const fr = events.map((e) => e.fraction);
  for (let i = 1; i < fr.length; i++) assert.ok(fr[i] >= fr[i - 1] - 1e-12);
  assert.ok(events.some((e) => e.phase === "epoch") && events.some((e) => e.phase === "batch"));
  // forward.js runs the exported weights
  const x = Array.from(smokeTest.X.subarray(0, 42));
  const a = Array.from(smokeTest.avail.subarray(0, 4));
  assert.ok(Math.abs(forward(x, a, res.weights).score - ts[0]) < 1e-9);
  t.diagnostic(
    `epochs ${h.epochsRun}, loss ${h.trainLoss[0].toFixed(4)} -> ${h.trainLoss.at(-1).toFixed(4)}, ` +
      `best epoch ${h.bestEpoch} (val AP ${h.valAp[h.bestEpoch].toFixed(4)}), test AP ${ap.toFixed(4)}, ${res.elapsedMs.toFixed(0)} ms`,
  );
});

test("training: deterministic for a seed", async () => {
  const small = syntheticDataset(4, 300);
  const cfg = { epochs: 2, seed: 9 };
  const a = await trainTesseraBase({ train: small, val: null, config: cfg });
  const b = await trainTesseraBase({ train: small, val: null, config: cfg });
  assert.deepEqual(a.weights, b.weights);
  assert.deepEqual(a.history.trainLoss, b.history.trainLoss);
  assert.equal(a.history.bestEpoch, 1);
});

test("training: early stopping stops after `patience` non-improving epochs and restores the best", async () => {
  // lr 0 => weights never change => val AP is flat => epoch 0 is best, stop after patience more.
  const res = await trainTesseraBase({
    train: smokeTrain,
    val: smokeVal,
    config: { epochs: 30, lr: 0, patience: 2, seed: 1 },
  });
  assert.equal(res.history.bestEpoch, 0);
  assert.equal(res.history.epochsRun, 3);
  assert.equal(res.history.stoppedEarly, true);
  const init = initWeights(makeRng(1, "train").fork("init"));
  assert.deepEqual(flattenParams(res.weights), flattenParams(init));
});

/** The rows of a dataset with a given label only (a one-class validation set). */
function onlyClass(d, label) {
  const idx = [];
  for (let r = 0; r < d.n; r++) if (d.y[r] === label) idx.push(r);
  const X = new Float32Array(idx.length * 42);
  const y = new Uint8Array(idx.length);
  const avail = new Uint8Array(idx.length * 4);
  idx.forEach((r, i) => {
    X.set(d.X.subarray(r * 42, r * 42 + 42), i * 42);
    avail.set(d.avail.subarray(r * 4, r * 4 + 4), i * 4);
    y[i] = d.y[r];
  });
  return { X, y, avail, n: idx.length };
}

test("earlyStopMetricFor: val_ap with both classes, val_loss with one class, none without rows", () => {
  assert.equal(earlyStopMetricFor(smokeVal), "val_ap");
  assert.equal(earlyStopMetricFor(onlyClass(smokeVal, 0)), "val_loss");
  assert.equal(earlyStopMetricFor(onlyClass(smokeVal, 1)), "val_loss");
  assert.equal(earlyStopMetricFor(null), "none");
  assert.equal(earlyStopMetricFor({ X: new Float32Array(0), y: new Uint8Array(0), avail: new Uint8Array(0), n: 0 }), "none");
});

test("training: a validation set with no attacks early-stops on validation loss, never NaN", async () => {
  const benignVal = onlyClass(smokeVal, 0);
  assert.ok(benignVal.n > 50);
  const events = [];
  const res = await trainTesseraBase({
    train: smokeTrain,
    val: benignVal,
    config: { epochs: 12, patience: 3, batchSize: 64, seed: 2 },
    onProgress: (e) => e.phase === "epoch" && events.push(e),
  });
  const h = res.history;
  assert.equal(h.earlyStopMetric, "val_loss");
  assert.deepEqual(h.valAp, [], "AP is undefined on one class: not reported as a number");
  assert.equal(h.valLoss.length, h.epochsRun);
  for (const v of [...h.valLoss, ...h.trainLoss]) assert.ok(Number.isFinite(v), `finite loss ${v}`);
  const best = h.valLoss.indexOf(Math.min(...h.valLoss));
  assert.equal(h.bestEpoch, best, "kept the lowest validation loss");
  for (const e of events) {
    assert.equal(e.valAp, null);
    assert.ok(Number.isFinite(e.valLoss));
  }
  // the returned weights are that epoch's: their validation loss is the recorded minimum
  const s = predict(res.weights, benignVal).scores;
  let loss = 0;
  for (const p of s) loss += -Math.log1p(-p);
  assert.ok(Math.abs(loss / s.length - h.valLoss[h.bestEpoch]) < 1e-9, "restored the best-loss epoch");
  assert.ok(countParams(res.weights) === 5005);
});

test("training: lr 0 with a one-class validation set stops after `patience` flat-loss epochs", async () => {
  const res = await trainTesseraBase({
    train: smokeTrain,
    val: onlyClass(smokeVal, 1),
    config: { epochs: 30, lr: 0, patience: 2, seed: 1 },
  });
  assert.equal(res.history.earlyStopMetric, "val_loss");
  assert.equal(res.history.bestEpoch, 0);
  assert.equal(res.history.epochsRun, 3);
  assert.equal(res.history.stoppedEarly, true);
});

test("training: no validation rows keeps the final epoch and never stops early", async () => {
  const empty = { X: new Float32Array(0), y: new Uint8Array(0), avail: new Uint8Array(0), n: 0 };
  for (const val of [null, empty]) {
    const res = await trainTesseraBase({ train: smokeTrain, val, config: { epochs: 3, lr: 0, patience: 1, seed: 1 } });
    assert.equal(res.history.earlyStopMetric, "none");
    assert.equal(res.history.epochsRun, 3);
    assert.equal(res.history.bestEpoch, 2);
    assert.equal(res.history.stoppedEarly, false);
    assert.deepEqual(res.history.valAp, []);
  }
  const withVal = await trainTesseraBase({ train: smokeTrain, val: smokeVal, config: { epochs: 1, seed: 1 } });
  assert.equal(withVal.history.earlyStopMetric, "val_ap");
});

test("logreg: one-class validation keeps the lowest validation loss; no validation keeps the last step", async () => {
  const benign = await trainLogReg({ train: smokeTrain, val: onlyClass(smokeVal, 0), config: { epochs: 40 } });
  const h = benign.history;
  assert.equal(h.earlyStopMetric, "val_loss");
  assert.deepEqual(h.valAp, []);
  assert.equal(h.valLoss.length, 40);
  for (const v of h.valLoss) assert.ok(Number.isFinite(v));
  assert.equal(h.bestEpoch, h.valLoss.indexOf(Math.min(...h.valLoss)));
  for (const v of predictLogReg(benign, smokeTest)) assert.ok(Number.isFinite(v));
  const none = await trainLogReg({ train: smokeTrain, val: null, config: { epochs: 7 } });
  assert.equal(none.history.earlyStopMetric, "none");
  assert.equal(none.history.bestEpoch, 6);
  const both = await trainLogReg({ train: smokeTrain, val: smokeVal, config: { epochs: 5 } });
  assert.equal(both.history.earlyStopMetric, "val_ap");
  assert.equal(both.history.valAp.length, 5);
});

test("training: shouldCancel stops promptly with cancelled = true", async () => {
  let batches = 0;
  let cancelAt = null;
  const res = await trainTesseraBase({
    train: smokeTrain,
    val: smokeVal,
    config: { epochs: 30, batchSize: 64 },
    onProgress: (e) => {
      if (e.phase === "batch") batches++;
      if (batches === 10 && cancelAt === null) cancelAt = batches;
    },
    shouldCancel: () => cancelAt !== null,
  });
  assert.equal(res.cancelled, true);
  assert.equal(batches, 10, "no batch ran after cancellation was requested");
  assert.equal(res.history.epochsRun, 0);
  assert.equal(countParams(res.weights), 5005);
});

test("training yields to the event loop (progress and cancel work inside a worker)", async () => {
  let ticks = 0;
  const timer = setInterval(() => ticks++, 0);
  await trainTesseraBase({ train: smokeTrain, val: null, config: { epochs: 1, batchSize: 64 }, yieldEvery: 2 });
  clearInterval(timer);
  assert.ok(ticks > 0, "a timer fired while training");
});

test("logreg: trains and beats chance on the same data", async (t) => {
  const model = await trainLogReg({ train: smokeTrain, val: smokeVal, config: { epochs: 200 } });
  const s = predictLogReg(model, smokeTest);
  const ap = averagePrecisionScore(smokeTest.y, s);
  let prev = 0;
  for (const v of smokeTest.y) prev += v;
  prev /= smokeTest.n;
  assert.ok(ap > prev + 0.3, `logreg AP ${ap} vs prevalence ${prev}`);
  assert.ok(model.history.trainLoss.at(-1) < model.history.trainLoss[0]);
  assert.equal(JSON.parse(JSON.stringify(model)).w.length, 46, "plain JSON model");
  let calls = 0;
  const cancelled = await trainLogReg({ train: smokeTrain, val: null, shouldCancel: () => ++calls > 5 });
  assert.equal(cancelled.cancelled, true);
  t.diagnostic(`logreg test AP ${ap.toFixed(4)} (prevalence ${prev.toFixed(3)}), best epoch ${model.history.bestEpoch}`);
});

// ---------------------------------------------------------------------------
// 6. Performance: 8,000 rows at batch 256

test("performance: time per epoch for 8,000 rows at batch 256", async (t) => {
  const big = syntheticDataset(5, 8000);
  const val = syntheticDataset(6, 1500);
  const epochs = 3;
  const times = [];
  let last = performance.now();
  await trainTesseraBase({
    train: big,
    val,
    config: { epochs, patience: 99 },
    onProgress: (e) => {
      if (e.phase === "epoch") {
        const now = performance.now();
        times.push(now - last);
        last = now;
      }
    },
  });
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const msg = `8,000 rows, batch 256 (+1,500-row validation pass): ${times.map((x) => x.toFixed(0)).join(", ")} ms per epoch, mean ${mean.toFixed(0)} ms`;
  t.diagnostic(msg);
  console.log(`# ${msg}`);
  assert.ok(mean < 2000, msg);
});
