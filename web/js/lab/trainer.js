// TESSERA-base TRAINING, written from scratch in vanilla JS (float64).
//
// This file trains the same network web/js/forward.js runs - the real
// tessera/models/tessera_base.py _TesseraBaseNet - with the same recipe as
// TesseraBase.fit: AdamW (decay on every parameter), cosine LR stepped per
// epoch, BCE-with-logits with pos_weight = n_neg / n_pos, global L2 gradient
// clipping at 1.0, batch 256, reshuffle every epoch, dropout 0.2 on the head,
// early stopping on validation average precision (patience 5) restoring the
// best epoch. When the validation set is all one class (AP undefined) early
// stopping falls back to the validation loss; with no validation rows the final
// epoch is kept (history.earlyStopMetric says which). There is no autograd here: the backward pass is written out by
// hand, and web/tests/trainer.test.mjs proves loss, every gradient, the grad
// norm, clipping and three AdamW steps against PyTorch autograd in float64
// (web/data/train_golden.json, from `uv run python -m tessera.demo.golden_train`).
//
// Performance: parameters, gradients and optimiser state live in single flat
// Float64Arrays; every intermediate a sample needs is pre-allocated once per
// network, and each row is pushed forward and straight back (its loss
// gradient depends only on its own logit), so the hot loop allocates nothing.

import { makeRng } from "./rng.js";

const FLOAT32_MIN = -3.4028234663852886e38; // torch.finfo(float32).min - the GMU mask sentinel
const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const GELU_C = 0.044715;

const DEFAULT_SLICES = [
  [0, 8],
  [8, 32],
  [32, 34],
  [34, 42],
];
const MODALITY_NAMES = ["m1_log", "m2_metrics", "m3_identity", "m4_graph"];

// ---------------------------------------------------------------------------
// Parameter layout: one flat vector, in net.parameters() order, which is a
// depth-first walk of the weights.json structure.

function tensorsOf(w) {
  const out = [];
  w.encoders.forEach((enc, m) => {
    out.push([`encoders.${m}.linear0.weight`, enc.linear0.weight]);
    out.push([`encoders.${m}.linear0.bias`, enc.linear0.bias]);
    out.push([`encoders.${m}.groupnorm.weight`, enc.groupnorm.weight]);
    out.push([`encoders.${m}.groupnorm.bias`, enc.groupnorm.bias]);
    out.push([`encoders.${m}.linear1.weight`, enc.linear1.weight]);
    out.push([`encoders.${m}.linear1.bias`, enc.linear1.bias]);
  });
  out.push(["fusion.gate_linear0.weight", w.fusion.gate_linear0.weight]);
  out.push(["fusion.gate_linear0.bias", w.fusion.gate_linear0.bias]);
  out.push(["fusion.gate_linear1.weight", w.fusion.gate_linear1.weight]);
  out.push(["fusion.gate_linear1.bias", w.fusion.gate_linear1.bias]);
  out.push(["head.linear0.weight", w.head.linear0.weight]);
  out.push(["head.linear0.bias", w.head.linear0.bias]);
  out.push(["head.linear1.weight", w.head.linear1.weight]);
  out.push(["head.linear1.bias", w.head.linear1.bias]);
  return out;
}

function tensorSize(t) {
  return Array.isArray(t[0]) ? t.length * t[0].length : t.length;
}

/** Flatten weights (or a grads object of the same shape) to one Float64Array, net.parameters() order. */
export function flattenParams(w) {
  const tensors = tensorsOf(w);
  let size = 0;
  for (const [, t] of tensors) size += tensorSize(t);
  const flat = new Float64Array(size);
  let o = 0;
  for (const [, t] of tensors) {
    if (Array.isArray(t[0])) {
      for (const row of t) for (let i = 0; i < row.length; i++) flat[o++] = row[i];
    } else {
      for (let i = 0; i < t.length; i++) flat[o++] = t[i];
    }
  }
  return flat;
}

/** Write a flat vector back into a nested weights/grads object in place. */
function writeFlat(w, flat) {
  let o = 0;
  for (const [, t] of tensorsOf(w)) {
    if (Array.isArray(t[0])) {
      for (const row of t) for (let i = 0; i < row.length; i++) row[i] = flat[o++];
    } else {
      for (let i = 0; i < t.length; i++) t[i] = flat[o++];
    }
  }
  return w;
}

/** Tensor names, shapes and offsets for a weights object (matches golden param_layout). */
export function paramLayout(w) {
  const out = [];
  let offset = 0;
  for (const [name, t] of tensorsOf(w)) {
    const shape = Array.isArray(t[0]) ? [t.length, t[0].length] : [t.length];
    out.push({ name, shape, offset, size: tensorSize(t) });
    offset += tensorSize(t);
  }
  return out;
}

function cloneStructure(w) {
  return JSON.parse(JSON.stringify(w));
}

// Grads mirror the weights structure: every Linear weight/bias and GroupNorm weight/bias.
function gradsFromFlat(w, flat) {
  const lin = (l) => ({ weight: l.weight.map((r) => r.slice()), bias: l.bias.slice() });
  const g = {
    encoders: w.encoders.map((e) => ({
      linear0: lin(e.linear0),
      groupnorm: { weight: e.groupnorm.weight.slice(), bias: e.groupnorm.bias.slice() },
      linear1: lin(e.linear1),
    })),
    fusion: { gate_linear0: lin(w.fusion.gate_linear0), gate_linear1: lin(w.fusion.gate_linear1) },
    head: { linear0: lin(w.head.linear0), linear1: lin(w.head.linear1) },
  };
  return writeFlat(g, flat);
}

// ---------------------------------------------------------------------------
// Network core over a flat parameter vector.

const coreCache = new WeakMap();

function getCore(w) {
  let core = coreCache.get(w);
  if (!core) {
    core = makeCore(w);
    coreCache.set(w, core);
  }
  return core;
}

function makeCore(w) {
  const slices = w.modality_slices || DEFAULT_SLICES;
  const nF = w.n_features || slices[slices.length - 1][1];
  const M = w.encoders.length;
  const E = w.fusion.embed_dim;
  const layout = paramLayout(w);
  const off = {};
  for (const t of layout) off[t.name] = t.offset;

  const enc = w.encoders.map((e, m) => ({
    start: slices[m][0],
    dIn: slices[m][1] - slices[m][0],
    H: e.linear0.weight.length,
    groups: e.groupnorm.num_groups,
    eps: e.groupnorm.eps,
    W0: off[`encoders.${m}.linear0.weight`],
    b0: off[`encoders.${m}.linear0.bias`],
    gw: off[`encoders.${m}.groupnorm.weight`],
    gb: off[`encoders.${m}.groupnorm.bias`],
    W1: off[`encoders.${m}.linear1.weight`],
    b1: off[`encoders.${m}.linear1.bias`],
  }));
  const Gin = M * E + M;
  const G1 = w.fusion.gate_linear0.weight.length;
  const Hh = w.head.linear0.weight.length;
  const Wg0 = off["fusion.gate_linear0.weight"];
  const bg0 = off["fusion.gate_linear0.bias"];
  const Wg1 = off["fusion.gate_linear1.weight"];
  const bg1 = off["fusion.gate_linear1.bias"];
  const Wh0 = off["head.linear0.weight"];
  const bh0 = off["head.linear0.bias"];
  const Wh1 = off["head.linear1.weight"];
  const bh1 = off["head.linear1.bias"];
  const nParams = layout.reduce((a, t) => a + t.size, 0);

  // Forward caches (one sample at a time).
  const xhat = enc.map((e) => new Float64Array(e.H));
  const inv = enc.map((e) => new Float64Array(e.groups));
  const nrm = enc.map((e) => new Float64Array(e.H));
  const tn = enc.map((e) => new Float64Array(e.H)); // tanh inside GELU(nrm)
  const g1 = enc.map((e) => new Float64Array(e.H));
  const a0 = enc.map((e) => new Float64Array(e.H));
  const emb = new Float64Array(M * E);
  const th = new Float64Array(M * E); // tanh(e)
  const gin = new Float64Array(Gin);
  const q0 = new Float64Array(G1);
  const tq = new Float64Array(G1);
  const q1 = new Float64Array(G1);
  const lg = new Float64Array(M);
  const gate = new Float64Array(M);
  const fused = new Float64Array(E);
  const r0 = new Float64Array(Hh);
  const tr = new Float64Array(Hh);
  const r1 = new Float64Array(Hh);
  const r2 = new Float64Array(Hh);
  // Backward scratch.
  const dfused = new Float64Array(E);
  const dr0 = new Float64Array(Hh);
  const dp = new Float64Array(M);
  const dlg = new Float64Array(M);
  const dq0 = new Float64Array(G1);
  const de = new Float64Array(M * E);
  const maxH = Math.max(...enc.map((e) => e.H));
  const dg1 = new Float64Array(maxH);
  const dxh = new Float64Array(maxH);
  const da0 = new Float64Array(maxH);
  let allAbsent = false;

  const geluInto = (src, tanhOut, dst, len) => {
    for (let i = 0; i < len; i++) {
      const v = src[i];
      const t = Math.tanh(SQRT_2_OVER_PI * (v + GELU_C * v * v * v));
      tanhOut[i] = t;
      dst[i] = 0.5 * v * (1 + t);
    }
  };
  const geluGrad = (v, t) => 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * SQRT_2_OVER_PI * (1 + 3 * GELU_C * v * v);

  // Forward one row; returns the logit. mask: keep flags (Float64Array, offset mo) or null.
  function forwardRow(P, X, xo, A, ao, mask, mo, scale) {
    for (let m = 0; m < M; m++) {
      const c = enc[m];
      const H = c.H;
      const dIn = c.dIn;
      const xs = xo + c.start;
      const a = a0[m];
      for (let o = 0; o < H; o++) {
        let acc = P[c.b0 + o];
        const wo = c.W0 + o * dIn;
        for (let i = 0; i < dIn; i++) acc += X[xs + i] * P[wo + i];
        a[o] = acc;
      }
      const gs = H / c.groups;
      const xh = xhat[m];
      const nr = nrm[m];
      for (let g = 0; g < c.groups; g++) {
        const s0 = g * gs;
        let mean = 0;
        for (let i = 0; i < gs; i++) mean += a[s0 + i];
        mean /= gs;
        let variance = 0;
        for (let i = 0; i < gs; i++) {
          const d = a[s0 + i] - mean;
          variance += d * d;
        }
        variance /= gs;
        const is = 1 / Math.sqrt(variance + c.eps);
        inv[m][g] = is;
        for (let i = 0; i < gs; i++) {
          const k = s0 + i;
          const v = (a[k] - mean) * is;
          xh[k] = v;
          nr[k] = v * P[c.gw + k] + P[c.gb + k];
        }
      }
      const gg = g1[m];
      geluInto(nr, tn[m], gg, H);
      const eo = m * E;
      for (let o = 0; o < E; o++) {
        let acc = P[c.b1 + o];
        const wo = c.W1 + o * H;
        for (let i = 0; i < H; i++) acc += gg[i] * P[wo + i];
        emb[eo + o] = acc;
        gin[eo + o] = acc;
        th[eo + o] = Math.tanh(acc);
      }
    }
    allAbsent = true;
    for (let m = 0; m < M; m++) {
      const av = A[ao + m];
      gin[M * E + m] = av;
      if (av >= 0.5) allAbsent = false;
    }
    for (let k = 0; k < G1; k++) {
      let acc = P[bg0 + k];
      const wo = Wg0 + k * Gin;
      for (let i = 0; i < Gin; i++) acc += gin[i] * P[wo + i];
      q0[k] = acc;
    }
    geluInto(q0, tq, q1, G1);
    let mx = -Infinity;
    for (let m = 0; m < M; m++) {
      let acc = P[bg1 + m];
      const wo = Wg1 + m * G1;
      for (let k = 0; k < G1; k++) acc += q1[k] * P[wo + k];
      const v = allAbsent ? 0 : A[ao + m] < 0.5 ? FLOAT32_MIN : acc;
      lg[m] = v;
      if (v > mx) mx = v;
    }
    let sum = 0;
    for (let m = 0; m < M; m++) {
      const ex = Math.exp(lg[m] - mx);
      gate[m] = ex;
      sum += ex;
    }
    for (let m = 0; m < M; m++) gate[m] /= sum;
    fused.fill(0);
    for (let m = 0; m < M; m++) {
      const gm = gate[m];
      const eo = m * E;
      for (let d = 0; d < E; d++) fused[d] += gm * th[eo + d];
    }
    for (let j = 0; j < Hh; j++) {
      let acc = P[bh0 + j];
      const wo = Wh0 + j * E;
      for (let d = 0; d < E; d++) acc += fused[d] * P[wo + d];
      r0[j] = acc;
    }
    geluInto(r0, tr, r1, Hh);
    let z = P[bh1];
    if (mask) {
      for (let j = 0; j < Hh; j++) {
        const v = (r1[j] * mask[mo + j]) / scale;
        r2[j] = v;
        z += v * P[Wh1 + j];
      }
    } else {
      for (let j = 0; j < Hh; j++) {
        r2[j] = r1[j];
        z += r1[j] * P[Wh1 + j];
      }
    }
    return z;
  }

  // Backprop dz (dLoss/dlogit for this row) into G, using the caches of the last forwardRow.
  function backwardRow(P, G, X, xo, dz, mask, mo, scale) {
    G[bh1] += dz;
    for (let j = 0; j < Hh; j++) {
      G[Wh1 + j] += dz * r2[j];
      let d = dz * P[Wh1 + j];
      if (mask) d = (d * mask[mo + j]) / scale;
      dr0[j] = d * geluGrad(r0[j], tr[j]);
    }
    dfused.fill(0);
    for (let j = 0; j < Hh; j++) {
      const d = dr0[j];
      G[bh0 + j] += d;
      const wo = Wh0 + j * E;
      for (let e = 0; e < E; e++) {
        G[wo + e] += d * fused[e];
        dfused[e] += P[wo + e] * d;
      }
    }
    let s = 0;
    for (let m = 0; m < M; m++) {
      const eo = m * E;
      let acc = 0;
      for (let d = 0; d < E; d++) {
        const t = th[eo + d];
        acc += dfused[d] * t;
        de[eo + d] = gate[m] * dfused[d] * (1 - t * t);
      }
      dp[m] = acc;
      s += gate[m] * acc;
    }
    for (let m = 0; m < M; m++) dlg[m] = allAbsent ? 0 : gate[m] * (dp[m] - s);
    for (let k = 0; k < G1; k++) {
      let acc = 0;
      for (let m = 0; m < M; m++) acc += P[Wg1 + m * G1 + k] * dlg[m];
      dq0[k] = acc * geluGrad(q0[k], tq[k]);
    }
    for (let m = 0; m < M; m++) {
      const d = dlg[m];
      G[bg1 + m] += d;
      const wo = Wg1 + m * G1;
      for (let k = 0; k < G1; k++) G[wo + k] += d * q1[k];
    }
    const nE = M * E;
    for (let k = 0; k < G1; k++) {
      const d = dq0[k];
      G[bg0 + k] += d;
      const wo = Wg0 + k * Gin;
      for (let i = 0; i < Gin; i++) G[wo + i] += d * gin[i];
      for (let i = 0; i < nE; i++) de[i] += P[wo + i] * d;
    }
    for (let m = 0; m < M; m++) {
      const c = enc[m];
      const H = c.H;
      const eo = m * E;
      const gg = g1[m];
      dg1.fill(0, 0, H);
      for (let o = 0; o < E; o++) {
        const d = de[eo + o];
        G[c.b1 + o] += d;
        const wo = c.W1 + o * H;
        for (let i = 0; i < H; i++) {
          G[wo + i] += d * gg[i];
          dg1[i] += P[wo + i] * d;
        }
      }
      const xh = xhat[m];
      const nr = nrm[m];
      const tt = tn[m];
      for (let k = 0; k < H; k++) {
        const dn = dg1[k] * geluGrad(nr[k], tt[k]);
        G[c.gw + k] += dn * xh[k];
        G[c.gb + k] += dn;
        dxh[k] = dn * P[c.gw + k];
      }
      const gs = H / c.groups;
      for (let g = 0; g < c.groups; g++) {
        const s0 = g * gs;
        let m1 = 0;
        let m2 = 0;
        for (let i = 0; i < gs; i++) {
          m1 += dxh[s0 + i];
          m2 += dxh[s0 + i] * xh[s0 + i];
        }
        m1 /= gs;
        m2 /= gs;
        const is = inv[m][g];
        for (let i = 0; i < gs; i++) {
          const k = s0 + i;
          da0[k] = is * (dxh[k] - m1 - xh[k] * m2);
        }
      }
      const dIn = c.dIn;
      const xs = xo + c.start;
      for (let o = 0; o < H; o++) {
        const d = da0[o];
        G[c.b0 + o] += d;
        const wo = c.W0 + o * dIn;
        for (let i = 0; i < dIn; i++) {
          const xv = X[xs + i];
          if (xv !== 0) G[wo + i] += d * xv;
        }
      }
    }
  }

  // Mean BCE-with-logits(pos_weight) over rows idx[start..start+count) (or start.. if idx null);
  // accumulates the batch-mean gradient into G (caller zeroes G).
  function batchLossGrad(P, G, X, y, A, idx, start, count, posWeight, mask, dropoutP) {
    const scale = 1 - dropoutP;
    let lossSum = 0;
    for (let b = 0; b < count; b++) {
      const r = idx ? idx[start + b] : start + b;
      const z = forwardRow(P, X, r * nF, A, r * M, mask, b * Hh, scale);
      const t = y[r] >= 0.5 ? 1 : 0;
      const lw = (posWeight - 1) * t + 1;
      // torch: (1 - y) * z - log_weight * log_sigmoid(z), with a stable log_sigmoid.
      const logSig = Math.min(z, 0) - Math.log1p(Math.exp(-Math.abs(z)));
      lossSum += (1 - t) * z - lw * logSig;
      const sig = 1 / (1 + Math.exp(-z));
      const dz = (1 - t + lw * (sig - 1)) / count;
      backwardRow(P, G, X, r * nF, dz, mask, b * Hh, scale);
    }
    return lossSum / count;
  }

  function predictInto(P, X, A, n, scores, attribution) {
    for (let r = 0; r < n; r++) {
      const z = forwardRow(P, X, r * nF, A, r * M, null, 0, 1);
      scores[r] = 1 / (1 + Math.exp(-z));
      if (attribution) for (let m = 0; m < M; m++) attribution[r * M + m] = gate[m];
    }
  }

  // Eval-mode mean BCE-with-logits (pos_weight) over n rows, from the logits (stable, never NaN).
  function evalLoss(P, X, y, A, n, posWeight) {
    let lossSum = 0;
    for (let r = 0; r < n; r++) {
      const z = forwardRow(P, X, r * nF, A, r * M, null, 0, 1);
      const t = y[r] >= 0.5 ? 1 : 0;
      const lw = (posWeight - 1) * t + 1;
      lossSum += (1 - t) * z - lw * (Math.min(z, 0) - Math.log1p(Math.exp(-Math.abs(z))));
    }
    return lossSum / Math.max(n, 1);
  }

  return { layout, nParams, nF, M, Hh, batchLossGrad, predictInto, evalLoss };
}

// ---------------------------------------------------------------------------
// Public API

/**
 * Fresh weights in weights.json format with PyTorch's default init: every
 * Linear weight and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)); GroupNorm
 * weight 1, bias 0, eps 1e-5, 4 groups (1 if hiddenDim is not divisible by 4).
 */
export function initWeights(rng, { embedDim = 16, hiddenDim = 32 } = {}) {
  const linear = (inDim, outDim) => {
    const bound = 1 / Math.sqrt(inDim);
    const weight = [];
    for (let o = 0; o < outDim; o++) {
      const row = new Array(inDim);
      for (let i = 0; i < inDim; i++) row[i] = rng.uniform(-bound, bound);
      weight.push(row);
    }
    const bias = new Array(outDim);
    for (let o = 0; o < outDim; o++) bias[o] = rng.uniform(-bound, bound);
    return { weight, bias };
  };
  const groups = hiddenDim % 4 === 0 ? 4 : 1;
  const M = DEFAULT_SLICES.length;
  const encoders = DEFAULT_SLICES.map(([a, b]) => ({
    linear0: linear(b - a, hiddenDim),
    groupnorm: {
      num_groups: groups,
      num_channels: hiddenDim,
      eps: 1e-5,
      weight: new Array(hiddenDim).fill(1),
      bias: new Array(hiddenDim).fill(0),
    },
    linear1: linear(hiddenDim, embedDim),
  }));
  const fusion = {
    n_modalities: M,
    embed_dim: embedDim,
    gate_linear0: linear(M * embedDim + M, 2 * M),
    gate_linear1: linear(2 * M, M),
  };
  const head = { linear0: linear(embedDim, hiddenDim), linear1: linear(hiddenDim, 1) };
  return {
    architecture: "tessera_base_v1",
    gelu: "tanh_approx",
    modality_slices: DEFAULT_SLICES.map((s) => s.slice()),
    modality_names: MODALITY_NAMES.slice(),
    n_features: DEFAULT_SLICES[M - 1][1],
    encoders,
    fusion,
    head,
  };
}

/** Number of trainable parameters (5,005 for the default TESSERA-base). */
export function countParams(weights) {
  return tensorsOf(weights).reduce((a, [, t]) => a + tensorSize(t), 0);
}

/**
 * Mean BCE-with-logits (pos_weight) loss and its exact gradient for one batch.
 * batch = {X (b*42), y (b), avail (b*4), b}. dropoutMask null = eval mode;
 * otherwise a Float64Array(b*32) of 0/1 keep flags applied as h * mask / (1 - p).
 * Returns {loss, grads} with grads mirroring the weights structure.
 */
export function lossAndGrads(weights, batch, { posWeight = 1, dropoutMask = null, dropoutP = 0.2 } = {}) {
  const core = getCore(weights);
  const P = flattenParams(weights);
  const G = new Float64Array(P.length);
  const loss = core.batchLossGrad(P, G, batch.X, batch.y, batch.avail, null, 0, batch.b, posWeight, dropoutMask, dropoutP);
  return { loss, grads: gradsFromFlat(weights, G) };
}

function perTensorNorm(g, layout) {
  // torch.nn.utils.clip_grad_norm_: norm of the per-tensor L2 norms.
  let total = 0;
  for (const t of layout) {
    let s = 0;
    for (let i = t.offset; i < t.offset + t.size; i++) s += g[i] * g[i];
    total += s; // (sqrt(s))^2
  }
  return Math.sqrt(total);
}

function clipFlat(g, maxNorm, layout) {
  const norm = layout ? perTensorNorm(g, layout) : Math.sqrt(g.reduce((a, v) => a + v * v, 0));
  const coef = maxNorm / (norm + 1e-6);
  if (coef < 1) for (let i = 0; i < g.length; i++) g[i] *= coef;
  return norm;
}

/**
 * Global L2 gradient clipping, exactly torch.nn.utils.clip_grad_norm_: scales
 * every gradient by maxNorm / (norm + 1e-6) when that is < 1. Works on a
 * nested grads object (in place) or a flat Float64Array. Returns the norm
 * before clipping.
 */
export function clipGradNorm(grads, maxNorm = 1) {
  if (ArrayBuffer.isView(grads)) return clipFlat(grads, maxNorm, null);
  const flat = flattenParams(grads);
  const norm = clipFlat(flat, maxNorm, paramLayout(grads));
  writeFlat(grads, flat);
  return norm;
}

/**
 * torch.optim.AdamW, exactly (decoupled decay p *= 1 - lr*wd first, then Adam
 * with bias correction). `params` is a nested weights object (updated in place)
 * or a flat Float64Array (updated in place). step(grads, lr) takes grads in the
 * same form.
 */
export function createAdamW(params, { lr = 1e-3, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 1e-2 } = {}) {
  const nested = !ArrayBuffer.isView(params);
  const P = nested ? flattenParams(params) : params;
  const mBuf = new Float64Array(P.length);
  const vBuf = new Float64Array(P.length);
  const [b1, b2] = betas;
  let t = 0;
  return {
    get stepCount() {
      return t;
    },
    step(grads, stepLr = lr) {
      const g = ArrayBuffer.isView(grads) ? grads : flattenParams(grads);
      t += 1;
      const bc1 = 1 - b1 ** t;
      const bc2Sqrt = Math.sqrt(1 - b2 ** t);
      const stepSize = stepLr / bc1;
      const decay = 1 - stepLr * weightDecay;
      for (let i = 0; i < P.length; i++) {
        const gi = g[i];
        P[i] *= decay;
        const m = mBuf[i] + (1 - b1) * (gi - mBuf[i]); // torch lerp_
        const v = vBuf[i] * b2 + (1 - b2) * (gi * gi); // torch mul_(b2).addcmul_(g, g, 1 - b2)
        mBuf[i] = m;
        vBuf[i] = v;
        P[i] -= stepSize * (m / (Math.sqrt(v) / bc2Sqrt + eps));
      }
      if (nested) writeFlat(params, P);
    },
  };
}

/** CosineAnnealingLR(T_max = totalEpochs, eta_min = 0): the LR used during epoch `epoch` (0-based). */
export function cosineLr(baseLr, epoch, totalEpochs, etaMin = 0) {
  return etaMin + ((baseLr - etaMin) * (1 + Math.cos((Math.PI * epoch) / totalEpochs))) / 2;
}

/**
 * sklearn.metrics.average_precision_score (step-wise, ties grouped by
 * distinct score). Returns null when y has a single class.
 */
export function averagePrecisionScore(y, s, n = y.length) {
  let nPos = 0;
  for (let i = 0; i < n; i++) if (y[i] >= 0.5) nPos++;
  if (nPos === 0 || nPos === n) return null;
  const order = new Int32Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => s[b] - s[a]);
  let tp = 0;
  let fp = 0;
  let prevRecall = 0;
  let ap = 0;
  for (let k = 0; k < n; k++) {
    const i = order[k];
    if (y[i] >= 0.5) tp++;
    else fp++;
    if (k === n - 1 || s[order[k + 1]] !== s[i]) {
      const recall = tp / nPos;
      ap += (recall - prevRecall) * (tp / (tp + fp));
      prevRecall = recall;
    }
  }
  return ap;
}

/** Scores (sigmoid) and gate attributions for n rows; equals forward.js row by row. */
export function predict(weights, { X, avail, n }) {
  const core = getCore(weights);
  const P = flattenParams(weights);
  const scores = new Float64Array(n);
  const attribution = new Float64Array(n * core.M);
  core.predictInto(P, X, avail, n, scores, attribution);
  return { scores, attribution };
}

const yieldToLoop = () => new Promise((r) => setTimeout(r, 0));

/**
 * earlyStopMetricFor(val) -> 'val_ap' when the validation set has both classes,
 * 'val_loss' when it has rows but only one class (AP is undefined there),
 * 'none' when there are no validation rows.
 */
export function earlyStopMetricFor(val) {
  if (!val || !(val.n > 0)) return "none";
  let nPos = 0;
  for (let i = 0; i < val.n; i++) if (val.y[i] >= 0.5) nPos++;
  return nPos > 0 && nPos < val.n ? "val_ap" : "val_loss";
}

/**
 * Train TESSERA-base with the TesseraBase.fit recipe (see file header).
 * train/val = {X (n*42), y (n), avail (n*4), n}. Returns {weights (best
 * validation epoch, weights.json format), history, nParams, elapsedMs, cancelled}.
 * history.earlyStopMetric: 'val_ap' (the fit() recipe), 'val_loss' (validation
 * is one class, so AP is undefined: best = lowest validation loss, same
 * patience) or 'none' (no validation rows: the final epoch is kept, no early
 * stop). history.valLoss holds the per-epoch validation loss in 'val_loss' mode.
 */
export async function trainTesseraBase({
  train,
  val = null,
  config = {},
  onProgress = null,
  shouldCancel = null,
  yieldEvery = 4,
}) {
  const {
    epochs = 30,
    lr = 1e-3,
    weightDecay = 1e-2,
    batchSize = 256,
    patience = 5,
    seed = 0,
    dropout = 0.2,
    clipNorm = 1,
    embedDim = 16,
    hiddenDim = 32,
  } = config;
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now()) - t0;

  const rng = makeRng(seed, "train");
  const weights = initWeights(rng.fork("init"), { embedDim, hiddenDim });
  const shuffleRng = rng.fork("shuffle");
  const dropRng = rng.fork("dropout");
  const core = getCore(weights);
  const P = flattenParams(weights);
  const G = new Float64Array(P.length);
  const opt = createAdamW(P, { lr, weightDecay });

  const n = train.n;
  let nPos = 0;
  for (let i = 0; i < n; i++) if (train.y[i] >= 0.5) nPos++;
  const posWeight = (n - nPos) / Math.max(nPos, 1);

  const perm = new Int32Array(n);
  for (let i = 0; i < n; i++) perm[i] = i;
  const batches = Math.max(1, Math.ceil(n / batchSize));
  const mask = new Float64Array(batchSize * core.Hh);
  const keepP = 1 - dropout;
  const earlyStopMetric = earlyStopMetricFor(val);
  const haveVal = earlyStopMetric !== "none";
  const byLoss = earlyStopMetric === "val_loss";
  const valScores = haveVal ? new Float64Array(val.n) : null;

  const history = { trainLoss: [], valAp: [], valLoss: [], lr: [], bestEpoch: -1, stoppedEarly: false, epochsRun: 0, earlyStopMetric };
  let bestAp = -1;
  let bestLoss = Infinity;
  let bestP = null;
  let badEpochs = 0;
  let cancelled = false;
  let sinceYield = 0;

  for (let epoch = 0; epoch < epochs && !cancelled; epoch++) {
    const epochLr = cosineLr(lr, epoch, epochs);
    shuffleRng.shuffle(perm);
    let lossSum = 0;
    let nb = 0;
    for (let b = 0; b < batches; b++) {
      if (shouldCancel && shouldCancel()) {
        cancelled = true;
        break;
      }
      const start = b * batchSize;
      const count = Math.min(batchSize, n - start);
      const useMask = dropout > 0 ? mask : null;
      if (useMask) for (let i = 0; i < count * core.Hh; i++) mask[i] = dropRng.next() < keepP ? 1 : 0;
      G.fill(0);
      const loss = core.batchLossGrad(P, G, train.X, train.y, train.avail, perm, start, count, posWeight, useMask, dropout);
      clipFlat(G, clipNorm, core.layout);
      opt.step(G, epochLr);
      lossSum += loss;
      nb++;
      if (onProgress) {
        onProgress({
          phase: "batch",
          epoch,
          epochs,
          batch: b,
          batches,
          fraction: (epoch * batches + b + 1) / (epochs * batches),
          loss,
          valAp: null,
          lr: epochLr,
        });
      }
      if (++sinceYield >= yieldEvery) {
        sinceYield = 0;
        await yieldToLoop();
      }
    }
    if (cancelled) break;

    const epochLoss = lossSum / Math.max(nb, 1);
    history.trainLoss.push(epochLoss);
    history.lr.push(epochLr);
    history.epochsRun = epoch + 1;
    let valAp = null;
    let valLoss = null;
    if (byLoss) {
      valLoss = core.evalLoss(P, val.X, val.y, val.avail, val.n, posWeight);
      history.valLoss.push(valLoss);
      if (valLoss < bestLoss) {
        bestLoss = valLoss;
        history.bestEpoch = epoch;
        bestP = P.slice();
        badEpochs = 0;
      } else {
        badEpochs++;
      }
    } else if (haveVal) {
      core.predictInto(P, val.X, val.avail, val.n, valScores, null);
      valAp = averagePrecisionScore(val.y, valScores, val.n) ?? 0; // both classes are present here
      history.valAp.push(valAp);
      if (valAp > bestAp) {
        bestAp = valAp;
        history.bestEpoch = epoch;
        bestP = P.slice();
        badEpochs = 0;
      } else {
        badEpochs++;
      }
    }
    if (onProgress) {
      onProgress({
        phase: "epoch",
        epoch,
        epochs,
        batch: batches - 1,
        batches,
        fraction: (epoch + 1) / epochs,
        loss: epochLoss,
        valAp,
        valLoss,
        lr: epochLr,
      });
    }
    if (haveVal && badEpochs >= patience) {
      history.stoppedEarly = true; // as fit(): have_val and bad_epochs >= patience
      break;
    }
    await yieldToLoop();
    sinceYield = 0;
  }

  if (!haveVal) history.bestEpoch = history.epochsRun - 1;
  const finalP = haveVal && bestP ? bestP : P;
  const out = writeFlat(cloneStructure(weights), finalP);
  return {
    weights: out,
    history,
    nParams: core.nParams,
    elapsedMs: now(),
    cancelled,
  };
}
