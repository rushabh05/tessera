// Logistic-regression baseline for the Training Lab (vanilla JS, float64).
//
// A deliberately simple reference point next to TESSERA-base: every raw
// feature goes through log1p (the real features are zero-inflated and
// heavy-tailed, so a linear model on raw values would be dominated by a few
// huge counts), is standardised with the TRAINING set's mean and std only (no
// validation/test statistics leak into the transform), and the 4 availability
// bits are appended. Trained full-batch with Adam on class-weighted BCE
// (pos_weight = n_neg / n_pos, as TESSERA-base) plus an L2 penalty on the
// weights (not the bias). If a validation set with both classes is given, the
// epoch with the best validation average precision is kept; a one-class
// validation set (AP undefined) keeps the lowest validation loss instead, and
// no validation set keeps the final epoch (history.earlyStopMetric says which).

import { makeRng } from "./rng.js";
import { averagePrecisionScore, earlyStopMetricFor } from "./trainer.js";

const N_AVAIL = 4;

function transformInto(model, X, avail, n, out) {
  const d = model.nFeatures;
  const D = d + N_AVAIL;
  for (let r = 0; r < n; r++) {
    const xo = r * d;
    const zo = r * D;
    for (let j = 0; j < d; j++) {
      const v = X[xo + j];
      out[zo + j] = (Math.log1p(v > 0 ? v : 0) - model.mean[j]) / model.std[j];
    }
    for (let m = 0; m < N_AVAIL; m++) out[zo + d + m] = avail[r * N_AVAIL + m] >= 0.5 ? 1 : 0;
  }
  return out;
}

function scoresFromDesign(w, b, Z, n, D, out) {
  for (let r = 0; r < n; r++) {
    let z = b;
    const zo = r * D;
    for (let j = 0; j < D; j++) z += w[j] * Z[zo + j];
    out[r] = 1 / (1 + Math.exp(-z));
  }
  return out;
}

// Mean class-weighted BCE (from the logits, stable) of the model on a design matrix; no L2 term.
function designLoss(w, b, Z, y, n, D, posWeight) {
  let loss = 0;
  for (let r = 0; r < n; r++) {
    let z = b;
    const zo = r * D;
    for (let j = 0; j < D; j++) z += w[j] * Z[zo + j];
    const t = y[r] >= 0.5 ? 1 : 0;
    const lw = (posWeight - 1) * t + 1;
    loss += (1 - t) * z - lw * (Math.min(z, 0) - Math.log1p(Math.exp(-Math.abs(z))));
  }
  return loss / Math.max(n, 1);
}

const yieldToLoop = () => new Promise((r) => setTimeout(r, 0));

/**
 * trainLogReg({train, val, config:{epochs=200, lr=0.05, l2=1e-3, seed=0}, onProgress, shouldCancel})
 * train/val = {X (n*42), y (n), avail (n*4), n}. Returns a plain-JSON model for predictLogReg.
 */
export async function trainLogReg({ train, val = null, config = {}, onProgress = null, shouldCancel = null }) {
  const { epochs = 200, lr = 0.05, l2 = 1e-3, seed = 0 } = config;
  const t0 = Date.now();
  const n = train.n;
  const d = Math.round(train.X.length / Math.max(n, 1)) || 42;
  const D = d + N_AVAIL;

  const mean = new Array(d).fill(0);
  const std = new Array(d).fill(0);
  for (let r = 0; r < n; r++) for (let j = 0; j < d; j++) mean[j] += Math.log1p(Math.max(0, train.X[r * d + j]));
  for (let j = 0; j < d; j++) mean[j] /= Math.max(n, 1);
  for (let r = 0; r < n; r++) {
    for (let j = 0; j < d; j++) {
      const dv = Math.log1p(Math.max(0, train.X[r * d + j])) - mean[j];
      std[j] += dv * dv;
    }
  }
  for (let j = 0; j < d; j++) {
    const s = Math.sqrt(std[j] / Math.max(n, 1));
    std[j] = s > 1e-12 ? s : 1; // constant column: leave it centred, unscaled
  }
  const model = { kind: "logreg", nFeatures: d, transform: "log1p+train-standardise", mean, std, w: null, b: 0 };

  const Z = transformInto(model, train.X, train.avail, n, new Float64Array(n * D));
  const earlyStopMetric = earlyStopMetricFor(val);
  const haveVal = earlyStopMetric !== "none";
  const byLoss = earlyStopMetric === "val_loss";
  const Zv = haveVal ? transformInto(model, val.X, val.avail, val.n, new Float64Array(val.n * D)) : null;
  const sv = haveVal ? new Float64Array(val.n) : null;

  let nPos = 0;
  for (let r = 0; r < n; r++) if (train.y[r] >= 0.5) nPos++;
  const posWeight = (n - nPos) / Math.max(nPos, 1);

  const rng = makeRng(seed, "logreg");
  const w = new Float64Array(D);
  for (let j = 0; j < D; j++) w[j] = 0.01 * rng.normal();
  let b = 0;
  const gw = new Float64Array(D);
  const mW = new Float64Array(D);
  const vW = new Float64Array(D);
  let mB = 0;
  let vB = 0;
  const b1 = 0.9;
  const b2 = 0.999;
  const eps = 1e-8;

  const history = { trainLoss: [], valAp: [], valLoss: [], bestEpoch: -1, epochsRun: 0, earlyStopMetric };
  let best = null;
  let bestAp = -1;
  let bestLoss = Infinity;
  let cancelled = false;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (shouldCancel && shouldCancel()) {
      cancelled = true;
      break;
    }
    gw.fill(0);
    let gb = 0;
    let loss = 0;
    for (let r = 0; r < n; r++) {
      const zo = r * D;
      let z = b;
      for (let j = 0; j < D; j++) z += w[j] * Z[zo + j];
      const t = train.y[r] >= 0.5 ? 1 : 0;
      const lw = (posWeight - 1) * t + 1;
      const logSig = Math.min(z, 0) - Math.log1p(Math.exp(-Math.abs(z)));
      loss += (1 - t) * z - lw * logSig;
      const g = (1 - t + lw * (1 / (1 + Math.exp(-z)) - 1)) / n;
      gb += g;
      for (let j = 0; j < D; j++) gw[j] += g * Z[zo + j];
    }
    loss /= Math.max(n, 1);
    let reg = 0;
    for (let j = 0; j < D; j++) {
      reg += w[j] * w[j];
      gw[j] += l2 * w[j];
    }
    loss += 0.5 * l2 * reg;

    const k = epoch + 1;
    const bc1 = 1 - b1 ** k;
    const bc2 = 1 - b2 ** k;
    for (let j = 0; j < D; j++) {
      mW[j] = b1 * mW[j] + (1 - b1) * gw[j];
      vW[j] = b2 * vW[j] + (1 - b2) * gw[j] * gw[j];
      w[j] -= (lr * (mW[j] / bc1)) / (Math.sqrt(vW[j] / bc2) + eps);
    }
    mB = b1 * mB + (1 - b1) * gb;
    vB = b2 * vB + (1 - b2) * gb * gb;
    b -= (lr * (mB / bc1)) / (Math.sqrt(vB / bc2) + eps);

    history.trainLoss.push(loss);
    history.epochsRun = k;
    let valAp = null;
    let valLoss = null;
    if (byLoss) {
      valLoss = designLoss(w, b, Zv, val.y, val.n, D, posWeight);
      history.valLoss.push(valLoss);
      if (valLoss < bestLoss) {
        bestLoss = valLoss;
        history.bestEpoch = epoch;
        best = { w: Array.from(w), b };
      }
    } else if (haveVal) {
      scoresFromDesign(w, b, Zv, val.n, D, sv);
      valAp = averagePrecisionScore(val.y, sv, val.n) ?? 0;
      history.valAp.push(valAp);
      if (valAp > bestAp) {
        bestAp = valAp;
        history.bestEpoch = epoch;
        best = { w: Array.from(w), b };
      }
    }
    if (onProgress) onProgress({ phase: "epoch", epoch, epochs, fraction: k / epochs, loss, valAp, valLoss, lr });
    if (k % 10 === 0) await yieldToLoop();
  }

  if (best) {
    model.w = best.w;
    model.b = best.b;
  } else {
    model.w = Array.from(w);
    model.b = b;
    history.bestEpoch = history.epochsRun - 1;
  }
  model.history = history;
  model.elapsedMs = Date.now() - t0;
  model.cancelled = cancelled;
  model.nParams = D + 1;
  return model;
}

/** Scores in [0, 1] for n rows. */
export function predictLogReg(model, { X, avail, n }) {
  const D = model.nFeatures + N_AVAIL;
  const Z = transformInto(model, X, avail, n, new Float64Array(n * D));
  return scoresFromDesign(model.w, model.b, Z, n, D, new Float64Array(n));
}
