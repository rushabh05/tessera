// k-nearest-neighbour "memoriser" for the Training Lab (vanilla JS, float64).
//
// Why it is here: the project's central finding is that a leaky evaluation
// flatters a model. The real Finding 1 used a gradient-boosted tree ensemble
// (LightGBM), a model that can memorise individual training windows. The
// Lab's own TESSERA-base (5,005 parameters) cannot memorise much, so on its own
// it hides the mechanism. A nearest-neighbour model is the purest memoriser:
// it has no parameters at all and scores a test window by copying the labels
// of the most similar training windows. When a split lets near-copies of test
// windows sit in training (a random row-level split), it looks brilliant; when
// the test comes from somewhere training has not seen, it does not.
//
// Recipe (deterministic, exact):
//   * features: the columns of splits.featureColumns(featureSet) ('all' = 42,
//     'm2' = the 24 network-metric columns); M3 host identity stays in for 'all';
//   * transform: log1p (the features are zero-inflated and heavy-tailed), then
//     standardised with the TRAINING rows' mean and std only (std floor 1e-6);
//   * distance: squared Euclidean in that space;
//   * score: sum_j w_j * label_j / sum_j w_j over the k nearest training rows,
//     w_j = 2^-(j-1) with j = 1 for the nearest, so the nearest neighbour
//     dominates and the others refine ties;
//   * ties: equal distances are ordered by training-row index (lower first).
// Exactness: training rows are scanned in index order and a row enters the
// current best-k only if its distance is strictly smaller than the k-th best,
// so the result equals "sort all rows by (distance, index), take k". A
// partially summed distance that already reaches the k-th best is abandoned
// early; the squares are non-negative, so that never changes the answer.

import { featureColumns } from "./splits.js";

const STD_FLOOR = 1e-6;
const YIELD_EVERY = 64; // test rows between progress reports / event-loop yields
const yieldToLoop = () => new Promise((r) => setTimeout(r, 0));

const logv = (v) => Math.log1p(v > 0 ? v : 0);

/**
 * trainKnn({train, config:{k = 5, featureSet = 'all'}}) -> model.
 * train = {X (n*42), y (n), n}. "Training" only stores the transformed rows:
 * model = {kind:'knn', k, featureSet, cols (Int32Array), d, nFeatures (42),
 *          mean, std (Float64Array(d)), Z (Float64Array(n*d)), y (Uint8Array(n)), n,
 *          nStored, elapsedMs}.
 */
export function trainKnn({ train, config = {} }) {
  const { k = 5, featureSet = "all" } = config;
  if (!Number.isInteger(k) || k < 1) throw new Error(`k must be a positive whole number (got ${k}).`);
  const t0 = Date.now();
  const cols = featureColumns(featureSet);
  const d = cols.length;
  const n = train.n;
  const F = Math.round(train.X.length / Math.max(n, 1)) || 42;

  const mean = new Float64Array(d);
  const std = new Float64Array(d);
  const Z = new Float64Array(n * d);
  for (let r = 0; r < n; r++) {
    const xo = r * F;
    const zo = r * d;
    for (let j = 0; j < d; j++) {
      const v = logv(train.X[xo + cols[j]]);
      Z[zo + j] = v;
      mean[j] += v;
    }
  }
  for (let j = 0; j < d; j++) mean[j] /= Math.max(n, 1);
  for (let r = 0; r < n; r++) {
    const zo = r * d;
    for (let j = 0; j < d; j++) {
      const dv = Z[zo + j] - mean[j];
      std[j] += dv * dv;
    }
  }
  for (let j = 0; j < d; j++) std[j] = Math.max(Math.sqrt(std[j] / Math.max(n, 1)), STD_FLOOR);
  for (let r = 0; r < n; r++) {
    const zo = r * d;
    for (let j = 0; j < d; j++) Z[zo + j] = (Z[zo + j] - mean[j]) / std[j];
  }
  const y = new Uint8Array(n);
  for (let r = 0; r < n; r++) y[r] = train.y[r] >= 0.5 ? 1 : 0;

  return {
    kind: "knn",
    k,
    featureSet,
    cols,
    d,
    nFeatures: F,
    transform: "log1p+train-standardise",
    mean,
    std,
    Z,
    y,
    n,
    nStored: n,
    elapsedMs: Date.now() - t0,
  };
}

/** Neighbour weights w_j = 2^-(j-1), j = 1..k. */
export function knnWeights(k) {
  const w = new Float64Array(k);
  for (let j = 0; j < k; j++) w[j] = 2 ** -j;
  return w;
}

/**
 * async predictKnn(model, {X, n}, {onProgress, shouldCancel}) -> Float64Array(n) scores in [0, 1],
 * or null if shouldCancel() returned true (checked every 64 test rows).
 * onProgress({done, total, fraction}) is called about every 64 rows and once at the end.
 */
export async function predictKnn(model, { X, n }, { onProgress = null, shouldCancel = null } = {}) {
  const { Z, y, d, cols, mean, std } = model;
  const nTrain = model.n;
  const F = model.nFeatures || Math.round(X.length / Math.max(n, 1)) || 42;
  const k = Math.min(model.k, nTrain);
  const w = knnWeights(Math.max(k, 1));
  let wSum = 0;
  for (let j = 0; j < k; j++) wSum += w[j];

  const q = new Float64Array(d);
  const bestD = new Float64Array(Math.max(k, 1));
  const bestI = new Int32Array(Math.max(k, 1));
  const scores = new Float64Array(n);

  for (let r = 0; r < n; r++) {
    if (r % YIELD_EVERY === 0) {
      if (shouldCancel && shouldCancel()) return null;
      if (r > 0) {
        if (onProgress) onProgress({ done: r, total: n, fraction: r / n });
        await yieldToLoop();
        if (shouldCancel && shouldCancel()) return null;
      }
    }
    if (k === 0) {
      scores[r] = 0;
      continue;
    }
    const xo = r * F;
    for (let j = 0; j < d; j++) q[j] = (logv(X[xo + cols[j]]) - mean[j]) / std[j];

    let filled = 0;
    let worst = Infinity; // k-th best distance once filled
    for (let i = 0; i < nTrain; i++) {
      const zo = i * d;
      let s = 0;
      let j = 0;
      for (; j < d; j++) {
        const dv = Z[zo + j] - q[j];
        s += dv * dv;
        if (s >= worst) break; // cannot be strictly better than the k-th best
      }
      if (j < d) continue;
      if (filled < k) {
        // insert keeping (distance, index) order; a later index goes after equal distances
        let p = filled++;
        while (p > 0 && bestD[p - 1] > s) {
          bestD[p] = bestD[p - 1];
          bestI[p] = bestI[p - 1];
          p--;
        }
        bestD[p] = s;
        bestI[p] = i;
        if (filled === k) worst = bestD[k - 1];
      } else if (s < worst) {
        let p = k - 1;
        while (p > 0 && bestD[p - 1] > s) {
          bestD[p] = bestD[p - 1];
          bestI[p] = bestI[p - 1];
          p--;
        }
        bestD[p] = s;
        bestI[p] = i;
        worst = bestD[k - 1];
      }
    }
    let acc = 0;
    for (let j = 0; j < k; j++) acc += w[j] * y[bestI[j]];
    scores[r] = acc / wSum;
  }
  if (onProgress) onProgress({ done: n, total: n, fraction: 1 });
  return scores;
}
