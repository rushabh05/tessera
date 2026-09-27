// Evaluation metrics for the Training Lab, written to reproduce scikit-learn
// 1.9 and tessera.eval (metrics.py / stats.py) exactly - checked value-for-value
// against golden numbers that sklearn itself produced
// (web/data/metrics_golden.json, `uv run python -m tessera.demo.golden_metrics`,
// asserted by web/tests/metrics.test.mjs to 1e-12).
//
// Reporting order follows the project's rule: average precision (AP) first,
// MCC second; accuracy is computed only so the UI can show it de-emphasised.
//
// Conventions shared with the Python side:
//   * a window is predicted "attack" iff score >= threshold (binary_metrics);
//   * ties are grouped by distinct score value (sklearn's _binary_clf_curve), so
//     duplicated rows - which produce identical scores - are handled the way
//     sklearn handles them, not by an arbitrary order;
//   * a single-class input has no ranking metric: ap / rocAuc / curves are null,
//     never NaN and never a throw.
//
// Every function takes y as a Uint8Array or array of 0/1 and s as a
// Float32Array / Float64Array / array. Ranking work sorts once (O(n log n));
// evaluateAll shares that sort across AP, ROC AUC and both curves.

export const MIN_SUPPORT_FOR_RATES = 20;

// ------------------------------------------------------------------ helpers

/** Neumaier-compensated sum: within ~1 ulp of the exact sum, so it agrees with
 *  numpy's pairwise summation far below the 1e-12 test tolerance. */
function csum(values, n = values.length) {
  let sum = 0;
  let c = 0;
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const t = sum + v;
    if (Math.abs(sum) >= Math.abs(v)) c += sum - t + v;
    else c += v - t + sum;
    sum = t;
  }
  return sum + c;
}

function checkInputs(y, s) {
  if (!y || !s || typeof y.length !== "number" || typeof s.length !== "number") {
    throw new TypeError("metrics: y and s must be array-like");
  }
  if (y.length !== s.length) {
    throw new RangeError(`metrics: y has ${y.length} entries but s has ${s.length}`);
  }
}

function countPositives(y) {
  let p = 0;
  for (let i = 0; i < y.length; i++) if (y[i] === 1 || y[i] === true) p++;
  return p;
}

const isPos = (v) => v === 1 || v === true;

/**
 * sklearn's confusion_matrix_at_thresholds (formerly _binary_clf_curve):
 * scores sorted descending, one entry per distinct score (the last index of each
 * tie group). Returns null for empty input. Throws only on non-finite scores
 * (sklearn raises on those too - they indicate a bug upstream).
 */
function clfCurve(y, s) {
  checkInputs(y, s);
  const n = y.length;
  if (n === 0) return null;
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(s[i])) throw new RangeError(`metrics: score ${i} is not finite (${s[i]})`);
    idx[i] = i;
  }
  idx.sort((a, b) => s[b] - s[a]);
  const tps = [];
  const fps = [];
  const thresholds = [];
  let tp = 0;
  for (let k = 0; k < n; k++) {
    const i = idx[k];
    if (isPos(y[i])) tp++;
    // end of a tie group: last element, or the next sorted score differs
    if (k === n - 1 || s[idx[k + 1]] !== s[i]) {
      tps.push(tp);
      fps.push(k + 1 - tp);
      thresholds.push(+s[i]);
    }
  }
  return { tps, fps, thresholds, nPos: tp, nNeg: n - tp, n };
}

// ------------------------------------------------------------------ curves

function prFromCurve(c) {
  if (!c || c.nPos === 0 || c.nNeg === 0) return null;
  const m = c.tps.length;
  const precision = new Array(m + 1);
  const recall = new Array(m + 1);
  const thresholds = new Array(m);
  const totalPos = c.tps[m - 1];
  for (let j = 0; j < m; j++) {
    const src = m - 1 - j; // reversed so recall is decreasing
    const tp = c.tps[src];
    const ps = tp + c.fps[src];
    precision[j] = ps !== 0 ? tp / ps : 0;
    recall[j] = tp / totalPos;
    thresholds[j] = c.thresholds[src];
  }
  precision[m] = 1;
  recall[m] = 0;
  return { precision, recall, thresholds };
}

function rocFromCurve(c) {
  if (!c || c.nPos === 0 || c.nNeg === 0) return null;
  let { tps, fps, thresholds } = c;
  const m = tps.length;
  if (m > 2) {
    // drop_intermediate=True: keep a point only where the second difference of
    // fps or tps is non-zero (a corner), plus both ends.
    const keep = [0];
    for (let i = 1; i < m - 1; i++) {
      const d2f = fps[i + 1] - 2 * fps[i] + fps[i - 1];
      const d2t = tps[i + 1] - 2 * tps[i] + tps[i - 1];
      if (d2f !== 0 || d2t !== 0) keep.push(i);
    }
    keep.push(m - 1);
    tps = keep.map((i) => c.tps[i]);
    fps = keep.map((i) => c.fps[i]);
    thresholds = keep.map((i) => c.thresholds[i]);
  }
  const lastF = fps[fps.length - 1];
  const lastT = tps[tps.length - 1];
  const fpr = [0];
  const tpr = [0];
  const thr = [Infinity];
  for (let i = 0; i < tps.length; i++) {
    fpr.push(fps[i] / lastF);
    tpr.push(tps[i] / lastT);
    thr.push(thresholds[i]);
  }
  return { fpr, tpr, thresholds: thr };
}

function apFromPr(pr) {
  if (!pr) return null;
  const { precision, recall } = pr;
  const terms = new Float64Array(recall.length - 1);
  for (let i = 0; i < terms.length; i++) terms[i] = (recall[i + 1] - recall[i]) * precision[i];
  return Math.max(0, -csum(terms));
}

function aucFromRoc(roc) {
  if (!roc) return null;
  const { fpr, tpr } = roc;
  const terms = new Float64Array(fpr.length - 1);
  for (let i = 0; i < terms.length; i++) terms[i] = ((fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i])) / 2;
  return csum(terms);
}

/** = sklearn.metrics.precision_recall_curve (drop_intermediate=False), including
 *  the final (precision 1, recall 0) point. null for single-class input. */
export function prCurve(y, s) {
  return prFromCurve(clfCurve(y, s));
}

/** = sklearn.metrics.roc_curve (drop_intermediate=True); thresholds[0] is
 *  Infinity as in sklearn. null for single-class input. */
export function rocCurve(y, s) {
  return rocFromCurve(clfCurve(y, s));
}

/** = sklearn.metrics.average_precision_score. null for single-class input. */
export function averagePrecision(y, s) {
  return apFromPr(prCurve(y, s));
}

/** = sklearn.metrics.roc_auc_score. null for single-class input. */
export function rocAuc(y, s) {
  return aucFromRoc(rocCurve(y, s));
}

// ------------------------------------------------------------------ threshold metrics

/** Confusion counts at a threshold; a window is flagged iff s >= thr. */
export function confusionAt(y, s, thr = 0.5) {
  checkInputs(y, s);
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  for (let i = 0; i < y.length; i++) {
    const pred = s[i] >= thr;
    if (isPos(y[i])) {
      if (pred) tp++;
      else fn++;
    } else if (pred) fp++;
    else tn++;
  }
  return { tp, fp, tn, fn };
}

/** sklearn.metrics.matthews_corrcoef from confusion counts (same float64 formula). */
function mccFromConfusion({ tp, fp, tn, fn }) {
  const t0 = tn + fp; // true-class row sums (labels 0, 1)
  const t1 = fn + tp;
  const p0 = tn + fn; // predicted-class column sums
  const p1 = fp + tp;
  const nCorrect = tn + tp;
  const nSamples = p0 + p1;
  const covYtYp = nCorrect * nSamples - (t0 * p0 + t1 * p1);
  const covYpYp = nSamples ** 2 - (p0 * p0 + p1 * p1);
  const covYtYt = nSamples ** 2 - (t0 * t0 + t1 * t1);
  const prod = covYpYp * covYtYt;
  if (prod === 0) return 0;
  return covYtYp / Math.sqrt(prod);
}

/** Metrics derived from confusion counts alone (zero-division -> 0, as sklearn's
 *  zero_division=0). balancedAccuracy follows sklearn: the mean recall over the
 *  classes actually present in y. */
export function metricsFromConfusion(confusion) {
  const { tp, fp, tn, fn } = confusion;
  const n = tp + fp + tn + fn;
  const div = (a, b) => (b !== 0 ? a / b : 0);
  const recalls = [];
  if (tn + fp > 0) recalls.push(tn / (tn + fp));
  if (tp + fn > 0) recalls.push(tp / (tp + fn));
  return {
    precision: div(tp, tp + fp),
    recall: div(tp, tp + fn),
    f1: div(2 * tp, 2 * tp + fp + fn),
    mcc: mccFromConfusion(confusion),
    accuracy: div(tp + tn, n),
    balancedAccuracy: recalls.length === 2 ? (recalls[0] + recalls[1]) / 2 : recalls.length === 1 ? recalls[0] : 0,
    specificity: div(tn, tn + fp),
    fpr: div(fp, fp + tn),
    confusion: { tp, fp, tn, fn },
  };
}

/** {precision, recall, f1, mcc, accuracy, balancedAccuracy, specificity, fpr,
 *  confusion, threshold} at threshold thr. */
export function metricsAt(y, s, thr = 0.5) {
  return { ...metricsFromConfusion(confusionAt(y, s, thr)), threshold: thr };
}

// ------------------------------------------------------------------ calibration

// np.linspace(0, 1, nBins + 1): i * (1 / nBins), last edge exactly 1.
function linspaceEdges(nBins) {
  const step = 1 / nBins;
  const edges = new Array(nBins + 1);
  for (let i = 0; i < nBins; i++) edges[i] = i * step;
  edges[nBins] = 1;
  return edges;
}

// tessera.eval.metrics.class_conditional_ece's inner _ece, on scores already
// clipped to [0, 1]: bins (lo, hi], the first bin [0, hi].
function eceMasked(y, p, mask, nBins) {
  let total = 0;
  let m = 0;
  for (let i = 0; i < p.length; i++) if (mask(i)) m++;
  if (m === 0) return null;
  const edges = linspaceEdges(nBins);
  const counts = new Float64Array(nBins);
  const posCounts = new Float64Array(nBins);
  const pVals = Array.from({ length: nBins }, () => []);
  const inBin = (v, k) => (k === 0 ? v >= edges[0] && v <= edges[1] : v > edges[k] && v <= edges[k + 1]);
  for (let i = 0; i < p.length; i++) {
    if (!mask(i)) continue;
    const v = p[i];
    // Same membership test as the Python loop ((lo, hi], first bin [0, hi]).
    // The bins tile [0, 1], so the arithmetic guess is off by at most one.
    let b = Math.min(nBins - 1, Math.max(0, Math.floor(v * nBins)));
    if (!inBin(v, b)) {
      if (b > 0 && inBin(v, b - 1)) b -= 1;
      else if (b < nBins - 1 && inBin(v, b + 1)) b += 1;
      else continue; // unreachable for v in [0, 1]
    }
    counts[b]++;
    if (isPos(y[i])) posCounts[b]++;
    pVals[b].push(v);
  }
  for (let k = 0; k < nBins; k++) {
    if (counts[k] === 0) continue;
    const meanY = posCounts[k] / counts[k];
    const meanP = csum(pVals[k]) / counts[k];
    total += (counts[k] / m) * Math.abs(meanY - meanP);
  }
  return total;
}

function clip01(s) {
  const p = new Float64Array(s.length);
  for (let i = 0; i < s.length; i++) p[i] = s[i] < 0 ? 0 : s[i] > 1 ? 1 : +s[i];
  return p;
}

/** Aggregate expected calibration error = tessera.eval.metrics
 *  class_conditional_ece(y, clip(s, 0, 1), n_bins)["ece_aggregate"]. null for n = 0.
 *  At attack prevalences this low it is dominated by the benign class - prefer
 *  classConditionalEce for the calibration claim. */
export function ece(y, s, nBins = 15) {
  checkInputs(y, s);
  return eceMasked(y, clip01(s), () => true, nBins);
}

/** {positive, negative, aggregate} = class_conditional_ece's three values
 *  (null where the class is empty). */
export function classConditionalEce(y, s, nBins = 15) {
  checkInputs(y, s);
  const p = clip01(s);
  return {
    positive: eceMasked(y, p, (i) => isPos(y[i]), nBins),
    negative: eceMasked(y, p, (i) => !isPos(y[i]), nBins),
    aggregate: eceMasked(y, p, () => true, nBins),
    nBins,
  };
}

// ------------------------------------------------------------------ histogram

/** Score histogram on [0, 1], split by class = np.histogram(clip(s, 0, 1),
 *  bins=nBins, range=(0, 1)) per class: bins [a, b), the last [a, b]. */
export function histogram(y, s, nBins = 20) {
  checkInputs(y, s);
  const edges = linspaceEdges(nBins);
  const benign = new Array(nBins).fill(0);
  const attack = new Array(nBins).fill(0);
  for (let i = 0; i < s.length; i++) {
    const v = s[i] < 0 ? 0 : s[i] > 1 ? 1 : +s[i];
    // numpy's uniform-bin fast path, including its edge corrections
    let b = Math.floor(v * nBins);
    if (b === nBins) b -= 1;
    if (v < edges[b]) b -= 1;
    else if (b !== nBins - 1 && v >= edges[b + 1]) b += 1;
    if (isPos(y[i])) attack[b]++;
    else benign[b]++;
  }
  return { edges, benign, attack };
}

// ------------------------------------------------------------------ summaries

/** = tessera.eval.stats.summarise_seeds: mean, std with ddof = 1 (0 for a single
 *  value), min, max, n. Empty input gives nulls. Non-finite / null entries are
 *  dropped (the caller should pass only defined fold values). */
export function summarise(values) {
  const v = [];
  for (const x of values || []) if (x !== null && x !== undefined && Number.isFinite(+x)) v.push(+x);
  const n = v.length;
  if (n === 0) return { mean: null, std: null, min: null, max: null, n: 0 };
  const mean = csum(v) / n;
  let std = 0;
  if (n > 1) {
    const sq = v.map((x) => (x - mean) * (x - mean));
    std = Math.sqrt(csum(sq) / (n - 1));
  }
  let min = v[0];
  let max = v[0];
  for (const x of v) {
    if (x < min) min = x;
    if (x > max) max = x;
  }
  return { mean, std, min, max, n };
}

// ------------------------------------------------------------------ bundle

/**
 * Everything the results panels need from one (y, s) pair, with one sort:
 * {ap, rocAuc, prevalence, n, nPositive, atThreshold, pr, roc, ece, eceByClass,
 *  histogram, lowSupport, threshold}.
 * ap / rocAuc / pr / roc are null for a single-class input.
 */
export function evaluateAll(y, s, { threshold = 0.5 } = {}) {
  checkInputs(y, s);
  const n = y.length;
  const nPositive = countPositives(y);
  const curve = clfCurve(y, s);
  const pr = prFromCurve(curve);
  const roc = rocFromCurve(curve);
  return {
    ap: apFromPr(pr),
    rocAuc: aucFromRoc(roc),
    prevalence: n ? nPositive / n : null,
    n,
    nPositive,
    lowSupport: nPositive < MIN_SUPPORT_FOR_RATES,
    threshold,
    atThreshold: metricsAt(y, s, threshold),
    pr,
    roc,
    ece: ece(y, s),
    eceByClass: classConditionalEce(y, s),
    histogram: histogram(y, s),
  };
}
