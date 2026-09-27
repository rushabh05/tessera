// The Training Lab pipeline: synthetic data -> split -> leakage certificate ->
// train -> score the test set -> metrics -> Merkle ledger, as one cancellable,
// progress-reporting async function.
//
// Pure: no DOM, no fetch. It runs unchanged in the Web Worker (worker.js) and in
// Node (web/tests/pipeline.test.mjs). The caller supplies the real-data summary
// statistics (web/data/replica_stats.json) and, optionally, the pretrained
// real-AIT weights (web/data/weights.json): only the 'pretrained' model needs
// them; without them every other model still runs and `fidelity` is null.
//
// Honesty notes baked into the code:
//   * every row this pipeline trains or tests on is SYNTHETIC (datagen.js);
//   * the 'pretrained' model was trained on REAL AIT data from every replica
//     except santos, so its leakage certificate carries a
//     'pretrained-contamination' check;
//   * `fidelity` is the real pretrained model's AP on this synthetic test set:
//     whether the real model separates synthetic attacks from synthetic benign
//     windows the way it does on real data - a necessary check on the
//     generator, not proof of realism and not a result on real data;
//   * leave-one-replica-out summaries exclude low-support folds (< 20 test
//     positives) from the mean/std, exactly like the project's Python reports,
//     and also give the naive all-folds mean for comparison; result.metrics in
//     LORO is pooled over the out-of-fold predictions of the same scoreable
//     folds only (result.metricsScope says so);
//   * the certificate's leakage verdict (overall) and its support verdict
//     (support) are separate: a test set without attacks is unscoreable, not
//     leaky;
//   * without Web Crypto (an insecure http:// page) the ledger stage is skipped,
//     not failed, so the run's results are kept;
//   * config.features = 'm2' restricts TESSERA-base, logistic regression and
//     the nearest-neighbour memoriser to the 24 network-metric columns (the
//     setup of the real Finding 1); the pretrained model was trained on all 42
//     features, so for it the setting is overridden (and said so) - never
//     silently applied to a model that cannot honour it;
//   * the leakage certificate compares rows on exactly the columns the model sees.

import { generateCorpus, corpusSummary, sliceRows } from "./datagen.js";
import {
  DEFAULT_SPLIT,
  PRETRAINED_TRAIN_REPLICAS,
  buildSplit,
  FEATURE_SETS,
  cloneSplitConfig,
  featureColumns,
  leakageCertificate,
  loroFolds,
  pctOf,
  validateSplitConfig,
  worstSeverity,
} from "./splits.js";
import { trainTesseraBase, predict, countParams } from "./trainer.js";
import { trainLogReg, predictLogReg } from "./logreg.js";
import { trainKnn, predictKnn } from "./knn.js";
import { evaluateAll, metricsAt, averagePrecision, summarise, MIN_SUPPORT_FOR_RATES } from "./metrics.js";
import { MerkleLog, setSha256, bytesToHex } from "../merkle.js";

/** Pipeline stages in order, with TESSERA-base's share (percent) of overall
 *  progress. Other models spend their time elsewhere: see stageWeights(). */
export const STAGES = Object.freeze([
  Object.freeze({ id: "generate", label: "Generate synthetic data", weight: 3 }),
  Object.freeze({ id: "split", label: "Split into train, validation and test", weight: 1 }),
  Object.freeze({ id: "leakage", label: "Check the split for leakage", weight: 2 }),
  Object.freeze({ id: "train", label: "Train the model", weight: 88 }),
  Object.freeze({ id: "test", label: "Score the test set", weight: 2 }),
  Object.freeze({ id: "evaluate", label: "Compute metrics", weight: 2 }),
  Object.freeze({ id: "ledger", label: "Commit verdicts to the ledger", weight: 2 }),
]);
const STAGE_IDS = STAGES.map((s) => s.id);
const FOLD_STAGES = ["split", "leakage", "train", "test", "evaluate"];

// Share (percent, sums to 100) of a run's work per stage, per model, from
// timings measured in Node at the default data size (scale 0.05): TESSERA-base
// spends its time training; the memoriser has nothing to fit and spends it
// searching for neighbours while scoring the test set; the pretrained model
// skips training.
const STAGE_WEIGHTS = Object.freeze({
  tessera: Object.freeze({ generate: 3, split: 1, leakage: 2, train: 88, test: 2, evaluate: 2, ledger: 2 }),
  logreg: Object.freeze({ generate: 13, split: 1, leakage: 2, train: 68, test: 3, evaluate: 2, ledger: 11 }),
  knn: Object.freeze({ generate: 18, split: 1, leakage: 2, train: 4, test: 62, evaluate: 2, ledger: 11 }),
  pretrained: Object.freeze({ generate: 48, split: 2, leakage: 3, train: 0, test: 12, evaluate: 3, ledger: 32 }),
});

/**
 * stageWeights(modelKind) -> {generate, split, leakage, train, test, evaluate,
 * ledger}: each stage's share (percent, summing to 100) of the overall progress
 * for that model. The same numbers ride on every 'stage' event (ev.weights), so
 * a UI's "x% of the work" labels always match the run. Unknown kinds get
 * TESSERA-base's weights.
 */
export function stageWeights(modelKind) {
  return { ...(STAGE_WEIGHTS[modelKind] || STAGE_WEIGHTS.tessera) };
}

/** At most this many verdicts are committed to the Merkle ledger per run. */
export const LEDGER_CAP = 5000;
/** The ledger stage's detail when no SHA-256 is available (no Web Crypto). */
export const LEDGER_SKIPPED_NO_CRYPTO = "Ledger skipped: Web Crypto needs https:// or localhost";
const MODEL_KINDS = ["tessera", "pretrained", "logreg", "knn"];
/** The nearest-neighbour memoriser's default number of neighbours. */
export const KNN_DEFAULT_K = 5;
const PROGRESS_INTERVAL_MS = 50; // <= 20 progress events per second
const SCORE_CHUNK = 2048;

export const DEFAULT_PIPELINE_CONFIG = Object.freeze({
  dataset: Object.freeze({ scale: 0.05, seed: 0 }),
  split: DEFAULT_SPLIT,
  model: Object.freeze({ kind: "tessera", epochs: 30, lr: 1e-3, batchSize: 256, patience: 5, seed: 0 }),
  features: "all",
  threshold: 0.5,
  ledger: true,
});

/** Plain-language names of the feature sets. */
export const FEATURE_LABELS = Object.freeze({
  all: "all four sources (42 features)",
  m2: "network metrics only (24 features)",
});

const nowMs = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const tick = () => new Promise((r) => setTimeout(r, 0));
const fmtInt = (n) => Math.round(n).toLocaleString("en");
const fmt3 = (v) => (v == null || !Number.isFinite(v) ? "n/a" : v.toFixed(3));

class Cancelled extends Error {
  constructor() {
    super("Cancelled");
    this.name = "Cancelled";
  }
}

const isPosInt = (v) => Number.isInteger(v) && v >= 1;
const isPosNum = (v) => typeof v === "number" && Number.isFinite(v) && v > 0;

/**
 * normaliseConfig(config, stats) -> {config (defaults filled, deep-copied), errors[], warnings[]}.
 * errors are plain-language sentences; warnings say what was adjusted (e.g. the
 * pretrained model always sees all 42 features, so features 'm2' becomes 'all').
 */
export function normaliseConfig(config = {}, stats = null) {
  const errors = [];
  const c = config || {};
  const dataset = { scale: 0.05, seed: 0, ...(c.dataset || {}) };
  if (Array.isArray(dataset.replicas)) dataset.replicas = [...dataset.replicas];
  const split = cloneSplitConfig(c.split || DEFAULT_SPLIT);
  const model = {
    kind: "tessera",
    epochs: 30,
    lr: 1e-3,
    batchSize: 256,
    patience: 5,
    seed: 0,
    weightDecay: 1e-2,
    dropout: 0.2,
    clipNorm: 1,
    ...(c.model || {}),
  };
  if (model.logreg) model.logreg = { ...model.logreg };
  if (model.kind === "knn" && model.k === undefined) model.k = KNN_DEFAULT_K;
  const threshold = c.threshold ?? 0.5;
  const ledger = c.ledger ?? true;
  const warnings = [];
  let features = c.features ?? "all";

  if (!isPosNum(dataset.scale) || dataset.scale > 1) {
    errors.push(`The dataset size must be between 0 and 100% of the real replica sizes (got ${dataset.scale}).`);
  }
  if (!Number.isInteger(dataset.seed)) errors.push(`The data seed must be a whole number (got ${dataset.seed}).`);
  if (!MODEL_KINDS.includes(model.kind)) {
    errors.push(
      `Unknown model "${model.kind}". Choose TESSERA-base (train), the pretrained model, logistic regression, or the nearest-neighbour memoriser.`,
    );
  }
  if (!FEATURE_SETS.includes(features)) {
    errors.push(`Unknown feature set "${features}". Choose all four sources ("all") or network metrics only ("m2").`);
  } else if (model.kind === "pretrained" && features !== "all") {
    warnings.push(
      "The pretrained model was trained on all 42 features of the real data, so it always sees all four sources: the network-metrics-only setting does not apply to it and was ignored.",
    );
    features = "all";
  }
  if (model.kind === "knn" && (!Number.isInteger(model.k) || model.k < 1 || model.k > 100)) {
    errors.push(`The number of neighbours must be a whole number from 1 to 100 (got ${model.k}).`);
  }
  if (model.kind === "tessera") {
    if (!isPosInt(model.epochs) || model.epochs > 500) errors.push(`Epochs must be a whole number from 1 to 500 (got ${model.epochs}).`);
    if (!isPosNum(model.lr) || model.lr > 1) errors.push(`The learning rate must be a positive number up to 1 (got ${model.lr}).`);
    if (!isPosInt(model.batchSize)) errors.push(`The batch size must be a positive whole number (got ${model.batchSize}).`);
    if (!isPosInt(model.patience)) errors.push(`Early-stopping patience must be a positive whole number (got ${model.patience}).`);
  }
  if (!Number.isInteger(model.seed)) errors.push(`The training seed must be a whole number (got ${model.seed}).`);
  if (typeof threshold !== "number" || !(threshold >= 0 && threshold <= 1)) {
    errors.push(`The decision threshold must be between 0 and 1 (got ${threshold}).`);
  }

  if (stats && Array.isArray(stats.replicas)) {
    const allIds = stats.replicas.map((r) => r.id);
    if (dataset.replicas != null) {
      for (const id of dataset.replicas) if (!allIds.includes(id)) errors.push(`"${id}" is not a replica in this dataset.`);
    }
    const genIds = dataset.replicas == null ? allIds : allIds.filter((id) => dataset.replicas.includes(id));
    const v = validateSplitConfig(split, genIds);
    errors.push(...v.errors);
  }
  return { config: { dataset, split, model, features, threshold, ledger: !!ledger }, errors, warnings };
}

/** Overall-progress plan: monotone segments generate | (fold stages) x K | ledger. */
function makeProgressPlan(K, weight = STAGE_WEIGHTS.tessera) {
  const segs = new Map();
  let acc = 0;
  const add = (stage, fold, w) => {
    segs.set(`${stage}#${fold}`, { start: acc, w });
    acc += w;
  };
  add("generate", 0, weight.generate);
  for (let k = 0; k < K; k++) for (const st of FOLD_STAGES) add(st, k, weight[st] / K);
  add("ledger", 0, weight.ledger);
  return (stage, fold, p) => {
    const seg = segs.get(`${stage}#${FOLD_STAGES.includes(stage) ? fold : 0}`);
    if (!seg) return 0;
    const q = Math.max(0, Math.min(1, p));
    return acc > 0 ? Math.min(1, (seg.start + seg.w * q) / acc) : 0;
  };
}

/** Keep flags for the 42 columns of a feature set. */
function keepMask(featureSet) {
  const keep = new Uint8Array(42);
  for (const j of featureColumns(featureSet)) keep[j] = 1;
  return keep;
}

/**
 * Network-metrics-only rows for TESSERA-base (in place on a copy made by
 * sliceRows): every column outside M2 is zeroed and the M1 / M4 availability
 * bits are cleared, so the log and graph encoders are gated out. M3 (host
 * identity) keeps availability 1 with zeroed features, so the gate always has
 * one present modality besides M2 and the network's shape is unchanged.
 */
function maskRowsToM2(rows) {
  const keep = keepMask("m2");
  const { X, avail, n } = rows;
  for (let r = 0; r < n; r++) {
    const o = r * 42;
    for (let j = 0; j < 42; j++) if (!keep[j]) X[o + j] = 0;
    avail[r * 4] = 0;
    avail[r * 4 + 3] = 0;
  }
  return rows;
}

/**
 * Rows restricted to the columns of a feature set, for logistic regression:
 * X becomes n * |cols| (so the model only has weights for those columns) and
 * only the M2 and M3 availability bits are kept (M1 / M4 set to 0, a constant
 * the model cannot learn from).
 */
function projectRows(rows, featureSet) {
  const cols = featureColumns(featureSet);
  const d = cols.length;
  const X = new Float32Array(rows.n * d);
  const avail = new Uint8Array(rows.n * 4);
  for (let r = 0; r < rows.n; r++) {
    for (let j = 0; j < d; j++) X[r * d + j] = rows.X[r * 42 + cols[j]];
    avail[r * 4 + 1] = rows.avail[r * 4 + 1];
    avail[r * 4 + 2] = rows.avail[r * 4 + 2];
  }
  return { ...rows, X, avail };
}

/** Plain words for how the kept epoch was chosen. */
function earlyStopText(h, unit = "epoch") {
  if (h.earlyStopMetric === "val_loss") {
    return ` · validation AP is undefined (the validation windows are all one class), so the ${unit} was chosen by validation loss ${fmt3(h.valLoss[h.bestEpoch])}`;
  }
  if (h.earlyStopMetric === "none") return ` · no validation windows, so the final ${unit} was kept (no early stopping)`;
  return h.valAp.length ? ` · validation AP ${fmt3(h.valAp[h.bestEpoch])}` : "";
}

const SEVERITY_WORD = { ok: "passed", warn: "warning", fail: "failed" };

/** One line for the leakage stage: the leakage verdict, then test support. */
function checksDetail(cert) {
  const leak = cert.checks.filter((c) => c.group !== "support");
  const fails = leak.filter((c) => c.severity === "fail");
  const warns = leak.filter((c) => c.severity === "warn");
  let text;
  if (!fails.length && !warns.length) text = `All ${leak.length} leakage checks passed`;
  else {
    const parts = [];
    if (fails.length) parts.push(`${fails.length} failed (${fails.map((c) => c.label.toLowerCase()).join("; ")})`);
    if (warns.length) parts.push(`${warns.length} warning${warns.length === 1 ? "" : "s"} (${warns.map((c) => c.label.toLowerCase()).join("; ")})`);
    text = `Leakage checks: ${parts.join(", ")}`;
  }
  const support = cert.support ?? "ok";
  const nPos = cert.testPositives;
  const supportText =
    support === "ok"
      ? ""
      : nPos === 0
        ? " · test support failed: no attack windows, so AP is undefined (not a leak)"
        : ` · test support ${SEVERITY_WORD[support]}: only ${fmtInt(nPos)} attack window${nPos === 1 ? "" : "s"} (not a leak)`;
  return text + supportText;
}

const plural = (n, one, many = `${one}s`) => `${fmtInt(n)} ${n === 1 ? one : many}`;
const joinList = (items) => (items.length <= 1 ? items.join("") : `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`);

/** An aggregate sentence for a leakage check that is ok in every fold, so the
 *  merged certificate never shows one fold's numbers as if they were all folds'. */
function allFoldsOkDetail(id, certs, K) {
  if (id === "replica-disjoint") {
    return `In each of the ${K} folds, no replica appears on both sides: every test set is a testbed its model never saw.`;
  }
  if (id === "stretch-disjoint") {
    return `In each of the ${K} folds, every attack episode and every benign stretch sits entirely on one side of the split.`;
  }
  if (id === "duplicate-rows") {
    const m2 = certs[0]?.featureSet === "m2";
    const onCols = m2 ? "on the 24 network-metric columns the model sees" : "in all 42 features";
    const quiet = certs.map((c) => c.quietDuplicates ?? 0);
    const sumQuiet = quiet.reduce((a, b) => a + b, 0);
    const nTest = certs.reduce((a, c) => a + (c.nTest ?? 0), 0);
    if (sumQuiet === 0) return `In none of the ${K} folds is a test window an exact copy of a training or validation window ${onCols}.`;
    const lo = Math.min(...quiet);
    const hi = Math.max(...quiet);
    const range = lo === hi ? `${fmtInt(lo)} in every fold` : `${fmtInt(lo)} to ${fmtInt(hi)} per fold`;
    return `Across the ${K} folds, ${plural(sumQuiet, "test window")} (${nTest ? `${pctOf(sumQuiet, nTest)} of all test windows, ` : ""}${range}) equal a training or validation window ${onCols}, and every one of them is a quiet window (${m2 ? "all 24 metric columns zero" : "no log, network-metric or graph activity, only the host identity"}). Such windows look alike on every testbed, so they reveal nothing about the held-out data and are not a leak. No fold has an active duplicate.`;
  }
  if (id === "pretrained-contamination") {
    return `The pretrained model never trained on the replica held out in any of the ${K} folds.`;
  }
  return `Passed in each of the ${K} folds.`;
}

/** The merged 'test-support' check: low-support folds are shown but left out of
 *  the summary (as in loroSummary), so they cap support at 'warn'; only a run
 *  with no scoreable fold at all fails it. */
function mergedSupportCheck(folds, label) {
  const K = folds.length;
  const low = folds.filter((f) => f.certificate.lowSupport);
  const kept = K - low.length;
  const counts = low.map((f) => `${f.heldOut}: ${plural(f.certificate.testPositives, "attack window")}${f.certificate.testPositives === 0 ? ", AP undefined" : ""}`);
  let severity;
  let detail;
  if (kept === 0) {
    severity = "fail";
    detail = `No fold has at least ${MIN_SUPPORT_FOR_RATES} attack windows in its test set (${counts.join("; ")}), so there is no summary to report. This is missing support, not a leak.`;
  } else if (low.length) {
    severity = "warn";
    const one = low.length === 1;
    detail = `${joinList(low.map((f) => f.heldOut))} ${one ? "has" : "have"} fewer than ${MIN_SUPPORT_FOR_RATES} attack windows in ${one ? "its" : "their"} test set (${counts.join("; ")}), so ${one ? "that fold is" : "those folds are"} shown but left out of the summary: ${kept} of ${K} folds are summarised. This is missing support, not a leak.`;
  } else {
    severity = "ok";
    const fewest = folds.reduce((a, f) => (f.certificate.testPositives < a.certificate.testPositives ? f : a), folds[0]);
    detail = `Every fold has at least ${MIN_SUPPORT_FOR_RATES} attack windows in its test set (fewest: ${fmtInt(fewest.certificate.testPositives)}, when ${fewest.heldOut} is held out).`;
  }
  return { id: "test-support", label, group: "support", pass: severity !== "fail", severity, detail, flaggedFolds: low.map((f) => f.heldOut) };
}

/**
 * mergeCertificates(folds) -> one certificate for a leave-one-replica-out run.
 * folds: [{heldOut, certificate}] (a {heldOut, cert} pair from a preview works too).
 * Leakage checks: the worst fold's severity, with the flagged folds named; a check
 * that is ok in every fold gets an aggregate sentence (never one fold's numbers).
 * 'test-support' is merged by mergedSupportCheck. overall = worst leakage
 * severity across folds; support = the merged support severity. Counts
 * (stretchOverlap, attackEpisodeOverlap, testRowsDuplicatingTrain,
 * quietDuplicates, activeDuplicates, testPositives, nTest) are summed over folds.
 * Also: excludedLowSupport (fold ids left out of the summary), lowSupport (no
 * scoreable fold at all), perFold [{heldOut, overall, support, lowSupport,
 * testPositives}].
 */
export function mergeCertificates(folds) {
  const fs = folds.map((f) => ({ heldOut: f.heldOut, certificate: f.certificate ?? f.cert }));
  const K = fs.length;
  const certs = fs.map((f) => f.certificate);
  const byId = new Map();
  for (const f of fs) {
    for (const ch of f.certificate.checks) {
      let m = byId.get(ch.id);
      if (!m) byId.set(ch.id, (m = { id: ch.id, label: ch.label, group: ch.group ?? "leakage", worst: ch, worstFold: f, flagged: [] }));
      if (ch.severity !== "ok") m.flagged.push(f.heldOut);
      if (worstSeverity([ch.severity, m.worst.severity]) !== m.worst.severity) {
        m.worst = ch;
        m.worstFold = f;
      }
    }
  }
  const checks = [...byId.values()].map((m) => {
    if (m.id === "test-support") return mergedSupportCheck(fs, m.label);
    return {
      id: m.id,
      label: m.label,
      group: m.group,
      pass: m.worst.severity !== "fail",
      severity: m.worst.severity,
      detail: m.flagged.length
        ? `${m.flagged.length} of ${K} folds flagged (${m.flagged.join(", ")}). Worst, fold ${m.worstFold.heldOut}: ${m.worst.detail}`
        : allFoldsOkDetail(m.id, certs, K),
      flaggedFolds: m.flagged,
    };
  });
  const union = new Set();
  const sum = (k) => certs.reduce((a, c) => a + (c[k] ?? 0), 0);
  for (const c of certs) c.replicaOverlap.forEach((r) => union.add(r));
  const excludedLowSupport = fs.filter((f) => f.certificate.lowSupport).map((f) => f.heldOut);
  return {
    checks,
    replicaOverlap: [...union],
    stretchOverlap: sum("stretchOverlap"),
    attackEpisodeOverlap: sum("attackEpisodeOverlap"),
    testRowsDuplicatingTrain: sum("testRowsDuplicatingTrain"),
    quietDuplicates: sum("quietDuplicates"),
    activeDuplicates: sum("activeDuplicates"),
    nTest: sum("nTest"),
    featureSet: certs[0]?.featureSet ?? "all",
    testPositives: sum("testPositives"),
    lowSupport: excludedLowSupport.length === K,
    excludedLowSupport,
    overall: worstSeverity(checks.filter((c) => c.group !== "support").map((c) => c.severity)),
    support: worstSeverity(checks.filter((c) => c.group === "support").map((c) => c.severity)),
    perFold: fs.map((f) => ({
      heldOut: f.heldOut,
      overall: f.certificate.overall,
      support: f.certificate.support,
      lowSupport: f.certificate.lowSupport,
      testPositives: f.certificate.testPositives,
    })),
  };
}

function subsetBy(labels, key, y, s) {
  const idx = [];
  for (let i = 0; i < labels.length; i++) if (labels[i] === key) idx.push(i);
  const ys = new Uint8Array(idx.length);
  const ss = new Float64Array(idx.length);
  let nPos = 0;
  idx.forEach((i, r) => {
    ys[r] = y[i];
    ss[r] = s[i];
    nPos += y[i];
  });
  return { ys, ss, n: idx.length, nPositive: nPos };
}

/**
 * runPipeline(config, {stats, pretrainedWeights, onEvent, shouldCancel, runId, sha256})
 * -> Promise<result>. See the data contract (6) in the project brief; deviations
 * and additions are documented next to the result object below.
 */
export async function runPipeline(
  config,
  { stats, pretrainedWeights = null, onEvent = null, shouldCancel = null, runId = null, sha256 = null } = {},
) {
  const t0 = nowMs();
  const id = runId ?? `run-${Date.now().toString(36)}`;
  const emit = (ev) => {
    if (onEvent) onEvent({ ...ev, runId: id, tMs: Math.round(nowMs() - t0) });
  };
  const checkCancel = () => {
    if (shouldCancel && shouldCancel()) throw new Cancelled();
  };

  const state = { stage: null, fold: 0, nFolds: 1, heldOut: null, stageT0: t0, lastProgressAt: -Infinity, overall: 0 };
  const timings = {};
  let weights = stageWeights(config?.model?.kind);
  let plan = makeProgressPlan(1, weights);
  let cfg = null;
  let isLoro = false;

  const foldExtra = () =>
    isLoro ? { foldIndex: state.fold, nFolds: state.nFolds, heldOut: state.heldOut } : {};
  // Stage progress as the UI's stepper sees it: in LORO the repeated stages
  // report cumulative progress across folds, so a bar never jumps backwards.
  const stageProgressOf = (stage, p) =>
    isLoro && FOLD_STAGES.includes(stage) ? (state.fold + Math.max(0, Math.min(1, p))) / state.nFolds : p;
  const foldPrefix = () => (isLoro ? `Fold ${state.fold + 1}/${state.nFolds} (${state.heldOut}) · ` : "");

  const reportProgress = (p, force = false) => {
    const o = Math.max(state.overall, plan(state.stage, state.fold, p));
    state.overall = o;
    const t = nowMs();
    if (!force && t - state.lastProgressAt < PROGRESS_INTERVAL_MS) return;
    state.lastProgressAt = t;
    const elapsed = t - t0;
    emit({
      type: "progress",
      overall: o,
      stage: state.stage,
      stageProgress: stageProgressOf(state.stage, p),
      etaMs: o > 0.05 && o < 1 ? Math.round((elapsed * (1 - o)) / o) : o >= 1 ? 0 : null,
      ...foldExtra(),
    });
  };
  const stageEvent = (stage, status, p, detail) =>
    emit({
      type: "stage",
      stage,
      status,
      progress: stageProgressOf(stage, p),
      detail,
      elapsedMs: Math.round(nowMs() - state.stageT0),
      weights,
      ...foldExtra(),
    });
  const stageStart = (stage, detail) => {
    state.stage = stage;
    state.stageT0 = nowMs();
    stageEvent(stage, "running", 0, detail);
    reportProgress(0);
  };
  const stageUpdate = (p, detail) => {
    stageEvent(state.stage, "running", p, detail);
  };
  const stageDone = (detail, status = "done") => {
    const stage = state.stage;
    timings[stage] = Math.round((timings[stage] ?? 0) + (nowMs() - state.stageT0));
    stageEvent(stage, status, 1, detail);
    reportProgress(1, true);
  };

  let corpus = null;
  let corpusSum = null;
  const folds = [];
  const modelInfo = { kind: null, nParams: null, weights: null, history: null, trainMs: 0 };
  let notes = [];

  try {
    if (!stats || !Array.isArray(stats.replicas)) throw new Error("The replica statistics (replica_stats.json) were not provided.");
    const norm = normaliseConfig(config, stats);
    if (norm.errors.length) throw new Error(`These settings cannot run: ${norm.errors.join(" ")}`);
    cfg = norm.config;
    notes = norm.warnings.slice();
    const kind = cfg.model.kind;
    const features = cfg.features;
    modelInfo.kind = kind;
    modelInfo.features = features;
    if (kind === "knn") modelInfo.k = cfg.model.k;
    weights = stageWeights(kind);
    if (kind === "pretrained" && !pretrainedWeights) {
      throw new Error(
        "The pretrained model's weights (weights.json) could not be loaded, so the pretrained model cannot run. TESSERA-base, logistic regression and the memoriser do not need them.",
      );
    }
    isLoro = cfg.split.mode === "loro";
    const allIds = stats.replicas.map((r) => r.id);
    const genIds = cfg.dataset.replicas == null ? allIds : allIds.filter((r) => cfg.dataset.replicas.includes(r));
    const foldConfigs = isLoro ? loroFolds(cfg.split, genIds) : [cfg.split];
    const K = foldConfigs.length;
    state.nFolds = K;
    plan = makeProgressPlan(K, weights);
    emit({ type: "plan", weights: { ...weights }, nFolds: K, modelKind: kind, features, notes: notes.slice() });

    // ---------------------------------------------------------------- generate
    stageStart("generate", `Sampling synthetic windows from real-data summary statistics (scale ${+(cfg.dataset.scale * 100).toFixed(2)}%)`);
    await tick();
    checkCancel();
    corpus = generateCorpus(stats, {
      scale: cfg.dataset.scale,
      seed: cfg.dataset.seed,
      replicas: genIds,
      ...(cfg.dataset.dependence ? { dependence: cfg.dataset.dependence } : {}),
    });
    corpusSum = corpusSummary(corpus);
    emit({ type: "corpus", summary: corpusSum });
    stageDone(
      `${fmtInt(corpusSum.n)} synthetic windows (${fmtInt(corpusSum.nPositive)} attacks) across ${corpus.replicaIds.length} replica${corpus.replicaIds.length === 1 ? "" : "s"}`,
    );
    await tick();
    checkCancel();

    const pretrainedParams = pretrainedWeights ? countParams(pretrainedWeights) : null;
    const certOpts = kind === "pretrained" ? { pretrainedTrainReplicas: [...PRETRAINED_TRAIN_REPLICAS], featureSet: features } : { featureSet: features };
    const featWords = features === "m2" ? " · network metrics only (24 features)" : "";

    // ---------------------------------------------------------------- folds
    for (let k = 0; k < K; k++) {
      const splitCfg = foldConfigs[k];
      state.fold = k;
      state.heldOut = isLoro ? splitCfg.test[0] : null;

      // split
      stageStart("split", `${foldPrefix()}Assigning windows to train, validation and test`);
      await tick();
      checkCancel();
      const split = buildSplit(corpus, splitCfg);
      const sn = split.summary.n;
      stageDone(`${foldPrefix()}Train ${fmtInt(sn.train)} · validation ${fmtInt(sn.val)} · test ${fmtInt(sn.test)} windows`);
      await tick();
      checkCancel();

      // leakage
      stageStart("leakage", `${foldPrefix()}Checking replicas, stretches and duplicate rows${features === "m2" ? " (on the 24 network-metric columns the model sees)" : ""}`);
      await tick();
      const certificate = leakageCertificate(corpus, split, certOpts);
      emit({ type: "split", summary: split.summary, certificate, ...foldExtra() });
      stageDone(`${foldPrefix()}${checksDetail(certificate)}`);
      await tick();
      checkCancel();

      // train
      let predictFn = null;
      let knnModel = null;
      let foldHistory = null;
      let foldTrainMs = 0;
      let foldWeights = null;
      if (kind === "pretrained") {
        state.stage = "train";
        state.stageT0 = nowMs();
        stageDone(
          `${foldPrefix()}Using the model pretrained on real AIT data (7 replicas, santos held out)${notes.length ? ". It always sees all 42 features: the network-metrics-only setting does not apply to it" : ""}`,
          "skipped",
        );
        modelInfo.nParams = pretrainedParams;
        predictFn = (rows) => predict(pretrainedWeights, rows);
      } else if (kind === "knn") {
        stageStart("train", `${foldPrefix()}Storing ${fmtInt(split.train.length)} training windows${featWords}`);
        await tick();
        checkCancel();
        const tr0 = nowMs();
        knnModel = trainKnn({ train: sliceRows(corpus, split.train), config: { k: cfg.model.k, featureSet: features } });
        foldTrainMs = nowMs() - tr0;
        modelInfo.nParams = null;
        modelInfo.nStored = knnModel.nStored;
        stageDone(
          `${foldPrefix()}Stored ${fmtInt(knnModel.nStored)} training windows (${knnModel.d} features each). Nothing to fit: the memoriser copies the labels of the ${cfg.model.k} most similar training windows when it scores the test set${split.val.length ? " (it needs no validation windows)" : ""}`,
        );
      } else {
        stageStart("train", `${foldPrefix()}Preparing ${fmtInt(split.train.length)} training windows${featWords}`);
        await tick();
        checkCancel();
        let train = sliceRows(corpus, split.train);
        let val = split.val.length ? sliceRows(corpus, split.val) : null;
        if (kind === "tessera") {
          if (features === "m2") {
            maskRowsToM2(train);
            if (val) maskRowsToM2(val);
          }
          const epochs = cfg.model.epochs;
          const res = await trainTesseraBase({
            train,
            val,
            config: {
              epochs,
              lr: cfg.model.lr,
              weightDecay: cfg.model.weightDecay,
              batchSize: cfg.model.batchSize,
              patience: cfg.model.patience,
              seed: cfg.model.seed,
              dropout: cfg.model.dropout,
              clipNorm: cfg.model.clipNorm,
            },
            onProgress: (ev) => {
              reportProgress(ev.fraction);
              if (ev.phase === "epoch") {
                emit({ type: "epoch", epoch: ev.epoch + 1, epochs, trainLoss: ev.loss, valAp: ev.valAp, valLoss: ev.valLoss ?? null, lr: ev.lr, ...foldExtra() });
                stageUpdate(
                  ev.fraction,
                  `${foldPrefix()}Epoch ${ev.epoch + 1}/${epochs} · loss ${fmt3(ev.loss)}${ev.valAp != null ? ` · validation AP ${fmt3(ev.valAp)}` : ev.valLoss != null ? ` · validation loss ${fmt3(ev.valLoss)} (validation AP undefined)` : ""}`,
                );
              }
            },
            shouldCancel,
          });
          if (res.cancelled) throw new Cancelled();
          foldHistory = res.history;
          foldTrainMs = res.elapsedMs;
          foldWeights = res.weights;
          modelInfo.nParams = res.nParams;
          const h = res.history;
          stageDone(
            `${foldPrefix()}${h.epochsRun} epoch${h.epochsRun === 1 ? "" : "s"}${h.stoppedEarly ? " (stopped early)" : ""} · kept epoch ${h.bestEpoch + 1}${earlyStopText(h)}`,
          );
          predictFn = (rows) => predict(foldWeights, rows);
        } else {
          if (features !== "all") {
            train = projectRows(train, features);
            if (val) val = projectRows(val, features);
          }
          const lc = { epochs: 200, lr: 0.05, l2: 1e-3, seed: cfg.model.seed, ...(cfg.model.logreg || {}) };
          const tr0 = nowMs();
          const m = await trainLogReg({
            train,
            val,
            config: lc,
            onProgress: (ev) => {
              reportProgress(ev.fraction);
              emit({ type: "epoch", epoch: ev.epoch + 1, epochs: lc.epochs, trainLoss: ev.loss, valAp: ev.valAp, valLoss: ev.valLoss ?? null, lr: ev.lr, ...foldExtra() });
              if ((ev.epoch + 1) % 20 === 0 || ev.epoch + 1 === lc.epochs) {
                stageUpdate(
                  ev.fraction,
                  `${foldPrefix()}Step ${ev.epoch + 1}/${lc.epochs} · loss ${fmt3(ev.loss)}${ev.valAp != null ? ` · validation AP ${fmt3(ev.valAp)}` : ev.valLoss != null ? ` · validation loss ${fmt3(ev.valLoss)} (validation AP undefined)` : ""}`,
                );
              }
            },
            shouldCancel,
          });
          if (m.cancelled) throw new Cancelled();
          foldTrainMs = nowMs() - tr0;
          foldHistory = {
            trainLoss: m.history.trainLoss,
            valAp: m.history.valAp,
            valLoss: m.history.valLoss,
            lr: m.history.trainLoss.map(() => lc.lr),
            bestEpoch: m.history.bestEpoch,
            stoppedEarly: false,
            epochsRun: m.history.epochsRun,
            earlyStopMetric: m.history.earlyStopMetric,
          };
          modelInfo.nParams = m.nParams;
          stageDone(`${foldPrefix()}${m.history.epochsRun} full-batch steps · kept step ${m.history.bestEpoch + 1}${earlyStopText(foldHistory, "step")}`);
          predictFn = features !== "all" ? (rows) => ({ scores: predictLogReg(m, projectRows(rows, features)), attribution: null }) : (rows) => ({ scores: predictLogReg(m, rows), attribution: null });
        }
      }
      modelInfo.trainMs += foldTrainMs;
      await tick();
      checkCancel();

      // test
      stageStart("test", `${foldPrefix()}Scoring ${fmtInt(split.test.length)} test windows`);
      await tick();
      const test = sliceRows(corpus, split.test);
      // what the model sees (TESSERA-base with 'm2': the masked copy); the
      // pretrained fidelity pass always sees the full windows
      const modelTest = kind === "tessera" && features === "m2" ? maskRowsToM2({ ...test, X: test.X.slice(), avail: test.avail.slice() }) : test;
      const withFidelity = !isLoro && !!pretrainedWeights && kind !== "pretrained";
      const nT = test.n;
      const totalWork = Math.max(1, nT * (withFidelity ? 2 : 1));
      const scores = new Float32Array(nT);
      let attribution = null;
      let pretrainedScores = null;
      let done = 0;
      const scoreInto = async (fn, outScores, wantAttr, rows = test) => {
        let attr = null;
        for (let o = 0; o < nT; o += SCORE_CHUNK) {
          checkCancel();
          const e = Math.min(nT, o + SCORE_CHUNK);
          const res = fn({ X: rows.X.subarray(o * 42, e * 42), avail: rows.avail.subarray(o * 4, e * 4), n: e - o });
          outScores.set(res.scores, o);
          if (wantAttr && res.attribution) {
            if (!attr) attr = new Float32Array(nT * 4);
            attr.set(res.attribution, o * 4);
          }
          done += e - o;
          reportProgress(done / totalWork);
          await tick();
        }
        return attr;
      };
      if (kind === "knn") {
        let lastDetailAt = -Infinity;
        const s = await predictKnn(knnModel, test, {
          shouldCancel,
          onProgress: (ev) => {
            reportProgress(ev.done / totalWork);
            const t = nowMs();
            if (t - lastDetailAt >= 250 && ev.done < ev.total) {
              lastDetailAt = t;
              stageUpdate(
                ev.done / totalWork,
                `${foldPrefix()}Finding the ${cfg.model.k} nearest training windows: ${fmtInt(ev.done)} of ${fmtInt(nT)} test windows`,
              );
            }
          },
        });
        if (!s) throw new Cancelled();
        scores.set(s);
        done = nT;
        await tick();
      } else {
        attribution = await scoreInto(predictFn, scores, true, modelTest);
      }
      if (withFidelity) {
        pretrainedScores = new Float32Array(nT);
        await scoreInto((rows) => predict(pretrainedWeights, rows), pretrainedScores, false);
      }
      stageDone(
        `${foldPrefix()}Scored ${fmtInt(nT)} test windows${kind === "knn" ? ` by their ${cfg.model.k} nearest training windows` : ""}${withFidelity ? " (and with the real pretrained model on the full windows, for fidelity)" : ""}`,
      );
      await tick();
      checkCancel();

      // evaluate
      stageStart("evaluate", `${foldPrefix()}Average precision, curves and calibration`);
      await tick();
      const metrics = evaluateAll(test.y, scores, { threshold: cfg.threshold });
      let fidelityAp = null;
      if (withFidelity) fidelityAp = averagePrecision(test.y, pretrainedScores);
      else if (!isLoro && kind === "pretrained") fidelityAp = metrics.ap;
      const fold = {
        foldIndex: k,
        heldOut: state.heldOut,
        testIdx: split.test,
        test,
        scores,
        attribution,
        metrics,
        certificate,
        splitSummary: split.summary,
        history: foldHistory,
        weights: foldWeights,
        trainMs: foldTrainMs,
        fidelityAp,
        fidelityAvailable: withFidelity || (!isLoro && kind === "pretrained"),
      };
      folds.push(fold);
      if (isLoro) {
        emit({
          type: "fold",
          foldIndex: k,
          nFolds: K,
          heldOut: state.heldOut,
          n: metrics.n,
          nPositive: metrics.nPositive,
          lowSupport: metrics.nPositive < MIN_SUPPORT_FOR_RATES,
          ap: metrics.ap,
          rocAuc: metrics.rocAuc,
          mcc: metrics.atThreshold.mcc,
        });
      }
      stageDone(
        `${foldPrefix()}AP ${fmt3(metrics.ap)} · MCC ${fmt3(metrics.atThreshold.mcc)} on ${fmtInt(metrics.nPositive)} attack windows${metrics.nPositive < MIN_SUPPORT_FOR_RATES ? " (low support)" : ""}`,
      );
      await tick();
      checkCancel();
    }

    // ---------------------------------------------------------------- pool
    // result.test keeps every pooled test row (with its fold and whether it is
    // counted); result.metrics covers the counted rows only. In LORO the
    // low-support folds (< MIN_SUPPORT_FOR_RATES test positives) are left out of
    // the metrics exactly as loroSummary leaves them out of the mean.
    let pooled;
    let metricsScope;
    if (!isLoro) {
      const f = folds[0];
      pooled = {
        test: f.test,
        testIdx: f.testIdx,
        scores: f.scores,
        attribution: f.attribution,
        fold: new Uint8Array(f.test.n),
        included: new Uint8Array(f.test.n).fill(1),
        metrics: f.metrics,
      };
      metricsScope = { kind: "single-split" };
    } else {
      let n = 0;
      for (const f of folds) n += f.test.n;
      const y = new Uint8Array(n);
      const replica = new Uint8Array(n);
      const host = new Uint8Array(n);
      const scores = new Float32Array(n);
      const testIdx = new Int32Array(n);
      const foldOf = new Uint8Array(n);
      const included = new Uint8Array(n);
      const hasAttr = folds.every((f) => f.attribution);
      const attribution = hasAttr ? new Float32Array(n * 4) : null;
      const excluded = [];
      let nIncl = 0;
      let o = 0;
      for (const f of folds) {
        const keep = f.metrics.nPositive >= MIN_SUPPORT_FOR_RATES;
        if (keep) nIncl += f.test.n;
        else excluded.push(f.heldOut);
        y.set(f.test.y, o);
        replica.set(f.test.replica, o);
        host.set(f.test.host, o);
        scores.set(f.scores, o);
        testIdx.set(f.testIdx, o);
        foldOf.fill(f.foldIndex, o, o + f.test.n);
        if (keep) included.fill(1, o, o + f.test.n);
        if (hasAttr) attribution.set(f.attribution, o * 4);
        o += f.test.n;
      }
      const yIn = new Uint8Array(nIncl);
      const sIn = new Float32Array(nIncl);
      for (let i = 0, r = 0; i < n; i++) {
        if (!included[i]) continue;
        yIn[r] = y[i];
        sIn[r] = scores[i];
        r++;
      }
      const nPooled = K - excluded.length;
      pooled = {
        test: { y, replica, host, n },
        testIdx,
        scores,
        attribution,
        fold: foldOf,
        included,
        metrics: evaluateAll(yIn, sIn, { threshold: cfg.threshold }),
      };
      metricsScope = { kind: "pooled-folds", nFolds: nPooled, nModels: nPooled, nFoldsTotal: K, excluded, n: nIncl };
    }

    const { y: ty, replica: tRep, host: tHost, n: tn } = pooled.test;
    let tPos = 0;
    for (let i = 0; i < tn; i++) tPos += ty[i];
    const perReplica = [];
    corpus.replicaIds.forEach((rid, ri) => {
      const sub = subsetBy(tRep, ri, ty, pooled.scores);
      if (!sub.n) return;
      perReplica.push({
        id: rid,
        n: sub.n,
        nPositive: sub.nPositive,
        ap: averagePrecision(sub.ys, sub.ss),
        mcc: metricsAt(sub.ys, sub.ss, cfg.threshold).mcc,
        lowSupport: sub.nPositive < MIN_SUPPORT_FOR_RATES,
      });
    });
    // per host: over the rows result.metrics counts (in LORO, the scoreable folds)
    const hostKey = isLoro ? Int16Array.from(tHost, (h, i) => (pooled.included[i] ? h : -1)) : tHost;
    const perHost = [];
    corpus.hostIds.forEach((hid, hi) => {
      const sub = subsetBy(hostKey, hi, ty, pooled.scores);
      if (!sub.n) return;
      perHost.push({ id: hid, label: corpus.hostLabels[hi], n: sub.n, nPositive: sub.nPositive, ap: averagePrecision(sub.ys, sub.ss) });
    });
    let attributionMean = null;
    if (pooled.attribution && tn > 0) {
      attributionMean = [0, 0, 0, 0];
      for (let i = 0; i < tn; i++) for (let m = 0; m < 4; m++) attributionMean[m] += pooled.attribution[i * 4 + m];
      attributionMean = attributionMean.map((v) => v / tn);
    }

    let fidelity = null;
    if (!isLoro && folds[0].fidelityAvailable) {
      const testIds = cfg.split.test ?? cfg.split.replicas ?? [];
      const seen = testIds.filter((r) => PRETRAINED_TRAIN_REPLICAS.includes(r));
      fidelity = {
        pretrainedAp: folds[0].fidelityAp,
        note:
          "Average precision of the real pretrained model (trained on real AIT windows from every replica except santos) on these synthetic test windows. " +
          "High means the real model separates synthetic attacks from synthetic benign windows the way it does on real data: a necessary check on the generator, not proof of realism and not a result on real data." +
          (seen.length
            ? ` Caveat: the pretrained model saw the real ${seen.join(", ")} data these synthetic windows are calibrated from, so this is optimistic for ${seen.length === 1 ? "that replica" : "those replicas"}.`
            : ""),
      };
    }

    let loroSummary = null;
    let foldsOut = null;
    if (isLoro) {
      foldsOut = folds.map((f) => ({
        foldIndex: f.foldIndex,
        heldOut: f.heldOut,
        n: f.metrics.n,
        nPositive: f.metrics.nPositive,
        lowSupport: f.metrics.nPositive < MIN_SUPPORT_FOR_RATES,
        ap: f.metrics.ap,
        rocAuc: f.metrics.rocAuc,
        mcc: f.metrics.atThreshold.mcc,
        prevalence: f.metrics.prevalence,
        certificate: f.certificate,
        splitSummary: f.splitSummary,
        history: f.history,
        trainMs: Math.round(f.trainMs),
      }));
      const kept = foldsOut.filter((f) => !f.lowSupport);
      loroSummary = {
        ap: summarise(kept.map((f) => f.ap)),
        mcc: summarise(kept.map((f) => f.mcc)),
        excludedLowSupport: foldsOut.filter((f) => f.lowSupport).map((f) => f.heldOut),
        naiveAp: summarise(foldsOut.map((f) => f.ap)),
        minSupport: MIN_SUPPORT_FOR_RATES,
      };
    }

    // ---------------------------------------------------------------- ledger
    let ledger = null;
    let ledgerStatus = "off";
    const digest =
      sha256 ||
      (globalThis.crypto && globalThis.crypto.subtle
        ? async (bytes) => new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", bytes))
        : null);
    if (cfg.ledger && !digest) {
      // An insecure page (http:// from a LAN address) has no crypto.subtle. The
      // ledger is an add-on: skip it and keep the run's results.
      ledgerStatus = "skipped-no-crypto";
      state.stage = "ledger";
      state.stageT0 = nowMs();
      stageDone(LEDGER_SKIPPED_NO_CRYPTO, "skipped");
    } else if (cfg.ledger) {
      ledgerStatus = "done";
      stageStart("ledger", "Committing one verdict per test window to a Merkle log");
      await tick();
      setSha256(digest);
      const enc = new TextEncoder();
      const hostHash = [];
      for (const hid of corpus.hostIds) {
        hostHash.push(bytesToHex(await digest(enc.encode(`tessera-lab/synthetic-host/${hid}`))).slice(0, 16));
      }
      const windowSeconds = stats.window_seconds ?? 60;
      const nLeaves = Math.min(tn, LEDGER_CAP);
      const log = new MerkleLog();
      for (let r = 0; r < nLeaves; r++) {
        const ci = pooled.testIdx[r];
        const s = pooled.scores[r];
        await log.appendJson({
          window_id: `syn-${corpus.replicaIds[corpus.replica[ci]]}-${corpus.hostIds[corpus.host[ci]]}-${corpus.t[ci]}`,
          host_hash: hostHash[corpus.host[ci]],
          ts_bucket: corpus.t[ci] * windowSeconds,
          verdict: s >= cfg.threshold ? 1 : 0,
          score: s.toFixed(6),
          model_git_sha: "lab",
        });
        if ((r + 1) % 256 === 0) {
          checkCancel();
          reportProgress((0.8 * (r + 1)) / Math.max(1, nLeaves));
          await tick();
        }
      }
      const rootHex = bytesToHex(await log.root());
      ledger = { rootHex, nLeaves, capped: tn > LEDGER_CAP, nTest: tn };
      stageDone(
        `Committed ${fmtInt(nLeaves)} verdicts${tn > LEDGER_CAP ? ` (capped at ${fmtInt(LEDGER_CAP)} of ${fmtInt(tn)} test windows)` : ""} · root ${rootHex.slice(0, 12)}…`,
      );
    } else {
      state.stage = "ledger";
      state.stageT0 = nowMs();
      stageDone("Ledger turned off for this run", "skipped");
    }

    let certificate;
    let splitSummary;
    if (isLoro) {
      certificate = mergeCertificates(folds);
      splitSummary = { mode: "loro", nFolds: folds.length, heldOut: folds.map((f) => f.heldOut), perFold: folds.map((f) => f.splitSummary) };
    } else {
      certificate = folds[0].certificate;
      splitSummary = folds[0].splitSummary;
    }

    if (!isLoro && kind !== "pretrained") {
      modelInfo.weights = folds[0].weights; // null for logreg
      modelInfo.history = folds[0].history;
    }
    modelInfo.trainMs = Math.round(modelInfo.trainMs);

    // Additions to the contract:
    //   test.fold      Uint8Array: fold index of each pooled test row (0 outside LORO)
    //   test.included  Uint8Array: 1 = the row is counted in `metrics`
    //   metricsScope   {kind: 'single-split'} or {kind: 'pooled-folds', nFolds,
    //                  nModels, nFoldsTotal, excluded: [held-out ids], n}
    //   perHost        over the rows `metrics` counts; perReplica over every row
    //   ledgerStatus   'done' | 'off' | 'skipped-no-crypto' (ledger is null unless 'done')
    const result = {
      runId: id,
      config: cfg,
      mode: cfg.split.mode,
      features: cfg.features,
      notes,
      synthetic: true,
      model: modelInfo,
      corpus: corpusSum,
      test: {
        n: tn,
        nPositive: tPos,
        scores: pooled.scores,
        y: ty,
        replica: tRep,
        host: tHost,
        fold: pooled.fold,
        included: pooled.included,
        attribution: pooled.attribution,
        replicaIds: corpus.replicaIds.slice(),
        hostIds: corpus.hostIds.slice(),
        hostLabels: corpus.hostLabels.slice(),
      },
      metrics: pooled.metrics,
      metricsScope,
      perReplica,
      perHost,
      attributionMean,
      fidelity,
      certificate,
      splitSummary,
      ledger,
      ledgerStatus,
      folds: foldsOut,
      loroSummary,
      timings,
      elapsedMs: Math.round(nowMs() - t0),
      cancelled: false,
    };
    state.overall = 1;
    emit({ type: "progress", overall: 1, stage: "ledger", stageProgress: 1, etaMs: 0 });
    emit({ type: "result", result });
    return result;
  } catch (err) {
    if (err instanceof Cancelled) {
      // Close out the current stage and everything after it, then report a
      // cancelled (partial) result. No 'error' event: cancelling is not a failure.
      const from = Math.max(0, STAGE_IDS.indexOf(state.stage ?? "generate"));
      for (let i = from; i < STAGE_IDS.length; i++) {
        emit({
          type: "stage",
          stage: STAGE_IDS[i],
          status: "skipped",
          progress: i === from ? stageProgressOf(STAGE_IDS[i], 0) : 0,
          detail: i === from ? "Cancelled" : "Not run (cancelled)",
          elapsedMs: i === from ? Math.round(nowMs() - state.stageT0) : 0,
          weights,
          cancelled: true,
          ...(i === from ? foldExtra() : {}),
        });
      }
      const result = {
        runId: id,
        config: cfg ?? config,
        mode: cfg?.split?.mode ?? config?.split?.mode ?? null,
        features: cfg?.features ?? config?.features ?? "all",
        notes,
        synthetic: true,
        model: { ...modelInfo, weights: null, trainMs: Math.round(modelInfo.trainMs) },
        corpus: corpusSum,
        test: null,
        metrics: null,
        metricsScope: null,
        perReplica: null,
        perHost: null,
        attributionMean: null,
        fidelity: null,
        certificate: null,
        splitSummary: null,
        ledger: null,
        ledgerStatus: null,
        folds: isLoro
          ? folds.map((f) => ({
              foldIndex: f.foldIndex,
              heldOut: f.heldOut,
              n: f.metrics.n,
              nPositive: f.metrics.nPositive,
              lowSupport: f.metrics.nPositive < MIN_SUPPORT_FOR_RATES,
              ap: f.metrics.ap,
              rocAuc: f.metrics.rocAuc,
              mcc: f.metrics.atThreshold.mcc,
            }))
          : null,
        loroSummary: null,
        timings,
        elapsedMs: Math.round(nowMs() - t0),
        cancelled: true,
        cancelledAt: state.stage,
      };
      emit({ type: "result", result });
      return result;
    }
    const message = err && err.message ? err.message : String(err);
    if (state.stage) {
      emit({ type: "stage", stage: state.stage, status: "error", progress: stageProgressOf(state.stage, 0), detail: message, elapsedMs: Math.round(nowMs() - state.stageT0), weights, ...foldExtra() });
    }
    emit({ type: "error", message, stage: state.stage });
    const e = new Error(message);
    e.pipelineReported = true;
    throw e;
  }
}
