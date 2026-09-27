// Train / validation / test splits for the Training Lab, and the leakage
// certificate that says - in plain language - whether a split lets the test set
// see what training saw.
//
// The four protocols mirror the project's evaluation regimes:
//   'replica'       hold out whole replicas (testbeds). The honest default:
//                   test on santos, train on the other 7, validation = a seeded
//                   random valFraction of the training rows (as the Python code
//                   does: permute, take round(n * valFraction)).
//   'random'        row-level shuffle across the chosen replicas - the LEAKY
//                   protocol most published numbers use (neighbouring windows of
//                   one attack land on both sides of the split).
//   'chronological' per replica + host timeline, the first trainPct% of windows
//                   train, the next valPct% validate, the rest test. Each
//                   boundary is then purged, like the real
//                   tessera.eval.splits.r1_chronological drops the windows within
//                   a gap after each cut: every validation / test window whose
//                   stretch (attack episode or benign block) also occurs in an
//                   earlier partition moves to 'unused' (counted in
//                   summary.purged), so no stretch straddles a boundary.
//   'loro'          leave-one-replica-out: a meta-mode; loroFolds() expands it
//                   into one 'replica' split per held-out replica.
//
// Everything is pure and deterministic: randomness comes only from
// makeRng(seed, 'split').

import { makeRng } from "./rng.js";
import { MIN_SUPPORT_FOR_RATES } from "./metrics.js";

/** The replicas in the project's canonical order (tessera.data.ait.unpack.REPLICAS). */
export const REPLICA_ORDER = Object.freeze([
  "russellmitchell",
  "santos",
  "harrison",
  "fox",
  "wheeler",
  "wardbeck",
  "wilson",
  "shaw",
]);

/** The replicas the pretrained real model (web/data/weights.json) was trained on:
 *  every replica except santos. */
export const PRETRAINED_TRAIN_REPLICAS = Object.freeze(REPLICA_ORDER.filter((r) => r !== "santos"));

export const DEFAULT_SPLIT = Object.freeze({
  mode: "replica",
  train: Object.freeze(REPLICA_ORDER.filter((r) => r !== "santos")),
  test: Object.freeze(["santos"]),
  valFraction: 0.15,
  seed: 0,
});

const MODES = ["replica", "random", "chronological", "loro"];

/** A mutable deep copy of a split config (DEFAULT_SPLIT is frozen). */
export function cloneSplitConfig(config = DEFAULT_SPLIT) {
  const out = { ...config };
  for (const k of ["train", "test", "replicas"]) if (Array.isArray(config[k])) out[k] = [...config[k]];
  return out;
}

// ------------------------------------------------------------------ validation

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

function checkIds(list, field, known, errors, { label }) {
  if (!Array.isArray(list) || list.length === 0) {
    errors.push(`Choose at least one replica for ${label}.`);
    return [];
  }
  const seen = new Set();
  for (const id of list) {
    if (!known.has(id)) errors.push(`"${id}" (in ${label}) is not a replica in this dataset.`);
    if (seen.has(id)) errors.push(`"${id}" is listed twice in ${label}.`);
    seen.add(id);
  }
  return list;
}

function checkValFraction(config, errors) {
  const v = config.valFraction ?? 0.15;
  if (!isNum(v) || v < 0.05 || v > 0.4) {
    errors.push(`The validation share must be between 5% and 40% of the training rows (got ${fmtPct(v)}).`);
  }
}

function checkSeed(config, errors) {
  if (config.seed !== undefined && !Number.isInteger(config.seed)) {
    errors.push(`The seed must be a whole number (got ${config.seed}).`);
  }
}

function fmtPct(v) {
  return isNum(v) ? `${+(v * 100).toFixed(2)}%` : String(v);
}

/**
 * validateSplitConfig(config, replicaIds) -> {ok, errors}
 * errors are plain-language sentences suitable for showing under the form.
 */
export function validateSplitConfig(config, replicaIds = REPLICA_ORDER) {
  const errors = [];
  if (!config || typeof config !== "object") {
    return { ok: false, errors: ["No split settings were given."] };
  }
  const known = new Set(replicaIds);
  const mode = config.mode;
  if (!MODES.includes(mode)) {
    return {
      ok: false,
      errors: [`Unknown split type "${mode}". Choose one of: by replica, random, chronological, leave-one-replica-out.`],
    };
  }

  if (mode === "replica") {
    const train = checkIds(config.train, "train", known, errors, { label: "training" });
    const test = checkIds(config.test, "test", known, errors, { label: "testing" });
    const both = train.filter((id) => test.includes(id));
    if (both.length) {
      errors.push(
        `${both.join(", ")} ${both.length === 1 ? "is" : "are"} in both training and testing - a replica can only be on one side.`,
      );
    }
    checkValFraction(config, errors);
    checkSeed(config, errors);
  } else if (mode === "random" || mode === "chronological") {
    checkIds(config.replicas, "replicas", known, errors, { label: "this split" });
    const pcts = { trainPct: "training", valPct: "validation", testPct: "test" };
    let allNum = true;
    for (const [k, label] of Object.entries(pcts)) {
      const v = config[k];
      if (!isNum(v)) {
        errors.push(`The ${label} percentage is missing or not a number.`);
        allNum = false;
      } else if (v < 0) {
        errors.push(`The ${label} percentage cannot be negative (got ${v}%).`);
      }
    }
    if (allNum) {
      const sum = config.trainPct + config.valPct + config.testPct;
      if (Math.abs(sum - 100) > 1e-9) {
        errors.push(`Training, validation and test must add up to 100% (they add up to ${+sum.toFixed(4)}%).`);
      }
      if (config.testPct < 5) errors.push(`Keep at least 5% for testing (got ${config.testPct}%).`);
      if (config.trainPct < 20) errors.push(`Keep at least 20% for training (got ${config.trainPct}%).`);
    }
    if (mode === "random") checkSeed(config, errors);
  } else if (mode === "loro") {
    const reps = checkIds(config.replicas, "replicas", known, errors, { label: "leave-one-replica-out" });
    if (reps.length === 1) {
      errors.push("Leave-one-replica-out needs at least 2 replicas (one to hold out, at least one to train on).");
    }
    checkValFraction(config, errors);
    checkSeed(config, errors);
  }
  return { ok: errors.length === 0, errors };
}

// ------------------------------------------------------------------ building

function sortedInt32(list) {
  const a = Int32Array.from(list);
  a.sort();
  return a;
}

function rowsByReplica(corpus) {
  const k = corpus.replicaIds.length;
  const buckets = Array.from({ length: k }, () => []);
  for (let i = 0; i < corpus.n; i++) buckets[corpus.replica[i]].push(i);
  return buckets;
}

const PARTS = ["train", "val", "test", "unused"];

function summariseSplit(corpus, parts, mode, purged = null) {
  const n = {};
  const positives = {};
  const pct = {};
  const role = new Int8Array(corpus.n); // 0 train, 1 val, 2 test, 3 unused
  PARTS.forEach((p, code) => {
    let pos = 0;
    for (const i of parts[p]) {
      role[i] = code;
      pos += corpus.y[i];
    }
    n[p] = parts[p].length;
    positives[p] = pos;
    pct[p] = corpus.n ? (100 * parts[p].length) / corpus.n : 0;
  });
  n.total = corpus.n;
  const perReplica = {};
  corpus.replicaIds.forEach((id) => {
    perReplica[id] = { role: "unused", n: 0, nPositive: 0, parts: { train: 0, val: 0, test: 0, unused: 0 } };
  });
  for (let i = 0; i < corpus.n; i++) {
    const r = perReplica[corpus.replicaIds[corpus.replica[i]]];
    r.n++;
    r.nPositive += corpus.y[i];
    r.parts[PARTS[role[i]]]++;
  }
  for (const r of Object.values(perReplica)) {
    const trainSide = r.parts.train + r.parts.val;
    if (r.n === 0 || r.parts.unused === r.n) r.role = "unused";
    else if (trainSide === r.n) r.role = "train";
    else if (r.parts.test === r.n) r.role = "test";
    else r.role = "mixed";
  }
  return { mode, n, positives, pct, perReplica, purged };
}

/**
 * buildSplit(corpus, config) -> {train, val, test, unused (sorted Int32Array row
 * indices), summary}. Throws an Error whose message lists the problems if the
 * config is invalid. 'loro' is a meta-mode: expand it with loroFolds() first.
 * summary = {mode, n, positives, pct, perReplica, purged}; purged is null except
 * for 'chronological', where it is {val, test, positives: {val, test}}: the
 * windows (and attack windows among them) moved to 'unused' because their
 * stretch continues from an earlier partition. A chronological split whose test
 * set is emptied by the purge throws.
 */
export function buildSplit(corpus, config) {
  const { ok, errors } = validateSplitConfig(config, corpus.replicaIds);
  if (!ok) throw new Error(`Invalid split: ${errors.join(" ")}`);
  if (config.mode === "loro") {
    throw new Error("Leave-one-replica-out runs one split per replica: expand it with loroFolds(config, replicaIds).");
  }
  const byRep = rowsByReplica(corpus);
  const repIndex = new Map(corpus.replicaIds.map((id, i) => [id, i]));
  const parts = { train: [], val: [], test: [], unused: [] };
  const seed = config.seed ?? 0;
  let purged = null; // chronological only: rows moved to 'unused' at the boundaries

  if (config.mode === "replica") {
    const trainSet = new Set(config.train.map((id) => repIndex.get(id)));
    const testSet = new Set(config.test.map((id) => repIndex.get(id)));
    const pool = [];
    byRep.forEach((rows, r) => {
      const dest = trainSet.has(r) ? pool : testSet.has(r) ? parts.test : parts.unused;
      for (const i of rows) dest.push(i);
    });
    const rng = makeRng(seed, "split");
    rng.shuffle(pool);
    const nVal = Math.round(pool.length * (config.valFraction ?? 0.15));
    parts.val = pool.slice(0, nVal);
    parts.train = pool.slice(nVal);
  } else if (config.mode === "random") {
    const selected = new Set(config.replicas.map((id) => repIndex.get(id)));
    const pool = [];
    byRep.forEach((rows, r) => {
      const dest = selected.has(r) ? pool : parts.unused;
      for (const i of rows) dest.push(i);
    });
    const rng = makeRng(seed, "split");
    rng.shuffle(pool);
    const nTest = Math.round((pool.length * config.testPct) / 100);
    const nVal = Math.min(pool.length - nTest, Math.round((pool.length * config.valPct) / 100));
    parts.test = pool.slice(0, nTest);
    parts.val = pool.slice(nTest, nTest + nVal);
    parts.train = pool.slice(nTest + nVal);
  } else {
    // chronological, per replica + host timeline
    const selected = new Set(config.replicas.map((id) => repIndex.get(id)));
    const timelines = new Map();
    for (let i = 0; i < corpus.n; i++) {
      if (!selected.has(corpus.replica[i])) {
        parts.unused.push(i);
        continue;
      }
      const key = corpus.replica[i] * 256 + corpus.host[i];
      let tl = timelines.get(key);
      if (!tl) timelines.set(key, (tl = []));
      tl.push(i);
    }
    const val = [];
    const test = [];
    for (const tl of timelines.values()) {
      tl.sort((a, b) => corpus.t[a] - corpus.t[b] || a - b);
      const m = tl.length;
      const nTrain = Math.min(m, Math.round((m * config.trainPct) / 100));
      const nVal = Math.min(m - nTrain, Math.round((m * config.valPct) / 100));
      for (let k = 0; k < m; k++) {
        if (k < nTrain) parts.train.push(tl[k]);
        else if (k < nTrain + nVal) val.push(tl[k]);
        else test.push(tl[k]);
      }
    }
    // Purge each boundary (the Lab's version of r1_chronological's gap): a later
    // partition keeps no window whose stretch already occurs in an earlier one.
    // Stretch ids are unique across the corpus, so one set per boundary suffices.
    const earlier = new Set();
    for (const i of parts.train) earlier.add(corpus.stretch[i]);
    const purge = (rows, into) => {
      let dropped = 0;
      let droppedPos = 0;
      for (const i of rows) {
        if (earlier.has(corpus.stretch[i])) {
          parts.unused.push(i);
          dropped++;
          droppedPos += corpus.y[i];
        } else into.push(i);
      }
      return [dropped, droppedPos];
    };
    const [pv, pvPos] = purge(val, parts.val);
    for (const i of parts.val) earlier.add(corpus.stretch[i]);
    const [pt, ptPos] = purge(test, parts.test);
    purged = { val: pv, test: pt, positives: { val: pvPos, test: ptPos } };
    if (parts.test.length === 0) {
      throw new Error(
        `Invalid split: after removing the ${plural(pt, "test window")} that continue an attack episode or quiet period from an earlier part, no test windows are left. Give the test set a larger share, or use more data.`,
      );
    }
  }

  const out = {};
  for (const p of PARTS) out[p] = sortedInt32(parts[p]);
  out.summary = summariseSplit(corpus, out, config.mode, purged);
  return out;
}

/** loroFolds(config, replicaIds) -> one 'replica' SplitConfig per held-out
 *  replica (in replicaIds order), training on the other selected replicas. */
export function loroFolds(config, replicaIds = REPLICA_ORDER) {
  const chosen = new Set(config.replicas ?? replicaIds);
  const reps = replicaIds.filter((id) => chosen.has(id));
  return reps.map((heldOut) => ({
    mode: "replica",
    train: reps.filter((id) => id !== heldOut),
    test: [heldOut],
    valFraction: config.valFraction ?? 0.15,
    seed: config.seed ?? 0,
  }));
}

// ------------------------------------------------------------------ leakage certificate

/** [start, stop) of each modality inside the 42-vector: M1 log, M2 network
 *  metrics, M3 host identity, M4 graph (the data contract, as in trainer.js). */
const MODALITY_SLICES = Object.freeze([
  Object.freeze([0, 8]),
  Object.freeze([8, 32]),
  Object.freeze([32, 34]),
  Object.freeze([34, 42]),
]);
const [M3_START, M3_STOP] = MODALITY_SLICES[2];

/** Feature sets a Lab model can see: every column, or the network metrics (M2) only. */
export const FEATURE_SETS = Object.freeze(["all", "m2"]);

const FEATURE_COLUMNS = {
  all: Int32Array.from({ length: MODALITY_SLICES[3][1] }, (_, j) => j),
  m2: Int32Array.from({ length: MODALITY_SLICES[1][1] - MODALITY_SLICES[1][0] }, (_, j) => MODALITY_SLICES[1][0] + j),
};

/**
 * featureColumns(featureSet = 'all') -> Int32Array of the column indices a model
 * with that feature set sees: 'all' = the 42 columns 0..41, 'm2' = the 24
 * network-metric columns 8..31. The single source of truth for "which columns a
 * feature set uses"; returns a fresh copy, so callers may keep or change it.
 * Throws on an unknown feature set.
 */
export function featureColumns(featureSet = "all") {
  const cols = Object.prototype.hasOwnProperty.call(FEATURE_COLUMNS, featureSet) ? FEATURE_COLUMNS[featureSet] : null;
  if (!cols) throw new Error(`Unknown feature set "${featureSet}". Choose one of: ${FEATURE_SETS.join(", ")}.`);
  return cols.slice();
}

// Row fingerprint over the raw float32 bits of the compared columns (two
// independent 32-bit hashes combined into one safe integer); a hash hit is then
// confirmed by comparing the values, so the count is exact even if two rows ever
// collide.
function rowKey(bits, off, cols) {
  let h1 = 0x811c9dc5;
  let h2 = 0x9747b28c;
  for (let j = 0; j < cols.length; j++) {
    let v = bits[off + cols[j]];
    if (v === 0x80000000) v = 0; // -0 equals +0
    h1 = Math.imul(h1 ^ v, 0x01000193);
    h1 ^= h1 >>> 15;
    h2 = Math.imul(h2 + v, 0x5bd1e995);
    h2 ^= h2 >>> 13;
  }
  return (h1 >>> 0) * 2097152 + ((h2 >>> 0) & 0x1fffff);
}

function float32Rows(corpus) {
  const X = corpus.X instanceof Float32Array ? corpus.X : Float32Array.from(corpus.X);
  return { X, bits: new Uint32Array(X.buffer, X.byteOffset, X.length) };
}

function sameRow(X, a, b, d, cols) {
  const oa = a * d;
  const ob = b * d;
  for (let j = 0; j < cols.length; j++) if (X[oa + cols[j]] !== X[ob + cols[j]]) return false;
  return true;
}

/** Quiet = every compared column outside M3 (host identity) is zero: the row
 *  has no log, network-metric or graph activity for the model to recognise. */
function isQuiet(X, i, d, activityCols) {
  const o = i * d;
  for (let j = 0; j < activityCols.length; j++) if (X[o + activityCols[j]] !== 0) return false;
  return true;
}

/** -> {total, quiet, active, quietAttacks, activeAttacks}: test rows equal (on
 *  `cols`) to at least one training-side row, split into quiet and active. */
function countTestDuplicates(corpus, trainRows, testRows, cols) {
  const d = corpus.nFeatures ?? 42;
  const { X, bits } = float32Rows(corpus);
  const activityCols = cols.filter((j) => j < M3_START || j >= M3_STOP);
  const index = new Map(); // key -> row index, or array of row indices on collision
  for (const i of trainRows) {
    const key = rowKey(bits, i * d, cols);
    const hit = index.get(key);
    if (hit === undefined) index.set(key, i);
    else {
      const list = Array.isArray(hit) ? hit : [hit];
      if (!list.some((r) => sameRow(X, r, i, d, cols))) {
        list.push(i);
        index.set(key, list);
      }
    }
  }
  const out = { total: 0, quiet: 0, active: 0, quietAttacks: 0, activeAttacks: 0 };
  for (const i of testRows) {
    const hit = index.get(rowKey(bits, i * d, cols));
    if (hit === undefined) continue;
    const list = Array.isArray(hit) ? hit : [hit];
    if (!list.some((r) => sameRow(X, r, i, d, cols))) continue;
    out.total++;
    if (isQuiet(X, i, d, activityCols)) {
      out.quiet++;
      out.quietAttacks += corpus.y[i];
    } else {
      out.active++;
      out.activeAttacks += corpus.y[i];
    }
  }
  return out;
}

const plural = (n, one, many = `${one}s`) => `${n.toLocaleString("en")} ${n === 1 ? one : many}`;

/** k of n as a percentage with one decimal, never "0.0%" for a non-zero count
 *  ("<0.1%") nor "100.0%" for a count short of n (">99.9%"). */
export function pctOf(k, n) {
  if (!(n > 0)) return "0%";
  const p = (100 * k) / n;
  if (k > 0 && p < 0.05) return "<0.1%";
  if (k < n && p >= 99.95) return ">99.9%";
  return `${p.toFixed(1)}%`;
}

/** -> [severity, detail] for the duplicate-rows check. Active duplicates above
 *  1% of the test set fail, any active duplicate warns, quiet ones never count. */
function duplicateVerdict(d, nTest, featureSet) {
  const m2 = featureSet === "m2";
  const onCols = m2 ? "on the 24 network-metric columns the model sees" : "in all 42 features";
  const quietWhat = m2
    ? "all 24 network-metric columns are zero, so the model sees an empty row, and an empty row is the same on every testbed"
    : "no log, network-metric or graph activity, only the host identity, which is the same on every testbed";
  const quietAtt = d.quietAttacks
    ? ` ${plural(d.quietAttacks, "of them is", "of them are")} labelled as an attack, but a row with no activity holds nothing specific to memorise.`
    : "";
  if (d.total === 0) {
    return ["ok", `No test window is an exact copy of a training or validation window ${onCols}.`];
  }
  if (d.active === 0) {
    return [
      "ok",
      `${plural(d.quiet, "test window")} (${pctOf(d.quiet, nTest)} of the test set) ${d.quiet === 1 ? "equals" : "equal"} a training or validation window, but ${d.quiet === 1 ? "it is a quiet window" : "all of them are quiet windows"}: ${quietWhat}. Such windows look alike on every replica, so they reveal nothing about the held-out data and are not a leak.${quietAtt}`,
    ];
  }
  const severity = d.active / nTest > 0.01 ? "fail" : "warn";
  const quietShort = m2 ? "all metric columns zero" : "no activity, host identity only";
  const quietNote = d.quiet
    ? ` A further ${plural(d.quiet, "quiet window")} (${quietShort}) also ${d.quiet === 1 ? "matches" : "match"}; ${d.quiet === 1 ? "it is" : "those are"} identical on every testbed and not counted as a leak.`
    : "";
  return [
    severity,
    `${plural(d.active, "test window")} with real ${m2 ? "network-metric " : ""}activity (${pctOf(d.active, nTest)} of the test set) exactly ${d.active === 1 ? "equals" : "equal"} a training or validation window ${onCols}, so the model is marked on rows it was trained on.${severity === "warn" ? " Under 1%: minor, but worth knowing." : ""}${quietNote}`,
  ];
}

/** The checks that decide the LEAKAGE verdict (certificate.overall). The
 *  'test-support' check is about whether a score can be measured at all, not
 *  about leakage; it decides certificate.support instead. */
export const LEAKAGE_CHECK_IDS = Object.freeze(["replica-disjoint", "stretch-disjoint", "duplicate-rows", "pretrained-contamination"]);
export const SUPPORT_CHECK_IDS = Object.freeze(["test-support"]);

const SEVERITY_RANK = { ok: 0, warn: 1, fail: 2 };

/** The worst of a list of severities ('ok' for an empty list). */
export function worstSeverity(severities) {
  let worst = "ok";
  for (const s of severities) if (SEVERITY_RANK[s] > SEVERITY_RANK[worst]) worst = s;
  return worst;
}

function joinList(items) {
  if (items.length <= 1) return items.join("");
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/**
 * leakageCertificate(corpus, split, {pretrainedTrainReplicas, featureSet}) ->
 * {checks:[{id, label, group:'leakage'|'support', pass, severity:'ok'|'warn'|'fail', detail}],
 *  replicaOverlap, stretchOverlap, attackEpisodeOverlap, testRowsDuplicatingTrain,
 *  quietDuplicates, activeDuplicates, featureSet, nTest, testPositives, lowSupport,
 *  overall:'ok'|'warn'|'fail', support:'ok'|'warn'|'fail'}.
 *
 * Two separate verdicts:
 *   overall  the LEAKAGE verdict, the worst severity of the leakage checks only
 *            (LEAKAGE_CHECK_IDS: replica-disjoint, stretch-disjoint,
 *            duplicate-rows and, when asked, pretrained-contamination);
 *   support  whether the test set can be scored at all ('test-support'): 'fail'
 *            with no attack windows (AP undefined), 'warn' under
 *            MIN_SUPPORT_FOR_RATES positives. A test set without attacks is a
 *            support failure, never a leak.
 *
 * Severities: sharing a testbed between the sides ('replica-disjoint') is a
 * 'warn': the split then measures generalisation within networks the model has
 * seen, a narrower question than generalising to a new one, but not by itself a
 * leak. The leaks that inflate a score - a stretch cut in two, repeated active
 * windows, a pretrained model that saw the test replica - are 'fail'.
 *
 * "Training side" means training + validation rows: validation picks the
 * early-stopping epoch, so anything it shares with the test set leaks too.
 * pass is false only for severity 'fail'.
 *
 * Duplicates are judged on the columns the model sees (featureColumns(featureSet),
 * 'all' by default). A duplicated test row is "quiet" when all of those columns
 * outside M3 are zero (for 'm2': all 24 metric columns zero): with no log, metric
 * or graph activity, only the host identity is left, and that is identical on
 * every testbed by construction, so the copy reveals nothing about the held-out
 * data. Only "active" duplicates (a window with real activity repeated on both
 * sides, the mechanism behind temporal-neighbour leakage) can fail the check.
 */
export function leakageCertificate(corpus, split, { pretrainedTrainReplicas = null, featureSet = "all" } = {}) {
  const cols = featureColumns(featureSet);
  const ids = corpus.replicaIds;
  const trainSide = [...split.train, ...split.val];
  const test = split.test;

  // replicas
  const trainReps = new Set();
  const testReps = new Set();
  for (const i of trainSide) trainReps.add(corpus.replica[i]);
  for (const i of test) testReps.add(corpus.replica[i]);
  const replicaOverlap = ids.filter((_, r) => trainReps.has(r) && testReps.has(r));

  // stretches (attack episodes and benign blocks)
  const trainStretch = new Set();
  for (const i of trainSide) trainStretch.add(corpus.stretch[i]);
  const shared = new Map(); // stretch id -> is attack
  for (const i of test) {
    const st = corpus.stretch[i];
    if (trainStretch.has(st)) shared.set(st, (shared.get(st) || false) || corpus.y[i] === 1);
  }
  const stretchOverlap = shared.size;
  let attackEpisodeOverlap = 0;
  for (const isAttack of shared.values()) if (isAttack) attackEpisodeOverlap++;

  // exact duplicates
  const dups = test.length
    ? countTestDuplicates(corpus, trainSide, test, cols)
    : { total: 0, quiet: 0, active: 0, quietAttacks: 0, activeAttacks: 0 };
  const testRowsDuplicatingTrain = dups.total;
  const quietDuplicates = dups.quiet;
  const activeDuplicates = dups.active;

  // support
  let testPositives = 0;
  for (const i of test) testPositives += corpus.y[i];
  const lowSupport = testPositives < MIN_SUPPORT_FOR_RATES;

  const checks = [];
  const add = (id, label, severity, detail) =>
    checks.push({ id, label, group: SUPPORT_CHECK_IDS.includes(id) ? "support" : "leakage", pass: severity !== "fail", severity, detail });

  const who =
    replicaOverlap.length > 2 && replicaOverlap.length === ids.length
      ? `All ${replicaOverlap.length} replicas`
      : joinList(replicaOverlap);
  add(
    "replica-disjoint",
    "Test replicas are unseen",
    replicaOverlap.length ? "warn" : "ok",
    replicaOverlap.length
      ? `${who} ${replicaOverlap.length === 1 ? "has" : "have"} windows in both training and testing. With the same testbeds on both sides, this measures how well the model generalises within networks it has already seen (to later windows, or to other rows), not to a new network, so the score is not comparable with a replica hold-out. That is a narrower question, not a leak by itself: the leaks that inflate a score are cut stretches and repeated active windows, checked below.`
      : "No replica appears on both sides: the test set is a testbed the model never saw.",
  );

  const purged = split.summary?.purged;
  const nPurged = purged ? purged.val + purged.test : 0;
  add(
    "stretch-disjoint",
    "No attack or quiet period is cut in two",
    stretchOverlap ? "fail" : "ok",
    stretchOverlap
      ? `${plural(stretchOverlap, "stretch", "stretches")} (${plural(attackEpisodeOverlap, "attack episode")}) ${stretchOverlap === 1 ? "has" : "have"} neighbouring windows on both sides of the split. Adjacent minutes of the same episode look almost identical, so the model can recognise them instead of detecting the attack (temporal-neighbour leakage).`
      : `Every attack episode and every benign stretch sits entirely on one side of the split.${nPurged ? ` To get there, the chronological split set aside ${plural(nPurged, "window")} (${plural(purged.positives.val + purged.positives.test, "attack window")}) at the start of each replica + host timeline's validation and test parts, because they continued a stretch from the part before; the real r1_chronological protocol drops a fixed 600-second gap after each cut instead.` : ""}`,
  );

  add("duplicate-rows", "Test rows are not copies of training rows", ...duplicateVerdict(dups, test.length, featureSet));

  add(
    "test-support",
    "Enough attacks in the test set",
    testPositives === 0 ? "fail" : lowSupport ? "warn" : "ok",
    testPositives === 0
      ? "The test set contains no attack windows, so average precision is undefined: there is nothing to detect. This is missing support, not a leak: no AP can be reported for this test set, and in leave-one-replica-out such a fold is shown but left out of the summary."
      : lowSupport
        ? `Only ${plural(testPositives, "attack window")} in the test set (fewer than ${MIN_SUPPORT_FOR_RATES}). Scores from this few positives are noise; the result is shown but kept out of summary statistics. This is a support limit, not a leak.`
        : `${plural(testPositives, "attack window")} in the test set (at least ${MIN_SUPPORT_FOR_RATES} needed for a rate to mean anything).`,
  );

  if (Array.isArray(pretrainedTrainReplicas)) {
    const seen = new Set(pretrainedTrainReplicas);
    const contaminated = ids.filter((id, r) => testReps.has(r) && seen.has(id));
    add(
      "pretrained-contamination",
      "The pretrained model has not seen the test replicas",
      contaminated.length ? "fail" : "ok",
      contaminated.length
        ? `The pretrained model was trained on the real ${joinList(contaminated)} data, and this test set uses ${contaminated.length === 1 ? "that replica" : "those replicas"}. Its score here would be inflated; only santos is unseen by it.`
        : "The pretrained model never trained on any replica in this test set.",
    );
  }

  const overall = worstSeverity(checks.filter((c) => c.group === "leakage").map((c) => c.severity));
  const support = worstSeverity(checks.filter((c) => c.group === "support").map((c) => c.severity));

  return {
    checks,
    replicaOverlap,
    stretchOverlap,
    attackEpisodeOverlap,
    testRowsDuplicatingTrain,
    quietDuplicates,
    activeDuplicates,
    featureSet,
    nTest: test.length,
    testPositives,
    lowSupport,
    overall,
    support,
  };
}
