// splits.js: partitions, protocols, validation messages and the leakage
// certificate, on a small hand-made corpus that follows data contract (2)
// (deliberately independent of datagen.js), plus a few end-to-end checks of the
// certificate on the real calibrated generator (datagen.js + replica_stats.json).
//
// Run: node --test "web/tests/*.test.mjs"   (Node 24 does not accept a bare directory)

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_SPLIT,
  REPLICA_ORDER,
  PRETRAINED_TRAIN_REPLICAS,
  validateSplitConfig,
  buildSplit,
  loroFolds,
  leakageCertificate,
  cloneSplitConfig,
  featureColumns,
  FEATURE_SETS,
  LEAKAGE_CHECK_IDS,
  pctOf,
  worstSeverity,
} from "../js/lab/splits.js";
import { makeRng } from "../js/lab/rng.js";
import { generateCorpus } from "../js/lab/datagen.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const D = 42;
const HOSTS = ["vpn", "intranet_server", "inet-firewall"];

/**
 * Fake corpus: every replica has 3 host timelines of `T` windows, cut into
 * stretches of `stretchLen`. On inet-firewall, every other stretch is an attack
 * episode (except in `fewAttacks` replicas, which get one short episode). Rows are
 * random; `dupRate` of rows copy their predecessor within a stretch, like the
 * real data's exact-duplicate runs.
 */
function fakeCorpus({ T = 48, stretchLen = 8, dupRate = 0.2, fewAttacks = ["shaw"], seed = 1 } = {}) {
  const rng = makeRng(seed, "fake-corpus");
  const rows = [];
  let stretchId = 0;
  REPLICA_ORDER.forEach((rep, r) => {
    HOSTS.forEach((host, h) => {
      for (let start = 0; start < T; start += stretchLen) {
        const k = start / stretchLen;
        let attack = h === 2 && k % 2 === 1;
        if (fewAttacks.includes(rep)) attack = h === 2 && k === 1;
        const sid = stretchId++;
        for (let t = start; t < Math.min(T, start + stretchLen); t++) {
          let isAttack = attack;
          if (fewAttacks.includes(rep) && attack && t >= start + 3) isAttack = false; // 3 positives only
          let x;
          let isDup = 0;
          const prev = rows[rows.length - 1];
          if (t > start && prev.stretch === sid && rng.bernoulli(dupRate) && prev.y === (isAttack ? 1 : 0)) {
            x = prev.x.slice();
            isDup = 1;
          } else {
            x = Array.from({ length: D }, () => (rng.bernoulli(0.3) ? 0 : Math.exp(rng.normal())));
          }
          rows.push({ r, h, t, stretch: sid, y: isAttack ? 1 : 0, x, isDup });
        }
      }
    });
  });
  return toCorpus(rows);
}

function toCorpus(rows) {
  const n = rows.length;
  const X = new Float32Array(n * D);
  const y = new Uint8Array(n);
  const avail = new Uint8Array(n * 4);
  const replica = new Uint8Array(n);
  const host = new Uint8Array(n);
  const t = new Int32Array(n);
  const stretch = new Int32Array(n);
  const isDuplicate = new Uint8Array(n);
  rows.forEach((row, i) => {
    X.set(row.x, i * D);
    y[i] = row.y;
    avail.set([1, 1, 1, 1], i * 4);
    replica[i] = row.r;
    host[i] = row.h;
    t[i] = row.t;
    stretch[i] = row.stretch;
    isDuplicate[i] = row.isDup;
  });
  return {
    n,
    nFeatures: D,
    X,
    y,
    avail,
    replica,
    host,
    t,
    stretch,
    isDuplicate,
    replicaIds: [...REPLICA_ORDER],
    hostIds: [...HOSTS],
    hostLabels: ["VPN Gateway", "Intranet Server", "Internet Firewall"],
    featureNames: Array.from({ length: D }, (_, j) => `f${j}`),
    meta: { scale: 1, seed: 0, generator: "test-fake", perReplica: [] },
  };
}

const corpus = fakeCorpus();
// Chronological splits purge every stretch that continues across a boundary, so
// they need timelines long enough to keep windows after the purge: 96 windows in
// stretches of 8 (with T = 48, the 7 test windows of each timeline all continue
// the stretch that starts in validation, and the test set would be empty).
const chronoCorpus = fakeCorpus({ T: 96 });

const RANDOM = { mode: "random", replicas: [...REPLICA_ORDER], trainPct: 70, valPct: 15, testPct: 15, seed: 0 };
const CHRONO = { mode: "chronological", replicas: [...REPLICA_ORDER], trainPct: 70, valPct: 15, testPct: 15 };

function assertPartition(split, n, label) {
  const seen = new Uint8Array(n);
  let total = 0;
  for (const p of ["train", "val", "test", "unused"]) {
    assert.ok(split[p] instanceof Int32Array, `${label}: ${p} is an Int32Array`);
    for (let k = 0; k < split[p].length; k++) {
      const i = split[p][k];
      if (k > 0) assert.ok(split[p][k - 1] < i, `${label}: ${p} is sorted and unique`);
      assert.equal(seen[i], 0, `${label}: row ${i} appears twice`);
      seen[i] = 1;
      total++;
    }
  }
  assert.equal(total, n, `${label}: every row is assigned exactly once`);
  const s = split.summary;
  assert.equal(s.n.total, n);
  assert.equal(s.n.train + s.n.val + s.n.test + s.n.unused, n);
  const pctSum = s.pct.train + s.pct.val + s.pct.test + s.pct.unused;
  assert.ok(Math.abs(pctSum - 100) < 1e-9, `${label}: percentages sum to 100`);
}

// ------------------------------------------------------------------ partitions

test("every mode partitions the rows disjointly and completely", () => {
  const configs = {
    default: DEFAULT_SPLIT,
    replicaSubset: { mode: "replica", train: ["fox", "wheeler"], test: ["wilson"], valFraction: 0.2, seed: 3 },
    random: RANDOM,
    randomSubset: { ...RANDOM, replicas: ["santos", "harrison"] },
  };
  for (const [label, cfg] of Object.entries(configs)) assertPartition(buildSplit(corpus, cfg), corpus.n, label);
  assertPartition(buildSplit(chronoCorpus, CHRONO), chronoCorpus.n, "chronological");
  assertPartition(buildSplit(chronoCorpus, { ...CHRONO, replicas: ["fox"] }), chronoCorpus.n, "chronologicalSubset");
});

test("replica mode never puts a replica on both sides, and unused replicas stay unused", () => {
  const cfg = { mode: "replica", train: ["fox", "wheeler", "wardbeck"], test: ["santos", "shaw"], valFraction: 0.15, seed: 0 };
  const split = buildSplit(corpus, cfg);
  const repsOf = (idx) => new Set(Array.from(idx, (i) => corpus.replicaIds[corpus.replica[i]]));
  const trainSide = new Set([...repsOf(split.train), ...repsOf(split.val)]);
  const test = repsOf(split.test);
  for (const r of test) assert.ok(!trainSide.has(r), `${r} on both sides`);
  assert.deepEqual([...test].sort(), ["santos", "shaw"]);
  assert.deepEqual([...trainSide].sort(), ["fox", "wardbeck", "wheeler"]);
  const pr = split.summary.perReplica;
  assert.equal(pr.fox.role, "train");
  assert.equal(pr.santos.role, "test");
  assert.equal(pr.russellmitchell.role, "unused");
  assert.equal(pr.shaw.nPositive, 3);
  assert.equal(split.summary.mode, "replica");
});

test("replica-mode validation is a seeded random valFraction of the training rows", () => {
  const a = buildSplit(corpus, DEFAULT_SPLIT);
  const b = buildSplit(corpus, cloneSplitConfig(DEFAULT_SPLIT));
  const c = buildSplit(corpus, { ...cloneSplitConfig(DEFAULT_SPLIT), seed: 7 });
  const pool = a.train.length + a.val.length;
  assert.equal(a.val.length, Math.round(pool * 0.15));
  assert.deepEqual(a.val, b.val, "same seed, same validation rows");
  assert.deepEqual(a.train, b.train);
  assert.notDeepEqual(a.val, c.val, "different seed, different validation rows");
  assert.equal(c.val.length, a.val.length);
  // validation is drawn from several replicas, not one block
  const valReps = new Set(Array.from(a.val, (i) => corpus.replica[i]));
  assert.ok(valReps.size >= 5);
  const d = buildSplit(corpus, { ...cloneSplitConfig(DEFAULT_SPLIT), valFraction: 0.3 });
  assert.equal(d.val.length, Math.round((d.train.length + d.val.length) * 0.3));
});

test("random mode honours the percentages to within one row", () => {
  for (const [tr, va, te] of [
    [70, 15, 15],
    [60, 20, 20],
    [80, 0, 20],
    [33.3, 33.3, 33.4],
  ]) {
    const split = buildSplit(corpus, { ...RANDOM, trainPct: tr, valPct: va, testPct: te });
    const N = corpus.n;
    assert.ok(Math.abs(split.test.length - (N * te) / 100) <= 1, `test ${split.test.length} vs ${(N * te) / 100}`);
    assert.ok(Math.abs(split.val.length - (N * va) / 100) <= 1, `val ${split.val.length}`);
    assert.ok(Math.abs(split.train.length - (N * tr) / 100) <= 1, `train ${split.train.length}`);
    assert.equal(split.unused.length, 0);
  }
  const again = buildSplit(corpus, RANDOM);
  assert.deepEqual(again.test, buildSplit(corpus, RANDOM).test, "seeded");
  assert.notDeepEqual(again.test, buildSplit(corpus, { ...RANDOM, seed: 1 }).test);
  assert.equal(again.summary.perReplica.fox.role, "mixed");
});

test("chronological mode orders train < validation < test inside every replica+host timeline", () => {
  const c = chronoCorpus;
  const split = buildSplit(c, CHRONO);
  const role = new Map();
  for (const p of ["train", "val", "test", "unused"]) for (const i of split[p]) role.set(i, p);
  const timelines = new Map();
  for (let i = 0; i < c.n; i++) {
    const key = `${c.replica[i]}/${c.host[i]}`;
    if (!timelines.has(key)) timelines.set(key, { train: [], val: [], test: [], unused: [] });
    timelines.get(key)[role.get(i)].push(c.t[i]);
  }
  assert.equal(timelines.size, REPLICA_ORDER.length * HOSTS.length);
  for (const [key, tl] of timelines) {
    assert.ok(tl.train.length && tl.val.length && tl.test.length, `${key}: all three parts present`);
    assert.ok(Math.max(...tl.train) < Math.min(...tl.val), `${key}: train before validation`);
    assert.ok(Math.max(...tl.val) < Math.min(...tl.test), `${key}: validation before test`);
  }
  const tlLen = 96;
  assert.equal(split.train.length, REPLICA_ORDER.length * HOSTS.length * Math.round(tlLen * 0.7));
  assert.equal(split.summary.mode, "chronological");
});

test("chronological mode purges each boundary: windows continuing an earlier stretch move to unused", () => {
  const c = chronoCorpus;
  const split = buildSplit(c, CHRONO);
  // Per 96-window timeline (stretches of 8): train = t 0..66, validation t 67..80,
  // test t 81..95 before the purge. Validation t 67..71 continue stretch 64..71
  // from training (5 windows); test t 81..87 continue stretch 80..87 from
  // validation (7 windows). 24 timelines.
  const nTl = REPLICA_ORDER.length * HOSTS.length;
  assert.deepEqual(split.summary.purged, { val: 5 * nTl, test: 7 * nTl, positives: { val: 0, test: 0 } });
  assert.equal(split.unused.length, 12 * nTl);
  assert.equal(split.summary.n.unused, 12 * nTl);
  // no stretch appears in two of train / validation / test
  const partOf = new Map();
  for (const p of ["train", "val", "test"]) {
    for (const i of split[p]) {
      const st = c.stretch[i];
      assert.ok(!partOf.has(st) || partOf.get(st) === p, `stretch ${st} in ${partOf.get(st)} and ${p}`);
      partOf.set(st, p);
    }
  }
  // so the certificate's stretch check passes by construction, and says why
  const cert = leakageCertificate(c, split);
  assert.equal(check(cert, "stretch-disjoint").severity, "ok");
  assert.equal(cert.stretchOverlap, 0);
  assert.match(check(cert, "stretch-disjoint").detail, /set aside 288 windows .* 600-second gap/);
  // the other modes do not purge
  assert.equal(buildSplit(c, RANDOM).summary.purged, null);
  assert.equal(buildSplit(c, DEFAULT_SPLIT).summary.purged, null);
  // an attack episode that straddles a boundary loses its later windows, and they are counted
  const attackCut = buildSplit(c, { ...CHRONO, trainPct: 20, valPct: 40, testPct: 40 });
  assert.ok(attackCut.summary.purged.positives.val + attackCut.summary.purged.positives.test > 0);
  let unusedPos = 0;
  for (const i of attackCut.unused) unusedPos += c.y[i];
  assert.equal(unusedPos, attackCut.summary.purged.positives.val + attackCut.summary.purged.positives.test);
  // a test set emptied by the purge is refused in plain language
  assert.throws(() => buildSplit(corpus, CHRONO), /no test windows are left/);
});

// ------------------------------------------------------------------ validation

test("validateSplitConfig accepts the defaults and rejects bad configs in plain language", () => {
  const ids = corpus.replicaIds;
  assert.deepEqual(validateSplitConfig(DEFAULT_SPLIT, ids), { ok: true, errors: [] });
  assert.equal(validateSplitConfig(RANDOM, ids).ok, true);
  assert.equal(validateSplitConfig(CHRONO, ids).ok, true);
  assert.equal(validateSplitConfig({ mode: "loro", replicas: ["fox", "santos"], valFraction: 0.15, seed: 0 }, ids).ok, true);

  const bad = [
    [{ mode: "replica", train: [], test: ["santos"] }, /at least one replica for training/],
    [{ mode: "replica", train: ["fox"], test: [] }, /at least one replica for testing/],
    [{ mode: "replica", train: ["fox", "santos"], test: ["santos"] }, /santos is in both training and testing/],
    [{ mode: "replica", train: ["atlantis"], test: ["santos"] }, /"atlantis".*not a replica/],
    [{ mode: "replica", train: ["fox"], test: ["santos"], valFraction: 0.6 }, /between 5% and 40%/],
    [{ mode: "replica", train: ["fox"], test: ["santos"], valFraction: 0.01 }, /between 5% and 40%/],
    [{ ...RANDOM, trainPct: 70, valPct: 20, testPct: 20 }, /add up to 100%/],
    [{ ...RANDOM, trainPct: 90, valPct: 7, testPct: 3 }, /at least 5% for testing/],
    [{ ...RANDOM, trainPct: 10, valPct: 45, testPct: 45 }, /at least 20% for training/],
    [{ ...RANDOM, trainPct: 110, valPct: -10, testPct: 0 }, /cannot be negative/],
    [{ ...CHRONO, replicas: [] }, /at least one replica/],
    [{ ...CHRONO, testPct: undefined }, /test percentage is missing/],
    [{ mode: "loro", replicas: ["fox"] }, /at least 2 replicas/],
    [{ mode: "sideways" }, /Unknown split type/],
  ];
  for (const [cfg, pattern] of bad) {
    const res = validateSplitConfig(cfg, ids);
    assert.equal(res.ok, false, JSON.stringify(cfg));
    assert.ok(res.errors.length > 0);
    assert.ok(res.errors.some((e) => pattern.test(e)), `${JSON.stringify(cfg)} -> ${res.errors.join(" | ")}`);
    for (const e of res.errors) assert.match(e, /^[A-Z"a-z].*[.]$/, `readable sentence: ${e}`);
  }
  assert.equal(validateSplitConfig(null, ids).ok, false);
  assert.throws(() => buildSplit(corpus, { mode: "replica", train: [], test: ["santos"] }), /Invalid split/);
  assert.throws(() => buildSplit(corpus, { mode: "loro", replicas: ["fox", "santos"] }), /loroFolds/);
});

test("loroFolds gives one fold per replica, each holding out exactly that replica", () => {
  const folds = loroFolds({ mode: "loro", replicas: [...REPLICA_ORDER], valFraction: 0.2, seed: 4 }, corpus.replicaIds);
  assert.equal(folds.length, REPLICA_ORDER.length);
  folds.forEach((f, k) => {
    assert.equal(f.mode, "replica");
    assert.deepEqual(f.test, [REPLICA_ORDER[k]]);
    assert.equal(f.train.length, REPLICA_ORDER.length - 1);
    assert.ok(!f.train.includes(REPLICA_ORDER[k]));
    assert.equal(f.valFraction, 0.2);
    assert.equal(f.seed, 4);
    assert.equal(validateSplitConfig(f, corpus.replicaIds).ok, true);
  });
  const sub = loroFolds({ mode: "loro", replicas: ["shaw", "fox", "santos"] }, corpus.replicaIds);
  assert.deepEqual(sub.map((f) => f.test[0]), ["santos", "fox", "shaw"], "canonical replica order");
  assert.deepEqual(sub[0].train, ["fox", "shaw"]);
  assert.equal(sub[0].valFraction, 0.15);
});

// ------------------------------------------------------------------ leakage certificate

const check = (cert, id) => cert.checks.find((c) => c.id === id);

test("the default replica split passes replica- and stretch-disjointness", () => {
  const split = buildSplit(corpus, DEFAULT_SPLIT);
  const cert = leakageCertificate(corpus, split);
  assert.equal(check(cert, "replica-disjoint").pass, true);
  assert.equal(check(cert, "replica-disjoint").severity, "ok");
  assert.equal(check(cert, "stretch-disjoint").pass, true);
  assert.equal(check(cert, "duplicate-rows").severity, "ok");
  assert.equal(check(cert, "test-support").severity, "ok");
  assert.deepEqual(cert.replicaOverlap, []);
  assert.equal(cert.stretchOverlap, 0);
  assert.equal(cert.testRowsDuplicatingTrain, 0);
  assert.equal(cert.lowSupport, false);
  assert.equal(cert.overall, "ok");
  assert.equal(check(cert, "pretrained-contamination"), undefined, "only reported when asked");
  for (const c of cert.checks) {
    assert.equal(typeof c.label, "string");
    assert.ok(c.detail.length > 20);
  }
});

test("a random row-level split is flagged: shared replicas (warn), cut stretches and copied rows (fail)", () => {
  const cert = leakageCertificate(corpus, buildSplit(corpus, RANDOM));
  // sharing testbeds narrows the question (within-network generalisation); it is not by itself a leak
  assert.equal(check(cert, "replica-disjoint").severity, "warn");
  assert.equal(check(cert, "replica-disjoint").pass, true);
  assert.match(check(cert, "replica-disjoint").detail, /^All 8 replicas have windows in both training and testing/);
  assert.match(check(cert, "replica-disjoint").detail, /not to a new network/);
  assert.equal(cert.replicaOverlap.length, REPLICA_ORDER.length);
  assert.equal(check(cert, "stretch-disjoint").severity, "fail");
  assert.ok(cert.stretchOverlap > 10);
  assert.ok(cert.attackEpisodeOverlap > 0);
  assert.match(check(cert, "stretch-disjoint").detail, /attack episode/);
  // the fake corpus copies ~20% of rows from their predecessor
  assert.ok(cert.testRowsDuplicatingTrain > 0);
  assert.equal(cert.overall, "fail");
});

test("a chronological split shares testbeds (warn) but, once purged, cuts no stretch", () => {
  const cert = leakageCertificate(chronoCorpus, buildSplit(chronoCorpus, CHRONO));
  assert.equal(check(cert, "replica-disjoint").severity, "warn");
  assert.equal(check(cert, "stretch-disjoint").severity, "ok");
  assert.notEqual(cert.overall, "ok", "the shared testbeds still show as a warning");
});

test("leakage and support are separate verdicts; every check says which it belongs to", () => {
  assert.deepEqual([...LEAKAGE_CHECK_IDS], ["replica-disjoint", "stretch-disjoint", "duplicate-rows", "pretrained-contamination"]);
  assert.equal(worstSeverity([]), "ok");
  assert.equal(worstSeverity(["ok", "warn", "ok"]), "warn");
  assert.equal(worstSeverity(["warn", "fail", "ok"]), "fail");
  const opts = { pretrainedTrainReplicas: [...PRETRAINED_TRAIN_REPLICAS] };
  const cert = leakageCertificate(corpus, buildSplit(corpus, DEFAULT_SPLIT), opts);
  for (const ch of cert.checks) assert.equal(ch.group, ch.id === "test-support" ? "support" : "leakage", ch.id);
  assert.equal(cert.support, "ok");
  assert.equal(cert.nTest, buildSplit(corpus, DEFAULT_SPLIT).test.length);

  // no attacks in the test set: support fails, but that is not a leak
  const noAttacks = { ...corpus, y: new Uint8Array(corpus.n) };
  const none = leakageCertificate(noAttacks, buildSplit(noAttacks, DEFAULT_SPLIT));
  assert.equal(check(none, "test-support").severity, "fail");
  assert.equal(none.support, "fail");
  assert.equal(none.overall, "ok", "a test set without attacks is unscoreable, not leaky");
  assert.match(check(none, "test-support").detail, /average precision is undefined/);
  assert.match(check(none, "test-support").detail, /not a leak/);
  assert.match(check(none, "test-support").detail, /left out of the summary/);

  // a few attacks: support warns, the leakage verdict is untouched
  const shawCfg = { mode: "replica", train: REPLICA_ORDER.filter((r) => r !== "shaw"), test: ["shaw"], valFraction: 0.15, seed: 0 };
  const shaw = leakageCertificate(corpus, buildSplit(corpus, shawCfg));
  assert.equal(shaw.support, "warn");
  assert.equal(shaw.overall, "ok");

  // the leakage verdict ignores support but follows every leakage check
  const leaky = leakageCertificate(noAttacks, buildSplit(noAttacks, RANDOM));
  assert.equal(leaky.overall, "fail");
  assert.equal(leaky.support, "fail");
});

test("pctOf never prints 0.0% for a non-zero count or 100.0% for a partial one", () => {
  assert.equal(pctOf(1, 2500), "<0.1%");
  assert.equal(pctOf(1, 2001), "<0.1%");
  assert.equal(pctOf(1, 1333), "0.1%", "0.075% rounds to 0.1%");
  assert.equal(pctOf(0, 1333), "0.0%");
  assert.equal(pctOf(173, 999), "17.3%");
  assert.equal(pctOf(9999, 10000), ">99.9%");
  assert.equal(pctOf(10, 10), "100.0%");
  assert.equal(pctOf(0, 0), "0%");
});

test("injected duplicate rows are flagged: fail above 1% of the test set, warn below", () => {
  const split = buildSplit(corpus, DEFAULT_SPLIT);
  const trainRow = split.train[0];
  const copyInto = (c, targets) => {
    const X = c.X.slice();
    for (const i of targets) X.copyWithin(i * D, trainRow * D, trainRow * D + D);
    return { ...c, X };
  };
  const one = leakageCertificate(copyInto(corpus, [split.test[5]]), split);
  assert.equal(one.testRowsDuplicatingTrain, 1);
  assert.equal(check(one, "duplicate-rows").severity, "warn");
  assert.equal(check(one, "duplicate-rows").pass, true);
  assert.equal(one.overall, "warn");

  const many = Array.from(split.test.slice(0, Math.ceil(split.test.length * 0.05)));
  const lots = leakageCertificate(copyInto(corpus, many), split);
  assert.equal(lots.testRowsDuplicatingTrain, many.length);
  assert.equal(check(lots, "duplicate-rows").severity, "fail");
  assert.equal(check(lots, "duplicate-rows").pass, false);

  // a copy of a VALIDATION row also counts: validation picks the stopping epoch
  const X = corpus.X.slice();
  X.copyWithin(split.test[0] * D, split.val[0] * D, split.val[0] * D + D);
  assert.equal(leakageCertificate({ ...corpus, X }, split).testRowsDuplicatingTrain, 1);

  // -0 and +0 are the same value
  const Z = corpus.X.slice();
  Z.copyWithin(split.test[1] * D, trainRow * D, trainRow * D + D);
  for (let j = 0; j < D; j++) if (Z[split.test[1] * D + j] === 0) Z[split.test[1] * D + j] = -0;
  assert.equal(leakageCertificate({ ...corpus, X: Z }, split).testRowsDuplicatingTrain, 1);
});

test("the certificate reports quiet and active duplicate counts", () => {
  const split = buildSplit(corpus, RANDOM);
  const cert = leakageCertificate(corpus, split);
  assert.equal(cert.featureSet, "all");
  assert.equal(cert.quietDuplicates + cert.activeDuplicates, cert.testRowsDuplicatingTrain);
  // the fake rows are random (30% zeros per column), so every copy is active
  assert.equal(cert.quietDuplicates, 0);
  assert.equal(check(cert, "duplicate-rows").severity, "fail");
  assert.match(check(cert, "duplicate-rows").detail, /real activity/);
});

// ------------------------------------------------------------------ quiet vs active duplicates

const M3 = [32, 33];
/** Make row i "quiet": every column zero except the two M3 host-identity ones. */
function quieten(X, i, m3 = [3, 1]) {
  X.fill(0, i * D, i * D + D);
  X[i * D + M3[0]] = m3[0];
  X[i * D + M3[1]] = m3[1];
}
const copyRow = (X, to, from) => X.copyWithin(to * D, from * D, from * D + D);

test("featureColumns is the single source of the compared columns", () => {
  assert.deepEqual(FEATURE_SETS, ["all", "m2"]);
  const all = featureColumns("all");
  assert.ok(all instanceof Int32Array);
  assert.deepEqual(Array.from(all), Array.from({ length: 42 }, (_, j) => j));
  assert.deepEqual(Array.from(featureColumns()), Array.from(all), "'all' is the default");
  const m2 = featureColumns("m2");
  assert.ok(m2 instanceof Int32Array);
  assert.equal(m2.length, 24);
  assert.deepEqual(Array.from(m2), Array.from({ length: 24 }, (_, j) => 8 + j));
  m2[0] = 99;
  assert.equal(featureColumns("m2")[0], 8, "each call returns a fresh copy");
  assert.throws(() => featureColumns("m5"), /Unknown feature set "m5"/);
  assert.throws(() => leakageCertificate(corpus, buildSplit(corpus, DEFAULT_SPLIT), { featureSet: "bogus" }), /Unknown feature set/);
});

test("quiet duplicates (only M3 host identity set) pass; active ones warn, then fail above 1%", () => {
  const split = buildSplit(corpus, DEFAULT_SPLIT);
  assert.equal(leakageCertificate(corpus, split).testRowsDuplicatingTrain, 0, "clean to start with");
  const nTest = split.test.length;
  const nQuiet = Math.ceil(nTest * 0.05); // well above the 1% fail line
  const X = corpus.X.slice();
  quieten(X, split.train[0]);
  for (let k = 0; k < nQuiet; k++) copyRow(X, split.test[k], split.train[0]);
  // a quiet test row with a DIFFERENT host identity is not a duplicate at all
  quieten(X, split.test[nQuiet], [3, 2]);

  const quiet = leakageCertificate({ ...corpus, X }, split);
  assert.equal(quiet.testRowsDuplicatingTrain, nQuiet);
  assert.equal(quiet.quietDuplicates, nQuiet);
  assert.equal(quiet.activeDuplicates, 0);
  const qc = check(quiet, "duplicate-rows");
  assert.equal(qc.severity, "ok");
  assert.equal(qc.pass, true);
  assert.match(qc.detail, new RegExp(`^${nQuiet} test windows`));
  assert.match(qc.detail, /quiet/);
  assert.match(qc.detail, /not a leak/);
  assert.equal(quiet.overall, "ok");

  // -0 in the activity columns still counts as quiet
  const Z = X.slice();
  const zRow = split.test[0];
  for (let j = 0; j < D; j++) if (!M3.includes(j)) Z[zRow * D + j] = -0;
  assert.equal(leakageCertificate({ ...corpus, X: Z }, split).quietDuplicates, nQuiet);

  // one active duplicate on top: warn, and the detail still mentions the quiet ones
  const A = X.slice();
  const activeTrain = split.train[1];
  copyRow(A, split.test[nQuiet + 1], activeTrain);
  const one = leakageCertificate({ ...corpus, X: A }, split);
  assert.equal(one.activeDuplicates, 1);
  assert.equal(one.quietDuplicates, nQuiet);
  assert.equal(one.testRowsDuplicatingTrain, nQuiet + 1);
  const oc = check(one, "duplicate-rows");
  assert.equal(oc.severity, "warn");
  assert.equal(oc.pass, true);
  assert.match(oc.detail, /^1 test window with real activity/);
  assert.match(oc.detail, new RegExp(`${nQuiet} quiet windows`));

  // active duplicates above 1% of the test set fail
  const nActive = Math.ceil(nTest * 0.02);
  for (let k = 0; k < nActive; k++) copyRow(A, split.test[nQuiet + 1 + k], activeTrain);
  const many = leakageCertificate({ ...corpus, X: A }, split);
  assert.equal(many.activeDuplicates, nActive);
  assert.equal(check(many, "duplicate-rows").severity, "fail");
  assert.equal(check(many, "duplicate-rows").pass, false);
  assert.equal(many.overall, "fail");
});

test("featureSet 'm2' compares only the 24 metric columns the model sees", () => {
  const split = buildSplit(corpus, DEFAULT_SPLIT);
  const X = corpus.X.slice();
  // test[0]: same M2 columns as a training row, different everything else
  const src = split.train[0];
  X.copyWithin(split.test[0] * D + 8, src * D + 8, src * D + 32);
  // test[1] and train[1]: all 24 M2 columns zero, other columns differ
  X.fill(0, split.test[1] * D + 8, split.test[1] * D + 32);
  X.fill(0, split.train[1] * D + 8, split.train[1] * D + 32);
  const c = { ...corpus, X };

  const all = leakageCertificate(c, split);
  assert.equal(all.testRowsDuplicatingTrain, 0, "not copies on all 42 columns");

  const m2 = leakageCertificate(c, split, { featureSet: "m2" });
  assert.equal(m2.featureSet, "m2");
  assert.equal(m2.testRowsDuplicatingTrain, 2);
  assert.equal(m2.activeDuplicates, 1, "the copied metric row");
  assert.equal(m2.quietDuplicates, 1, "the all-zero metric row");
  const mc = check(m2, "duplicate-rows");
  assert.equal(mc.severity, "warn");
  assert.match(mc.detail, /24 network-metric columns/);

  // with only the quiet one left, m2 passes
  const Q = corpus.X.slice();
  Q.fill(0, split.test[1] * D + 8, split.test[1] * D + 32);
  Q.fill(0, split.train[1] * D + 8, split.train[1] * D + 32);
  const q = leakageCertificate({ ...corpus, X: Q }, split, { featureSet: "m2" });
  assert.equal(q.quietDuplicates, 1);
  assert.equal(q.activeDuplicates, 0);
  assert.equal(check(q, "duplicate-rows").severity, "ok");
  assert.match(check(q, "duplicate-rows").detail, /^1 test window .*it is a quiet window/);
});

// ------------------------------------------------------------------ on the real calibrated generator

const here = dirname(fileURLToPath(import.meta.url));
const stats = JSON.parse(readFileSync(join(here, "..", "data/replica_stats.json"), "utf8"));
let synth = null;
const synthCorpus = () => (synth ??= generateCorpus(stats, { scale: 0.05, seed: 0 }));

test("synthetic data (scale 0.05, seed 0): the default split passes every check", () => {
  const c = synthCorpus();
  for (const featureSet of FEATURE_SETS) {
    const cert = leakageCertificate(c, buildSplit(c, DEFAULT_SPLIT), { featureSet });
    for (const ch of cert.checks) assert.equal(ch.severity, "ok", `${featureSet}: ${ch.id} - ${ch.detail}`);
    assert.equal(cert.overall, "ok");
    assert.equal(cert.activeDuplicates, 0, featureSet);
    // quiet windows (host identity only) do repeat across testbeds - reported, not failed
    assert.ok(cert.quietDuplicates > 0, featureSet);
    assert.equal(cert.quietDuplicates, cert.testRowsDuplicatingTrain);
  }
  const opts = { pretrainedTrainReplicas: [...PRETRAINED_TRAIN_REPLICAS] };
  assert.equal(leakageCertificate(c, buildSplit(c, DEFAULT_SPLIT), opts).overall, "ok");
});

test("synthetic data (scale 0.05, seed 0): a random split over all 8 replicas still fails", () => {
  const c = synthCorpus();
  const split = buildSplit(c, RANDOM);
  const cert = leakageCertificate(c, split);
  assert.equal(check(cert, "stretch-disjoint").severity, "fail");
  assert.equal(check(cert, "duplicate-rows").severity, "fail");
  assert.ok(cert.activeDuplicates / split.test.length > 0.01, `${cert.activeDuplicates} active of ${split.test.length}`);
  assert.equal(cert.overall, "fail");
});

test("synthetic data (scale 0.05, seed 0): no LORO fold leaks; shaw's fold fails on support only", () => {
  const c = synthCorpus();
  for (const fc of loroFolds({ mode: "loro", replicas: [...REPLICA_ORDER] }, c.replicaIds)) {
    const cert = leakageCertificate(c, buildSplit(c, fc));
    assert.equal(cert.activeDuplicates, 0, fc.test[0]);
    assert.equal(check(cert, "duplicate-rows").severity, "ok", fc.test[0]);
    assert.equal(check(cert, "stretch-disjoint").severity, "ok", fc.test[0]);
    assert.equal(cert.overall, "ok", `${fc.test[0]}: leakage verdict`);
    if (fc.test[0] === "shaw") {
      // measured: 0 synthetic attack windows in shaw at this size and seed
      assert.equal(cert.testPositives, 0);
      assert.equal(cert.support, "fail");
    }
  }
});

test("synthetic data (scale 0.05, seed 0): a purged chronological split over 8 replicas cuts no stretch", () => {
  const c = synthCorpus();
  const split = buildSplit(c, CHRONO);
  const p = split.summary.purged;
  assert.ok(p.val + p.test > 0, "the boundaries had stretches to purge");
  for (const featureSet of FEATURE_SETS) {
    const cert = leakageCertificate(c, split, { featureSet });
    assert.equal(check(cert, "stretch-disjoint").severity, "ok", featureSet);
    assert.equal(check(cert, "replica-disjoint").severity, "warn", featureSet);
    assert.notEqual(cert.overall, "fail", `${featureSet}: ${cert.checks.map((ch) => `${ch.id}=${ch.severity}`).join(", ")}`);
    assert.equal(cert.support, "ok");
  }
});

test("pretrained contamination fails on a non-santos test replica and passes for santos", () => {
  const opts = { pretrainedTrainReplicas: [...PRETRAINED_TRAIN_REPLICAS] };
  const santos = leakageCertificate(corpus, buildSplit(corpus, DEFAULT_SPLIT), opts);
  assert.equal(check(santos, "pretrained-contamination").severity, "ok");
  assert.equal(check(santos, "pretrained-contamination").pass, true);

  const foxCfg = { mode: "replica", train: REPLICA_ORDER.filter((r) => r !== "fox"), test: ["fox"], valFraction: 0.15, seed: 0 };
  const fox = leakageCertificate(corpus, buildSplit(corpus, foxCfg), opts);
  const c = check(fox, "pretrained-contamination");
  assert.equal(c.severity, "fail");
  assert.equal(c.pass, false);
  assert.match(c.detail, /fox/);
  assert.equal(check(fox, "replica-disjoint").pass, true, "the split itself is still clean");
});

test("low test support is flagged (warn under 20 positives, fail at zero)", () => {
  const shawCfg = { mode: "replica", train: REPLICA_ORDER.filter((r) => r !== "shaw"), test: ["shaw"], valFraction: 0.15, seed: 0 };
  const shaw = leakageCertificate(corpus, buildSplit(corpus, shawCfg));
  assert.equal(shaw.testPositives, 3);
  assert.equal(shaw.lowSupport, true);
  assert.equal(check(shaw, "test-support").severity, "warn");
  assert.match(check(shaw, "test-support").detail, /fewer than 20/);

  const noAttacks = { ...corpus, y: new Uint8Array(corpus.n) };
  const none = leakageCertificate(noAttacks, buildSplit(noAttacks, DEFAULT_SPLIT));
  assert.equal(none.testPositives, 0);
  assert.equal(check(none, "test-support").severity, "fail");
  assert.equal(none.support, "fail");
  assert.equal(none.overall, "ok");
});

test("the certificate on ~20k rows takes well under 100 ms", () => {
  const big = fakeCorpus({ T: 840, stretchLen: 60, dupRate: 0.25, seed: 9 });
  assert.ok(big.n >= 20000, `n = ${big.n}`);
  const split = buildSplit(big, RANDOM);
  leakageCertificate(big, split); // warm-up (JIT)
  const t0 = performance.now();
  const cert = leakageCertificate(big, split);
  const ms = performance.now() - t0;
  assert.ok(cert.testRowsDuplicatingTrain > 0);
  assert.ok(ms < 100, `certificate took ${ms.toFixed(1)} ms`);
  console.log(`certificate on ${big.n} rows: ${ms.toFixed(1)} ms`);
  const t1 = performance.now();
  buildSplit(big, DEFAULT_SPLIT);
  assert.ok(performance.now() - t1 < 200, "buildSplit is fast too");
});
