// Tests for web/js/lab/datagen.js - the Training Lab's SYNTHETIC corpus,
// sampled from aggregate statistics of the real AIT cache
// (web/data/replica_stats.json). Run: node --test web/tests/datagen.test.mjs
//
// The last block is the generator-fidelity check: the REAL pretrained
// TESSERA-base model (web/data/weights.json, trained on the 7 replicas other
// than santos) scores synthetic santos windows, and its AP is compared with the
// AP it achieves on real santos (0.9995, RESULTS.md). A faithful simulation
// should look like real data to a model that has only seen real data.

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { generateCorpus, corpusSummary, sliceRows, normCdf, normInv } from "../js/lab/datagen.js";
import { forward } from "../js/forward.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const readJson = (rel) => JSON.parse(fs.readFileSync(path.join(here, "..", rel), "utf8"));
const stats = readJson("data/replica_stats.json");
const weights = readJson("data/weights.json");
const F = 42;

const cache = new Map();
function corpus(opts) {
  const key = JSON.stringify(opts);
  if (!cache.has(key)) cache.set(key, generateCorpus(stats, opts));
  return cache.get(key);
}
const row = (c, i) => c.X.subarray(i * F, i * F + F);
const bytes = (ta) => Buffer.from(ta.buffer, ta.byteOffset, ta.byteLength);
const repStats = (id) => stats.replicas.find((r) => r.id === id);

/** sklearn average_precision_score: sum over distinct thresholds of (R_k - R_{k-1}) * P_k. */
function localAp(y, s) {
  const n = y.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => s[b] - s[a]);
  let nPos = 0;
  for (let i = 0; i < n; i++) nPos += y[i];
  if (nPos === 0 || nPos === n) return null;
  let tp = 0;
  let fp = 0;
  let prevRecall = 0;
  let ap = 0;
  for (let k = 0; k < n; k++) {
    const i = order[k];
    if (y[i]) tp++;
    else fp++;
    if (k + 1 < n && s[order[k + 1]] === s[i]) continue; // tie group: one threshold
    const recall = tp / nPos;
    ap += (recall - prevRecall) * (tp / (tp + fp));
    prevRecall = recall;
  }
  return ap;
}

function scoreWithRealModel(c) {
  const s = new Float64Array(c.n);
  for (let i = 0; i < c.n; i++) {
    s[i] = forward(Array.from(row(c, i)), Array.from(c.avail.subarray(i * 4, i * 4 + 4)), weights).score;
  }
  return s;
}

// ------------------------------------------------------------------ shape

test("corpus has the contract's shape and metadata", () => {
  const c = corpus({ scale: 0.05, seed: 0 });
  assert.equal(c.nFeatures, 42);
  assert.ok(c.X instanceof Float32Array && c.X.length === c.n * F);
  for (const k of ["y", "replica", "host", "isDuplicate"]) assert.ok(c[k] instanceof Uint8Array && c[k].length === c.n, k);
  assert.ok(c.avail instanceof Uint8Array && c.avail.length === c.n * 4);
  assert.ok(c.t instanceof Int32Array && c.t.length === c.n);
  assert.ok(c.stretch instanceof Int32Array && c.stretch.length === c.n);
  assert.deepEqual(c.replicaIds, stats.replicas.map((r) => r.id));
  assert.deepEqual(c.hostIds, ["vpn", "intranet_server", "inet-firewall"]);
  assert.deepEqual(c.hostLabels, stats.hosts.map((h) => h.label));
  assert.deepEqual(c.featureNames, stats.feature_names);
  assert.equal(c.meta.generator, "tessera-synth/v1");
  assert.equal(c.meta.scale, 0.05);
  assert.equal(c.meta.seed, 0);
  assert.equal(c.meta.dependence, "copula");
  assert.equal(c.meta.perReplica.reduce((a, r) => a + r.n, 0), c.n);
});

// ------------------------------------------------------------------ determinism

test("same seed -> byte-identical corpus; different seed -> different", () => {
  const a = generateCorpus(stats, { scale: 0.05, seed: 7 });
  const b = generateCorpus(stats, { scale: 0.05, seed: 7 });
  for (const k of ["X", "y", "avail", "replica", "host", "t", "stretch", "isDuplicate"]) {
    assert.ok(bytes(a[k]).equals(bytes(b[k])), k);
  }
  const c = generateCorpus(stats, { scale: 0.05, seed: 8 });
  assert.ok(!bytes(a.X).equals(bytes(c.X)));
});

test("generating a subset of replicas never changes any replica's rows", () => {
  const full = corpus({ scale: 0.05, seed: 0 });
  for (const ids of [["santos"], ["shaw", "fox"]]) {
    const sub = generateCorpus(stats, { scale: 0.05, seed: 0, replicas: ids });
    assert.deepEqual(sub.replicaIds, full.replicaIds.filter((id) => ids.includes(id)));
    const fullIdx = [];
    for (let i = 0; i < full.n; i++) if (ids.includes(full.replicaIds[full.replica[i]])) fullIdx.push(i);
    assert.equal(sub.n, fullIdx.length);
    fullIdx.forEach((i, r) => {
      assert.equal(sub.replicaIds[sub.replica[r]], full.replicaIds[full.replica[i]]);
      assert.equal(sub.y[r], full.y[i]);
      assert.equal(sub.host[r], full.host[i]);
      assert.equal(sub.t[r], full.t[i]);
      assert.equal(sub.stretch[r], full.stretch[i]);
      assert.equal(sub.isDuplicate[r], full.isDuplicate[i]);
    });
    const fullRows = sliceRows(full, Int32Array.from(fullIdx));
    assert.ok(bytes(fullRows.X).equals(bytes(sub.X)));
    assert.ok(bytes(fullRows.avail).equals(bytes(sub.avail)));
  }
});

test("rejects unknown replicas and bad scales", () => {
  assert.throws(() => generateCorpus(stats, { replicas: ["atlantis"] }), /unknown replica/);
  assert.throws(() => generateCorpus(stats, { scale: 0 }), /scale/);
});

// ------------------------------------------------------------------ calibration

test("per-host row and positive counts track the real counts x scale", () => {
  for (const scale of [0.05, 0.1]) {
    const c = corpus({ scale, seed: 0 });
    c.replicaIds.forEach((id, ri) => {
      const rs = repStats(id);
      let nRep = 0;
      let posRep = 0;
      c.hostIds.forEach((h, hi) => {
        let n = 0;
        let pos = 0;
        for (let i = 0; i < c.n; i++) {
          if (c.replica[i] === ri && c.host[i] === hi) {
            n++;
            pos += c.y[i];
          }
        }
        const hs = rs.hosts[h];
        assert.equal(n, Math.max(4, Math.round(hs.n_windows * scale)), `${id}/${h} n`);
        // stochastic rounding: within one window of the expectation
        assert.ok(Math.abs(pos - hs.n_positive * scale) < 1, `${id}/${h} positives ${pos} vs ${hs.n_positive * scale}`);
        nRep += n;
        posRep += pos;
      });
      const pr = c.meta.perReplica[ri];
      assert.equal(pr.n, nRep);
      assert.equal(pr.nPositive, posRep);
      assert.ok(Math.abs(nRep - rs.n_windows * scale) <= 3, id);
    });
    const sum = corpusSummary(c);
    assert.equal(sum.n, c.n);
    assert.equal(sum.nPositive, c.y.reduce((a, v) => a + v, 0));
    assert.equal(sum.perReplica.length, c.replicaIds.length);
    for (const r of sum.perReplica) assert.ok(r.prevalence >= 0 && r.prevalence <= 1);
  }
});

test("shaw stays a faithfully low-support replica (~0 positives at scale 0.05)", () => {
  for (const seed of [0, 1, 2, 3, 4]) {
    const c = generateCorpus(stats, { scale: 0.05, seed, replicas: ["shaw"] });
    const pos = c.y.reduce((a, v) => a + v, 0);
    assert.ok(pos <= 2, `shaw seed ${seed}: ${pos} positives`);
    assert.ok(pos < stats.min_support);
  }
});

test("attacks come only in contiguous episodes, no more than the real episode count", () => {
  const c = corpus({ scale: 0.1, seed: 0 });
  c.replicaIds.forEach((id, ri) => {
    c.hostIds.forEach((h, hi) => {
      const idx = [];
      for (let i = 0; i < c.n; i++) if (c.replica[i] === ri && c.host[i] === hi) idx.push(i);
      // rows are stored in time order, t = 0..T-1
      idx.forEach((i, k) => assert.equal(c.t[i], k));
      let runs = 0;
      const attackStretches = new Map();
      idx.forEach((i, k) => {
        if (c.y[i] && (k === 0 || !c.y[idx[k - 1]])) runs++;
        if (c.y[i]) {
          const st = c.stretch[i];
          if (!attackStretches.has(st)) attackStretches.set(st, []);
          attackStretches.get(st).push(c.t[i]);
        }
      });
      const maxEp = repStats(id).hosts[h].n_attack_episodes;
      assert.ok(runs <= maxEp, `${id}/${h}: ${runs} runs > ${maxEp} episodes`);
      assert.ok(attackStretches.size <= maxEp, `${id}/${h}: ${attackStretches.size} episodes > ${maxEp}`);
      for (const ts of attackStretches.values()) {
        ts.forEach((v, k) => k && assert.equal(v, ts[k - 1] + 1, `${id}/${h}: episode not contiguous`));
      }
    });
  });
  // a stretch never mixes classes, hosts or replicas; benign blocks are short
  const seen = new Map();
  for (let i = 0; i < c.n; i++) {
    const key = `${c.replica[i]}|${c.host[i]}|${c.y[i]}`;
    const prev = seen.get(c.stretch[i]);
    if (prev) assert.equal(prev.key, key);
    else seen.set(c.stretch[i], { key, n: 0 });
    seen.get(c.stretch[i]).n++;
  }
  const block = Math.max(2, Math.round(stats.benign_block_windows * 0.1));
  for (let i = 0; i < c.n; i++) if (!c.y[i]) assert.ok(seen.get(c.stretch[i]).n <= block);
});

test("exact-duplicate rate is close to the real per-replica rate", () => {
  const c = corpus({ scale: 0.1, seed: 0 });
  let sumAbs = 0;
  const report = [];
  c.replicaIds.forEach((id, ri) => {
    const seen = new Set();
    let d = 0;
    let n = 0;
    for (let i = 0; i < c.n; i++) {
      if (c.replica[i] !== ri) continue;
      n++;
      const k = Array.from(row(c, i)).join(",");
      if (seen.has(k)) d++;
      else seen.add(k);
    }
    const real = repStats(id).exact_duplicate_rate;
    sumAbs += Math.abs(d / n - real);
    report.push(`${id} ${(d / n).toFixed(3)} (real ${real.toFixed(3)})`);
  });
  const meanAbs = sumAbs / c.replicaIds.length;
  console.log(`# exact-duplicate rate, synthetic vs real: ${report.join("; ")}; mean |diff| ${meanAbs.toFixed(3)}`);
  assert.ok(meanAbs < 0.1, `mean |diff| ${meanAbs}`);
  // flagged duplicates are exact copies of the previous window of the same stretch
  for (let i = 0; i < c.n; i++) {
    if (!c.isDuplicate[i]) continue;
    assert.equal(c.stretch[i - 1], c.stretch[i]);
    assert.deepEqual(Array.from(row(c, i)), Array.from(row(c, i - 1)));
  }
});

// ------------------------------------------------------------------ value validity

test("availability, M3 identity, integrality and bounds hold for every row", () => {
  const c = corpus({ scale: 0.1, seed: 0 });
  const mods = stats.modalities.map((m) => m.slice);
  const meta = stats.feature_meta;
  const pairs = stats.identical_feature_pairs ?? [];
  for (let i = 0; i < c.n; i++) {
    const x = row(c, i);
    for (let m = 0; m < 4; m++) {
      const [lo, hi] = mods[m];
      let any = 0;
      for (let j = lo; j < hi; j++) if (x[j] !== 0) any = 1;
      assert.equal(c.avail[i * 4 + m], m === 2 ? 1 : any, `row ${i} modality ${m}`);
    }
    assert.equal(x[32], stats.hosts[c.host[i]].host_bucket);
    assert.ok([1, 2, 3].includes(x[33]));
    for (let j = 0; j < F; j++) {
      const v = x[j];
      assert.ok(Number.isFinite(v), `row ${i} f${j} not finite`);
      assert.ok(v >= 0 && v <= Math.fround(meta[j].upper), `row ${i} f${j}=${v} outside [0, ${meta[j].upper}]`);
      if (meta[j].integer) assert.equal(v, Math.round(v), `row ${i} f${j}=${v} not an integer`);
    }
    for (const [a, b] of pairs) assert.equal(x[a], x[b]);
  }
  const buckets = new Set();
  for (let i = 0; i < c.n; i++) buckets.add(row(c, i)[32]);
  assert.deepEqual([...buckets].sort((a, b) => a - b), [10, 34, 47]);
});

test("the copula keeps each feature's marginal (zero rate, log-mean) from the stats", () => {
  // Large santos sample; firewall attacks have tau 0 and dup_rate 0 there, so
  // the synthetic marginals should sit on the exported ones.
  const c = generateCorpus(stats, { scale: 1, seed: 3, replicas: ["santos"] });
  const cs = repStats("santos").hosts["inet-firewall"].attack;
  const fw = c.hostIds.indexOf("inet-firewall");
  const mods = stats.modalities.map((m) => m.slice);
  const availCol = (j) => mods.findIndex(([lo, hi]) => j >= lo && j < hi);
  for (const j of [0, 6, 9, 11, 12, 25, 35]) {
    const f = cs.features[j];
    let present = 0;
    let zeros = 0;
    let sumLog = 0;
    for (let i = 0; i < c.n; i++) {
      if (c.host[i] !== fw || !c.y[i] || !c.avail[i * 4 + availCol(j)]) continue;
      present++;
      const v = row(c, i)[j];
      if (v === 0) zeros++;
      else sumLog += Math.log(v);
    }
    assert.ok(present > 1000);
    assert.ok(Math.abs(zeros / present - f.zero_rate) < 0.04, `${stats.feature_names[j]} zero rate ${zeros / present} vs ${f.zero_rate}`);
    if (f.zero_rate < 0.9 && !stats.feature_meta[j].integer) {
      const m = sumLog / (present - zeros);
      assert.ok(Math.abs(m - f.mu) < 0.1 + 0.1 * f.sigma, `${stats.feature_names[j]} log-mean ${m} vs ${f.mu}`);
    }
  }
});

test("normal CDF / quantile helpers are accurate and mutually inverse", () => {
  assert.ok(Math.abs(normCdf(0) - 0.5) < 1e-7);
  assert.ok(Math.abs(normCdf(1.959963984540054) - 0.975) < 1e-7);
  assert.ok(Math.abs(normInv(0.975) - 1.959963984540054) < 1e-8);
  for (const p of [1e-6, 0.01, 0.2, 0.5, 0.7, 0.99, 1 - 1e-6]) {
    assert.ok(Math.abs(normCdf(normInv(p)) - p) < 1e-7 * Math.max(1, p / (1 - p + 1e-12)) + 1e-9, `p=${p}`);
  }
});

test("sliceRows copies the selected rows", () => {
  const c = corpus({ scale: 0.05, seed: 0 });
  const idx = Int32Array.from([5, 0, c.n - 1]);
  const s = sliceRows(c, idx);
  assert.equal(s.n, 3);
  idx.forEach((i, r) => {
    assert.deepEqual(Array.from(s.X.subarray(r * F, r * F + F)), Array.from(row(c, i)));
    assert.deepEqual(Array.from(s.avail.subarray(r * 4, r * 4 + 4)), Array.from(c.avail.subarray(i * 4, i * 4 + 4)));
    assert.equal(s.y[r], c.y[i]);
    assert.equal(s.replica[r], c.replica[i]);
    assert.equal(s.host[r], c.host[i]);
  });
  s.X[0] = -1;
  assert.notEqual(c.X[5 * F], -1);
});

test("fast: the full corpus at scale 0.1 generates well under a second", () => {
  generateCorpus(stats, { scale: 0.1, seed: 11 }); // warm-up (JIT)
  const t0 = performance.now();
  const c = generateCorpus(stats, { scale: 0.1, seed: 12 });
  const ms = performance.now() - t0;
  console.log(`# generateCorpus(scale 0.1, all 8 replicas): ${c.n} rows in ${ms.toFixed(1)} ms`);
  assert.ok(ms < 1000, `${ms} ms`);
});

// ------------------------------------------------------------------ fidelity

test("generator fidelity: the REAL pretrained model scores synthetic santos like real santos", async () => {
  let metricsAp = null;
  try {
    ({ averagePrecision: metricsAp } = await import("../js/lab/metrics.js"));
  } catch {
    metricsAp = null;
  }
  const rows = [];
  for (const dependence of ["independent", "copula"]) {
    for (const seed of [0, 1, 2]) {
      const c = generateCorpus(stats, { scale: 0.1, seed, replicas: ["santos"], dependence });
      assert.equal(c.meta.dependence, dependence);
      const s = scoreWithRealModel(c);
      const ap = localAp(c.y, s);
      if (metricsAp) assert.ok(Math.abs(metricsAp(c.y, s) - ap) < 1e-12, "local AP disagrees with metrics.js");
      rows.push({ dependence, seed, n: c.n, nPositive: c.y.reduce((a, v) => a + v, 0), ap });
    }
  }
  for (const r of rows) {
    console.log(`# fidelity: synthetic santos scale 0.1 seed ${r.seed} ${r.dependence}: n=${r.n} pos=${r.nPositive} AP=${r.ap.toFixed(4)} (real santos: 0.9995)`);
  }
  const cop = rows.filter((r) => r.dependence === "copula");
  const ind = rows.filter((r) => r.dependence === "independent");
  // Thresholds are what the generator actually achieves (measured: copula
  // 0.9998 / 1.0000 / 0.9996, independent 0.9748 / 0.9695 / 0.9763).
  for (const r of cop) assert.ok(r.ap >= 0.995, `copula seed ${r.seed}: AP ${r.ap}`);
  for (const r of ind) assert.ok(r.ap >= 0.95, `independent seed ${r.seed}: AP ${r.ap}`);
  const mean = (a) => a.reduce((x, r) => x + r.ap, 0) / a.length;
  assert.ok(mean(cop) > mean(ind), "the copula should make synthetic data more faithful");
});
