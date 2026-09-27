// Synthetic AIT-like corpus for the Training Lab.
//
// Real AIT rows can never ship (CC BY-NC-SA; see LICENSE-DATA.md), so the lab
// trains on windows SAMPLED from aggregate statistics of the real cache
// (web/data/replica_stats.json, written by
// `uv run python -m tessera.demo.export_web_data`, every statistic aggregating
// >= 20 real windows). Everything this module produces is synthetic.
//
// What the generator reproduces, per replica and per host:
//   * the window count and attack count (times `scale`), with attacks arranged
//     in the real number of contiguous episodes;
//   * which modalities are present (joint availability / n_sources patterns);
//   * per feature: zero-inflation, a log-normal body (mu, sigma), and a
//     per-stretch random offset (tau) so neighbouring windows resemble each
//     other the way consecutive real minutes do;
//   * runs of exact duplicate windows (dup_rate);
//   * columns that are identical in every real window stay identical;
//   * when stats.copula is present (default), the features' co-movement: one
//     correlated latent normal per feature (a Gaussian copula fitted to the real
//     pooled within-host rank correlations) decides both whether the feature is
//     zero and how large it is, so every marginal above is kept exactly while
//     e.g. "long log lines" and "many events" no longer co-occur at random.
//     Measured effect (web/tests/datagen.test.mjs, synthetic santos, scale 0.1,
//     seed 0, scored by the real pretrained model): AP 0.9748 with independent
//     features -> 0.9998 with the copula (the model scores real santos 0.9995).
//     Pass {dependence: 'independent'} to reproduce the plain version.
//
// Deterministic for (stats, scale, seed, replicas): every replica draws from its
// own forked stream, so choosing a subset of replicas never changes any
// replica's rows. Pure: no DOM, no fetch.

import { makeRng } from "./rng.js";

export const GENERATOR = "tessera-synth/v1";
const N_FEATURES = 42;
const N_AVAIL = 4;
// Stretch ids are offset per replica (by its index in stats.replicas) so they are
// globally unique AND identical whichever subset of replicas is generated.
const STRETCH_ID_STRIDE = 1 << 22;

function stochasticRound(x, rng) {
  const f = Math.floor(x);
  return f + (rng.next() < x - f ? 1 : 0);
}

function clampInt(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/** k distinct sorted integers from [1, m] (Floyd's algorithm; k <= m). */
function sampleDistinctSorted(m, k, rng) {
  const chosen = new Set();
  for (let j = m - k + 1; j <= m; j++) {
    const t = 1 + rng.int(j);
    chosen.add(chosen.has(t) ? j : t);
  }
  return Array.from(chosen).sort((a, b) => a - b);
}

/** Random composition of `total` into `parts` positive integers (total >= parts >= 1). */
function positiveComposition(total, parts, rng) {
  if (parts === 1) return [total];
  const cuts = sampleDistinctSorted(total - 1, parts - 1, rng);
  const out = [];
  let prev = 0;
  for (const c of cuts) {
    out.push(c - prev);
    prev = c;
  }
  out.push(total - prev);
  return out;
}

/** Random weak composition of `total` into `parts` non-negative integers. */
function weakComposition(total, parts, rng) {
  // Stars and bars: a positive composition of total + parts, minus one each.
  return positiveComposition(total + parts, parts, rng).map((v) => v - 1);
}

/** ClassStats with the contract's fallback: host -> replica pooled -> global pooled. */
function resolveClassStats(stats, rep, hostId, cls) {
  const h = rep.hosts?.[hostId];
  return h?.[cls] ?? rep.pooled?.[cls] ?? stats.global_pooled?.[cls] ?? null;
}

// ---- Gaussian copula helpers (optional stats.copula)

/** Standard normal CDF via erfc (Numerical Recipes erfcc, |rel err| < 1.2e-7). */
export function normCdf(x) {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))))
    );
  return x >= 0 ? 1 - 0.5 * r : 0.5 * r;
}

/** Standard normal quantile (Acklam's rational approximation, |rel err| < 1.2e-9). */
export function normInv(p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/**
 * Cholesky factor (row-major k*k lower triangle) of a correlation matrix given as
 * lower-triangle rows (row i has i+1 entries, the last being the diagonal 1).
 * A tiny ridge keeps a rounded, barely-positive-definite matrix factorisable.
 */
function choleskyFromLowerRows(rows) {
  const k = rows.length;
  const A = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    for (let j = 0; j <= i; j++) {
      const v = i === j ? 1 : rows[i][j];
      A[i * k + j] = v;
      A[j * k + i] = v;
    }
  }
  for (let ridge = 0; ridge <= 0.5; ridge = ridge ? ridge * 4 : 1e-6) {
    const L = new Float64Array(k * k);
    let ok = true;
    for (let i = 0; i < k && ok; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = A[i * k + j] + (i === j ? ridge : 0);
        for (let m = 0; m < j; m++) sum -= L[i * k + m] * L[j * k + m];
        if (i === j) {
          if (!(sum > 0)) {
            ok = false;
            break;
          }
          L[i * k + i] = Math.sqrt(sum);
        } else {
          L[i * k + j] = sum / L[j * k + j];
        }
      }
    }
    if (ok) {
      // rescale rows so every latent keeps unit variance despite the ridge
      for (let i = 0; i < k; i++) {
        let ss = 0;
        for (let j = 0; j <= i; j++) ss += L[i * k + j] * L[i * k + j];
        const f = 1 / Math.sqrt(ss);
        for (let j = 0; j <= i; j++) L[i * k + j] *= f;
      }
      return L;
    }
  }
  return null; // not factorisable: caller falls back to independent sampling
}

function compileCopula(copula) {
  if (!copula || !Array.isArray(copula.feature_index)) return null;
  const k = copula.feature_index.length;
  const out = { features: Int16Array.from(copula.feature_index), k, L: {} };
  for (const cls of ["benign", "attack"]) {
    const rows = copula[cls]?.lower;
    if (!Array.isArray(rows) || rows.length !== k) continue;
    const L = choleskyFromLowerRows(rows);
    if (L) out.L[cls === "attack" ? 1 : 0] = L;
  }
  return out;
}

/**
 * Precompute everything the inner loop needs for one ClassStats: flat arrays of
 * zero rate / mu / sigma / tau and the pattern table.
 */
function compileClassStats(cs) {
  const zero = new Float64Array(N_FEATURES).fill(1);
  const mu = new Float64Array(N_FEATURES);
  const sigma = new Float64Array(N_FEATURES);
  const tau = new Float64Array(N_FEATURES);
  for (let j = 0; j < N_FEATURES; j++) {
    const f = cs.features[j];
    if (!f) continue;
    zero[j] = f.zero_rate;
    mu[j] = f.mu;
    sigma[j] = f.sigma;
    tau[j] = f.tau;
  }
  const probs = cs.patterns.map((p) => p.p);
  const patterns = cs.patterns.map((p) => ({ a: p.a, nSources: p.n_sources }));
  // latent threshold below which a copula-driven feature is zero: P(z < c) = zero_rate
  const zeroCut = Float64Array.from(zero, (z) => (z <= 0 ? -Infinity : z >= 1 ? Infinity : normInv(z)));
  return { zero, zeroCut, mu, sigma, tau, probs, patterns, dupRate: cs.dup_rate ?? 0 };
}

/**
 * generateCorpus(stats, {scale = 0.05, seed = 0, replicas = all ids,
 *                        dependence = 'copula' | 'independent'}) -> corpus
 * {n, nFeatures: 42, X: Float32Array(n*42), y: Uint8Array(n), avail: Uint8Array(n*4),
 *  replica, host: Uint8Array(n), t, stretch: Int32Array(n), isDuplicate: Uint8Array(n),
 *  replicaIds, hostIds, hostLabels, featureNames,
 *  meta: {scale, seed, generator, dependence, perReplica: [{id, n, nPositive, nEpisodes}]}}
 * Rows are ordered replica (stats order) -> host (stats.hosts order) -> t.
 * `replicaIds` lists the generated replicas in stats order; `stretch` ids are
 * unique across the whole corpus and do not depend on which replicas were chosen.
 */
export function generateCorpus(stats, { scale = 0.05, seed = 0, replicas = null, dependence = "copula" } = {}) {
  if (!stats || !Array.isArray(stats.replicas)) throw new Error("generateCorpus: stats.replicas missing");
  if (!(scale > 0) || !Number.isFinite(scale)) throw new Error("generateCorpus: scale must be a positive number");
  const allIds = stats.replicas.map((r) => r.id);
  const wanted = replicas == null ? new Set(allIds) : new Set(replicas);
  for (const id of wanted) {
    if (!allIds.includes(id)) throw new Error(`generateCorpus: unknown replica "${id}"`);
  }
  const replicaIds = allIds.filter((id) => wanted.has(id));
  const hostIds = stats.hosts.map((h) => h.id);
  const hostLabels = stats.hosts.map((h) => h.label);
  const meta = stats.feature_meta;
  const upper = Float64Array.from(meta, (m) => m.upper);
  const isInt = Uint8Array.from(meta, (m) => (m.integer ? 1 : 0));
  const hbCol = meta.findIndex((m) => m.kind === "host_bucket");
  const nsCol = meta.findIndex((m) => m.kind === "n_sources");
  const mods = stats.modalities.map((m) => m.slice);
  // [start, stop) of the three sampled modalities, in pattern order [M1, M2, M4].
  const sampled = [mods[0], mods[1], mods[3]];
  // Columns that are identical in every real window: copy instead of sampling twice.
  const copyFrom = new Int16Array(N_FEATURES).fill(-1);
  for (const [a, b] of stats.identical_feature_pairs ?? []) copyFrom[b] = a;
  const blockLen = Math.max(2, Math.round(stats.benign_block_windows * scale));
  // Optional between-feature dependence (a Gaussian copula fitted to the real
  // cache's pooled within-host rank correlations). Without it, features are
  // sampled independently given the class - the plain contract.
  const cop = dependence === "copula" ? compileCopula(stats.copula) : null;
  const latentOf = new Int16Array(N_FEATURES).fill(-1);
  if (cop) cop.features.forEach((j, i) => (latentOf[j] = i));
  const eps = new Float64Array(cop ? cop.k : 0);
  const zLat = new Float64Array(cop ? cop.k : 0);
  const root = makeRng(seed, "datagen");

  // ---- pass 1: per replica, lay out each host's timeline (stretches + labels)
  const plans = [];
  let n = 0;
  for (const rid of replicaIds) {
    const repIndex = allIds.indexOf(rid);
    const rep = stats.replicas[repIndex];
    const rng = root.fork(rid);
    const hostPlans = [];
    let stretchLocal = 0;
    let nPos = 0;
    let nEp = 0;
    let nRep = 0;
    stats.hosts.forEach((host, hi) => {
      const hs = rep.hosts?.[host.id];
      if (!hs || !(hs.n_windows > 0)) return;
      const T = Math.max(4, Math.round(hs.n_windows * scale));
      const A = clampInt(stochasticRound(hs.n_positive * scale, rng), 0, T - 1);
      const E = A > 0 ? clampInt(hs.n_attack_episodes, 1, A) : 0;
      const epLens = E > 0 ? positiveComposition(A, E, rng) : [];
      const gaps = weakComposition(T - A, E + 1, rng);
      // segments: [{len, attack, stretchId}] in time order
      const segs = [];
      const pushBenign = (len) => {
        for (let s = 0; s < len; s += blockLen) {
          segs.push({ len: Math.min(blockLen, len - s), attack: 0, stretch: repIndex * STRETCH_ID_STRIDE + stretchLocal++ });
        }
      };
      for (let e = 0; e <= E; e++) {
        pushBenign(gaps[e]);
        if (e < E) segs.push({ len: epLens[e], attack: 1, stretch: repIndex * STRETCH_ID_STRIDE + stretchLocal++ });
      }
      hostPlans.push({ hi, host, segs, T });
      nPos += A;
      nEp += E;
      nRep += T;
    });
    plans.push({ rid, repIndex, rep, rng, hostPlans, n: nRep, nPositive: nPos, nEpisodes: nEp });
    n += nRep;
  }

  const X = new Float32Array(n * N_FEATURES);
  const y = new Uint8Array(n);
  const avail = new Uint8Array(n * N_AVAIL);
  const replica = new Uint8Array(n);
  const hostArr = new Uint8Array(n);
  const t = new Int32Array(n);
  const stretch = new Int32Array(n);
  const isDuplicate = new Uint8Array(n);
  const delta = new Float64Array(N_FEATURES);

  // ---- pass 2: sample the rows
  let row = 0;
  plans.forEach((plan, ri) => {
    const { rep, rng } = plan;
    for (const hp of plan.hostPlans) {
      const compiled = {
        0: compileFor(resolveClassStats(stats, rep, hp.host.id, "benign")),
        1: compileFor(resolveClassStats(stats, rep, hp.host.id, "attack")),
      };
      let tt = 0;
      for (const seg of hp.segs) {
        const cs = compiled[seg.attack];
        if (!cs) throw new Error(`generateCorpus: no ${seg.attack ? "attack" : "benign"} statistics for ${plan.rid}/${hp.host.id}`);
        for (let j = 0; j < N_FEATURES; j++) delta[j] = cs.tau[j] > 0 ? cs.tau[j] * rng.normal() : 0;
        for (let w = 0; w < seg.len; w++, row++, tt++) {
          const o = row * N_FEATURES;
          y[row] = seg.attack;
          replica[row] = ri;
          hostArr[row] = hp.hi;
          t[row] = tt;
          stretch[row] = seg.stretch;
          if (w > 0 && cs.dupRate > 0 && rng.next() < cs.dupRate) {
            X.copyWithin(o, o - N_FEATURES, o);
            isDuplicate[row] = 1;
          } else {
            const pat = cs.patterns[rng.pick(cs.probs)];
            const L = cop ? cop.L[seg.attack] : null;
            if (L) {
              const k = cop.k;
              for (let i = 0; i < k; i++) eps[i] = rng.normal();
              for (let i = 0; i < k; i++) {
                let acc = 0;
                const ro = i * k;
                for (let q = 0; q <= i; q++) acc += L[ro + q] * eps[q];
                zLat[i] = acc;
              }
            }
            for (let m = 0; m < 3; m++) {
              if (!pat.a[m]) continue; // absent modality stays all-zero
              const [lo, hi] = sampled[m];
              for (let j = lo; j < hi; j++) {
                if (copyFrom[j] >= 0) continue;
                let w;
                const li = L ? latentOf[j] : -1;
                if (li >= 0) {
                  // copula: one latent normal decides both "is it zero" and "how big"
                  const z = zLat[li];
                  const zr = cs.zero[j];
                  if (z < cs.zeroCut[j]) continue;
                  if (zr <= 0) w = z;
                  else {
                    const q = (normCdf(z) - zr) / (1 - zr);
                    w = normInv(Math.min(1 - 1e-12, Math.max(1e-12, q)));
                  }
                } else {
                  if (rng.next() < cs.zero[j]) continue;
                  w = rng.normal();
                }
                let v = Math.exp(cs.mu[j] + delta[j] + cs.sigma[j] * w);
                if (isInt[j]) v = Math.max(1, Math.round(v));
                if (v > upper[j]) v = upper[j];
                X[o + j] = v;
              }
            }
            for (let j = 0; j < N_FEATURES; j++) if (copyFrom[j] >= 0) X[o + j] = X[o + copyFrom[j]];
            if (hbCol >= 0) X[o + hbCol] = hp.host.host_bucket;
            if (nsCol >= 0) X[o + nsCol] = pat.nSources;
          }
          // derived availability from the final row (M3 is always present)
          const ao = row * N_AVAIL;
          for (let m = 0; m < N_AVAIL; m++) {
            if (m === 2) {
              avail[ao + m] = 1;
              continue;
            }
            const [lo, hi] = mods[m];
            let any = 0;
            for (let j = lo; j < hi; j++) {
              if (X[o + j] !== 0) {
                any = 1;
                break;
              }
            }
            avail[ao + m] = any;
          }
        }
      }
    }
  });

  function compileFor(cs) {
    return cs ? compileClassStats(cs) : null;
  }

  return {
    n,
    nFeatures: N_FEATURES,
    X,
    y,
    avail,
    replica,
    host: hostArr,
    t,
    stretch,
    isDuplicate,
    replicaIds,
    hostIds,
    hostLabels,
    featureNames: stats.feature_names.slice(),
    meta: {
      scale,
      seed,
      generator: GENERATOR,
      dependence: cop ? "copula" : "independent",
      perReplica: plans.map((p) => ({ id: p.rid, n: p.n, nPositive: p.nPositive, nEpisodes: p.nEpisodes })),
    },
  };
}

/** corpusSummary(corpus) -> {n, nPositive, prevalence, perReplica:[{id, n, nPositive, prevalence, nEpisodes}]} */
export function corpusSummary(corpus) {
  let nPositive = 0;
  for (let i = 0; i < corpus.n; i++) nPositive += corpus.y[i];
  const perReplica = corpus.meta.perReplica.map((r) => ({
    id: r.id,
    n: r.n,
    nPositive: r.nPositive,
    prevalence: r.n ? r.nPositive / r.n : 0,
    nEpisodes: r.nEpisodes,
  }));
  return { n: corpus.n, nPositive, prevalence: corpus.n ? nPositive / corpus.n : 0, perReplica };
}

/** sliceRows(corpus, idx: Int32Array) -> {X, y, avail, n, replica, host} (copies). */
export function sliceRows(corpus, idx) {
  const k = idx.length;
  const F = corpus.nFeatures;
  const X = new Float32Array(k * F);
  const y = new Uint8Array(k);
  const avail = new Uint8Array(k * N_AVAIL);
  const replica = new Uint8Array(k);
  const host = new Uint8Array(k);
  for (let r = 0; r < k; r++) {
    const i = idx[r];
    X.set(corpus.X.subarray(i * F, i * F + F), r * F);
    avail.set(corpus.avail.subarray(i * N_AVAIL, i * N_AVAIL + N_AVAIL), r * N_AVAIL);
    y[r] = corpus.y[i];
    replica[r] = corpus.replica[i];
    host[r] = corpus.host[i];
  }
  return { X, y, avail, n: k, replica, host };
}
