"""Aggregate summary statistics of the real AIT feature cache, for the browser
Training Lab's synthetic-data generator (``web/js/lab/datagen.js``).

Why this exists. The Training Lab lets a visitor train TESSERA-base in the
browser under different split protocols and watch leakage inflate the score.
That needs data with the real corpus's structure - per-replica and per-host
prevalence, contiguous attack episodes, zero-inflated heavy-tailed features,
missing modalities, runs of exact duplicate windows - but the real rows can
never ship: the AIT Log Data Set is CC BY-NC-SA and LICENSE-DATA commits to
never redistributing derived features. So this module publishes only
aggregate statistics, and the browser samples synthetic windows from them.

What makes the output licence-safe (and what the tests assert):

* No row-level value. Every number aggregates at least ``min_support`` (20,
  ``tessera.eval.metrics.MIN_SUPPORT_FOR_RATES``) real windows. A class group
  (host x class, replica x class, or global x class) below that is emitted as
  ``null``. Inside a group, a statistic whose own support is below 20 - a
  feature that is nonzero in only a handful of windows, say - is not computed
  from those few windows: it is inherited from the next coarser group (host ->
  replica pooled -> global pooled), and the inherited feature indices are listed
  in ``inherited_features``.
* ``upper`` (the clip bound per feature) is the 20th-largest real value, not
  the maximum, so it never discloses a single window's exact value.
* No window, feature vector, log line, IP, timestamp or host name leaves this
  module - only the three generic host ROLE ids already used by the demo.

Definitions follow the web data contract ``tessera-replica-stats/v1``:

* derived availability of M1/M2/M4 = any nonzero value in that modality's slice
  (M3 is always present);
* an attack stretch is a maximal run of ``y == 1`` in one host's time-ordered
  windows (an episode); a benign stretch is a maximal run of ``y == 0`` chopped
  into blocks of ``BENIGN_BLOCK_WINDOWS`` (60) windows; a run also breaks at any
  gap in the 60 s window grid;
* per feature, over windows whose modality is present: ``zero_rate`` =
  P(x == 0), ``mu`` = mean of ln(x) over x > 0, ``sigma`` = pooled
  within-stretch std of ln(x) (floored at 1e-3), ``tau`` = between-stretch std
  of stretch-mean ln(x), a method-of-moments random-effects estimate
  ``max(0, var(stretch means) - mean(within var / n_s))`` over stretches with at
  least 5 positive values (0 when fewer than 3 such stretches exist);
* ``dup_rate`` = P(a window's 42-vector exactly equals the previous window of
  the same host in the same stretch), over windows that have such a
  predecessor.

Optional top-level ``copula`` (added so the synthetic windows keep the real
features' co-movement - sampling each feature independently put, for example, a
long mean log-line length next to a high event count, a combination real attack
minutes almost never show, and the real pretrained model scored such windows as
benign): per class, a Gaussian-copula correlation matrix over the sampled
features, estimated as the pooled within-(replica, host) correlation of normal
scores (van der Waerden: ``Phi^-1(rank / (n + 1))``, ties at mid-rank, computed
over the windows whose modality is present). Only groups of at least
``min_support`` windows contribute, a pair needs ``min_support`` co-present
windows (else 0), the matrix is projected to positive definite, and entries are
rounded to ``COPULA_DECIMALS`` places. One aggregate matrix per class - no
per-window value. Columns identical to an earlier column
(``identical_feature_pairs``) are left out; the generator copies them.
"""

from __future__ import annotations

import math

import numpy as np

from tessera.data.ait.unpack import REPLICAS
from tessera.demo.synthetic_demo_data import DEMO_HOST_LABELS, DEMO_HOSTS
from tessera.eval import leakage
from tessera.eval.metrics import MIN_SUPPORT_FOR_RATES
from tessera.features.m1_log import M1_FEATURE_NAMES
from tessera.features.m2_metrics import FEATURE_NAMES as M2_FEATURE_NAMES
from tessera.features.m3_identity import M3_FEATURE_NAMES, host_bucket
from tessera.features.m4_graph import M4_FEATURE_NAMES
from tessera.features.pipeline import (
    M1_SLICE,
    M2_SLICE,
    M3_SLICE,
    M4_SLICE,
    N_TOTAL_FEATURES,
)

SCHEMA = "tessera-replica-stats/v1"
GENERATED_BY = "uv run python -m tessera.demo.export_web_data"
BENIGN_BLOCK_WINDOWS = 60
WINDOW_SECONDS = 60
SIGMA_FLOOR = 1e-3
TAU_MIN_VALUES_PER_STRETCH = 5
TAU_MIN_STRETCHES = 3
SIGNIFICANT_DIGITS = 6
COPULA_DECIMALS = 2
COPULA_MIN_EIGENVALUE = 1e-2

FEATURE_NAMES = (*M1_FEATURE_NAMES, *M2_FEATURE_NAMES, *M3_FEATURE_NAMES, *M4_FEATURE_NAMES)
assert len(FEATURE_NAMES) == N_TOTAL_FEATURES

MODALITIES = (
    ("m1_log", "Log templates", M1_SLICE),
    ("m2_metrics", "Network metrics", M2_SLICE),
    ("m3_identity", "Host identity", M3_SLICE),
    ("m4_graph", "Graph structure", M4_SLICE),
)
HOST_BUCKET_COL = M3_SLICE.start
N_SOURCES_COL = M3_SLICE.start + 1

# Derived-availability columns, in contract order [M1, M2, M4] (M3 is always present).
_AVAIL_SLICES = (M1_SLICE, M2_SLICE, M4_SLICE)
# feature index -> the derived-availability column that governs it (None for M3).
_AVAIL_COL = [None] * N_TOTAL_FEATURES
for _c, _sl in enumerate(_AVAIL_SLICES):
    for _j in range(_sl.start, _sl.stop):
        _AVAIL_COL[_j] = _c

LICENCE_NOTE = (
    "Aggregate summary statistics of the AIT Log Data Set V2.1 (Landauer et al., "
    "Zenodo record 19483937, CC BY-NC-SA 4.0) - summary statistics only; no row-level "
    "data. No window, feature vector, log line, IP address, timestamp or host name is "
    "included (only the three generic host role ids). Every statistic aggregates at "
    "least min_support real windows: smaller groups are null, and a statistic with "
    "smaller support is inherited from the next coarser group (listed in "
    "inherited_features). 'upper' is the min_support-th largest real value, not the "
    "maximum, so no single window's value is disclosed. See LICENSE-DATA."
)


# ---------------------------------------------------------------- helpers


def round_sig(obj, digits: int = SIGNIFICANT_DIGITS):
    """Recursively round every float to ``digits`` significant digits.

    Whole-number floats become ints (``1.0`` -> ``1``) - JavaScript has no
    int/float distinction, and it keeps the JSON short. Non-finite floats are
    refused rather than written as the invalid-JSON tokens NaN/Infinity.
    """
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        if not math.isfinite(x):
            raise ValueError(f"non-finite value {x!r} cannot be exported as JSON")
        r = float(f"{x:.{digits}g}")
        return int(r) if r.is_integer() and abs(r) < 1e15 else r
    if isinstance(obj, dict):
        return {k: round_sig(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_sig(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return [round_sig(v, digits) for v in obj.tolist()]
    raise TypeError(f"cannot export {type(obj).__name__}")


def derived_availability(X: np.ndarray) -> np.ndarray:
    """(n, 3) bool: any nonzero value in the M1, M2, M4 slices."""
    return np.stack([(X[:, sl] != 0).any(axis=1) for sl in _AVAIL_SLICES], axis=1)


def _default_feature() -> dict:
    """The contract's value for a feature with no positive values in a class."""
    return {"zero_rate": 1.0, "mu": 0.0, "sigma": SIGMA_FLOOR, "tau": 0.0}


def _log_moments(lx: np.ndarray, sid: np.ndarray) -> tuple[float, float, float]:
    """(mu, sigma, tau) of log-values ``lx`` grouped by stretch id ``sid``."""
    mu = float(lx.mean())
    _, inv = np.unique(sid, return_inverse=True)
    cnt = np.bincount(inv).astype(np.float64)
    means = np.bincount(inv, weights=lx) / cnt
    ss = np.bincount(inv, weights=(lx - means[inv]) ** 2)
    dof = float((cnt - 1).sum())
    # Pooled within-stretch variance. If every stretch holds a single positive value
    # there is no within-stretch information at all; fall back to the total variance
    # (tau is then 0 below, so the total spread is still represented once).
    within_var = float(ss.sum() / dof) if dof > 0 else float(lx.var())
    sigma = max(SIGMA_FLOOR, math.sqrt(max(within_var, 0.0)))

    big = cnt >= TAU_MIN_VALUES_PER_STRETCH
    tau = 0.0
    if int(big.sum()) >= TAU_MIN_STRETCHES:
        m = means[big]
        wv = ss[big] / (cnt[big] - 1)
        tau2 = float(m.var(ddof=1) - np.mean(wv / cnt[big]))
        tau = math.sqrt(tau2) if tau2 > 0 else 0.0
    return mu, sigma, tau


# ---------------------------------------------------------------- the row table


class _Table:
    """Every real window of every replica, in (replica, host, time) order, with the
    stretch structure precomputed. Internal only - never serialised."""

    def __init__(self, datasets: dict[str, dict], replicas: tuple[str, ...]):
        Xs, ys, reps, hosts, sids, firsts, dups, ws_all = [], [], [], [], [], [], [], []
        self.n_attack_episodes: dict[tuple[int, int], int] = {}
        offset = 0
        for ri, r in enumerate(replicas):
            d = datasets[r]
            X = np.asarray(d["X"], dtype=np.float64) + 0.0  # also turns -0.0 into 0.0
            y = np.asarray(d["y"]).astype(np.int8).ravel()
            host = np.asarray(d["host"]).astype(str)
            ws = np.asarray(d["window_start"], dtype=np.int64).ravel()
            if X.shape != (len(y), N_TOTAL_FEATURES):
                raise ValueError(f"{r}: X has shape {X.shape}, expected (n, {N_TOTAL_FEATURES})")
            unknown = sorted(set(host.tolist()) - set(DEMO_HOSTS))
            if unknown:
                raise ValueError(f"{r}: hosts {unknown} are not among {DEMO_HOSTS}")
            for hi, h in enumerate(DEMO_HOSTS):
                idx = np.flatnonzero(host == h)
                if idx.size == 0:
                    continue
                idx = idx[np.argsort(ws[idx], kind="stable")]
                Xh, yh, wh = X[idx], y[idx], ws[idx]
                n = idx.size
                run_break = np.ones(n, dtype=bool)
                run_break[1:] = (yh[1:] != yh[:-1]) | (np.diff(wh) != WINDOW_SECONDS)
                run_id = np.cumsum(run_break) - 1
                pos_in_run = np.arange(n) - np.flatnonzero(run_break)[run_id]
                first = run_break | ((yh == 0) & (pos_in_run % BENIGN_BLOCK_WINDOWS == 0))
                stretch = offset + np.cumsum(first) - 1
                offset = int(stretch[-1]) + 1
                dup = np.zeros(n, dtype=bool)
                dup[1:] = (Xh[1:] == Xh[:-1]).all(axis=1) & ~first[1:]
                self.n_attack_episodes[(ri, hi)] = int((run_break & (yh == 1)).sum())

                Xs.append(Xh)
                ys.append(yh)
                reps.append(np.full(n, ri, dtype=np.int16))
                hosts.append(np.full(n, hi, dtype=np.int16))
                sids.append(stretch)
                firsts.append(first)
                dups.append(dup)
                ws_all.append(wh)

        self.X = np.concatenate(Xs, axis=0)
        self.y = np.concatenate(ys)
        self.replica = np.concatenate(reps)
        self.host = np.concatenate(hosts)
        self.stretch = np.concatenate(sids)
        self.first = np.concatenate(firsts)
        self.dup_prev = np.concatenate(dups)
        self.window_start = np.concatenate(ws_all)
        self.avail = derived_availability(self.X)


def _class_stats(tab: _Table, idx: np.ndarray, parent: dict | None, min_support: int):
    """ClassStats for the windows ``idx`` (all one class), or None below min_support.

    ``parent`` is the already-resolved ClassStats of the next coarser group of the
    same class (None at the global level); statistics with too little support of
    their own are inherited from it instead of being computed from a few windows.
    """
    n = int(idx.size)
    if n < min_support:
        return None
    Xg, ag, sg = tab.X[idx], tab.avail[idx], tab.stretch[idx]

    nsrc = np.rint(Xg[:, N_SOURCES_COL]).astype(np.int64)
    code = ag[:, 0].astype(np.int64) + 2 * ag[:, 1] + 4 * ag[:, 2] + 8 * nsrc
    codes, counts = np.unique(code, return_counts=True)
    patterns = []
    for k in np.lexsort((codes, -counts)):  # most frequent first, ties by code
        c = int(codes[k])
        patterns.append(
            {
                "a": [c & 1, (c >> 1) & 1, (c >> 2) & 1],
                "n_sources": c >> 3,
                "p": float(counts[k] / n),
            }
        )

    features: list[dict | None] = []
    inherited: list[int] = []
    for j in range(N_TOTAL_FEATURES):
        col = _AVAIL_COL[j]
        if col is None:
            features.append(None)
            continue
        present = ag[:, col]
        n_present = int(present.sum())
        x = Xg[present, j]
        pos = x > 0
        n_pos = int(pos.sum())
        pf = parent["features"][j] if parent is not None else None

        if n_present < min_support:
            # Too few windows even to estimate how often the feature is zero.
            if pf is None:
                entry = _default_feature()
            else:
                entry = dict(pf)
                inherited.append(j)
        elif n_pos == 0:
            entry = _default_feature()
        elif n_pos < min_support:
            # zero_rate is an aggregate over >= min_support windows; the log-moments
            # would be computed from fewer, so they come from the coarser group.
            if pf is None or pf["zero_rate"] >= 1.0:
                entry = _default_feature()
            else:
                entry = {
                    "zero_rate": 1.0 - n_pos / n_present,
                    "mu": pf["mu"],
                    "sigma": pf["sigma"],
                    "tau": pf["tau"],
                }
                inherited.append(j)
        else:
            mu, sigma, tau = _log_moments(np.log(x[pos]), sg[present][pos])
            entry = {"zero_rate": 1.0 - n_pos / n_present, "mu": mu, "sigma": sigma, "tau": tau}
        features.append(entry)

    nonfirst = ~tab.first[idx]
    n_nonfirst = int(nonfirst.sum())
    dup_inherited = False
    if n_nonfirst >= min_support:
        dup_rate = float(tab.dup_prev[idx][nonfirst].mean())
    elif parent is not None:
        dup_rate, dup_inherited = float(parent["dup_rate"]), True
    else:
        dup_rate = 0.0

    return {
        "n": n,
        "patterns": patterns,
        "features": features,
        "dup_rate": dup_rate,
        "inherited_features": inherited,
        "dup_rate_inherited": dup_inherited,
    }


def _feature_meta(X: np.ndarray, min_support: int) -> list[dict]:
    k = min(min_support, X.shape[0])
    # k-th largest per column: a value reached by at least k real windows.
    upper = -np.partition(-X, k - 1, axis=0)[k - 1]
    is_int = (np.round(X) == X).all(axis=0)
    meta = []
    for j, name in enumerate(FEATURE_NAMES):
        modality = next(mid for mid, _, sl in MODALITIES if sl.start <= j < sl.stop)
        kind = (
            "host_bucket"
            if j == HOST_BUCKET_COL
            else "n_sources"
            if j == N_SOURCES_COL
            else "continuous"
        )
        meta.append(
            {
                "name": name,
                "modality": modality,
                "integer": bool(is_int[j]),
                "upper": float(upper[j]),
                "kind": kind,
            }
        )
    return meta


def _identical_feature_pairs(X: np.ndarray) -> list[list[int]]:
    """Column pairs equal in every real window (e.g. two names for one count).

    Constant columns are skipped: two all-zero columns are trivially "identical" and
    say nothing about the features (the real cache has none, but a small or
    filtered input can)."""
    varying = [j for j in range(X.shape[1]) if X[:, j].min() != X[:, j].max()]
    pairs = []
    for a, i in enumerate(varying):
        for j in varying[a + 1 :]:
            if np.array_equal(X[:, i], X[:, j]):
                pairs.append([i, j])
    return pairs


def _copula(tab: _Table, identical_pairs: list[list[int]], min_support: int) -> dict:
    """Pooled within-(replica, host) normal-score correlation per class (see the
    module docstring). Deterministic; aggregates only."""
    from scipy.special import ndtri
    from scipy.stats import rankdata

    copies = {b for _, b in identical_pairs}
    cols = [j for j in range(N_TOTAL_FEATURES) if _AVAIL_COL[j] is not None and j not in copies]
    k = len(cols)
    out: dict = {
        "method": "gaussian copula; pooled within-(replica, host) correlation of normal scores",
        "feature_index": cols,
    }
    for c, name in ((0, "benign"), (1, "attack")):
        S = np.zeros((k, k))
        N = np.zeros((k, k))
        n_groups = 0
        for ri in np.unique(tab.replica):
            for hi in np.unique(tab.host):
                idx = np.flatnonzero((tab.replica == ri) & (tab.host == hi) & (tab.y == c))
                if idx.size < min_support:
                    continue
                n_groups += 1
                Z = np.zeros((idx.size, k))
                M = np.zeros((idx.size, k))
                for a, j in enumerate(cols):
                    present = tab.avail[idx, _AVAIL_COL[j]]
                    x = tab.X[idx[present], j]
                    if x.size < min_support or x.min() == x.max():
                        continue
                    z = ndtri(rankdata(x) / (x.size + 1))
                    sd = z.std()
                    if not sd > 0:
                        continue
                    Z[present, a] = (z - z.mean()) / sd
                    M[present, a] = 1.0
                S += Z.T @ Z
                N += M.T @ M
        C = np.where(min_support <= N, S / np.maximum(N, 1.0), 0.0)
        C = (C + C.T) / 2
        np.fill_diagonal(C, 1.0)
        w, V = np.linalg.eigh(C)
        C = (V * np.maximum(w, COPULA_MIN_EIGENVALUE)) @ V.T
        d = np.sqrt(np.diag(C))
        C = np.clip(C / np.outer(d, d), -1.0, 1.0)
        out[name] = {
            "n_groups": n_groups,
            # lower triangle, row a holds a+1 entries (the last is the diagonal 1)
            "lower": [
                [round(float(C[a, b]), COPULA_DECIMALS) for b in range(a + 1)] for a in range(k)
            ],
        }
    return out


# ---------------------------------------------------------------- public API


def load_real_datasets(replicas: tuple[str, ...] = REPLICAS) -> dict[str, dict]:
    """The real (host, window) feature cache for each replica (gitignored, local)."""
    from tessera.eval.loro_real import build_or_load_replica_dataset

    return {r: build_or_load_replica_dataset(r) for r in replicas}


def compute_replica_stats(
    datasets: dict[str, dict] | None = None,
    *,
    replicas: tuple[str, ...] = REPLICAS,
    min_support: int = MIN_SUPPORT_FOR_RATES,
) -> dict:
    """The ``tessera-replica-stats/v1`` payload, floats rounded for export.

    ``datasets`` maps replica id -> {X (n, 42), y, host, window_start}, as returned
    by ``tessera.eval.loro_real.build_or_load_replica_dataset``; loaded from the
    real cache when omitted. Deterministic for identical inputs.
    """
    replicas = tuple(replicas)
    if datasets is None:
        datasets = load_real_datasets(replicas)
    missing = [r for r in replicas if r not in datasets]
    if missing:
        raise ValueError(f"datasets missing replicas {missing}")
    tab = _Table(datasets, replicas)
    identical_pairs = _identical_feature_pairs(tab.X)

    rows_by_class = {c: tab.y == c for c in (0, 1)}
    cls_name = {0: "benign", 1: "attack"}
    global_pooled = {
        cls_name[c]: _class_stats(tab, np.flatnonzero(m), None, min_support)
        for c, m in rows_by_class.items()
    }

    replica_out = []
    for ri, r in enumerate(replicas):
        in_r = tab.replica == ri
        n_r = int(in_r.sum())
        pos_r = int(tab.y[in_r].sum())
        ws_r = tab.window_start[in_r]
        pooled = {
            cls_name[c]: _class_stats(
                tab, np.flatnonzero(in_r & m), global_pooled[cls_name[c]], min_support
            )
            for c, m in rows_by_class.items()
        }
        hosts_out = {}
        for hi, h in enumerate(DEMO_HOSTS):
            in_h = in_r & (tab.host == hi)
            n_h = int(in_h.sum())
            if n_h == 0:
                continue
            hosts_out[h] = {
                "n_windows": n_h,
                "n_positive": int(tab.y[in_h].sum()),
                "n_attack_episodes": tab.n_attack_episodes.get((ri, hi), 0),
                **{
                    cls_name[c]: _class_stats(
                        tab, np.flatnonzero(in_h & m), pooled[cls_name[c]], min_support
                    )
                    for c, m in rows_by_class.items()
                },
            }
        replica_out.append(
            {
                "id": r,
                "n_windows": n_r,
                "n_positive": pos_r,
                "prevalence": pos_r / n_r if n_r else 0.0,
                "span_hours": float(ws_r.max() - ws_r.min() + WINDOW_SECONDS) / 3600.0,
                # Same definition as the leakage certificate's control 1 (rows beyond
                # the first copy of each distinct 42-vector, as a fraction of all rows).
                "exact_duplicate_rate": leakage.exact_duplicates(
                    np.asarray(datasets[r]["X"])
                ).duplicate_fraction,
                "hosts": hosts_out,
                "pooled": pooled,
            }
        )

    payload = {
        "schema": SCHEMA,
        "generated_by": GENERATED_BY,
        "licence_note": LICENCE_NOTE,
        "min_support": min_support,
        "benign_block_windows": BENIGN_BLOCK_WINDOWS,
        "window_seconds": WINDOW_SECONDS,
        "n_features": N_TOTAL_FEATURES,
        "feature_names": list(FEATURE_NAMES),
        "modalities": [
            {"id": mid, "label": label, "slice": [sl.start, sl.stop]}
            for mid, label, sl in MODALITIES
        ],
        "feature_meta": _feature_meta(tab.X, min_support),
        "identical_feature_pairs": identical_pairs,
        "hosts": [
            {"id": h, "label": label, "host_bucket": host_bucket(h)}
            for h, label in zip(DEMO_HOSTS, DEMO_HOST_LABELS, strict=True)
        ],
        "replicas": replica_out,
        "global_pooled": global_pooled,
        "copula": _copula(tab, identical_pairs, min_support),
    }
    return round_sig(payload)


def iter_class_stats(payload: dict):
    """Yield (where, ClassStats-or-None) for every ClassStats slot in a payload."""
    for rep in payload["replicas"]:
        for h, hs in rep["hosts"].items():
            for c in ("benign", "attack"):
                yield f"{rep['id']}/{h}/{c}", hs[c]
        for c in ("benign", "attack"):
            yield f"{rep['id']}/pooled/{c}", rep["pooled"][c]
    for c in ("benign", "attack"):
        yield f"global/{c}", payload["global_pooled"][c]
