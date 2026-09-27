"""Export the static demo site's data files into ``web/data/``.

    uv run python -m tessera.demo.export_web_data          # or: just web-data

Writes three files, every number in them either RECOMPUTED here from the local
real-data cache / chain simulator, or TRANSCRIBED from RESULTS.md /
NEGATIVE_RESULTS.md with a ``source`` naming the test or ablation that recorded
it. Nothing is invented, and each block says which of the two it is
(``provenance``).

* ``replica_stats.json`` - aggregate statistics the browser's synthetic data
  generator is calibrated from (:mod:`tessera.demo.replica_stats`). No row-level
  data; see that module's docstring for the licence argument.
* ``real_results.json`` - the project's real-data findings: the 8-fold
  leave-one-replica-out run and the dataset aggregates are recomputed from the
  cache (LightGBM, ~15 s), and so is the shortcut floor next to it (the same
  folds with a model that sees only the availability mask, only host_bucket, or
  both; ~15 s); the duplicate-leakage finding and the all-modalities
  R0/R1 comparison are recomputed on the cached russellmitchell replica with the
  exact protocol of the tests that record them (~2 s); the findings that need
  slow neural training or code that no longer exists (the calendar-feature
  ablation) are transcribed.
* ``design_notes.json`` - TESSERA's own design decisions: the same calls
  ``tessera.chainsim.report`` makes (optimiser comparison against exhaustive
  ground truth, EHO structural unreachability, segment-objective invariance with
  one fh value per NSC), plus the design decisions this project made and the
  measured rationale behind each one.

Every payload passes :func:`licence_scan` before it is written, so a list long
enough to carry row-level data cannot reach the public site by accident.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import time
from pathlib import Path

import numpy as np

from tessera.data.ait.unpack import REPLICAS
from tessera.demo.replica_stats import (
    WINDOW_SECONDS,
    compute_replica_stats,
    load_real_datasets,
    round_sig,
)
from tessera.eval.metrics import MIN_SUPPORT_FOR_RATES
from tessera.eval.stats import summarise_seeds
from tessera.paths import REPO_ROOT

WEB_DATA_DIR = REPO_ROOT / "web" / "data"
GENERATED_BY = "uv run python -m tessera.demo.export_web_data"

# The licence scan: no list in a shipped payload may be long enough to carry rows.
MAX_LIST_LEN = 64
# ...except these per-feature arrays, which must then have exactly one entry per feature.
PER_FEATURE_KEYS = ("feature_names", "feature_meta", "features")
N_FEATURES = 42

AIT_DATASET = {
    "name": "AIT Log Data Set V2.1",
    "authors": "Landauer et al.",
    "zenodo_record": 19483937,
    "licence": "CC BY-NC-SA 4.0",
}
SUBSET = (
    "3 hosts (vpn, intranet_server, inet-firewall) x 4 log sources, 60 s windows, "
    "42 features (M1 log templates 8, M2 network metrics 24, M3 host identity 2, "
    "M4 graph structure 8)"
)


# ---------------------------------------------------------------- serialisation


class LicenceScanError(ValueError):
    """A payload contains something that could carry row-level data."""


def licence_scan(obj, *, path: str = "$") -> None:
    """Refuse any list longer than MAX_LIST_LEN, except per-feature arrays of 42."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, list) and k in PER_FEATURE_KEYS and len(v) > MAX_LIST_LEN:
                raise LicenceScanError(f"{path}.{k}: per-feature list of length {len(v)}")
            if isinstance(v, list) and k in PER_FEATURE_KEYS:
                if len(v) != N_FEATURES:
                    raise LicenceScanError(f"{path}.{k}: expected {N_FEATURES} entries")
                for i, item in enumerate(v):
                    licence_scan(item, path=f"{path}.{k}[{i}]")
                continue
            licence_scan(v, path=f"{path}.{k}")
    elif isinstance(obj, list):
        if len(obj) > MAX_LIST_LEN:
            raise LicenceScanError(
                f"{path}: list of length {len(obj)} > {MAX_LIST_LEN} - row-level data "
                "must never reach the public site"
            )
        for i, item in enumerate(obj):
            licence_scan(item, path=f"{path}[{i}]")


def dumps_web(obj, *, indent: int = 1, inline_width: int = 160) -> str:
    """Compact-but-readable JSON: short containers on one line, long ones one item
    per line. ``allow_nan=False`` so an invalid JSON token can never be written."""

    def compact(o) -> str:
        return json.dumps(o, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def walk(o, depth: int) -> str:
        s = compact(o)
        if len(s) <= inline_width or not isinstance(o, (dict, list)) or not o:
            return s
        pad, pad_in = " " * (indent * depth), " " * (indent * (depth + 1))
        if isinstance(o, dict):
            body = ",\n".join(f"{pad_in}{compact(k)}:{walk(v, depth + 1)}" for k, v in o.items())
            return "{\n" + body + "\n" + pad + "}"
        body = ",\n".join(pad_in + walk(v, depth + 1) for v in o)
        return "[\n" + body + "\n" + pad + "]"

    return walk(obj, 0) + "\n"


def write_payload(path: Path, payload: dict, *, indent: int = 1, inline_width: int = 200) -> int:
    """Licence-scan, round, serialise and write; returns the byte size."""
    payload = round_sig(payload)
    licence_scan(payload)
    text = dumps_web(payload, indent=indent, inline_width=inline_width)
    json.loads(text)  # round-trip guard
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------- real_results.json


def _summary(d: dict) -> dict:
    """summarise_seeds output in the web contract's shape ({mean,std,min,max,n})."""
    return {
        "mean": d.get("mean"),
        "std": d.get("std"),
        "min": d.get("min"),
        "max": d.get("max"),
        "n": d.get("n_seeds", 0),
    }


def loro_block(datasets: dict[str, dict]) -> dict:
    """The 8-fold leave-one-replica-out run, recomputed (RESULTS.md finding 3)."""
    from tessera.eval.loro_real import run_leave_one_replica_out

    res = run_leave_one_replica_out(datasets, seed=0, n_estimators=200)
    folds = []
    for f in res["folds"]:
        n_test = f["n_test"]
        folds.append(
            {
                "held_out": f["held_out"],
                "n_train": f["n_train"],
                "n_test": n_test,
                "n_pos_test": f["n_pos_test"],
                "test_prevalence": f.get("test_prevalence", f["n_pos_test"] / max(n_test, 1)),
                # None (not NaN) when the held-out replica has a single class.
                "average_precision": f.get("average_precision"),
                "mcc": f.get("mcc"),
                "low_support": bool(f["low_support"]),
            }
        )
    s = res["summary"]
    scored = [f for f in folds if f["average_precision"] is not None]
    return {
        "provenance": "recomputed",
        "source": (
            "recomputed by tessera.demo.export_web_data from the real AIT cache: "
            "tessera.eval.loro_real.run_leave_one_replica_out(build_all_replica_datasets()) "
            "- LightGBM (tessera.models.baselines.gbdt.make_gbdt, 200 trees, seed 0) "
            "trained on 7 replicas and tested on the 8th, MCC at threshold 0.5; the same "
            "run tests/test_loro_real.py::test_leave_one_replica_out_generalises_strongly "
            "asserts on (RESULTS.md finding 3)"
        ),
        "model": "LightGBM, 200 trees, seed 0, raw 42-feature vectors",
        "folds": folds,
        "summary": {
            "n_folds": s["n_folds"],
            "n_scored_folds": s["n_scored_folds"],
            "min_support": s["min_support_for_rates"],
            "excluded_low_support": [f["held_out"] for f in folds if f["low_support"]],
            "average_precision": _summary(s["average_precision"]),
            "mcc": _summary(s["mcc"]),
            # What a naive table would report: every scored fold averaged, shaw included.
            "naive_all_folds_ap": _summary(
                summarise_seeds([f["average_precision"] for f in scored])
            ),
            "std_ddof": 1,
            "note": (
                f"Folds with fewer than {MIN_SUPPORT_FOR_RATES} held-out positives are "
                "reported in full but excluded from the summary mean/std; "
                "naive_all_folds_ap shows what averaging them in would read."
            ),
        },
    }


def dataset_block(datasets: dict[str, dict]) -> dict:
    """Per-replica aggregates of the real cache (counts, prevalence, capture span)."""
    reps = []
    for r in REPLICAS:
        d = datasets[r]
        y, ws = np.asarray(d["y"]), np.asarray(d["window_start"], dtype=np.int64)
        n, n_pos = int(y.size), int(y.sum())
        reps.append(
            {
                "id": r,
                "n_windows": n,
                "n_positive": n_pos,
                "prevalence": n_pos / n,
                "span_hours": float(ws.max() - ws.min() + WINDOW_SECONDS) / 3600.0,
            }
        )
    return {
        "provenance": "recomputed",
        "source": (
            "recomputed by tessera.demo.export_web_data from the real AIT cache "
            "(data/processed/ait/<replica>_v1.npz, built by "
            "tessera.eval.loro_real.build_all_replica_datasets); counts only"
        ),
        **AIT_DATASET,
        "subset": SUBSET,
        "window_seconds": WINDOW_SECONDS,
        "replicas": reps,
        "totals": {
            "n_windows": sum(r["n_windows"] for r in reps),
            "n_positive": sum(r["n_positive"] for r in reps),
        },
    }


def _r0_r1(X: np.ndarray, y: np.ndarray, window_start: np.ndarray) -> dict:
    """R0 random vs R1 chronological AP/MCC - the protocol of
    tests/test_pipeline_real_data.py, verbatim (seed 0, 600 s gap, 200 trees)."""
    from tessera.eval.metrics import evaluate
    from tessera.eval.splits import r0_random, r1_chronological
    from tessera.models.baselines.gbdt import make_gbdt
    from tessera.train.seed import seed_everything

    seed_everything(0)
    out = {}
    r0 = r0_random(y, seed=0)
    r1 = r1_chronological(window_start.astype(np.float64), gap_seconds=600)
    for name, split in (("r0", r0), ("r1", r1)):
        m = make_gbdt(seed=0, n_estimators=200)
        m.fit(X[split.train_idx], y[split.train_idx])
        b = evaluate(y[split.test_idx], m.predict_proba(X[split.test_idx])[:, 1])["binary"]
        out[f"{name}_ap"], out[f"{name}_mcc"] = b["average_precision"], b["mcc"]
    out["r0_split"] = r0
    return out


def leakage_blocks(datasets: dict[str, dict]) -> tuple[dict, dict]:
    """RESULTS.md findings 1 and 2 (all-modalities table), recomputed on the cached
    russellmitchell replica - the cache is built by the same build_real_dataset call
    with the same hosts and sources the tests extract, and reproduces them exactly."""
    from tessera.eval import leakage
    from tessera.features.pipeline import M2_SLICE

    d = datasets["russellmitchell"]
    X, y, ws = np.asarray(d["X"]), np.asarray(d["y"]), np.asarray(d["window_start"])

    X2 = X[:, M2_SLICE]
    m2 = _r0_r1(X2, y, ws)
    r0 = m2.pop("r0_split")
    exact = leakage.exact_duplicates(X2, y, train_idx=r0.train_idx, test_idx=r0.test_idx)
    near = leakage.near_duplicates(X2, seed=0, train_idx=r0.train_idx, test_idx=r0.test_idx)
    dup = {
        "provenance": "recomputed",
        "source": (
            "recomputed by tessera.demo.export_web_data on the cached russellmitchell "
            "replica with the protocol of tests/test_pipeline_real_data.py::"
            "test_random_split_is_inflated_by_duplicate_leakage_on_m2_alone (R0 = "
            "tessera.eval.splits.r0_random seed 0; R1 = r1_chronological, 600 s gap; "
            "LightGBM 200 trees, seed 0) and the leakage certificate's controls 1-2 "
            "(tessera.eval.leakage.exact_duplicates / near_duplicates over the R0 "
            "split, as in ::test_leakage_certificate_finds_real_duplicates); "
            "RESULTS.md finding 1"
        ),
        "replica": "russellmitchell",
        "features": "M2 only (Suricata eve.json aggregates, 24 features)",
        "n_windows": int(y.size),
        "n_test_rows": int(r0.test_idx.size),
        "r0_random_ap": m2["r0_ap"],
        "r1_chronological_ap": m2["r1_ap"],
        "exact_duplicate_rate": exact.duplicate_fraction,
        "near_duplicate_rate": near["near_duplicate_fraction"],
        "near_duplicate_method": near["method"],
        "test_rows_identical_to_train": int(exact.cross_split_twins),
    }

    full = _r0_r1(X, y, ws)
    full.pop("r0_split")
    allmod = {
        "provenance": "recomputed",
        "source": (
            "recomputed by tessera.demo.export_web_data on the cached russellmitchell "
            "replica with the protocol of tests/test_pipeline_real_data.py::"
            "test_full_feature_set_generalises_well_without_calendar_leakage (R0 random "
            "vs R1 chronological, LightGBM 200 trees, seed 0); RESULTS.md finding 2, "
            "all-four-modalities table"
        ),
        "replica": "russellmitchell",
        "features": "all four modalities (42 features)",
        "r0_ap": full["r0_ap"],
        "r0_mcc": full["r0_mcc"],
        "r1_ap": full["r1_ap"],
        "r1_mcc": full["r1_mcc"],
        "gap": full["r0_ap"] - full["r1_ap"],
    }
    return dup, allmod


def calendar_leak_block(datasets: dict[str, dict]) -> dict:
    """The transcribed calendar-feature ablation, plus a live re-check of its
    'without calendar features' row on today's cache (M1+M2+M3 = the first 34
    columns). The recorded ablation predates the current cache, and the re-check
    does not reproduce it exactly - so both are shown, each labelled, rather than
    either being silently preferred."""
    from tessera.features.pipeline import M3_SLICE

    d = datasets["russellmitchell"]
    X = np.asarray(d["X"])[:, : M3_SLICE.stop]
    r = _r0_r1(X, np.asarray(d["y"]), np.asarray(d["window_start"]))
    r.pop("r0_split")
    return {
        **CALENDAR_LEAK,
        "recheck_without_calendar": {
            "provenance": "recomputed",
            "source": (
                "recomputed by tessera.demo.export_web_data: the 'without calendar "
                "features' row re-run on today's cached russellmitchell replica, "
                "M1+M2+M3 columns only, protocol of tests/test_pipeline_real_data.py"
            ),
            "r0_ap": r["r0_ap"],
            "r1_ap": r["r1_ap"],
            "gap": r["r0_ap"] - r["r1_ap"],
            "note": (
                "The recorded rows above predate the current feature cache, so a re-run "
                "differs slightly; the with-calendar row cannot be re-run at all because "
                "the calendar features were deleted. The finding - calendar position "
                "leaks, and removing it closes most of the R0/R1 gap - does not depend "
                "on the difference."
            ),
        },
    }


# Transcribed blocks: numbers copied exactly from RESULTS.md (and NEGATIVE_RESULTS.md),
# because reproducing them needs slow neural training or code that was deliberately
# deleted. tests/test_web_exports.py checks each value still appears in RESULTS.md.

CALENDAR_LEAK = {
    "provenance": "transcribed",
    "source": (
        "transcribed from RESULTS.md (finding 2; also NEGATIVE_RESULTS.md C9 and the "
        "tessera.features.m3_identity docstring); recorded by a same-seed, same-split "
        "ablation on russellmitchell (M1+M2+M3, R0 random vs R1 chronological, LightGBM) "
        "made before the four calendar features were deleted from m3_identity.py, so no "
        "test can re-run it; the corrected state is locked in by "
        "tests/test_pipeline_real_data.py::test_calendar_features_are_not_present_in_m3 "
        "and ::test_full_feature_set_generalises_well_without_calendar_leakage"
    ),
    "replica": "russellmitchell",
    "rows": [
        {
            "features": "M1+M2+M3 with calendar features (hour of day, day of week, "
            "is weekend, minute of hour)",
            "r0_ap": 0.9998,
            "r1_ap": 0.5156,
            "gap": 0.484,
        },
        {
            "features": "M1+M2+M3 without calendar features",
            "r0_ap": 1.0000,
            "r1_ap": 0.9813,
            "gap": 0.019,
        },
    ],
    "top_feature": "hour_of_day (importance 572; next highest 410)",
    "note": "NEGATIVE_RESULTS.md C9 records the gaps to four decimals: 0.4843 and 0.0187.",
}

TESSERA_BASE = {
    "provenance": "transcribed",
    "source": (
        "transcribed from RESULTS.md (finding 4); recorded by "
        "tests/test_tessera_base.py::test_tessera_base_matches_lightgbm_on_real_data "
        "(slow: trains TESSERA-base on the 7 replicas other than santos, less a seeded "
        "15% validation carve, 30 epochs with early stopping, on MPS). The LightGBM row "
        "is the santos fold of the leave-one-replica-out run, recomputed in loro.folds."
    ),
    "held_out": "santos",
    "n_train_windows": 152629,
    "train_seconds": 82.5,
    "device": "MPS (Apple GPU)",
    "epochs": 30,
    "n_parameters": 5005,
    "tessera_ap": 0.9995,
    "tessera_mcc": 0.9969,
    "lightgbm_ap": 0.9994,
    "lightgbm_mcc": 0.9975,
}

ATTRIBUTION_VS_ABLATION = {
    "provenance": "transcribed",
    "source": (
        "transcribed from RESULTS.md (finding 4; also NEGATIVE_RESULTS.md F9); the mean "
        "attribution and the two TESSERA-base rows are recorded by "
        "tests/test_tessera_base.py::test_gmu_attribution_does_not_match_naive_ablation_importance "
        "(slow; it asserts the direction of the gap, not these exact figures); the three "
        "GBDT single-modality rows are a one-off ablation on the same santos fold that no "
        "test re-runs"
    ),
    "held_out": "santos",
    "mean_attribution": {
        "m1_log": 0.14,
        "m2_metrics": 0.02,
        "m3_identity": 0.83,
        "m4_graph": 0.01,
    },
    "ablation": [
        {
            "check": "GBDT on M1 alone",
            "ap": 0.9996,
            "note": "M1 alone carries nearly all the signal",
        },
        {"check": "GBDT on M3 alone", "ap": 0.6917, "note": ""},
        {
            "check": "GBDT on host_bucket alone (1 feature)",
            "ap": 0.4977,
            "note": "MCC = 0; prevalence floor 0.1666",
        },
        {"check": "TESSERA-base with M1 (full model)", "ap": 0.9995, "note": ""},
        {
            "check": "TESSERA-base without M1 (zeroed and marked unavailable)",
            "ap": 0.9846,
            "note": "the honest removal: M1 matters far more than its 14% gate weight says",
        },
    ],
    "finding": (
        "The gate weights say M1 14% and M3 83%, but removing M1 costs real AP and M3's "
        "host_bucket alone is barely above the prevalence floor: gate values are a signal, "
        "never ground truth for what the model used. Per-modality ablation is the reliable "
        "importance measure."
    ),
}


# ---------------------------------------------------------------- the shortcut floor

# The model mask_only_score (tessera.eval.leakage, control 4) fits: the same factory
# and size, so this block is that control run leave-one-replica-out.
FLOOR_N_ESTIMATORS = 100
FLOOR_HOST = "inet-firewall"


def availability_mask(X: np.ndarray) -> np.ndarray:
    """(n, 3) bool [M1 present, M2 present, M4 present], exactly as
    build_real_dataset's ``availability`` array, recovered from the cached features.

    The cache stores no mask, and "any nonzero value" is not enough on its own: a
    present M2 window can be all zeros (a Suricata window holding only events with no
    protocol, port or address, e.g. ``stats`` records). It is exact because M1 present
    means ``n_events >= 1``, M4 present means at least one peer (``n_unique_peers >=
    1``), and M3's ``n_sources_active`` is the row sum of the true mask - so M2 is
    what is left over.
    """
    from tessera.features.pipeline import M1_SLICE, M3_SLICE, M4_SLICE

    X = np.asarray(X)
    m1 = (X[:, M1_SLICE] != 0).any(axis=1)
    m4 = (X[:, M4_SLICE] != 0).any(axis=1)
    m2 = np.rint(X[:, M3_SLICE.start + 1]).astype(np.int64) - m1 - m4
    if not np.isin(m2, (0, 1)).all():
        raise AssertionError("n_sources_active disagrees with the M1/M4 slices")
    return np.stack([m1, m2.astype(bool), m4], axis=1)


def _loro_floor(datasets: dict[str, dict], featurise) -> dict:
    """Leave-one-replica-out AP of a LightGBM that sees only ``featurise(X)``.

    Same folds, same AP function and the same low-support rule as loro_block, so
    the summary excludes exactly the folds the headline mean excludes.
    """
    from tessera.eval.metrics import evaluate
    from tessera.models.baselines.gbdt import make_gbdt

    names = sorted(datasets)
    feats = {n: np.asarray(featurise(datasets[n]["X"]), dtype=np.float64) for n in names}
    folds = []
    for held in names:
        train = [n for n in names if n != held]
        y_test = np.asarray(datasets[held]["y"])
        n_pos = int(y_test.sum())
        ap = None
        if len(np.unique(y_test)) == 2:
            model = make_gbdt(seed=0, n_estimators=FLOOR_N_ESTIMATORS)
            model.fit(
                np.concatenate([feats[n] for n in train]),
                np.concatenate([np.asarray(datasets[n]["y"]) for n in train]),
            )
            proba = model.predict_proba(feats[held])[:, 1]
            ap = evaluate(y_test, proba)["binary"]["average_precision"]
        folds.append(
            {
                "held_out": held,
                "n_pos_test": n_pos,
                "average_precision": ap,
                "low_support": n_pos < MIN_SUPPORT_FOR_RATES,
            }
        )
    reliable = [
        f["average_precision"]
        for f in folds
        if not f["low_support"] and f["average_precision"] is not None
    ]
    return {
        "folds": folds,
        "summary": {
            "excluded_low_support": [f["held_out"] for f in folds if f["low_support"]],
            "average_precision": _summary(summarise_seeds(reliable)),
        },
    }


def mask_only_floor_block(datasets: dict[str, dict]) -> dict:
    """How much of the cross-replica AP is available without reading a feature value.

    Labels exist only for the log files M1 is computed from, so every attack window
    has M1 activity by construction, and nearly every attack is on one host. This
    block measures what that structure alone is worth, leave-one-replica-out: the
    availability mask alone (tessera.eval.leakage.mask_only_score's control, run on
    the LORO folds instead of 5-fold CV), host_bucket alone, and both together.
    """
    from tessera.features.pipeline import M3_SLICE

    host_col = M3_SLICE.start
    mask = _loro_floor(datasets, lambda X: availability_mask(X))
    host = _loro_floor(datasets, lambda X: np.asarray(X)[:, [host_col]])
    both = _loro_floor(
        datasets,
        lambda X: np.column_stack([availability_mask(X), np.asarray(X)[:, host_col]]),
    )

    A = np.concatenate([availability_mask(d["X"]) for d in datasets.values()])
    y = np.concatenate([np.asarray(d["y"]) for d in datasets.values()]).astype(bool)
    hosts = np.concatenate([np.asarray(d["host"]).astype(str) for d in datasets.values()])
    n_attack, n_benign = int(y.sum()), int((~y).sum())
    on_host = hosts == FLOOR_HOST
    patterns = {}
    for cls, m in (("attack", y), ("benign", ~y)):
        keys, counts = np.unique(A[m].astype(np.int8), axis=0, return_counts=True)
        patterns[cls] = [
            {"a": [int(v) for v in k], "n": int(c)}
            for k, c in sorted(
                zip(keys.tolist(), counts.tolist(), strict=True), key=lambda t: -t[1]
            )
        ]

    return {
        "provenance": "recomputed",
        "source": (
            "recomputed by tessera.demo.export_web_data from the real AIT cache: the "
            "shortcut-floor control of tessera.eval.leakage.mask_only_score (make_gbdt, "
            f"{FLOOR_N_ESTIMATORS} trees, seed 0, on the [M1, M2, M4] availability bits "
            "only) run on the same 8 leave-one-replica-out folds as loro instead of 5-fold "
            "CV; host_only and mask_and_host repeat it on m3_host_bucket alone and on "
            "both. The mask is build_real_dataset's availability array, recovered exactly "
            "from the cache (M1 and M4 from their slices, M2 from n_sources_active). "
            f"Folds with fewer than {MIN_SUPPORT_FOR_RATES} held-out positives are "
            "excluded from each summary, as in loro. "
            "tests/test_web_exports.py::test_shipped_mask_only_floor_is_fresh re-runs it "
            "from the cache and checks the shipped numbers."
        ),
        "model": f"LightGBM, {FLOOR_N_ESTIMATORS} trees, seed 0 (mask_only_score's configuration)",
        "features": "availability mask only: [M1 present, M2 present, M4 present]",
        "labelling_rule": (
            "A window is an attack window only if a labelled log line falls inside it "
            "(tessera.data.ait.window_builder). AIT labels exist only for the four log "
            "files M1 is computed from, so every attack window has M1 activity by "
            "construction; Suricata events carry no labels."
        ),
        "folds": mask["folds"],
        "summary": mask["summary"],
        "host_only": {
            "features": "m3_host_bucket only (1 feature)",
            "folds": host["folds"],
            "summary": host["summary"],
        },
        "mask_and_host": {
            "features": "availability mask + m3_host_bucket (4 features)",
            "folds": both["folds"],
            "summary": both["summary"],
        },
        "attack_share": {
            "n_attack_windows": n_attack,
            "n_benign_windows": n_benign,
            "m1_present_rate_attack": float(A[y, 0].mean()),
            "m1_present_rate_benign": float(A[~y, 0].mean()),
            "patterns": patterns,
            "host": FLOOR_HOST,
            "n_attack_on_host": int((y & on_host).sum()),
            "share_of_attacks_on_host": float((y & on_host).sum() / max(n_attack, 1)),
            "prevalence_on_host": float(y[on_host].mean()) if on_host.any() else None,
            "prevalence_elsewhere": float(y[~on_host].mean()) if (~on_host).any() else None,
        },
    }


def build_real_results(datasets: dict[str, dict]) -> dict:
    dup, allmod = leakage_blocks(datasets)
    return {
        "schema": "tessera-real-results/v1",
        "generated_by": GENERATED_BY,
        "generated_utc": _utc_now(),
        "provenance_note": (
            "Real results on real, held-out AIT data (aggregates only; no row-level data). "
            "Each block's 'provenance' says how its numbers got here: 'recomputed' = "
            "computed by this exporter from the local real-data cache at export time; "
            "'transcribed' = copied exactly from RESULTS.md / NEGATIVE_RESULTS.md, with "
            "'source' naming the test or ablation that recorded it. Average precision is "
            "the primary metric, MCC second; accuracy is not reported."
        ),
        "loro": loro_block(datasets),
        "mask_only_floor": mask_only_floor_block(datasets),
        "leakage_duplicates": dup,
        "calendar_leak": calendar_leak_block(datasets),
        "all_modalities_r0_r1": allmod,
        "tessera_base": TESSERA_BASE,
        "attribution_vs_ablation": ATTRIBUTION_VS_ABLATION,
        "dataset": dataset_block(datasets),
    }


# ---------------------------------------------------------------- design_notes.json

# NEGATIVE_RESULTS.md N2's recorded run, for comparison with this export's re-run
# (the cost model's hash/sign/verify timings are measured on the exporting machine).
RECORDED_OPTIMISER_RUN = {
    "provenance": "transcribed",
    "label": "recorded run (NEGATIVE_RESULTS N2)",
    "source": "transcribed from NEGATIVE_RESULTS.md N2; recorded by uv run python -m tessera.chainsim.report",
    "ground_truth": {"best_segment_length": 184, "best_objective": 0.945294},
    "results": [
        {
            "optimiser": "random search",
            "best_segment_length": 187,
            "best_objective": 0.945332,
            "optimality_gap_pct": 0.004,
        },
        {
            "optimiser": "Optuna TPE",
            "best_segment_length": 185,
            "best_objective": 0.945354,
            "optimality_gap_pct": 0.006,
        },
        {
            "optimiser": "EHO (NH=10, NI=6, LH=0.5)",
            "best_segment_length": 16680,
            "best_objective": 1.204840,
            "optimality_gap_pct": 27.457,
        },
    ],
}

DESIGN_DECISIONS = [
    {
        "decision": (
            "Report average precision and MCC on leakage-checked splits, not accuracy "
            "on a random split"
        ),
        "why": (
            "A random split scored 29 points higher on average precision than a "
            "chronological split on the same data (0.928 vs 0.639), which is exactly "
            "the leakage this project's own harness is built to catch; reporting a "
            "random-split accuracy number instead would launder that leakage into a "
            "headline metric. Accuracy is also uninformative at AIT's low attack "
            "prevalence, so it is not reported at all."
        ),
        "evidence": (
            "On real AIT data (M2-only features), a random split scores average "
            "precision 0.928 against 0.639 for a chronological split, with 28% "
            "exact-duplicate rows and 1,170 test rows byte-identical to a training row."
        ),
        "source_file": "RESULTS.md; README.md; src/tessera/eval/leakage.py; src/tessera/eval/metrics.py",
    },
    {
        "decision": (
            "Reject the draft delay/energy objective for chain segment-length tuning "
            "because it never depends on the segment length"
        ),
        "why": (
            "An objective that cannot change with the decision variable cannot certify "
            "any optimisation gain attributed to it, no matter which optimiser searches "
            "it; TESSERA replaced it with the cost model in cost_model.py, whose terms "
            "genuinely rise and fall with segment length."
        ),
        "evidence": (
            "Evaluated literally, the draft objective fh is bit-identical for every "
            "candidate segment length from 8 to 2048 (absolute spread 0.0), so "
            "d(fh)/d(segment length) = 0."
        ),
        "source_file": "src/tessera/chainsim/segment_objective.py",
    },
    {
        "decision": (
            "Discard the draft delay/energy objective as dimensionally ill-posed "
            "before ever tuning it"
        ),
        "why": (
            "An objective whose units resolve to neither a delay nor an energy, and "
            "whose summation bound is never pinned down, cannot be validated by any "
            "optimiser result; the well-posed replacement (tessera.chainsim.cost_model) "
            "reports total delay in seconds instead."
        ),
        "evidence": (
            "A sum of delays multiplied by an energy has units of joule-seconds, which "
            "is neither a delay nor an energy, and the draft's summation bound was left "
            "unspecified while its normaliser was fixed."
        ),
        "source_file": "src/tessera/chainsim/segment_objective.py",
    },
    {
        "decision": ("Store only a hash of six fields on the ledger, never raw IPs/URLs/bodies"),
        "why": (
            "Many blockchain-logging designs conflate immutability with security and "
            "store raw request data on-chain; once written to a shared immutable "
            "ledger, that data can never be redacted. TESSERA states its own threat "
            "model explicitly and commits only a hash, so the ledger proves retroactive "
            "tampering without ever holding personal data itself."
        ),
        "evidence": (
            "Each TESSERA log entry is an RFC 6962 leaf hash (SHA-256 over a 0x00 prefix "
            "and the canonical JSON) of six fields: {window_id, host_hash, ts_bucket, "
            "verdict, score, model_git_sha}. host_hash is meant to be a salted "
            "pseudonym of the host (the salt helper in ledger/sth.py is not yet wired "
            "in, and the leaf hash itself is unsalted); no IP, geolocation, username, "
            "URL, header or body is included. The ledger proves only that an auditor "
            "holding a past signed tree head can detect retroactive modification or "
            "deletion of a retained verdict."
        ),
        "source_file": (
            "THREAT_MODEL.md; src/tessera/ledger/merkle.py; src/tessera/data/contract.py"
        ),
    },
]


def _gap_pct_text(v: float) -> str:
    """The audit tab's gapPct format: 0.005% below 0.1%, else one decimal (27.6%)."""
    if v == 0:
        return "0%"
    return f"{v:.3f}%" if abs(v) < 0.1 else f"{round(v, 1):g}%"


def eho_decision(
    *,
    optimum: int,
    max_lh: float,
    interval_at_half: list,
    eho_gap_pct: float,
    random_gap_pct: float,
    budget: int,
) -> dict:
    """Written from THIS export's recomputed values - the same ones the optimiser
    and reachability charts draw - so the page never shows two optima.
    NEGATIVE_RESULTS.md N2's recorded run stays in optimiser_comparison.recorded_run."""
    a, b = (int(v) for v in interval_at_half)
    return {
        "decision": (
            "Evaluate EHO, TPE and random search against exhaustive ground truth for "
            "chain segment-length tuning, and reject EHO"
        ),
        "why": (
            "The herd update rule only averages existing herds and defines no mutation, "
            "so the herd population can never leave the interval it was initialised in - "
            "a structural limit, not a tuning failure, that no herd count, iteration "
            "budget or seed can fix. Random search and TPE, which are not confined this "
            "way, land within a fraction of a percent of the exhaustive optimum on the "
            "same evaluation budget."
        ),
        "evidence": (
            f"The reachable set at LH = 0.5 is [{a}, {b}], while the exhaustive optimum "
            f"in this export is S = {optimum} (reaching it would need LH <= {max_lh:.4f}). "
            f"Measured optimality gap on the same {budget}-evaluation budget: EHO "
            f"{_gap_pct_text(eho_gap_pct)}, random search {_gap_pct_text(random_gap_pct)}. "
            "The recorded run (NEGATIVE_RESULTS N2), timed on another machine, is shown "
            "separately on the optimiser card."
        ),
        "source_file": "NEGATIVE_RESULTS.md (N2); src/tessera/chainsim/optimisers/eho.py",
    }


def build_design_notes() -> dict:
    """The calls tessera.chainsim.report.main makes, captured as data."""
    from tessera.chainsim.benchmark import HI, LO, run
    from tessera.chainsim.cost_model import Workload, calibrate, evaluate_cost
    from tessera.chainsim.optimisers import eho, exhaustive
    from tessera.chainsim.segment_objective import (
        demonstrate_objective_invariance,
        segment_delay_objective,
    )

    cal = calibrate(n_hash=20000, n_sign=1500)
    res = run(budget=60, seed=0, cal=cal)
    grid = res["ground_truth"]

    wl = Workload(n_entries=res["workload"]["n_entries"])

    def obj(S: int) -> float:
        return evaluate_cost(S, wl, cal).total_delay_s

    # benchmark.run's "ground truth" evaluates every 8th S. The objective jumps with
    # ceil(N/S), so an off-grid S can beat that grid optimum and a metaheuristic would
    # then show a NEGATIVE optimality gap (measured while building this export: TPE
    # found S=172 below the grid's S=168). Every integer in the domain is cheap to
    # evaluate (~65k calls, well under a second), so the exported ground truth is the
    # genuinely exhaustive one and every gap is measured against it.
    truth = exhaustive.search(obj, LO, HI, step=1)
    optimum, best = int(truth.best_x), float(truth.best_value)
    metaheuristics = [r for r in res["results"] if r["optimiser"] != grid["optimiser"]]
    beat_grid = [
        r["optimiser"] for r in metaheuristics if r["best_objective"] < grid["best_objective"]
    ]

    unreach = eho.demonstrate_structural_unreachability(obj, LO, HI, optimum)
    inv = demonstrate_objective_invariance()
    # One real fh value per NSC, recomputed directly; must equal what
    # demonstrate_objective_invariance measured (bit-identical by construction of
    # the draft objective).
    fh_values = [segment_delay_objective(n, seed=0) for n in inv["nsc_values"]]
    if fh_values != list(inv["fh_values"]):
        raise AssertionError(
            "segment_delay_objective disagrees with demonstrate_objective_invariance"
        )

    results = [
        {
            "optimiser": r["optimiser"],
            "best_segment_length": r["best_segment_length"],
            "best_objective": r["best_objective"],
            "n_evaluations": r["n_evaluations"],
            "optimality_gap": r["best_objective"] - best,
            "optimality_gap_pct": 100.0 * (r["best_objective"] - best) / best,
        }
        for r in metaheuristics
    ]
    max_lh = 2.0 * optimum / HI
    eho_row = next(r for r in results if r["optimiser"].startswith("EHO"))
    rnd_row = next(r for r in results if r["optimiser"] == "random search")
    half = next(row for row in unreach["rows"] if row["learning_rate"] == 0.5)
    design_decisions = list(DESIGN_DECISIONS)
    design_decisions.insert(
        3,
        eho_decision(
            optimum=optimum,
            max_lh=max_lh,
            interval_at_half=half["reachable_interval"],
            eho_gap_pct=eho_row["optimality_gap_pct"],
            random_gap_pct=rnd_row["optimality_gap_pct"],
            budget=res["budget_per_metaheuristic"],
        ),
    )

    return {
        "schema": "tessera-design-notes/v1",
        "generated_by": GENERATED_BY,
        "generated_utc": _utc_now(),
        "provenance_note": (
            "Recomputed at export time by the same calls tessera.chainsim.report makes "
            "(uv run python -m tessera.chainsim.report). The cost model's hash, sign and "
            "verify timings are measured on the exporting machine, so objective values - "
            "and possibly the optimum S by a few units - differ slightly from the run "
            "recorded in NEGATIVE_RESULTS.md N2 (kept below as recorded_run). Energy terms "
            "are modelled, not measured. design_decisions summarise the cited files; the "
            "fourth decision's numbers are written from this export's recomputed values, "
            "never the recorded run."
        ),
        "optimiser_comparison": {
            "provenance": "recomputed",
            "source": "recomputed: tessera.chainsim.benchmark.run(budget=60, seed=0, cal=calibrate(n_hash=20000, n_sign=1500))",
            "objective": "total delay in seconds of the segmented verdict log (well-posed cost model)",
            "search_domain": [LO, HI],
            "budget_per_metaheuristic": res["budget_per_metaheuristic"],
            "workload": res["workload"],
            "calibration": {
                k: res["calibration"][k]
                for k in ("t_hash_s", "t_sign_s", "t_verify_s", "measured_on", "note")
            },
            "ground_truth": {
                "optimiser": "exhaustive, every integer S (ground truth)",
                "best_segment_length": optimum,
                "best_objective": best,
                "n_evaluations": truth.n_evaluations,
            },
            "grid_ground_truth": {
                "optimiser": grid["optimiser"],
                "step": 8,
                "best_segment_length": grid["best_segment_length"],
                "best_objective": grid["best_objective"],
                "n_evaluations": grid["n_evaluations"],
                "beaten_by": beat_grid,
                "note": (
                    "tessera.chainsim.benchmark.run's ground truth searches every 8th S; "
                    "the gaps below are measured against the every-integer search instead, "
                    "so none can be negative."
                ),
            },
            "results": results,
            "eho_history": [
                {
                    "iteration": h["iteration"],
                    "herd_spread": h["herd_spread"],
                    "matriarch": h["matriarch"],
                    "n_reconfigured": h["n_reconfigured"],
                    "mean_fitness": h["mean_fitness"],
                }
                for h in (res["eho_history"] or [])
            ],
            "recorded_run": RECORDED_OPTIMISER_RUN,
        },
        "eho_unreachability": {
            "provenance": "recomputed",
            "source": "recomputed: tessera.chainsim.optimisers.eho.demonstrate_structural_unreachability",
            "chain_length_n": HI,
            "rows": [
                {
                    "learning_rate": row["learning_rate"],
                    "reachable_interval": row["reachable_interval"],
                    "optimum_reachable": row["optimum_reachable"],
                    "best_possible_objective": row["best_possible_value_in_interval"],
                }
                for row in unreach["rows"]
            ],
            "optimum": optimum,
            "optimum_objective": best,
            "search_domain": [LO, HI],
            # The herd-init interval's lower end is LH*N/2, so the optimum is admitted
            # only when LH <= 2S*/N.
            "max_learning_rate_admitting_optimum": max_lh,
            "optimum_reachable_for_any_tested_lh": unreach["optimum_reachable_for_any_tested_lh"],
            "argument": unreach["argument"],
            "finding": unreach["finding"],
        },
        "fh_invariance": {
            "provenance": "recomputed",
            "source": (
                "recomputed: tessera.chainsim.segment_objective.demonstrate_objective_invariance "
                "and segment_delay_objective(NSC) per value"
            ),
            "equation": inv["equation"],
            "nsc_values": list(inv["nsc_values"]),
            "fh_values": fh_values,
            "absolute_spread": inv["absolute_spread"],
            "relative_spread": inv["relative_spread"],
            "d_fh_d_nsc_is_zero": inv["d_fh_d_nsc_is_zero"],
            "units": inv["units"],
            "undefined_symbol": inv["undefined_symbol"],
            "finding": inv["finding"],
            "parameters_note": (
                "Per-block read/write/hash/verify delays and energy are "
                "segment_delay_objective's illustrative constants with seeded 1% jitter; "
                "the finding is that NSC never enters the computation, not the "
                "magnitude of fh."
            ),
        },
        "design_decisions": design_decisions,
    }


# ---------------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=Path, default=WEB_DATA_DIR)
    ap.add_argument(
        "--only",
        choices=("stats", "results", "design"),
        action="append",
        help="export only these files (repeatable); default: all three",
    )
    args = ap.parse_args(argv)
    only = set(args.only or ("stats", "results", "design"))
    out: Path = args.out_dir

    datasets = load_real_datasets() if only & {"stats", "results"} else None

    if "stats" in only:
        t0 = time.perf_counter()
        stats = compute_replica_stats(datasets)
        size = write_payload(out / "replica_stats.json", stats, indent=0, inline_width=160)
        print(
            f"wrote {out / 'replica_stats.json'}  ({size / 1024:.1f} KiB, {time.perf_counter() - t0:.1f}s)"
        )

    if "results" in only:
        t0 = time.perf_counter()
        results = build_real_results(datasets)
        size = write_payload(out / "real_results.json", results)
        s = results["loro"]["summary"]
        print(
            f"wrote {out / 'real_results.json'}  ({size / 1024:.1f} KiB, {time.perf_counter() - t0:.1f}s)"
        )
        print(
            f"  LORO AP {s['average_precision']['mean']:.4f} +/- {s['average_precision']['std']:.4f} "
            f"over {s['average_precision']['n']} folds (excluded: {s['excluded_low_support']}); "
            f"naive all-folds {s['naive_all_folds_ap']['mean']:.3f} +/- {s['naive_all_folds_ap']['std']:.3f}; "
            f"MCC {s['mcc']['mean']:.3f} +/- {s['mcc']['std']:.3f}"
        )
        for f in results["loro"]["folds"]:
            ap_s = "n/a" if f["average_precision"] is None else f"{f['average_precision']:.4f}"
            mcc_s = "n/a" if f["mcc"] is None else f"{f['mcc']:.3f}"
            print(
                f"    {f['held_out']:<16} n={f['n_test']:>6} pos={f['n_pos_test']:>5} "
                f"AP={ap_s} MCC={mcc_s}" + ("  LOW SUPPORT" if f["low_support"] else "")
            )
        d, a = results["leakage_duplicates"], results["all_modalities_r0_r1"]
        print(
            f"  M2-only R0 {d['r0_random_ap']:.4f} vs R1 {d['r1_chronological_ap']:.4f}; exact dup "
            f"{d['exact_duplicate_rate']:.4f}, near dup {d['near_duplicate_rate']:.4f}, "
            f"{d['test_rows_identical_to_train']} test rows identical to train"
        )
        print(
            f"  all-modalities R0 {a['r0_ap']:.4f} vs R1 {a['r1_ap']:.4f} (R1 MCC {a['r1_mcc']:.3f})"
        )
        fl = results["mask_only_floor"]
        sh = fl["attack_share"]
        print(
            "  shortcut floor (LORO, same exclusions): "
            + "; ".join(
                f"{name} AP {b['summary']['average_precision']['mean']:.3f} "
                f"+/- {b['summary']['average_precision']['std']:.3f}"
                for name, b in (
                    ("mask only", fl),
                    ("host only", fl["host_only"]),
                    ("mask + host", fl["mask_and_host"]),
                )
            )
        )
        print(
            f"  M1 present in {sh['m1_present_rate_attack']:.1%} of attack vs "
            f"{sh['m1_present_rate_benign']:.1%} of benign windows; "
            f"{sh['share_of_attacks_on_host']:.1%} of attack windows on {sh['host']}"
        )

    if "design" in only:
        t0 = time.perf_counter()
        design_notes = build_design_notes()
        size = write_payload(out / "design_notes.json", design_notes)
        old_audit = out / "audit.json"
        if old_audit.exists():
            old_audit.unlink()
        oc = design_notes["optimiser_comparison"]
        print(
            f"wrote {out / 'design_notes.json'}  ({size / 1024:.1f} KiB, {time.perf_counter() - t0:.1f}s)"
        )
        print(
            f"  ground truth S={oc['ground_truth']['best_segment_length']} "
            f"({oc['ground_truth']['best_objective']:.6f}s); "
            + "; ".join(
                f"{r['optimiser']}: S={r['best_segment_length']} gap {r['optimality_gap_pct']:.3f}%"
                for r in oc["results"]
            )
        )
        print(f"  fh spread across NSC: {design_notes['fh_invariance']['absolute_spread']!r}")


if __name__ == "__main__":
    main()
