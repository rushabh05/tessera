"""Leave-one-replica-out (R3) on real AIT data, across all 8 downloaded bundles.

Named per the project's own established framing: 'replica', not 'organisation'.
The 8 AIT testbeds are parameter-randomised executions of ONE scenario and attack
repertoire, so this measures robustness to that randomisation - not transfer to a
genuinely different environment. See tests/test_cross_replica.py for the 2-replica
version this extends, and RESULTS.md for why that distinction matters.

Each replica's (host, window) dataset is built once and CACHED to
data/processed/ait/<replica>.npz, because building one from raw eve.json/label
files (extract -> join -> M1/M2/M3/M4) costs real time (tens of seconds) and disk
churn (streaming-unpack extract + cleanup per replica), and 8 folds each need
every OTHER replica's data in memory simultaneously. The cache is derived,
regenerable data - gitignored, never a substitute for the raw manifest-hashed
zips, and rebuilt automatically if the source pipeline code's cache-format
version changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tessera.data.ait.unpack import REPLICAS, cleanup, extract
from tessera.eval.metrics import MIN_SUPPORT_FOR_RATES, evaluate
from tessera.eval.stats import summarise_seeds
from tessera.features.pipeline import N_TOTAL_FEATURES, build_real_dataset
from tessera.models.baselines.gbdt import make_gbdt
from tessera.paths import PROCESSED_DIR

HOSTS = ("vpn", "intranet_server", "inet-firewall")
PREFIXES = (
    "gather/vpn/logs/openvpn.log",
    "labels/vpn/logs/openvpn.log",
    "gather/vpn/logs/suricata/eve.json",
    "gather/intranet_server/logs/auth.log",
    "labels/intranet_server/logs/auth.log",
    "gather/intranet_server/logs/audit/audit.log",
    "labels/intranet_server/logs/audit/audit.log",
    "gather/intranet_server/logs/suricata/eve.json",
    "gather/inet-firewall/logs/dnsmasq.log",
    "labels/inet-firewall/logs/dnsmasq.log",
    "gather/inet-firewall/logs/suricata/eve.json",
    "dataset.yaml",
)

# Bump this if build_real_dataset's feature layout ever changes, so a stale cache
# built under an old feature set cannot silently be reused as if it still matched.
CACHE_VERSION = 1


def cache_path(replica: str) -> Path:
    d = PROCESSED_DIR / "ait"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{replica}_v{CACHE_VERSION}.npz"


def build_or_load_replica_dataset(replica: str, *, force: bool = False) -> dict:
    """Build one replica's (host, window) feature set, or load it from cache."""
    cp = cache_path(replica)
    if cp.exists() and not force:
        with np.load(cp, allow_pickle=True) as z:
            return {
                "X": z["X"],
                "y": z["y"],
                "host": z["host"],
                "window_start": z["window_start"],
                "n_features": int(z["n_features"]),
            }

    r = extract(replica, only_prefixes=PREFIXES)
    try:
        ds = build_real_dataset(replica_dir=r.extract_dir, capture_year=2022, hosts=list(HOSTS))
    finally:
        cleanup(replica)

    assert ds.X.shape[1] == N_TOTAL_FEATURES, (
        f"{replica}: built {ds.X.shape[1]} features, expected {N_TOTAL_FEATURES} "
        f"- the cache format has drifted; bump CACHE_VERSION"
    )
    np.savez_compressed(
        cp,
        X=ds.X,
        y=ds.y,
        host=ds.host,
        window_start=ds.window_start,
        n_features=ds.X.shape[1],
    )
    return {
        "X": ds.X,
        "y": ds.y,
        "host": ds.host,
        "window_start": ds.window_start,
        "n_features": ds.X.shape[1],
    }


def build_all_replica_datasets(
    replicas: tuple = REPLICAS, *, force: bool = False
) -> dict[str, dict]:
    return {r: build_or_load_replica_dataset(r, force=force) for r in replicas}


def run_leave_one_replica_out(
    datasets: dict[str, dict],
    *,
    seed: int = 0,
    n_estimators: int = 200,
    min_support: int = MIN_SUPPORT_FOR_RATES,
) -> dict:
    """Train on 7 replicas, test on the held-out 8th, for every replica in turn.

    Per the plan's own convention (eval/metrics.py's MIN_SUPPORT_FOR_RATES,
    applied per-class within one dataset; applied here per-FOLD across replicas,
    which the AIT scenario's own execution-time randomisation makes necessary): a
    fold whose held-out replica has fewer than ``min_support`` positive windows is
    flagged ``low_support`` and EXCLUDED from the summary statistics, not silently
    averaged in as if it carried the same statistical weight as a fold with
    thousands of positives. Found empirically: one replica (shaw) has only 6
    positive windows for this project's fixed 3-host/4-source subset - a real,
    honest consequence of "attack parameters and execution order vary per
    replica" (Zenodo's own description), not a pipeline defect. A fold that thin
    is still REPORTED (never dropped from the per-fold table), only excluded from
    the headline mean/std.
    """
    names = sorted(datasets)
    folds = []
    for held in names:
        train_names = [n for n in names if n != held]
        X_train = np.concatenate([datasets[n]["X"] for n in train_names], axis=0)
        y_train = np.concatenate([datasets[n]["y"] for n in train_names], axis=0)
        X_test, y_test = datasets[held]["X"], datasets[held]["y"]
        n_pos_test = int(y_test.sum())

        if len(np.unique(y_test)) < 2:
            folds.append(
                {
                    "held_out": held,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                    "n_pos_test": n_pos_test,
                    "low_support": True,
                    "note": "held-out replica has a single class; AP undefined",
                }
            )
            continue

        model = make_gbdt(seed=seed, n_estimators=n_estimators)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba)["binary"]
        folds.append(
            {
                "held_out": held,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_pos_test": n_pos_test,
                "test_prevalence": float(y_test.mean()),
                "average_precision": metrics["average_precision"],
                "mcc": metrics["mcc"],
                "low_support": n_pos_test < min_support,
            }
        )

    scored = [f for f in folds if "average_precision" in f]
    reliable = [f for f in scored if not f["low_support"]]
    low_support = [f for f in scored if f["low_support"]]

    summary = {
        "n_folds": len(folds),
        "n_scored_folds": len(scored),
        "min_support_for_rates": min_support,
        "n_low_support_folds_excluded": len(low_support),
        "low_support_replicas": [f["held_out"] for f in low_support],
        "average_precision": summarise_seeds([f["average_precision"] for f in reliable]),
        "mcc": summarise_seeds([f["mcc"] for f in reliable]),
    }
    return {"folds": folds, "summary": summary}
