"""The full 8-replica leave-one-replica-out result - the real R3 evidence,
extending the 2-replica check in test_cross_replica.py to every downloaded
bundle.

Uses the on-disk cache in data/processed/ait/ (built by build_all_replica_datasets)
so this runs in seconds once the cache exists, rather than re-extracting and
re-parsing ~7 GB of raw data per replica on every test run. Skipped if the cache
is not present - building it from scratch takes real time (measured: ~11 minutes
for all 8 replicas) and is not something a routine test run should trigger
implicitly.
"""

from __future__ import annotations

import pytest

from tessera.data.ait.unpack import REPLICAS
from tessera.eval.loro_real import cache_path, run_leave_one_replica_out

pytestmark = pytest.mark.skipif(
    not all(cache_path(r).exists() for r in REPLICAS),
    reason="processed dataset cache not built for all 8 replicas; run "
    "build_all_replica_datasets() first (~11 min on a cold cache)",
)


@pytest.fixture(scope="module")
def all_replica_datasets():
    from tessera.eval.loro_real import build_all_replica_datasets

    return build_all_replica_datasets()  # cache hit for every replica


def test_all_eight_replicas_have_cached_datasets(all_replica_datasets):
    assert set(all_replica_datasets) == set(REPLICAS)
    for name, ds in all_replica_datasets.items():
        assert ds["X"].shape[0] > 0, name
        assert ds["n_features"] == ds["X"].shape[1]


def test_shaw_is_the_genuinely_low_support_replica(all_replica_datasets):
    """A real, honest finding, not a bug: shaw's fixed 3-host/4-source subset has
    only 6 positive windows out of 29207, because its attack episode for these
    specific hosts is much briefer than the other 7 replicas' (a 44-minute
    cluster near the end of a 162-hour capture, vs. hours-to-days elsewhere).
    Every other replica has hundreds to thousands of positives. If this ever
    changes, either the source data changed or the pipeline regressed."""
    y_shaw = all_replica_datasets["shaw"]["y"]
    assert y_shaw.sum() < 20, (
        f"shaw now has {y_shaw.sum()} positives, no longer the low-support "
        "outlier this test documents - update the finding, don't just widen this bound"
    )
    for name, ds in all_replica_datasets.items():
        if name == "shaw":
            continue
        assert ds["y"].sum() >= 100, f"{name} unexpectedly low-support too"


def test_leave_one_replica_out_generalises_strongly(all_replica_datasets):
    """THE finding: excluding the one honestly-flagged low-support fold (shaw),
    average precision across the remaining 7 leave-one-replica-out folds is high
    and TIGHT (small std) - real evidence the M1+M2+M3+M4 features transfer
    across independently-randomised executions of the scenario, not merely
    across the one pair (russellmitchell -> santos) checked in
    test_cross_replica.py."""
    result = run_leave_one_replica_out(all_replica_datasets, seed=0)

    assert result["summary"]["n_folds"] == 8
    assert result["summary"]["low_support_replicas"] == ["shaw"]

    ap = result["summary"]["average_precision"]
    assert ap["n_seeds"] == 7, "expected 7 reliable folds after excluding shaw"
    assert ap["mean"] > 0.95, f"LORO mean AP dropped to {ap['mean']:.4f}"
    assert ap["std"] < 0.05, f"LORO AP std widened to {ap['std']:.4f}; a fold destabilised"
    assert ap["min"] > 0.9, f"weakest reliable fold dropped to {ap['min']:.4f}"


def test_low_support_fold_is_flagged_not_silently_averaged():
    """Direct check of the flagging mechanism itself, independent of the real
    data: a fold with fewer positives than MIN_SUPPORT_FOR_RATES must be marked
    low_support=True and excluded from the summary statistics."""
    import numpy as np

    from tessera.eval.loro_real import run_leave_one_replica_out

    rng = np.random.default_rng(0)
    fake = {
        "big_a": {
            "X": rng.normal(0, 1, (500, 4)).astype("float32"),
            "y": (rng.random(500) < 0.3).astype("int8"),
        },
        "big_b": {
            "X": rng.normal(0, 1, (500, 4)).astype("float32"),
            "y": (rng.random(500) < 0.3).astype("int8"),
        },
        "thin": {
            "X": rng.normal(0, 1, (500, 4)).astype("float32"),
            "y": np.zeros(500, dtype="int8"),
        },
    }
    fake["thin"]["y"][:3] = 1  # 3 positives - well under the default min_support=20

    result = run_leave_one_replica_out(fake, seed=0, min_support=20)
    thin_fold = next(f for f in result["folds"] if f["held_out"] == "thin")
    assert thin_fold["low_support"] is True
    assert result["summary"]["low_support_replicas"] == ["thin"]
    assert result["summary"]["average_precision"]["n_seeds"] == 2
