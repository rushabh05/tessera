"""THE headline finding of this project, reproduced on REAL AIT data end to end:
random-split evaluation is dominated by duplicate-row leakage, exactly the failure
mode this harness exists to catch and the base paper's own methodology exhibits.

ingest -> label join -> M2 features -> split -> GBDT -> leakage certificate,
entirely on real russellmitchell data (vpn, intranet_server, inet-firewall).

Skipped if the bundle or the labelled sources are not extracted locally.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.data.ait.unpack import cleanup, extract, zip_path
from tessera.eval.certificate import build_certificate
from tessera.eval.halt_gate import check_e1
from tessera.eval.metrics import evaluate
from tessera.eval.splits import (
    assert_partitions_have_positives,
    r0_random,
    r1_chronological,
)
from tessera.features.pipeline import build_real_dataset
from tessera.models.baselines.gbdt import make_gbdt
from tessera.train.seed import seed_everything

pytestmark = pytest.mark.skipif(
    not zip_path("russellmitchell").exists(),
    reason="AIT bundle not downloaded locally; run the P1 data step first",
)


@pytest.fixture(scope="module")
def real_dataset():
    r = extract(
        "russellmitchell",
        only_prefixes=(
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
        ),
    )
    ds = build_real_dataset(
        replica_dir=r.extract_dir,
        capture_year=2022,
        hosts=["vpn", "intranet_server", "inet-firewall"],
    )
    yield ds
    cleanup("russellmitchell")


def test_dataset_has_real_shape_and_prevalence(real_dataset):
    assert real_dataset.X.shape[0] > 15000
    assert real_dataset.X.shape[1] == 24
    assert 0.0 < real_dataset.y.mean() < 1.0
    assert set(real_dataset.host) == {"vpn", "intranet_server", "inet-firewall"}


def test_random_split_is_inflated_by_duplicate_leakage(real_dataset):
    """THE headline finding. Locked with a wide tolerance band (real LightGBM
    numbers vary slightly by environment) but the DIRECTION and MAGNITUDE of the
    gap are the actual claim under test."""
    seed_everything(0)
    X, y = real_dataset.X, real_dataset.y

    r0 = r0_random(y, seed=0)
    assert_partitions_have_positives(r0, y)
    m0 = make_gbdt(seed=0, n_estimators=200)
    m0.fit(X[r0.train_idx], y[r0.train_idx])
    p0 = m0.predict_proba(X[r0.test_idx])[:, 1]
    ap0 = evaluate(y[r0.test_idx], p0)["binary"]["average_precision"]

    w_start = real_dataset.window_start.astype(np.float64)
    r1 = r1_chronological(w_start, gap_seconds=600)
    assert_partitions_have_positives(r1, y)
    m1 = make_gbdt(seed=0, n_estimators=200)
    m1.fit(X[r1.train_idx], y[r1.train_idx])
    p1 = m1.predict_proba(X[r1.test_idx])[:, 1]
    ap1 = evaluate(y[r1.test_idx], p1)["binary"]["average_precision"]

    # R0 (random, same hosts) must score well above R1 (chronological, same
    # hosts) - the gap isolates duplicate-row leakage, since host distribution is
    # held constant between the two splits.
    assert ap0 > ap1 + 0.15, (
        f"expected random-split inflation > 0.15, got R0={ap0:.4f} R1={ap1:.4f} "
        f"(gap {ap0 - ap1:.4f}) - the headline finding did not reproduce"
    )

    # The E1 gate (required elsewhere in the codebase) must actually fire on this
    # real gap, not just on synthetic data.
    gate = check_e1(ap0, ap1, raise_on_fail=False)
    assert gate.passed, gate.message


def test_leakage_certificate_finds_real_duplicates(real_dataset):
    """The certificate must find real cross-split twins on real data, not just on
    the synthetic fixture with deliberately injected duplicates."""
    X, y = real_dataset.X, real_dataset.y
    r0 = r0_random(y, seed=0)
    cert = build_certificate(split=r0, y=y, X=X, seed=0, run_permutation=False)
    d = cert.exact_duplicates
    assert d["cross_split_twins"] > 0, (
        "expected real cross-split duplicate rows on this feature set (sparse "
        "network activity produces many identical all-zero or near-zero windows); "
        "found none - either the data changed or the control regressed"
    )


def test_permutation_check_still_passes_on_real_data(real_dataset):
    """The one control that must NOT fire on real data: shuffled labels must
    still score at chance, confirming the leakage is in the SPLIT, not a bug
    in the evaluation pipeline itself."""
    X, y = real_dataset.X, real_dataset.y
    r0 = r0_random(y, seed=0)
    cert = build_certificate(split=r0, y=y, X=X, seed=0, run_permutation=True)
    assert cert.permutation["within_tolerance"] is True, cert.permutation
