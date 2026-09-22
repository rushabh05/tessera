"""Two real findings from real AIT data, both locked in with regression tests.

1. THE HEADLINE FINDING: random-split evaluation is inflated by duplicate-row
   leakage - the project's central thesis, reproduced in its own pipeline.
2. A SELF-FOUND BUG: calendar-position features (hour-of-day etc.) are a severe
   temporal leakage vector on a short, non-repeating capture - found, measured,
   and removed from M3 (see m3_identity.py's module docstring for the numbers).

ingest -> label join -> M1+M2+M3 features -> split -> GBDT -> leakage
certificate, entirely on real russellmitchell data (vpn, intranet_server,
inet-firewall). Skipped if the bundle is not extracted locally.
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
from tessera.features.m1_log import N_M1_FEATURES
from tessera.features.m2_metrics import N_M2_FEATURES
from tessera.features.m3_identity import N_M3_FEATURES
from tessera.features.m4_graph import N_M4_FEATURES
from tessera.features.pipeline import N_TOTAL_FEATURES, build_real_dataset
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
    assert real_dataset.X.shape[1] == N_TOTAL_FEATURES == 42
    assert N_M1_FEATURES + N_M2_FEATURES + N_M3_FEATURES + N_M4_FEATURES == N_TOTAL_FEATURES
    assert 0.0 < real_dataset.y.mean() < 1.0
    assert set(real_dataset.host) == {"vpn", "intranet_server", "inet-firewall"}


def test_build_report_per_host_prevalence_is_populated(real_dataset):
    """Regression test: a rewrite of build_real_dataset (to share the source
    parse between label windowing and M1 template mining) stopped populating
    n_windows/n_attack_windows/per_host_prevalence on the returned report -
    every field silently stayed at its dataclass default ({}), only caught by
    inspecting a real cross-replica run where the printed report was empty
    despite y being fully populated."""
    report = real_dataset.build_report
    assert report.n_windows == real_dataset.X.shape[0]
    assert report.n_attack_windows == int(real_dataset.y.sum())
    assert set(report.per_host_prevalence) == {"vpn", "intranet_server", "inet-firewall"}
    for stats in report.per_host_prevalence.values():
        assert stats["n_windows"] > 0
        assert 0.0 <= stats["prevalence"] <= 1.0


def test_random_split_is_inflated_by_duplicate_leakage_on_m2_alone(real_dataset):
    """THE headline finding, isolated to M2 (Suricata) columns only, which are
    dominated by sparse/near-zero rows and therefore show the classic duplicate-
    leakage pattern most clearly. Wide tolerance: the direction and rough
    magnitude of the gap are the claim under test, not an exact figure."""
    seed_everything(0)
    m2_start = N_M1_FEATURES
    X = real_dataset.X[:, m2_start : m2_start + N_M2_FEATURES]
    y = real_dataset.y

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

    assert ap0 > ap1 + 0.15, (
        f"expected random-split inflation > 0.15 on M2-alone features, got "
        f"R0={ap0:.4f} R1={ap1:.4f} (gap {ap0 - ap1:.4f})"
    )
    gate = check_e1(ap0, ap1, raise_on_fail=False)
    assert gate.passed, gate.message


def test_full_feature_set_generalises_well_without_calendar_leakage(real_dataset):
    """With M1+M2+M3 combined (M3 already excludes the removed calendar
    features), the chronological score should be high and close to the random
    score - the corrected state, after the temporal-leakage bug was found and
    fixed. A regression here means the leaky features crept back in."""
    seed_everything(0)
    X, y = real_dataset.X, real_dataset.y

    r0 = r0_random(y, seed=0)
    m0 = make_gbdt(seed=0, n_estimators=200)
    m0.fit(X[r0.train_idx], y[r0.train_idx])
    p0 = m0.predict_proba(X[r0.test_idx])[:, 1]
    ap0 = evaluate(y[r0.test_idx], p0)["binary"]["average_precision"]

    w_start = real_dataset.window_start.astype(np.float64)
    r1 = r1_chronological(w_start, gap_seconds=600)
    m1 = make_gbdt(seed=0, n_estimators=200)
    m1.fit(X[r1.train_idx], y[r1.train_idx])
    p1 = m1.predict_proba(X[r1.test_idx])[:, 1]
    ap1 = evaluate(y[r1.test_idx], p1)["binary"]["average_precision"]

    assert ap1 > 0.9, f"chronological AP dropped to {ap1:.4f}; check for regressions"
    assert ap0 - ap1 < 0.10, (
        f"R0-R1 gap widened to {ap0 - ap1:.4f}; a leaky feature may have crept "
        "back into M1/M2/M3 (see m3_identity.py's calendar-feature history)"
    )


def test_calendar_features_are_not_present_in_m3():
    """Direct lock on the fix: M3 must not include absolute calendar position."""
    assert N_M3_FEATURES == 2
    from tessera.features.m3_identity import M3_FEATURE_NAMES

    for leaky in ("hour", "day", "minute", "weekend"):
        assert not any(leaky in name for name in M3_FEATURE_NAMES), (
            f"M3_FEATURE_NAMES contains a calendar-like feature ('{leaky}'); "
            "this was measured to cause severe temporal leakage - see "
            "m3_identity.py's module docstring before reintroducing it"
        )


def test_leakage_certificate_finds_real_duplicates(real_dataset):
    """The certificate must find real cross-split twins on real data."""
    X, y = real_dataset.X, real_dataset.y
    r0 = r0_random(y, seed=0)
    cert = build_certificate(split=r0, y=y, X=X, seed=0, run_permutation=False)
    d = cert.exact_duplicates
    assert d["cross_split_twins"] > 0


def test_permutation_check_still_passes_on_real_data(real_dataset):
    """Shuffled labels must still score at chance - confirms the leakage found
    above is in the split, not a bug in the evaluation pipeline itself."""
    X, y = real_dataset.X, real_dataset.y
    r0 = r0_random(y, seed=0)
    cert = build_certificate(split=r0, y=y, X=X, seed=0, run_permutation=True)
    assert cert.permutation["within_tolerance"] is True, cert.permutation
