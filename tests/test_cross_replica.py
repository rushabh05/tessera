"""The real generalisation test: train on one AIT replica, evaluate on a
DIFFERENT, never-seen replica - as opposed to R0/R1, which only ever split
within one replica.

Named 'cross-replica', not 'cross-organisation': the AIT replicas are
parameter-randomised executions of the SAME underlying scenario and attack
repertoire (verified from the Zenodo record description), so this measures
robustness to that randomisation - a real and useful property, but not evidence
of transfer to a genuinely different environment or attack type.
"""

from __future__ import annotations

import pytest

from tessera.data.ait.unpack import cleanup, extract, zip_path
from tessera.eval.metrics import evaluate
from tessera.eval.splits import r0_random
from tessera.features.pipeline import build_real_dataset
from tessera.models.baselines.gbdt import make_gbdt
from tessera.train.seed import seed_everything

HOSTS = ["vpn", "intranet_server", "inet-firewall"]
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

pytestmark = pytest.mark.skipif(
    not (zip_path("russellmitchell").exists() and zip_path("santos").exists()),
    reason="both AIT bundles must be downloaded locally for this test",
)


@pytest.fixture(scope="module")
def cross_replica_datasets():
    r_train = extract("russellmitchell", only_prefixes=PREFIXES)
    r_test = extract("santos", only_prefixes=PREFIXES)
    ds_train = build_real_dataset(replica_dir=r_train.extract_dir, capture_year=2022, hosts=HOSTS)
    ds_test = build_real_dataset(replica_dir=r_test.extract_dir, capture_year=2022, hosts=HOSTS)
    yield ds_train, ds_test
    cleanup("russellmitchell")
    cleanup("santos")


def test_datasets_are_genuinely_different_replicas(cross_replica_datasets):
    ds_train, ds_test = cross_replica_datasets
    # Different window counts / timestamps confirm these are not the same data.
    assert set(ds_train.window_start.tolist()) != set(ds_test.window_start.tolist())
    assert (
        ds_train.X.shape[0] != ds_test.X.shape[0]
        or not (ds_test.X[: ds_train.X.shape[0]] == ds_train.X).all()
    )


def test_cross_replica_generalisation_is_strong(cross_replica_datasets):
    """The finding: a model trained entirely on russellmitchell scores nearly as
    well on santos - a replica it has never seen - as a model trained AND
    tested within santos itself. Locked with a wide tolerance (the exact figure
    can shift slightly with environment/LightGBM version) but the CLAIM under
    test is that cross-replica performance is close to same-replica performance,
    not merely 'above chance'."""
    seed_everything(0)
    ds_train, ds_test = cross_replica_datasets

    m_cross = make_gbdt(seed=0, n_estimators=200)
    m_cross.fit(ds_train.X, ds_train.y)
    p_cross = m_cross.predict_proba(ds_test.X)[:, 1]
    ap_cross = evaluate(ds_test.y, p_cross)["binary"]["average_precision"]

    r0 = r0_random(ds_test.y, seed=0)
    m_same = make_gbdt(seed=0, n_estimators=200)
    m_same.fit(ds_test.X[r0.train_idx], ds_test.y[r0.train_idx])
    p_same = m_same.predict_proba(ds_test.X[r0.test_idx])[:, 1]
    ap_same = evaluate(ds_test.y[r0.test_idx], p_same)["binary"]["average_precision"]

    assert ap_cross > 0.9, f"cross-replica AP {ap_cross:.4f} too low to call this a finding"
    assert abs(ap_cross - ap_same) < 0.05, (
        f"cross-replica AP {ap_cross:.4f} diverges from same-replica AP "
        f"{ap_same:.4f} by more than the expected small margin"
    )
