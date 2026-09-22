"""TESSERA-base: encoder/fusion mechanics on synthetic data (fast, always runs),
plus the real-data training result (slow, skipped if the LORO cache is absent -
see tests/test_loro_real.py for why that cache may not exist)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tessera.models.encoders.small_mlp import SmallMLPEncoder
from tessera.models.fusion.gmu import GatedMultimodalUnit
from tessera.models.tessera_base import N_MODALITIES, TesseraBase, best_device


def test_small_mlp_encoder_output_shape():
    enc = SmallMLPEncoder(in_dim=8, hidden_dim=16, out_dim=4)
    x = torch.randn(5, 8)
    out = enc(x)
    assert out.shape == (5, 4)


def test_small_mlp_encoder_falls_back_to_one_group_on_indivisible_hidden_dim():
    """hidden_dim=6 is not divisible by the default groups=4; must not raise."""
    enc = SmallMLPEncoder(in_dim=8, hidden_dim=6, out_dim=4)
    out = enc(torch.randn(3, 8))
    assert out.shape == (3, 4)


def test_gmu_gates_sum_to_one_over_available_modalities():
    gmu = GatedMultimodalUnit(n_modalities=4, embed_dim=8)
    embeds = torch.randn(10, 4, 8)
    availability = torch.ones(10, 4)
    availability[:, 2] = 0  # modality 2 absent for every row
    fused, gates = gmu(embeds, availability)
    assert fused.shape == (10, 8)
    assert gates.shape == (10, 4)
    assert torch.allclose(gates.sum(dim=-1), torch.ones(10), atol=1e-5)


def test_gmu_absent_modality_gets_exactly_zero_gate():
    """The core availability-masking guarantee: an absent modality must receive
    EXACTLY zero weight, not merely a small one - 'not present' must mean the
    model literally cannot lean on it."""
    gmu = GatedMultimodalUnit(n_modalities=4, embed_dim=8)
    embeds = torch.randn(20, 4, 8)
    availability = torch.ones(20, 4)
    availability[:, 1] = 0
    _, gates = gmu(embeds, availability)
    assert torch.allclose(gates[:, 1], torch.zeros(20), atol=1e-6)


def test_gmu_all_modalities_absent_does_not_produce_nan():
    """An edge case that should not occur in practice (M3 is always derived and
    always available) but is guarded rather than assumed: if every modality were
    absent, softmax-of-all-(-inf) must not propagate NaN into training."""
    gmu = GatedMultimodalUnit(n_modalities=4, embed_dim=8)
    embeds = torch.randn(3, 4, 8)
    availability = torch.zeros(3, 4)
    fused, gates = gmu(embeds, availability)
    assert torch.isfinite(fused).all()
    assert torch.isfinite(gates).all()


def test_tessera_base_fits_and_predicts_on_synthetic_data():
    """Fast, CI-friendly: a tiny synthetic dataset, 3 epochs, just checking the
    mechanics work end to end - shapes, no NaNs, attribution sums to 1."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, 42)).astype("float32")
    y = (rng.random(n) < 0.3).astype("int8")
    avail = (rng.random((n, 3)) < 0.8).astype(bool)

    model = TesseraBase(seed=0, epochs=3, batch_size=32)
    model.fit(X[:150], y[:150], avail[:150])

    proba = model.predict_proba(X[150:], avail[150:])
    assert proba.shape == (50,)
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()

    score, attribution = model.predict_with_attribution(X[150:], avail[150:])
    assert attribution.shape == (50, N_MODALITIES)
    assert np.allclose(attribution.sum(axis=1), 1.0, atol=1e-4)


def test_tessera_base_availability_masking_zeroes_absent_modality_attribution():
    """Integration check: a modality marked absent for EVERY row in a batch must
    receive zero attribution for every row, end to end through fit+predict."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(0, 1, (n, 42)).astype("float32")
    y = (rng.random(n) < 0.3).astype("int8")
    avail = np.ones((n, 3), dtype=bool)
    avail[:, 1] = False  # M2 absent for every window

    model = TesseraBase(seed=0, epochs=3, batch_size=32)
    model.fit(X, y, avail)
    _, attribution = model.predict_with_attribution(X, avail)
    assert np.allclose(attribution[:, 1], 0.0, atol=1e-5)


def test_describe_reports_real_config():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (60, 42)).astype("float32")
    y = (rng.random(60) < 0.3).astype("int8")
    avail = np.ones((60, 3), dtype=bool)
    model = TesseraBase(seed=0, epochs=2, batch_size=16)
    model.fit(X, y, avail)
    d = model.describe()
    assert d["name"] == "tessera_base"
    assert d["n_parameters"] > 0
    assert d["device"] in ("cpu", "mps")


def test_best_device_returns_a_valid_torch_device():
    d = best_device()
    assert d.type in ("cpu", "mps")


# ---------------------------------------------------------------- real data


def _loro_cache_complete() -> bool:
    from tessera.data.ait.unpack import REPLICAS
    from tessera.eval.loro_real import cache_path

    return all(cache_path(r).exists() for r in REPLICAS)


@pytest.mark.slow  # ~90s: trains a real model on ~150k real windows
@pytest.mark.skipif(
    not _loro_cache_complete(),
    reason="processed dataset cache not built; see test_loro_real.py",
)
def test_tessera_base_matches_lightgbm_on_real_data():
    """THE real-data result: TESSERA-base trained on 7 replicas, tested on the
    8th (santos), should score close to the tuned LightGBM baseline on the same
    fold - not necessarily beat it (that is not the claim), but land in the
    same strong regime, confirming the neural architecture is not broken."""
    from tessera.eval.loro_real import build_all_replica_datasets
    from tessera.eval.metrics import evaluate
    from tessera.features.pipeline import M1_SLICE, M2_SLICE, M4_SLICE
    from tessera.train.seed import seed_everything

    seed_everything(0)
    datasets = build_all_replica_datasets()
    held = "santos"
    train_names = [n for n in sorted(datasets) if n != held]
    X_train = np.concatenate([datasets[n]["X"] for n in train_names], axis=0)
    y_train = np.concatenate([datasets[n]["y"] for n in train_names], axis=0)
    X_test, y_test = datasets[held]["X"], datasets[held]["y"]

    def derive_availability(X):
        a = np.zeros((X.shape[0], 3), dtype=bool)
        a[:, 0] = np.abs(X[:, M1_SLICE]).sum(axis=1) > 0
        a[:, 1] = np.abs(X[:, M2_SLICE]).sum(axis=1) > 0
        a[:, 2] = np.abs(X[:, M4_SLICE]).sum(axis=1) > 0
        return a

    avail_train = derive_availability(X_train)
    avail_test = derive_availability(X_test)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y_train))
    n_val = int(0.15 * len(y_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = TesseraBase(seed=0, epochs=30, batch_size=256)
    model.fit(
        X_train[tr_idx],
        y_train[tr_idx],
        avail_train[tr_idx],
        X_val=X_train[val_idx],
        y_val=y_train[val_idx],
        availability_val=avail_train[val_idx],
    )
    proba = model.predict_proba(X_test, avail_test)
    ap = evaluate(y_test, proba)["binary"]["average_precision"]

    assert ap > 0.98, f"TESSERA-base AP {ap:.4f} on held-out santos, expected > 0.98"


@pytest.mark.slow  # ~180s: trains two real models (with/without M1) for the ablation
@pytest.mark.skipif(
    not _loro_cache_complete(),
    reason="processed dataset cache not built; see test_loro_real.py",
)
def test_gmu_attribution_does_not_match_naive_ablation_importance():
    """A real, documented finding (RESULTS.md / NEGATIVE_RESULTS.md): the GMU's
    self-reported attribution should NOT be taken at face value as ground truth
    for what the model relies on. Measured: with M1 present, the gate attributes
    only ~14% mean weight to it, yet removing M1 (marking it unavailable) costs
    real performance - confirming M1 matters more than its own attribution
    claims. This test locks in the DIRECTION of that gap, not exact figures."""
    from tessera.eval.loro_real import build_all_replica_datasets
    from tessera.eval.metrics import evaluate
    from tessera.features.pipeline import M1_SLICE, M2_SLICE, M4_SLICE
    from tessera.train.seed import seed_everything

    seed_everything(0)
    datasets = build_all_replica_datasets()
    held = "santos"
    train_names = [n for n in sorted(datasets) if n != held]
    X_train = np.concatenate([datasets[n]["X"] for n in train_names], axis=0)
    y_train = np.concatenate([datasets[n]["y"] for n in train_names], axis=0)
    X_test, y_test = datasets[held]["X"], datasets[held]["y"]

    def derive_availability(X):
        a = np.zeros((X.shape[0], 3), dtype=bool)
        a[:, 0] = np.abs(X[:, M1_SLICE]).sum(axis=1) > 0
        a[:, 1] = np.abs(X[:, M2_SLICE]).sum(axis=1) > 0
        a[:, 2] = np.abs(X[:, M4_SLICE]).sum(axis=1) > 0
        return a

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(y_train))
    n_val = int(0.15 * len(y_train))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    # WITH M1
    avail_train = derive_availability(X_train)
    avail_test = derive_availability(X_test)
    model_with = TesseraBase(seed=0, epochs=30, batch_size=256)
    model_with.fit(
        X_train[tr_idx],
        y_train[tr_idx],
        avail_train[tr_idx],
        X_val=X_train[val_idx],
        y_val=y_train[val_idx],
        availability_val=avail_train[val_idx],
    )
    proba_with, attribution_with = model_with.predict_with_attribution(X_test, avail_test)
    ap_with = evaluate(y_test, proba_with)["binary"]["average_precision"]
    m1_attribution = float(attribution_with[:, 0].mean())

    # WITHOUT M1 (zeroed and marked unavailable - the honest removal)
    X_train_noM1, X_test_noM1 = X_train.copy(), X_test.copy()
    X_train_noM1[:, M1_SLICE] = 0.0
    X_test_noM1[:, M1_SLICE] = 0.0
    avail_train_noM1 = avail_train.copy()
    avail_train_noM1[:, 0] = False
    avail_test_noM1 = avail_test.copy()
    avail_test_noM1[:, 0] = False

    model_without = TesseraBase(seed=0, epochs=30, batch_size=256)
    model_without.fit(
        X_train_noM1[tr_idx],
        y_train[tr_idx],
        avail_train_noM1[tr_idx],
        X_val=X_train_noM1[val_idx],
        y_val=y_train[val_idx],
        availability_val=avail_train_noM1[val_idx],
    )
    proba_without = model_without.predict_proba(X_test_noM1, avail_test_noM1)
    ap_without = evaluate(y_test, proba_without)["binary"]["average_precision"]

    # The gap this test locks in: attribution UNDERSTATES M1's real importance.
    assert m1_attribution < 0.3, (
        f"M1's mean attribution ({m1_attribution:.3f}) is no longer low - the "
        "documented divergence may have changed, update RESULTS.md if so"
    )
    assert ap_with - ap_without > 0.005, (
        f"removing M1 barely changed AP ({ap_with:.4f} -> {ap_without:.4f}); "
        "the 'attribution understates true importance' finding did not reproduce"
    )
