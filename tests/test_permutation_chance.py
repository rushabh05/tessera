"""Permuted labels must score at chance. Above chance means the pipeline leaks."""

from __future__ import annotations

from tessera.data.synthetic import make_synthetic
from tessera.eval.leakage import permutation_check


def test_permuted_labels_score_at_chance():
    ws = make_synthetic(n_replicas=2, hosts_per_replica=2, windows_per_host=300, seed=0)
    res = permutation_check(ws.flat_features(), ws.y_bin, seed=0, n_repeats=2, cv=3)
    assert res["within_tolerance"] is True, (
        f"permuted AP {res['permuted_average_precision_mean']:.4f} vs chance "
        f"{res['chance_level']:.4f}; the pipeline has a defect"
    )


def test_real_labels_score_above_chance():
    """The converse: the fixture must contain real signal, or the check is vacuous."""
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import cross_val_predict

    from tessera.models.baselines.gbdt import make_gbdt

    ws = make_synthetic(n_replicas=2, hosts_per_replica=2, windows_per_host=300, seed=0)
    X, y = ws.flat_features(), ws.y_bin
    p = cross_val_predict(make_gbdt(seed=0, n_estimators=60), X, y, cv=3, method="predict_proba")[
        :, 1
    ]
    ap = average_precision_score(y, p)
    assert ap > 3 * y.mean(), f"AP {ap:.4f} barely above prevalence {y.mean():.4f}"
