"""The frozen data contract must reject every shape and semantic violation."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.data.contract import M1_MAX_EVENTS, MODALITIES
from tessera.data.synthetic import make_fixture, make_synthetic


def test_fixture_satisfies_contract():
    fx = make_fixture()
    fx.validate()
    assert fx.n > 0
    assert fx.availability.shape[1] == len(MODALITIES)
    assert fx.flat_features().shape[0] == fx.n


def test_subset_preserves_contract():
    fx = make_fixture()
    sub = fx.subset(np.arange(0, fx.n, 3))
    sub.validate()
    assert sub.n == len(range(0, fx.n, 3))


@pytest.mark.mustfail
def test_unlabellable_window_cannot_be_positive():
    fx = make_fixture()
    bad = np.where(~fx.labellable)[0][0]
    fx.y_bin[bad] = 1
    with pytest.raises(ValueError, match="unlabellable"):
        fx.validate()


@pytest.mark.mustfail
def test_sequence_length_bound_is_enforced():
    fx = make_fixture()
    fx.m1_length[0] = M1_MAX_EVENTS + 1
    with pytest.raises(ValueError, match="m1_length"):
        fx.validate()


def test_synthetic_reproduces_the_availability_shortcut():
    """If the fixture has no mask shortcut, the C2 machinery is never exercised."""
    from tessera.eval.leakage import mask_only_score

    ws = make_synthetic(n_replicas=4, hosts_per_replica=4, windows_per_host=300, seed=0)
    m = mask_only_score(ws.availability, ws.y_bin, seed=0)
    assert m["average_precision"] > m["prevalence"] * 1.3, (
        "the synthetic corpus lacks label-correlated missingness, so the "
        "missing-modality experiments would be testing nothing"
    )


def test_synthetic_is_deterministic_under_seed():
    a = make_synthetic(n_replicas=2, hosts_per_replica=2, windows_per_host=60, seed=7)
    b = make_synthetic(n_replicas=2, hosts_per_replica=2, windows_per_host=60, seed=7)
    np.testing.assert_array_equal(a.y_bin, b.y_bin)
    np.testing.assert_allclose(a.m2_numeric, b.m2_numeric)
