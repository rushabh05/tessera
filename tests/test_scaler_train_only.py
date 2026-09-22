"""The scaler must be fittable only on training rows."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.features.scalers import ScalerLeakageError, TrainOnlyScaler


def test_fit_uses_only_train_rows():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 6))
    X[200:] += 100.0  # test rows on a wildly different scale
    train_idx = np.arange(200)

    s = TrainOnlyScaler().fit(X, train_idx=train_idx)
    ref = TrainOnlyScaler().fit(X[train_idx])

    np.testing.assert_allclose(s.center_, ref.center_)
    np.testing.assert_allclose(s.scale_, ref.scale_)
    # The shifted test rows must NOT have influenced the statistics.
    assert np.abs(s.center_).max() < 10.0


def test_transform_before_fit_is_refused():
    with pytest.raises(ScalerLeakageError, match="before fit"):
        TrainOnlyScaler().transform(np.zeros((3, 3)))


def test_fit_transform_requires_train_idx():
    """The convenient leaky call must not exist."""
    with pytest.raises(TypeError):
        TrainOnlyScaler().fit_transform(np.zeros((10, 3)))  # no train_idx


def test_constant_feature_does_not_produce_infinities():
    X = np.ones((50, 3))
    out = TrainOnlyScaler().fit(X, train_idx=np.arange(50)).transform(X)
    assert np.isfinite(out).all()
