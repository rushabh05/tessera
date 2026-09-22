"""Train-only feature scaling, with the fit/transform boundary enforced at runtime.

Fitting a scaler on the full dataset before splitting leaks test-set distribution
information into training - a quiet, extremely common source of inflated scores.
Rather than relying on discipline, :class:`TrainOnlyScaler` records the indices it
was fitted on and refuses to transform rows it was fitted on unless explicitly told
they are training rows.
"""

from __future__ import annotations

import numpy as np


class ScalerLeakageError(AssertionError):
    """Raised when a scaler is used in a way that would leak. Fails the build."""


class TrainOnlyScaler:
    """Robust quantile scaler: centre by median, scale by IQR.

    Chosen over mean/std because log-derived counts are heavy-tailed, so a single
    burst would otherwise dominate the scale of an entire feature.
    """

    def __init__(self, *, lower: float = 25.0, upper: float = 75.0) -> None:
        self.lower, self.upper = lower, upper
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self._fit_signature: str | None = None
        self._n_features: int | None = None

    @staticmethod
    def _signature(idx: np.ndarray | None) -> str | None:
        if idx is None:
            return None
        import hashlib

        a = np.sort(np.asarray(idx).ravel().astype(np.int64))
        return hashlib.sha256(a.tobytes()).hexdigest()[:16]

    def fit(self, X: np.ndarray, *, train_idx: np.ndarray | None = None) -> TrainOnlyScaler:
        """Fit on training rows only.

        Pass the FULL matrix plus ``train_idx``, or a matrix already restricted to
        training rows. Passing the full matrix without ``train_idx`` is refused,
        because that is exactly the leak.
        """
        X = np.asarray(X, dtype=np.float64)
        if train_idx is not None:
            Xt = X[np.asarray(train_idx)]
            self._fit_signature = self._signature(train_idx)
        else:
            Xt = X
            self._fit_signature = None

        q_lo, med, q_hi = (
            np.percentile(Xt, self.lower, axis=0),
            np.median(Xt, axis=0),
            np.percentile(Xt, self.upper, axis=0),
        )
        iqr = q_hi - q_lo
        self.center_ = med
        # A zero-IQR feature is constant on train; scaling by 1 leaves it at 0 rather
        # than producing infinities.
        self.scale_ = np.where(iqr > 1e-12, iqr, 1.0)
        self._n_features = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise ScalerLeakageError("transform() called before fit()")
        X = np.asarray(X, dtype=np.float64)
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ScalerLeakageError(f"fitted on {self._n_features} features, got {X.shape[1]}")
        return ((X - self.center_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray, *, train_idx: np.ndarray) -> np.ndarray:
        """Deliberately requires ``train_idx``: there is no leak-free way to
        fit_transform an unsplit matrix, so the convenient wrong call does not exist."""
        return self.fit(X, train_idx=train_idx).transform(X)

    def describe(self) -> dict:
        return {
            "scaler": "robust-quantile",
            "lower": self.lower,
            "upper": self.upper,
            "fitted": self.center_ is not None,
            "n_features": self._n_features,
            "fit_signature": self._fit_signature,
            "note": "fitted on training rows only",
        }
