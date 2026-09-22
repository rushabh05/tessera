"""Floor baselines. Every headline number must clear all of these to mean anything.

``MaskOnly`` is the important one. It sees ONLY which modalities were present - no
features at all - so it measures how much of a score is obtainable from the
telemetry availability pattern. A detector that fails to beat it has learned which
monitoring agents were installed, not what an attack looks like.
"""

from __future__ import annotations

import numpy as np

from tessera.data.contract import WindowSet
from tessera.models.base import Prediction, uniform_attribution
from tessera.models.baselines.gbdt import gbdt_backend, make_gbdt


class RandomScorer:
    """Uniform random scores. The P0 end-to-end smoke model.

    Exists so the whole harness - splits, certificate, metrics, tables - can be
    validated before any modelling code is written. Its average precision must come
    out at roughly the class prevalence; anything else means the harness is broken.
    """

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, train: WindowSet, val: WindowSet | None = None):
        return self

    def predict(self, data: WindowSet) -> Prediction:
        rng = np.random.default_rng(self.seed)
        s = rng.random(data.n).astype(np.float32)
        return Prediction(
            score=s,
            confidence=np.abs(s - 0.5).astype(np.float32),
            attribution=uniform_attribution(data.n),
        )

    def describe(self) -> dict:
        return {"name": self.name, "seed": self.seed, "kind": "floor/smoke"}


class MajorityClass:
    """Predicts the training-set majority. The trivial floor."""

    name = "majority"

    def __init__(self) -> None:
        self.p = 0.0

    def fit(self, train: WindowSet, val: WindowSet | None = None):
        self.p = float(train.y_bin.mean())
        return self

    def predict(self, data: WindowSet) -> Prediction:
        s = np.full(data.n, self.p, dtype=np.float32)
        return Prediction(
            score=s,
            confidence=np.ones(data.n, dtype=np.float32),
            attribution=uniform_attribution(data.n),
        )

    def describe(self) -> dict:
        return {"name": self.name, "train_prevalence": self.p, "kind": "floor"}


class MaskOnly:
    """GBDT on the availability mask ALONE - the shortcut floor.

    If this scores near the full model, the full model's advantage is mostly the
    missingness pattern, which is an artifact of monitoring deployment rather than
    a detection capability.
    """

    name = "nullmask"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.model = None

    def fit(self, train: WindowSet, val: WindowSet | None = None):
        X = train.availability.astype(np.float32)
        y = train.y_bin
        if len(np.unique(y)) < 2:
            self.model = None
            self._const = float(y.mean())
            return self
        self.model = make_gbdt(seed=self.seed, n_estimators=100).fit(X, y)
        return self

    def predict(self, data: WindowSet) -> Prediction:
        if self.model is None:
            s = np.full(data.n, getattr(self, "_const", 0.0), dtype=np.float32)
        else:
            s = self.model.predict_proba(data.availability.astype(np.float32))[:, 1].astype(
                np.float32
            )
        return Prediction(
            score=s,
            confidence=np.abs(s - 0.5).astype(np.float32),
            attribution=uniform_attribution(data.n),
            extra={"note": "sees only the availability mask, no features"},
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "backend": gbdt_backend(),
            "kind": "shortcut floor",
        }


class GbdtFlat:
    """Tuned-family gradient boosting on the flat tabular view.

    This is the baseline expected to be hard to beat within-dataset, and it is given
    a genuine chance rather than being crippled to flatter the deep model.
    """

    name = "lightgbm"

    def __init__(self, seed: int = 0, n_estimators: int = 300) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.model = None

    def fit(self, train: WindowSet, val: WindowSet | None = None):
        X, y = train.flat_features(), train.y_bin
        if len(np.unique(y)) < 2:
            self.model = None
            self._const = float(y.mean())
            return self
        self.model = make_gbdt(seed=self.seed, n_estimators=self.n_estimators).fit(X, y)
        return self

    def predict(self, data: WindowSet) -> Prediction:
        if self.model is None:
            s = np.full(data.n, getattr(self, "_const", 0.0), dtype=np.float32)
        else:
            s = self.model.predict_proba(data.flat_features())[:, 1].astype(np.float32)
        return Prediction(
            score=s,
            confidence=np.abs(s - 0.5).astype(np.float32),
            attribution=uniform_attribution(data.n),
        )

    def describe(self) -> dict:
        return {
            "name": self.name,
            "seed": self.seed,
            "n_estimators": self.n_estimators,
            "backend": gbdt_backend(),
            "kind": "strong classical baseline",
        }


REGISTRY = {
    "random": RandomScorer,
    "majority": MajorityClass,
    "nullmask": MaskOnly,
    "lightgbm": GbdtFlat,
}


def build(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"unknown model '{name}'; available: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
