"""THE FROZEN MODEL OUTPUT CONTRACT.

Every detector - the dummy scorer, the classical baselines, the two base-paper
reimplementations, TESSERA-base and TESSERA-EV - implements this. Freezing it now
means the demo track can build its whole UI against a random scorer in week 1 and
swap in the real model later without touching a line of frontend code.

Attribution is part of the contract, not an afterthought: a detector that cannot say
which modality drove an alert cannot be explained in a demo or a viva.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from tessera.data.contract import MODALITIES, WindowSet


@dataclass
class Prediction:
    """What every model returns. Arrays are aligned to the input WindowSet."""

    score: np.ndarray  # (N,) anomaly score in [0, 1]
    confidence: np.ndarray  # (N,) higher = more confident; drives abstention
    attribution: np.ndarray  # (N, 4) per-modality contribution, MODALITIES order
    y_coarse_logits: np.ndarray | None = None  # (N, C) optional class head
    uncertainty: np.ndarray | None = None  # (N,) evidential vacuity, if the model has one
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.score)
        if self.confidence.shape != (n,):
            raise ValueError(f"confidence must be ({n},), got {self.confidence.shape}")
        if self.attribution.shape != (n, len(MODALITIES)):
            raise ValueError(
                f"attribution must be ({n}, {len(MODALITIES)}), got {self.attribution.shape}"
            )
        if not np.all(np.isfinite(self.score)):
            raise ValueError("score contains non-finite values")

    def top_modality(self) -> np.ndarray:
        """Name of the highest-attributed modality per window, for the demo panel."""
        return np.array(MODALITIES, dtype=object)[np.argmax(self.attribution, axis=1)]


@runtime_checkable
class Detector(Protocol):
    """The interface. Implementations must be usable without any torch import."""

    name: str

    def fit(self, train: WindowSet, val: WindowSet | None = None) -> Detector: ...

    def predict(self, data: WindowSet) -> Prediction: ...

    def describe(self) -> dict:
        """Config actually used, recorded in the run record."""
        ...


def uniform_attribution(n: int) -> np.ndarray:
    """Equal attribution, for models with no notion of modality contribution.

    Returned explicitly rather than left as None so a downstream consumer never has
    to special-case it, and so a table can show plainly that these models attribute
    nothing.
    """
    return np.full((n, len(MODALITIES)), 1.0 / len(MODALITIES), dtype=np.float32)
