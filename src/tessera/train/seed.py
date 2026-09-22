"""Seeding for reproducibility across Python, NumPy, torch, MPS and dataloaders.

Seeding the main process is not enough: DataLoader workers each get their own RNG
state, and unseeded workers silently randomise augmentation and shuffling order.
:func:`worker_init_fn` and :func:`make_generator` close that gap.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedReport:
    """What seeding actually achieved, recorded per run so it is auditable."""

    seed: int
    deterministic_algorithms: bool
    cudnn_benchmark_disabled: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "seed": self.seed,
            "deterministic_algorithms": self.deterministic_algorithms,
            "cudnn_benchmark_disabled": self.cudnn_benchmark_disabled,
            "note": self.note,
        }


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedReport:
    """Seed every RNG this project touches.

    ``deterministic=True`` requests deterministic kernels. Several ops have no
    deterministic MPS implementation, so we use ``warn_only`` and record what we
    got rather than crashing or, worse, silently claiming determinism we lack.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch

    from tessera import ensure_single_thread

    # Required for process stability, not just determinism: with more than one
    # intra-op thread, a CPU conv after a LightGBM fit deadlocks. See tessera/__init__.
    ensure_single_thread()

    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - no CUDA on this machine
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    note = ""
    got_deterministic = False
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            got_deterministic = True
        except Exception as exc:  # pragma: no cover
            note = f"use_deterministic_algorithms unavailable: {exc}"

    cudnn_off = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        cudnn_off = True

    return SeedReport(
        seed=seed,
        deterministic_algorithms=got_deterministic,
        cudnn_benchmark_disabled=cudnn_off,
        note=note,
    )


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible RNG state."""
    import torch

    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_generator(seed: int):
    """A torch Generator for DataLoader shuffling, so batch order is reproducible."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
