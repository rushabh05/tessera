"""TESSERA: multimodal cloud anomaly detection with a leakage-instrumented harness.

OPENMP GUARD (do not reorder anything below)
-------------------------------------------
On macOS/arm64, torch bundles its own libomp while LightGBM's dylib loads the
Homebrew ``libomp``, so two OpenMP runtimes end up in one process. Measured on this
machine, both failure directions are real and both are fatal:

* ``import torch`` then ``lgb.fit(...)``              -> SIGSEGV (exit 139)
* ``lgb.fit(...)`` then a CPU ``Conv1d`` forward pass -> DEADLOCK (hangs forever)

Two fixes are needed, and the widely-cited workarounds do not work:

1. **Import LightGBM before torch.** Claiming the OpenMP runtime first removes the
   segfault direction. Every entry point imports :mod:`tessera` before anything
   else, so doing it here makes the ordering automatic.
2. **Pin torch to a single intra-op thread.** This removes the deadlock direction.
   The bound is not negotiable: thread counts of 2, 4, 8, 10 and 14 were each
   measured and each deadlocked; only 1 survives.

Measured and REJECTED: ``KMP_DUPLICATE_LIB_OK=TRUE`` still deadlocks, and setting
LightGBM's own ``num_threads=1`` still segfaults. The constraint is on torch's
thread pool specifically.

What this costs, stated plainly: CPU torch runs single-threaded. Training runs on
MPS, where this is irrelevant, and the GBDT baselines already pin
``num_threads=1`` for determinism, so little is lost. Parallelism comes from
running independent configurations as separate PROCESSES (Hydra ``--multirun``)
rather than threads within one - which is how the experiment grid runs anyway.

``tests/test_openmp_import_order.py`` fails the build if either direction regresses.
"""

from __future__ import annotations

import os

# Must precede any torch import. Wrapped because LightGBM is optional at runtime
# (the GBDT baseline can fall back to sklearn's HistGradientBoosting).
try:  # pragma: no cover - environment-dependent
    import lightgbm as _lightgbm  # noqa: F401

    LIGHTGBM_AVAILABLE = True
except Exception:  # pragma: no cover
    LIGHTGBM_AVAILABLE = False

# Deterministic cuBLAS/oneDNN reductions where the backend honours it. Set before
# torch initialises its backends, so it belongs here and not in seed().
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Fix (2) above: bound every OpenMP-aware runtime to one thread BEFORE torch loads.
# Setting the environment is what makes this stick for libraries that read it at
# import time; TORCH_NUM_THREADS is applied again defensively in ensure_single_thread().
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")


def ensure_single_thread() -> int:
    """Pin torch to one intra-op thread. Idempotent; safe to call repeatedly.

    Called by :func:`tessera.train.seed.seed_everything`, so any entry point that
    seeds is protected without having to remember this.
    """
    try:
        import torch
    except Exception:  # pragma: no cover
        return 0
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    return torch.get_num_threads()


__version__ = "0.1.0"

__all__ = ["LIGHTGBM_AVAILABLE", "__version__", "ensure_single_thread"]
