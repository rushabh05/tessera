"""The OpenMP guard must hold in BOTH failure directions.

On macOS/arm64 torch bundles its own libomp while LightGBM's dylib loads the
Homebrew one, putting two OpenMP runtimes in one process. Measured on this machine:

* ``import torch`` then ``lgb.fit(...)``              -> SIGSEGV (exit 139)
* ``lgb.fit(...)`` then a CPU ``Conv1d`` forward pass -> DEADLOCK (hangs forever)

The second direction is the one that actually broke the test suite: the parity test
passed alone and crashed once ``test_e2e`` had fitted a LightGBM model first.

Both run in subprocesses because one mode is a hard SIGSEGV and the other is an
unrecoverable hang; either would take the whole session down in-process.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tessera import LIGHTGBM_AVAILABLE

# Direction 1: torch imported, then a LightGBM fit. Unguarded -> SIGSEGV.
SEGFAULT_DIRECTION = """
import tessera            # claims the OpenMP runtime first
import torch
import lightgbm as lgb, numpy as np
X = np.random.RandomState(0).randn(200, 5); y = (X[:, 0] > 0).astype(int)
lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
if torch.backends.mps.is_available():
    t = torch.randn(64, 64, device="mps"); torch.mps.synchronize(); _ = (t @ t).sum().item()
lgb.LGBMClassifier(n_estimators=10, verbose=-1).fit(X, y)
print("OK")
"""

# Direction 2: a LightGBM fit, then a CPU conv. Unguarded -> DEADLOCK.
DEADLOCK_DIRECTION = """
import tessera
from tessera.train.seed import seed_everything
seed_everything(0)
import lightgbm as lgb, numpy as np, torch
assert torch.get_num_threads() == 1, f"threads={torch.get_num_threads()}; >1 deadlocks"
X = np.random.RandomState(0).randn(300, 6); y = (X[:, 0] > 0).astype(int)
lgb.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)
conv = torch.nn.Conv1d(16, 32, 3, dilation=2)
out = conv(torch.randn(8, 16, 64))
print("OK", tuple(out.shape))
"""


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
def test_torch_then_lightgbm_fit_does_not_segfault():
    out = subprocess.run(
        [sys.executable, "-c", SEGFAULT_DIRECTION],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, (
        f"exit {out.returncode} (139 or -11 = SIGSEGV; the import-order guard has "
        f"regressed).\nstderr:\n{out.stderr[-2000:]}"
    )
    assert "OK" in out.stdout


@pytest.mark.skipif(not LIGHTGBM_AVAILABLE, reason="LightGBM not installed")
def test_lightgbm_fit_then_cpu_conv_does_not_deadlock():
    """The direction that actually broke the suite. A timeout here IS the bug."""
    try:
        out = subprocess.run(
            [sys.executable, "-c", DEADLOCK_DIRECTION],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "DEADLOCK: a CPU conv after a LightGBM fit hung. torch must be pinned to "
            "one intra-op thread; counts 2, 4, 8, 10 and 14 were each measured and "
            "each deadlocked."
        ) from None
    assert out.returncode == 0, f"exit {out.returncode}\nstderr:\n{out.stderr[-2000:]}"
    assert "OK" in out.stdout


def test_single_thread_is_actually_pinned():
    from tessera import ensure_single_thread

    assert ensure_single_thread() == 1
