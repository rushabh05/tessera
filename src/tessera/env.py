"""Environment capture, recorded with every run.

An examiner asking "on what machine, with which library versions, at which commit
was this number produced?" must be answerable from the run directory alone.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def capture_env() -> dict:
    """A JSON-serialisable snapshot of the runtime, including torch device support."""
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "packages": {
            name: _pkg(name)
            for name in (
                "torch",
                "numpy",
                "polars",
                "scikit-learn",
                "lightgbm",
                "drain3",
                "hydra-core",
                "optuna",
                "pymoo",
                "networkx",
                "cryptography",
                "scipy",
            )
        },
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            # A dirty tree means the code that produced this number is not the
            # code in the commit. Recorded so it can never be claimed otherwise.
            "dirty": bool(_git("status", "--porcelain")),
        },
    }

    try:
        import torch

        info["torch_devices"] = {
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
            "cuda_available": torch.cuda.is_available(),
            "num_threads": torch.get_num_threads(),
        }
    except Exception as exc:  # pragma: no cover
        info["torch_devices"] = {"error": str(exc)}

    try:
        info["cpu"] = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            ).stdout.strip()
            or None
        )
    except Exception:  # pragma: no cover
        info["cpu"] = None

    return info


def env_json() -> str:
    return json.dumps(capture_env(), indent=2, sort_keys=True)
