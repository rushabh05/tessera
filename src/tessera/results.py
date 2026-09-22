"""Append-only run records. Every reported number originates from one of these.

Design rules, each defending against a specific failure:

* A run directory is never overwritten. Re-running with identical settings creates
  a new record, so a result can never be silently replaced by a later one.
* Config, environment, seed report and the leakage certificate are written next to
  the metrics, so a number is never separable from the conditions that produced it.
* ``results/index.jsonl`` is append-only and is what the table generator reads, so
  tables cannot be assembled from anything but recorded runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tessera.env import capture_env


def results_dir() -> Path:
    """Resolved at call time so TESSERA_RESULTS_DIR applies to the test suite."""
    from tessera import paths

    return paths.RESULTS_DIR


def runs_dir() -> Path:
    from tessera import paths

    return paths.RUNS_DIR


def index_path() -> Path:
    return results_dir() / "index.jsonl"


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_fingerprint(config: dict) -> str:
    """Short stable hash of a config, so identical settings are recognisable."""
    return hashlib.sha256(_canonical(config).encode()).hexdigest()[:12]


@dataclass
class RunRecord:
    """One experimental run. Build it, add metrics, then :meth:`write`."""

    config: dict
    tags: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    seed_report: dict | None = None
    leakage_certificate: dict | None = None
    artifacts: dict = field(default_factory=dict)

    run_id: str = field(init=False)
    created_utc: str = field(init=False)
    _dir: Path | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        now = datetime.now(UTC)
        self.created_utc = now.isoformat(timespec="seconds")
        stamp = now.strftime("%Y%m%dT%H%M%S")
        # Timestamp keeps runs ordered; fingerprint makes the settings visible in
        # the name; pid disambiguates runs started in the same second.
        self.run_id = f"{stamp}-{config_fingerprint(self.config)}-{os.getpid()}"

    @property
    def dir(self) -> Path:
        if self._dir is None:
            self._dir = runs_dir() / self.run_id
        return self._dir

    def log(self, **kwargs: Any) -> None:
        self.metrics.update(kwargs)

    def write(self) -> Path:
        """Persist the run. Refuses to overwrite an existing record."""
        if self.dir.exists():
            raise FileExistsError(
                f"run directory already exists, refusing to overwrite: {self.dir}"
            )
        self.dir.mkdir(parents=True)

        payload = {
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "config_fingerprint": config_fingerprint(self.config),
            "tags": self.tags,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }

        (self.dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
        (self.dir / "config.json").write_text(json.dumps(self.config, indent=2, default=str))
        (self.dir / "env.json").write_text(json.dumps(capture_env(), indent=2, default=str))
        if self.seed_report is not None:
            (self.dir / "seed.json").write_text(json.dumps(self.seed_report, indent=2))
        if self.leakage_certificate is not None:
            (self.dir / "leakage_certificate.json").write_text(
                json.dumps(self.leakage_certificate, indent=2, default=str)
            )

        idx = index_path()
        idx.parent.mkdir(parents=True, exist_ok=True)
        index_line = {
            **payload,
            "dir": str(self.dir.relative_to(results_dir())),
            "has_leakage_certificate": self.leakage_certificate is not None,
        }
        with idx.open("a") as fh:
            fh.write(_canonical(index_line) + "\n")

        return self.dir


def load_index() -> list[dict]:
    """Read every recorded run. The only permitted source for a reported number."""
    idx = index_path()
    if not idx.exists():
        return []
    rows = []
    for line in idx.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows
