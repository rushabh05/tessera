"""The P3 vertical slice: real AIT data -> labelled windows -> M2 features ->
a numpy array the P0 harness (splits, leakage controls, GBDT baseline) can score.

Deliberately M2-only for this first pass - M1 (log templates), M3 (identity) and
M4 (graph) extend the same pattern but are separate, larger pieces of work. A
window with no eve.json data for its host gets an all-zero M2 vector, which is
exactly what the "availability mask" contract element is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tessera.data.ait.window_builder import BuildReport, build_host_windows
from tessera.features.m2_metrics import N_M2_FEATURES, stream_eve_windows


@dataclass
class RealDataset:
    """Real, labelled, feature-extracted windows. What the P0 harness consumes."""

    X: np.ndarray  # (n, N_M2_FEATURES)
    y: np.ndarray  # (n,) int8
    host: np.ndarray  # (n,) str — for group-disjoint splitting
    window_start: np.ndarray  # (n,) int64 — for chronological splitting
    availability: np.ndarray  # (n,) bool — whether eve.json covered this window
    build_report: BuildReport


def build_real_dataset(*, replica_dir: Path, capture_year: int, hosts: list[str]) -> RealDataset:
    """Join labels (window_builder) with M2 features (eve.json) for the given hosts."""
    windows, report = build_host_windows(
        replica_dir=replica_dir, capture_year=capture_year, hosts=hosts
    )

    eve_by_host: dict[str, dict[int, np.ndarray]] = {}
    for host in hosts:
        eve_path = replica_dir / "gather" / host / "logs" / "suricata" / "eve.json"
        if not eve_path.exists():
            continue
        stats = stream_eve_windows(eve_path)
        eve_by_host[host] = {w: s.to_vector() for w, s in stats.items()}

    # Union of label windows and eve windows: a window can exist in one source
    # without the other (a host with no Suricata activity that minute, or Suricata
    # activity with no labelled log entries that minute — both real).
    all_keys = set(windows) | {(h, w) for h, wm in eve_by_host.items() for w in wm}

    rows = sorted(all_keys, key=lambda k: (k[0], k[1]))
    n = len(rows)
    X = np.zeros((n, N_M2_FEATURES), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    host_arr = np.empty(n, dtype=object)
    w_start_arr = np.zeros(n, dtype=np.int64)
    avail = np.zeros(n, dtype=bool)

    for i, (host, w_start) in enumerate(rows):
        host_arr[i] = host
        w_start_arr[i] = w_start
        hw = windows.get((host, w_start))
        if hw is not None:
            y[i] = 1 if hw.is_attack else 0
        vec = eve_by_host.get(host, {}).get(w_start)
        if vec is not None:
            X[i] = vec
            avail[i] = True

    return RealDataset(
        X=X,
        y=y,
        host=host_arr,
        window_start=w_start_arr,
        availability=avail,
        build_report=report,
    )
