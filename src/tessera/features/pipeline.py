"""The P3 vertical slice: real AIT data -> labelled windows -> M1+M2+M3 features
-> a numpy array the P0 harness (splits, leakage controls, GBDT baseline) can score.

M4 (graph) is not yet included - it needs its own per-window entity graph
construction, a larger piece of infrastructure than M1/M2/M3 each needed
individually, and is the next increment. A window with no eve.json data for its
host gets an all-zero M2 vector; the availability mask records this rather than
hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tessera.data.ait.label_join import join_source
from tessera.data.ait.window_builder import KNOWN_SOURCES, BuildReport, bucket_events
from tessera.features.m1_log import (
    N_M1_FEATURES,
    make_template_miner,
    mine_events_into_windows,
)
from tessera.features.m2_metrics import N_M2_FEATURES, stream_eve_windows
from tessera.features.m3_identity import N_M3_FEATURES, window_identity_vector

N_TOTAL_FEATURES = N_M1_FEATURES + N_M2_FEATURES + N_M3_FEATURES


@dataclass
class RealDataset:
    """Real, labelled, feature-extracted windows. What the P0 harness consumes."""

    X: np.ndarray  # (n, N_TOTAL_FEATURES)
    y: np.ndarray  # (n,) int8
    host: np.ndarray  # (n,) str — for group-disjoint splitting
    window_start: np.ndarray  # (n,) int64 — for chronological splitting
    availability: np.ndarray  # (n, 2) bool — [M1 (log) present, M2 (metrics) present]
    build_report: BuildReport
    feature_names: tuple = ()


def _join_all_sources(
    replica_dir: Path, hosts: list[str], capture_year: int
) -> tuple[dict, BuildReport]:
    """Join every known labelled source for every host - shared by the label
    windowing and the M1 template mining, so the raw parse happens once."""
    gather_root = replica_dir / "gather"
    labels_root = replica_dir / "labels"
    report = BuildReport(replica=replica_dir.name)
    events_by_host_source: dict[tuple[str, str], list] = {}

    for host in hosts:
        joined_here = []
        for base_name, subpath in KNOWN_SOURCES:
            gdir = gather_root / host / subpath
            ldir = labels_root / host / subpath
            if not (gdir / base_name).exists() and not any(gdir.glob(f"{base_name}.*")):
                continue
            try:
                events, _jr = join_source(
                    host=host,
                    base_name=base_name,
                    gather_dir=gdir,
                    labels_dir=ldir,
                    capture_year=capture_year,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not fatal to the run
                report.errors.append(f"{host}/{base_name}: {exc}")
                continue
            events_by_host_source[(host, base_name)] = events
            joined_here.append(base_name)
        if joined_here:
            report.hosts_processed.append(host)
            report.sources_joined[host] = joined_here

    return events_by_host_source, report


def build_real_dataset(*, replica_dir: Path, capture_year: int, hosts: list[str]) -> RealDataset:
    """Join labels with M1 (templates), M2 (Suricata) and M3 (identity) features."""
    events_by_hs, report = _join_all_sources(replica_dir, hosts, capture_year)

    # --- labels: reuse the join output to build (host, window) -> is_attack ----
    label_windows: dict[tuple[str, int], object] = {}
    for (_host, source), events in events_by_hs.items():
        bucket_events(events, source, label_windows)

    # --- M1: mine templates across every joined source, bucketed into windows --
    miner = make_template_miner()
    m1_windows = mine_events_into_windows(events_by_hs, miner=miner)

    # --- M2: stream each host's eve.json independently -------------------------
    eve_by_host: dict[str, dict[int, np.ndarray]] = {}
    for host in hosts:
        eve_path = replica_dir / "gather" / host / "logs" / "suricata" / "eve.json"
        if eve_path.exists():
            stats = stream_eve_windows(eve_path)
            eve_by_host[host] = {w: s.to_vector() for w, s in stats.items()}

    all_keys = (
        set(label_windows) | set(m1_windows) | {(h, w) for h, wm in eve_by_host.items() for w in wm}
    )
    rows = sorted(all_keys)
    n = len(rows)

    X = np.zeros((n, N_TOTAL_FEATURES), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)
    host_arr = np.empty(n, dtype=object)
    w_start_arr = np.zeros(n, dtype=np.int64)
    avail = np.zeros((n, 2), dtype=bool)

    for i, (host, w_start) in enumerate(rows):
        host_arr[i] = host
        w_start_arr[i] = w_start

        lw = label_windows.get((host, w_start))
        if lw is not None:
            y[i] = 1 if lw.is_attack else 0

        m1acc = m1_windows.get((host, w_start))
        if m1acc is not None:
            X[i, :N_M1_FEATURES] = m1acc.to_vector()
            avail[i, 0] = True

        m2vec = eve_by_host.get(host, {}).get(w_start)
        if m2vec is not None:
            X[i, N_M1_FEATURES : N_M1_FEATURES + N_M2_FEATURES] = m2vec
            avail[i, 1] = True

        n_sources_active = int(avail[i, 0]) + int(avail[i, 1])
        X[i, N_M1_FEATURES + N_M2_FEATURES :] = window_identity_vector(
            host=host, n_sources_active=n_sources_active
        )

    # Populate the window/prevalence counts on the report. These were computed by
    # build_host_windows() in the old M2-only pipeline; this rewrite joins sources
    # itself (to share the parse with M1's template mining) and must derive them
    # from the actual output arrays instead, or every per-host prevalence figure
    # silently reports empty - caught by inspecting a real cross-replica run where
    # `per_host_prevalence` printed as `{}` despite y being fully populated.
    report.n_windows = n
    report.n_attack_windows = int(y.sum())
    per_host: dict[str, list[int]] = {}
    for i in range(n):
        h = host_arr[i]
        counts = per_host.setdefault(h, [0, 0])
        counts[0] += 1
        counts[1] += int(y[i])
    report.per_host_prevalence = {
        h: {"n_windows": c[0], "n_attack": c[1], "prevalence": round(c[1] / max(c[0], 1), 4)}
        for h, c in sorted(per_host.items())
    }

    return RealDataset(
        X=X,
        y=y,
        host=host_arr,
        window_start=w_start_arr,
        availability=avail,
        build_report=report,
        feature_names=(
            *(
                f"m1_{n}"
                for n in (
                    "n_events",
                    "n_unique_templates",
                    "template_entropy",
                    "dominant_template_id",
                    "dominant_template_frac",
                    "n_new_templates",
                    "mean_line_length",
                    "max_line_length",
                )
            ),
            *(f"m2_{i}" for i in range(N_M2_FEATURES)),
            *(f"m3_{n}" for n in ("host_bucket", "n_sources_active")),
        ),
    )
