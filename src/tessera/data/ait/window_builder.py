"""The window builder: ties label_join.py's output into 60s (host, window) buckets
- the sample unit the WindowSet contract (P0) requires.

A window's label is positive if ANY labelled event from ANY source for that host
falls inside it. This is deliberately inclusive (not per-source): the label taxonomy
is attack-STEP based, not attack-FILE based, so a single attack step often only
shows up in one source (e.g. an auth.log escalation) while others in the same
60s window stay clean - the label must reflect "something bad happened on this host
in this window", which the window builder computes across every joined source.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from tessera.data.ait.label_join import JoinedEvent, LabelJoinError, join_source
from tessera.data.contract import WINDOW_SECONDS

# The labelled sources known for this dataset. Each entry is (base_name,
# gather_subpath, labels_subpath) relative to a host's directory. Not every host
# has every source - join_source is only called when the raw file exists.
KNOWN_SOURCES: tuple[tuple[str, str], ...] = (
    ("auth.log", "logs"),
    ("dnsmasq.log", "logs"),
    ("openvpn.log", "logs"),
    ("audit.log", "logs/audit"),
)


@dataclass
class HostWindow:
    """One (host, 60s bucket) - the pre-feature-extraction sample unit."""

    host: str
    window_start: int  # epoch seconds, floor(t / 60) * 60
    is_attack: bool = False
    labels: set = field(default_factory=set)
    events_by_source: dict = field(default_factory=dict)  # source -> list[JoinedEvent]

    @property
    def window_id(self) -> str:
        # Deterministic, stable across re-runs: a hash of (host, window_start).
        h = hashlib.sha256(f"{self.host}:{self.window_start}".encode()).hexdigest()[:16]
        return f"{self.host}-{self.window_start}-{h}"


def bucket_events(
    events: list[JoinedEvent], source: str, windows: dict[tuple[str, int], HostWindow]
) -> None:
    """Assign each timestamped event to its (host, 60s bucket), creating buckets
    as needed. Events with no parseable timestamp are skipped and counted by the
    caller via the returned skip count."""
    for e in events:
        if e.timestamp is None:
            continue
        w_start = int(e.timestamp // WINDOW_SECONDS) * WINDOW_SECONDS
        key = (e.host, w_start)
        hw = windows.get(key)
        if hw is None:
            hw = HostWindow(host=e.host, window_start=w_start)
            windows[key] = hw
        hw.events_by_source.setdefault(source, []).append(e)
        if e.is_attack:
            hw.is_attack = True
            hw.labels |= set(e.labels)


@dataclass
class BuildReport:
    replica: str
    hosts_processed: list = field(default_factory=list)
    sources_joined: dict = field(default_factory=dict)  # host -> [source names]
    n_windows: int = 0
    n_attack_windows: int = 0
    per_host_prevalence: dict = field(default_factory=dict)
    n_events_skipped_no_timestamp: int = 0
    errors: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "replica": self.replica,
            "hosts_processed": self.hosts_processed,
            "sources_joined": self.sources_joined,
            "n_windows": self.n_windows,
            "n_attack_windows": self.n_attack_windows,
            "prevalence": round(self.n_attack_windows / max(self.n_windows, 1), 4),
            # Reported per-host because it is NOT uniform: one continuous-exfiltration
            # host can be >90% positive while others sit under 3%. Pooling into a
            # single number would hide that (see NEGATIVE_RESULTS.md).
            "per_host_prevalence": self.per_host_prevalence,
            "n_events_skipped_no_timestamp": self.n_events_skipped_no_timestamp,
            "errors": self.errors,
        }


def build_host_windows(
    *, replica_dir: Path, capture_year: int, hosts: list[str] | None = None
) -> tuple[dict[tuple[str, int], HostWindow], BuildReport]:
    """Join every known labelled source for every host in a replica and bucket the
    results into 60s (host, window) units.

    ``replica_dir`` is the streaming-unpack extract_dir (must already contain
    ``gather/`` and ``labels/`` for this replica; see ``unpack.extract``).
    """
    gather_root = replica_dir / "gather"
    labels_root = replica_dir / "labels"
    report = BuildReport(replica=replica_dir.name)

    if hosts is None:
        hosts = sorted(
            p.name for p in gather_root.iterdir() if p.is_dir() and p.name != "attacker_0"
        )

    windows: dict[tuple[str, int], HostWindow] = {}
    for host in hosts:
        joined_here = []
        for base_name, subpath in KNOWN_SOURCES:
            gdir = gather_root / host / subpath
            ldir = labels_root / host / subpath
            if not (gdir / base_name).exists() and not any(gdir.glob(f"{base_name}.*")):
                continue
            try:
                events, jreport = join_source(
                    host=host,
                    base_name=base_name,
                    gather_dir=gdir,
                    labels_dir=ldir,
                    capture_year=capture_year,
                )
            except LabelJoinError as exc:
                report.errors.append(f"{host}/{base_name}: {exc}")
                continue
            bucket_events(events, base_name, windows)
            joined_here.append(base_name)
            report.n_events_skipped_no_timestamp += sum(1 for e in events if e.timestamp is None)
        if joined_here:
            report.hosts_processed.append(host)
            report.sources_joined[host] = joined_here

    report.n_windows = len(windows)
    report.n_attack_windows = sum(1 for w in windows.values() if w.is_attack)

    per_host: dict[str, list[int]] = {}
    for w in windows.values():
        n, a = per_host.setdefault(w.host, [0, 0])
        per_host[w.host][0] = n + 1
        per_host[w.host][1] = a + (1 if w.is_attack else 0)
    report.per_host_prevalence = {
        h: {"n_windows": n, "n_attack": a, "prevalence": round(a / max(n, 1), 4)}
        for h, (n, a) in sorted(per_host.items())
    }
    return windows, report
