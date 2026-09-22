"""Per-file-type labelling-coverage table — the second P2 exit criterion the plan
requires before any missing-modality experiment (C2) can be trusted.

Only ~8 of the ~20 file types AIT collects carry labels, and system-monitoring
logs are almost entirely absent from that set (confirmed: 1 host-day out of ~28
possible in russellmitchell — one CPU-metric log, one host, one day). This
directly confounds the availability-masking claim: if labelling coverage
correlates with file type, then "modality present/absent" partly encodes "this
window COULD have been labelled" rather than a genuine deployment property.

This module produces the coverage table from the zip's central directory alone —
no extraction needed, so it can run for all eight bundles cheaply even before P1's
"process one at a time" pipeline exists.
"""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from tessera.data.ait.unpack import REPLICAS, zip_path


@dataclass
class FileTypeCoverage:
    """One (host, file-type) pair's presence in gather/ vs labels/."""

    host: str
    base_name: str
    present_in_gather: bool
    present_in_labels: bool
    label_bytes: int

    @property
    def labelled(self) -> bool:
        return self.present_in_gather and self.present_in_labels


_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


def _base_name(path: str) -> str:
    """Normalise a filename to a FILE TYPE, collapsing two things that are NOT
    distinct types but were measured to inflate a naive per-filename count from
    ~20 real telemetry sources to 300+:

    * a trailing numeric ROTATION suffix (auth.log.1, auth.log.2, ...)
    * a leading DATE prefix on logstash-derived per-day monitoring exports
      (2022-01-22-system.filesystem.log, 2022-01-23-system.filesystem.log, ...) -
      confirmed present under gather/monitoring/logs/logstash/<host>/: the same
      metric, one file per calendar day, which is not a distinct file type any
      more than auth.log.1 vs auth.log.2 is.
    """
    name = Path(path).name
    name = _DATE_PREFIX_RE.sub("", name)
    parts = name.rsplit(".", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def scan_replica(replica: str, *, raw_dir: Path | None = None) -> list[FileTypeCoverage]:
    """Enumerate every (host, file-type) under gather/ and labels/ for one replica,
    reading only the zip's central directory."""
    zp = zip_path(replica, raw_dir=raw_dir) if raw_dir else zip_path(replica)
    gather: dict[tuple[str, str], bool] = {}
    labels: dict[tuple[str, str], int] = {}

    with zipfile.ZipFile(zp) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            if len(parts) < 2:
                continue
            root, host = parts[0], parts[1]
            if root not in ("gather", "labels"):
                continue
            base = _base_name(info.filename)
            # Exclude configuration/rules trees - not telemetry, and every host
            # has near-identical copies that would dominate the coverage table.
            if "/configs/" in info.filename or "/rules/" in info.filename:
                continue
            # Exclude decoy documents planted by the attack simulation (invoices,
            # spreadsheets under downloads/) and the attacker's own capture host -
            # neither is defender telemetry a real deployment would collect.
            if "/downloads/" in info.filename or host == "attacker_0":
                continue
            key = (host, base)
            if root == "gather":
                gather[key] = True
            else:
                labels[key] = labels.get(key, 0) + info.file_size

    all_keys = set(gather) | set(labels)
    return [
        FileTypeCoverage(
            host=h,
            base_name=b,
            present_in_gather=(h, b) in gather,
            present_in_labels=(h, b) in labels,
            label_bytes=labels.get((h, b), 0),
        )
        for h, b in sorted(all_keys)
    ]


@dataclass
class CoverageSummary:
    """Two granularities, deliberately both reported:

    * ``n_*_types`` - distinct file TYPES across the whole deployment (auth.log,
      apache access, eve.json, ...), collapsing across hosts. Close to the plan's
      original "~20 file types" estimate.
    * ``n_*_host_type_pairs`` - (host, type) pairs. The operationally relevant
      number for the C2 confound: a window's per-host availability depends on
      whether THAT host emits THAT type, not on whether the type exists anywhere
      in the deployment.
    """

    replica: str
    n_gather_types: int
    n_labelled_types: int
    n_gather_host_type_pairs: int
    n_labelled_host_type_pairs: int
    labelled_types: list = field(default_factory=list)
    labelled_host_type_pairs: list = field(default_factory=list)
    unlabelled_host_type_pairs: list = field(default_factory=list)

    @property
    def type_coverage_fraction(self) -> float:
        return self.n_labelled_types / max(self.n_gather_types, 1)

    @property
    def host_type_coverage_fraction(self) -> float:
        return self.n_labelled_host_type_pairs / max(self.n_gather_host_type_pairs, 1)

    def as_dict(self) -> dict:
        return {
            "replica": self.replica,
            "n_gather_types": self.n_gather_types,
            "n_labelled_types": self.n_labelled_types,
            "type_coverage_fraction": round(self.type_coverage_fraction, 3),
            "n_gather_host_type_pairs": self.n_gather_host_type_pairs,
            "n_labelled_host_type_pairs": self.n_labelled_host_type_pairs,
            "host_type_coverage_fraction": round(self.host_type_coverage_fraction, 3),
            "labelled_types": sorted(self.labelled_types),
            "labelled_host_type_pairs": [f"{h}/{b}" for h, b in self.labelled_host_type_pairs],
            "unlabelled_host_type_pairs_sample": [
                f"{h}/{b}" for h, b in self.unlabelled_host_type_pairs[:15]
            ],
            "n_unlabelled_host_type_pairs": len(self.unlabelled_host_type_pairs),
        }


def summarise(rows: list[FileTypeCoverage], *, replica: str) -> CoverageSummary:
    gather_pairs = {(r.host, r.base_name) for r in rows if r.present_in_gather}
    labelled_pairs = {(r.host, r.base_name) for r in rows if r.labelled}
    unlabelled_pairs = sorted(gather_pairs - labelled_pairs)

    gather_types = {b for _, b in gather_pairs}
    labelled_types = {b for _, b in labelled_pairs}

    return CoverageSummary(
        replica=replica,
        n_gather_types=len(gather_types),
        n_labelled_types=len(labelled_types),
        n_gather_host_type_pairs=len(gather_pairs),
        n_labelled_host_type_pairs=len(labelled_pairs),
        labelled_types=sorted(labelled_types),
        labelled_host_type_pairs=sorted(labelled_pairs),
        unlabelled_host_type_pairs=unlabelled_pairs,
    )


def render_markdown(summaries: list[CoverageSummary]) -> str:
    """The P2 exit-criterion artifact: a table, not a claim. Reports BOTH
    granularities so the confound is visible at the resolution the C2 experiment
    actually operates at (per host, not per deployment)."""
    lines = [
        "| replica | file types (total / labelled) | host-type pairs (total / labelled) | type coverage | host-type coverage |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(
            f"| `{s.replica}` | {s.n_gather_types} / {s.n_labelled_types} "
            f"| {s.n_gather_host_type_pairs} / {s.n_labelled_host_type_pairs} "
            f"| {s.type_coverage_fraction:.1%} | {s.host_type_coverage_fraction:.1%} |"
        )
    lines.append("")
    lines.append(
        "**Labelled file TYPES, union across replicas** (the plan's original ~20-type framing):"
    )
    labelled_union = sorted({t for s in summaries for t in s.labelled_types})
    for t in labelled_union:
        lines.append(f"- `{t}`")
    lines.append("")
    lines.append(
        "**Labelled (host, type) pairs, union across replicas** (the C2-relevant granularity):"
    )
    pair_union = sorted({p for s in summaries for p in s.labelled_host_type_pairs})
    for h, b in pair_union:
        lines.append(f"- `{h}/{b}`")
    return "\n".join(lines)


def scan_all_downloaded(*, raw_dir: Path | None = None) -> tuple[list[CoverageSummary], str]:
    summaries = []
    for r in REPLICAS:
        zp = zip_path(r, raw_dir=raw_dir) if raw_dir else zip_path(r)
        if not zp.exists():
            continue
        try:
            rows = scan_replica(r, raw_dir=raw_dir)
        except zipfile.BadZipFile:
            continue  # in-progress download; same handling as unpack.py
        summaries.append(summarise(rows, replica=r))
    return summaries, render_markdown(summaries)
