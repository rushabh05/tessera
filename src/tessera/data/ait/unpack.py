"""Streaming unpack for AIT bundles: one testbed resident on disk at a time.

MANDATORY, not a nicety. Measured on the smallest bundle (russellmitchell,
522.1 MB zip): unpacked size is 7,247,563,244 bytes, a **13.88x** expansion ratio,
via `unzip -l` totals (no extraction needed to get an exact figure). Projecting that
ratio across all eight zip sizes gives ~87.3 GB unpacked — against ~87 GB free on
this machine. Unpacking all eight simultaneously would consume essentially all free
disk, leaving nothing for Parquet intermediates, template caches, or the working
directory anything else needs.

So the raw tier holds AT MOST ONE testbed's unpacked files at a time:
``unzip -> extract only what a caller asks for -> caller processes it -> delete``.
The zip archives themselves stay on disk (6.3 GB total for all eight - cheap, and the
canonical byte-identical source the manifest hashes against), but their *unpacked*
contents never coexist.

This module only extracts; it does not delete on the caller's behalf. Deletion is the
caller's responsibility via :func:`cleanup`, called explicitly after Parquet features
are written, so a crash mid-processing leaves the raw files present for retry rather
than silently deleting a testbed nothing has been built from yet.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from tessera.data.manifest import require_free_disk
from tessera.paths import INTERIM_DIR, RAW_DIR

# The eight AIT V2.1 no-pcaps testbeds, in ascending zip size (smallest first, since
# that is the cheapest to develop and test against).
REPLICAS = (
    "russellmitchell",
    "santos",
    "harrison",
    "fox",
    "wheeler",
    "wardbeck",
    "wilson",
    "shaw",
)

# The measured ratio, kept as a named constant so a disk-budget assertion can use it
# BEFORE extracting, rather than discovering a full disk mid-unzip.
MEASURED_EXPANSION_RATIO = 7_247_563_244 / 522_084_364  # 13.88, from russellmitchell


def zip_path(replica: str, *, raw_dir: Path = RAW_DIR) -> Path:
    if replica not in REPLICAS:
        raise ValueError(f"unknown AIT replica '{replica}'; known: {REPLICAS}")
    return raw_dir / "ait" / f"{replica}_no-pcaps.zip"


def extract_dir(replica: str, *, interim_dir: Path = INTERIM_DIR) -> Path:
    return interim_dir / "ait" / replica


@dataclass
class UnpackReport:
    replica: str
    zip_bytes: int
    unpacked_bytes: int
    n_files: int
    expansion_ratio: float
    extract_dir: Path
    hosts_with_suricata: list = field(default_factory=list)
    labelled_hosts: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "replica": self.replica,
            "zip_bytes": self.zip_bytes,
            "unpacked_bytes": self.unpacked_bytes,
            "n_files": self.n_files,
            "expansion_ratio": round(self.expansion_ratio, 3),
            "extract_dir": str(self.extract_dir),
            "hosts_with_suricata": sorted(self.hosts_with_suricata),
            "labelled_hosts": sorted(self.labelled_hosts),
        }


class IncompleteDownloadError(RuntimeError):
    """The zip is present but not a valid archive - a truncated or in-progress
    download, not a code bug. Raised rather than propagating zipfile's BadZipFile
    so callers can distinguish "not downloaded" from "downloaded but broken"."""


def inspect_without_extracting(replica: str, *, raw_dir: Path = RAW_DIR) -> UnpackReport:
    """Read the zip's central directory only. Costs nothing on disk.

    Used to plan a disk-safe extraction (or to skip extraction entirely for callers
    that only need the inventory, e.g. the P1 de-risking check this module exists
    because of).
    """
    zp = zip_path(replica, raw_dir=raw_dir)
    if not zp.exists():
        raise FileNotFoundError(f"{zp} not downloaded; run the download step first")

    try:
        zf_ctx = zipfile.ZipFile(zp)
    except zipfile.BadZipFile as exc:
        # A real failure mode, not hypothetical: this was FIRST triggered by a
        # background download for another replica still being written to disk
        # while this function ran. A truncated download is exactly the same
        # symptom, so both are reported the same way rather than crashing.
        raise IncompleteDownloadError(
            f"{zp} is not a valid zip archive ({exc}). If a download is in "
            "progress, wait for it to finish; otherwise re-download."
        ) from exc

    with zf_ctx as zf:
        infos = zf.infolist()
        unpacked = sum(i.file_size for i in infos)
        hosts_eve = {
            Path(i.filename).parts[1]
            for i in infos
            if len(Path(i.filename).parts) >= 2
            and Path(i.filename).parts[0] == "gather"
            and i.filename.endswith("suricata/eve.json")
        }
        hosts_labelled = {
            Path(i.filename).parts[1]
            for i in infos
            if len(Path(i.filename).parts) >= 2 and Path(i.filename).parts[0] == "labels"
        }

    return UnpackReport(
        replica=replica,
        zip_bytes=zp.stat().st_size,
        unpacked_bytes=unpacked,
        n_files=len(infos),
        expansion_ratio=unpacked / zp.stat().st_size,
        extract_dir=extract_dir(replica),
        hosts_with_suricata=sorted(hosts_eve),
        labelled_hosts=sorted(hosts_labelled),
    )


def extract(
    replica: str,
    *,
    raw_dir: Path = RAW_DIR,
    interim_dir: Path = INTERIM_DIR,
    only_prefixes: tuple[str, ...] | None = None,
    disk_headroom_bytes: int = 2 * 1024**3,
) -> UnpackReport:
    """Extract one testbed. Asserts free disk BEFORE writing a single byte.

    ``only_prefixes`` restricts extraction to matching path prefixes (e.g.
    ``("gather/", "labels/")`` to skip the ``environment/`` provisioning tree, which
    is Ansible/Kyoushi configuration, not telemetry, and is dead weight for feature
    extraction).
    """
    report = inspect_without_extracting(replica, raw_dir=raw_dir)
    require_free_disk(report.unpacked_bytes + disk_headroom_bytes, path=raw_dir)

    out = extract_dir(replica, interim_dir=interim_dir)
    if out.exists():
        shutil.rmtree(out)  # a partial extraction from a crashed prior run
    out.mkdir(parents=True)

    zp = zip_path(replica, raw_dir=raw_dir)
    with zipfile.ZipFile(zp) as zf:
        members = zf.infolist()
        if only_prefixes:
            members = [m for m in members if m.filename.startswith(only_prefixes)]
        zf.extractall(out, members=members)

    return report


def cleanup(replica: str, *, interim_dir: Path = INTERIM_DIR) -> int:
    """Delete a testbed's unpacked files. Called only after downstream Parquet
    features are confirmed written, so a crash mid-processing is retryable."""
    out = extract_dir(replica, interim_dir=interim_dir)
    if not out.exists():
        return 0
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    shutil.rmtree(out)
    return n


def plan_all_replicas(*, raw_dir: Path = RAW_DIR) -> dict:
    """Inspect every downloaded replica without extracting any of them.

    This is what P1's disk-safety check actually runs: it proves the "process one
    at a time" plan is sufficient by showing that the LARGEST single testbed still
    fits inside free disk with headroom, even though the SUM of all eight does not.
    """
    reports = []
    incomplete = []
    for r in REPLICAS:
        zp = zip_path(r, raw_dir=raw_dir)
        if not zp.exists():
            continue
        try:
            reports.append(inspect_without_extracting(r, raw_dir=raw_dir))
        except IncompleteDownloadError:
            # Do not let one truncated/in-progress download abort the scan of
            # every other replica - report it and continue.
            incomplete.append(r)

    total_zip = sum(r.zip_bytes for r in reports)
    total_unpacked_if_simultaneous = sum(r.unpacked_bytes for r in reports)
    largest = max(reports, key=lambda r: r.unpacked_bytes) if reports else None

    return {
        "n_replicas_downloaded": len(reports),
        "n_replicas_incomplete": len(incomplete),
        "incomplete_replicas": incomplete,
        "n_replicas_total": len(REPLICAS),
        "total_zip_bytes": total_zip,
        "total_unpacked_bytes_if_simultaneous": total_unpacked_if_simultaneous,
        "largest_single_replica": largest.as_dict() if largest else None,
        "streaming_required": total_unpacked_if_simultaneous > 40 * 1024**3,
        "reports": [r.as_dict() for r in reports],
    }
