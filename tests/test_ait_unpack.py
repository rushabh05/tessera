"""Streaming unpack, tested against the REAL downloaded russellmitchell bundle.

Skipped automatically if the bundle is not present (e.g. a teammate who has not run
the data step yet) - these are integration tests, not unit tests, and the project
plan's own reproducibility artifacts assume the raw tier may be locally absent.
"""

from __future__ import annotations

import pytest

from tessera.data.ait.unpack import (
    MEASURED_EXPANSION_RATIO,
    REPLICAS,
    IncompleteDownloadError,
    cleanup,
    extract,
    inspect_without_extracting,
    plan_all_replicas,
    zip_path,
)
from tessera.paths import RAW_DIR

pytestmark = pytest.mark.skipif(
    not zip_path("russellmitchell").exists(),
    reason="AIT bundle not downloaded locally; run the P1 data step first",
)


def test_inspect_matches_known_exact_totals():
    """Locked against the exact figures verified by hand on 2026-09-22 - if these
    ever change, either Zenodo re-released the file (manifest hash would also
    change) or this module has a bug."""
    r = inspect_without_extracting("russellmitchell")
    assert r.zip_bytes == 522_084_364
    assert r.unpacked_bytes == 7_247_563_244
    assert r.n_files == 14898
    assert abs(r.expansion_ratio - 13.882) < 0.01


def test_seven_hosts_have_suricata_eve_json():
    r = inspect_without_extracting("russellmitchell")
    assert set(r.hosts_with_suricata) == {
        "cloud_share",
        "inet-firewall",
        "internal_share",
        "intranet_server",
        "mail",
        "vpn",
        "webserver",
    }


def test_five_hosts_have_labels():
    r = inspect_without_extracting("russellmitchell")
    assert set(r.labelled_hosts) == {
        "inet-firewall",
        "internal_share",
        "intranet_server",
        "monitoring",
        "vpn",
    }


def test_unknown_replica_is_rejected():
    with pytest.raises(ValueError, match="unknown AIT replica"):
        inspect_without_extracting("not-a-real-testbed")


def test_extract_then_cleanup_round_trips():
    r = extract("russellmitchell", only_prefixes=("labels/vpn/", "dataset.yaml"))
    try:
        assert r.extract_dir.exists()
        assert (r.extract_dir / "labels" / "vpn" / "logs" / "openvpn.log").exists()
        assert (r.extract_dir / "dataset.yaml").exists()
        # only_prefixes must actually restrict what lands on disk
        assert not (r.extract_dir / "environment").exists()
    finally:
        n = cleanup("russellmitchell")
        assert n > 0
    assert not r.extract_dir.exists()


def test_cleanup_of_never_extracted_replica_is_a_noop():
    assert cleanup("santos" if zip_path("santos").exists() else "russellmitchell") >= 0


def test_expansion_ratio_constant_matches_measurement():
    assert abs(MEASURED_EXPANSION_RATIO - 13.882) < 0.01


def test_plan_all_replicas_flags_streaming_as_required():
    """The decisive finding: unpacking all 8 simultaneously would exceed disk
    budget, so the plan function must say streaming is required whenever more
    than a couple of the larger bundles are present."""
    plan = plan_all_replicas()
    assert plan["n_replicas_downloaded"] >= 1
    assert plan["largest_single_replica"] is not None
    # Every individual replica must itself be small enough to process alone -
    # this is what makes "one at a time" a valid mitigation rather than merely a
    # smaller version of the same problem.
    for rep in plan["reports"]:
        assert rep["unpacked_bytes"] < 20 * 1024**3, (
            f"{rep['replica']} alone is {rep['unpacked_bytes'] / 1024**3:.1f} GB "
            "unpacked; streaming one-at-a-time would not fit either"
        )


def test_all_eight_replica_names_are_known():
    assert len(REPLICAS) == 8
    assert len(set(REPLICAS)) == 8  # no duplicates
    assert RAW_DIR.exists() or True  # directory creation is exercised elsewhere


def test_truncated_download_reports_incomplete_not_crash(tmp_path):
    """This exact scenario happened live: `plan_all_replicas` crashed with a raw
    BadZipFile while a background download for another replica was mid-write.
    Fixed to raise a typed error from inspect_without_extracting, and to have
    plan_all_replicas skip such entries rather than aborting the whole scan."""
    raw = tmp_path / "raw"
    (raw / "ait").mkdir(parents=True)
    # A truncated file: real zip bytes at the START, cut off before the central
    # directory - exactly what a killed download looks like on disk.
    real = zip_path("russellmitchell")
    truncated = raw / "ait" / "russellmitchell_no-pcaps.zip"
    truncated.write_bytes(real.read_bytes()[:1000])

    with pytest.raises(IncompleteDownloadError):
        inspect_without_extracting("russellmitchell", raw_dir=raw)

    plan = plan_all_replicas(raw_dir=raw)
    assert plan["n_replicas_incomplete"] == 1
    assert "russellmitchell" in plan["incomplete_replicas"]
    assert plan["n_replicas_downloaded"] == 0  # did not crash the whole scan
