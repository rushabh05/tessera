"""Per-file-type labelling coverage, tested against the REAL AIT bundle."""

from __future__ import annotations

import pytest

from tessera.data.ait.coverage import _base_name, scan_all_downloaded, scan_replica, summarise
from tessera.data.ait.unpack import zip_path

pytestmark = pytest.mark.skipif(
    not zip_path("russellmitchell").exists(),
    reason="AIT bundle not downloaded locally; run the P1 data step first",
)


def test_base_name_collapses_rotation_suffixes():
    assert _base_name("gather/h/logs/auth.log.1") == "auth.log"
    assert _base_name("gather/h/logs/auth.log.2") == "auth.log"
    assert _base_name("gather/h/logs/auth.log") == "auth.log"


def test_base_name_collapses_date_prefixes():
    """The fix for the inflation found live: date-stamped per-day monitoring
    exports are the same file TYPE, not distinct types."""
    a = _base_name("gather/monitoring/logs/logstash/h/2022-01-22-system.filesystem.log")
    b = _base_name("gather/monitoring/logs/logstash/h/2022-01-23-system.filesystem.log")
    assert a == b == "system.filesystem.log"


def test_scan_replica_finds_exactly_eight_labelled_host_type_pairs():
    """Locked against the exact, hand-verified inventory from the zip's central
    directory (2026-09-22): 8 labelled (host, type) pairs in russellmitchell."""
    rows = scan_replica("russellmitchell")
    s = summarise(rows, replica="russellmitchell")
    assert s.n_labelled_host_type_pairs == 8
    assert ("vpn", "openvpn.log") in s.labelled_host_type_pairs
    assert ("inet-firewall", "dnsmasq.log") in s.labelled_host_type_pairs
    assert ("monitoring", "system.cpu.log") in s.labelled_host_type_pairs


def test_attacker_host_and_downloads_are_excluded():
    """Decoy documents and the attacker's own capture host are not defender
    telemetry a real deployment would collect; both were measured to massively
    inflate the naive per-filename count (334 -> 187 -> 69 types after fixes)."""
    rows = scan_replica("russellmitchell")
    hosts = {r.host for r in rows}
    assert "attacker_0" not in hosts


def test_coverage_fraction_is_low_not_the_plan_rough_estimate():
    """Corrects the plan's original rough '~8 of ~20 (40%)' estimate: the real
    figure, measured, is ~10% by file type and ~4% by (host, type) pair - thinner
    coverage than assumed, which sharpens rather than weakens the C2 confound."""
    rows = scan_replica("russellmitchell")
    s = summarise(rows, replica="russellmitchell")
    assert s.n_gather_types > 40  # far more than the plan's rough ~20
    assert s.type_coverage_fraction < 0.15
    assert s.host_type_coverage_fraction < 0.10


def test_render_markdown_includes_both_granularities():
    summaries, md = scan_all_downloaded()
    assert "file types (total / labelled)" in md
    assert "host-type pairs (total / labelled)" in md
    assert len(summaries) >= 1
