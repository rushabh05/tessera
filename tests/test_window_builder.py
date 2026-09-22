"""Window building, tested against REAL AIT data: joins labelled sources per host
and buckets them into 60s (host, window) units."""

from __future__ import annotations

import pytest

from tessera.data.ait.unpack import cleanup, extract, zip_path
from tessera.data.ait.window_builder import HostWindow, bucket_events, build_host_windows

pytestmark = pytest.mark.skipif(
    not zip_path("russellmitchell").exists(),
    reason="AIT bundle not downloaded locally; run the P1 data step first",
)


@pytest.fixture(scope="module")
def three_host_subset():
    r = extract(
        "russellmitchell",
        only_prefixes=(
            "gather/vpn/logs/openvpn.log",
            "labels/vpn/logs/openvpn.log",
            "gather/intranet_server/logs/auth.log",
            "labels/intranet_server/logs/auth.log",
            "gather/intranet_server/logs/audit/audit.log",
            "labels/intranet_server/logs/audit/audit.log",
            "gather/inet-firewall/logs/dnsmasq.log",
            "labels/inet-firewall/logs/dnsmasq.log",
            "dataset.yaml",
        ),
    )
    yield r.extract_dir
    cleanup("russellmitchell")


def test_builds_windows_for_three_hosts(three_host_subset):
    windows, report = build_host_windows(
        replica_dir=three_host_subset,
        capture_year=2022,
        hosts=["vpn", "intranet_server", "inet-firewall"],
    )
    assert set(report.hosts_processed) == {"vpn", "intranet_server", "inet-firewall"}
    assert report.n_windows > 0
    assert report.errors == []
    assert report.n_events_skipped_no_timestamp == 0


def test_per_host_prevalence_is_not_uniform(three_host_subset):
    """The real finding: one host's labelled activity spans nearly the whole
    capture (continuous DNS exfiltration), while others are sparse. Pooling
    would hide this, so per-host reporting must show the spread."""
    _, report = build_host_windows(
        replica_dir=three_host_subset,
        capture_year=2022,
        hosts=["vpn", "intranet_server", "inet-firewall"],
    )
    prev = {h: v["prevalence"] for h, v in report.per_host_prevalence.items()}
    assert prev["vpn"] < 0.05
    assert prev["intranet_server"] < 0.10
    assert prev["inet-firewall"] > 0.80  # the continuous-exfiltration host


def test_attack_window_carries_correct_labels(three_host_subset):
    windows, _ = build_host_windows(
        replica_dir=three_host_subset,
        capture_year=2022,
        hosts=["vpn"],
    )
    attacks = [w for w in windows.values() if w.is_attack]
    assert len(attacks) >= 1
    w = sorted(attacks, key=lambda w: w.window_start)[0]
    assert "attacker_vpn" in w.labels
    assert "openvpn.log" in w.events_by_source


def test_window_id_is_deterministic():
    a = HostWindow(host="h", window_start=1000)
    b = HostWindow(host="h", window_start=1000)
    c = HostWindow(host="h", window_start=1060)
    assert a.window_id == b.window_id
    assert a.window_id != c.window_id


def test_bucket_events_skips_events_with_no_timestamp():
    from tessera.data.ait.label_join import JoinedEvent

    events = [
        JoinedEvent(
            host="h", source="s", global_line=1, raw_text="x", timestamp=None, is_attack=False
        ),
        JoinedEvent(
            host="h",
            source="s",
            global_line=2,
            raw_text="y",
            timestamp=120.0,
            is_attack=True,
            labels=("x",),
        ),
    ]
    windows: dict = {}
    bucket_events(events, "s", windows)
    assert len(windows) == 1
    ((_host, w_start), hw) = next(iter(windows.items()))
    assert w_start == 120  # floor(120 / 60) * 60 = 120
    assert hw.is_attack


def test_unknown_hosts_list_produces_no_windows(three_host_subset):
    windows, report = build_host_windows(
        replica_dir=three_host_subset,
        capture_year=2022,
        hosts=["does-not-exist"],
    )
    assert windows == {}
    assert report.hosts_processed == []
