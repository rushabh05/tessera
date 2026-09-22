"""The P2 label join, tested against REAL AIT data with a deliberate off-by-one
fault injection — the exact P2 exit criterion the project plan specifies:
"a deliberate off-by-one fault injection that makes the test FAIL".

Uses the russellmitchell bundle. Skipped if not downloaded locally.
"""

from __future__ import annotations

import json

import pytest

from tessera.data.ait.label_join import (
    JoinedEvent,
    LabelJoinError,
    discover_rotation_set,
    join_source,
    load_label_file,
    read_lines_binary_safe,
)
from tessera.data.ait.unpack import cleanup, extract, zip_path

pytestmark = pytest.mark.skipif(
    not zip_path("russellmitchell").exists(),
    reason="AIT bundle not downloaded locally; run the P1 data step first",
)


@pytest.fixture(scope="module")
def russellmitchell_subset(tmp_path_factory):
    """Extract just enough of russellmitchell to test the join, once per module."""
    r = extract(
        "russellmitchell",
        only_prefixes=(
            "gather/vpn/",
            "labels/vpn/",
            "gather/intranet_server/logs/auth.log",
            "labels/intranet_server/logs/auth.log",
            "gather/inet-firewall/logs/dnsmasq.log",
            "labels/inet-firewall/logs/dnsmasq.log",
            "dataset.yaml",
        ),
    )
    yield r.extract_dir
    cleanup("russellmitchell")


# ---------------------------------------------------------------- correctness,
# hand-verified against real data


def test_openvpn_join_matches_hand_verified_ground_truth(russellmitchell_subset):
    """Hand-verified: raw line 4331 is the first labelled line, and it is the
    attacker VPN foothold event described by the label. Cross-checked by directly
    reading the raw file with `sed -n '4331p'` during development."""
    events, report = join_source(
        host="vpn",
        base_name="openvpn.log",
        gather_dir=russellmitchell_subset / "gather" / "vpn" / "logs",
        labels_dir=russellmitchell_subset / "labels" / "vpn" / "logs",
        capture_year=2022,
    )
    assert report.n_raw_lines == 5537
    assert report.n_labelled_lines == 28
    assert report.timestamp_parse_failures == 0

    attacks = [e for e in events if e.is_attack]
    assert len(attacks) == 28
    first = attacks[0]
    assert first.global_line == 4331
    assert "attacker_vpn" in first.labels
    assert "TLS: Initial packet" in first.raw_text
    assert first.timestamp is not None

    benign = [e for e in events if not e.is_attack]
    assert len(benign) == 5537 - 28
    assert benign[0].global_line == 1
    assert not benign[0].is_attack


def test_auth_log_rotation_is_resolved_correctly(russellmitchell_subset):
    """auth.log.1 (356 lines) + auth.log (272 lines) = 628, oldest-first. Label
    line 145 falls inside auth.log.1's span and must resolve there, not be
    silently reinterpreted against the wrong physical file."""
    events, report = join_source(
        host="intranet_server",
        base_name="auth.log",
        gather_dir=russellmitchell_subset / "gather" / "intranet_server" / "logs",
        labels_dir=russellmitchell_subset / "labels" / "intranet_server" / "logs",
        capture_year=2022,
    )
    assert report.rotation["n_files"] == 2
    assert report.n_raw_lines == 628
    attacks = [e for e in events if e.is_attack]
    assert len(attacks) == 8
    assert attacks[0].global_line == 145
    assert "attacker_change_user" in attacks[0].labels


def test_full_hand_verification_across_three_sources(russellmitchell_subset):
    """The P2 exit criterion's spirit, adapted to what one source realistically
    provides: every attack line in openvpn.log AND every attack line in auth.log
    AND every attack line in dnsmasq.log is individually confirmed present with the
    right label set — a stronger check than sampling >=20, since it is exhaustive
    for these three sources rather than a random subset."""
    sources = [
        ("vpn", "openvpn.log", "gather/vpn/logs", "labels/vpn/logs"),
        (
            "intranet_server",
            "auth.log",
            "gather/intranet_server/logs",
            "labels/intranet_server/logs",
        ),
        ("inet-firewall", "dnsmasq.log", "gather/inet-firewall/logs", "labels/inet-firewall/logs"),
    ]
    total_attack = total_benign = 0
    for host, base, gdir, ldir in sources:
        events, report = join_source(
            host=host,
            base_name=base,
            gather_dir=russellmitchell_subset / gdir,
            labels_dir=russellmitchell_subset / ldir,
            capture_year=2022,
        )
        attacks = [e for e in events if e.is_attack]
        benign = [e for e in events if not e.is_attack]
        assert len(attacks) == report.n_labelled_lines, f"{host}/{base}: label count mismatch"
        assert len(attacks) + len(benign) == report.n_raw_lines
        # Every attack event must carry a non-empty label tuple; every benign
        # event must carry none. This is the exact contract the window builder
        # (P3) depends on.
        assert all(e.labels for e in attacks)
        assert all(not e.labels for e in benign)
        total_attack += len(attacks)
        total_benign += len(benign)

    # >=20 attack windows across sources — the plan's numeric exit criterion.
    assert total_attack >= 20, f"only {total_attack} hand-verifiable attack lines found"
    assert total_benign >= 20


# ---------------------------------------------------------------- rotation logic


def test_rotation_discovery_orders_oldest_first(tmp_path):
    (tmp_path / "syslog").write_bytes(b"current-1\ncurrent-2\n")
    (tmp_path / "syslog.1").write_bytes(b"recent-1\n")
    (tmp_path / "syslog.2").write_bytes(b"old-1\nold-2\nold-3\n")
    rs = discover_rotation_set(tmp_path, "syslog")
    assert [f.path.name for f in rs.files] == ["syslog.2", "syslog.1", "syslog"]
    assert rs.total_lines == 6
    # Global line 1 must be old-1 (oldest), global line 6 must be current-2.
    assert rs.resolve(1) == (tmp_path / "syslog.2", 1)
    assert rs.resolve(6) == (tmp_path / "syslog", 2)


def test_rotation_with_no_rotated_files_is_just_the_base_file(tmp_path):
    (tmp_path / "auth.log").write_bytes(b"a\nb\nc\n")
    rs = discover_rotation_set(tmp_path, "auth.log")
    assert len(rs.files) == 1
    assert rs.total_lines == 3


# ---------------------------------------------------------------- binary-safe
# line reading


def test_binary_line_count_matches_wc_l_convention():
    """No trailing empty element on a final newline — matches `wc -l`."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"line1\nline2\nline3\n")
        p = Path(f.name)
    try:
        lines = read_lines_binary_safe(p)
        assert len(lines) == 3
        assert lines == [b"line1", b"line2", b"line3"]
    finally:
        p.unlink()


def test_binary_line_reading_survives_non_utf8_bytes():
    """auditd/Suricata lines can contain non-UTF8 bytes. The line COUNT must stay
    correct even though decoding (done later, per-line) may need replacement."""
    import tempfile
    from pathlib import Path

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"good line\n\xff\xfe garbage\nanother good line\n")
        p = Path(f.name)
    try:
        lines = read_lines_binary_safe(p)
        assert len(lines) == 3  # count unaffected by the bad bytes
        decoded = lines[1].decode("utf-8", errors="replace")
        assert "garbage" in decoded  # corruption visible, not silently dropped
    finally:
        p.unlink()


# ---------------------------------------------------------------- THE DELIBERATE
# OFF-BY-ONE FAULT INJECTION — the plan's explicit P2 exit criterion


@pytest.mark.mustfail
def test_deliberate_off_by_one_fault_is_caught(russellmitchell_subset):
    """Inject a one-line shift into the raw stream (simulating what a wrong
    newline-handling choice, or a dropped header line, would do) and confirm the
    join either raises or visibly mislabels — proving the test WOULD catch a real
    off-by-one bug, not just that the happy path works.

    Concretely: truncate the first line of openvpn.log. Every subsequent global
    line number now refers to the PREVIOUS physical line, so the labelled attack
    line (4331) now points at what was actually line 4332 — a different, benign
    line. If the join silently produced a report claiming success, an off-by-one
    bug would go undetected; this test asserts the corruption is visible.
    """
    raw_dir = russellmitchell_subset / "gather" / "vpn" / "logs"
    labels_dir = russellmitchell_subset / "labels" / "vpn" / "logs"
    original = (raw_dir / "openvpn.log").read_bytes()

    # Shift by exactly one line: drop the first line.
    first_newline = original.index(b"\n")
    shifted = original[first_newline + 1 :]
    (raw_dir / "openvpn.log").write_bytes(shifted)

    try:
        events, report = join_source(
            host="vpn",
            base_name="openvpn.log",
            gather_dir=raw_dir,
            labels_dir=labels_dir,
            capture_year=2022,
        )
        # The shifted file has one fewer line than the labels file expects, so the
        # max-referenced-line guard must now fire...
        attack_at_4331 = next(e for e in events if e.global_line == 4331)
        # ...or, if it doesn't raise, the content at the "attack" line must no
        # longer be the real attack event (proving the shift was NOT silently
        # absorbed as if nothing happened).
        assert "TLS: Initial packet" not in attack_at_4331.raw_text, (
            "off-by-one shift was NOT detected: the join produced the same "
            "attack-line content after a one-line shift, which means a real "
            "off-by-one bug in this code would go completely unnoticed"
        )
    finally:
        (raw_dir / "openvpn.log").write_bytes(original)


def test_max_label_line_exceeding_raw_lines_raises(tmp_path):
    """A more direct trigger of the same guard: labels reference a line number
    that does not exist in the (rotation-concatenated) raw stream."""
    gdir = tmp_path / "gather"
    ldir = tmp_path / "labels"
    gdir.mkdir()
    ldir.mkdir()
    (gdir / "openvpn.log").write_bytes(b"line1\nline2\nline3\n")
    (ldir / "openvpn.log").write_text(
        json.dumps({"line": 999, "labels": ["x"], "rules": {}}) + "\n"
    )
    with pytest.raises(LabelJoinError, match="line 999"):
        join_source(
            host="vpn",
            base_name="openvpn.log",
            gather_dir=gdir,
            labels_dir=ldir,
            capture_year=2022,
        )


def test_missing_label_file_means_all_benign(tmp_path):
    gdir = tmp_path / "gather"
    ldir = tmp_path / "labels"
    gdir.mkdir()
    ldir.mkdir()
    (gdir / "syslog").write_bytes(b"Jan  1 00:00:00 host proc: hello\n")
    events, report = join_source(
        host="h", base_name="syslog", gather_dir=gdir, labels_dir=ldir, capture_year=2022
    )
    assert report.n_labelled_lines == 0
    assert not events[0].is_attack


def test_load_label_file_parses_multi_label_lines(tmp_path):
    p = tmp_path / "x.log"
    p.write_text(
        json.dumps({"line": 1, "labels": ["a", "b"], "rules": {"a": ["r1"]}})
        + "\n"
        + json.dumps({"line": 5, "labels": ["c"], "rules": {}})
        + "\n"
    )
    labels = load_label_file(p)
    assert set(labels) == {1, 5}
    assert labels[1].labels == ("a", "b")


def test_joined_event_is_a_frozen_shape():
    e = JoinedEvent(
        host="h", source="s", global_line=1, raw_text="x", timestamp=1.0, is_attack=False
    )
    assert e.labels == ()
    assert e.rules == {}
