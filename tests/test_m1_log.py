"""M1 log-template mining: timestamp stripping and Drain3 wiring."""

from __future__ import annotations

from tessera.features.m1_log import (
    M1WindowAccumulator,
    make_template_miner,
    strip_timestamp_prefix,
)


def test_strip_syslog_prefix():
    line = "Jan 23 06:25:05 intranet-server CRON[22883]: pam_unix(cron:session): session closed for user root"
    out = strip_timestamp_prefix(line, source_kind="syslog")
    assert not out.startswith("Jan")
    assert "CRON" in out


def test_strip_ait_iso_prefix():
    line = "2022-01-21 00:09:11 jhall/192.168.230.165:46011 TLS: soft reset"
    out = strip_timestamp_prefix(line, source_kind="ait_iso")
    assert out == "jhall/192.168.230.165:46011 TLS: soft reset"


def test_strip_unknown_kind_is_a_noop():
    line = "some raw line with no known timestamp format"
    assert strip_timestamp_prefix(line, source_kind=None) == line
    assert strip_timestamp_prefix(line, source_kind="auditd") == line


def test_similar_lines_merge_into_one_template_once_timestamp_stripped():
    """The exact case that motivated stripping: two structurally identical CRON
    lines a few minutes apart must merge into ONE template once the varying
    timestamp is removed - they did NOT merge when fed with timestamps intact."""
    miner = make_template_miner()
    lines = [
        "Jan 23 06:25:05 intranet-server CRON[22883]: pam_unix(cron:session): session closed for user root",
        "Jan 23 07:12:33 intranet-server CRON[24001]: pam_unix(cron:session): session closed for user root",
    ]
    clusters = set()
    for line in lines:
        stripped = strip_timestamp_prefix(line, source_kind="syslog")
        r = miner.add_log_message(stripped)
        clusters.add(r["cluster_id"])
    assert len(clusters) == 1, f"expected one merged cluster, got {clusters}"


def test_structurally_different_lines_stay_in_different_clusters():
    miner = make_template_miner()
    r1 = miner.add_log_message("session opened for user root by (uid=0)")
    r2 = miner.add_log_message("TLS: soft reset sec=100/100 bytes=1/1 pkts=1/0")
    assert r1["cluster_id"] != r2["cluster_id"]


def test_window_accumulator_vector_shape_and_bounds():
    from tessera.features.m1_log import N_M1_FEATURES

    acc = M1WindowAccumulator()
    acc.template_ids[1] += 3
    acc.template_ids[2] += 1
    acc.line_lengths = [10, 20, 30]
    v = acc.to_vector()
    assert v.shape == (N_M1_FEATURES,)
    assert v[0] == 4  # n_events
    assert v[1] == 2  # n_unique_templates
    assert v[2] >= 0  # entropy non-negative
    assert v[3] == 1  # dominant template id (id 1 has count 3)
    assert abs(v[4] - 0.75) < 1e-6  # dominant_template_frac = 3/4


def test_empty_window_accumulator_is_all_zero():
    from tessera.features.m1_log import N_M1_FEATURES

    v = M1WindowAccumulator().to_vector()
    assert v.shape == (N_M1_FEATURES,)
    assert (v == 0).all()
