"""M4 graph structural features: subnet reduction and cross-window peer history."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from tessera.features.m4_graph import (
    N_M4_FEATURES,
    dest_subnet,
    stream_eve_graph_windows,
)


def test_dest_subnet_reduces_ipv4_to_slash24():
    assert dest_subnet("172.19.130.55") == "172.19.130.0/24"
    assert dest_subnet("10.0.0.1") == "10.0.0.0/24"


def test_dest_subnet_buckets_ipv6_separately():
    assert dest_subnet("fe80::1") == "ipv6"


def test_dest_subnet_invalid_address_is_labelled():
    assert dest_subnet("not-an-ip") == "invalid"


def _write_eve(records: list[dict]) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return Path(f.name)


def _flow(ts: str, dest_ip: str, dest_port: int = 443) -> dict:
    return {
        "timestamp": ts,
        "event_type": "flow",
        "dest_ip": dest_ip,
        "dest_port": dest_port,
        "flow": {},
    }


def test_first_window_all_peers_are_new():
    path = _write_eve(
        [
            _flow("2022-01-21T00:00:01.000000+0000", "10.0.0.1"),
            _flow("2022-01-21T00:00:02.000000+0000", "10.0.1.1"),
        ]
    )
    try:
        windows = stream_eve_graph_windows(path)
        assert len(windows) == 1
        v = next(iter(windows.values()))
        assert v.shape == (N_M4_FEATURES,)
        assert v[0] == 2  # n_unique_peers
        assert v[1] == 2  # n_new_peers - everything is new in the first window
        assert v[2] == 1.0  # frac_new_peers
        assert v[7] == 2  # cumulative_peer_count
    finally:
        path.unlink()


def test_repeated_peer_across_windows_is_not_new_but_reappears():
    path = _write_eve(
        [
            _flow("2022-01-21T00:00:01.000000+0000", "10.0.0.1"),
            _flow("2022-01-21T00:01:01.000000+0000", "10.0.0.1"),  # same subnet, next window
        ]
    )
    try:
        windows = stream_eve_graph_windows(path)
        assert len(windows) == 2
        keys = sorted(windows)
        w0, w1 = windows[keys[0]], windows[keys[1]]
        assert w0[1] == 1  # window 0: new
        assert w1[1] == 0  # window 1: NOT new (seen in window 0)
        assert w1[6] == 1.0  # reappearance rate: was in the immediately prior window
        assert w1[7] == 1  # cumulative stays at 1 - same peer, not a new one
    finally:
        path.unlink()


def test_cumulative_peer_count_is_monotone_non_decreasing():
    path = _write_eve(
        [_flow("2022-01-21T00:00:01.000000+0000", f"10.0.{i}.1") for i in range(5)]
        + [_flow(f"2022-01-21T00:0{m}:01.000000+0000", "10.0.0.1") for m in range(1, 4)]
    )
    try:
        windows = stream_eve_graph_windows(path)
        counts = [windows[k][7] for k in sorted(windows)]
        assert counts == sorted(counts), "cumulative_peer_count decreased somewhere"
    finally:
        path.unlink()


def test_diverse_peers_have_higher_entropy_than_one_dominant_peer():
    diverse = _write_eve(
        [_flow("2022-01-21T00:00:01.000000+0000", f"10.0.{i}.1") for i in range(8)]
    )
    concentrated = _write_eve(
        [_flow("2022-01-21T00:00:01.000000+0000", "10.0.0.1")] * 7
        + [_flow("2022-01-21T00:00:01.000000+0000", "10.0.1.1")]
    )
    try:
        v_diverse = next(iter(stream_eve_graph_windows(diverse).values()))
        v_conc = next(iter(stream_eve_graph_windows(concentrated).values()))
        assert v_diverse[3] > v_conc[3]  # entropy
        assert v_conc[4] > v_diverse[4]  # max_peer_share
    finally:
        diverse.unlink()
        concentrated.unlink()


def test_empty_file_produces_no_windows():
    path = _write_eve([])
    try:
        assert stream_eve_graph_windows(path) == {}
    finally:
        path.unlink()


def test_non_flow_events_are_ignored():
    path = _write_eve(
        [
            {
                "timestamp": "2022-01-21T00:00:01.000000+0000",
                "event_type": "dns",
                "dest_ip": "10.0.0.1",
            },
        ]
    )
    try:
        assert stream_eve_graph_windows(path) == {}
    finally:
        path.unlink()


def test_malformed_line_does_not_crash_the_stream():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json\n")
        f.write(json.dumps(_flow("2022-01-21T00:00:01.000000+0000", "10.0.0.1")) + "\n")
    path = Path(f.name)
    try:
        windows = stream_eve_graph_windows(path)
        assert len(windows) == 1
    finally:
        path.unlink()
