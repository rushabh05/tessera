"""M4: graph structural features over a per-window (host -> destination /24
subnet) multigraph, built from the same Suricata `eve.json` flow records M2
already parses.

Deliberately NOT a restatement of M2. M2 computes per-window AGGREGATES (byte/
packet sums, protocol mix) that treat every flow independently. M4 asks a
different, genuinely graph-shaped question that requires state ACROSS windows:
is this host talking to peers it has talked to before, or is its peer set
expanding? A host whose destination set is stable looks structurally different
from one whose destination set is growing every window, even if the two have
IDENTICAL M2 byte/packet/protocol statistics - which is exactly the case a
lateral-movement or scanning host would present.

Kept deliberately boring per the architecture's own design note: plain
structural counts into a small MLP, no GNN. Graph CONSTRUCTION (peer identity,
history) is the actual work; the encoder is intentionally trivial.

Destination IPs are reduced to /24 subnets before anything else touches them -
this is the same prefix-preserving pseudonymisation posture as the ledger
threat model (structure preserved, exact host address discarded).
"""

from __future__ import annotations

import ipaddress
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from tessera.data.ait.timestamps import parse_suricata_eve

N_M4_FEATURES = 8
M4_FEATURE_NAMES = (
    "n_unique_peers",
    "n_new_peers",
    "frac_new_peers",
    "peer_entropy",
    "max_peer_share",
    "n_unique_peer_port_pairs",
    "peer_reappearance_rate",
    "cumulative_peer_count",
)
assert len(M4_FEATURE_NAMES) == N_M4_FEATURES


def dest_subnet(ip: str) -> str:
    """Reduce an address to its /24 (IPv4) or a fixed 'ipv6' bucket - the same
    prefix-preserving posture as the ledger's pseudonymisation, applied here for
    feature stability rather than privacy (an exact /32 destination would make
    almost every peer a singleton, destroying the entropy/concentration signal)."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "invalid"
    if addr.version == 4:
        octets = ip.split(".")
        return f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
    return "ipv6"


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


@dataclass
class _HostPeerHistory:
    """Running state for one host, carried across windows in temporal order."""

    ever_seen: set = field(default_factory=set)
    prev_window_peers: set = field(default_factory=set)


def stream_eve_graph_windows(eve_path, *, window_seconds: int = 60) -> dict[int, np.ndarray]:
    """Stream one host's eve.json, in TIMESTAMP ORDER, maintaining peer history
    across windows, and emit one M4 feature vector per window.

    Unlike M2 (order-independent aggregation), this genuinely needs temporal
    order - "new peer" and "reappearance" are meaningless without it - so the
    events are sorted by timestamp before the single pass that builds state.
    """
    events: list[tuple[float, str, int]] = []  # (ts, subnet, dest_port)
    with eve_path.open("r", encoding="utf-8", errors="replace") as f:
        import json as _json

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if rec.get("event_type") != "flow":
                continue
            ts = parse_suricata_eve(rec)
            if ts is None:
                continue
            dip = rec.get("dest_ip")
            if not dip:
                continue
            events.append((ts, dest_subnet(dip), rec.get("dest_port") or 0))

    events.sort(key=lambda e: e[0])

    windows: dict[int, np.ndarray] = {}
    history = _HostPeerHistory()
    cur_w_start: int | None = None
    peer_counts: Counter = Counter()
    peer_port_pairs: set = set()

    def flush(w_start: int) -> None:
        peers = set(peer_counts)
        new_peers = peers - history.ever_seen
        reappearing = peers & history.prev_window_peers

        v = np.zeros(N_M4_FEATURES, dtype=np.float32)
        v[0] = len(peers)
        v[1] = len(new_peers)
        v[2] = len(new_peers) / max(len(peers), 1)
        v[3] = _shannon_entropy(peer_counts)
        v[4] = (max(peer_counts.values()) / sum(peer_counts.values())) if peer_counts else 0.0
        v[5] = len(peer_port_pairs)
        v[6] = len(reappearing) / max(len(history.prev_window_peers), 1)
        history.ever_seen |= peers
        v[7] = len(history.ever_seen)
        windows[w_start] = v

        history.prev_window_peers = peers

    for ts, subnet, port in events:
        w_start = int(ts // window_seconds) * window_seconds
        if cur_w_start is None:
            cur_w_start = w_start
        elif w_start != cur_w_start:
            flush(cur_w_start)
            peer_counts = Counter()
            peer_port_pairs = set()
            cur_w_start = w_start
        peer_counts[subnet] += 1
        peer_port_pairs.add((subnet, port))

    if cur_w_start is not None:
        flush(cur_w_start)

    return windows
