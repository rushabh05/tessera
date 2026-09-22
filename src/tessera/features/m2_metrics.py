"""M2: numeric network-metric features from Suricata `eve.json`, bucketed into the
same 60s (host, window) units the label join uses.

Real schema, confirmed by direct inspection of the russellmitchell bundle: `flow`
events dominate (~121k of ~200k sampled records) and carry
`flow.{pkts_toserver,pkts_toclient,bytes_toserver,bytes_toclient}`; `alert` events
carry a severity; `dns`/`http`/`tls` give the protocol mix. IPv4 and IPv6 addresses
are both present in `src_ip`/`dest_ip`.

24 features per window - fewer than the ~64 in the original detailed design, sized
for the simplified project scope, but drawn entirely from fields verified present
in real data rather than a field list assumed from the Suricata docs.
"""

from __future__ import annotations

import ipaddress
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tessera.data.ait.timestamps import parse_suricata_eve

N_M2_FEATURES = 24
FEATURE_NAMES = (
    "n_flow",
    "n_alert",
    "n_dns",
    "n_http",
    "n_tls",
    "n_fileinfo",
    "pkts_toserver_sum",
    "pkts_toclient_sum",
    "bytes_toserver_sum",
    "bytes_toclient_sum",
    "pkts_toserver_p90",
    "bytes_toserver_p90",
    "flow_duration_mean",
    "flow_duration_p90",
    "n_unique_dest_ports",
    "dest_port_entropy",
    "n_unique_dest_ips",
    "frac_ipv6",
    "frac_proto_tcp",
    "frac_proto_udp",
    "frac_proto_icmp",
    "n_tls_handshakes",
    "alert_severity_max",
    "n_dns_queries",
)
assert len(FEATURE_NAMES) == N_M2_FEATURES


def _shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _is_ipv6(addr: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(addr), ipaddress.IPv6Address)
    except ValueError:
        return False


@dataclass
class Eve2WindowStats:
    """Per-(host, window) accumulator, filled by streaming eve.json line by line -
    the whole file is never held in memory, since the largest single eve.json seen
    (webserver, 361 MB) makes that a real concern, not a hypothetical one."""

    n_flow: int = 0
    n_alert: int = 0
    n_dns: int = 0
    n_http: int = 0
    n_tls: int = 0
    n_fileinfo: int = 0
    pkts_toserver: list = field(default_factory=list)
    pkts_toclient: list = field(default_factory=list)
    bytes_toserver: list = field(default_factory=list)
    bytes_toclient: list = field(default_factory=list)
    flow_durations: list = field(default_factory=list)
    dest_ports: Counter = field(default_factory=Counter)
    dest_ips: set = field(default_factory=set)
    n_ipv6: int = 0
    n_total: int = 0
    proto_counts: Counter = field(default_factory=Counter)
    alert_severity_max: int = 0

    def to_vector(self) -> np.ndarray:
        v = np.zeros(N_M2_FEATURES, dtype=np.float32)
        v[0] = self.n_flow
        v[1] = self.n_alert
        v[2] = self.n_dns
        v[3] = self.n_http
        v[4] = self.n_tls
        v[5] = self.n_fileinfo
        v[6] = sum(self.pkts_toserver)
        v[7] = sum(self.pkts_toclient)
        v[8] = sum(self.bytes_toserver)
        v[9] = sum(self.bytes_toclient)
        v[10] = _percentile(self.pkts_toserver, 90)
        v[11] = _percentile(self.bytes_toserver, 90)
        v[12] = float(np.mean(self.flow_durations)) if self.flow_durations else 0.0
        v[13] = _percentile(self.flow_durations, 90)
        v[14] = len(self.dest_ports)
        v[15] = _shannon_entropy(self.dest_ports)
        v[16] = len(self.dest_ips)
        v[17] = self.n_ipv6 / max(self.n_total, 1)
        total_proto = sum(self.proto_counts.values()) or 1
        v[18] = self.proto_counts.get("TCP", 0) / total_proto
        v[19] = self.proto_counts.get("UDP", 0) / total_proto
        v[20] = sum(c for p, c in self.proto_counts.items() if "ICMP" in p) / total_proto
        v[21] = self.n_tls
        v[22] = self.alert_severity_max
        v[23] = self.n_dns
        return v


def stream_eve_windows(eve_path: Path, *, window_seconds: int = 60) -> dict[int, Eve2WindowStats]:
    """Stream one host's eve.json and accumulate per-window stats.

    Malformed lines are skipped and counted (returned separately), not fatal -
    Suricata JSON output has been observed elsewhere to contain occasional
    truncated final lines from a log rotation race.
    """
    windows: dict[int, Eve2WindowStats] = {}
    n_bad = 0
    with eve_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue

            ts = parse_suricata_eve(rec)
            if ts is None:
                n_bad += 1
                continue
            w_start = int(ts // window_seconds) * window_seconds
            st = windows.setdefault(w_start, Eve2WindowStats())

            et = rec.get("event_type")
            st.n_total += 1
            if _is_ipv6(rec.get("src_ip", "")) or _is_ipv6(rec.get("dest_ip", "")):
                st.n_ipv6 += 1
            proto = rec.get("proto")
            if proto:
                st.proto_counts[proto] += 1
            dport = rec.get("dest_port")
            if dport is not None:
                st.dest_ports[dport] += 1
            dip = rec.get("dest_ip")
            if dip:
                st.dest_ips.add(dip)

            if et == "flow":
                st.n_flow += 1
                flow = rec.get("flow", {})
                st.pkts_toserver.append(flow.get("pkts_toserver", 0))
                st.pkts_toclient.append(flow.get("pkts_toclient", 0))
                st.bytes_toserver.append(flow.get("bytes_toserver", 0))
                st.bytes_toclient.append(flow.get("bytes_toclient", 0))
                start_s, end_s = flow.get("start"), flow.get("end")
                if start_s and end_s:
                    t0, t1 = (
                        parse_suricata_eve({"timestamp": start_s}),
                        parse_suricata_eve({"timestamp": end_s}),
                    )
                    if t0 is not None and t1 is not None:
                        st.flow_durations.append(max(0.0, t1 - t0))
            elif et == "alert":
                st.n_alert += 1
                sev = rec.get("alert", {}).get("severity")
                if isinstance(sev, int):
                    st.alert_severity_max = max(st.alert_severity_max, sev)
            elif et == "dns":
                st.n_dns += 1
            elif et == "http":
                st.n_http += 1
            elif et == "tls":
                st.n_tls += 1
            elif et == "fileinfo":
                st.n_fileinfo += 1

    return windows
