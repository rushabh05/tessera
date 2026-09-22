"""M1: log template mining via Drain3, bucketed into the same 60s (host, window)
units as M2 and the labels.

Templates are mined on the log BODY, with the leading timestamp stripped first -
feeding Drain3 the raw line (timestamp included) was tried first and produces
templates that vary by time-of-day rather than by message structure, since Drain3
treats a differing token position as a wildcard candidate. Verified: two CRON
session lines a few minutes apart correctly merge into one template only once the
timestamp prefix is removed.

One TemplateMiner is shared across a whole replica (not per host, not per source),
matching the architecture's "shared template vocabulary" design: the same
structural pattern (e.g. a PAM session message) can appear on more than one host.

For the classical/GBDT view this reduces to summary statistics (count, unique
templates, entropy, dominant template id) - the same reduction WindowSet.flat_features()
already applies to the full M1 sequence tensor, so this module can feed either
representation without redesigning the contract.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from tessera.data.ait.timestamps import (
    _AIT_ISO_RE,
    _APACHE_RE,
    _AUDITD_RE,
    _SYSLOG_RE,
    parser_for_filename,
)

N_M1_FEATURES = 8
M1_FEATURE_NAMES = (
    "n_events",
    "n_unique_templates",
    "template_entropy",
    "dominant_template_id",
    "dominant_template_frac",
    "n_new_templates",
    "mean_line_length",
    "max_line_length",
)
assert len(M1_FEATURE_NAMES) == N_M1_FEATURES

# Reuse the timestamp regexes as PREFIX STRIPPERS: everything up to end-of-match is
# the timestamp, the remainder is the message body Drain3 should cluster on.
_PREFIX_PATTERNS = {
    "syslog": _SYSLOG_RE,
    "apache": _APACHE_RE,
    "ait_iso": _AIT_ISO_RE,
    "auditd": _AUDITD_RE,  # not a prefix match (embedded), stripped separately below
}


def strip_timestamp_prefix(line: str, *, source_kind: str | None) -> str:
    """Remove the leading timestamp so Drain3 clusters on message structure, not
    time-of-day. auditd's timestamp is embedded mid-line (inside msg=audit(...)),
    so it is left in place - it is a small, bounded, high-cardinality token Drain3
    already treats as a wildcard candidate after a couple of examples, and
    stripping it would require reconstructing the line rather than trimming a
    prefix.
    """
    if source_kind in ("syslog", "apache", "ait_iso"):
        pat = _PREFIX_PATTERNS[source_kind]
        m = pat.match(line) if source_kind != "apache" else pat.search(line)
        if m:
            return line[m.end() :].lstrip(" :")
    return line


def make_template_miner(*, max_clusters: int = 4096) -> TemplateMiner:
    """A Drain3 miner with in-memory persistence and a bounded cluster count, so a
    single malformed or highly variable source cannot grow the vocabulary
    unboundedly.

    ``TemplateMinerConfig()`` already carries usable defaults without a config
    file - calling ``.load(None)`` was tried first and raises inside configparser,
    since drain3 expects `.load` to receive a real path or nothing at all.
    """
    cfg = TemplateMinerConfig()
    cfg.profiling_enabled = False
    cfg.drain_max_clusters = max_clusters
    return TemplateMiner(config=cfg)


@dataclass
class M1WindowAccumulator:
    """Per-(host, window) accumulator, filled as events are mined."""

    template_ids: Counter = field(default_factory=Counter)
    line_lengths: list = field(default_factory=list)
    n_new_templates: int = 0

    def to_vector(self) -> list[float]:
        import numpy as np

        n = sum(self.template_ids.values())
        v = np.zeros(N_M1_FEATURES, dtype=np.float32)
        v[0] = n
        v[1] = len(self.template_ids)
        if n > 0:
            v[2] = -sum((c / n) * math.log2(c / n) for c in self.template_ids.values() if c > 0)
            dom_id, dom_count = self.template_ids.most_common(1)[0]
            v[3] = dom_id
            v[4] = dom_count / n
        v[5] = self.n_new_templates
        if self.line_lengths:
            v[6] = float(np.mean(self.line_lengths))
            v[7] = float(max(self.line_lengths))
        return v


def mine_events_into_windows(
    events_by_host_source: dict, *, miner: TemplateMiner, window_seconds: int = 60
) -> dict[tuple[str, int], M1WindowAccumulator]:
    """Mine templates for a set of already-joined events (from label_join.py /
    window_builder.py) and bucket the results into (host, window) accumulators.

    ``events_by_host_source`` maps (host, source_base_name) -> list[JoinedEvent].
    """
    windows: dict[tuple[str, int], M1WindowAccumulator] = {}
    for (host, source), events in events_by_host_source.items():
        kind = parser_for_filename(source)
        for e in events:
            if e.timestamp is None:
                continue
            body = strip_timestamp_prefix(e.raw_text, source_kind=kind)
            result = miner.add_log_message(body)
            w_start = int(e.timestamp // window_seconds) * window_seconds
            key = (host, w_start)
            acc = windows.setdefault(key, M1WindowAccumulator())
            acc.template_ids[result["cluster_id"]] += 1
            acc.line_lengths.append(len(body))
            if result["change_type"] == "cluster_created":
                acc.n_new_templates += 1
    return windows
