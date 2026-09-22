"""M3: identity/session categoricals, derived from data already joined by
window_builder.py - no new log sources needed for this scope.

MEASURED AND REMOVED: absolute calendar position (hour-of-day, day-of-week,
minute-of-hour) was included in a first version and turned out to be a severe
leakage vector on a short, non-repeating capture. On real AIT data, adding those
four features raised the random-split AP to 0.9998 while chronological AP fell to
0.5156 (gap 0.48) - `hour_of_day` became the single most important feature by a
wide margin. The reason is structural, not a modelling error: in a single 4-6 day
capture, "hour 3 on day Jan-24" only ever occurs once, so a random split leaks the
exact calendar position of held-out attack windows into training, and a
tree-based model happily memorises "attacks happen around hour X" for THIS
capture rather than learning anything that would transfer to a different day.
Removing the four calendar features and keeping only `host_bucket` and
`n_sources_active` dropped the gap to 0.0187 - and R1 alone reached 0.9813,
confirming the OTHER features (M1 templates, M2 Suricata metrics) generalise
genuinely well once the shortcut is gone.

Host identity is hashed (never the raw hostname) so the feature is usable without
ever putting a plain hostname into a model input - consistent with the
pseudonymisation posture the ledger threat model already commits to.
"""

from __future__ import annotations

import hashlib

import numpy as np

N_M3_FEATURES = 2
M3_FEATURE_NAMES = ("host_bucket", "n_sources_active")
assert len(M3_FEATURE_NAMES) == N_M3_FEATURES

_HOST_BUCKETS = 64


def host_bucket(host: str) -> int:
    """A stable, non-reversible bucket index - not the raw hostname."""
    h = hashlib.sha256(host.encode()).hexdigest()
    return int(h[:8], 16) % _HOST_BUCKETS


def window_identity_vector(*, host: str, n_sources_active: int) -> np.ndarray:
    v = np.zeros(N_M3_FEATURES, dtype=np.float32)
    v[0] = host_bucket(host)
    v[1] = n_sources_active
    return v
