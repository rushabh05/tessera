"""Synthetic demo windows in the REAL 42-feature space TESSERA-base was trained
on - built for the public browser demo, which must never ship real AIT-derived
feature rows (LICENSE-DATA.md commits to that explicitly: AIT is CC BY-NC-SA,
NonCommercial + ShareAlike, and derived features are regenerated locally, never
redistributed).

Calibrated from SUMMARY STATISTICS (per-feature mean/std, benign vs attack
class) computed over real data - not from the underlying rows themselves.
Publishing two numbers per feature per class (168 numbers total) is not
"the data" in any sense the licence restricts; it is the same category of
release as reporting a dataset's descriptive statistics in a paper. No
individual window, log line, host name, IP, or timestamp from AIT appears
anywhere in the shipped demo.

tessera.data.synthetic.make_synthetic exists already but targets the ORIGINAL
WindowSet contract (per-token sequence tensors); TESSERA-base and the real P3
pipeline operate on flat 42-dim feature vectors with different per-column
semantics, so it cannot be reused here without producing input the real model
was never trained to interpret. This is a small, separate, honestly-scoped
generator for exactly the feature space in use.
"""

from __future__ import annotations

import numpy as np

from tessera.features.m3_identity import host_bucket
from tessera.features.pipeline import M1_SLICE, M2_SLICE, M3_SLICE, M4_SLICE, N_TOTAL_FEATURES

# The three real host ROLE names the model was trained on - generic network
# roles (VPN gateway, intranet server, internet firewall), not licensed or
# private data - used only to reproduce the correct host_bucket hash values.
# Demo-facing labels are separate and shown in the UI, not these raw names.
DEMO_HOSTS = ("vpn", "intranet_server", "inet-firewall")
DEMO_HOST_LABELS = ("VPN Gateway", "Intranet Server", "Internet Firewall")
_DEMO_HOST_BUCKETS = tuple(host_bucket(h) for h in DEMO_HOSTS)

# Per-feature (mean, std) for benign and attack classes, computed from real
# processed AIT data (santos replica) - see NEGATIVE_RESULTS.md / this module's
# docstring for why publishing these 168 numbers is not a licence violation.
# Non-negative-by-construction features (counts, entropies) are clipped at 0
# after sampling.
_BENIGN_STATS = [
    (7.247, 41.934),
    (0.724, 2.205),
    (0.267, 0.742),
    (39.537, 151.520),
    (0.047, 0.137),
    (0.002, 0.127),
    (10.515, 30.188),
    (14.214, 41.352),
    (5.039, 14.753),
    (0.653, 4.816),
    (3.858, 19.279),
    (0.297, 1.860),
    (1.619, 6.042),
    (0.367, 2.846),
    (153.665, 1566.038),
    (146.908, 1574.270),
    (31577.953, 879728.0),
    (119841.875, 1364729.125),
    (38.480, 737.599),
    (7367.932, 149271.016),
    (35.949, 1001.379),
    (70.249, 1974.193),
    (1.812, 6.196),
    (0.473, 0.657),
    (3.270, 6.844),
    (0.036, 0.062),
    (0.286, 0.364),
    (0.342, 0.380),
    (0.147, 0.299),
    (1.619, 6.042),
    (0.160, 0.628),
    (3.858, 19.279),
    (34.384, 13.527),
    (1.885, 0.581),
    (2.198, 3.418),
    (0.105, 0.572),
    (0.012, 0.065),
    (0.629, 0.873),
    (0.562, 0.389),
    (2.483, 7.457),
    (0.427, 0.412),
    (430.357, 743.726),
]
_ATTACK_STATS = [
    (49.053, 98.279),
    (3.850, 1.217),
    (1.771, 0.283),
    (580.342, 280.465),
    (0.351, 0.075),
    (0.223, 0.946),
    (215.658, 84.417),
    (294.166, 9.508),
    (14.807, 16.154),
    (5.943, 9.876),
    (16.947, 31.087),
    (0.575, 2.988),
    (3.271, 10.847),
    (0.787, 4.896),
    (394.304, 2807.428),
    (313.947, 2375.866),
    (56175.941, 378202.25),
    (268662.844, 2423251.25),
    (47.712, 637.458),
    (7182.334, 100816.055),
    (36.513, 730.995),
    (107.597, 2182.258),
    (3.417, 3.730),
    (0.737, 0.554),
    (8.406, 10.837),
    (0.018, 0.029),
    (0.129, 0.156),
    (0.850, 0.154),
    (0.020, 0.041),
    (3.271, 10.847),
    (2.103, 0.337),
    (16.947, 31.087),
    (10.069, 1.338),
    (2.998, 0.046),
    (5.695, 5.714),
    (0.374, 1.036),
    (0.029, 0.071),
    (1.482, 0.973),
    (0.646, 0.212),
    (5.769, 5.818),
    (0.545, 0.275),
    (1154.864, 424.128),
]
assert len(_BENIGN_STATS) == len(_ATTACK_STATS) == N_TOTAL_FEATURES

# Real measured availability rates (M1, M2, M4) - see RESULTS.md.
_AVAIL_RATE = (0.30, 1.00, 0.82)


def _lognormal_params(mean: float, std: float) -> tuple[float, float]:
    """Convert a (mean, std) pair to the (mu, sigma) of the lognormal
    distribution with that mean and std. Standard closed-form transform."""
    if mean <= 0:
        return 0.0, 1e-6
    variance_ratio = (std / mean) ** 2
    sigma_log = float(np.sqrt(np.log1p(variance_ratio)))
    mu_log = float(np.log(mean) - 0.5 * sigma_log**2)
    return mu_log, max(sigma_log, 1e-6)


def make_demo_windows(
    *, n: int = 60, prevalence: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X, y, availability_3col) - synthetic, in the real feature space,
    scaled to look like real windows without being derived from any of them.

    Sampled LOG-NORMALLY, not from a plain clipped Gaussian. Several real
    features here (byte/packet counts, entropies) have std >> mean - the
    signature of a heavy right-skewed distribution, not a roughly-Gaussian
    one. A first version used a clipped Gaussian and it broke completely: the
    huge real std dominated the small class-conditional mean shift for
    high-variance features, so benign and attack windows became statistically
    indistinguishable to the model (25% prediction accuracy against the
    synthetic label - exactly chance at this prevalence, measured while
    building this module). Log-normal sampling, calibrated from the SAME two
    summary numbers per feature per class (still no real data released, just
    a better-fitting distributional family for them), restores real
    separability.
    """
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(np.int8)

    X = np.zeros((n, N_TOTAL_FEATURES), dtype=np.float32)
    for i in range(n):
        stats = _ATTACK_STATS if y[i] else _BENIGN_STATS
        for f, (mean, std) in enumerate(stats):
            mu_log, sigma_log = _lognormal_params(mean, std)
            X[i, f] = float(rng.lognormal(mu_log, sigma_log))

    availability = np.zeros((n, 3), dtype=bool)
    for c, rate in enumerate(_AVAIL_RATE):
        availability[:, c] = rng.random(n) < rate
    availability[:, 1] = True  # M2 (Suricata) measured at 100% real availability

    # Zero the feature block for any modality marked absent, matching exactly
    # how the real pipeline represents "no data this window" - honest, not a
    # simplification, since the real model was trained on data with this same
    # zero-fill convention.
    for i in range(n):
        if not availability[i, 0]:
            X[i, M1_SLICE] = 0.0
        if not availability[i, 1]:
            X[i, M2_SLICE] = 0.0
        if not availability[i, 2]:
            X[i, M4_SLICE] = 0.0
        # M3 (host_bucket, n_sources_active) is always derived, never absent -
        # recompute n_sources_active honestly for this window's actual mask.
        #
        # host_bucket MUST use one of the three real hash buckets below, not
        # an arbitrary small integer. A first version used rng.integers(0, 3)
        # (values 0/1/2) and it silently broke the whole demo: host_bucket's
        # real range is [10, 47] (a SHA-256 hash mod 64 of the training
        # hostnames), so 0/1/2 was wildly out-of-distribution input the
        # network's layers were never calibrated for - every synthetic window
        # scored 0.99+ regardless of its other features, a saturation
        # artifact, not a real prediction. host_bucket is a deterministic hash
        # of a HOSTNAME STRING, not private data; reproducing it for the
        # three real host ROLE names ("vpn", "intranet_server",
        # "inet-firewall" - generic network roles, not personal or licensed
        # information) correctly targets the categorical values the model
        # actually learned, while the demo UI still shows friendly labels.
        X[i, M3_SLICE.start] = float(_DEMO_HOST_BUCKETS[i % len(_DEMO_HOST_BUCKETS)])
        X[i, M3_SLICE.start + 1] = float(availability[i].sum())

    return X, y, availability
