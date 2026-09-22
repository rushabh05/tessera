"""Synthetic demo data: correct shape, license-safe (no real AIT data), and
locks in the two real bugs found while building it."""

from __future__ import annotations

import numpy as np

from tessera.demo.synthetic_demo_data import (
    _DEMO_HOST_BUCKETS,
    DEMO_HOSTS,
    make_demo_windows,
)
from tessera.features.m3_identity import host_bucket
from tessera.features.pipeline import M3_SLICE, N_TOTAL_FEATURES


def test_shape_and_range():
    X, y, avail = make_demo_windows(n=50, seed=0)
    assert X.shape == (50, N_TOTAL_FEATURES)
    assert y.shape == (50,)
    assert avail.shape == (50, 3)
    assert set(np.unique(y).tolist()) <= {0, 1}
    assert np.isfinite(X).all()
    assert (X >= 0).all()  # every real feature here is non-negative


def test_deterministic_under_seed():
    X1, y1, a1 = make_demo_windows(n=20, seed=3)
    X2, y2, a2 = make_demo_windows(n=20, seed=3)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(a1, a2)


def test_host_bucket_uses_real_hash_values_not_small_integers():
    """Locks in the fix for the real bug found live: host_bucket's true range
    is [10, 47] (a SHA-256 hash of the real training hostnames, mod 64); a
    first version substituted rng.integers(0, 3), silently pushing every
    synthetic window out of the distribution the network was calibrated for
    and saturating every score near 1.0 regardless of the other features."""
    X, _, _ = make_demo_windows(n=100, seed=0)
    observed = set(X[:, M3_SLICE.start].tolist())
    expected = {float(host_bucket(h)) for h in DEMO_HOSTS}
    assert observed <= expected, (
        f"host_bucket values {observed} are not among the real hash buckets "
        f"{expected} for the training hostnames"
    )
    # And they must not be small sequential integers - the exact regression.
    assert not observed <= {0.0, 1.0, 2.0}


def test_demo_host_buckets_are_computed_not_hardcoded():
    """The buckets must come from the real hash function, so if the hashing
    scheme in m3_identity.py ever changes, this module's constants update
    with it rather than silently going stale."""
    assert tuple(host_bucket(h) for h in DEMO_HOSTS) == _DEMO_HOST_BUCKETS


def test_lognormal_sampling_produces_realistic_scale():
    """A regression guard for the OTHER real bug: a plain clipped-Gaussian
    sampler (tried first) produced values dominated by noise for
    high-variance features, destroying class separability entirely. This
    checks the sampled values land within a sane order of magnitude of the
    real calibration statistics rather than exploding or collapsing to zero."""
    X, y, _ = make_demo_windows(n=500, seed=1)
    # feature 16 (an M2 byte-count feature) has a huge real mean (~30k-56k);
    # log-normal sampling should produce mostly-reasonable values, not mostly
    # zeros (a collapsed distribution) or a mean many orders of magnitude off.
    assert X[:, 16].mean() > 100  # not collapsed to ~0
    assert X[:, 16].mean() < 1e8  # not exploded


def test_availability_zeroes_absent_modality_features():
    from tessera.features.pipeline import M1_SLICE, M4_SLICE

    X, _, avail = make_demo_windows(n=200, seed=2)
    for i in range(len(avail)):
        if not avail[i, 0]:
            assert (X[i, M1_SLICE] == 0).all()
        if not avail[i, 2]:
            assert (X[i, M4_SLICE] == 0).all()
    assert avail[:, 1].all()  # M2 always available, matching the real measured rate


def test_no_real_ait_identifiers_anywhere_in_module():
    """A static guard: the module must never contain a real AIT log line,
    IP address, or timestamp - only the three generic host ROLE names (not
    licensed data) and derived summary statistics."""
    import inspect

    import tessera.demo.synthetic_demo_data as mod

    src = inspect.getsource(mod)
    for forbidden in ("192.168.", "172.19.", "2022-01-2", "smith.russellmitchell"):
        assert forbidden not in src, f"found a real-AIT-looking string: {forbidden!r}"
