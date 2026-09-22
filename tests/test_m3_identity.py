"""M3 identity features - and the guard against the calendar-leakage regression."""

from __future__ import annotations

from tessera.features.m3_identity import (
    N_M3_FEATURES,
    host_bucket,
    window_identity_vector,
)


def test_host_bucket_is_stable_and_discriminating():
    assert host_bucket("vpn") == host_bucket("vpn")
    assert host_bucket("vpn") != host_bucket("inet-firewall")
    assert 0 <= host_bucket("vpn") < 64


def test_host_bucket_never_leaks_the_raw_hostname():
    b = host_bucket("very-identifiable-hostname-12345")
    assert isinstance(b, int)
    assert str(b) != "very-identifiable-hostname-12345"


def test_window_identity_vector_shape():
    v = window_identity_vector(host="vpn", n_sources_active=2)
    assert v.shape == (N_M3_FEATURES,)
    assert v[1] == 2


def test_no_epoch_timestamp_parameter_exists():
    """The old signature took window_start (epoch seconds) and derived hour/day/
    minute from it. That parameter is gone - passing it must fail loudly rather
    than silently being ignored, so a caller cannot accidentally reintroduce the
    leaky computation by resurrecting an old call site."""
    import inspect

    sig = inspect.signature(window_identity_vector)
    assert "window_start" not in sig.parameters
