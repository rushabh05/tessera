"""Property-based tests for the transparency log.

These are the invariants the tamper-evidence claim rests on, so they are checked
across many tree sizes and indices rather than one hand-picked example.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tessera.ledger.merkle import (
    MerkleLog,
    canonical_bytes,
    merkle_tree_hash,
    verify_consistency,
    verify_inclusion,
)


def _log(n: int) -> MerkleLog:
    log = MerkleLog()
    for i in range(n):
        log.append_json({"window_id": f"w{i}", "verdict": i % 2, "score": i / max(n, 1)})
    return log


@settings(max_examples=60, deadline=None)
@given(n=st.integers(min_value=1, max_value=64))
def test_every_leaf_inclusion_proof_verifies(n):
    log, root = _log(n), None
    root = log.root()
    for i in range(n):
        assert log.inclusion_proof(i).verify(root), f"leaf {i} of {n} failed"


@settings(max_examples=60, deadline=None)
@given(n=st.integers(min_value=1, max_value=48), m=st.integers(min_value=1, max_value=48))
def test_every_consistency_proof_verifies(n, m):
    first, second = min(n, m), max(n, m)
    log = _log(second)
    assert verify_consistency(
        first, log.root(first), second, log.root(second), log.consistency_proof(first, second)
    ), f"consistency {first} -> {second} failed"


@settings(max_examples=40, deadline=None)
@given(n=st.integers(min_value=2, max_value=40))
def test_tampered_leaf_breaks_inclusion(n):
    log = _log(n)
    root_before = log.root()
    log.tamper(n // 2, canonical_bytes({"window_id": "FORGED", "verdict": 0, "score": 0.0}))
    # The retained proof no longer matches the rebuilt tree...
    assert log.root() != root_before
    # ...and the old proof does not verify against the new root.
    assert not log.inclusion_proof(n // 2).verify(root_before)


@settings(max_examples=40, deadline=None)
@given(n=st.integers(min_value=4, max_value=40))
def test_tampered_history_breaks_consistency_against_retained_head(n):
    """The core guarantee: editing a past entry is detectable by an auditor holding
    an earlier signed tree head."""
    log = _log(n)
    first = n // 2
    retained_root = log.root(first)  # the auditor keeps this
    log.tamper(0, canonical_bytes({"window_id": "REWRITTEN", "verdict": 0, "score": 0.0}))
    assert not verify_consistency(
        first, retained_root, n, log.root(n), log.consistency_proof(first, n)
    )


def test_append_only_root_of_past_size_is_stable():
    log = _log(8)
    r4 = log.root(4)
    for i in range(8, 20):
        log.append_json({"window_id": f"w{i}", "verdict": 0, "score": 0.0})
    assert log.root(4) == r4, "a past root changed; the log is not append-only"


def test_known_rfc6962_vectors():
    """Anchor the implementation to the spec, not just to itself."""
    import hashlib

    assert merkle_tree_hash([]) == hashlib.sha256(b"").digest()
    log = MerkleLog()
    i0 = log.append(b"a")
    assert log.root() == log.inclusion_proof(i0).leaf  # single leaf: root == leaf hash


def test_proof_size_grows_logarithmically():
    small = _log(16).inclusion_proof(0)
    large = _log(1024).inclusion_proof(0)
    assert len(small.path) == 4 and len(large.path) == 10


@pytest.mark.mustfail
def test_out_of_range_index_is_refused():
    with pytest.raises(IndexError):
        _log(4).inclusion_proof(9)


def test_forged_proof_does_not_verify():
    log = _log(16)
    p = log.inclusion_proof(3)
    bad = list(p.path)
    bad[0] = b"\x00" * 32
    assert not verify_inclusion(p.leaf, 3, 16, bad, log.root())
