"""A leaked split must FAIL the build, not produce a quietly inflated number."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.eval.leakage import LeakageError, assert_no_group_overlap
from tessera.eval.splits import SplitError, r2_group_split, r2_grouped


def test_disjoint_groups_pass():
    info = assert_no_group_overlap(np.array([1, 2, 3]), np.array([4, 5]), name="host_id")
    assert info["n_shared"] == 0 and info["asserted_disjoint"]


@pytest.mark.mustfail
def test_overlapping_groups_raise():
    with pytest.raises(LeakageError, match="overlap between train and test"):
        assert_no_group_overlap(np.array([1, 2, 3]), np.array([3, 4]), name="host_id")


@pytest.mark.mustfail
def test_deliberately_leaked_split_is_refused():
    """Inject the leak directly: a split whose test hosts also appear in train."""
    rng = np.random.default_rng(0)
    n = 600
    host = rng.integers(0, 5, n)  # only 5 long-lived hosts -> unavoidable overlap

    class Leaked:
        name, regime = "leaked", "deliberately leaked"
        train_idx = np.arange(0, 400)
        test_idx = np.arange(200, 600)  # overlaps train by construction

    with pytest.raises(LeakageError):
        assert_no_group_overlap(host[Leaked.train_idx], host[Leaked.test_idx], name="host_id")


@pytest.mark.mustfail
def test_long_lived_entity_cannot_be_chronologically_disjoint():
    """R2 must refuse the impossible combination rather than return an empty split."""
    rng = np.random.default_rng(0)
    n = 2000
    t = np.sort(rng.random(n) * 86400 * 5)
    host = rng.integers(0, 12, n)  # spans the whole capture
    with pytest.raises(SplitError, match="collapsed to an empty partition"):
        r2_grouped(t, {"host_id": host})


def test_entity_disjoint_split_is_actually_disjoint():
    rng = np.random.default_rng(0)
    host = rng.integers(0, 40, 4000)
    s = r2_group_split(host, seed=0)
    assert not (set(host[s.train_idx]) & set(host[s.test_idx]))
    assert not (set(host[s.train_idx]) & set(host[s.val_idx]))
    assert len(s.test_idx) > 0 and len(s.val_idx) > 0
