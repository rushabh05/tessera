"""The split regimes that form the inflation cascade - this project's central figure.

Reporting R0 through R4 side by side shows how much of a headline number is a
percentage-based-split artifact of how the data was divided, rather than a
detector's actual capability.

* **R0** random stratified - the leakage upper bound, for comparability only.
* **R1** chronological per testbed, with a gap at each boundary so a window
  straddling the cut cannot appear on both sides.
* **R2** R1 plus entity disjointness (host / user / session).
* **R3** leave-one-replica-out. The eight AIT testbeds are parameter- and
  order-randomised replicas of ONE environment and attack repertoire, not eight
  independent organisations, so this measures robustness to that randomisation. It
  is deliberately not called cross-organisation transfer.
* **R4** cross-corpus (AIT <-> CIC-IDS2017-corrected) - the only genuine
  cross-corpus generalisation evidence available here, and therefore non-cuttable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tessera.eval.leakage import LeakageError, assert_no_group_overlap


@dataclass
class Split:
    """One train/val/test partition, carrying the provenance of how it was made."""

    name: str
    regime: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    meta: dict = field(default_factory=dict)

    def sizes(self) -> dict:
        return {
            "n_train": int(len(self.train_idx)),
            "n_val": int(len(self.val_idx)),
            "n_test": int(len(self.test_idx)),
        }

    def positives(self, y: np.ndarray) -> dict:
        y = np.asarray(y).ravel()
        return {
            "pos_train": int(y[self.train_idx].sum()),
            "pos_val": int(y[self.val_idx].sum()),
            "pos_test": int(y[self.test_idx].sum()),
        }

    def as_dict(self, y: np.ndarray | None = None) -> dict:
        d = {"name": self.name, "regime": self.regime, **self.sizes(), **self.meta}
        if y is not None:
            d.update(self.positives(y))
        return d


class SplitError(RuntimeError):
    pass


def assert_partitions_have_positives(split: Split, y: np.ndarray) -> None:
    """Every partition must contain at least one positive.

    AIT packs one multi-step attack into a 4-6 day capture, so a naive chronological
    cut can put the entire attack inside a single partition. Metrics computed on a
    partition with zero positives are undefined, and average precision silently
    returns a meaningless value rather than raising - hence an explicit guard.
    """
    pos = split.positives(y)
    empty = [k for k, v in pos.items() if v == 0]
    if empty:
        raise SplitError(
            f"split '{split.name}' ({split.regime}) has zero positives in: {', '.join(empty)}. "
            "Use attack-aware chronological splitting, or widen the partition."
        )


# ------------------------------------------------------------------ R0


def r0_random(
    y: np.ndarray, *, seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2
) -> Split:
    """Random stratified split. The leakage upper bound, never a capability claim."""
    from sklearn.model_selection import train_test_split

    y = np.asarray(y).ravel()
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
    rel_val = val_frac / (1.0 - test_frac)
    train, val = train_test_split(rest, test_size=rel_val, random_state=seed, stratify=y[rest])
    return Split(
        name="r0_random",
        regime="R0 random stratified (leakage upper bound)",
        train_idx=np.sort(train),
        val_idx=np.sort(val),
        test_idx=np.sort(test),
        meta={
            "seed": seed,
            "caption": "leakage upper bound; for comparability with prior work, not a capability claim",
        },
    )


# ------------------------------------------------------------------ R1


def r1_chronological(
    timestamps: np.ndarray,
    *,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    gap_seconds: float = 600.0,
) -> Split:
    """Chronological 60/20/20 with a gap at each boundary.

    The gap discards windows adjacent to a cut so temporally-correlated neighbours
    cannot sit on both sides of it.
    """
    t = np.asarray(timestamps, dtype=np.float64).ravel()
    order = np.argsort(t, kind="stable")
    n = len(t)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise SplitError(f"chronological split leaves an empty partition at n={n}")

    train = order[:n_train]
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :]

    # Drop from the later partition anything within gap_seconds of the boundary.
    def drop_gap(earlier: np.ndarray, later: np.ndarray) -> np.ndarray:
        if len(earlier) == 0 or len(later) == 0:
            return later
        boundary = t[earlier].max()
        return later[t[later] > boundary + gap_seconds]

    n_pre_gap = (len(train), len(val), len(test))
    val = drop_gap(train, val)
    test = drop_gap(val if len(val) else train, test)

    # The gap must be small relative to the time span. If it empties a partition the
    # split is unusable, and returning it empty would surface later as a confusing
    # downstream error instead of here as the actual cause.
    if len(val) == 0 or len(test) == 0:
        span = float(t.max() - t.min())
        raise SplitError(
            f"chronological gap of {gap_seconds:g}s emptied a partition "
            f"(val {n_pre_gap[1]}->{len(val)}, test {n_pre_gap[2]}->{len(test)}). "
            f"The data spans only {span:g}s, so the gap is {gap_seconds / max(span, 1e-9):.1%} "
            "of the timeline. Reduce gap_seconds, or use a corpus covering a longer "
            "period - real captures span days, where a 600s gap is negligible."
        )

    return Split(
        name="r1_chrono",
        regime="R1 chronological per testbed",
        train_idx=np.sort(train),
        val_idx=np.sort(val),
        test_idx=np.sort(test),
        meta={
            "gap_seconds": gap_seconds,
            "n_dropped_to_gap": int(n - (len(train) + len(val) + len(test))),
        },
    )


# ------------------------------------------------------------------ R2

# Entity lifetime governs which disjointness is even achievable.
#
# A host in an AIT testbed exists for the whole 4-6 day capture, so every host
# present in a chronologically-later partition is also present earlier. Taking a
# chronological cut and then dropping rows whose host appears in train therefore
# removes EVERY row - measured on synthetic data with 12 hosts: 794 rows removed,
# val and test both empty. Host-level disjointness is not obtainable within a single
# testbed; it is delivered instead by R3, which holds out whole testbeds.
#
# Sessions are short-lived, so session-level disjointness costs little and is worth
# enforcing. Hence two distinct regimes rather than one overloaded function:
#
#   r2_grouped     - chronological + disjointness on SHORT-LIVED entities
#   r2_group_split - true entity-disjoint partitioning, chronology sacrificed
#
# Reporting both is more informative than pretending one regime delivers both
# properties, which it cannot.


def r2_grouped(
    timestamps: np.ndarray,
    groups: dict[str, np.ndarray],
    *,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    gap_seconds: float = 600.0,
) -> Split:
    """Chronological, plus disjointness on the given SHORT-LIVED entities.

    Pass sessions here, not hosts. If a group is long-lived the partitions collapse
    and this raises :class:`SplitError` pointing at :func:`r2_group_split`, rather
    than returning a degenerate split that would silently break downstream metrics.
    """
    base = r1_chronological(
        timestamps, val_frac=val_frac, test_frac=test_frac, gap_seconds=gap_seconds
    )
    train, val, test = base.train_idx, base.val_idx, base.test_idx
    n_before = len(val) + len(test)

    removed: dict[str, int] = {}
    for name, g in groups.items():
        g = np.asarray(g).ravel()
        train_vals = set(g[train].tolist())
        before = len(val) + len(test)
        val = val[~np.isin(g[val], list(train_vals))]
        test = test[~np.isin(g[test], list(train_vals))]
        removed[name] = before - (len(val) + len(test))

    if len(val) == 0 or len(test) == 0:
        culprits = [n for n, c in removed.items() if c > 0]
        raise SplitError(
            "R2 chronological+entity-disjoint collapsed to an empty partition "
            f"(removed {n_before - len(val) - len(test)} of {n_before} val/test rows; "
            f"groups responsible: {culprits}). These entities span the whole capture, "
            "so chronological and entity-disjoint cannot hold simultaneously. Enforce "
            "only short-lived entities (sessions) here and use r2_group_split() for "
            "true entity-disjoint partitioning, or rely on R3 for host disjointness."
        )

    split = Split(
        name="r2_grouped",
        regime="R2 chronological + short-lived-entity-disjoint",
        train_idx=np.sort(train),
        val_idx=np.sort(val),
        test_idx=np.sort(test),
        meta={
            "gap_seconds": gap_seconds,
            "removed_for_group_overlap": removed,
            "enforced_groups": sorted(groups),
            "note": "host-level disjointness is NOT claimed here; see R3",
        },
    )

    for name, g in groups.items():
        g = np.asarray(g).ravel()
        assert_no_group_overlap(g[split.train_idx], g[split.test_idx], name=name)
        assert_no_group_overlap(g[split.train_idx], g[split.val_idx], name=f"{name} (val)")
    return split


def r2_group_split(
    groups: np.ndarray, *, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 0
) -> Split:
    """True entity-disjoint partitioning: whole entities are assigned to one side.

    Chronology is sacrificed, which is stated in the regime name so the two
    properties are never conflated in a results table.
    """
    from sklearn.model_selection import GroupShuffleSplit

    groups = np.asarray(groups).ravel()
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise SplitError(f"entity-disjoint split needs >= 3 distinct groups, got {n_groups}")
    idx = np.arange(len(groups))

    rest, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed).split(
            idx, groups=groups
        )
    )
    rel_val = val_frac / (1.0 - test_frac)
    tr_rel, va_rel = next(
        GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed).split(
            rest, groups=groups[rest]
        )
    )
    train, val = rest[tr_rel], rest[va_rel]

    split = Split(
        name="r2_group_split",
        regime="R2b entity-disjoint (chronology not preserved)",
        train_idx=np.sort(train),
        val_idx=np.sort(val),
        test_idx=np.sort(test),
        meta={"seed": seed, "n_groups": int(n_groups)},
    )
    assert_no_group_overlap(groups[split.train_idx], groups[split.test_idx], name="entity")
    assert_no_group_overlap(groups[split.train_idx], groups[split.val_idx], name="entity (val)")
    return split


# ------------------------------------------------------------------ R3


def r3_leave_one_replica_out(
    replica: np.ndarray, *, val_frac: float = 0.2, seed: int = 0
) -> list[Split]:
    """One fold per held-out replica testbed.

    Named 'replica', not 'organisation': the AIT testbeds share one environment and
    attack repertoire, varying only parameters and execution order.
    """
    replica = np.asarray(replica).ravel()
    names = np.unique(replica)
    if len(names) < 2:
        raise SplitError(f"leave-one-replica-out needs >= 2 replicas, got {len(names)}")

    rng = np.random.default_rng(seed)
    folds: list[Split] = []
    for held in names:
        test = np.where(replica == held)[0]
        rest = np.where(replica != held)[0]
        # Validation comes from whole held-in replicas where possible, so the
        # validation set is not a random slice of the training distribution.
        rest_names = [n for n in names if n != held]
        n_val_rep = max(1, int(round(len(rest_names) * val_frac)))
        val_names = set(rng.choice(rest_names, size=n_val_rep, replace=False).tolist())
        val = np.array([i for i in rest if replica[i] in val_names], dtype=int)
        train = np.array([i for i in rest if replica[i] not in val_names], dtype=int)
        folds.append(
            Split(
                name=f"r3_loro_{held}",
                regime="R3 leave-one-replica-out (parameter/order-randomised replicas of one scenario)",
                train_idx=np.sort(train),
                val_idx=np.sort(val),
                test_idx=np.sort(test),
                meta={
                    "held_out_replica": str(held),
                    "val_replicas": sorted(map(str, val_names)),
                    "n_replicas": int(len(names)),
                },
            )
        )
    return folds


# ------------------------------------------------------------------ R4


def r4_cross_corpus(
    corpus: np.ndarray, *, train_corpus: str, test_corpus: str, val_frac: float = 0.2, seed: int = 0
) -> Split:
    """Train on one corpus, test on another. The only true cross-corpus evidence."""
    corpus = np.asarray(corpus).ravel().astype(str)
    tr_all = np.where(corpus == train_corpus)[0]
    test = np.where(corpus == test_corpus)[0]
    if len(tr_all) == 0 or len(test) == 0:
        raise SplitError(
            f"cross-corpus split empty: train='{train_corpus}' n={len(tr_all)}, "
            f"test='{test_corpus}' n={len(test)}"
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(tr_all)
    n_val = int(round(len(perm) * val_frac))
    return Split(
        name=f"r4_{train_corpus}_to_{test_corpus}",
        regime="R4 cross-corpus",
        train_idx=np.sort(perm[n_val:]),
        val_idx=np.sort(perm[:n_val]),
        test_idx=np.sort(test),
        meta={"train_corpus": train_corpus, "test_corpus": test_corpus, "seed": seed},
    )


__all__ = [
    "LeakageError",
    "Split",
    "SplitError",
    "assert_no_group_overlap",
    "assert_partitions_have_positives",
    "r0_random",
    "r1_chronological",
    "r2_grouped",
    "r2_group_split",
    "r3_leave_one_replica_out",
    "r4_cross_corpus",
]
