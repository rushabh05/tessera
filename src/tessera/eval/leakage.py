"""The five leakage controls. Every run emits all of them.

The base paper reports 99.4% accuracy on a corpus formed by pooling AWS/Azure/GCP
logs with NSL-KDD, UNSW-NB15, KDD99, CERT and NAB under a random split. Two
mechanisms can produce a number like that without any detection capability:

1. **Duplicate records across the split.** NSL-KDD *is* deduplicated KDD99, so
   pooling both puts identical rows on either side of a random split.
2. **Source provenance.** Seven datasets with disjoint schemas, zero-filled into a
   union table, leave a missingness fingerprint that identifies the source dataset
   - and therefore the label distribution - without reading a single feature.

These controls measure both, plus a group-overlap assertion, a shortcut floor and a
permutation check. They are diagnostics run on *our* pipeline, and the same code
quantifies the base paper's pooled corpus as a negative control.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import cross_val_predict

from tessera.models.baselines.gbdt import gbdt_backend, make_gbdt


class LeakageError(AssertionError):
    """Raised when a leakage guard fails. Intended to fail the build."""


# ---------------------------------------------------------------- control 1


def _row_hashes(X: np.ndarray, *, decimals: int = 6) -> np.ndarray:
    """Stable per-row hashes. Rounded so float noise does not hide true duplicates."""
    Xr = np.round(np.asarray(X, dtype=np.float64), decimals)
    return np.array(
        [hashlib.sha256(row.tobytes()).hexdigest()[:16] for row in np.ascontiguousarray(Xr)]
    )


@dataclass
class DuplicateReport:
    n_rows: int
    n_unique: int
    n_exact_duplicate_rows: int
    duplicate_fraction: float
    cross_split_twins: int | None = None
    cross_split_twin_fraction: float | None = None
    per_class: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_unique": self.n_unique,
            "n_exact_duplicate_rows": self.n_exact_duplicate_rows,
            "duplicate_fraction": round(self.duplicate_fraction, 6),
            "cross_split_twins": self.cross_split_twins,
            "cross_split_twin_fraction": (
                None
                if self.cross_split_twin_fraction is None
                else round(self.cross_split_twin_fraction, 6)
            ),
            "per_class": self.per_class,
        }


def exact_duplicates(
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
) -> DuplicateReport:
    """Count exact duplicate rows, and how many test rows have a train-side twin.

    ``cross_split_twins`` is the number that matters: a test row byte-identical to a
    training row is memorised, not predicted.
    """
    h = _row_hashes(X)
    n = len(h)
    uniq, counts = np.unique(h, return_counts=True)
    n_dup_rows = int((counts[counts > 1]).sum() - (counts > 1).sum())

    per_class: dict = {}
    if y is not None:
        for cls in np.unique(y):
            m = y == cls
            hc, cc = np.unique(h[m], return_counts=True)
            per_class[str(cls)] = {
                "n_rows": int(m.sum()),
                "n_unique": int(len(hc)),
                "n_exact_duplicate_rows": int((cc[cc > 1]).sum() - (cc > 1).sum()),
            }

    twins = twin_frac = None
    if train_idx is not None and test_idx is not None:
        train_set = set(h[train_idx].tolist())
        test_h = h[test_idx]
        twins = int(sum(1 for x in test_h if x in train_set))
        twin_frac = twins / max(len(test_h), 1)

    return DuplicateReport(
        n_rows=n,
        n_unique=int(len(uniq)),
        n_exact_duplicate_rows=n_dup_rows,
        duplicate_fraction=n_dup_rows / max(n, 1),
        cross_split_twins=twins,
        cross_split_twin_fraction=twin_frac,
        per_class=per_class,
    )


# ---------------------------------------------------------------- control 2


def near_duplicates(
    X: np.ndarray,
    *,
    n_bits: int = 32,
    seed: int = 0,
    train_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
) -> dict:
    """Random-projection LSH (SimHash) count of near-duplicate rows.

    Exact hashing misses rows that differ only by rounding or a rescaled field.
    Signing ``n_bits`` random projections of the standardised rows buckets
    near-identical rows together without an extra dependency.
    """
    X = np.asarray(X, dtype=np.float64)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    Z = (X - mu) / np.where(sd > 0, sd, 1.0)

    rng = np.random.default_rng(seed)
    P = rng.standard_normal((Z.shape[1], n_bits))
    bits = (Z @ P) > 0
    weights = (1 << np.arange(n_bits)).astype(np.uint64)
    sig = (bits.astype(np.uint64) * weights).sum(axis=1)

    uniq, counts = np.unique(sig, return_counts=True)
    n_near = int((counts[counts > 1]).sum() - (counts > 1).sum())

    out = {
        "method": f"simhash-{n_bits}bit-random-projection",
        "seed": seed,
        "n_rows": int(len(sig)),
        "n_buckets": int(len(uniq)),
        "n_near_duplicate_rows": n_near,
        "near_duplicate_fraction": round(n_near / max(len(sig), 1), 6),
    }
    if train_idx is not None and test_idx is not None:
        train_sig = set(sig[train_idx].tolist())
        test_sig = sig[test_idx]
        n_twin = int(sum(1 for s in test_sig.tolist() if s in train_sig))
        out["cross_split_near_twins"] = n_twin
        out["cross_split_near_twin_fraction"] = round(n_twin / max(len(test_sig), 1), 6)
    return out


# ---------------------------------------------------------------- control 3


def assert_no_group_overlap(
    train_groups: np.ndarray, test_groups: np.ndarray, *, name: str = "group"
) -> dict:
    """Assert train and test share no entity. Raises :class:`LeakageError`.

    Wired into ``tests/test_no_group_overlap.py`` so a leaked split fails the build
    rather than producing a quietly inflated number.
    """
    tr = set(np.asarray(train_groups).ravel().tolist())
    te = set(np.asarray(test_groups).ravel().tolist())
    shared = tr & te
    if shared:
        sample = sorted(map(str, shared))[:10]
        raise LeakageError(
            f"{name} overlap between train and test: {len(shared)} shared value(s); "
            f"examples: {sample}"
        )
    return {
        "group": name,
        "n_train_groups": len(tr),
        "n_test_groups": len(te),
        "n_shared": 0,
        "asserted_disjoint": True,
    }


# ---------------------------------------------------------------- control 4


def mask_only_score(
    availability_mask: np.ndarray, y: np.ndarray, *, seed: int = 0, cv: int = 5
) -> dict:
    """The shortcut floor: how well the availability mask ALONE predicts the label.

    In multimodal telemetry, modality absence is structural - a host without a
    monitoring agent never emits metrics - so the mask correlates with host role and
    therefore with the label. Any model score at or below this floor has demonstrated
    nothing beyond exploiting which sources were switched on.
    """
    M = np.asarray(availability_mask, dtype=np.float64).reshape(len(y), -1)
    y = np.asarray(y).ravel()
    out: dict = {
        "backend": gbdt_backend(),
        "n_features": int(M.shape[1]),
        "prevalence": float(np.mean(y)),
    }
    if len(np.unique(y)) < 2:
        out["note"] = "single-class target; score undefined"
        return out

    proba = cross_val_predict(
        make_gbdt(seed=seed, n_estimators=100), M, y, cv=cv, method="predict_proba"
    )[:, 1]
    out["average_precision"] = float(average_precision_score(y, proba))
    out["roc_auc"] = float(roc_auc_score(y, proba))
    return out


# ---------------------------------------------------------------- control 5


def provenance_auc(X: np.ndarray, source: np.ndarray, *, seed: int = 0, cv: int = 3) -> dict:
    """How identifiable is a row's SOURCE DATASET from its features?

    Run on the pooled reconstruction of the base paper's corpus, this quantifies how
    much of a 99.x% headline is obtainable from schema signature alone. A high value
    means the pooled table is separable by provenance, so a classifier can route on
    "which dataset is this" instead of "is this an attack".
    """
    X = np.asarray(X, dtype=np.float64)
    source = np.asarray(source).ravel()
    classes = np.unique(source)
    out: dict = {
        "backend": gbdt_backend(),
        "n_sources": int(len(classes)),
        "sources": [str(c) for c in classes],
    }
    if len(classes) < 2:
        out["note"] = "single source; provenance AUC undefined"
        return out

    y = np.searchsorted(classes, source)
    proba = cross_val_predict(
        make_gbdt(seed=seed, n_estimators=100), X, y, cv=cv, method="predict_proba"
    )
    out["macro_ovr_roc_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    out["chance_level"] = 0.5
    return out


# ---------------------------------------------------------------- control 6


def permutation_check(
    X: np.ndarray, y: np.ndarray, *, seed: int = 0, cv: int = 3, n_repeats: int = 3
) -> dict:
    """Shuffle the labels; the pipeline must score at chance.

    A pipeline that scores above chance on permuted labels has a defect - leakage
    through preprocessing fitted on all data, or an index misalignment - and every
    real number it produces is suspect.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).ravel()
    prevalence = float(np.mean(y))
    rng = np.random.default_rng(seed)

    aps = []
    for _ in range(n_repeats):
        yp = rng.permutation(y)
        if len(np.unique(yp)) < 2:
            continue
        proba = cross_val_predict(
            make_gbdt(seed=seed, n_estimators=60), X, yp, cv=cv, method="predict_proba"
        )[:, 1]
        aps.append(float(average_precision_score(yp, proba)))

    return {
        "n_repeats": len(aps),
        "permuted_average_precision_mean": float(np.mean(aps)) if aps else None,
        "permuted_average_precision_std": float(np.std(aps)) if aps else None,
        # For average precision, chance equals the positive prevalence.
        "chance_level": prevalence,
        "within_tolerance": (
            None if not aps else bool(abs(float(np.mean(aps)) - prevalence) < 0.10)
        ),
    }
