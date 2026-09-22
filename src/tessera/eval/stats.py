"""Significance testing and effect sizes. No comparison is reported without both.

A difference in mean average precision across seeds means nothing on its own: with
5 seeds, a 2-point gap can easily be noise. Every headline comparison therefore
carries a paired test, a multiple-comparison correction over the family, and an
effect size - because a significant p-value on a trivial difference is still a
trivial difference.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# Nemenyi two-tailed critical values at alpha=0.05, indexed by number of models k.
# Source: Demsar (2006), "Statistical Comparisons of Classifiers over Multiple
# Data Sets", JMLR 7:1-30, Table 5a.
_NEMENYI_Q05 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
}


def cliffs_delta(a, b) -> dict:
    """Cliff's delta: P(a > b) - P(a < b). Non-parametric, no distributional claim.

    Magnitude thresholds follow Romano et al. (2006): |d| < 0.147 negligible,
    < 0.33 small, < 0.474 medium, otherwise large.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return {"delta": None, "magnitude": "undefined"}
    diff = np.sign(a[:, None] - b[None, :])
    d = float(diff.sum() / (a.size * b.size))
    ad = abs(d)
    magnitude = (
        "negligible"
        if ad < 0.147
        else "small"
        if ad < 0.33
        else "medium"
        if ad < 0.474
        else "large"
    )
    return {"delta": d, "magnitude": magnitude}


def paired_wilcoxon(a, b, *, label_a: str = "a", label_b: str = "b") -> dict:
    """Wilcoxon signed-rank on paired measurements, plus the effect size."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"paired test needs equal lengths, got {a.size} and {b.size}")

    out = {
        "test": "wilcoxon signed-rank (paired, two-sided)",
        "n_pairs": int(a.size),
        f"mean_{label_a}": float(a.mean()),
        f"mean_{label_b}": float(b.mean()),
        f"std_{label_a}": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        f"std_{label_b}": float(b.std(ddof=1)) if b.size > 1 else 0.0,
        "mean_difference": float((a - b).mean()),
        "effect_size": cliffs_delta(a, b),
    }
    if a.size < 6:
        out["p_value"] = None
        out["note"] = (
            f"n={a.size} pairs is too few for a meaningful Wilcoxon test "
            "(minimum detectable two-sided p at n=5 is 0.0625); report the effect "
            "size and the paired differences instead of a p-value"
        )
        return out
    if np.allclose(a, b):
        out["p_value"] = 1.0
        out["note"] = "identical measurements"
        return out
    try:
        stat, p = stats.wilcoxon(a, b)
        out["statistic"] = float(stat)
        out["p_value"] = float(p)
    except Exception as exc:  # pragma: no cover
        out["p_value"] = None
        out["note"] = f"test failed: {exc}"
    return out


def holm_correction(p_values: dict[str, float], *, alpha: float = 0.05) -> dict:
    """Holm-Bonferroni step-down correction over a family of comparisons.

    Running one test per model against a baseline and reporting raw p-values
    inflates the family-wise error rate; the correction is applied over the whole
    family actually examined, not a subset chosen afterwards.
    """
    items = [(k, v) for k, v in p_values.items() if v is not None]
    if not items:
        return {"alpha": alpha, "n_comparisons": 0, "results": {}}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    results, running_max, still_rejecting = {}, 0.0, True
    for i, (key, p) in enumerate(items):
        adj = min(1.0, max(running_max, p * (m - i)))
        running_max = adj
        if adj > alpha:
            still_rejecting = False
        results[key] = {
            "p_raw": p,
            "p_holm_adjusted": adj,
            "reject_at_alpha": bool(still_rejecting and adj <= alpha),
        }
    for key, v in p_values.items():
        if v is None:
            results[key] = {"p_raw": None, "p_holm_adjusted": None, "reject_at_alpha": None}
    return {"alpha": alpha, "n_comparisons": m, "results": results}


def friedman_nemenyi(scores: dict[str, list[float]], *, alpha: float = 0.05) -> dict:
    """Friedman test across datasets/folds, then Nemenyi critical difference.

    ``scores`` maps model name to one score per fold, in a consistent fold order.
    Produces the mean ranks and critical difference that a CD diagram plots.
    """
    names = list(scores)
    k = len(names)
    if k < 2:
        return {"note": f"need >= 2 models, got {k}"}
    mat = np.asarray([scores[n] for n in names], dtype=np.float64)
    if len({len(v) for v in scores.values()}) != 1:
        raise ValueError("every model needs the same number of folds")
    n_folds = mat.shape[1]

    # Rank per fold, rank 1 = best (highest score).
    ranks = np.apply_along_axis(lambda col: stats.rankdata(-col), 0, mat)
    mean_ranks = ranks.mean(axis=1)

    out = {
        "n_models": k,
        "n_folds": n_folds,
        "mean_ranks": {n: float(r) for n, r in zip(names, mean_ranks, strict=True)},
        "best_by_mean_rank": names[int(np.argmin(mean_ranks))],
        "alpha": alpha,
    }

    if n_folds < 2:
        out["note"] = "need >= 2 folds for a Friedman test"
        return out
    try:
        stat, p = stats.friedmanchisquare(*[mat[i] for i in range(k)])
        out["friedman_statistic"] = float(stat)
        out["friedman_p_value"] = float(p)
        out["friedman_rejects_all_equal"] = bool(p <= alpha)
    except Exception as exc:  # pragma: no cover
        out["friedman_p_value"] = None
        out["note"] = f"Friedman failed: {exc}"

    q = _NEMENYI_Q05.get(k)
    if q is None:
        out["critical_difference"] = None
        out["cd_note"] = f"no tabulated q_0.05 for k={k}"
    else:
        cd = float(q * np.sqrt(k * (k + 1) / (6.0 * n_folds)))
        out["critical_difference"] = cd
        out["cd_note"] = (
            "models whose mean ranks differ by less than the critical difference are "
            "not distinguishable at this alpha"
        )
        out["indistinguishable_pairs"] = [
            [names[i], names[j]]
            for i in range(k)
            for j in range(i + 1, k)
            if abs(mean_ranks[i] - mean_ranks[j]) < cd
        ]
    return out


def summarise_seeds(values: list[float]) -> dict:
    """mean +/- std with the seed count, the only permitted form for a table cell."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        return {"n_seeds": 0, "mean": None, "std": None}
    return {
        "n_seeds": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
    }
