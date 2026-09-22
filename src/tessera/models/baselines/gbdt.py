"""One gradient-boosting factory, used by both the headline baseline and the
mask-only leakage floor, so the two are directly comparable.

Prefers LightGBM (the standard for this benchmark family) and falls back to
sklearn's ``HistGradientBoosting`` when LightGBM's OpenMP runtime is unavailable,
so a missing system library degrades the baseline rather than blocking the repo.
"""

from __future__ import annotations

from tessera import LIGHTGBM_AVAILABLE


def make_gbdt(*, seed: int = 0, n_estimators: int = 200, **kwargs):
    """A binary-classification GBDT with a stated, reproducible configuration."""
    if LIGHTGBM_AVAILABLE:
        import lightgbm as lgb

        params = {
            "n_estimators": n_estimators,
            "random_state": seed,
            "deterministic": True,
            "force_row_wise": True,  # silences the threading heuristic warning
            "num_threads": 1,  # determinism over speed; these models are tiny
            "verbose": -1,
        }
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=n_estimators, random_state=seed, **kwargs)


def gbdt_backend() -> str:
    return "lightgbm" if LIGHTGBM_AVAILABLE else "sklearn-histgradientboosting"
