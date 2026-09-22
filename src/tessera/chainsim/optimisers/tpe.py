"""Optuna TPE: a real, widely-used Bayesian optimiser as a competent comparator."""

from __future__ import annotations

from tessera.chainsim.optimisers.exhaustive import OptResult


def search(objective_fn, lo: int, hi: int, *, budget: int = 60, seed: int = 0) -> OptResult:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda t: objective_fn(t.suggest_int("segment_length", lo, hi, log=True)),
        n_trials=budget,
        show_progress_bar=False,
    )
    return OptResult(
        "Optuna TPE", int(study.best_params["segment_length"]), float(study.best_value), budget
    )
