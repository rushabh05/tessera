"""Run every optimiser on the same objective and budget, and report optimality gaps."""

from __future__ import annotations

from tessera.chainsim.cost_model import Workload, calibrate, evaluate_cost
from tessera.chainsim.optimisers import eho, exhaustive, random_search, tpe

LO, HI = 8, 65536


def run(*, n_entries: int = 200_000, budget: int = 60, seed: int = 0, cal=None) -> dict:
    cal = cal or calibrate(n_hash=20000, n_sign=1500)
    wl = Workload(n_entries=n_entries)

    calls = {"n": 0}

    def obj(S: int) -> float:
        calls["n"] += 1
        return evaluate_cost(S, wl, cal).total_delay_s

    # Ground truth first, on a coarse-but-dense grid over a log-spaced range.
    truth = exhaustive.search(obj, LO, HI, step=8)

    results = [truth]
    for _name, fn, kw in (
        ("random", random_search.search, {"budget": budget, "seed": seed}),
        ("tpe", tpe.search, {"budget": budget, "seed": seed}),
        (
            "eho",
            eho.search,
            {"n_herds": 10, "n_iterations": budget // 10, "learning_rate": 0.5, "seed": seed},
        ),
    ):
        r = fn(obj, LO, HI, **kw)
        r.optimality_gap = r.best_value - truth.best_value
        r.optimality_gap_pct = (
            100.0 * r.optimality_gap / truth.best_value if truth.best_value else None
        )
        results.append(r)

    return {
        "calibration": cal.as_dict(),
        "workload": {"n_entries": n_entries, "proof_requests": wl.proof_requests},
        "search_range": [LO, HI],
        "budget_per_metaheuristic": budget,
        "ground_truth": truth.as_dict(),
        "results": [r.as_dict() for r in results],
        "eho_history": next((r.history for r in results if r.history), None),
    }
