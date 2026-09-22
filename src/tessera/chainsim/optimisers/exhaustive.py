"""Exhaustive search: the GROUND-TRUTH optimum.

The search space is one bounded integer, so the true optimum is computable. That
turns every metaheuristic comparison from "which heuristic beat which" into a
measured OPTIMALITY GAP against the actual best, which is the only comparison that
settles anything. The base paper reports neither.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptResult:
    name: str
    best_x: int
    best_value: float
    n_evaluations: int
    history: list = None
    optimality_gap: float | None = None
    optimality_gap_pct: float | None = None

    def as_dict(self) -> dict:
        return {
            "optimiser": self.name,
            "best_segment_length": self.best_x,
            "best_objective": self.best_value,
            "n_evaluations": self.n_evaluations,
            "optimality_gap": self.optimality_gap,
            "optimality_gap_pct": self.optimality_gap_pct,
        }


def search(objective_fn, lo: int, hi: int, *, step: int = 1) -> OptResult:
    """Evaluate every candidate in ``[lo, hi]``."""
    best_x, best_v, n = lo, float("inf"), 0
    for x in range(lo, hi + 1, step):
        v = objective_fn(x)
        n += 1
        if v < best_v:
            best_x, best_v = x, v
    return OptResult(
        "exhaustive (ground truth)", best_x, best_v, n, optimality_gap=0.0, optimality_gap_pct=0.0
    )
