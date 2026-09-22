"""Uniform random search: the baseline every metaheuristic must actually beat.

Random search on a low-dimensional problem is a surprisingly strong baseline, and
omitting it is how metaphor-driven metaheuristics come to look effective. Included
here with an identical evaluation budget so the comparison is fair.
"""

from __future__ import annotations

import numpy as np

from tessera.chainsim.optimisers.exhaustive import OptResult


def search(objective_fn, lo: int, hi: int, *, budget: int = 60, seed: int = 0) -> OptResult:
    rng = np.random.default_rng(seed)
    best_x, best_v = lo, float("inf")
    for _ in range(budget):
        x = int(rng.integers(lo, hi + 1))
        v = objective_fn(x)
        if v < best_v:
            best_x, best_v = x, v
    return OptResult("random search", best_x, best_v, budget)
