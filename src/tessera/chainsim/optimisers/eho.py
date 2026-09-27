"""Elephant Herding Optimizer (EHO), a candidate optimiser TESSERA evaluated for
chain segment-length tuning.

    herd init:    NSC = STOCH(LH * N / 2, N / 2)
    fitness:      fh  = (1/NEB) * sum(dr + dw + dh + dv) * em
    threshold:    fth = (1/NH) * sum_i fh(i) * LH
    herd update:  NSC(new) = (NSC(old) + NSC(Matriarch)) / 2

The herd initialisation draws each herd's candidate split point from an interval
scaled by the learning-rate parameter ``LH``: ``[LH*N/2, N/2]``, where ``N`` is the
current chain length. The update rule then moves every herd whose fitness exceeds a
mean-scaled threshold toward the midpoint between itself and the matriarch (the
best-fitness herd) - a pure averaging step, with no mutation operator anywhere in
that update rule.

Design choices made while implementing it, recorded rather than silently chosen:

* ``STOCH`` is implemented as a uniform draw over the stated interval, the weakest
  assumption that reproduces the described behaviour.
* ``LH`` (herd learning rate) has no single canonical value, so it is swept, and the
  value used is reported alongside every result.
* The threshold scales the mean fitness by ``LH``. For ``LH < 1`` the threshold
  falls BELOW the mean, so most herds exceed it and are reconfigured every
  iteration; the search therefore collapses toward the matriarch rather than
  exploring. That behaviour is a property of the update rule itself, not of this
  implementation.
* The update rule moves every above-threshold herd to the midpoint between itself
  and the matriarch, with no mutation, so herd diversity contracts monotonically.
  There is no mechanism by which it can recover.

Judged against exhaustive ground truth and random search on an identical budget.
"""

from __future__ import annotations

import numpy as np

from tessera.chainsim.optimisers.exhaustive import OptResult


def search(
    objective_fn,
    lo: int,
    hi: int,
    *,
    n_herds: int = 10,
    n_iterations: int = 6,
    learning_rate: float = 0.5,
    seed: int = 0,
) -> OptResult:
    rng = np.random.default_rng(seed)
    LH = learning_rate

    # herd init: each herd proposes a split point. N is the current chain length,
    # which here is the top of the search range.
    N = hi
    lo_draw, hi_draw = LH * N / 2.0, N / 2.0
    if lo_draw > hi_draw:
        lo_draw, hi_draw = hi_draw, lo_draw
    herds = np.clip(rng.uniform(lo_draw, hi_draw, n_herds), lo, hi)

    n_eval = 0
    best_x, best_v = int(herds[0]), float("inf")
    history = []

    for it in range(n_iterations):
        fitness = np.array([objective_fn(int(round(x))) for x in herds])
        n_eval += len(herds)

        i_best = int(np.argmin(fitness))
        if fitness[i_best] < best_v:
            best_v, best_x = float(fitness[i_best]), int(round(herds[i_best]))

        matriarch = herds[i_best]  # minimum-fitness herd
        f_th = float(np.mean(fitness) * LH)  # threshold rule
        reconfig = fitness > f_th
        herds = np.where(reconfig, (herds + matriarch) / 2.0, herds)  # herd update
        herds = np.clip(herds, lo, hi)

        history.append(
            {
                "iteration": it,
                "f_threshold": f_th,
                "mean_fitness": float(np.mean(fitness)),
                "n_reconfigured": int(reconfig.sum()),
                "herd_spread": float(herds.max() - herds.min()),
                "matriarch": float(matriarch),
            }
        )

    return OptResult(
        f"EHO (NH={n_herds}, NI={n_iterations}, LH={LH})", best_x, best_v, n_eval, history=history
    )


def reachable_interval(
    chain_length: int, learning_rate: float, lo: int, hi: int
) -> tuple[float, float]:
    """The set of split points EHO can EVER evaluate, for this initialisation/update rule.

    Herd init draws every initial herd from ``[LH*N/2, N/2]``. The herd update
    replaces a herd with ``(herd + matriarch)/2``, a convex combination of two
    points already in the population. A convex combination cannot leave the convex
    hull of the population, and no mutation operator exists in that update rule.
    Therefore the reachable set is exactly the herd-init interval, for any number of
    iterations.
    """
    a, b = learning_rate * chain_length / 2.0, chain_length / 2.0
    a, b = (a, b) if a <= b else (b, a)
    return max(float(lo), a), min(float(hi), b)


def demonstrate_structural_unreachability(
    objective_fn,
    lo: int,
    hi: int,
    true_optimum: int,
    *,
    learning_rates=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
) -> dict:
    """Show that the optimum lies outside EHO's reachable set, and for which LH.

    This distinguishes a tuning problem from a structural one. If the optimum is
    outside the interval, no herd count, iteration budget or seed can find it.
    """
    rows = []
    for lh in learning_rates:
        a, b = reachable_interval(hi, lh, lo, hi)
        reachable = a <= true_optimum <= b
        rows.append(
            {
                "learning_rate": lh,
                "reachable_interval": [a, b],
                "interval_width_fraction_of_range": (b - a) / (hi - lo),
                "true_optimum": true_optimum,
                "optimum_reachable": bool(reachable),
                "best_possible_value_in_interval": min(
                    objective_fn(int(round(x))) for x in (a, (a + b) / 2, b)
                ),
            }
        )
    any_reachable = any(r["optimum_reachable"] for r in rows)
    return {
        "equations": "herd initialisation and herd update",
        "argument": (
            "Herd init confines every initial herd to [LH*N/2, N/2]. The herd "
            "update replaces a herd with the midpoint between it and the Matriarch "
            "- a convex combination of existing population members - and this "
            "update rule defines no mutation. The population therefore never "
            "leaves the convex hull of its initialisation, so the reachable set "
            "equals the herd-init interval for any herd count, iteration budget or "
            "seed."
        ),
        "rows": rows,
        "optimum_reachable_for_any_tested_lh": any_reachable,
        "finding": (
            "The optimality gap is STRUCTURAL, not a tuning failure: for every tested "
            "learning rate the true optimum lies outside the interval this "
            "initialisation rule permits."
            if not any_reachable
            else "Some learning rates do admit the optimum; the gap is then partly a tuning issue."
        ),
    }
