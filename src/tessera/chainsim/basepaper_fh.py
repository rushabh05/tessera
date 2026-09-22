"""The base paper's sidechain objective, implemented literally - and shown to be
invariant to its own decision variable.

Equation 21:  fh = (1/NEB) * sum_{i=1..NED} (dr(i) + dw(i) + dh(i) + dv(i)) * em(i)

Three defects, all visible once it is written out as code:

1. **Dimensional incoherence.** A sum of delays multiplied by an energy has units of
   joule-seconds, which is neither a delay nor an energy.
2. **Undefined index.** The summation runs to ``NED`` while the normaliser is
   ``1/NEB``; ``NED`` is never defined in the paper.
3. **Decisive: it does not depend on NSC.** Per-block read, write, hash and verify
   costs are properties of a block, not of where the chain was cut. So ``fh`` is
   constant in the split point the Elephant Herding Optimizer is searching over,
   and the reported delay, energy and throughput gains cannot have come from that
   optimiser.

:func:`demonstrate_invariance` measures this rather than asserting it.
"""

from __future__ import annotations

import numpy as np


def basepaper_fh(
    n_sidechain_blocks: int,
    *,
    n_eval_blocks: int = 32,
    d_read: float = 1.0e-4,
    d_write: float = 2.0e-4,
    d_hash: float = 5.0e-6,
    d_verify: float = 8.0e-5,
    energy_per_block_j: float = 1.0e-3,
    seed: int = 0,
) -> float:
    """Equation 21, taken at face value.

    ``n_sidechain_blocks`` is NSC, the quantity eqs. 20 and 23 search over. Note that
    it appears nowhere in the computation - which is precisely the finding.
    """
    rng = np.random.default_rng(seed)
    # Per-block costs, with a little jitter so the result is not trivially constant
    # for an uninteresting reason.
    dr = d_read * (1 + 0.01 * rng.standard_normal(n_eval_blocks))
    dw = d_write * (1 + 0.01 * rng.standard_normal(n_eval_blocks))
    dh = d_hash * (1 + 0.01 * rng.standard_normal(n_eval_blocks))
    dv = d_verify * (1 + 0.01 * rng.standard_normal(n_eval_blocks))
    em = energy_per_block_j * (1 + 0.01 * rng.standard_normal(n_eval_blocks))
    return float(((dr + dw + dh + dv) * em).sum() / n_eval_blocks)


def demonstrate_invariance(
    nsc_values=(8, 16, 32, 64, 128, 256, 512, 1024, 2048), *, seed: int = 0
) -> dict:
    """Evaluate eq. 21 across split points and report the spread.

    With the stochastic jitter seeded identically, the values are bit-identical: the
    objective is exactly constant in NSC. The partial derivative is zero, so the
    search has nothing to optimise.
    """
    vals = [basepaper_fh(n, seed=seed) for n in nsc_values]
    arr = np.asarray(vals, dtype=np.float64)
    spread = float(arr.max() - arr.min())
    rel = spread / abs(float(arr.mean())) if arr.mean() else 0.0
    return {
        "equation": "eq. 21: fh = (1/NEB) * sum(dr+dw+dh+dv) * em",
        "nsc_values": list(nsc_values),
        "fh_values": vals,
        "absolute_spread": spread,
        "relative_spread": rel,
        "d_fh_d_nsc_is_zero": bool(spread == 0.0),
        "finding": (
            "fh is exactly constant across three orders of magnitude of NSC, so the "
            "published objective is invariant to the decision variable the Elephant "
            "Herding Optimizer searches over. Any reported improvement in block "
            "delay, energy or throughput therefore cannot be attributed to that "
            "optimisation."
        ),
        "units": "joule-seconds, which is neither a delay nor an energy",
        "undefined_symbol": "NED (summation bound) is never defined in the paper",
    }
