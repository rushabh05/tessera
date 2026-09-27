"""Chain-simulator report: optimiser comparison, optimality gaps, and findings from
evaluating candidate objectives and optimisers for sidechain segment-length tuning."""

from __future__ import annotations

import json

from tessera.chainsim.benchmark import HI, LO, run
from tessera.chainsim.cost_model import Workload, calibrate, evaluate_cost
from tessera.chainsim.optimisers import eho
from tessera.chainsim.segment_objective import demonstrate_objective_invariance
from tessera.paths import TABLES_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    cal = calibrate(n_hash=20000, n_sign=1500)
    res = run(budget=60, seed=0, cal=cal)

    print("=" * 78)
    print("OPTIMISER COMPARISON vs EXHAUSTIVE GROUND TRUTH")
    print("=" * 78)
    gt = res["ground_truth"]
    print(
        f"ground truth: S={gt['best_segment_length']}  "
        f"delay={gt['best_objective']:.6f}s  ({gt['n_evaluations']} evaluations)\n"
    )
    print(f"{'optimiser':<34}{'best S':>10}{'objective':>13}{'gap':>12}{'gap %':>9}")
    for r in res["results"]:
        pct = "" if r["optimality_gap_pct"] is None else f"{r['optimality_gap_pct']:8.3f}%"
        print(
            f"{r['optimiser']:<34}{r['best_segment_length']:>10}"
            f"{r['best_objective']:>13.6f}{r['optimality_gap'] or 0.0:>12.6f}{pct:>9}"
        )

    wl = Workload(n_entries=res["workload"]["n_entries"])
    obj = lambda S: evaluate_cost(S, wl, cal).total_delay_s  # noqa: E731
    unreach = eho.demonstrate_structural_unreachability(obj, LO, HI, gt["best_segment_length"])

    print("\n" + "=" * 78)
    print("FINDING 1 - EHO's optimality gap is STRUCTURAL, not a tuning failure")
    print("=" * 78)
    print(f"{'LH':>6}  {'reachable interval (eq. 20)':>30}  {'optimum inside?':>16}")
    for row in unreach["rows"]:
        a, b = row["reachable_interval"]
        print(
            f"{row['learning_rate']:>6.1f}  [{a:>12.0f}, {b:>12.0f}]  {str(row['optimum_reachable']):>16}"
        )
    print("\n" + unreach["argument"])
    print("\n-> " + unreach["finding"])

    inv = demonstrate_objective_invariance()
    print("\n" + "=" * 78)
    print("FINDING 2 - a candidate objective we drafted is invariant to its own decision variable")
    print("=" * 78)
    print(f"  {inv['equation']}")
    print(f"  NSC swept over {inv['nsc_values']}")
    print(f"  absolute spread across three orders of magnitude: {inv['absolute_spread']!r}")
    print(f"  d(fh)/d(NSC) == 0: {inv['d_fh_d_nsc_is_zero']}")
    print(f"  units: {inv['units']}")
    print(f"  undefined symbol: {inv['undefined_symbol']}")
    print("\n-> " + inv["finding"])

    payload = {
        "optimiser_comparison": res,
        "finding_1_unreachability": unreach,
        "finding_2_invariance": inv,
    }
    out = TABLES_DIR / "t9_chainsim.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
