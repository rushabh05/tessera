"""Table and figure generation. Every number is read from results/, never typed.

``check_no_hardcoded_numbers.py`` fails the build if a numeral in a generated
document does not trace back to a recorded run, which is the mechanism that makes
the base paper's Section-IV-vs-Conclusion contradiction impossible here.
"""

from __future__ import annotations

import json

from tessera.eval.halt_gate import check_e1
from tessera.paths import TABLES_DIR, ensure_dirs
from tessera.report.collect import (
    available_datasets,
    collect,
    ordered_models,
    ordered_regimes,
    summarise,
)

DASH = "--"


def _cell(entry: dict | None, *, decimals: int = 4) -> str:
    if not entry or entry.get("mean") is None or entry.get("n_seeds", 0) == 0:
        return DASH
    mean, std, n = entry["mean"], entry.get("std") or 0.0, entry["n_seeds"]
    if n == 1:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def inflation_cascade_markdown(
    metric: str = "binary.average_precision", dataset: str | None = None
) -> str:
    """T2: the project's central table. Models by split regime, strictness increasing."""
    summ = summarise(collect(metric, dataset=dataset))
    models, regimes = ordered_models(summ), ordered_regimes(summ)
    if not models:
        return "_no recorded runs_\n"

    lines = [
        f"**Metric:** `{metric}` — mean ± std across seeds. "
        f"Columns increase in split strictness left to right; a value that falls is "
        f"the leakage the looser split was hiding.",
        "",
        "| model | " + " | ".join(regimes) + " | seeds |",
        "|" + "---|" * (len(regimes) + 2),
    ]
    for m in models:
        row = summ["summary"].get(m, {})
        seeds = {e.get("n_seeds", 0) for e in row.values()} or {0}
        lines.append(
            f"| `{m}` | " + " | ".join(_cell(row.get(r)) for r in regimes) + f" | {max(seeds)} |"
        )

    synthetic = any(
        (summ["meta"].get(m, {}).get(r, {}) or {}).get("is_synthetic")
        for m in models
        for r in regimes
    )
    if synthetic:
        lines += [
            "",
            "> **SYNTHETIC DATA.** Validates the harness against known ground truth. Not a detection result.",
        ]

    floors = {
        (summ["meta"].get(m, {}).get(r, {}) or {}).get("mask_only_floor_ap")
        for m in models
        for r in regimes
    } - {None}
    if floors:
        lines += [
            "",
            f"> **Shortcut floor** (availability mask alone): AP = "
            f"{min(floors):.4f}–{max(floors):.4f}. A model at or below this range has "
            f"learned which telemetry sources were switched on, not what an attack looks like.",
        ]
    return "\n".join(lines) + "\n"


def e1_gate_status(dataset: str | None = None, required_drop: float = 0.02) -> dict:
    """Evaluate the E1 gate from recorded runs rather than from a live pipeline."""
    summ = summarise(collect("binary.average_precision", dataset=dataset))
    out = {"required_drop": required_drop, "checks": {}}
    for m, row in summ["summary"].items():
        r0, r1 = row.get("r0_random"), row.get("r1_chrono")
        if not r0 or not r1 or r0.get("mean") is None or r1.get("mean") is None:
            continue
        res = check_e1(r0["mean"], r1["mean"], required_drop=required_drop, raise_on_fail=False)
        out["checks"][m] = res.as_dict()
    # Floor models cannot inflate: they ignore the features a looser split leaks.
    discriminative = {
        m: c for m, c in out["checks"].items() if m not in {"random", "majority", "nullmask"}
    }
    out["verdict"] = (
        "no discriminative model has both R0 and R1 recorded"
        if not discriminative
        else "PASS"
        if all(c["passed"] for c in discriminative.values())
        else "HALT"
    )
    out["evaluated_on"] = sorted(discriminative)
    return out


def write_all(dataset: str | None = None) -> dict:
    """Regenerate every table, ONE SET PER DATASET. Invoked by `just tables`.

    Tables are always per-dataset: a single table mixing corpora would present a
    between-corpus difference as seed variance.
    """
    ensure_dirs()
    targets = [dataset] if dataset else available_datasets()
    written: dict = {}
    verdicts: dict = {}

    # Remove stale tables first. A generated file left over from an earlier schema
    # still contains numbers, and a reader cannot tell it is obsolete - the
    # hardcoded-number check flagged exactly this case.
    if dataset is None:
        for stale in TABLES_DIR.glob("*.md"):
            stale.unlink()

    for ds in targets:
        suffix = f"__{ds}"
        md = inflation_cascade_markdown(dataset=ds)
        p = TABLES_DIR / f"t2_inflation_cascade{suffix}.md"
        p.write_text(f"# T2 - Split-regime inflation cascade (dataset: {ds})\n\n" + md)
        written[f"t2{suffix}"] = str(p)

        gate = e1_gate_status(dataset=ds)
        p = TABLES_DIR / f"e1_gate{suffix}.json"
        p.write_text(json.dumps(gate, indent=2))
        written[f"e1_gate{suffix}"] = str(p)
        verdicts[ds] = gate["verdict"]

        for metric, slug in (
            ("binary.mcc", "t2b_mcc"),
            ("deployment.p_attack_given_alert_at_base_rate_0.0001", "t4_base_rate"),
        ):
            p = TABLES_DIR / f"{slug}{suffix}.md"
            p.write_text(
                f"# {slug} - `{metric}` (dataset: {ds})\n\n"
                + inflation_cascade_markdown(metric, dataset=ds)
            )
            written[f"{slug}{suffix}"] = str(p)

    return {"written": written, "e1_verdicts": verdicts, "datasets": targets}


def main() -> None:
    res = write_all()
    for ds in res["datasets"]:
        print(f"===== dataset: {ds} =====")
        print(inflation_cascade_markdown(dataset=ds))
        print(f"E1 gate verdict: {res['e1_verdicts'][ds]}\n")
    for name in sorted(res["written"]):
        print(f"  wrote {name}")


if __name__ == "__main__":
    main()
