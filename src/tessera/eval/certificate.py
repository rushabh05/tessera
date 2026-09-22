"""The per-run leakage certificate, printed beneath every results table.

A results table without its certificate is not reportable. Making the certificate a
machine-emitted artifact rather than an appendix experiment means the controls
cannot quietly stop being run once the numbers look good.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from tessera.eval import leakage


@dataclass
class Certificate:
    """Assembled leakage evidence for one run."""

    split_name: str
    regime: str
    sizes: dict
    positives: dict
    exact_duplicates: dict | None = None
    near_duplicates: dict | None = None
    group_disjointness: list = field(default_factory=list)
    mask_only_floor: dict | None = None
    provenance: dict | None = None
    permutation: dict | None = None
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "split": {
                "name": self.split_name,
                "regime": self.regime,
                **self.sizes,
                **self.positives,
            },
            "control_1_exact_duplicates": self.exact_duplicates,
            "control_2_near_duplicates": self.near_duplicates,
            "control_3_group_disjointness": self.group_disjointness,
            "control_4_mask_only_floor": self.mask_only_floor,
            "control_5_provenance": self.provenance,
            "control_6_label_permutation": self.permutation,
            "warnings": self.warnings,
        }

    def render(self, *, width: int = 92) -> str:
        """One-paragraph plain-text form, printed under a results table."""
        lines = ["-" * width, f"LEAKAGE CERTIFICATE  |  split={self.split_name}  ({self.regime})"]
        s, p = self.sizes, self.positives
        lines.append(
            f"  sizes      train={s.get('n_train')} val={s.get('n_val')} test={s.get('n_test')}"
            f"   positives train={p.get('pos_train')} val={p.get('pos_val')} test={p.get('pos_test')}"
        )
        if self.exact_duplicates:
            e = self.exact_duplicates
            twins = e.get("cross_split_twins")
            lines.append(
                f"  dup(exact) {e['n_exact_duplicate_rows']} dup rows "
                f"({e['duplicate_fraction']:.4%}); test rows with a train-side twin: {twins}"
            )
        if self.near_duplicates:
            nd = self.near_duplicates
            lines.append(
                f"  dup(near)  {nd['n_near_duplicate_rows']} rows "
                f"({nd['near_duplicate_fraction']:.4%}) via {nd['method']}"
                + (
                    f"; cross-split near-twins: {nd['cross_split_near_twins']}"
                    if "cross_split_near_twins" in nd
                    else ""
                )
            )
        for g in self.group_disjointness:
            lines.append(
                f"  disjoint   {g['group']}: train={g['n_train_groups']} test={g['n_test_groups']} "
                f"shared={g['n_shared']} (asserted)"
            )
        if self.mask_only_floor:
            m = self.mask_only_floor
            ap = m.get("average_precision")
            lines.append(
                f"  mask-only  AP={ap:.4f} (shortcut floor; a model at or below this has "
                f"shown nothing)"
                if ap is not None
                else f"  mask-only  {m.get('note')}"
            )
        if self.provenance:
            pv = self.provenance
            auc = pv.get("macro_ovr_roc_auc")
            lines.append(
                f"  provenance source-ID AUC={auc:.4f} over {pv['n_sources']} sources "
                f"(chance {pv.get('chance_level')})"
                if auc is not None
                else f"  provenance {pv.get('note')}"
            )
        if self.permutation:
            pm = self.permutation
            got, chance = pm.get("permuted_average_precision_mean"), pm.get("chance_level")
            if got is not None:
                flag = "OK" if pm.get("within_tolerance") else "SUSPECT"
                lines.append(f"  permuted   AP={got:.4f} vs chance {chance:.4f}  [{flag}]")
        for w in self.warnings:
            lines.append(f"  WARNING    {w}")
        lines.append("-" * width)
        return "\n".join(lines)


def build_certificate(
    *,
    split,
    y: np.ndarray,
    X: np.ndarray | None = None,
    groups: dict[str, np.ndarray] | None = None,
    availability_mask: np.ndarray | None = None,
    source: np.ndarray | None = None,
    seed: int = 0,
    run_permutation: bool = True,
) -> Certificate:
    """Run every applicable control for one split and assemble the certificate.

    Controls needing data that is not present are skipped and recorded as skipped,
    never silently omitted.
    """
    y = np.asarray(y).ravel()
    cert = Certificate(
        split_name=split.name,
        regime=split.regime,
        sizes=split.sizes(),
        positives=split.positives(y),
    )

    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        cert.exact_duplicates = leakage.exact_duplicates(
            X, y, train_idx=split.train_idx, test_idx=split.test_idx
        ).as_dict()
        cert.near_duplicates = leakage.near_duplicates(
            X, seed=seed, train_idx=split.train_idx, test_idx=split.test_idx
        )
        twins = cert.exact_duplicates.get("cross_split_twins") or 0
        if twins:
            cert.warnings.append(
                f"{twins} test rows are byte-identical to a training row; those "
                "predictions are memorised, not generalised"
            )
    else:
        cert.warnings.append("controls 1-2 skipped: no feature matrix supplied")

    if groups:
        for name, g in groups.items():
            g = np.asarray(g).ravel()
            try:
                cert.group_disjointness.append(
                    leakage.assert_no_group_overlap(
                        g[split.train_idx], g[split.test_idx], name=name
                    )
                )
            except leakage.LeakageError as exc:
                cert.group_disjointness.append(
                    {"group": name, "n_shared": -1, "error": str(exc), "asserted_disjoint": False}
                )
                cert.warnings.append(f"group overlap on {name}: {exc}")

    if availability_mask is not None:
        cert.mask_only_floor = leakage.mask_only_score(availability_mask, y, seed=seed)
    else:
        cert.warnings.append("control 4 skipped: no availability mask supplied")

    if source is not None and X is not None:
        cert.provenance = leakage.provenance_auc(X, source, seed=seed)

    if run_permutation and X is not None:
        cert.permutation = leakage.permutation_check(X, y, seed=seed)
        if cert.permutation.get("within_tolerance") is False:
            cert.warnings.append(
                "permuted labels scored away from chance; the pipeline has a defect "
                "and every real number it produces is suspect"
            )

    return cert
