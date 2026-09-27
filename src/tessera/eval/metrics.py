"""Metrics, ordered so the informative ones lead and accuracy cannot headline.

At 1-8% attack prevalence, accuracy is dominated by the benign class: a detector
that predicts "benign" for everything scores 99%+ and detects nothing, so a
headline accuracy number is actively misleading at low prevalence. This module
therefore puts average precision first,
reports MCC, and makes the deployment consequence explicit through alert volume and
P(attack | alert) at realistic base rates.

Two deliberate omissions:

* **No point adjustment.** The time-series anomaly-detection convention of marking a
  whole ground-truth segment correct if any point in it is flagged inflates scores
  dramatically and is not used here.
* **Aggregate ECE is not reported as the calibration number.** At 95-99% benign
  prevalence an aggregate ECE is essentially a statistic about the benign bin, so
  calibration is reported class-conditionally.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

# A class with fewer positives than this is reported as a COUNT, never as a rate.
# A recall of "1.0" computed from 3 positives is noise presented as a measurement.
MIN_SUPPORT_FOR_RATES = 20


def binary_metrics(y_true: np.ndarray, score: np.ndarray, *, threshold: float = 0.5) -> dict:
    """Primary binary metrics. ``score`` is a probability or any monotone score."""
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(score, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2:
        return {
            "n": int(len(y)),
            "prevalence": float(y.mean()) if len(y) else None,
            "note": "single-class test partition; ranking metrics undefined",
        }

    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    return {
        "n": int(len(y)),
        "prevalence": float(y.mean()),
        # primary
        "average_precision": float(average_precision_score(y, s)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        # secondary, annotated
        "roc_auc": float(roc_auc_score(y, s)),
        "roc_auc_note": "prevalence-insensitive; reported for comparability only",
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
        # accuracy is present for completeness but is never the headline
        "accuracy": (tp + tn) / len(y),
        "accuracy_note": "uninformative at this prevalence; do not headline",
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "threshold": threshold,
    }


def tpr_at_fpr(y_true: np.ndarray, score: np.ndarray, fprs=(1e-2, 1e-3)) -> dict:
    """Recall at operating points a SOC would actually tolerate."""
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(score, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2:
        return {"note": "single-class partition"}
    fpr, tpr, thr = roc_curve(y, s)
    out = {}
    for target in fprs:
        ok = fpr <= target
        if not ok.any():
            out[f"tpr_at_fpr_{target:g}"] = None
            out[f"threshold_at_fpr_{target:g}"] = None
            continue
        i = int(np.argmax(np.where(ok, tpr, -np.inf)))
        out[f"tpr_at_fpr_{target:g}"] = float(tpr[i])
        out[f"threshold_at_fpr_{target:g}"] = float(thr[i])
    return out


def deployment_consequence(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    windows_per_day: float,
    base_rates=(0.01, 0.001, 0.0001),
    target_fpr: float = 1e-2,
) -> dict:
    """Alert volume and P(attack | alert) at deployment base rates.

    The test partition's prevalence is an artifact of dataset construction. In
    production the base rate is far lower, and precision collapses accordingly - the
    base-rate fallacy Axelsson described for intrusion detection. Reporting
    P(attack | alert) at several assumed base rates makes that explicit instead of
    letting a test-set precision imply operational usefulness.
    """
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(score, dtype=np.float64).ravel()
    if len(np.unique(y)) < 2:
        return {"note": "single-class partition"}

    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= target_fpr
    if not ok.any():
        return {"note": f"no operating point with fpr <= {target_fpr:g}"}
    i = int(np.argmax(np.where(ok, tpr, -np.inf)))
    op_tpr, op_fpr = float(tpr[i]), float(fpr[i])

    ppv = {}
    for pi in base_rates:
        num = op_tpr * pi
        den = num + op_fpr * (1.0 - pi)
        ppv[f"p_attack_given_alert_at_base_rate_{pi:g}"] = float(num / den) if den > 0 else None

    return {
        "operating_point": {"tpr": op_tpr, "fpr": op_fpr, "target_fpr": target_fpr},
        "windows_per_day": float(windows_per_day),
        # At the chosen FPR, benign windows dominate, so alert volume is ~ FPR * volume
        "false_alerts_per_day": float(op_fpr * windows_per_day),
        **ppv,
        "interpretation": (
            "P(attack|alert) is what an analyst experiences; test-set precision is not."
        ),
    }


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, *, labels=None, min_support: int = MIN_SUPPORT_FOR_RATES
) -> dict:
    """Per-class precision/recall/F1 WITH support, suppressing rates on thin classes."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    labels = np.unique(np.concatenate([y_true, y_pred])) if labels is None else np.asarray(labels)
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {"min_support_for_rates": min_support, "classes": {}}
    for i, lab in enumerate(labels):
        entry = {"support": int(sup[i])}
        if sup[i] >= min_support:
            entry.update({"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])})
        else:
            entry["rates_suppressed"] = (
                f"support {int(sup[i])} < {min_support}; reported count-only"
            )
        out["classes"][str(lab)] = entry
    return out


def class_conditional_ece(y_true: np.ndarray, proba: np.ndarray, *, n_bins: int = 15) -> dict:
    """Expected calibration error computed WITHIN each true class.

    An aggregate ECE at 95-99% benign prevalence measures the benign bin and hides
    the fact that positive-class confidences may be badly miscalibrated.
    """
    y = np.asarray(y_true).ravel().astype(int)
    p = np.asarray(proba, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def _ece(mask: np.ndarray) -> float | None:
        if mask.sum() == 0:
            return None
        pm, ym = p[mask], y[mask]
        total = 0.0
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            sel = (pm > lo) & (pm <= hi) if lo > 0 else (pm >= lo) & (pm <= hi)
            if not sel.any():
                continue
            total += (sel.sum() / len(pm)) * abs(ym[sel].mean() - pm[sel].mean())
        return float(total)

    return {
        "n_bins": n_bins,
        "ece_positive_class": _ece(y == 1),
        "ece_negative_class": _ece(y == 0),
        "ece_aggregate": _ece(np.ones_like(y, dtype=bool)),
        "ece_aggregate_note": (
            "dominated by the benign bin at this prevalence; the class-conditional "
            "values carry the calibration claim"
        ),
    }


def risk_coverage(y_true: np.ndarray, score: np.ndarray, confidence: np.ndarray) -> dict:
    """Risk-coverage curve and AURC for selective prediction.

    Abstaining on low-confidence windows is how an uncertainty-aware detector earns
    its keep: error on the retained set should fall as coverage falls. AURC
    summarises that. Ordering is by ``confidence``, which for the evidential head is
    ``1 - vacuity`` rather than the class probability.
    """
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(score, dtype=np.float64).ravel()
    c = np.asarray(confidence, dtype=np.float64).ravel()
    pred = (s >= 0.5).astype(int)
    err = (pred != y).astype(np.float64)

    order = np.argsort(-c, kind="stable")
    err_sorted = err[order]
    n = len(err_sorted)
    cum_err = np.cumsum(err_sorted)
    k = np.arange(1, n + 1)
    risk = cum_err / k
    coverage = k / n

    return {
        "aurc": float(np.trapezoid(risk, coverage)) if n > 1 else None,
        "risk_at_full_coverage": float(risk[-1]) if n else None,
        "risk_at_50pct_coverage": float(risk[max(0, n // 2 - 1)]) if n else None,
        "risk_at_20pct_coverage": float(risk[max(0, n // 5 - 1)]) if n else None,
        "curve": {
            "coverage": [round(float(x), 4) for x in coverage[:: max(1, n // 50)]],
            "risk": [round(float(x), 6) for x in risk[:: max(1, n // 50)]],
        },
    }


def evaluate(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    windows_per_day: float = 86400 / 60,
    threshold: float = 0.5,
) -> dict:
    """The standard bundle emitted for every run, in reporting order."""
    y = np.asarray(y_true).ravel().astype(int)
    s = np.asarray(score, dtype=np.float64).ravel()
    out = {"binary": binary_metrics(y, s, threshold=threshold)}
    if len(np.unique(y)) < 2:
        return out
    out["operating_points"] = tpr_at_fpr(y, s)
    out["deployment"] = deployment_consequence(y, s, windows_per_day=windows_per_day)
    out["calibration"] = class_conditional_ece(y, np.clip(s, 0, 1))
    out["per_class"] = per_class_report(y, (s >= threshold).astype(int))
    out["selective"] = risk_coverage(
        y, s, confidence if confidence is not None else np.abs(s - 0.5)
    )
    out["point_adjustment"] = "not used"
    return out
