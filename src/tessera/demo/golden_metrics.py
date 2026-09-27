"""Golden metric values for the browser Training Lab's metrics.js.

The Training Lab (web/js/lab/metrics.js) re-implements average precision, ROC AUC,
the PR / ROC curves, threshold metrics, ECE and seed summaries in plain JavaScript,
because the demo is a static site with no server. A re-implementation is only
trustworthy if it is checked against the reference, so this module computes every
value with scikit-learn and tessera.eval themselves on a set of deliberately
awkward inputs (heavy ties like duplicated rows produce, all-equal scores, a single
positive, perfect and inverted rankings, scores exactly at the threshold, tiny n,
and one n=3000 case) and writes them to web/data/metrics_golden.json.
web/tests/metrics.test.mjs then asserts metrics.js reproduces each value to 1e-12.

All inputs are synthetic (seeded numpy draws); no AIT data is involved.

Run from the repo root:  uv run python -m tessera.demo.golden_metrics
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from tessera.eval.metrics import class_conditional_ece
from tessera.eval.stats import summarise_seeds

OUT_PATH = Path(__file__).resolve().parents[3] / "web" / "data" / "metrics_golden.json"
THRESHOLDS = (0.5, 0.3)
ECE_BINS = 15
HIST_BINS = 20


def _floats(a) -> list:
    """JSON-safe float list; +inf (roc_curve's first threshold) becomes "inf"."""
    out = []
    for v in np.asarray(a, dtype=np.float64).ravel():
        if math.isinf(v):
            out.append("inf" if v > 0 else "-inf")
        else:
            out.append(float(v))
    return out


def _binary_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {
        "threshold": thr,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def _case(name: str, description: str, y, s, *, float32: bool = False) -> dict:
    y = np.asarray(y, dtype=np.int64).ravel()
    s = np.asarray(s, dtype=np.float32 if float32 else np.float64).ravel()
    assert len(np.unique(y)) == 2, f"{name}: golden cases must contain both classes"
    s64 = s.astype(np.float64)  # exact widening; what the JSON carries
    precision, recall, pr_thr = precision_recall_curve(y, s64)
    fpr, tpr, roc_thr = roc_curve(y, s64)
    clipped = np.clip(s64, 0.0, 1.0)
    cal = class_conditional_ece(y, clipped, n_bins=ECE_BINS)
    edges = np.linspace(0.0, 1.0, HIST_BINS + 1)
    return {
        "name": name,
        "description": description,
        "float32": float32,
        "y": [int(v) for v in y],
        "s": [float(v) for v in s64],
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "ap": float(average_precision_score(y, s64)),
        "roc_auc": float(roc_auc_score(y, s64)),
        "pr": {
            "precision": _floats(precision),
            "recall": _floats(recall),
            "thresholds": _floats(pr_thr),
        },
        "roc": {"fpr": _floats(fpr), "tpr": _floats(tpr), "thresholds": _floats(roc_thr)},
        "at_threshold": [_binary_at(y, s64, t) for t in THRESHOLDS],
        "ece": {
            "n_bins": ECE_BINS,
            "aggregate": cal["ece_aggregate"],
            "positive": cal["ece_positive_class"],
            "negative": cal["ece_negative_class"],
        },
        "histogram": {
            "n_bins": HIST_BINS,
            "edges": _floats(edges),
            "benign": [int(v) for v in np.histogram(clipped[y == 0], bins=edges)[0]],
            "attack": [int(v) for v in np.histogram(clipped[y == 1], bins=edges)[0]],
        },
    }


def build_cases() -> list[dict]:
    rng = np.random.default_rng(20260923)
    cases = []

    y = (rng.random(200) < 0.15).astype(int)
    y[:2] = [0, 1]
    cases.append(
        _case("random_uninformative", "uniform scores unrelated to the label", y, rng.random(200))
    )

    y = (rng.random(500) < 0.2).astype(int)
    logit = rng.normal(size=500) + 1.8 * y - 1.0
    cases.append(
        _case(
            "random_informative",
            "noisy but informative logistic scores",
            y,
            1 / (1 + np.exp(-logit)),
        )
    )

    y = (rng.random(400) < 0.25).astype(int)
    s = np.round(np.clip(rng.normal(0.35 + 0.3 * y, 0.2), 0, 1), 1)
    cases.append(
        _case(
            "heavy_ties",
            "scores rounded to one decimal: large tie groups, like duplicated rows",
            y,
            s,
        )
    )

    y = np.array([0, 1] * 10 + [0] * 30)
    cases.append(
        _case(
            "all_equal",
            "every score identical (0.5, exactly the threshold)",
            y,
            np.full(len(y), 0.5),
        )
    )

    y = np.zeros(100, dtype=int)
    y[37] = 1
    cases.append(
        _case("single_positive", "one positive among 100 (a low-support fold)", y, rng.random(100))
    )

    y = np.array([0] * 30 + [1] * 10)
    s = np.concatenate([rng.uniform(0.0, 0.45, 30), rng.uniform(0.55, 1.0, 10)])
    cases.append(_case("perfect_separation", "every attack scores above every benign window", y, s))

    s_inv = np.concatenate([rng.uniform(0.55, 1.0, 30), rng.uniform(0.0, 0.45, 10)])
    cases.append(
        _case("perfectly_inverted", "every attack scores below every benign window", y, s_inv)
    )

    y = (rng.random(120) < 0.4).astype(int)
    y[:2] = [0, 1]
    s = rng.choice([0.3, 0.5, 0.2, 0.7], size=120)
    s[:10] = 0.5
    cases.append(
        _case("at_threshold", "many scores exactly 0.5 and 0.3 (s >= threshold is positive)", y, s)
    )

    cases.append(_case("tiny_n2", "two windows, one of each class", [0, 1], [0.2, 0.9]))
    cases.append(
        _case("tiny_n3_tie", "three windows with a cross-class tie", [1, 0, 1], [0.4, 0.4, 0.1])
    )

    y = (rng.random(1000) < 0.02).astype(int)
    y[:2] = [0, 1]
    logit = rng.normal(size=1000) + 2.5 * y - 3.0
    s = 1 / (1 + np.exp(-logit))
    s[rng.random(1000) < 0.05] = 0.0
    s[:5] = [0.0, 1.0, 1.0, 0.0, 1.0]
    cases.append(_case("low_prevalence_edges", "2% prevalence with scores exactly 0 and 1", y, s))

    y = (rng.random(3000) < 0.08).astype(int)
    logit = rng.normal(size=3000) * 1.3 + 2.2 * y - 2.0
    s = 1 / (1 + np.exp(-logit))
    dup = rng.random(3000) < 0.2  # duplicated rows -> repeated scores
    s[1:][dup[1:]] = s[:-1][dup[1:]]
    cases.append(
        _case(
            "large_float32", "n=3000, float32 scores with 20% repeated values", y, s, float32=True
        )
    )
    return cases


def build_summaries() -> list[dict]:
    rng = np.random.default_rng(7)
    inputs = {
        "empty": [],
        "single": [0.4375],
        "two": [0.25, 0.75],
        "constant": [0.1, 0.1, 0.1],
        "folds_7": [float(v) for v in rng.uniform(0.2, 0.9, 7)],
        "seeds_5": [float(v) for v in rng.normal(0.6, 0.05, 5)],
    }
    return [{"name": k, "values": v, "expected": summarise_seeds(v)} for k, v in inputs.items()]


def main() -> None:
    payload = {
        "schema": "tessera-metrics-golden/v1",
        "generated_by": "uv run python -m tessera.demo.golden_metrics",
        "reference": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "ece": "tessera.eval.metrics.class_conditional_ece on clip(s, 0, 1)",
            "summary": "tessera.eval.stats.summarise_seeds",
            "histogram": "numpy.histogram(clip(s, 0, 1), bins=linspace(0, 1, 21)) per class",
            "binary": "sklearn precision/recall/f1 (zero_division=0), matthews_corrcoef, accuracy, balanced_accuracy on s >= threshold",
        },
        "note": "synthetic seeded inputs only; no AIT data",
        "cases": build_cases(),
        "summaries": build_summaries(),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    print(
        f"wrote {OUT_PATH} ({len(payload['cases'])} cases, {len(payload['summaries'])} summaries)"
    )


if __name__ == "__main__":
    main()
