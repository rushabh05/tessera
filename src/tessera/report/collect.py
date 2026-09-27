"""Read recorded runs and group them into table cells. The ONLY source for a number.

Nothing in the reporting path can invent a value: every cell is assembled from
``results/index.jsonl`` entries, and a cell with no runs renders as an em dash
rather than a plausible-looking number.
"""

from __future__ import annotations

from collections import defaultdict

from tessera.eval.stats import summarise_seeds
from tessera.results import load_index

# Regimes in cascade order. This ordering IS the argument: each column is a stricter
# split than the one to its left, so inflation reads left to right.
REGIME_ORDER = [
    "r0_random",
    "r1_chrono",
    "r2_grouped",
    "r2_group_split",
    "r3_loro",
    "r4_crosscorpus",
]

MODEL_ORDER = [
    "majority",
    "nullmask",
    "random",
    "knn",
    "iforest",
    "rf",
    "lightgbm",
    "draft_literal",
    "draft_charitable",
    "tessera_base",
    "tessera_ev",
]


def _metric(row: dict, path: str):
    node = row.get("metrics", {})
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


class PooledDatasetError(RuntimeError):
    """Raised when one table cell would mix runs from different datasets."""


def available_datasets(metric: str = "binary.average_precision") -> list[str]:
    """Datasets that have at least one recorded run of ``metric``."""
    found = set()
    for row in load_index():
        if _metric(row, metric) is not None:
            found.add(str(row.get("tags", {}).get("dataset", "?")))
    return sorted(found)


def collect(metric: str = "binary.average_precision", *, dataset: str | None = None) -> dict:
    """Group recorded runs into ``{model: {split: [values across seeds]}}``.

    ``dataset`` is mandatory whenever more than one dataset has recorded runs.
    Averaging a fixture run with a real run into a single "mean +/- std" cell is a
    silent correctness failure - the std looks like seed variance when it is really
    a difference of corpora - so it is refused rather than produced.
    """
    if dataset is None:
        found = available_datasets(metric)
        if len(found) > 1:
            raise PooledDatasetError(
                f"runs exist for {len(found)} datasets ({', '.join(found)}); pass "
                "dataset= explicitly. Pooling them into one cell would report a "
                "between-corpus difference as seed variance."
            )
        dataset = found[0] if found else None

    cells: dict = defaultdict(lambda: defaultdict(list))
    meta: dict = defaultdict(lambda: defaultdict(dict))
    for row in load_index():
        tags = row.get("tags", {})
        if dataset is not None and tags.get("dataset") != dataset:
            continue
        v = _metric(row, metric)
        if v is None:
            continue
        model, split = tags.get("model", "?"), tags.get("split", "?")
        # r3 produces one run per held-out replica; pool them under one column.
        col = "r3_loro" if str(split).startswith("r3_loro") else split
        cells[model][col].append(float(v))
        meta[model][col] = {
            "regime": tags.get("regime"),
            "is_synthetic": tags.get("is_synthetic"),
            "mask_only_floor_ap": _metric(row, "mask_only_floor_ap"),
        }
    return {
        "metric": metric,
        "dataset": dataset,
        "cells": {m: dict(v) for m, v in cells.items()},
        "meta": {m: dict(v) for m, v in meta.items()},
    }


def summarise(collected: dict) -> dict:
    """Reduce each cell to mean/std/n_seeds."""
    return {
        "metric": collected["metric"],
        "dataset": collected["dataset"],
        "summary": {
            model: {split: summarise_seeds(vals) for split, vals in splits.items()}
            for model, splits in collected["cells"].items()
        },
        "meta": collected["meta"],
    }


def ordered_models(summary: dict) -> list[str]:
    present = list(summary["summary"])
    known = [m for m in MODEL_ORDER if m in present]
    return known + sorted(m for m in present if m not in known)


def ordered_regimes(summary: dict) -> list[str]:
    present = {s for v in summary["summary"].values() for s in v}
    known = [r for r in REGIME_ORDER if r in present]
    return known + sorted(present - set(known))
