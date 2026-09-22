"""The single run orchestrator. Every reported number comes through here.

Order is deliberate: split, then certificate, then fit, then evaluate. The
certificate is built from the split BEFORE the model sees anything, so leakage
evidence cannot be shaped by having seen the result.
"""

from __future__ import annotations

import importlib
import sys

from omegaconf import DictConfig, OmegaConf

from tessera.data.contract import WindowSet
from tessera.eval import splits as S
from tessera.eval.certificate import build_certificate
from tessera.eval.metrics import evaluate
from tessera.paths import CONF_DIR, ensure_dirs
from tessera.results import RunRecord
from tessera.train.seed import seed_everything


def _resolve(path: str):
    module, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def build_dataset(cfg: DictConfig) -> WindowSet:
    builder = _resolve(cfg.dataset.builder)
    params = OmegaConf.to_container(cfg.dataset.params, resolve=True) or {}
    if cfg.dataset.get("is_synthetic", False):
        params.setdefault("seed", cfg.seed)
    return builder(**params)


def build_splits(cfg: DictConfig, ws: WindowSet) -> list[S.Split]:
    name = cfg.split.name
    p = OmegaConf.to_container(cfg.split.params, resolve=True) or {}
    if name == "r0_random":
        return [S.r0_random(ws.y_bin, seed=cfg.seed, **p)]
    if name == "r1_chrono":
        return [S.r1_chronological(ws.t_start, **p)]
    if name == "r2_grouped":
        groups = {g: ws.groups()[g] for g in cfg.split.enforce_groups}
        return [S.r2_grouped(ws.t_start, groups, **p)]
    if name == "r2_group_split":
        return [S.r2_group_split(ws.groups()[cfg.split.group], seed=cfg.seed, **p)]
    if name == "r3_loro":
        return S.r3_leave_one_replica_out(ws.replica, seed=cfg.seed, **p)
    raise KeyError(f"unknown split '{name}'")


def run_one(cfg: DictConfig, ws: WindowSet, split: S.Split) -> RunRecord:
    """Fit and evaluate one (model, split) pair and return the unwritten record."""
    S.assert_partitions_have_positives(split, ws.y_bin)

    train, test = ws.subset(split.train_idx), ws.subset(split.test_idx)
    val = ws.subset(split.val_idx) if len(split.val_idx) else None

    # Certificate first: leakage evidence must not be able to react to the result.
    cert = build_certificate(
        split=split,
        y=ws.y_bin,
        X=ws.flat_features(),
        groups={g: ws.groups()[g] for g in cfg.split.get("enforce_groups", [])} or None,
        availability_mask=ws.availability,
        source=ws.corpus if len(set(map(str, ws.corpus))) > 1 else None,
        seed=cfg.seed,
        run_permutation=bool(cfg.leakage.run_permutation),
    )

    build = _resolve(cfg.model.builder)
    params = OmegaConf.to_container(cfg.model.params, resolve=True) or {}
    if cfg.model.name != "majority":
        params.setdefault("seed", cfg.seed)
    model = build(cfg.model.name, **params)
    model.fit(train, val)
    pred = model.predict(test)

    metrics = evaluate(
        test.y_bin,
        pred.score,
        confidence=pred.confidence,
        windows_per_day=float(cfg.eval.windows_per_day),
        threshold=float(cfg.eval.threshold),
    )

    record = RunRecord(
        config=OmegaConf.to_container(cfg, resolve=True),
        tags={
            "model": cfg.model.name,
            "dataset": cfg.dataset.name,
            "is_synthetic": bool(cfg.dataset.get("is_synthetic", False)),
            "split": split.name,
            "regime": split.regime,
            "seed": int(cfg.seed),
        },
        seed_report=None,
        leakage_certificate=cert.as_dict(),
    )
    record.log(
        **metrics,
        model_describe=model.describe(),
        split_info=split.as_dict(ws.y_bin),
        dataset_summary=ws.summary(),
        # The floor this run must clear to mean anything, carried WITH the result
        # so a table cell can never be read without it.
        mask_only_floor_ap=(cert.mask_only_floor or {}).get("average_precision"),
    )
    return record


def main(cfg: DictConfig) -> None:
    ensure_dirs()
    report = seed_everything(int(cfg.seed), deterministic=bool(cfg.deterministic))
    ws = build_dataset(cfg)

    if cfg.dataset.get("is_synthetic", False):
        print(
            f"[synthetic] dataset='{cfg.dataset.name}' - validates the harness "
            "against known ground truth; not a reportable detection result."
        )

    folds = build_splits(cfg, ws)
    for split in folds:
        rec = run_one(cfg, ws, split)
        rec.seed_report = report.as_dict()
        path = rec.write()
        b = rec.metrics["binary"]
        floor = rec.metrics.get("mask_only_floor_ap")
        verdict = (
            "ABOVE floor" if floor is None or b["average_precision"] > floor else "AT/BELOW floor"
        )
        print(
            f"{cfg.model.name:<10} {split.name:<22} AP={b['average_precision']:.4f} "
            f"MCC={b['mcc']:+.4f} prev={b['prevalence']:.4f} "
            f"floor={floor if floor is None else round(floor, 4)} [{verdict}]  -> {path.name}"
        )


def cli() -> None:
    """Entry point. Hydra composes conf/ and applies command-line overrides."""
    import hydra

    rel = CONF_DIR.relative_to(
        __import__("pathlib").Path(__file__).resolve().parent.parent.parent.parent
    )
    hydra.main(version_base=None, config_path=str(("../" * 3) + rel.name), config_name="config")(
        main
    )()


if __name__ == "__main__":
    sys.exit(cli())
