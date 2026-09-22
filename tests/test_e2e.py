"""END-TO-END: the whole process, in one test, asserting real values at every hop.

ingest -> split -> leakage certificate -> fit -> predict -> explain -> verdict stream
-> ledger append -> inclusion proof -> consistency proof -> tamper -> DETECT
-> segmentation optimiser -> results record -> generated table

This is the test that answers "is any of this actually wired together, or is it a
pile of modules?" Every stage asserts on a value it computed, not on the absence of
an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.chainsim.cost_model import Calibration, Workload, evaluate_cost
from tessera.chainsim.optimisers import eho, exhaustive
from tessera.data.contract import MODALITIES, Verdict
from tessera.data.synthetic import make_fixture
from tessera.eval.certificate import build_certificate
from tessera.eval.metrics import evaluate
from tessera.eval.splits import assert_partitions_have_positives, r0_random
from tessera.ledger.checkpoint import SegmentedLog
from tessera.ledger.merkle import MerkleLog, canonical_bytes, verify_consistency
from tessera.models.baselines.dummy import build
from tessera.results import RunRecord
from tessera.train.seed import seed_everything

# Fast, deterministic calibration so the test never depends on machine timing.
FIXED_CAL = Calibration(t_hash_s=3.0e-7, t_sign_s=6.5e-5, t_verify_s=1.5e-4)


def test_whole_process_end_to_end():
    seed_everything(0, deterministic=True)

    # ---- 1. ingest ------------------------------------------------------
    ws = make_fixture(seed=0)
    ws.validate()
    assert ws.n > 0
    assert ws.y_bin.sum() > 0, "fixture has no positives; nothing downstream is testable"
    assert (ws.availability == 0).any(), "fixture has no absent modality"

    # ---- 2. split -------------------------------------------------------
    split = r0_random(ws.y_bin, seed=0)
    assert_partitions_have_positives(split, ws.y_bin)
    assert not (set(split.train_idx) & set(split.test_idx))

    # ---- 3. leakage certificate BEFORE the model sees anything ----------
    cert = build_certificate(
        split=split,
        y=ws.y_bin,
        X=ws.flat_features(),
        availability_mask=ws.availability,
        seed=0,
        run_permutation=False,
    )
    cd = cert.as_dict()
    assert cd["control_1_exact_duplicates"]["n_rows"] == ws.n
    floor = cd["control_4_mask_only_floor"]["average_precision"]
    assert 0.0 <= floor <= 1.0
    assert "LEAKAGE CERTIFICATE" in cert.render()

    # ---- 4/5. fit and predict -------------------------------------------
    train, test = ws.subset(split.train_idx), ws.subset(split.test_idx)
    model = build("lightgbm", seed=0).fit(train)
    pred = model.predict(test)
    assert pred.score.shape == (test.n,)
    assert np.isfinite(pred.score).all()

    # ---- 6. explain ------------------------------------------------------
    assert pred.attribution.shape == (test.n, len(MODALITIES))
    assert set(pred.top_modality().tolist()) <= set(MODALITIES)

    # ---- 7. evaluate -----------------------------------------------------
    metrics = evaluate(test.y_bin, pred.score, confidence=pred.confidence)
    ap = metrics["binary"]["average_precision"]
    assert 0.0 <= ap <= 1.0
    assert ap > test.y_bin.mean(), f"AP {ap:.4f} at or below prevalence; model learned nothing"
    assert metrics["point_adjustment"] == "not used"

    # ---- 8. verdict stream ----------------------------------------------
    verdicts = [
        Verdict(
            window_id=str(test.window_id[i]),
            host_hash=f"hh{abs(hash(str(test.host_id[i]))) % 10**8:08d}",
            ts_bucket=int(test.t_start[i] // 300),
            verdict=int(pred.score[i] >= 0.5),
            score=float(pred.score[i]),
            model_git_sha="test",
            attribution=tuple(float(x) for x in pred.attribution[i]),
        )
        for i in range(test.n)
    ]
    assert len(verdicts) == test.n
    # The ledger payload must carry no identifying field.
    payload = verdicts[0].canonical()
    for forbidden in ("ip", "url", "username", "user", "geolocation", "host_id"):
        assert forbidden not in payload, f"verdict payload leaks '{forbidden}'"

    # ---- 9. ledger append + inclusion proof ------------------------------
    log = MerkleLog()
    for v in verdicts:
        log.append_json(v.canonical())
    assert len(log) == len(verdicts)
    root_full = log.root()
    for i in (0, len(verdicts) // 2, len(verdicts) - 1):
        assert log.inclusion_proof(i).verify(root_full), f"inclusion proof {i} failed"

    # ---- 10. consistency proof -------------------------------------------
    half = len(verdicts) // 2
    retained_root = log.root(half)  # what an auditor keeps
    assert verify_consistency(
        half, retained_root, len(log), root_full, log.consistency_proof(half, len(log))
    )

    # ---- 11. TAMPER, and detect it ---------------------------------------
    log.tamper(0, canonical_bytes({"window_id": "FORGED", "verdict": 0, "score": 0.0}))
    assert not verify_consistency(
        half, retained_root, len(log), log.root(), log.consistency_proof(half, len(log))
    ), "a rewritten history verified as consistent; tamper-evidence is broken"

    # ---- 12. segmented log with signed checkpoints -----------------------
    seg = SegmentedLog(segment_length=64)
    for v in verdicts:
        seg.append_json(v.canonical())
    chain = seg.verify_chain()
    assert chain["intact"], chain["problems"]
    assert chain["n_segments"] >= 1

    # ---- 13. segmentation optimiser vs ground truth ----------------------
    wl = Workload(n_entries=len(verdicts) * 500)
    obj = lambda S: evaluate_cost(S, wl, FIXED_CAL).total_delay_s  # noqa: E731
    truth = exhaustive.search(obj, 8, 16384, step=8)
    assert 8 < truth.best_x < 16384, (
        f"optimum at a boundary ({truth.best_x}); cost model degenerate"
    )
    e = eho.search(obj, 8, 16384, n_herds=8, n_iterations=5, learning_rate=0.5, seed=0)
    assert e.best_value >= truth.best_value, (
        "a heuristic beat exhaustive search; ground truth is wrong"
    )

    # ---- 14. record the run and regenerate a table ------------------------
    rec = RunRecord(
        config={"dataset": {"name": "e2e_fixture"}, "model": {"name": "lightgbm"}, "seed": 0},
        tags={
            "model": "lightgbm",
            "dataset": "e2e_fixture",
            "split": split.name,
            "regime": split.regime,
            "seed": 0,
            "is_synthetic": True,
        },
        leakage_certificate=cd,
    )
    rec.log(**metrics, mask_only_floor_ap=floor)
    path = rec.write()
    assert (path / "metrics.json").exists()
    assert (path / "leakage_certificate.json").exists()
    assert (path / "env.json").exists()

    from tessera.report.tables import inflation_cascade_markdown

    table = inflation_cascade_markdown(dataset="e2e_fixture")
    assert "lightgbm" in table
    assert f"{ap:.4f}" in table, "the recorded AP does not appear in the generated table"


def test_run_record_refuses_to_overwrite():
    """Append-only: a result can never be silently replaced by a later one."""
    rec = RunRecord(config={"x": 1}, tags={"model": "t", "dataset": "t", "split": "s"})
    rec.log(binary={"average_precision": 0.5})
    rec.write()
    with pytest.raises(FileExistsError):
        rec.write()
