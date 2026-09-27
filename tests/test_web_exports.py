"""The static demo's exported data files (web/data/*.json): schema, licence
safety, honesty of provenance, and freshness against the real cache.

Three groups:

1. Checks on the SHIPPED files - always run (the files are part of the site):
   schema keys, no ClassStats below MIN_SUPPORT_FOR_RATES, pattern
   probabilities, the licence scan (no list long enough to carry row-level
   data), LORO coverage with shaw flagged, provenance strings, transcribed
   numbers still present in RESULTS.md, and the host-label fix.
2. Unit tests of the statistics code on a small hand-built dataset - always run.
3. Checks against the real processed cache - skipped when it is absent, like
   tests/test_loro_real.py: the statistics are deterministic and the shipped
   replica_stats.json is exactly what the exporter computes today.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np
import pytest

from tessera.data.ait.unpack import REPLICAS
from tessera.demo.export_web_data import (
    ATTRIBUTION_VS_ABLATION,
    CALENDAR_LEAK,
    RECORDED_OPTIMISER_RUN,
    TESSERA_BASE,
    LicenceScanError,
    availability_mask,
    dumps_web,
    licence_scan,
)
from tessera.demo.replica_stats import (
    FEATURE_NAMES,
    SIGMA_FLOOR,
    compute_replica_stats,
    iter_class_stats,
    round_sig,
)
from tessera.demo.synthetic_demo_data import DEMO_HOST_LABELS, DEMO_HOSTS
from tessera.eval.loro_real import cache_path
from tessera.eval.metrics import MIN_SUPPORT_FOR_RATES
from tessera.features.m3_identity import host_bucket
from tessera.features.pipeline import M1_SLICE, M2_SLICE, M3_SLICE, M4_SLICE
from tessera.paths import REPO_ROOT

WEB_DATA = REPO_ROOT / "web" / "data"
RESULTS_MD = (REPO_ROOT / "RESULTS.md").read_text(encoding="utf-8")
FIXED_LABELS = {
    "vpn": "VPN Gateway",
    "intranet_server": "Intranet Server",
    "inet-firewall": "Internet Firewall",
}

_cache_complete = all(cache_path(r).exists() for r in REPLICAS)
needs_cache = pytest.mark.skipif(
    not _cache_complete,
    reason="processed dataset cache not built for all 8 replicas; run "
    "build_all_replica_datasets() first (~11 min on a cold cache)",
)


def _load(name: str) -> dict:
    return json.loads((WEB_DATA / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stats() -> dict:
    return _load("replica_stats.json")


@pytest.fixture(scope="module")
def real_results() -> dict:
    return _load("real_results.json")


@pytest.fixture(scope="module")
def design() -> dict:
    return _load("design_notes.json")


# ---------------------------------------------------------------- 1. shipped files

CLASS_STATS_KEYS = {
    "n",
    "patterns",
    "features",
    "dup_rate",
    "inherited_features",
    "dup_rate_inherited",
}


def test_replica_stats_schema(stats):
    for key in (
        "schema",
        "generated_by",
        "licence_note",
        "min_support",
        "benign_block_windows",
        "n_features",
        "feature_names",
        "modalities",
        "feature_meta",
        "hosts",
        "replicas",
        "global_pooled",
    ):
        assert key in stats, key
    assert stats["schema"] == "tessera-replica-stats/v1"
    assert stats["generated_by"] == "uv run python -m tessera.demo.export_web_data"
    assert stats["min_support"] == MIN_SUPPORT_FOR_RATES == 20
    assert stats["benign_block_windows"] == 60
    assert stats["n_features"] == 42
    assert stats["feature_names"] == list(FEATURE_NAMES)
    assert [(m["id"], m["slice"]) for m in stats["modalities"]] == [
        ("m1_log", [M1_SLICE.start, M1_SLICE.stop]),
        ("m2_metrics", [M2_SLICE.start, M2_SLICE.stop]),
        ("m3_identity", [M3_SLICE.start, M3_SLICE.stop]),
        ("m4_graph", [M4_SLICE.start, M4_SLICE.stop]),
    ]
    assert [h["id"] for h in stats["hosts"]] == list(DEMO_HOSTS)
    for h in stats["hosts"]:
        assert h["label"] == FIXED_LABELS[h["id"]]
        assert h["host_bucket"] == host_bucket(h["id"])
    assert len(stats["feature_meta"]) == 42
    for j, fm in enumerate(stats["feature_meta"]):
        assert set(fm) == {"name", "modality", "integer", "upper", "kind"}
        assert fm["name"] == FEATURE_NAMES[j]
        assert fm["kind"] == {32: "host_bucket", 33: "n_sources"}.get(j, "continuous")
        assert fm["upper"] >= 0
    assert [r["id"] for r in stats["replicas"]] == list(REPLICAS)
    for rep in stats["replicas"]:
        for key in (
            "n_windows",
            "n_positive",
            "prevalence",
            "span_hours",
            "exact_duplicate_rate",
            "hosts",
            "pooled",
        ):
            assert key in rep, (rep["id"], key)
        assert list(rep["hosts"]) == list(DEMO_HOSTS)
        assert sum(h["n_windows"] for h in rep["hosts"].values()) == rep["n_windows"]
        assert sum(h["n_positive"] for h in rep["hosts"].values()) == rep["n_positive"]
        for hs in rep["hosts"].values():
            assert {"n_windows", "n_positive", "n_attack_episodes", "benign", "attack"} <= set(hs)
            assert (hs["n_attack_episodes"] > 0) == (hs["n_positive"] > 0)
    for where, cs in iter_class_stats(stats):
        if cs is not None:
            assert set(cs) == CLASS_STATS_KEYS, where


def test_no_class_stats_below_min_support(stats):
    """Any group of fewer than 20 real windows must be null, never a smaller group."""
    n_null = 0
    for where, cs in iter_class_stats(stats):
        if cs is None:
            n_null += 1
            continue
        assert cs["n"] >= MIN_SUPPORT_FOR_RATES, where
    by_id = {r["id"]: r for r in stats["replicas"]}
    for rep in stats["replicas"]:
        for h, hs in rep["hosts"].items():
            n_benign = hs["n_windows"] - hs["n_positive"]
            assert (hs["attack"] is None) == (hs["n_positive"] < MIN_SUPPORT_FOR_RATES), (rep, h)
            assert (hs["benign"] is None) == (n_benign < MIN_SUPPORT_FOR_RATES), (rep, h)
            if hs["attack"] is not None:
                assert hs["attack"]["n"] == hs["n_positive"]
    # The measured facts: vpn attacks are single windows everywhere, and shaw has only
    # 6 positive windows in total, so its pooled attack stats must be absent.
    assert all(r["hosts"]["vpn"]["attack"] is None for r in stats["replicas"])
    assert by_id["shaw"]["n_positive"] == 6
    assert by_id["shaw"]["pooled"]["attack"] is None
    assert n_null > 0


def test_patterns_are_a_probability_distribution(stats):
    for where, cs in iter_class_stats(stats):
        if cs is None:
            continue
        ps = [p["p"] for p in cs["patterns"]]
        assert abs(sum(ps) - 1.0) < 1e-4, (where, sum(ps))
        assert all(0 < p <= 1 for p in ps), where
        keys = set()
        for pat in cs["patterns"]:
            assert len(pat["a"]) == 3 and set(pat["a"]) <= {0, 1}, where
            assert pat["n_sources"] in (1, 2, 3), where
            keys.add((tuple(pat["a"]), pat["n_sources"]))
        assert len(keys) == len(cs["patterns"]), f"{where}: duplicate pattern"


def test_feature_stats_are_valid(stats):
    for where, cs in iter_class_stats(stats):
        if cs is None:
            continue
        feats = cs["features"]
        assert len(feats) == 42, where
        assert feats[32] is None and feats[33] is None, where
        for j, f in enumerate(feats):
            if j in (32, 33):
                continue
            assert set(f) == {"zero_rate", "mu", "sigma", "tau"}, (where, j)
            assert 0 <= f["zero_rate"] <= 1, (where, j)
            assert f["sigma"] >= SIGMA_FLOOR - 1e-12, (where, j)
            assert f["tau"] >= 0, (where, j)
            assert all(math.isfinite(v) for v in f.values()), (where, j)
        assert all(0 <= j < 42 and j not in (32, 33) for j in cs["inherited_features"])
        assert 0 <= cs["dup_rate"] <= 1, where


def _long_numeric_lists(obj, key=None, path="$"):
    """Independent of the exporter's own scan: every list holding more than 64
    numbers, other than the per-feature arrays of exactly 42 entries."""
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            found += _long_numeric_lists(v, k, f"{path}.{k}")
    elif isinstance(obj, list):
        per_feature = key in ("feature_names", "feature_meta", "features") and len(obj) == 42
        n_numbers = sum(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj)
        if len(obj) > 64 and not per_feature:
            found.append((path, len(obj), n_numbers))
        for i, v in enumerate(obj):
            found += _long_numeric_lists(v, None, f"{path}[{i}]")
    return found


@pytest.mark.parametrize("name", ["replica_stats.json", "real_results.json", "design_notes.json"])
def test_licence_safety_no_row_level_lists(name):
    """No list in a shipped file is long enough to carry row-level AIT data."""
    payload = _load(name)
    assert _long_numeric_lists(payload) == []
    licence_scan(payload)  # and the exporter's own gate agrees


def test_licence_scan_rejects_row_like_payloads():
    with pytest.raises(LicenceScanError):
        licence_scan({"scores": list(range(65))})
    with pytest.raises(LicenceScanError):
        licence_scan({"a": {"b": [[0.5] * 42] * 100}})
    with pytest.raises(LicenceScanError):
        licence_scan({"features": [0.0] * 43})
    licence_scan({"features": [None] * 42, "feature_names": ["x"] * 42, "k": list(range(64))})


def test_replica_stats_stays_small():
    assert (WEB_DATA / "replica_stats.json").stat().st_size < 160_000


def test_real_results_loro_covers_every_replica(real_results):
    loro = real_results["loro"]
    folds = {f["held_out"]: f for f in loro["folds"]}
    assert set(folds) == set(REPLICAS) and len(loro["folds"]) == 8
    assert folds["shaw"]["low_support"] is True
    assert folds["shaw"]["n_pos_test"] == 6
    assert [f["held_out"] for f in loro["folds"] if f["low_support"]] == ["shaw"]
    s = loro["summary"]
    assert s["n_folds"] == 8
    assert s["excluded_low_support"] == ["shaw"]
    assert s["average_precision"]["n"] == 7 and s["mcc"]["n"] == 7
    assert s["naive_all_folds_ap"]["n"] == 8
    for f in loro["folds"]:
        assert set(f) == {
            "held_out",
            "n_train",
            "n_test",
            "n_pos_test",
            "test_prevalence",
            "average_precision",
            "mcc",
            "low_support",
        }


def test_recomputed_loro_matches_the_recorded_finding(real_results):
    """The exporter re-ran LORO; it must agree with RESULTS.md finding 3 at the
    precision RESULTS.md reports (0.996 +/- 0.009; naive 0.907 +/- 0.253)."""
    s = real_results["loro"]["summary"]
    assert round(s["average_precision"]["mean"], 3) == 0.996
    assert round(s["average_precision"]["std"], 3) == 0.009
    assert round(s["mcc"]["mean"], 3) == 0.888
    assert round(s["naive_all_folds_ap"]["mean"], 3) == 0.907
    assert round(s["naive_all_folds_ap"]["std"], 3) == 0.253
    folds = {f["held_out"]: f for f in real_results["loro"]["folds"]}
    assert round(folds["harrison"]["average_precision"], 4) == 0.9766
    assert round(folds["harrison"]["mcc"], 3) == 0.224
    assert round(folds["shaw"]["average_precision"], 4) == 0.2814


def test_recomputed_leakage_matches_the_recorded_finding(real_results):
    d = real_results["leakage_duplicates"]
    assert d["replica"] == "russellmitchell"
    assert round(d["r0_random_ap"], 3) == 0.928
    assert round(d["r1_chronological_ap"], 3) == 0.639
    assert round(d["exact_duplicate_rate"], 2) == 0.28
    assert round(d["near_duplicate_rate"], 2) == 0.87
    assert d["test_rows_identical_to_train"] == 1170
    a = real_results["all_modalities_r0_r1"]
    assert round(a["r0_ap"], 3) == 1.000
    assert round(a["r1_ap"], 3) == 0.985
    assert round(a["r1_mcc"], 3) == 0.667


def test_every_block_states_its_provenance(real_results, design):
    blocks = {k: v for k, v in real_results.items() if isinstance(v, dict)}
    blocks.update({f"design.{k}": v for k, v in design.items() if isinstance(v, dict)})
    assert len(blocks) >= 10
    for name, b in blocks.items():
        assert b["provenance"] in ("recomputed", "transcribed"), name
        prefix = "recomputed" if b["provenance"] == "recomputed" else "transcribed from"
        assert b["source"].startswith(prefix), name
        if b["provenance"] == "transcribed":
            # names the test that recorded it, or says why no test can
            assert re.search(r"tests/test_\w+\.py|no test|chainsim\.report", b["source"]), name
    for name in ("calendar_leak", "tessera_base", "attribution_vs_ablation"):
        assert real_results[name]["provenance"] == "transcribed", name
    for name in (
        "loro",
        "mask_only_floor",
        "leakage_duplicates",
        "all_modalities_r0_r1",
        "dataset",
    ):
        assert real_results[name]["provenance"] == "recomputed", name


def test_transcribed_numbers_are_in_results_md(real_results):
    """Transcribed means copied: every transcribed figure must still be in RESULTS.md."""
    tb = TESSERA_BASE
    expected_text = {
        tb["tessera_ap"]: "0.9995",
        tb["tessera_mcc"]: "0.9969",
        tb["lightgbm_ap"]: "0.9994",
        tb["lightgbm_mcc"]: "0.9975",
        tb["n_train_windows"]: "152,629",
        tb["train_seconds"]: "82.5 seconds",
        tb["n_parameters"]: "5,005",
    }
    for value, text in expected_text.items():
        assert text in RESULTS_MD, (value, text)
        assert float(text.split()[0].replace(",", "")) == value
    att = ATTRIBUTION_VS_ABLATION["mean_attribution"]
    assert "M1 14%, M2 2%, M3 83%, M4 1%" in RESULTS_MD
    assert (att["m1_log"], att["m2_metrics"], att["m3_identity"], att["m4_graph"]) == (
        0.14,
        0.02,
        0.83,
        0.01,
    )
    for row in ATTRIBUTION_VS_ABLATION["ablation"]:
        assert f"{row['ap']:.4f}" in RESULTS_MD, row
    for row in CALENDAR_LEAK["rows"]:
        for k in ("r0_ap", "r1_ap"):
            assert f"{row[k]:.4f}" in RESULTS_MD, row
        assert f"{row['gap']:.3f}" in RESULTS_MD, row
    # the shipped file carries exactly these constants
    assert real_results["tessera_base"] == round_sig(TESSERA_BASE)
    assert real_results["attribution_vs_ablation"] == round_sig(ATTRIBUTION_VS_ABLATION)
    shipped_cal = {k: v for k, v in real_results["calendar_leak"].items() if k in CALENDAR_LEAK}
    assert shipped_cal == round_sig(CALENDAR_LEAK)


FLOOR_FOLD_KEYS = {"held_out", "n_pos_test", "average_precision", "low_support"}


def test_mask_only_floor_schema(real_results):
    """The shortcut floor: the same LORO folds, a model that sees only structure."""
    fl = real_results["mask_only_floor"]
    loro = real_results["loro"]
    loro_folds = {f["held_out"]: f for f in loro["folds"]}
    for key in ("source", "model", "features", "labelling_rule", "folds", "summary"):
        assert key in fl, key
    assert "mask_only_score" in fl["source"] and "leave-one-replica-out" in fl["source"]
    assert "tests/test_web_exports.py::test_shipped_mask_only_floor_is_fresh" in fl["source"]
    for block in (fl, fl["host_only"], fl["mask_and_host"]):
        folds = {f["held_out"]: f for f in block["folds"]}
        assert set(folds) == set(REPLICAS) and len(block["folds"]) == 8
        s = block["summary"]
        # exactly the folds the headline LORO mean excludes, for the same reason
        assert s["excluded_low_support"] == loro["summary"]["excluded_low_support"] == ["shaw"]
        included = []
        for name, f in folds.items():
            assert set(f) == FLOOR_FOLD_KEYS, name
            assert f["n_pos_test"] == loro_folds[name]["n_pos_test"], name
            assert f["low_support"] == loro_folds[name]["low_support"], name
            assert f["average_precision"] is None or 0 <= f["average_precision"] <= 1, name
            if not f["low_support"]:
                included.append(f["average_precision"])
        ap = s["average_precision"]
        assert ap["n"] == len(included) == 7
        assert ap["mean"] == pytest.approx(float(np.mean(included)), abs=1e-5)
        assert ap["std"] == pytest.approx(float(np.std(included, ddof=1)), abs=1e-5)
        assert ap["min"] == pytest.approx(min(included), abs=1e-6)


def test_mask_only_floor_attack_share_is_consistent(real_results):
    sh = real_results["mask_only_floor"]["attack_share"]
    totals = real_results["dataset"]["totals"]
    assert sh["n_attack_windows"] == totals["n_positive"]
    assert sh["n_attack_windows"] + sh["n_benign_windows"] == totals["n_windows"]
    for cls, n in (("attack", sh["n_attack_windows"]), ("benign", sh["n_benign_windows"])):
        pats = sh["patterns"][cls]
        assert sum(p["n"] for p in pats) == n, cls
        assert all(len(p["a"]) == 3 and set(p["a"]) <= {0, 1} for p in pats), cls
        m1 = sum(p["n"] for p in pats if p["a"][0] == 1) / n
        assert sh[f"m1_present_rate_{cls}"] == pytest.approx(m1, rel=1e-5), cls
    assert 0 < sh["n_attack_on_host"] <= sh["n_attack_windows"]
    share = sh["n_attack_on_host"] / sh["n_attack_windows"]
    assert sh["share_of_attacks_on_host"] == pytest.approx(share, rel=1e-5)
    assert sh["host"] == "inet-firewall"


def test_host_only_floor_agrees_with_the_transcribed_ablation(real_results):
    """RESULTS.md's one-off 'GBDT on host_bucket alone' on santos (0.4977) is the
    santos fold of the recomputed host-only floor: an independent cross-check of a
    transcribed number."""
    row = next(r for r in ATTRIBUTION_VS_ABLATION["ablation"] if "host_bucket" in r["check"])
    folds = {f["held_out"]: f for f in real_results["mask_only_floor"]["host_only"]["folds"]}
    assert round(folds["santos"]["average_precision"], 4) == row["ap"]


def test_availability_mask_recovers_an_all_zero_m2_window():
    """A present M2 window can be all zeros; n_sources_active is what reveals it."""
    X = np.zeros((4, 42), dtype=np.float32)
    X[0, M1_SLICE.start] = 3  # M1 only... plus an all-zero M2
    X[0, M3_SLICE.start + 1] = 2
    X[1, M2_SLICE.start] = 5  # M2 only
    X[1, M3_SLICE.start + 1] = 1
    X[2, M1_SLICE.start], X[2, M2_SLICE.start], X[2, M4_SLICE.start] = 1, 1, 1
    X[2, M3_SLICE.start + 1] = 3
    X[3, M4_SLICE.start] = 2  # M4 with an all-zero M2
    X[3, M3_SLICE.start + 1] = 2
    assert availability_mask(X).astype(int).tolist() == [
        [1, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 1],
    ]
    X[1, M3_SLICE.start + 1] = 3  # inconsistent with the slices
    with pytest.raises(AssertionError):
        availability_mask(X)


def test_design_numbers_come_from_one_export(design):
    """The EHO decision is written from the recomputed optimiser values the charts
    draw, and the recorded NEGATIVE_RESULTS run is kept only as a labelled note."""
    oc = design["optimiser_comparison"]
    un = design["eho_unreachability"]
    decision = next(c for c in design["design_decisions"] if "reject EHO" in c["decision"])
    ev = decision["evidence"]
    assert f"S = {oc['ground_truth']['best_segment_length']}" in ev
    assert f"LH <= {un['max_learning_rate_admitting_optimum']:.4f}" in ev
    for r in oc["results"]:
        if r["optimiser"].startswith("EHO") or r["optimiser"] == "random search":
            v = r["optimality_gap_pct"]
            text = f"{v:.3f}%" if v < 0.1 else f"{round(v, 1):g}%"
            assert text in ev, (r["optimiser"], text)
    recorded = RECORDED_OPTIMISER_RUN
    assert oc["recorded_run"]["label"] == "recorded run (NEGATIVE_RESULTS N2)"
    for c in design["design_decisions"]:
        assert f"{recorded['results'][-1]['optimality_gap_pct']}%" not in c["evidence"]
        if (
            recorded["ground_truth"]["best_segment_length"]
            != oc["ground_truth"]["best_segment_length"]
        ):
            assert f"S = {recorded['ground_truth']['best_segment_length']}" not in c["evidence"]


def test_ledger_decision_matches_what_the_code_hashes(design):
    """merkle.leaf_hash is SHA-256(0x00 || data) with no salt: the decision must not
    say the leaf is salted."""
    from tessera.ledger.merkle import LEAF_PREFIX, leaf_hash

    assert leaf_hash(b"x") == __import__("hashlib").sha256(LEAF_PREFIX + b"x").digest()
    decision = next(c for c in design["design_decisions"] if "ledger" in c["decision"].lower())
    assert "salted SHA-256" not in decision["evidence"]
    assert "RFC 6962" in decision["evidence"] and "unsalted" in decision["evidence"]


def test_design_payload(design):
    for key in (
        "schema",
        "generated_by",
        "optimiser_comparison",
        "eho_unreachability",
        "fh_invariance",
        "design_decisions",
    ):
        assert key in design, key
    assert design["schema"] == "tessera-design-notes/v1"
    oc = design["optimiser_comparison"]
    gt = oc["ground_truth"]
    assert {"best_segment_length", "best_objective", "n_evaluations"} <= set(gt)
    assert gt["n_evaluations"] == 65536 - 8 + 1  # every integer S: genuinely exhaustive
    for r in oc["results"]:
        assert {"optimiser", "best_segment_length", "best_objective", "optimality_gap"} <= set(r)
        assert r["optimality_gap"] >= 0 and r["optimality_gap_pct"] >= 0, r  # vs true optimum
        assert r["n_evaluations"] == oc["budget_per_metaheuristic"]
    eho = next(r for r in oc["results"] if r["optimiser"].startswith("EHO"))
    rnd = next(r for r in oc["results"] if r["optimiser"] == "random search")
    assert eho["optimality_gap_pct"] > 100 * rnd["optimality_gap_pct"]

    un = design["eho_unreachability"]
    assert un["optimum"] == gt["best_segment_length"]
    assert un["search_domain"] == [8, 65536]
    assert un["rows"] and not any(row["optimum_reachable"] for row in un["rows"])
    for row in un["rows"]:
        a, b = row["reachable_interval"]
        assert a <= b and not (a <= un["optimum"] <= b)

    fh = design["fh_invariance"]
    assert len(fh["fh_values"]) == len(fh["nsc_values"]) >= 5
    assert len(set(fh["fh_values"])) == 1 and fh["absolute_spread"] == 0
    assert fh["d_fh_d_nsc_is_zero"] is True


def test_design_decisions_cite_real_files(design):
    decisions = design["design_decisions"]
    assert any("accuracy" in c["decision"] for c in decisions)
    for c in decisions:
        assert set(c) == {"decision", "why", "evidence", "source_file"}
        assert c["decision"].strip() and c["why"].strip() and c["evidence"].strip()
        for f in c["source_file"].split(";"):
            fname = f.strip().split(" (")[0]
            assert (REPO_ROOT / fname).exists(), fname


def test_host_labels_are_fixed(stats):
    """DEMO_HOST_LABELS once mislabelled two of the three roles (intranet_server as
    'Internal Firewall', inet-firewall as 'Web / Mail Server')."""
    assert dict(zip(DEMO_HOSTS, DEMO_HOST_LABELS, strict=True)) == FIXED_LABELS
    demo = _load("demo_windows.json")
    assert len(demo["windows"]) == 60
    for w in demo["windows"]:
        assert w["host_label"] == FIXED_LABELS[w["host"]], w["window_id"]
    raw = (WEB_DATA / "demo_windows.json").read_text(encoding="utf-8")
    assert "Internal Firewall" not in raw and "Web / Mail Server" not in raw


# ---------------------------------------------------------------- 2. unit tests


def _toy_datasets(seed: int = 0) -> dict[str, dict]:
    """One replica, hand-built so every statistic has a known answer.

    vpn             180 benign windows = 3 benign blocks of 60; log(n_flow) has
                    block means 0, 2, 4 and noise sd 0.1 (so tau ~ 2, sigma ~ 0.1);
                    n_alert is nonzero in only 5 windows (too few -> inherited).
    intranet_server 40 benign, 10 attack, 50 benign: the attack group (10 < 20)
                    must be null, with one episode; n_alert is nonzero in 30 of
                    its benign windows, so the pooled benign group has support.
    inet-firewall   30 attack, 30 benign, 30 attack: two episodes; every attack
                    window at an odd position copies its predecessor exactly.
    """
    rng = np.random.default_rng(seed)
    parts = {"X": [], "y": [], "host": [], "window_start": []}

    def add(host: str, y: np.ndarray, X: np.ndarray):
        n = len(y)
        X[:, 32] = host_bucket(host)
        X[:, 33] = 1 + (X[:, M1_SLICE] != 0).any(1) + (X[:, M4_SLICE] != 0).any(1)
        parts["X"].append(X.astype(np.float32))
        parts["y"].append(y.astype(np.int8))
        parts["host"].append(np.array([host] * n, dtype=object))
        parts["window_start"].append(1_000_000 + 60 * np.arange(n, dtype=np.int64))

    X = np.zeros((180, 42))
    X[:, 8] = np.exp(np.repeat([0.0, 2.0, 4.0], 60) + 0.1 * rng.standard_normal(180))
    X[[3, 50, 70, 100, 170], 9] = 7.0
    add("vpn", np.zeros(180), X)

    y = np.r_[np.zeros(40), np.ones(10), np.zeros(50)]
    X = np.zeros((100, 42))
    X[:, 8] = np.exp(rng.standard_normal(100))
    X[:30, 9] = np.exp(1 + 0.5 * rng.standard_normal(30))
    add("intranet_server", y, X)

    y = np.r_[np.ones(30), np.zeros(30), np.ones(30)]
    X = np.zeros((90, 42))
    X[:, 8] = np.exp(3 + rng.standard_normal(90))
    X[:, 0] = np.where(y == 1, 5.0, 0.0)  # M1 present in attack windows only
    for i in range(1, 90):
        if y[i] == 1 and y[i - 1] == 1 and i % 2 == 1:
            X[i] = X[i - 1]
    add("inet-firewall", y, X)

    # shuffle row order: the code must sort each host's windows by time itself
    order = rng.permutation(370)
    return {"toy": {k: np.concatenate(v)[order] for k, v in parts.items()}}


@pytest.fixture(scope="module")
def toy_stats() -> dict:
    return compute_replica_stats(_toy_datasets(), replicas=("toy",))


def test_toy_stretches_episodes_and_null_groups(toy_stats):
    rep = toy_stats["replicas"][0]
    hosts = rep["hosts"]
    assert rep["n_windows"] == 370 and rep["n_positive"] == 70
    assert hosts["vpn"]["n_attack_episodes"] == 0 and hosts["vpn"]["attack"] is None
    assert hosts["intranet_server"]["n_attack_episodes"] == 1
    assert hosts["intranet_server"]["attack"] is None  # 10 windows < min_support
    assert hosts["intranet_server"]["benign"]["n"] == 90
    assert hosts["inet-firewall"]["n_attack_episodes"] == 2
    assert hosts["inet-firewall"]["attack"]["n"] == 60
    assert rep["pooled"]["attack"]["n"] == 70
    assert toy_stats["global_pooled"]["benign"]["n"] == 300
    assert round(rep["span_hours"], 6) == round(180 * 60 / 3600, 6)


def test_toy_log_moments_separate_within_and_between_stretch_spread(toy_stats):
    f = toy_stats["replicas"][0]["hosts"]["vpn"]["benign"]["features"][8]
    assert f["zero_rate"] == 0
    assert f["mu"] == pytest.approx(2.0, abs=0.05)
    assert f["sigma"] == pytest.approx(0.1, abs=0.02)
    assert f["tau"] == pytest.approx(2.0, abs=0.1)


def test_toy_low_support_statistics_are_inherited_not_computed(toy_stats):
    vpn = toy_stats["replicas"][0]["hosts"]["vpn"]["benign"]
    pooled = toy_stats["replicas"][0]["pooled"]["benign"]
    # n_alert: nonzero in only 5 of 180 vpn windows - the zero rate aggregates 180
    # windows, but the log-moments would come from 5, so they are the coarser
    # (replica-pooled, 35 nonzero values) group's.
    assert 9 not in pooled["inherited_features"]
    assert 9 in vpn["inherited_features"]
    assert vpn["features"][9]["zero_rate"] == pytest.approx(1 - 5 / 180, rel=1e-5)
    parent = pooled["features"][9]
    assert (vpn["features"][9]["mu"], vpn["features"][9]["sigma"]) == (
        parent["mu"],
        parent["sigma"],
    )
    # M1 is never present in benign windows here: no support at all, contract default.
    assert vpn["features"][0] == {"zero_rate": 1, "mu": 0, "sigma": SIGMA_FLOOR, "tau": 0}


def test_toy_duplicates_and_patterns(toy_stats):
    fw = toy_stats["replicas"][0]["hosts"]["inet-firewall"]
    # 58 attack windows have a predecessor in their episode; the odd-position ones copy it.
    n_copies = sum(1 for i in range(1, 90) if (i < 30 or i >= 61) and i % 2 == 1)
    assert fw["attack"]["dup_rate"] == pytest.approx(n_copies / 58, rel=1e-5)
    assert fw["benign"]["dup_rate"] == 0
    assert fw["attack"]["patterns"] == [{"a": [1, 1, 0], "n_sources": 2, "p": 1}]
    assert toy_stats["replicas"][0]["hosts"]["vpn"]["benign"]["patterns"] == [
        {"a": [0, 1, 0], "n_sources": 1, "p": 1}
    ]


def test_toy_stats_are_deterministic_and_licence_safe(toy_stats):
    again = compute_replica_stats(_toy_datasets(), replicas=("toy",))
    assert again == toy_stats
    licence_scan(toy_stats)
    assert json.loads(dumps_web(toy_stats, indent=0)) == toy_stats


def test_round_sig_and_writer():
    assert round_sig({"a": 0.123456789, "b": [1.0, 2.5e-9], "c": np.float32(3.0)}) == {
        "a": 0.123457,
        "b": [1, 2.5e-9],
        "c": 3,
    }
    with pytest.raises(ValueError):
        round_sig({"x": float("nan")})
    obj = {"short": [1, 2], "long": [{"k": i, "v": "x" * 30} for i in range(12)]}
    text = dumps_web(obj, indent=1, inline_width=60)
    assert json.loads(text) == obj
    assert '"short":[1,2]' in text and text.count("\n") > 12


# ---------------------------------------------------------------- 3. real cache


@pytest.fixture(scope="module")
def real_datasets():
    from tessera.demo.replica_stats import load_real_datasets

    return load_real_datasets()


@needs_cache
def test_replica_stats_deterministic_and_shipped_file_is_fresh(real_datasets, stats):
    first = compute_replica_stats(real_datasets)
    second = compute_replica_stats(real_datasets)
    assert first == second
    assert stats == first, (
        "web/data/replica_stats.json is stale - regenerate it with `just web-data`"
    )


@needs_cache
def test_shipped_dataset_block_matches_the_cache(real_datasets, real_results):
    from tessera.demo.export_web_data import dataset_block

    assert real_results["dataset"] == round_sig(dataset_block(real_datasets))


@needs_cache
def test_transcribed_tessera_base_training_size_matches_the_cache(real_datasets):
    """RESULTS.md's 152,629 training windows = the 7 non-santos replicas minus the
    test's int(0.15 * n) validation carve - a cross-check of a transcribed number."""
    n = sum(len(d["y"]) for r, d in real_datasets.items() if r != "santos")
    assert n - int(0.15 * n) == TESSERA_BASE["n_train_windows"]


@needs_cache
def test_availability_mask_on_the_real_cache(real_datasets):
    """Row sums equal n_sources_active, and every attack window has M1 activity -
    the labelling fact the About FAQ states."""
    for name, d in real_datasets.items():
        X, y = np.asarray(d["X"]), np.asarray(d["y"]).astype(bool)
        A = availability_mask(X)
        assert (A.sum(axis=1) == np.rint(X[:, M3_SLICE.start + 1])).all(), name
        assert A[y, 0].all(), name


@needs_cache
def test_shipped_mask_only_floor_is_fresh(real_datasets, real_results):
    from tessera.demo.export_web_data import mask_only_floor_block

    assert real_results["mask_only_floor"] == round_sig(mask_only_floor_block(real_datasets)), (
        "web/data/real_results.json's mask_only_floor is stale - regenerate it with "
        "`uv run python -m tessera.demo.export_web_data --only results`"
    )
