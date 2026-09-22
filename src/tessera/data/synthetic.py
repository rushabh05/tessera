"""Synthetic corpus emitting the frozen contract. The stub every track builds on.

This is not a toy. It reproduces, with known ground truth, the four properties that
make the real corpus hard - so the harness can be validated before AIT arrives and
the demo can be built in week 1:

1. **Long-lived hosts.** A host spans the whole capture, so host-disjoint and
   chronological splitting cannot both hold (see ``eval/splits.py``).
2. **Label-correlated structural missingness.** Attack-prone hosts run fewer
   monitoring agents, so the availability mask predicts the label on its own. This
   is the shortcut the mask-only floor exists to measure, and it must be present in
   the fixture or the C2 machinery is never exercised.
3. **Bursty attacks.** Positives occur in a few short windows of the timeline, so a
   naive chronological cut really can leave a partition with zero positives.
4. **Partial labelling coverage.** Some hosts emit only unlabelled file types and
   can never be positive, mirroring AIT's ~8-of-20 labelled file types.

``redundant_copy_of`` duplicates one modality into another, giving the controlled-
redundancy ground truth that the evidence-independence claim is tested against:
under perfect dependence, fused uncertainty must not fall.
"""

from __future__ import annotations

import numpy as np

from tessera.data.contract import (
    M1_MAX_EVENTS,
    M1_SRC_TYPES,
    M2_N_FEATURES,
    M3_N_CATEGORICAL,
    M4_N_FEATURES,
    MODALITIES,
    WINDOW_SECONDS,
    WindowSet,
)


def make_synthetic(
    *,
    n_replicas: int = 8,
    hosts_per_replica: int = 6,
    windows_per_host: int = 1440,  # one day at 60s windows; a 600s gap is then ~0.4%
    prevalence: float = 0.06,
    seed: int = 0,
    signal: dict | None = None,
    missingness_label_coupling: float = 0.35,
    unlabellable_host_fraction: float = 0.2,
    redundant_copy_of: str | None = None,
    corpus: str = "ait",
) -> WindowSet:
    """Build a synthetic :class:`WindowSet` with known ground truth.

    ``signal`` sets per-modality effect size; the defaults make M2 and M1 informative,
    M3 weak and M4 weakest, so modality-ablation tables have something to show.
    """
    rng = np.random.default_rng(seed)
    sig = {"m1_log": 0.9, "m2_metrics": 1.1, "m3_identity": 0.35, "m4_graph": 0.25}
    sig.update(signal or {})

    n_hosts = n_replicas * hosts_per_replica
    n = n_hosts * windows_per_host

    replica = np.repeat(
        [f"replica{i:02d}" for i in range(n_replicas)], hosts_per_replica * windows_per_host
    ).astype(object)
    host_idx = np.repeat(np.arange(n_hosts), windows_per_host)
    host_id = np.array([f"h{h:03d}" for h in host_idx], dtype=object)

    # Each host is observed across the whole capture: long-lived by construction.
    per_host_t = np.arange(windows_per_host) * WINDOW_SECONDS
    t_start = np.tile(per_host_t, n_hosts).astype(np.float64) + 1_700_000_000.0

    # Sessions are short-lived (~8 windows), so session-disjointness is achievable.
    session_id = np.array([f"s{i // 8:05d}" for i in range(n)], dtype=object)
    user_id = np.array(
        [f"u{(host_idx[i] * 7 + i // 40) % (n_hosts * 3):04d}" for i in range(n)], dtype=object
    )
    window_id = np.array(
        [f"{replica[i]}:{host_id[i]}:{int(per_host_t[i % windows_per_host])}" for i in range(n)],
        dtype=object,
    )

    # ---- property 3: bursty attacks, a few contiguous runs per replica --------
    y_bin = np.zeros(n, dtype=np.int8)
    target_pos = int(round(prevalence * n))
    burst_len = 12
    n_bursts = max(1, target_pos // burst_len)
    # Attacks concentrate on a subset of hosts, so the attack surface is not uniform.
    attack_hosts = rng.choice(n_hosts, size=max(1, n_hosts // 3), replace=False)
    for _ in range(n_bursts):
        h = int(rng.choice(attack_hosts))
        start = int(rng.integers(0, max(1, windows_per_host - burst_len)))
        lo = h * windows_per_host + start
        y_bin[lo : lo + burst_len] = 1

    # ---- property 4: some hosts emit only unlabelled file types ---------------
    n_unlab = int(round(unlabellable_host_fraction * n_hosts))
    unlab_hosts = set(
        rng.choice(
            [h for h in range(n_hosts) if h not in set(attack_hosts.tolist())],
            size=min(n_unlab, n_hosts - len(attack_hosts)),
            replace=False,
        ).tolist()
    )
    labellable = ~np.isin(host_idx, list(unlab_hosts))
    y_bin[~labellable] = 0  # cannot be positive; enforced by the contract validator

    # ---- property 2: availability correlates with the label -------------------
    # Attack-prone hosts run fewer agents, so absence carries label information.
    host_agent_quality = rng.uniform(0.55, 0.98, n_hosts)
    host_agent_quality[attack_hosts] -= missingness_label_coupling * 0.5
    host_agent_quality = np.clip(host_agent_quality, 0.15, 0.99)
    base_p = host_agent_quality[host_idx][:, None]
    p_present = np.clip(base_p - missingness_label_coupling * 0.25 * y_bin[:, None], 0.05, 0.99)
    availability = (rng.random((n, len(MODALITIES))) < p_present).astype(np.int8)
    availability[:, 0] = 1  # logs are always present; they are the ingestion source

    # ---- modality payloads ----------------------------------------------------
    n_templates = 400
    m1_length = np.clip(rng.poisson(38, n) + 2, 1, M1_MAX_EVENTS).astype(np.int32)
    m1_template_id = np.zeros((n, M1_MAX_EVENTS), dtype=np.int32)
    m1_src_type = np.zeros((n, M1_MAX_EVENTS), dtype=np.int8)
    m1_dt = np.zeros((n, M1_MAX_EVENTS), dtype=np.float32)
    # Attack windows draw from a shifted template distribution: the M1 signal.
    benign_p = rng.dirichlet(np.ones(n_templates) * 0.6)
    attack_p = benign_p.copy()
    hot = rng.choice(n_templates, size=25, replace=False)
    attack_p[hot] += sig["m1_log"] * 0.04
    attack_p = attack_p / attack_p.sum()
    for i in range(n):
        L = int(m1_length[i])
        p = attack_p if y_bin[i] else benign_p
        m1_template_id[i, :L] = rng.choice(n_templates, size=L, p=p) + 1
        m1_src_type[i, :L] = rng.integers(0, M1_SRC_TYPES, size=L)
        m1_dt[i, :L] = rng.exponential(WINDOW_SECONDS / max(L, 1), size=L)

    def _numeric(dim: int, strength: float) -> np.ndarray:
        X = rng.normal(0, 1, (n, dim)).astype(np.float32)
        informative = max(1, dim // 4)
        X[:, :informative] += (strength * y_bin[:, None]).astype(np.float32)
        return X

    m2_numeric = _numeric(M2_N_FEATURES, sig["m2_metrics"])
    m4_graph = _numeric(M4_N_FEATURES, sig["m4_graph"])

    m3_categorical = np.zeros((n, M3_N_CATEGORICAL), dtype=np.int32)
    m3_categorical[:, 0] = host_idx
    m3_categorical[:, 1] = [int(u[1:]) % 4096 for u in user_id]
    for c in range(2, M3_N_CATEGORICAL):
        card = (64, 8, 6, 4096, 32, 16)[min(c - 2, 5)]
        base = rng.integers(0, card, n)
        flip = rng.random(n) < (0.12 * sig["m3_identity"] * y_bin)
        m3_categorical[:, c] = np.where(flip, rng.integers(0, card, n), base)

    # Zero out payloads for absent modalities, so "absent" is unambiguous downstream.
    m2_numeric[availability[:, 1] == 0] = 0.0
    m3_categorical[availability[:, 2] == 0] = 0
    m4_graph[availability[:, 3] == 0] = 0.0

    if redundant_copy_of is not None:
        # Controlled redundancy: make one modality an exact copy of another, so
        # fused uncertainty has a ground-truth requirement (it must not fall).
        src = {"m2_metrics": m2_numeric, "m4_graph": m4_graph}[redundant_copy_of]
        m4_graph = src[:, :M4_N_FEATURES].copy()
        availability[:, 3] = availability[:, 1]

    y_coarse = np.where(y_bin == 1, rng.integers(0, 4, n), -1).astype(np.int8)
    y_step = np.where(y_bin == 1, rng.integers(0, 10, n), -1).astype(np.int8)

    return WindowSet(
        window_id=window_id,
        replica=replica,
        corpus=np.full(n, corpus, dtype=object),
        host_id=host_id,
        user_id=user_id,
        session_id=session_id,
        t_start=t_start,
        m1_template_id=m1_template_id,
        m1_src_type=m1_src_type,
        m1_dt=m1_dt,
        m1_length=m1_length,
        m2_numeric=m2_numeric,
        m3_categorical=m3_categorical,
        m4_graph=m4_graph,
        availability=availability,
        y_bin=y_bin,
        y_coarse=y_coarse,
        y_step=y_step,
        labellable=labellable,
        meta={
            "name": "synthetic",
            "seed": seed,
            "generator": "tessera.data.synthetic.make_synthetic",
            "signal": sig,
            "missingness_label_coupling": missingness_label_coupling,
            "redundant_copy_of": redundant_copy_of,
            "warning": "SYNTHETIC fixture with known ground truth. Never a reported result.",
        },
    )


def make_fixture(seed: int = 0) -> WindowSet:
    """The tiny corpus used by the end-to-end test. Small, but exercises every path:
    multiple replicas, absent modalities, unlabellable hosts and bursty positives."""
    # 400 windows/host = ~6.7h span, so the 600s boundary gap is ~2.5% of the
    # timeline rather than 25% of it. Still small and fast, but temporally sane.
    return make_synthetic(
        n_replicas=3, hosts_per_replica=3, windows_per_host=400, prevalence=0.10, seed=seed
    )
