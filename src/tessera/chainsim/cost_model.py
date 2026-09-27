"""Cost model for the segmented verdict log, with constants MEASURED on this machine.

Why this exists. Our first draft objective for tuning segment length was
``fh = (1/NEB) * sum(dr + dw + dh + dv) * em`` - the summed read, write, hash and
verify delays times mining energy. Per-block read/write/hash/verify cost does not
depend on WHERE the chain is cut, so that objective is constant in its own decision
variable. Optimising it cannot change anything (demonstrated in
``segment_objective.py``).

A well-posed objective needs terms that genuinely vary with segment length S:

* inclusion-proof size and verification cost grow as ``log2(S)``
* amortised checkpoint signing cost per append falls as ``C_sign / S``
* locating a historical proof grows as ``log2(N / S)`` through the segment index
* crash recovery must rehash the ACTIVE segment, costing ``O(S)`` per restart
* the resident working set grows as ``S``

Getting this model right took two corrections, both recorded here because a cost
model that does not reward the decision it is making is the same class of error as
the draft objective it replaced.

1. A first version charged a LINEAR SCAN of every archived segment per proof
   request. That term (``requests * n_segments * t_seek``) reached 1500 s at S=16 and
   swamped everything, making total delay monotonically decreasing in S. It was also
   simply wrong: a real log has a segment index, so a historical lookup is
   ``O(log n_segments)``, not ``O(n_segments)``.
2. A second version added recovery cost but kept the linear scan, so it stayed
   monotone; the recovery term alone is far too small to counteract it, because
   SHA-256 on this host runs at ~0.29 us per entry.

With an indexed archive, the terms that fall with S (checkpoint signatures,
``N/S * t_sign``) and those that rise with S (append path ``N log2(S) t_hash`` and
recovery rehash ``restarts * S * t_hash``) balance at a genuine interior optimum, so
an exhaustive search can establish the ground-truth optimum against which any
metaheuristic is judged.

Energy is MODELLED, not measured. Apple Silicon exposes no RAPL, ``powermetrics``
requires sudo, and thermal/DVFS noise exceeds the effect being measured. The model
below multiplies measured operation counts by a stated power constant; every energy
figure is printed with those constants attached so it is never mistaken for a
wall-plug measurement.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# Power constants for the energy MODEL. Stated, not measured.
CPU_ACTIVE_WATTS = 4.0  # single-core sustained draw, Apple M-series order of magnitude
STORAGE_JOULES_PER_BYTE = 2.5e-10  # NVMe read/write energy, order of magnitude


@dataclass
class Calibration:
    """Timings measured on the host, so the model is grounded in real op costs."""

    t_hash_s: float
    t_sign_s: float
    t_verify_s: float
    leaf_bytes: int = 32
    entry_bytes: int = 220
    # Indexed archive: one seek plus a B-tree-style index descent over the segment
    # count. NOT a linear scan of every segment - see the note in the module docstring.
    t_archive_seek_s: float = 2.0e-4
    t_index_step_s: float = 5.0e-6
    # Recovery: on restart the ACTIVE segment's tree must be rebuilt from its
    # entries, costing one hash per entry. Unavoidable in any real deployment, and
    # the term that makes segment length worth choosing at all.
    restarts_per_100k_entries: float = 2.0
    measured_on: str = ""
    note: str = "hash/sign/verify measured on this host; power constants are modelled"

    def as_dict(self) -> dict:
        return {
            "t_hash_s": self.t_hash_s,
            "t_sign_s": self.t_sign_s,
            "t_verify_s": self.t_verify_s,
            "t_archive_seek_s": self.t_archive_seek_s,
            "t_index_step_s": self.t_index_step_s,
            "restarts_per_100k_entries": self.restarts_per_100k_entries,
            "leaf_bytes": self.leaf_bytes,
            "entry_bytes": self.entry_bytes,
            "cpu_active_watts": CPU_ACTIVE_WATTS,
            "storage_joules_per_byte": STORAGE_JOULES_PER_BYTE,
            "measured_on": self.measured_on,
            "note": self.note,
        }


def calibrate(*, n_hash: int = 20000, n_sign: int = 2000) -> Calibration:
    """Measure hash, sign and verify cost on this machine."""
    import platform

    payload = b"x" * 220
    t0 = time.perf_counter()
    for _ in range(n_hash):
        hashlib.sha256(payload).digest()
    t_hash = (time.perf_counter() - t0) / n_hash

    key = Ed25519PrivateKey.generate()
    t0 = time.perf_counter()
    for _ in range(n_sign):
        key.sign(payload)
    t_sign = (time.perf_counter() - t0) / n_sign

    sig, pub = key.sign(payload), key.public_key()
    t0 = time.perf_counter()
    for _ in range(n_sign):
        pub.verify(sig, payload)
    t_verify = (time.perf_counter() - t0) / n_sign

    return Calibration(
        t_hash_s=t_hash,
        t_sign_s=t_sign,
        t_verify_s=t_verify,
        measured_on=f"{platform.machine()} / {platform.system()} {platform.release()}",
    )


@dataclass
class Workload:
    """The verdict stream. Driven by the detector's real output, not synthetic arrivals."""

    n_entries: int
    proof_requests: int = 0
    historical_proof_fraction: float = 0.3  # share of proof requests hitting an archive
    seconds_observed: float = 1.0

    def __post_init__(self) -> None:
        if self.proof_requests == 0:
            # One audit per 100 verdicts, a deliberately stated assumption.
            self.proof_requests = max(1, self.n_entries // 100)


@dataclass
class CostBreakdown:
    segment_length: int
    append_delay_s: float
    proof_verify_delay_s: float
    archive_fetch_delay_s: float
    recovery_delay_s: float
    total_delay_s: float
    energy_j: float
    throughput_entries_per_s: float
    resident_bytes: int
    n_segments: int
    mean_proof_path: float
    components: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "segment_length": self.segment_length,
            "append_delay_s": self.append_delay_s,
            "proof_verify_delay_s": self.proof_verify_delay_s,
            "archive_fetch_delay_s": self.archive_fetch_delay_s,
            "recovery_delay_s": self.recovery_delay_s,
            "total_delay_s": self.total_delay_s,
            "energy_j_modelled": self.energy_j,
            "throughput_entries_per_s": self.throughput_entries_per_s,
            "resident_bytes": self.resident_bytes,
            "n_segments": self.n_segments,
            "mean_proof_path": self.mean_proof_path,
            "components": self.components,
        }


def evaluate_cost(segment_length: int, workload: Workload, cal: Calibration) -> CostBreakdown:
    """Total delay, modelled energy and throughput at one segment length."""
    S = max(1, int(segment_length))
    N = workload.n_entries
    n_segments = max(1, math.ceil(N / S))

    # Appending: each entry is hashed, and the incremental tree update touches
    # ~log2(S) internal nodes within the active segment.
    nodes_per_append = math.log2(S) if S > 1 else 0.0
    append_hashes = N * (1.0 + nodes_per_append)
    # One checkpoint signature per closed segment, amortised across its entries.
    signs = max(0, n_segments - 1)
    append_delay = append_hashes * cal.t_hash_s + signs * cal.t_sign_s

    # Verifying an inclusion proof costs one hash per path element.
    mean_path = math.log2(S) if S > 1 else 0.0
    proof_verify = workload.proof_requests * (mean_path * cal.t_hash_s + cal.t_verify_s)

    # A historical proof locates its segment through an index: one seek plus a
    # logarithmic descent over the segment count.
    hist = workload.proof_requests * workload.historical_proof_fraction
    index_depth = math.log2(n_segments) if n_segments > 1 else 0.0
    archive_fetch = hist * (cal.t_archive_seek_s + index_depth * cal.t_index_step_s)

    # Recovery: each restart rehashes the active segment. Grows linearly in S and is
    # what stops the optimum running away to a single unbounded segment.
    n_restarts = (N / 100_000.0) * cal.restarts_per_100k_entries
    recovery = n_restarts * S * cal.t_hash_s

    total = append_delay + proof_verify + archive_fetch + recovery

    bytes_written = N * cal.entry_bytes + append_hashes * cal.leaf_bytes
    energy = total * CPU_ACTIVE_WATTS + bytes_written * STORAGE_JOULES_PER_BYTE
    resident = S * (cal.leaf_bytes + cal.entry_bytes)

    return CostBreakdown(
        segment_length=S,
        append_delay_s=append_delay,
        proof_verify_delay_s=proof_verify,
        archive_fetch_delay_s=archive_fetch,
        recovery_delay_s=recovery,
        total_delay_s=total,
        energy_j=energy,
        throughput_entries_per_s=(N / total if total > 0 else float("inf")),
        resident_bytes=int(resident),
        n_segments=n_segments,
        mean_proof_path=mean_path,
        components={
            "append_hashes": append_hashes,
            "checkpoint_signatures": signs,
            "n_restarts": n_restarts,
            "grows_with_S": "proof path (log2 S), recovery rehash (S), resident set (S)",
            "shrinks_with_S": "checkpoint signatures (N/S), archive seeks (N/S)",
        },
    )


def objective(segment_length: int, workload: Workload, cal: Calibration) -> float:
    """Scalarised objective: total delay. Used by the single-objective optimisers.

    The genuinely multi-objective delay/energy/throughput trade-off is handled by
    NSGA-II over the same cost model; this scalar exists so the metaheuristics can be
    compared against exhaustive ground truth on identical terms.
    """
    return evaluate_cost(segment_length, workload, cal).total_delay_s
