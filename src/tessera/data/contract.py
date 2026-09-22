"""THE FROZEN DATA CONTRACT. Change this only by team agreement.

Every track depends on this and nothing else:

* **Data/Features** produces objects of this shape from AIT, BGL and CIC-IDS2017.
* **Model/Eval** consumes this shape and never touches a raw log.
* **Demo/Frontend** builds against :func:`tessera.data.synthetic.make_synthetic`,
  which emits this shape, so the demo is buildable in week 1 with no data at all.
* **Ledger/Simulator** consumes only the verdict stream, described at the bottom.

Freezing the contract before the AIT label join is what stops three of four people
sitting idle behind the project's riskiest task.

One sample is a ``(host, 60s window)`` pair inside one replica testbed. Every
modality is a different view of THE SAME window - that co-observation is what makes
the multimodality real rather than four datasets concatenated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Modality order is fixed everywhere: masks, gates, evidence and attribution all
# index into this order, so a permutation here silently corrupts attribution.
MODALITIES = ("m1_log", "m2_metrics", "m3_identity", "m4_graph")

M1_MAX_EVENTS = 128  # L: events per window, padded/truncated
M1_SRC_TYPES = 12  # apache access/error, auth, syslog, auditd, exim, ...
M2_N_FEATURES = 64
M3_N_CATEGORICAL = 8
M4_N_FEATURES = 24

WINDOW_SECONDS = 60


@dataclass
class WindowSet:
    """A set of windows. All arrays share length N and index the same windows.

    Held as plain NumPy so the contract has no framework dependency: the features
    track can build it with Polars, the model track reads it into torch, and the
    demo track exports it to JSON, without any of them agreeing on a dataframe.
    """

    # --- identity and provenance -------------------------------------------
    window_id: np.ndarray  # (N,) str  - stable unique id
    replica: np.ndarray  # (N,) str  - AIT testbed name; R3 folds on this
    corpus: np.ndarray  # (N,) str  - 'ait' | 'cicids2017c' | 'bgl'
    host_id: np.ndarray  # (N,) str
    user_id: np.ndarray  # (N,) str  - salted hash, never a raw username
    session_id: np.ndarray  # (N,) str
    t_start: np.ndarray  # (N,) float64 epoch seconds; R1/R2 order on this

    # --- modality 1: log template sequence ---------------------------------
    m1_template_id: np.ndarray  # (N, L) int32, 0 = pad
    m1_src_type: np.ndarray  # (N, L) int8
    m1_dt: np.ndarray  # (N, L) float32 seconds since previous event
    m1_length: np.ndarray  # (N,)   int32 true length before padding

    # --- modality 2: numeric network + system metrics ----------------------
    m2_numeric: np.ndarray  # (N, 64) float32

    # --- modality 3: identity categoricals ---------------------------------
    m3_categorical: np.ndarray  # (N, 8) int32 bucket indices

    # --- modality 4: graph structural features -----------------------------
    m4_graph: np.ndarray  # (N, 24) float32

    # --- availability ------------------------------------------------------
    # 1 iff that modality's source emitted >= 1 event in this window. STRUCTURAL,
    # not synthetic: a host with no monitoring agent never emits metrics. Because
    # absence correlates with host role and therefore with the label, the mask is a
    # leakage channel - hence the mask-only floor in every certificate.
    availability: np.ndarray  # (N, 4) int8, column order == MODALITIES

    # --- labels ------------------------------------------------------------
    y_bin: np.ndarray  # (N,) int8  0 benign / 1 attack
    y_coarse: np.ndarray  # (N,) int8  L1 attack category, -1 if benign
    y_step: np.ndarray  # (N,) int8  L2 attack step, -1 if benign

    # --- bookkeeping -------------------------------------------------------
    # Only ~8 of ~20 AIT log file types carry labels, and system-monitoring logs are
    # not among them. A window on a host emitting only unlabelled file types can
    # never be positive even mid-attack, which confounds any missing-modality claim.
    # Such windows are excluded and counted, not silently treated as benign.
    labellable: np.ndarray  # (N,) bool
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def n(self) -> int:
        return len(self.window_id)

    def validate(self) -> None:
        """Fail loudly on any contract violation. Called on construction."""
        n = len(self.window_id)
        expect = {
            "replica": (n,),
            "corpus": (n,),
            "host_id": (n,),
            "user_id": (n,),
            "session_id": (n,),
            "t_start": (n,),
            "m1_template_id": (n, M1_MAX_EVENTS),
            "m1_src_type": (n, M1_MAX_EVENTS),
            "m1_dt": (n, M1_MAX_EVENTS),
            "m1_length": (n,),
            "m2_numeric": (n, M2_N_FEATURES),
            "m3_categorical": (n, M3_N_CATEGORICAL),
            "m4_graph": (n, M4_N_FEATURES),
            "availability": (n, len(MODALITIES)),
            "y_bin": (n,),
            "y_coarse": (n,),
            "y_step": (n,),
            "labellable": (n,),
        }
        for name, shape in expect.items():
            arr = getattr(self, name)
            if arr.shape != shape:
                raise ValueError(f"{name}: expected shape {shape}, got {arr.shape}")
        if not np.isin(self.y_bin, (0, 1)).all():
            raise ValueError("y_bin must be 0/1")
        if not np.isin(self.availability, (0, 1)).all():
            raise ValueError("availability must be 0/1")
        if (self.m1_length > M1_MAX_EVENTS).any() or (self.m1_length < 0).any():
            raise ValueError(f"m1_length must lie in [0, {M1_MAX_EVENTS}]")
        # A positive label on an unlabellable window is impossible by construction.
        if (self.y_bin[~self.labellable] == 1).any():
            raise ValueError(
                "a window marked unlabellable carries a positive label; the label "
                "join or the labellable flag is wrong"
            )

    # ------------------------------------------------------------------ views

    def flat_features(self) -> np.ndarray:
        """Concatenated tabular view, for the classical baselines and the leakage
        controls (which need one matrix, not four tensors).

        The M1 sequence is reduced to simple summary statistics here on purpose: this
        view is what a gradient-boosting baseline legitimately sees, and it must not
        be given the sequence model's representation.
        """
        valid = self.m1_template_id > 0
        counts = valid.sum(1, keepdims=True).astype(np.float32)
        uniq = np.array(
            [len(np.unique(row[row > 0])) for row in self.m1_template_id], dtype=np.float32
        )[:, None]
        dt_mean = np.where(
            counts > 0, (self.m1_dt * valid).sum(1, keepdims=True) / np.maximum(counts, 1), 0
        )
        return np.hstack(
            [
                self.m2_numeric.astype(np.float32),
                self.m3_categorical.astype(np.float32),
                self.m4_graph.astype(np.float32),
                counts,
                uniq,
                dt_mean,
                self.availability.astype(np.float32),
            ]
        )

    def groups(self) -> dict[str, np.ndarray]:
        """Entity groups for disjointness checks. Host is long-lived - see splits.py."""
        return {
            "host_id": self.host_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
        }

    def subset(self, idx: np.ndarray) -> WindowSet:
        idx = np.asarray(idx)
        return WindowSet(
            **{
                f: getattr(self, f)[idx]
                for f in (
                    "window_id",
                    "replica",
                    "corpus",
                    "host_id",
                    "user_id",
                    "session_id",
                    "t_start",
                    "m1_template_id",
                    "m1_src_type",
                    "m1_dt",
                    "m1_length",
                    "m2_numeric",
                    "m3_categorical",
                    "m4_graph",
                    "availability",
                    "y_bin",
                    "y_coarse",
                    "y_step",
                    "labellable",
                )
            },
            meta={
                **self.meta,
                "subset_of": self.meta.get("name", "unknown"),
                "n_selected": int(len(idx)),
            },
        )

    def summary(self) -> dict:
        avail_rate = self.availability.mean(0)
        return {
            "n_windows": self.n,
            "n_replicas": int(len(np.unique(self.replica))),
            "corpora": sorted(set(map(str, self.corpus))),
            "prevalence_y_bin": float(self.y_bin.mean()),
            "n_positive": int(self.y_bin.sum()),
            "n_labellable": int(self.labellable.sum()),
            "n_excluded_unlabellable": int((~self.labellable).sum()),
            "availability_rate": {
                m: round(float(r), 4) for m, r in zip(MODALITIES, avail_rate, strict=True)
            },
            "mean_modalities_present": float(self.availability.sum(1).mean()),
            "time_span_hours": float((self.t_start.max() - self.t_start.min()) / 3600)
            if self.n
            else 0.0,
        }


# ---------------------------------------------------------------------------
# The verdict stream: the ONLY thing the ledger and simulator tracks consume.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """One detector output, and the only input to the ledger.

    Deliberately carries no IP, username, URL, header or body. The base paper's
    ledger stores source/destination IPs, geolocation and full request/response
    sets on a shared immutable chain, which makes it a privacy liability rather
    than a privacy mechanism. Here the ledger commits to a hash and the data stays
    off-chain, so an immutable record never becomes an immutable PII leak.
    """

    window_id: str
    host_hash: str  # salted SHA-256, never a hostname
    ts_bucket: int  # coarsened timestamp, not the exact event time
    verdict: int  # 0 benign / 1 attack
    score: float
    model_git_sha: str
    attribution: tuple = ()  # per-modality gate or evidence, in MODALITIES order
    uncertainty: float | None = None

    def canonical(self) -> dict:
        """The exact dict that gets hashed. Field order is fixed for determinism.

        ``score`` is formatted as a FIXED-PRECISION STRING, not a bare JSON
        number. Found live while building the browser demo's JS ledger port:
        Python's ``json.dumps`` renders a whole-number float with its trailing
        zero (``0.0`` -> the text ``"0.0"``), while JavaScript has no int/float
        distinction and ``JSON.stringify`` renders the identical value as
        ``"0"`` - so a verdict with a score that happens to round to a whole
        number (score exactly 0.0 is a real case: a maximally-confident benign
        prediction) would hash to two DIFFERENT leaves depending on which
        language recomputed it. A bare float is therefore never safe to hash
        across a language boundary this ledger is explicitly designed to be
        verified across (see web/js/merkle.js); a fixed-width string removes
        the ambiguity because both languages serialise the same string
        identically. Verified: `web/merkle-parity.test.mjs` reproduces the
        real Python ledger's roots exactly using this format.
        """
        return {
            "window_id": self.window_id,
            "host_hash": self.host_hash,
            "ts_bucket": self.ts_bucket,
            "verdict": int(self.verdict),
            "score": f"{round(float(self.score), 6):.6f}",
            "model_git_sha": self.model_git_sha,
        }
