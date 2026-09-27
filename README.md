# TESSERA

Multimodal cloud anomaly detection with a leakage-instrumented evaluation harness and
a verdict transparency log.

Final-year major project. TESSERA is a leakage-instrumented, multimodal cloud
anomaly detector: it fuses genuinely separate telemetry modalities behind an
availability-masked gate, reports every result under a split-regime inflation
cascade with a per-run leakage certificate, and commits every verdict to a
tamper-evident transparency log — with a live, in-browser demo built and
parity-tested against the real Python implementation.

## What this is

Four co-observed feature groups (modalities) from the same `(host, 60s window)`, drawn
from two telemetry streams (log files and Suricata network events): log templates
(M1, from the logs), network metrics (M2) and destination-graph structure (M4, both
from the same Suricata flow records), and a derived host-identity group (M3). They are
fused by an availability-masked gate, with every result reported under a split-regime
inflation cascade and a per-run leakage certificate.

TESSERA fuses genuinely separate feature groups — log templates, network metrics,
destination-graph structure and host identity — rather than three encoders reading
one tabular vector split three ways; its GMU fusion produces a real per-modality
latent, not a whole latent space collapsed to a single scalar batch statistic; every
learnable component is actually learned, not a fixed-kernel convolution masquerading
as one; and an early drafted sidechain objective for tuning segment length turned
out to be invariant to the variable it was meant to optimise, so it was replaced. See
[NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) for the measured findings, including our
own corrections.

## Quick start

```bash
uv sync
uv run pytest tests -q
uv run python -m tessera.train.run model=lightgbm split=r2_grouped dataset=synthetic
uv run python -m tessera.report.tables
```

The whole pipeline, end to end, in one test — ingest → features → detect → explain →
ledger → inclusion proof → tamper detection → optimiser:

```bash
uv run pytest tests/test_e2e.py -q -v
```

Two findings from evaluating candidate objectives/optimisers for sidechain
segment-length tuning:

```bash
uv run python -m tessera.chainsim.report
```

## Guarantees the build enforces

These fail CI rather than appearing as caveats in a document:

| Guard | What it prevents |
|---|---|
| `test_no_group_overlap` | A leaked split producing a quietly inflated score |
| `test_scaler_train_only` | Scaler statistics fitted on test data |
| `test_permutation_chance` | A pipeline that scores above chance on shuffled labels |
| `test_merkle_proofs` | Broken tamper-evidence (property-based, many tree sizes) |
| `test_no_secrets_tracked` | A signing key or salt reaching git |
| `test_openmp_import_order` | The macOS LightGBM/torch segfault *and* deadlock |
| `check_no_hardcoded_numbers` | A reported metric no recorded run produced |
| E1 halt gate | A chronological split that fails to show expected inflation |

Every run emits a **leakage certificate** — exact and near-duplicate counts,
cross-split twins, asserted group disjointness, the availability-mask shortcut floor,
source-provenance AUC and a label-permutation check — printed beneath every table. A
table without its certificate is not reportable.

## Honest framing

Accuracy is never the headline. At 5 % prevalence a detector with ROC-AUC 0.98 and
97.4 % accuracy yields **P(attack | alert) = 0.008** at a realistic 1-in-10,000 base
rate. Primary metrics are average precision, MCC and class-conditional calibration,
with alert volume and base-rate-corrected precision reported alongside.

TESSERA reports the number its own leakage-checked evaluation produces, not the
higher number a leakier protocol would produce — that gap (0.928 vs 0.639 AP on the
same data, a corpus pooling KDD99 with NSL-KDD, and NSL-KDD *is* deduplicated KDD99,
under a random split) is measured directly, not assumed.

## Status

Phases P0 to P5 are complete: the leakage-instrumented harness (P0), real AIT
ingest, label join and window building (P1–P3), the 8-replica leave-one-replica-out
evaluation (P4) and the TESSERA-base neural model on real data (P5). The measured
findings, including the ones that went against us, are in [RESULTS.md](RESULTS.md)
and [NEGATIVE_RESULTS.md](NEGATIVE_RESULTS.md) (which records the findings of each
phase). Phases P6–P9 of the plan remain.

The static demo in [`web/`](web/README.md) is complete and parity-tested against the
Python model and ledger (`cd web && npm test`). It runs locally with `just serve` and
is built for GitHub Pages; see [web/README.md](web/README.md).

## Licence

Code: MIT. Derived data artifacts inherit **CC BY-NC-SA 4.0** from the AIT Log Data
Set (Landauer et al., Zenodo 19483937); see [LICENSE-DATA](LICENSE-DATA). The licence
allows non-commercial redistribution under the same terms; by project policy, derived
features are regenerated locally rather than redistributed, which keeps ShareAlike out
of the code. The demo site ships aggregate statistics only.

The dataset authors ask users to cite:

> M. Landauer et al., "Maintainable Log Datasets for Evaluation of Intrusion Detection
> Systems", *IEEE Transactions on Dependable and Secure Computing*, vol. 20, no. 4,
> pp. 3466–3482, 2023.
