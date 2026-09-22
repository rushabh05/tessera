# TESSERA

Multimodal cloud anomaly detection with a leakage-instrumented evaluation harness and
a verdict transparency log.

Final-year major project. Enhances **Nagarjun & Rajkumar, "Design of an Anomaly
Detection Framework for Delay and Privacy-Aware Blockchain-Based Cloud Deployments",
IEEE Access 12 (2024) 84843–84861** — treated as the *object of study*, not as the
source of the design.

## What this is

Four genuinely co-observed telemetry modalities from the same `(host, 60s window)` —
log template sequences, numeric network/system metrics, identity categoricals, and
graph structure — fused by an availability-masked gate, with every result reported
under a split-regime inflation cascade and a per-run leakage certificate.

The base paper's "multimodality" is three encoders reading one tabular vector; its
autoencoder output collapses the whole latent space to a single scalar batch
statistic; its convolution kernel has no learnable parameter; and its sidechain
objective is invariant to the variable it optimises. See
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

The two falsification findings about the base paper's sidechain optimisation:

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

We do not attempt to beat the base paper's 99.4 %. That number was produced under
conditions this harness is built to expose: a corpus pooling KDD99 with NSL-KDD — and
NSL-KDD *is* deduplicated KDD99 — under a random split.

## Status

P0 (harness) complete. See the plan for phases P1–P9.

## Licence

Code: MIT. Derived data artifacts inherit **CC BY-NC-SA 4.0** from the AIT Log Data
Set (Landauer et al., Zenodo 19483937); see [LICENSE-DATA](LICENSE-DATA). Derived
features are regenerated locally rather than redistributed.
