# Results

Real numbers, from real AIT data, reproducible with:

```bash
# Findings 1-2 (leakage), fast:
uv run pytest tests/test_pipeline_real_data.py tests/test_cross_replica.py -q -v
# Finding 3 (8-replica generalisation), fast via cache:
uv run pytest tests/test_loro_real.py -q -v
# Finding 4 (neural model + attribution honesty), slow (~5 min, trains real models):
uv run pytest tests/test_tessera_base.py -q -v -m slow
```

## Four findings, all self-demonstrated on real data

### 1. Random-split evaluation is inflated by duplicate-row leakage (the project's central thesis)

Three hosts from the `russellmitchell` replica (`vpn`, `intranet_server`,
`inet-firewall`), M2 (Suricata `eve.json`, 24 features) only:

| split | AP | what it shows |
|---|---|---|
| R0 random | **0.928** | looks great |
| R1 chronological, same hosts | **0.639** | the truth — a 29-point drop, 14× the E1 gate's required margin |

The leakage certificate explains why: **28% exact duplicate rows**, **87%
near-duplicates** (many windows have no or minimal Suricata activity, producing
identical or near-identical all-zero vectors), **1,170 test rows byte-identical to
a training row**. The permutation control confirms the pipeline has no bug
(shuffled labels score at chance) — the inflation is in the *split*.

This is the project's central thesis, demonstrated in its own pipeline — not only
inferred from auditing the base paper's described methodology.

### 2. A second, self-found leakage bug — in this project's own feature design

Adding M1 (log-template stats) and M3 (identity) features, a first version of M3
included absolute calendar position: hour-of-day, day-of-week, minute-of-hour.
Result:

| features | R0 | R1 | gap |
|---|---|---|---|
| M1+M2+M3 **with** calendar features | 0.9998 | 0.5156 | **0.484** |
| M1+M2+M3 **without** calendar features | 1.0000 | **0.9813** | **0.019** |

`hour_of_day` was the single most important feature by a wide margin. The reason
is structural: in a single 4-day capture, "hour 3 on day Jan-24" occurs exactly
once, so a random split leaks the *exact calendar position* of held-out attack
windows into training — the model learns "attacks happen around hour X for this
capture," which generalises to nothing. Confirmed by a direct ablation (same
seed, same split, features in vs. out): removing four calendar features shrank
the gap from 0.484 to 0.019.

**Fixed**: `m3_identity.py` no longer computes calendar features at all. Only
`host_bucket` (hashed, non-reversible) and `n_sources_active` remain. Locked in
by `test_calendar_features_are_not_present_in_m3` and a signature-level guard
(`window_identity_vector` no longer accepts a timestamp parameter, so the leaky
computation cannot be silently reintroduced by resurrecting an old call site).

**With the fix, all four modalities** (M1: 8 + M2: 24 + M3: 2 + M4: 8 = 42
features — M4 adds graph structural features over a per-window destination-subnet
multigraph, built from the same Suricata data but requiring genuine cross-window
state: peer novelty and reappearance, which M2's per-window aggregates cannot
express):

| split | AP | MCC |
|---|---|---|
| R0 random | 1.000 | — |
| R1 chronological | **0.985** | **0.667** |

Adding M4 narrowed the gap further (0.019 → 0.015), and M4's own features
(`cumulative_peer_count`, `peer_reappearance_rate`) rank in the top 10 by
importance — genuine contribution, not dead weight.

The E1 gate technically doesn't clear its 0.02 margin here (observed gap ~0.015) —
worth noting honestly rather than glossing over: the certificate still shows 27.6%
exact-duplicate rows in R0, but they no longer *drive* the score, because the
genuine M1+M2+M4 signal is strong enough that duplicate rows are classified
consistently regardless of which side of the split they land on. This is the gate
correctly flagging something worth a second look; the second look is benign.

### 3. Cross-replica generalisation is strong — confirmed across all 8 replicas

Full leave-one-replica-out (`tests/test_loro_real.py`): train on 7 replicas, test
on the 8th, repeated for every replica in turn — all four modalities (42
features):

| held out | n test | n positive | AP | MCC | |
|---|---|---|---|---|---|
| fox | 25,104 | 3,437 | 0.9998 | 0.998 | |
| harrison | 25,614 | 6,327 | 0.9766 | 0.224 | ← low MCC despite high AP, see below |
| russellmitchell | 20,463 | 5,161 | 0.9999 | 0.999 | |
| santos | 19,995 | 3,331 | 0.9994 | 0.998 | |
| **shaw** | 29,207 | **6** | 0.2814 | 0.471 | **LOW SUPPORT — excluded from summary** |
| wardbeck | 24,750 | 2,784 | 0.9996 | 0.998 | |
| wheeler | 24,762 | 4,662 | 0.9998 | 0.998 | |
| wilson | 29,663 | 4,981 | 0.9996 | 0.998 | |

**Summary, excluding the flagged low-support fold** (7 folds):

| metric | mean | std | min | max |
|---|---|---|---|---|
| AP | **0.996** | 0.009 | 0.977 | 1.000 |
| MCC | 0.888 | 0.293 | 0.224 | 0.999 |

Strong and *tight* — a 0.009 standard deviation across seven independent held-out
replicas is real evidence the M1+M2+M3+M4 features transfer across independently
randomised executions of the scenario, not a lucky single pair.

**Two things reported honestly rather than smoothed over:**

- **`shaw` is a genuinely low-support fold, not a bug.** Its fixed 3-host/4-source
  subset has only 6 positive windows out of 29,207 — every other replica has
  hundreds to thousands. Investigated: shaw's capture spans 162 hours (longer
  than the others), and its 6 positive windows all cluster within a single
  44-minute episode near the end, across all three hosts. This is a real
  consequence of "attack parameters and execution order vary per replica"
  (Zenodo's own description) — for this particular randomised execution, the
  attack against these three specific hosts was much briefer. Per this project's
  own `MIN_SUPPORT_FOR_RATES` convention (already used for per-class reporting in
  `eval/metrics.py`, now applied per-fold too), this fold is **reported in full**
  but **excluded from the summary statistics** rather than silently averaged in —
  a naive pooled mean±std across all 8 folds would read 0.907 ± 0.253, which
  understates how strong the other 7 folds actually are.
- **`harrison`'s MCC (0.224) diverges sharply from its AP (0.977).** AP is
  threshold-free (ranking quality); MCC here uses a fixed 0.5 cutoff. This
  pattern — near-perfect ranking, poor fixed-threshold classification — usually
  means probability outputs are shifted for that replica rather than that the
  model has failed to learn anything, and is exactly the reason this project
  reports AP first and treats accuracy-family metrics as secondary. Not
  chased further here; a per-replica calibrated threshold or the TPR-at-FPR
  metrics already in `eval/metrics.py` would be the next step.

**Scoped honestly, per this project's established framing**: named
*cross-replica*, not *cross-organisation*. The AIT replicas are
parameter-randomised executions of the **same underlying scenario and attack
repertoire** (confirmed from the Zenodo record description), so this measures
robustness to that randomisation — a real and useful property, but not evidence
of transfer to a genuinely different environment or attack type. That broader
claim would need attack scenarios AIT does not provide.

## What this validates

- The leakage-instrumented harness (P0) works on real data, catching leakage in
  two different, independent places (duplicate rows; a self-authored feature bug)
  using two different mechanisms (the leakage certificate; the E1 halt gate).
- The label join (P2) and window builder (P3) are correct enough to produce
  coherent, explicable results at scale, and to transfer across replicas.
- **The base paper's methodology (99.4% accuracy, random split, pooled corpus
  with cross-split duplicates) is exactly the failure mode measured in finding 1.**
- The actual TESSERA-base neural model (not just the GBDT baseline) trains
  correctly on real multimodal data and matches the tuned classical baseline —
  and its own explanation mechanism was checked against independent evidence
  rather than trusted, the same rigor applied everywhere else in this project.

## Scope of these results

All four modalities: M1 (log templates, Drain3-mined, summary-statistic view),
M2 (Suricata flow/alert/DNS/HTTP/TLS aggregates), M3 (identity, calendar-free),
M4 (graph — destination-subnet peer structure with cross-window history) — 42
features, a fixed 3-host/4-source subset applied identically across all 8
downloaded replicas. Full leave-one-replica-out is complete (§3). Not yet done:
extending beyond this fixed host/source subset (more of the ~70 file types per
replica), and the neural TESSERA-base model (GMU fusion) — everything so far
uses LightGBM as the baseline.

## 4. TESSERA-base: the real neural model, and an honest limit on its explanations

Everything above uses LightGBM as the classifier. TESSERA-base
(`models/tessera_base.py`) is the actual project model: four small per-modality
encoders (Linear → GroupNorm → GELU → Linear) feeding a Gated Multimodal Unit
(availability-aware softmax fusion) and one MLP head. 5,005 parameters total.

**Trained on real data**: 7 replicas pooled (152,629 windows after carving a
validation slice), tested on the 8th (`santos`, held out), 30 epochs with early
stopping, MPS, 82.5 seconds:

| model | AP | MCC |
|---|---|---|
| TESSERA-base (neural, 5,005 params) | **0.9995** | 0.9969 |
| LightGBM (same fold, for comparison) | 0.9994 | 0.9975 |

A from-scratch neural model matches the tuned gradient-boosting baseline on real
data — not the point of the exercise (LightGBM is expected to be highly
competitive on tabular features; see the project's own pre-committed framing),
but confirms the architecture trains correctly and isn't leaving obvious
performance on the table.

### The GMU's own attribution should not be trusted at face value

The gate values are meant to explain a verdict — "this alert fired mostly
because of M2" — and are the mechanism the planned demo uses for exactly that.
Measured mean attribution on the test set: **M1 14%, M2 2%, M3 83%, M4 1%.**
Taken at face value, this says the model barely uses M1 (log templates) and
relies almost entirely on M3 (2 features: a hashed host bucket and a source
count).

**That reading is wrong, and here is the check that proves it.** Two
independent ablations:

| check | AP |
|---|---|
| GBDT on M1 alone | 0.9996 |
| GBDT on M3 alone | 0.6917 |
| GBDT on host_bucket alone (1 feature) | 0.4977 (MCC=0, prevalence floor 0.1666) |
| TESSERA-base **with** M1 (full model) | 0.9995 |
| TESSERA-base **without** M1 (zeroed, marked unavailable — the honest removal) | 0.9846 |

M1 alone carries nearly the entire signal for a classical model, and genuinely
removing it from the neural model — not just down-weighting it, actually
zeroing it and marking it structurally absent — costs real, measurable
performance (0.9995 → 0.9846). **The gate's 14% figure understates M1's true
importance by a wide margin.** `host_bucket` alone, meanwhile, is barely above
the prevalence floor and has MCC=0 — nowhere near what an 83%-attribution
reading would suggest M3 is doing.

**Locked in by test**: `test_gmu_attribution_does_not_match_naive_ablation_importance`
asserts the direction of this gap (low attribution, real importance) so it
cannot silently disappear.

**Consequence for the demo and the report**: gate values are shown as *a*
signal, never presented as ground truth for "what the model used." A proper
per-modality ablation is the only reliable importance measure this project has,
and is what any explanation claim should cite.
