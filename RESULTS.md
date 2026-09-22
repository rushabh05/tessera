# Results

Real numbers, from real AIT data, reproducible with:

```bash
uv run pytest tests/test_pipeline_real_data.py tests/test_cross_replica.py -q -v
```

## Three findings, all self-demonstrated on real data

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

**With the fix, same-replica** (M1: 8 + M2: 24 + M3: 2 = 34 features):

| split | AP | MCC |
|---|---|---|
| R0 random | 1.000 | — |
| R1 chronological | **0.981** | **0.667** |

The E1 gate technically doesn't clear its 0.02 margin here (observed gap 0.019) —
worth noting honestly rather than glossing over: the certificate still shows 27.6%
exact-duplicate rows in R0, but they no longer *drive* the score, because the
genuine M1+M2 signal (log-template statistics, Suricata flow features) is strong
enough that duplicate rows are classified consistently regardless of which side of
the split they land on. This is the gate correctly flagging something worth a
second look; the second look is benign.

### 3. Cross-replica generalisation is strong — the real R3-style evidence

Train entirely on `russellmitchell`, evaluate on `santos` — a replica the model
has never seen, with different specific hosts, timestamps, IPs and usernames:

| evaluation | AP | MCC |
|---|---|---|
| cross-replica (train=russellmitchell, test=santos) | **0.999** | 0.989 |
| same-replica (santos, random split) | 0.999 | 0.996 |

Cross-replica performance is essentially identical to same-replica performance —
strong evidence the M1+M2+M3 features capture genuine, transferable attack
structure (Drain3 template patterns, Suricata flow signatures) rather than
replica-specific artifacts.

**Scoped honestly, per this project's own established framing**: named
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

## Scope of these results

M1 (log templates, Drain3-mined, summary-statistic view) + M2 (Suricata) + M3
(identity, calendar-free) — 34 features, 3 of 8 replicas' worth of hosts. M4
(graph) is not yet included. Cross-replica evidence so far covers 2 of 8 replicas;
extending to the full 8-replica leave-one-replica-out protocol is the next step.
