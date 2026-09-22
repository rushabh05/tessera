# Results

Real numbers, from real AIT data, reproducible with:

```bash
uv run pytest tests/test_pipeline_real_data.py -q -v
```

## Headline finding: random-split evaluation is inflated by duplicate-row leakage

This is the project's central thesis, demonstrated in its own pipeline on real
data — not just found in an audit of the base paper.

Three hosts from the `russellmitchell` replica (`vpn`, `intranet_server`,
`inet-firewall`), joined labels (line-number exact, hand-verified) with 24-feature
M2 (Suricata `eve.json`) windows, evaluated with a LightGBM baseline (200 trees,
seed 0):

| split | description | AP | MCC | n test |
|---|---|---|---|---|
| **R0** | random, stratified (leakage upper bound) | **0.928** | 0.880 | 4093 |
| **R1** | chronological, same hosts | **0.639** | 0.698 | 4063 |
| **R2b** | host-disjoint (3 hosts, 2 train / 1 test) | **0.003** | −0.003 | 6820 |

**R0 → R1 isolates the duplicate-leakage effect** (same hosts, same overall
distribution, only the split order changes): a **29-point drop**, 14× the E1 gate's
required 2-point margin. The leakage certificate explains why: **28.1% of rows are
exact duplicates**, **87.1% are near-duplicates** (many windows have no or minimal
Suricata activity, producing identical or near-identical all-zero feature vectors),
and **1,170 test rows in R0 are byte-identical to a training row** — those
predictions are memorised, not generalised.

**R2b's further collapse to near-chance is a separate, confounded finding** — with
only 3 hosts of wildly different character (`inet-firewall` is 93.8% positive from
a near-continuous DNS-exfiltration attack; `vpn` is 0.3% positive from a single
foothold event), training on 2 and testing on 1 conflates duplicate-leakage removal
with genuine distribution shift and a near-total absence of the test host's attack
pattern in training. This is not yet a clean cross-host generalisation claim — that
needs the full R3 leave-one-replica-out protocol across all 8 replicas, now that
all 8 are downloaded.

**The permutation control confirms the pipeline itself has no bug**: shuffled
labels score at chance (0.2522 ≈ prevalence 0.2522) — the inflation is in the
*split*, not an evaluation-harness defect.

## What this validates

- The leakage-instrumented harness (P0) works on real data, not only on
  synthetic fixtures with deliberately injected duplicates.
- The E1 halt gate fires correctly on a real 29-point gap.
- The label join (P2) — hand-verified against individual real attack events — is
  correct enough to produce a coherent, explicable result at scale.
- **The base paper's methodology (99.4% accuracy, random split, pooled corpus with
  cross-split duplicates) is exactly the failure mode measured here**, now shown
  directly rather than only inferred from its description.

## Scope of this result

M2 (Suricata network/flow metrics) only — 24 features, 3 of 8 replicas' worth of
hosts. M1 (log templates), M3 (identity), M4 (graph) are not yet wired in; adding
them is expected to *reduce* the near-duplicate rate (richer feature space, fewer
coincidental collisions) and is the next increment.
