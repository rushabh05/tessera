# Negative results and corrections

Opened on day one, per plan, so that "what did not work" is a running record rather
than something reconstructed at the end. Findings about the base paper and findings
about **our own** work are both recorded here; the second kind is the reason the file
is credible.

## Findings about the base framework

### N1. The sidechain objective is invariant to its own decision variable

Eq. 21, `fh = (1/NEB) * sum(dr + dw + dh + dv) * em`, contains no term that depends
on `NSC`, the split point that eqs. 20 and 23 search over. Per-block read, write,
hash and verify costs are properties of a block, not of where the chain was cut.

Measured: `fh` is **bit-identical** across `NSC ∈ {8 … 2048}` — absolute spread
`0.0`. So `∂fh/∂NSC = 0`, and the reported block-delay, energy and throughput gains
cannot be attributed to the optimisation that is claimed to produce them.

Two further defects in the same equation: its units are joule-seconds, which is
neither a delay nor an energy; and the summation bound `NED` is never defined, while
the normaliser is `1/NEB`.

Reproduce: `uv run python -m tessera.chainsim.report`

### N2. The published EHO cannot reach the optimum — structurally, not by mistuning

Eq. 20 draws every initial herd from `[LH·N/2, N/2]`. Eq. 23 replaces a herd with
`(herd + matriarch)/2`, a convex combination of two points already in the
population, and the paper defines **no mutation operator**. A convex combination
cannot leave the convex hull of the population, so the reachable set is exactly the
eq. 20 interval — for any herd count, iteration budget or seed.

On our well-posed cost model the optimum is at `S = 184`, while the reachable
interval is `[16384, 32768]` at `LH = 0.5`. Reaching `S = 184` would require
`LH ≤ 0.0056`.

Measured optimality gaps against exhaustive ground truth, identical budget:

| optimiser | best S | objective (s) | gap |
|---|---|---|---|
| exhaustive (ground truth) | 184 | 0.945294 | — |
| random search | 187 | 0.945332 | **0.004 %** |
| Optuna TPE | 185 | 0.945354 | **0.006 %** |
| EHO (NH=10, NI=6, LH=0.5) | 16680 | 1.204840 | **27.457 %** |

The herd dynamics show the mechanism: all 10 herds are reconfigured at every
iteration (because `LH < 1` puts the eq. 22 threshold below the mean), and the herd
spread halves geometrically — 7525 → 3762 → 1881 → 941 → 470 → 235 — while the
matriarch never leaves its initial neighbourhood.

**Random search beats it by four orders of magnitude of relative gap.**

## Corrections to our own work

### C1. Our first cost model was monotone — the same class of error we criticise

The first version of the segmented-log cost model charged a **linear scan** of every
archived segment per proof request. That term reached 1500 s at `S = 16` and swamped
everything, making total delay monotonically decreasing in `S`; the optimum was
degenerate at "one segment containing everything", which defeats segmenting
entirely. It was also just wrong — a real log has a segment index, so a historical
lookup is `O(log n)`, not `O(n)`.

A second version added crash-recovery cost but kept the linear scan, so it stayed
monotone: SHA-256 runs at ~0.29 µs/entry on this host, far too fast to counteract it.

This is worth recording because it is precisely the defect N1 identifies in the base
paper — an objective that does not reward the decision it is supposedly making. We
made the same mistake, twice, and caught it by checking whether the optimum was
interior rather than assuming it.

### C2. Five seeds cannot support a Wilcoxon test

The plan specified "Wilcoxon signed-rank for the headline pair, 5 seeds". The
minimum attainable two-sided p-value at n=5 is **0.0625**, so the test can never
reach p < 0.05 regardless of effect size. `eval/stats.py` refuses to emit a p-value
below n=6 and says why. The headline paired comparison must therefore run over the
**8 leave-one-replica-out folds** (minimum p = 0.0078), not across seeds.

### C3. Chronological and host-disjoint splitting cannot both hold

Hosts in an AIT testbed live for the whole 4–6 day capture, so every host in a later
partition also appears earlier. Taking a chronological cut then dropping rows whose
host appears in train removes **every** row — measured on synthetic data with 12
hosts: 794 of 794 val/test rows removed, both partitions empty.

R2 was therefore split into two regimes that are each achievable, and the impossible
combination now raises with an actionable message instead of silently returning an
empty split. Host-level disjointness is delivered by R3 (leave-one-replica-out).

### C4. Test runs were polluting the reportable results index

`test_e2e` wrote a run record into `results/index.jsonl`, and its fixture score
appeared in generated tables indistinguishably from a real result. The results tier
is now redirected to a temp directory for the whole test session, asserted by a
session fixture. A related bug: table generation pooled runs from *different
datasets* into one `mean ± std` cell, reporting a between-corpus difference as seed
variance. Both are now refused rather than patched.

### C5. "Eight independent testbeds" was wrong

An early draft described the eight AIT testbeds as independent organisations. Zenodo
states the attack *parameters and execution orders* vary while the environment and
attack repertoire are shared. R3 is therefore named **leave-one-replica-out** and
measures robustness to parameter/order randomisation — a real result, but not
cross-organisation transfer. R4 (cross-corpus) consequently became non-cuttable.

## Environment findings

### E1. LightGBM and torch deadlock together on macOS/arm64 unless torch is single-threaded

Two OpenMP runtimes in one process. Both directions are fatal and both were measured:
`import torch` then `lgb.fit` → SIGSEGV; `lgb.fit` then a CPU `Conv1d` → **deadlock**.

The fix is two-part: import LightGBM before torch, and pin torch to one intra-op
thread. Thread counts of 2, 4, 8, 10 and 14 were each measured and each deadlocked.

Measured and **rejected**: `KMP_DUPLICATE_LIB_OK=TRUE`, the most widely cited
workaround, still deadlocks; setting LightGBM's own `num_threads=1` still segfaults.

Consequence: CPU torch is single-threaded, so parallelism comes from running
configurations as separate processes (`--multirun`), not threads. Training is on MPS,
where the bound is irrelevant.
