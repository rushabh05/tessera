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

## P1 findings (measured against real AIT data, 2026-09-22)

### F3. Unpacking all eight AIT bundles simultaneously would exceed free disk

Measured exactly on the smallest bundle (`russellmitchell_no-pcaps.zip`, 522,084,364
bytes): unpacked size is **7,247,563,244 bytes — a 13.88× expansion ratio**, read from
the zip's central directory with no extraction needed (`unzip -l` totals; confirmed
identical via a real extraction of the `gather/`+`labels/` subset: 6.6 GB on disk).

Projecting that ratio across all eight zip sizes: **~87.3 GB unpacked**, against ~87 GB
free on this machine. Unpacking all eight at once would consume essentially all free
disk, leaving nothing for Parquet intermediates or template caches.

This is not a hypothetical risk the plan flagged defensively — it is a hard
architectural constraint, now implemented as such in `tessera/data/ait/unpack.py`:
the raw tier holds **at most one testbed's unpacked files at a time**
(`extract → caller processes → cleanup`), never all eight. The zip archives
themselves (6.3 GB total) stay resident as the canonical hashed source; their
unpacked contents never coexist.

### F4. The label format is line-number references, not log-line copies

`labels/<host>/<relpath>` files are JSONL: `{"line": N, "labels": [...], "rules":
{...}}`, one record per *labelled* line, keyed by 1-indexed line number in the
corresponding raw file at `gather/<host>/<relpath>`. Absence of a line number =
benign. Verified against `vpn/logs/openvpn.log`: raw file has 5537 lines, 28 are
labelled, line 4331 is exactly the attacker VPN event the label predicted (`TLS:
Initial packet from ... sid=62b69fbd`), and line 1 (unlabelled) is an unrelated
benign line. The join mechanic is simple and exact — no fuzzy matching needed.

### F5. Exactly 8 labelled files across 5 hosts in russellmitchell — and the
system-monitoring finding is sharper than first estimated

Complete, non-truncated inventory from the zip's central directory:

| File | Host | Size |
|---|---|---|
| `dnsmasq.log` | inet-firewall | 11.2 MB |
| `audit/audit.log` | internal_share | 376 B |
| `apache2/...access.log.2` | intranet_server | 1.6 MB |
| `apache2/...error.log.2` | intranet_server | 9.7 KB |
| `audit/audit.log` | intranet_server | 2.4 KB |
| `auth.log` | intranet_server | 2.5 KB |
| `logstash/.../2022-01-24-system.cpu.log` | monitoring | 7.9 KB |
| `openvpn.log` | vpn | 4.3 KB |

The earlier plan text said "system-monitoring logs are not among the labelled file
types" — **not quite right**. There is a sliver of coverage: one CPU-metric log, for
**one host (`intranet-server`), one day (`2022-01-24`)**, out of a 4-day capture
across roughly 7 monitored hosts (~28 possible host-days). That is **1 of ~28**, which
if anything *sharpens* rather than contradicts the confound this project flags for
C2: system-monitoring label coverage is present but vanishingly thin, and any
per-file-type coverage table (P2 exit criterion) must report this exactly, not as a
flat zero.

### F6. Four distinct timestamp conventions, confirmed and now handled

Real samples (not synthetic) from the bundle:

| Source | Format | Sample |
|---|---|---|
| `auth.log`, `dnsmasq.log` | syslog, **no year** | `Jan 23 06:25:05` |
| `apache2` access/error | CLF, year + UTC offset | `[23/Jan/2022:06:36:13 +0000]` |
| `audit.log` | Unix epoch **embedded inside** `msg=` | `audit(1642724221.475:149)` |
| `openvpn.log` | ISO-like, no timezone marker | `2022-01-21 00:09:11` |
| `eve.json` (Suricata) | RFC3339 + microseconds | (field `timestamp`) |

Implemented in `tessera/data/ait/timestamps.py`, with the year for syslog-format
lines taken explicitly from `dataset.yaml`'s `start` field rather than defaulting to
"now" — the sharpest hazard in that format. The cross-source anchor test the plan
required (`a known event visible in two sources must land in the same window`) is
implemented and passes against both a constructed pair and a real pair from the
bundle (`auth.log` / `dnsmasq.log`, both `Jan 23 06:25:05`, delta 0.0s).

### C6. `plan_all_replicas` crashed on a real in-progress download

While downloading the remaining 7 bundles in the background, a test run of
`plan_all_replicas()` hit a raw `zipfile.BadZipFile` on `harrison_no-pcaps.zip`,
which was mid-write at the time. A truncated file is exactly what a killed
download, a network drop, or a full disk produces — not a hypothetical edge case.
Fixed: `inspect_without_extracting` now raises a typed `IncompleteDownloadError`,
and `plan_all_replicas` catches it per-replica and continues the scan rather than
aborting on one bad file. Regression test constructs a truncated zip by hand and
asserts the scan still completes.

### C7. The first coverage-table implementation was itself badly diluted

Building the per-file-type coverage table (P2 exit criterion), the first version
counted every unique filename under `gather/` as a distinct "file type" — 334
"types" for russellmitchell, giving a coverage fraction of 2.4%. Two real inflation
sources, both fixed:

1. **Date-stamped per-day monitoring exports** (`2022-01-22-system.filesystem.log`,
   `2022-01-23-system.filesystem.log`, ...) were each counted as a distinct type,
   when they are the same signal on different days — the same defect class as an
   unrecognised rotation suffix. Fixed by stripping a leading `YYYY-MM-DD-` prefix
   alongside the existing trailing-rotation-suffix handling.
2. **Decoy documents** (`2010_invoices.xlsx` ... one per year, planted by the
   `dnsteal` exfiltration scenario) and the **attacker's own capture host**
   (`attacker_0/`) were counted as telemetry file types. Neither is something a
   real deployment's defender-side telemetry would collect. Excluded.

After both fixes: 334 → **69–70** file types (still far more than the plan's rough
"~20" estimate — a useful correction in itself), of which **7 are labelled (10.1%
by type, 4.3% by (host, type) pair)**. The plan's original "~8 of ~20 (~40%)"
estimate was too generous by a factor of 4; the real coverage is thinner than
assumed, which *sharpens* rather than weakens the C2 confound this table exists to
quantify.

Two granularities are now reported side by side, because they answer different
questions: file-type coverage (close to the plan's original framing) and
(host, type)-pair coverage (the resolution the C2 missing-modality experiment
actually operates at, since availability is a per-host property).

## P3 finding (window builder, real data, 2026-09-22)

### F7. Per-host attack prevalence is wildly non-uniform, not a small fraction

Joining `openvpn.log`/vpn, `auth.log`+`audit.log`/intranet_server, and
`dnsmasq.log`/inet-firewall into 60s windows: vpn is 0.33% positive,
intranet_server 2.12% — both realistic for anomaly detection. But
**inet-firewall is 93.76% positive**, because the labelled DNS-exfiltration
activity (`dnsteal`) spans **85.9 of the 96-hour capture** on that host —
essentially continuous, not a discrete event, with only the final ~10 hours
benign.

A single pooled prevalence number would hide this bimodal split entirely. The
window builder now reports prevalence **per host** (`BuildReport.per_host_prevalence`),
not only in aggregate — consistent with the harness's existing per-class support
reporting.

### C8. My own test arithmetic error, caught by running the test

Wrote `assert w_start == 60` for a window computed as `floor(120/60)*60`, which is
120, not 60. Caught immediately because the test failed rather than passed
silently — the value the test asserted was simply wrong, not the code under test.

## P3 finding: a second self-found leakage bug (2026-09-22)

### C9. Calendar-position features caused severe temporal leakage

Building M3 (identity features), a first version included absolute calendar
position derived from the window's epoch timestamp: hour-of-day, day-of-week,
is-weekend, minute-of-hour. Measured on real data (M1+M2+M3 combined, 3 hosts):

| | R0 (random) | R1 (chronological) | gap |
|---|---|---|---|
| with calendar features | 0.9998 | 0.5156 | 0.4843 |
| without | 1.0000 | 0.9813 | 0.0187 |

`hour_of_day` was the single most important feature by a wide margin (feature
importance 572, next-highest 410). Confirmed by direct ablation (identical seed,
identical split, features included vs. excluded) — not merely observed in one run.

The mechanism is structural, not a training bug: in a single 4-6 day capture,
"hour 3 on day 2022-01-24" occurs exactly once. Under a random split, that exact
calendar position of a held-out attack window is present verbatim in training
(shifted to a different, arbitrary row), so a tree model learns "attacks cluster
around hour X in this capture" rather than any transferable signal. This is the
same failure class as F1 (duplicate-row leakage) but through a different
mechanism — feature-level rather than row-level identity leakage — and it is
worse: F1's gap was 0.29 points; this one was 0.48.

Fixed by removing the four calendar features from `m3_identity.py` entirely
(kept: `host_bucket`, hashed and non-reversible, and `n_sources_active`). Locked
in two ways: a test that inspects `M3_FEATURE_NAMES` for anything
calendar-shaped, and a signature-level guard — `window_identity_vector` no
longer accepts a `window_start` parameter at all, so a future call site cannot
silently resurrect the leaky computation by passing a timestamp back in.

This is exactly the class of self-caught error `NEGATIVE_RESULTS.md` exists to
record: found by testing the project's own pipeline against real data with the
same rigor applied to auditing the base paper, not exempted from it.

## P4 finding: attack prevalence varies by orders of magnitude ACROSS replicas too (2026-09-23)

### F8. One replica (shaw) has near-zero positives for the fixed host/source subset

F7 found per-host prevalence non-uniform WITHIN one replica (0.3%–94%). Running
the full 8-replica leave-one-replica-out protocol found the same pattern ACROSS
replicas: 7 of 8 have 2,784–6,327 positive windows on the fixed 3-host/4-source
subset; `shaw` has **6**. Investigated, not a bug: shaw's capture spans 162
hours (longer than the ~96h of the other replicas measured), and all 6 positive
windows cluster within a single 44-minute episode near the end, across all three
hosts — a real consequence of the per-replica execution-time randomisation
(Zenodo: "attack parameters and their execution orders vary in each dataset"),
not a label-join or feature-extraction defect.

Extended the project's own `MIN_SUPPORT_FOR_RATES` convention (already used for
per-class reporting within one dataset) to leave-one-replica-out folds: a fold
with fewer than 20 held-out positives is reported in full in the per-fold table
but excluded from the summary mean/std, because a naive pooled average across
all 8 folds (0.907 ± 0.253) would understate how strong and consistent the other
seven folds actually are (0.996 ± 0.009 once the degenerate fold is excluded
rather than silently blended in).

## P5 finding: the model's own explanation mechanism is misleading (2026-09-23)

### F9. GMU attribution understates a modality's true importance by a wide margin

TESSERA-base's Gated Multimodal Unit outputs a per-modality gate weight,
intended as the mechanism the demo uses to explain a verdict ("this alert fired
mostly because of M2"). Trained on real data (7 replicas, tested on the 8th),
the gate's own mean attribution was **M1 14%, M2 2%, M3 83%, M4 1%** — read at
face value, this says the model barely uses M1 (log-template statistics) and
relies almost entirely on M3 (2 features: a hashed host identifier and a source
count).

Checked with two independent ablations rather than trusted:

| check | AP |
|---|---|
| GBDT on M1 alone | 0.9996 |
| GBDT on M3 alone | 0.6917 |
| GBDT on `host_bucket` alone (1 feature) | 0.4977, MCC=0 |
| TESSERA-base with M1 present | 0.9995 |
| TESSERA-base with M1 genuinely removed (zeroed + marked unavailable) | 0.9846 |

M1 alone carries nearly the whole signal for a classical model, and actually
removing it from the neural model (not down-weighting — zeroing and marking it
structurally absent, the same mechanism the availability mask uses everywhere
else) costs real performance. `host_bucket` alone is barely above the
prevalence floor with MCC=0. **The gate's stated 14%/83% split does not match
what either model's own behaviour shows.**

This is the same category of error the project has caught in itself before
(the calendar-feature leakage, the empty per-host prevalence report): a
mechanism that *looks* like it is telling you something true, checked against
independent evidence rather than trusted because it is plausible or
convenient. Consequence: gate values are documented and used as *a* signal,
never presented as ground truth for "what the model used" — in the report, the
demo, or the viva. A proper per-modality ablation, not the gate, is this
project's only reliable importance measure. Locked in by
`test_gmu_attribution_does_not_match_naive_ablation_importance`.

## Demo finding: a real cross-language hashing bug in the production ledger (2026-09-23)

### C10. `Verdict.canonical()`'s score field hashed differently in Python vs JavaScript

Building the browser demo's JS Merkle port (`web/js/merkle.js`) and generating
real cross-language parity vectors surfaced a genuine bug in the **production**
ledger code, not a demo-only issue: `Verdict.canonical()` rounded `score` to a
Python float (`round(float(self.score), 6)`), then relied on `json.dumps` to
serialise it. Python's `json.dumps(0.0)` renders `"0.0"` — preserving the
trailing zero that marks it as a float — while JavaScript has no int/float
distinction, and `JSON.stringify(0.0)` renders `"0"`. Any verdict with a score
that rounds to a whole number (0.0 — a maximally-confident benign prediction —
or 1.0, both real, expected values, not edge cases) would therefore hash to
**two different leaves** depending on which language recomputed it, silently
defeating the entire point of a ledger this project explicitly designed to be
verified across languages (Python producing it, JavaScript checking it in a
browser).

Caught by generating real Python-computed Merkle roots and reproducing them in
Node.js: the first parity run failed outright (`root mismatch`), traced to a
single-field diff (`"score":0.0` vs `"score":0`) by comparing canonical bytes
side by side before touching the tree logic at all.

**Fixed**: `score` is now formatted as a fixed-precision string
(`f"{round(float(self.score), 6):.6f}"` → `"0.000000"`), removing the ambiguity
entirely — both languages serialise the same string identically, so there is
no float-formatting convention left to diverge. Verified: 11/11 real inclusion
proofs match Python exactly after the fix, including deliberate whole-number
scores at two positions. Regression-tested in Python
(`test_verdict_canonical_cross_language.py`) and locked into the JS parity
suite (`web/merkle-parity.test.mjs`).

This is the same category of error the project has caught in itself
repeatedly (calendar-feature leakage, the empty per-host prevalence report,
the misleading GMU attribution): a mechanism assumed correct because it
"obviously" round-trips through JSON, checked against independent evidence —
in this case, an actual second-language implementation — rather than trusted.
