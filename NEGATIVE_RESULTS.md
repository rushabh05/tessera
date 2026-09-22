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
