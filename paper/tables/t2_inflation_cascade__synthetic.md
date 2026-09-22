# T2 - Split-regime inflation cascade (dataset: synthetic)

**Metric:** `binary.average_precision` — mean ± std across seeds. Columns increase in split strictness left to right; a value that falls is the leakage the looser split was hiding.

| model | r0_random | r1_chrono | r2_grouped | seeds |
|---|---|---|---|---|
| `majority` | 0.0542 | 0.0501 | 0.0501 | 1 |
| `nullmask` | 0.1107 | 0.1064 | 0.1064 | 1 |
| `random` | 0.0544 | 0.0528 | 0.0528 | 1 |
| `lightgbm` | 0.7930 | 0.7488 | 0.7488 | 1 |

> **SYNTHETIC DATA.** Validates the harness against known ground truth. Not a detection result.

> **Shortcut floor** (availability mask alone): AP = 0.1099–0.1099. A model at or below this range has learned which telemetry sources were switched on, not what an attack looks like.
