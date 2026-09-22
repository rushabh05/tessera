# t2b_mcc - `binary.mcc` (dataset: synthetic)

**Metric:** `binary.mcc` — mean ± std across seeds. Columns increase in split strictness left to right; a value that falls is the leakage the looser split was hiding.

| model | r0_random | r1_chrono | r2_grouped | seeds |
|---|---|---|---|---|
| `majority` | 0.0000 | 0.0000 | 0.0000 | 1 |
| `nullmask` | 0.0000 | 0.0000 | 0.0000 | 1 |
| `random` | -0.0011 | 0.0105 | 0.0105 | 1 |
| `lightgbm` | 0.6814 | 0.6720 | 0.6720 | 1 |

> **SYNTHETIC DATA.** Validates the harness against known ground truth. Not a detection result.

> **Shortcut floor** (availability mask alone): AP = 0.1099–0.1099. A model at or below this range has learned which telemetry sources were switched on, not what an attack looks like.
