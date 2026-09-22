# T2 - Split-regime inflation cascade (dataset: fixture)

**Metric:** `binary.average_precision` — mean ± std across seeds. Columns increase in split strictness left to right; a value that falls is the leakage the looser split was hiding.

| model | r2_grouped | seeds |
|---|---|---|
| `random` | 0.0903 | 1 |

> **SYNTHETIC DATA.** Validates the harness against known ground truth. Not a detection result.

> **Shortcut floor** (availability mask alone): AP = 0.2149–0.2149. A model at or below this range has learned which telemetry sources were switched on, not what an attack looks like.
