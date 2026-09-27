# TESSERA — static demo

The whole site is static files: vanilla ES modules, no framework, no npm dependencies,
no CDN, relative URLs only, and no server logic at run time. Any static host works
(GitHub Pages is the target). Nothing a visitor does leaves their browser.

## The tabs

| Tab | What it shows | Data behind it |
| --- | --- | --- |
| **Overview** | What TESSERA is, the pipeline as an interactive diagram, headline real numbers, and a live self-check that re-runs the parity vectors in the visitor's browser. | real (`real_results.json`, `design_notes.json`) + live check |
| **Live Detector** | The real pretrained TESSERA-base model (5,005 parameters) scoring example windows in the browser. Visitors can switch telemetry sources off and tamper with the Merkle verdict ledger. Below it, **Score your own data** takes a CSV in the 42-feature layout and scores it locally. | real model, **synthetic** windows |
| **Training Lab** | The full protocol in a Web Worker: generate a corpus → split (replica / random / chronological / leave-one-replica-out) → leakage certificate → train TESSERA-base or a logistic-regression baseline → evaluate (AP first, MCC second) → commit verdicts to a ledger. | **synthetic** only (`tag-synthetic`), with real reference numbers beside it (`tag-real`) |
| **Real Results** | The recorded findings on real AIT data: LORO AP and MCC per fold (with low-support folds shown but excluded from the mean), the leakage gaps, and TESSERA-base against LightGBM. | real, each block with provenance |
| **Design Decisions** | Why TESSERA is built this way: the optimiser comparison against exhaustive ground truth, why EHO cannot reach the optimum, and the drafted objective's invariance. | `design_notes.json` (chainsim runs) |
| **About** | Method, honest limitations, a viva FAQ, and how to reproduce every number. | real + text |

A guided tour (top bar) walks through the anchors marked with `data-tour` attributes.

## Run it locally

```bash
just serve                                   # from the repo root; or:
python3 -m http.server 8743 --directory web  # then open http://localhost:8743/
```

The Lab needs module Web Workers (current Chrome, Firefox and Safari). Without them it
falls back to the main thread.

## How the numbers are verified

`just web-test` (which runs `cd web && npm test`) runs all of these, and every one must pass:

| Check | What it proves |
| --- | --- |
| `node parity.test.mjs` | `js/forward.js` reproduces the real trained PyTorch model on 40 golden vectors (`data/golden.json`, written by `tessera.models.export`). Max score delta measured at 4.4e-7, against a 1e-4 tolerance. |
| `node merkle-parity.test.mjs` | `js/merkle.js` reproduces the root and all 11 inclusion proofs of `tessera/ledger/merkle.py` (RFC 6962) byte for byte, and detects a tampered entry. |
| `tests/metrics.test.mjs` | AP, ROC-AUC, PR/ROC curves, confusion, MCC, ECE, histogram and `summarise` equal scikit-learn / `tessera.eval` to 1e-12 (`data/metrics_golden.json`). |
| `tests/trainer.test.mjs` | Loss, every gradient, grad-norm clipping, AdamW steps and the cosine LR schedule match torch autograd to a relative error of 1e-9 (`data/train_golden.json`). `predict()` also equals `forward.js`. |
| `tests/splits.test.mjs` | Every split mode partitions rows exactly, and the leakage certificate catches replica, stretch and duplicate leaks. |
| `tests/datagen.test.mjs` | The generator is deterministic per seed and its marginals and duplicate rate follow `replica_stats.json`. |
| `tests/pipeline.test.mjs` | The full pipeline delivers its events in order, keeps progress monotone, cancels cleanly, and is reproducible. |
| `tests/charts.test.mjs`, `tests/byod.test.mjs` | Chart scales, ticks and paths; the CSV parser; uploaded-row scoring equals `forward.js`. |
| `node dev/check-imports.mjs` | Every relative import resolves, every named import is really exported, every `data/*.json` URL exists, and every file passes `node --check`. |

On the Python side, `uv run pytest tests/test_web_exports.py` checks that the shipped JSON
is fresh: it must equal what the exporter computes now. It also checks the schema, the
licence scan and the file-size limit.

Use the glob form, as `npm test` does: `node --test "tests/*.test.mjs"`. Node 24 does not
accept a bare directory, so `node --test tests/` fails.

## Honesty and licence rules

- **No real AIT rows ship.** The AIT Log Data Set V2.1 is CC BY-NC-SA 4.0 (see
  `../LICENSE-DATA`). The site ships only aggregate summary statistics, and any number
  derived from real rows aggregates at least 20 rows (`MIN_SUPPORT_FOR_RATES = 20`).
  `export_web_data` runs a licence scan that refuses to write anything row-level.
- **Synthetic is labelled synthetic.** All Lab data (from `data/replica_stats.json`) and
  every example window (from per-class summary statistics) is generated, and marked
  `tag-synthetic`. Real numbers are
  marked `tag-real` and name the test or command that produced them: either recomputed
  by the exporter or transcribed from `RESULTS.md`. No number is invented.
- **Average precision is the headline metric**, with MCC second. Accuracy appears only
  de-emphasised, with a note explaining why it misleads at this prevalence.
- **Low-support folds** (fewer than 20 test attacks, e.g. shaw) are shown but left out
  of the summary mean and std.
- **Pretrained contamination.** The pretrained model was trained on every replica except
  santos. When the Lab evaluates that model, its leakage certificate fails any test set
  that uses one of those replicas.

## Regenerating the data

```bash
just web-data     # = uv run python -m tessera.demo.export_web_data
                  #   writes data/replica_stats.json, data/real_results.json, data/design_notes.json
                  #   (needs the local real-data cache, see tessera/eval/loro_real.py)
uv run python -m tessera.demo.golden_metrics   # data/metrics_golden.json (scikit-learn)
uv run python -m tessera.demo.golden_train     # data/train_golden.json (torch autograd)
```

`data/weights.json` and `data/golden.json` come from `tessera.models.export.export_all`.
`data/demo_windows.json` comes from `tessera.demo.synthetic_demo_data`. `design_notes.json`
changes a little on every export, because the chain simulator times hashing on the
exporting machine. The Design Decisions tab therefore reads its values from the JSON instead of
hard-coding them.
