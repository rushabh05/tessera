# TESSERA — static demo

Zero-dependency, zero-cost, always-on. Vanilla ES modules, no framework, no
CDN, no server required at runtime (`python3 -m http.server` here is only a
local dev convenience — any static host works, e.g. GitHub Pages).

## What's real here

- **`js/forward.js`** is a from-scratch JavaScript reimplementation of the
  trained TESSERA-base model (`tessera/models/tessera_base.py`), operating on
  the same 42-dim feature vectors and the exact trained weights
  (`data/weights.json`, exported by `tessera/models/export.py`).
- **`js/merkle.js`** is a from-scratch JS port of the RFC 6962 Merkle log
  (`tessera/ledger/merkle.py`) — the same construction Certificate
  Transparency uses.
- Both are proven correct against the real Python implementations, not just
  assumed to match:

```bash
npm test
# or individually:
node parity.test.mjs          # JS model vs. real trained PyTorch model
node merkle-parity.test.mjs   # JS ledger vs. real Python ledger + a live tamper check
```

## Demo data

`data/demo_windows.json` is **synthetic** — generated from summary statistics
(mean/std per feature, per class) computed over real processed AIT data, never
from the real feature rows themselves. This keeps the demo fully clear of the
AIT Log Data Set's CC BY-NC-SA licence (NonCommercial + ShareAlike), which the
project's `LICENSE-DATA.md` commits to never violating by redistributing
derived features. See `tessera/demo/synthetic_demo_data.py` for the generator
and the two real bugs found and fixed while building it (documented in
`NEGATIVE_RESULTS.md`).

Real, validated accuracy numbers — measured only on real held-out AIT data,
never on this synthetic demo set — are in the project root's `RESULTS.md`.

## Regenerating the exported data

```bash
# from the tessera/ project root, with the processed-data cache already built
# (see tessera/eval/loro_real.py):
uv run python -m tessera.models.export   # (or the inline export shown in NEGATIVE_RESULTS.md)
```

## Serving locally

```bash
npm run serve
# then open http://localhost:8743
```
