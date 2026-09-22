# TESSERA task runner.  `just --list` to see everything.
#
# Split by cost on purpose: `check` is what CI runs on every commit and finishes in
# minutes; `grid` is hundreds of runs and is invoked by hand. A CI step that re-runs
# the ablation grid is not a verification, it is a fantasy.

set shell := ["bash", "-uc"]

default:
    @just --list

# ---------------------------------------------------------------- fast (CI)

# Everything CI runs: lint, the full test suite, tables, and the numbers check.
check: lint test tables verify-numbers

lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

test:
    uv run pytest tests -q

# Guards that must FAIL the build when violated, run in isolation so a pass is visible.
test-guards:
    uv run pytest tests -q -m mustfail -v

# End-to-end: ingest -> features -> detect -> explain -> ledger -> proof -> optimise.
e2e:
    uv run pytest tests/test_e2e.py -q -v

# ---------------------------------------------------------------- data

# Verify dataset integrity against the SHA-256 manifest.
data-verify:
    uv run python -c "from tessera.data.manifest import verify; \
      [print(f'{r.status:>14}  {r.key}  {r.detail}') for r in verify()]"

# Free disk on the volume holding data/.
disk:
    uv run python -c "from tessera.data.manifest import free_disk_bytes; \
      print('free: %.1f GiB' % (free_disk_bytes()/2**30))"

# ---------------------------------------------------------------- experiments

# One run. e.g. `just train model=tessera_base split=r2_grouped seed=0`
train *ARGS:
    uv run python -m tessera.train.run {{ARGS}}

# The baseline ladder across the split-regime inflation cascade.
baselines:
    uv run python -m tessera.train.run --multirun \
      dataset=synthetic \
      model=random,majority,nullmask,lightgbm \
      split=r0_random,r1_chrono,r2_grouped \
      seed=0,1,2,3,4

# Leave-one-replica-out: the headline generalisation protocol.
loro:
    uv run python -m tessera.train.run --multirun \
      dataset=synthetic model=lightgbm split=r3_loro seed=0,1,2,3,4

# The full ablation grid. HOURS. Never run by CI.
grid: baselines loro
    @echo "grid complete; run `just tables` to regenerate reports"

# ---------------------------------------------------------------- chain / ledger

# Optimiser comparison against exhaustive ground truth, plus the optimality gap.
chainsim:
    uv run python -m tessera.chainsim.report

# ---------------------------------------------------------------- reporting

# Regenerate every table and figure from results/. The ONLY way numbers are produced.
tables:
    uv run python -m tessera.report.tables

# Fail if any document contains a metric no recorded run produced.
verify-numbers:
    uv run python -m tessera.report.check_no_hardcoded_numbers

# ---------------------------------------------------------------- housekeeping

fmt:
    uv run ruff format src tests
    uv run ruff check --fix src tests

# Remove generated reports and recorded runs. Does NOT touch data/.
clean-results:
    rm -rf results/runs results/index.jsonl paper/tables/* paper/figures/*

env:
    uv run python -c "from tessera.env import env_json; print(env_json())"
