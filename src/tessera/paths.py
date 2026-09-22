"""Canonical filesystem layout. Every module resolves paths through here.

Keeping this in one place means the repo can move (the project directory name
contains spaces, which breaks naive path handling) without edits elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/tessera/paths.py -> src/tessera -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

CONF_DIR = REPO_ROOT / "conf"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
# TESSERA_RESULTS_DIR redirects the results tier. The test suite sets it to a tmp
# directory so a test run can never contribute a row to the reportable index -
# without it, a fixture result and a real result are indistinguishable in a table.
RESULTS_DIR = Path(os.environ.get("TESSERA_RESULTS_DIR") or (REPO_ROOT / "results"))
RUNS_DIR = RESULTS_DIR / "runs"
PAPER_DIR = REPO_ROOT / "paper"
TABLES_DIR = PAPER_DIR / "tables"
FIGURES_DIR = PAPER_DIR / "figures"
TESTS_DIR = REPO_ROOT / "tests"

# Secrets live OUTSIDE the repo. tests/test_no_secrets_tracked.py enforces this.
SECRETS_DIR = REPO_ROOT.parent / ".tessera-secrets"


def ensure_dirs() -> None:
    """Create the directories that are gitignored and therefore absent on clone."""
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, RUNS_DIR, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
