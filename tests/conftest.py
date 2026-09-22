"""Test isolation: a test run must never contribute a reportable number.

Without this, ``test_e2e`` writes a run record into ``results/index.jsonl`` and its
fixture score appears in generated tables indistinguishably from a real result -
exactly the fixture-vs-real conflation this project exists to prevent. The results
tier is redirected to a tmp directory for the whole session, before any tessera
module caches a path.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = tempfile.mkdtemp(prefix="tessera-test-results-")
os.environ["TESSERA_RESULTS_DIR"] = _TMP


@pytest.fixture(scope="session", autouse=True)
def isolated_results_tier():
    """Assert the redirect took effect, then hand control to the tests."""
    from tessera import paths

    assert str(paths.RESULTS_DIR) == _TMP, (
        f"results tier not isolated: {paths.RESULTS_DIR} != {_TMP}. A test run would "
        "write into the reportable results index."
    )
    Path(_TMP).mkdir(parents=True, exist_ok=True)
    yield Path(_TMP)


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Per-test results tier, for tests that assert on index contents."""
    from tessera import paths

    d = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_DIR", d, raising=True)
    monkeypatch.setattr(paths, "RUNS_DIR", d / "runs", raising=True)
    return d
