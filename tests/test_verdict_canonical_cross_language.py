"""A real bug, caught while building the browser demo's JS Merkle port: a
whole-number score serialises to different JSON text in Python vs JavaScript,
so the SAME verdict would hash to two DIFFERENT leaves depending on which
language computed it - defeating the whole point of a cross-language-verifiable
ledger. See Verdict.canonical()'s docstring for the full story.
"""

from __future__ import annotations

import json

from tessera.data.contract import Verdict


def _make(score: float) -> Verdict:
    return Verdict(
        window_id="w0",
        host_hash="hh0",
        ts_bucket=1000,
        verdict=0,
        score=score,
        model_git_sha="abc123",
    )


def test_score_is_a_string_not_a_bare_number():
    """The actual fix: a bare JSON number's text representation is NOT
    guaranteed identical across languages for whole-number floats. A fixed-
    width string is, because both languages serialise the same string
    identically - there is no float-formatting convention left to diverge."""
    payload = _make(0.0).canonical()
    assert isinstance(payload["score"], str), (
        "score must be a string; a bare float here reintroduces the exact "
        "cross-language hash mismatch this test guards against"
    )


def test_whole_number_score_does_not_render_ambiguously():
    """The specific case that broke: score=0.0 (a maximally-confident benign
    prediction, a real and expected value, not an edge case to dismiss)."""
    payload = _make(0.0).canonical()
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert '"score":"0.000000"' in text
    # The old, broken form - a bare, unquoted zero - must NOT appear.
    assert '"score":0.0' not in text
    assert '"score":0,' not in text and not text.rstrip("}").endswith('"score":0')


def test_fractional_score_still_rounds_to_six_places():
    payload = _make(0.123456789).canonical()
    assert payload["score"] == "0.123457"


def test_score_of_exactly_one_also_safe():
    """The other whole-number case: score=1.0 (maximally-confident attack)."""
    payload = _make(1.0).canonical()
    assert payload["score"] == "1.000000"
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    assert '"score":1.0' not in text
    assert '"score":1,' not in text


def test_canonical_payload_still_has_no_identifying_fields():
    """Unaffected by the score-format fix, but worth re-asserting alongside
    it: the forbidden-field guard from test_e2e.py, isolated here."""
    payload = _make(0.5).canonical()
    for forbidden in ("ip", "url", "username", "user", "geolocation", "host_id"):
        assert forbidden not in payload
