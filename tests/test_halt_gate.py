"""The E1 gate must halt when a chronological split fails to show inflation."""

from __future__ import annotations

import pytest

from tessera.eval.halt_gate import HaltGateError, check_e1


def test_expected_inflation_passes():
    assert check_e1(0.86, 0.79).passed


@pytest.mark.mustfail
def test_missing_inflation_halts():
    with pytest.raises(HaltGateError, match="E1 HALT"):
        check_e1(0.86, 0.855)


def test_gate_can_report_without_raising():
    res = check_e1(0.80, 0.799, raise_on_fail=False)
    assert res.passed is False and "E1 HALT" in res.message
