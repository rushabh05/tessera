"""The E1 halt gate: a numeric stop condition enforced in the runner.

If a chronological split (R1) does not score materially below a random split (R0),
one of two things is true: the chronological split is not actually chronological, or
the random split is not actually leaking. Either way the split logic is wrong, and
every downstream number is untrustworthy.

The base paper reports only a random split, so this failure mode is invisible in it.
Here the gate halts the pipeline rather than letting the grid run for three weeks on
a broken split.
"""

from __future__ import annotations

from dataclasses import dataclass

# R1 average precision must fall at least this far below R0 for the split to be
# behaving as expected. Set before any results exist so it cannot be tuned to pass.
DEFAULT_MIN_DROP_AP = 0.02


class HaltGateError(RuntimeError):
    pass


@dataclass
class GateResult:
    passed: bool
    r0_ap: float
    r1_ap: float
    observed_drop: float
    required_drop: float
    message: str

    def as_dict(self) -> dict:
        return {
            "gate": "E1 random-vs-chronological inflation",
            "passed": self.passed,
            "r0_average_precision": self.r0_ap,
            "r1_average_precision": self.r1_ap,
            "observed_drop": self.observed_drop,
            "required_drop": self.required_drop,
            "message": self.message,
        }


def check_e1(
    r0_ap: float,
    r1_ap: float,
    *,
    required_drop: float = DEFAULT_MIN_DROP_AP,
    raise_on_fail: bool = True,
) -> GateResult:
    """Assert R0 -> R1 shows the expected inflation drop."""
    drop = float(r0_ap) - float(r1_ap)
    passed = drop >= required_drop
    if passed:
        msg = (
            f"R1 average precision is {drop:.4f} below R0 (>= {required_drop:.4f} required); "
            "the chronological split is behaving as expected"
        )
    else:
        msg = (
            f"E1 HALT: chronological R1 ({r1_ap:.4f}) did not fall at least "
            f"{required_drop:.4f} below random R0 ({r0_ap:.4f}); observed drop "
            f"{drop:.4f}. Either the chronological split is not ordering by time, or "
            "the random split is not leaking as expected. Debug the split before "
            "running anything else."
        )
    result = GateResult(passed, float(r0_ap), float(r1_ap), drop, float(required_drop), msg)
    if not passed and raise_on_fail:
        raise HaltGateError(msg)
    return result
