"""Fail the build if a document contains a metric that no recorded run produced.

The base paper's Section IV and Conclusion disagree on three numbers - the signature
of hand-transcribed results. This makes that failure mode impossible: every
metric-looking numeral in a generated document must trace back to
``results/index.jsonl``.

Only numerals with two or more decimal places are checked. Structural numbers -
years, RFC numbers, window sizes, section references - do not look like metrics and
are ignored, which keeps the check precise enough to stay switched on.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from tessera.paths import TABLES_DIR
from tessera.results import load_index

# A metric: optional sign, digits, dot, >= 2 decimals.
METRIC_RE = re.compile(r"(?<![\w.])(-?\d+\.\d{2,})(?![\w])")
ALLOW_RE = re.compile(r"<!--\s*allow-number:\s*([^>]+?)\s*-->")


def _walk(obj, out: set[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def recorded_numbers() -> set[float]:
    vals: set[float] = set()
    for row in load_index():
        _walk(row, vals)
    return vals


def _renderings(values: set[float]) -> set[str]:
    """Every string form a generator might legitimately print."""
    out: set[str] = set()
    for v in values:
        for d in (2, 3, 4, 5, 6):
            out.add(f"{v:.{d}f}")
            out.add(f"{abs(v):.{d}f}")
        out.add(repr(v))
        out.add(str(v))
    return out


def check_file(path: Path, allowed: set[str]) -> list[str]:
    text = path.read_text()
    explicit = {tok.strip() for m in ALLOW_RE.findall(text) for tok in m.split(",")}
    violations = []
    for match in METRIC_RE.finditer(text):
        tok = match.group(1)
        if tok in allowed or tok.lstrip("-") in allowed or tok in explicit:
            continue
        line = text.count("\n", 0, match.start()) + 1
        violations.append(f"{path.name}:{line}: '{tok}' does not trace to any recorded run")
    return violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="documents to check (default: generated tables)")
    args = ap.parse_args(argv)

    targets = [Path(p) for p in args.paths] or sorted(TABLES_DIR.glob("*.md"))
    targets = [p for p in targets if p.exists()]
    if not targets:
        print("no documents to check")
        return 0

    allowed = _renderings(recorded_numbers())
    print(f"checking {len(targets)} document(s) against {len(allowed)} recorded renderings")

    all_v: list[str] = []
    for p in targets:
        v = check_file(p, allowed)
        status = "OK" if not v else f"{len(v)} VIOLATION(S)"
        print(f"  {p.name}: {status}")
        all_v.extend(v)

    if all_v:
        print("\nHand-typed numbers detected:", file=sys.stderr)
        for v in all_v:
            print(f"  {v}", file=sys.stderr)
        print(
            "\nEvery reported metric must come from results/. Regenerate with "
            "`just tables`, or annotate a genuinely non-metric value with "
            "<!-- allow-number: X -->.",
            file=sys.stderr,
        )
        return 1
    print("\nall numbers trace to recorded runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
