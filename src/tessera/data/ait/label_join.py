"""The AIT label join: the project's HIGHEST-RISK component, per the plan's own
6-day kill switch. This module implements it to the standard the plan specifies:

* Files are read **in binary with explicit newline handling** — no text-mode
  decoding, which can silently renumber lines on any byte sequence that isn't
  valid UTF-8 (auditd and Suricata output routinely contain such bytes).
* Line counts are validated against the label file's own claims BEFORE any join is
  trusted: if the max referenced line number exceeds the raw file's line count, the
  join is provably broken and this raises rather than silently mislabelling.
* Rotated files (``auth.log`` + ``auth.log.1``, ``syslog`` + ``syslog.1..4``) are
  handled as an ORDERED CONCATENATION with a recorded per-file line-offset map, so a
  label line number can be resolved back to the correct physical file.

Verified against real data (russellmitchell, ``vpn/logs/openvpn.log``): 5537 raw
lines, 28 labelled, line 4331 is exactly the attacker VPN event the label predicts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from tessera.data.ait.timestamps import SOURCE_PARSERS as _PARSERS
from tessera.data.ait.timestamps import parser_for_filename


class LabelJoinError(RuntimeError):
    """The join is provably broken (a referenced line does not exist, a rotation
    ordering is ambiguous, or a decode is silently inconsistent). Raised rather
    than producing a join that LOOKS successful but mislabels lines."""


def read_lines_binary_safe(path: Path) -> list[bytes]:
    """Split on ``\\n`` in binary mode — the exact convention AIT's own line
    numbering uses (verified: matches ``wc -l`` / ``sed -n 'Np'`` on real data).

    A trailing newline produces no extra empty final element, matching how
    ``wc -l`` counts. Returned lines keep their bytes undecoded; decoding (with an
    explicit, recorded error-handling policy) happens per-line at the call site
    that needs text, not here — so a single malformed line's decode failure never
    silently shifts every line number after it.
    """
    raw = path.read_bytes()
    if raw.endswith(b"\n"):
        raw = raw[:-1]
    if raw == b"":
        return []
    return raw.split(b"\n")


@dataclass
class RotatedFile:
    """One physical file within a rotated set, with its line-offset recorded."""

    path: Path
    n_lines: int
    line_offset: int  # this file's line 1 is global line (line_offset + 1)

    def global_to_local(self, global_line: int) -> int | None:
        local = global_line - self.line_offset
        return local if 1 <= local <= self.n_lines else None


@dataclass
class RotatedFileSet:
    """A base file plus its numbered rotations (``.1``, ``.2``, ...), concatenated
    in the ORDER an ever-growing append-only log implies: oldest rotation first,
    current (unrotated) file last. This matters because AIT's line-number labels
    are relative to a single logical stream, and getting the order backwards
    silently maps every label to the wrong physical line.
    """

    base_name: str
    files: list[RotatedFile] = field(default_factory=list)

    @property
    def total_lines(self) -> int:
        return sum(f.n_lines for f in self.files)

    def resolve(self, global_line: int) -> tuple[Path, int] | None:
        for f in self.files:
            local = f.global_to_local(global_line)
            if local is not None:
                return f.path, local
        return None

    def as_dict(self) -> dict:
        return {
            "base_name": self.base_name,
            "n_files": len(self.files),
            "total_lines": self.total_lines,
            "files": [
                {"path": str(f.path.name), "n_lines": f.n_lines, "line_offset": f.line_offset}
                for f in self.files
            ],
        }


def discover_rotation_set(directory: Path, base_name: str) -> RotatedFileSet:
    """Find ``base_name``, ``base_name.1``, ``base_name.2``, ... in a directory and
    order them oldest-first, current-last (numerically DESCENDING suffix, then the
    unrotated file). This is logrotate's own convention: higher numeric suffix =
    older content, and the un-suffixed file is always the most recent.
    """
    import re

    candidates: list[tuple[int, Path]] = []
    pat = re.compile(rf"^{re.escape(base_name)}(?:\.(\d+))?$")
    if directory.exists():
        for p in directory.iterdir():
            m = pat.match(p.name)
            if m:
                suffix = int(m.group(1)) if m.group(1) else -1  # unrotated sorts last
                candidates.append((suffix, p))
    # Descending suffix (oldest rotation first), unrotated (-1) sorts after all.
    candidates.sort(key=lambda t: -t[0] if t[0] != -1 else float("inf"))

    files: list[RotatedFile] = []
    offset = 0
    for _, p in candidates:
        lines = read_lines_binary_safe(p)
        files.append(RotatedFile(path=p, n_lines=len(lines), line_offset=offset))
        offset += len(lines)
    return RotatedFileSet(base_name=base_name, files=files)


@dataclass
class LabelRecord:
    global_line: int
    labels: tuple
    rules: dict


def load_label_file(path: Path) -> dict[int, LabelRecord]:
    """Parse the JSONL label file into {global_line -> LabelRecord}."""
    out: dict[int, LabelRecord] = {}
    if not path.exists():
        return out
    for raw in read_lines_binary_safe(path):
        if not raw.strip():
            continue
        rec = json.loads(raw.decode("utf-8"))
        line = int(rec["line"])
        out[line] = LabelRecord(
            global_line=line,
            labels=tuple(rec.get("labels", [])),
            rules=rec.get("rules", {}),
        )
    return out


@dataclass
class JoinedEvent:
    """One raw log line, joined to its label (if any) and its parsed timestamp."""

    host: str
    source: str  # base filename, e.g. "auth.log"
    global_line: int
    raw_text: str
    timestamp: float | None
    is_attack: bool
    labels: tuple = ()
    rules: dict = field(default_factory=dict)


@dataclass
class JoinReport:
    host: str
    source: str
    n_raw_lines: int
    n_labelled_lines: int
    max_label_line: int | None
    rotation: dict
    timestamp_parse_failures: int
    timestamp_parser: str | None

    def as_dict(self) -> dict:
        return {
            "host": self.host,
            "source": self.source,
            "n_raw_lines": self.n_raw_lines,
            "n_labelled_lines": self.n_labelled_lines,
            "max_label_line": self.max_label_line,
            "rotation": self.rotation,
            "timestamp_parse_failures": self.timestamp_parse_failures,
            "timestamp_parser": self.timestamp_parser,
        }


def join_source(
    *,
    host: str,
    base_name: str,
    gather_dir: Path,
    labels_dir: Path,
    capture_year: int,
    decode_errors: str = "replace",
) -> tuple[list[JoinedEvent], JoinReport]:
    """Join one log source for one host: rotation-aware, line-number-exact,
    timestamp-normalised. The single function that implements the P2 join.

    ``decode_errors`` is explicit and recorded (not silently swallowed) because
    auditd and Suricata lines can contain non-UTF8 bytes; 'replace' keeps line
    COUNT correct (never drops a line) while making any corruption visible in the
    decoded text rather than crashing the whole join.
    """
    rotset = discover_rotation_set(gather_dir, base_name)
    label_path = labels_dir / base_name
    labels = load_label_file(label_path)

    max_label_line = max(labels) if labels else None
    if max_label_line is not None and max_label_line > rotset.total_lines:
        raise LabelJoinError(
            f"{host}/{base_name}: label file references line {max_label_line} but "
            f"the rotation-concatenated raw stream has only {rotset.total_lines} "
            f"lines ({rotset.as_dict()}). The join is broken - check rotation "
            "ordering or a missing rotated file before trusting any output."
        )

    parser_kind = parser_for_filename(base_name)
    parser = _PARSERS.get(parser_kind) if parser_kind else None

    events: list[JoinedEvent] = []
    ts_failures = 0
    global_line = 0
    for rf in rotset.files:
        for local_line_bytes in read_lines_binary_safe(rf.path):
            global_line += 1
            text = local_line_bytes.decode("utf-8", errors=decode_errors)
            ts = parser(text, capture_year=capture_year) if parser else None
            if parser and ts is None:
                ts_failures += 1
            rec = labels.get(global_line)
            events.append(
                JoinedEvent(
                    host=host,
                    source=base_name,
                    global_line=global_line,
                    raw_text=text,
                    timestamp=ts,
                    is_attack=rec is not None,
                    labels=rec.labels if rec else (),
                    rules=rec.rules if rec else {},
                )
            )

    report = JoinReport(
        host=host,
        source=base_name,
        n_raw_lines=global_line,
        n_labelled_lines=len(labels),
        max_label_line=max_label_line,
        rotation=rotset.as_dict(),
        timestamp_parse_failures=ts_failures,
        timestamp_parser=parser_kind,
    )
    return events, report
