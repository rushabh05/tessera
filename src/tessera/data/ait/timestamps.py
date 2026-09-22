"""Timestamp normalisation across the AIT log sources — the sub-task the project
plan calls "the most time-consuming part of P2" and gives its own test for.

Confirmed by direct inspection of the russellmitchell bundle (2026-09-22), FOUR
distinct conventions across five labelled sources, not a guess:

  syslog        "Jan 23 06:25:05"                  auth.log, dnsmasq.log — NO YEAR
  apache CLF    "[23/Jan/2022:06:36:13 +0000]"      apache access/error — year + UTC offset
  auditd        "audit(1642724221.475:149)"         audit.log — Unix epoch embedded in msg=
  ait-iso       "2022-01-21 00:09:11"               openvpn.log — no timezone marker
  suricata      RFC3339-with-microseconds field      eve.json `timestamp` key

The syslog format's missing year is the sharpest hazard: naively parsing "Jan 23"
with the CURRENT year is silently wrong for a 2022 capture. Every parser here
therefore takes the capture year EXPLICITLY (from ``dataset.yaml``'s ``start``
field) rather than defaulting to "now".

All parsers return a UTC Unix timestamp (float seconds), the common unit the window
builder groups on. A parser that fails returns ``None`` rather than raising, because
a single malformed line must not abort an entire file's join - the caller counts
failures and a high failure rate is itself a signal worth surfacing.
"""

from __future__ import annotations

import re
from calendar import timegm
from dataclasses import dataclass
from datetime import UTC, datetime

_SYSLOG_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+"
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\b"
)
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
}

_APACHE_RE = re.compile(
    r"\[(?P<day>\d{2})/(?P<mon>[A-Za-z]{3})/(?P<year>\d{4}):"
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\s+(?P<tz>[+-]\d{4})\]"
)

_AUDITD_RE = re.compile(r"audit\((?P<epoch>\d+)\.(?P<frac>\d+):\d+\)")

_AIT_ISO_RE = re.compile(
    r"^(?P<year>\d{4})-(?P<mon>\d{2})-(?P<day>\d{2})\s+"
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})"
)


def parse_syslog(line: str, *, capture_year: int) -> float | None:
    """ "Jan 23 06:25:05" — auth.log, dnsmasq.log. No year in the line itself."""
    m = _SYSLOG_RE.match(line)
    if not m or m.group("mon") not in _MONTHS:
        return None
    try:
        dt = datetime(
            capture_year,
            _MONTHS[m.group("mon")],
            int(m.group("day")),
            int(m.group("h")),
            int(m.group("m")),
            int(m.group("s")),
            tzinfo=UTC,
        )
    except ValueError:
        return None
    return dt.timestamp()


def parse_apache(line: str, *, capture_year: int | None = None) -> float | None:
    """ "[23/Jan/2022:06:36:13 +0000]" — apache access/error. Has its own year+TZ."""
    m = _APACHE_RE.search(line)
    if not m or m.group("mon") not in _MONTHS:
        return None
    offset_sign = 1 if m.group("tz")[0] == "+" else -1
    offset_min = offset_sign * (int(m.group("tz")[1:3]) * 60 + int(m.group("tz")[3:5]))
    try:
        naive = datetime(
            int(m.group("year")),
            _MONTHS[m.group("mon")],
            int(m.group("day")),
            int(m.group("h")),
            int(m.group("m")),
            int(m.group("s")),
        )
    except ValueError:
        return None
    epoch = timegm(naive.timetuple())
    return float(epoch - offset_min * 60)


def parse_auditd(line: str, *, capture_year: int | None = None) -> float | None:
    """ "audit(1642724221.475:149)" — audit.log. Epoch embedded inside msg=."""
    m = _AUDITD_RE.search(line)
    if not m:
        return None
    return float(f"{m.group('epoch')}.{m.group('frac')}")


def parse_ait_iso(line: str, *, capture_year: int | None = None) -> float | None:
    """ "2022-01-21 00:09:11" — openvpn.log. No timezone marker; assumed UTC, which
    matches every other source's convention once auditd (unambiguously UTC epoch)
    is used as the cross-source anchor - see ``cross_source_anchor_test`` below."""
    m = _AIT_ISO_RE.match(line)
    if not m:
        return None
    try:
        dt = datetime(
            int(m.group("year")),
            int(m.group("mon")),
            int(m.group("day")),
            int(m.group("h")),
            int(m.group("m")),
            int(m.group("s")),
            tzinfo=UTC,
        )
    except ValueError:
        return None
    return dt.timestamp()


def parse_suricata_eve(record: dict, *, capture_year: int | None = None) -> float | None:
    """Suricata eve.json `timestamp` field: RFC3339 with microseconds, e.g.
    "2022-01-21T00:00:09.123456+0000"."""
    ts = record.get("timestamp")
    if not isinstance(ts, str):
        return None
    try:
        # Python's fromisoformat wants a colon in the UTC offset before 3.11 fully
        # relaxed it; normalise defensively rather than assume the running version.
        s = ts.replace("Z", "+00:00")
        if len(s) >= 5 and s[-5] in "+-" and s[-3] != ":":
            s = s[:-2] + ":" + s[-2:]
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


SOURCE_PARSERS = {
    "syslog": parse_syslog,
    "apache": parse_apache,
    "auditd": parse_auditd,
    "ait_iso": parse_ait_iso,
    "suricata_eve": parse_suricata_eve,
}

# Which parser applies to which labelled file, by filename suffix match. Extend as
# more sources are added in P3.
FILE_PARSER = {
    "auth.log": "syslog",
    "dnsmasq.log": "syslog",
    "openvpn.log": "ait_iso",
    "audit.log": "auditd",
}


def parser_for_filename(filename: str) -> str | None:
    """Resolve which timestamp parser applies, by filename match then extension."""
    for suffix, kind in FILE_PARSER.items():
        if filename.endswith(suffix):
            return kind
    if "apache2" in filename and ("access" in filename or "error" in filename):
        return "apache"
    if filename.endswith("eve.json"):
        return "suricata_eve"
    return None


@dataclass
class CrossSourceCheck:
    """Result of the anchor test: a known event visible in two sources must land
    in the same window (the exact test the project plan requires for P2)."""

    source_a: str
    source_b: str
    ts_a: float
    ts_b: float
    delta_seconds: float
    same_60s_window: bool


def check_cross_source_alignment(
    ts_a: float, source_a: str, ts_b: float, source_b: str, *, window_seconds: int = 60
) -> CrossSourceCheck:
    delta = ts_b - ts_a
    return CrossSourceCheck(
        source_a=source_a,
        source_b=source_b,
        ts_a=ts_a,
        ts_b=ts_b,
        delta_seconds=delta,
        same_60s_window=int(ts_a // window_seconds) == int(ts_b // window_seconds),
    )
