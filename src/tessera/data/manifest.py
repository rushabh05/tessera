"""Dataset integrity manifest: URL, SHA-256, size, retrieval date and licence per file.

Two rules this module exists to enforce:

**Verify before use.** Every pipeline entry point checks recorded hashes, so a
truncated download or a silently re-released dataset surfaces immediately rather
than as an unexplained metric change weeks later.

**Substitution-refusal.** If a corrected dataset variant cannot be obtained, the
manifest records it as ``unobtainable``; the uncorrected original is never quietly
substituted. Using CIC-IDS2017 where CIC-IDS2017-corrected was promised would
invalidate the comparison this project is built on, so it is made impossible by
construction rather than by discipline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from tessera.paths import RAW_DIR

MANIFEST_PATH = Path(__file__).parent / "manifest.yaml"
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Entry:
    """One expected file."""

    key: str
    dataset: str
    filename: str
    url: str
    licence: str
    sha256: str | None = None  # None until first retrieval pins it
    size_bytes: int | None = None
    retrieved_utc: str | None = None
    unobtainable: bool = False
    unobtainable_reason: str | None = None
    notes: str | None = None

    @property
    def path(self) -> Path:
        return RAW_DIR / self.dataset / self.filename


class ManifestError(RuntimeError):
    pass


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Entry]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    entries = {}
    for key, body in (raw.get("files") or {}).items():
        entries[key] = Entry(key=key, **body)
    return entries


def save_manifest(entries: dict[str, Entry], path: Path = MANIFEST_PATH) -> None:
    body = {
        "_comment": (
            "Generated and updated by tessera.data.manifest. Hashes are pinned on "
            "first successful retrieval. An entry marked unobtainable is NEVER "
            "silently replaced by an uncorrected original."
        ),
        "files": {
            key: {k: v for k, v in asdict(e).items() if k != "key"}
            for key, e in sorted(entries.items())
        },
    }
    path.write_text(yaml.safe_dump(body, sort_keys=False, width=100))


@dataclass
class VerifyResult:
    key: str
    status: str  # "ok" | "missing" | "hash_mismatch" | "unpinned" | "unobtainable"
    detail: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.status in {"missing", "hash_mismatch"}


def verify(
    entries: dict[str, Entry] | None = None, *, require: list[str] | None = None
) -> list[VerifyResult]:
    """Check every manifest entry against the bytes on disk.

    ``require`` names keys that MUST be present and matching; anything else is
    reported but not blocking, so partial local datasets do not stop development.
    """
    entries = load_manifest() if entries is None else entries
    require = require or []
    results: list[VerifyResult] = []

    for key, e in sorted(entries.items()):
        if e.unobtainable:
            results.append(
                VerifyResult(
                    key, "unobtainable", e.unobtainable_reason or "recorded as unobtainable"
                )
            )
            continue
        if not e.path.exists():
            results.append(VerifyResult(key, "missing", f"absent: {e.path}"))
            continue
        if e.sha256 is None:
            results.append(VerifyResult(key, "unpinned", "present but hash not yet pinned"))
            continue
        actual = sha256_file(e.path)
        if actual != e.sha256:
            results.append(
                VerifyResult(key, "hash_mismatch", f"expected {e.sha256[:16]}, got {actual[:16]}")
            )
        else:
            results.append(
                VerifyResult(key, "ok", f"{e.size_bytes or e.path.stat().st_size} bytes")
            )

    blocking = [r for r in results if r.is_blocking and r.key in require]
    if blocking:
        raise ManifestError(
            "required dataset files failed verification:\n"
            + "\n".join(f"  {r.key}: {r.status} - {r.detail}" for r in blocking)
        )
    return results


def free_disk_bytes(path: Path = RAW_DIR) -> int:
    """Free bytes on the volume holding ``path``.

    The AIT no-pcaps bundles are 6.4 GB zipped and their unpacked footprint is not
    published anywhere, so a disk precondition is asserted before unpacking rather
    than discovered when the volume fills mid-parse.
    """
    import shutil

    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def require_free_disk(need_bytes: int, path: Path = RAW_DIR) -> None:
    free = free_disk_bytes(path)
    if free < need_bytes:
        raise ManifestError(
            f"insufficient free disk: need {need_bytes / 2**30:.1f} GiB, "
            f"have {free / 2**30:.1f} GiB on the volume holding {path}"
        )


def manifest_summary_json() -> str:
    return json.dumps(
        [
            asdict(r) if hasattr(r, "__dataclass_fields__") else r
            for r in [{"key": r.key, "status": r.status, "detail": r.detail} for r in verify()]
        ],
        indent=2,
    )
