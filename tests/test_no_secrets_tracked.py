"""No signing key or salt may ever be tracked by git.

A pseudonymisation salt committed to the repository voids the pseudonymisation
completely: anyone with the repo can re-derive every host and user hash. This is a
build-failing check rather than a note in a README.
"""

from __future__ import annotations

import subprocess

from tessera.paths import REPO_ROOT

FORBIDDEN_NAMES = ("ledger_ed25519.key", "pseudonymisation_salt.bin")
FORBIDDEN_SUFFIXES = (".key", ".pem")
FORBIDDEN_DIRS = ("secrets/", ".tessera-secrets/")


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]


def test_no_secret_material_is_tracked():
    tracked = _tracked_files()
    bad = [
        f
        for f in tracked
        if f.endswith(FORBIDDEN_NAMES)
        or f.endswith(FORBIDDEN_SUFFIXES)
        or any(d in f for d in FORBIDDEN_DIRS)
    ]
    assert not bad, f"secret material is tracked by git: {bad}"


def test_secrets_directory_is_outside_the_repo():
    from tessera.paths import SECRETS_DIR

    assert REPO_ROOT not in SECRETS_DIR.parents and SECRETS_DIR != REPO_ROOT, (
        f"secrets dir {SECRETS_DIR} must live outside the repository"
    )


def test_gitignore_covers_secret_patterns():
    gi = (REPO_ROOT / ".gitignore").read_text()
    for pattern in ("*.key", "secrets/", ".env"):
        assert pattern in gi, f".gitignore is missing '{pattern}'"
