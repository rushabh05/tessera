"""Signed Tree Heads, and the key handling that keeps signing material out of git.

A tree head is only useful if an auditor can tell it came from the log operator, so
heads are Ed25519-signed. The private key and the pseudonymisation salt live OUTSIDE
the repository, are generated per deployment, and are excluded from the dataset
manifest. ``tests/test_no_secrets_tracked.py`` fails the build if either appears in
a tracked file - a salt committed to the repo would void the pseudonymisation
entirely, so it is enforced rather than documented.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from tessera.paths import SECRETS_DIR

KEY_NAME = "ledger_ed25519.key"
PUB_NAME = "ledger_ed25519.pub"
SALT_NAME = "pseudonymisation_salt.bin"


def _ensure_secrets_dir() -> Path:
    SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(SECRETS_DIR, stat.S_IRWXU)  # owner-only
    return SECRETS_DIR


def load_or_create_key() -> Ed25519PrivateKey:
    """Load the signing key, generating it on first use. Never inside the repo."""
    d = _ensure_secrets_dir()
    path = d / KEY_NAME
    if path.exists():
        return serialization.load_pem_private_key(path.read_bytes(), password=None)

    key = Ed25519PrivateKey.generate()
    path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    (d / PUB_NAME).write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return key


def load_or_create_salt(n_bytes: int = 32) -> bytes:
    """The salt for host/user pseudonymisation. Per-deployment, never tracked."""
    d = _ensure_secrets_dir()
    path = d / SALT_NAME
    if path.exists():
        return path.read_bytes()
    salt = os.urandom(n_bytes)
    path.write_bytes(salt)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return salt


@dataclass
class SignedTreeHead:
    """A commitment to the log's state at one size."""

    tree_size: int
    root_hex: str
    timestamp: int  # caller-supplied, so runs stay reproducible
    signature_hex: str
    log_id: str = "tessera-verdict-log"

    def payload(self) -> bytes:
        return json.dumps(
            {
                "log_id": self.log_id,
                "tree_size": self.tree_size,
                "root": self.root_hex,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def verify(self, public_key: Ed25519PublicKey) -> bool:
        try:
            public_key.verify(bytes.fromhex(self.signature_hex), self.payload())
            return True
        except (InvalidSignature, ValueError):
            return False

    def as_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "tree_size": self.tree_size,
            "root": self.root_hex,
            "timestamp": self.timestamp,
            "signature": self.signature_hex,
        }


def sign_tree_head(
    root: bytes, tree_size: int, *, timestamp: int, key: Ed25519PrivateKey | None = None
) -> SignedTreeHead:
    key = key or load_or_create_key()
    sth = SignedTreeHead(
        tree_size=tree_size, root_hex=root.hex(), timestamp=timestamp, signature_hex=""
    )
    sth.signature_hex = key.sign(sth.payload()).hex()
    return sth
