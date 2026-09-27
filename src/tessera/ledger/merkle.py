"""RFC 6962 Merkle transparency log for detector verdicts.

Implements the Certificate Transparency constructions verbatim - Merkle Tree Hash,
inclusion proofs (audit paths) and consistency proofs - rather than an ad-hoc hash
chain, because those are the constructions with published verification algorithms
that an auditor can independently check.

Design goal. Storing raw PII - source/destination IPs, geolocation, full
request/response bodies - on an immutable ledger turns permanence into a privacy
liability: whatever goes on-chain can never be redacted. TESSERA's leaves commit
only a hash of {window_id, host_hash, ts_bucket, verdict, score, model_git_sha}, so
the log can be public and tamper-evident without ever being a PII sink, and the data
stays off-chain. Segmentation is done by SIGNED CHECKPOINT rather than by
truncating and re-mining a short chain (which would sever hash linkage to genesis
and destroy the tamper-evidence the design exists to provide): the first leaf of a
new segment commits to the previous segment's root, so the archive stays verifiable
from a retained tree head.

Claim licensed, and nothing stronger: *an auditor holding a past signed tree head
can detect any retroactive modification or deletion of any retained verdict.*
See THREAT_MODEL.md for what is explicitly out of scope (split-view without gossip,
an adversary holding the signing key, and archive destruction, which yields
unverifiability rather than tamper-evidence).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def _h(*parts: bytes) -> bytes:
    d = hashlib.sha256()
    for p in parts:
        d.update(p)
    return d.digest()


def leaf_hash(data: bytes) -> bytes:
    """RFC 6962 leaf hash. The 0x00 prefix is what stops second-preimage attacks
    that would otherwise let an internal node be passed off as a leaf."""
    return _h(LEAF_PREFIX, data)


def node_hash(left: bytes, right: bytes) -> bytes:
    return _h(NODE_PREFIX, left, right)


def _largest_power_of_two_below(n: int) -> int:
    k = 1
    while k * 2 < n:
        k *= 2
    return k


def merkle_tree_hash(leaves: list[bytes]) -> bytes:
    """MTH over a list of LEAF HASHES. Empty tree hashes the empty string."""
    n = len(leaves)
    if n == 0:
        return hashlib.sha256(b"").digest()
    if n == 1:
        return leaves[0]
    k = _largest_power_of_two_below(n)
    return node_hash(merkle_tree_hash(leaves[:k]), merkle_tree_hash(leaves[k:]))


def inclusion_path(index: int, leaves: list[bytes]) -> list[bytes]:
    """RFC 6962 PATH(m, D[n])."""
    n = len(leaves)
    if not 0 <= index < n:
        raise IndexError(f"index {index} out of range for tree of size {n}")
    if n == 1:
        return []
    k = _largest_power_of_two_below(n)
    if index < k:
        return inclusion_path(index, leaves[:k]) + [merkle_tree_hash(leaves[k:])]
    return inclusion_path(index - k, leaves[k:]) + [merkle_tree_hash(leaves[:k])]


def verify_inclusion(
    leaf: bytes, index: int, tree_size: int, path: list[bytes], root: bytes
) -> bool:
    """RFC 6962-bis inclusion verification. Returns False rather than raising, so a
    tampered entry produces a clean negative an auditor can act on."""
    if index >= tree_size or tree_size == 0:
        return False
    fn, sn, r = index, tree_size - 1, leaf
    for p in path:
        if sn == 0:
            return False  # proof longer than the tree can justify
        if (fn & 1) or fn == sn:
            r = node_hash(p, r)
            if not (fn & 1):
                while True:
                    fn >>= 1
                    sn >>= 1
                    if (fn & 1) or fn == 0:
                        break
        else:
            r = node_hash(r, p)
        fn >>= 1
        sn >>= 1
    return sn == 0 and r == root


def consistency_path(first: int, leaves: list[bytes]) -> list[bytes]:
    """RFC 6962 PROOF(m, D[n]): proves the size-``first`` tree is a prefix."""
    n = len(leaves)
    if first == 0 or first > n:
        return []
    if first == n:
        return []
    return _subproof(first, leaves, True)


def _subproof(m: int, leaves: list[bytes], b: bool) -> list[bytes]:
    n = len(leaves)
    if m == n:
        return [] if b else [merkle_tree_hash(leaves)]
    k = _largest_power_of_two_below(n)
    if m <= k:
        return _subproof(m, leaves[:k], b) + [merkle_tree_hash(leaves[k:])]
    return _subproof(m - k, leaves[k:], False) + [merkle_tree_hash(leaves[:k])]


def verify_consistency(
    first_size: int, first_root: bytes, second_size: int, second_root: bytes, path: list[bytes]
) -> bool:
    """RFC 6962-bis consistency verification: is the old tree a prefix of the new one?

    This is the check that detects a rewritten history: if an operator edits or
    deletes a past verdict, no valid consistency proof exists against a tree head
    the auditor already holds.
    """
    if first_size > second_size:
        return False
    if first_size == second_size:
        return not path and first_root == second_root
    if first_size == 0:
        return not path

    proof = list(path)
    # A tree whose size is an exact power of two has its root as an implicit node.
    if first_size & (first_size - 1) == 0:
        proof = [first_root] + proof
    if not proof:
        return False

    fn, sn = first_size - 1, second_size - 1
    while fn & 1:
        fn >>= 1
        sn >>= 1

    fr = sr = proof[0]
    for c in proof[1:]:
        if sn == 0:
            return False
        if (fn & 1) or fn == sn:
            fr = node_hash(c, fr)
            sr = node_hash(c, sr)
            if not (fn & 1):
                while True:
                    fn >>= 1
                    sn >>= 1
                    if (fn & 1) or fn == 0:
                        break
        else:
            sr = node_hash(sr, c)
        fn >>= 1
        sn >>= 1

    return sn == 0 and fr == first_root and sr == second_root


def canonical_bytes(obj: dict) -> bytes:
    """Deterministic serialisation. Any ambiguity here breaks every proof."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class InclusionProof:
    leaf_index: int
    tree_size: int
    path: list[bytes]
    leaf: bytes

    def verify(self, root: bytes) -> bool:
        return verify_inclusion(self.leaf, self.leaf_index, self.tree_size, self.path, root)

    def size_bytes(self) -> int:
        return sum(len(p) for p in self.path)

    def as_dict(self) -> dict:
        return {
            "leaf_index": self.leaf_index,
            "tree_size": self.tree_size,
            "path": [p.hex() for p in self.path],
            "leaf": self.leaf.hex(),
            "path_length": len(self.path),
            "proof_bytes": self.size_bytes(),
        }


class MerkleLog:
    """Append-only log of leaf hashes with inclusion and consistency proofs."""

    def __init__(self) -> None:
        self._leaves: list[bytes] = []
        self._entries: list[bytes] = []

    def __len__(self) -> int:
        return len(self._leaves)

    def append(self, data: bytes) -> int:
        """Append one entry and return its index."""
        self._entries.append(data)
        self._leaves.append(leaf_hash(data))
        return len(self._leaves) - 1

    def append_json(self, obj: dict) -> int:
        return self.append(canonical_bytes(obj))

    def root(self, size: int | None = None) -> bytes:
        size = len(self._leaves) if size is None else size
        if size > len(self._leaves):
            raise ValueError(f"requested root at size {size}, log has {len(self._leaves)}")
        return merkle_tree_hash(self._leaves[:size])

    def inclusion_proof(self, index: int, size: int | None = None) -> InclusionProof:
        size = len(self._leaves) if size is None else size
        return InclusionProof(
            leaf_index=index,
            tree_size=size,
            path=inclusion_path(index, self._leaves[:size]),
            leaf=self._leaves[index],
        )

    def consistency_proof(self, first_size: int, second_size: int | None = None) -> list[bytes]:
        second_size = len(self._leaves) if second_size is None else second_size
        return consistency_path(first_size, self._leaves[:second_size])

    def entry(self, index: int) -> bytes:
        return self._entries[index]

    def tamper(self, index: int, data: bytes) -> None:
        """Overwrite a past entry. FOR TESTS AND THE DEMO ONLY.

        A real log has no such operation; it exists so the demo can let a visitor
        edit a stored verdict and watch consistency verification fail against a
        retained tree head. That is the clearest possible demonstration that the
        guarantee is real rather than asserted.
        """
        self._entries[index] = data
        self._leaves[index] = leaf_hash(data)
