"""Signed-checkpoint segmentation: the correct replacement for "archive the long
chain and mine on the short one".

Growing per-append cost is capped by splitting the chain into segments rather than
by truncating and re-mining a short chain, which would sever hash linkage to
genesis. Here a new segment's FIRST leaf commits to the previous segment's root,
size, last leaf hash and signature, so an auditor holding any past tree head can
still verify inclusion in the archive. The cost is bounded the same way; the
guarantee survives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tessera.ledger.merkle import MerkleLog, canonical_bytes
from tessera.ledger.sth import SignedTreeHead, sign_tree_head


@dataclass
class Segment:
    index: int
    log: MerkleLog
    closing_sth: SignedTreeHead | None = None
    anchor: dict | None = None  # commitment to the PREVIOUS segment

    def size(self) -> int:
        return len(self.log)


@dataclass
class SegmentedLog:
    """A chain of segments linked by signed anchors.

    ``segment_length`` is the decision variable the chain simulator optimises. Note
    that our first draft objective for tuning it does not depend on it at all -
    per-block read/write/hash/verify cost is independent of where the chain is cut -
    which is demonstrated directly in ``chainsim/segment_objective.py``.
    """

    segment_length: int = 1024
    segments: list[Segment] = field(default_factory=list)
    _clock: int = 0

    def __post_init__(self) -> None:
        if self.segment_length < 1:
            raise ValueError("segment_length must be >= 1")
        if not self.segments:
            self.segments = [Segment(index=0, log=MerkleLog())]

    @property
    def current(self) -> Segment:
        return self.segments[-1]

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def append_json(self, obj: dict) -> tuple[int, int]:
        """Append a verdict, rolling to a new segment when the current one is full.

        Returns ``(segment_index, leaf_index)``.
        """
        if self.current.size() >= self.segment_length:
            self._close_and_open()
        return self.current.index, self.current.log.append_json(obj)

    def _close_and_open(self) -> None:
        closing = self.current
        size = closing.size()
        root = closing.log.root()
        closing.closing_sth = sign_tree_head(root, size, timestamp=self._tick())

        anchor = {
            "_anchor": True,
            "prev_segment_index": closing.index,
            "prev_root": root.hex(),
            "prev_size": size,
            "prev_last_leaf": closing.log.inclusion_proof(size - 1).leaf.hex() if size else None,
            "prev_sth_signature": closing.closing_sth.signature_hex,
        }
        new = Segment(index=closing.index + 1, log=MerkleLog(), anchor=anchor)
        # The anchor is the new segment's FIRST leaf, so the link is inside the
        # hashed structure rather than alongside it.
        new.log.append(canonical_bytes(anchor))
        self.segments.append(new)

    def verify_chain(self) -> dict:
        """Walk every anchor and confirm each segment still commits to its predecessor."""
        problems: list[str] = []
        for seg in self.segments[1:]:
            prev = self.segments[seg.index - 1]
            a = seg.anchor or {}
            if a.get("prev_root") != prev.log.root(a.get("prev_size", 0)).hex():
                problems.append(
                    f"segment {seg.index} anchor does not match segment {prev.index} root"
                )
            first_leaf = seg.log.entry(0)
            if first_leaf != canonical_bytes(a):
                problems.append(f"segment {seg.index} first leaf is not its anchor")
        return {
            "n_segments": len(self.segments),
            "segment_length": self.segment_length,
            "total_entries": sum(s.size() for s in self.segments),
            "intact": not problems,
            "problems": problems,
        }

    def stats(self) -> dict:
        sizes = [s.size() for s in self.segments]
        current = self.current
        proof = current.log.inclusion_proof(0) if current.size() else None
        return {
            "n_segments": len(self.segments),
            "segment_length": self.segment_length,
            "segment_sizes": sizes,
            "total_entries": sum(sizes),
            # These are the quantities that genuinely vary with segment_length, and
            # therefore make the optimisation problem well-posed.
            "current_proof_path_length": len(proof.path) if proof else 0,
            "current_proof_bytes": proof.size_bytes() if proof else 0,
            "archived_segments": max(0, len(self.segments) - 1),
        }
