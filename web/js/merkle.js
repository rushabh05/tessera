// RFC 6962 Merkle transparency log, ported from tessera/ledger/merkle.py.
// Uses SHA-256 via Web Crypto in the browser, or Node's crypto module when run
// under Node (for the parity test) - the same algorithm either way, so no
// behavioural difference between the demo page and its own test.

const LEAF_PREFIX = new Uint8Array([0x00]);
const NODE_PREFIX = new Uint8Array([0x01]);

let sha256Impl = null;
export function setSha256(fn) {
  sha256Impl = fn; // async (Uint8Array) => Uint8Array
}

function concatBytes(...parts) {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

export async function leafHash(dataBytes) {
  return sha256Impl(concatBytes(LEAF_PREFIX, dataBytes));
}

export async function nodeHash(left, right) {
  return sha256Impl(concatBytes(NODE_PREFIX, left, right));
}

function largestPowerOfTwoBelow(n) {
  let k = 1;
  while (k * 2 < n) k *= 2;
  return k;
}

// MTH(D[n]) over a list of LEAF HASHES (Uint8Array[]).
export async function merkleTreeHash(leaves) {
  const n = leaves.length;
  if (n === 0) return sha256Impl(new Uint8Array(0));
  if (n === 1) return leaves[0];
  const k = largestPowerOfTwoBelow(n);
  const left = await merkleTreeHash(leaves.slice(0, k));
  const right = await merkleTreeHash(leaves.slice(k));
  return nodeHash(left, right);
}

// PATH(m, D[n]): the inclusion audit path for leaf index `m`.
export async function inclusionPath(index, leaves) {
  const n = leaves.length;
  if (index < 0 || index >= n) throw new RangeError(`index ${index} out of range for tree of size ${n}`);
  if (n === 1) return [];
  const k = largestPowerOfTwoBelow(n);
  if (index < k) {
    const rest = await inclusionPath(index, leaves.slice(0, k));
    return [...rest, await merkleTreeHash(leaves.slice(k))];
  }
  const rest = await inclusionPath(index - k, leaves.slice(k));
  return [...rest, await merkleTreeHash(leaves.slice(0, k))];
}

function bytesEqual(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// RFC 6962-bis inclusion verification: does `leaf` at `index`, combined with
// `path`, reconstruct `root` for a tree of size `treeSize`? Returns false
// rather than throwing on a failed/tampered proof, so a UI can show a clean
// negative rather than a stack trace.
export async function verifyInclusion(leaf, index, treeSize, path, root) {
  if (index >= treeSize || treeSize === 0) return false;
  let fn = index;
  let sn = treeSize - 1;
  let r = leaf;
  for (const p of path) {
    if (sn === 0) return false;
    if (fn & 1 || fn === sn) {
      r = await nodeHash(p, r);
      if (!(fn & 1)) {
        while (true) {
          fn >>= 1;
          sn >>= 1;
          if (fn & 1 || fn === 0) break;
        }
      }
    } else {
      r = await nodeHash(r, p);
    }
    fn >>= 1;
    sn >>= 1;
  }
  return sn === 0 && bytesEqual(r, root);
}

export function bytesToHex(b) {
  return Array.from(b).map((x) => x.toString(16).padStart(2, "0")).join("");
}

export function canonicalBytes(obj) {
  // Deterministic serialisation matching merkle.py's canonical_bytes: sorted
  // keys, no extra whitespace. Any ambiguity here breaks every proof.
  const json = JSON.stringify(sortKeysDeep(obj));
  return new TextEncoder().encode(json);
}

function sortKeysDeep(value) {
  if (Array.isArray(value)) return value.map(sortKeysDeep);
  if (value && typeof value === "object") {
    const out = {};
    for (const k of Object.keys(value).sort()) out[k] = sortKeysDeep(value[k]);
    return out;
  }
  return value;
}

// A tiny append-only log wrapper, mirroring ledger/merkle.py's MerkleLog.
export class MerkleLog {
  constructor() {
    this._leaves = [];
    this._entries = [];
  }
  get length() {
    return this._leaves.length;
  }
  async append(dataBytes) {
    this._entries.push(dataBytes);
    this._leaves.push(await leafHash(dataBytes));
    return this._leaves.length - 1;
  }
  async appendJson(obj) {
    return this.append(canonicalBytes(obj));
  }
  async root(size = this._leaves.length) {
    if (size > this._leaves.length) throw new RangeError("root() size exceeds log length");
    return merkleTreeHash(this._leaves.slice(0, size));
  }
  async inclusionProof(index, size = this._leaves.length) {
    return {
      leafIndex: index,
      treeSize: size,
      path: await inclusionPath(index, this._leaves.slice(0, size)),
      leaf: this._leaves[index],
    };
  }
  entry(index) {
    return this._entries[index];
  }
  // FOR THE TAMPER DEMO ONLY - a real log has no such operation. Overwrites a
  // past entry so the page can show what happens next: the recomputed root
  // changes, and a proof recorded before the edit no longer verifies.
  tamper(index, dataBytes) {
    this._entries[index] = dataBytes;
    this._leaves[index] = null; // recomputed lazily
    return leafHash(dataBytes).then((h) => {
      this._leaves[index] = h;
    });
  }
}
