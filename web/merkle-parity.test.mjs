// Parity test: does the JS Merkle log reproduce the REAL Python ledger's roots
// and inclusion proofs, using vectors computed by tessera/ledger/merkle.py
// (RFC 6962 - the same construction Certificate Transparency uses)?
//
// Run: node web/merkle-parity.test.mjs

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { MerkleLog, setSha256, verifyInclusion, bytesToHex, canonicalBytes } from "./js/merkle.js";

setSha256(async (bytes) => new Uint8Array(createHash("sha256").update(bytes).digest()));

const vectors = JSON.parse(readFileSync(new URL("./merkle_vectors.json", import.meta.url)));

const log = new MerkleLog();
for (const v of vectors.verdicts) await log.appendJson(v);

const root = await log.root();
const rootHex = bytesToHex(root);
console.log(`MERKLE PARITY TEST (${vectors.n} real verdicts, RFC 6962)`);
console.log(`  root:     JS=${rootHex}`);
console.log(`            py=${vectors.root_hex}`);

let ok = rootHex === vectors.root_hex;
if (!ok) {
  console.error("FAIL: root mismatch between JS and Python");
  process.exit(1);
}
console.log("  root MATCHES Python exactly.\n");

let nProofsOk = 0;
for (const pv of vectors.proofs) {
  const proof = await log.inclusionProof(pv.leaf_index, pv.tree_size);
  const leafHex = bytesToHex(proof.leaf);
  const pathHex = proof.path.map(bytesToHex);
  const leafMatch = leafHex === pv.leaf_hex;
  const pathMatch = JSON.stringify(pathHex) === JSON.stringify(pv.path_hex);
  const verifies = await verifyInclusion(proof.leaf, pv.leaf_index, pv.tree_size, proof.path, root);
  if (leafMatch && pathMatch && verifies) {
    nProofsOk++;
  } else {
    console.error(`  proof ${pv.leaf_index}: leafMatch=${leafMatch} pathMatch=${pathMatch} verifies=${verifies}`);
    ok = false;
  }
}
console.log(`  ${nProofsOk}/${vectors.proofs.length} inclusion proofs match Python and verify.`);

// canonical_bytes must serialise identically too, or a leaf built in JS from a
// live verdict would hash differently than the same verdict hashed in Python.
const sample = { z: 1, a: [3, 2, 1], m: { b: 2, a: 1 } };
const jsBytes = canonicalBytes(sample);
const jsJson = new TextDecoder().decode(jsBytes);
console.log(`\n  canonical_bytes sample: ${jsJson}`);
const expectedJson = '{"a":[3,2,1],"m":{"a":1,"b":2},"z":1}';
if (jsJson !== expectedJson) {
  console.error(`FAIL: canonical serialisation mismatch, expected ${expectedJson}`);
  ok = false;
}

// The tamper check: editing a past entry must break its own recorded proof.
const before = await log.inclusionProof(3);
const rootBefore = await log.root();
await log.tamper(3, canonicalBytes({ window_id: "FORGED", verdict: false, score: 0 }));
const rootAfter = await log.root();
const stillVerifies = await verifyInclusion(before.leaf, 3, before.treeSize, before.path, rootBefore);
const staleProofOnNewRoot = await verifyInclusion(before.leaf, 3, before.treeSize, before.path, rootAfter);
console.log(`\n  TAMPER CHECK: root changed after edit: ${bytesToHex(rootBefore) !== bytesToHex(rootAfter)}`);
console.log(`  old proof still verifies against OLD root (expected true): ${stillVerifies}`);
console.log(`  old proof verifies against NEW (post-tamper) root (expected false): ${staleProofOnNewRoot}`);
if (!stillVerifies || staleProofOnNewRoot) {
  console.error("FAIL: tamper-evidence property did not hold");
  ok = false;
}

if (!ok) process.exit(1);
console.log("\nMERKLE PARITY CONFIRMED: the browser ledger matches RFC 6962 and the real Python implementation.");
