# Threat model

Many blockchain-logging designs conflate immutability with security and store raw
source and destination IPs, geolocation and full request/response bodies on a shared
immutable chain — a privacy liability presented as a privacy mechanism. TESSERA
states its threat model explicitly, which is why its ledger stores none of that. This
document states what the verdict transparency log does and, more importantly, what
it does not.

## Assets

| Asset | Sensitivity |
|---|---|
| Raw tenant logs | May contain PII: usernames, IPs, URLs, request bodies |
| Derived feature tables | Pseudonymised, but re-identifiable given side information |
| Model weights | Disclose training-set characteristics |
| Verdict transparency log | Integrity-critical, publicly auditable |
| Ed25519 signing key, pseudonymisation salt | Compromise breaks every guarantee below |

## Adversaries

1. **External attacker controlling inputs.** Generates the traffic being classified.
2. **Malicious insider with operator access** to the log store.
3. **Honest-but-curious peer tenant** sharing the deployment.
4. **Compromised log node.**
5. **Analyst holding pseudonymised logs** plus external side information.

## What the log guarantees

Exactly this, and nothing stronger:

> An auditor holding a past signed tree head can detect any retroactive modification
> or deletion of any retained verdict.

Mechanism: RFC 6962 Merkle tree, Ed25519-signed tree heads, inclusion and consistency
proofs, and signed-checkpoint segmentation in which each new segment's first leaf
commits to the previous segment's root, size, last leaf hash and signature.

Privacy mechanism: **hash-on-chain, data-off-chain.** A leaf commits to
`SHA-256(salt ‖ {window_id, host_hash, ts_bucket, verdict, score, model_git_sha})`.
No IP, geolocation, username, URL, header or body ever reaches the log. Enforced by a
test asserting the verdict payload contains no identifying field.

## Explicitly out of scope

- **Split-view attacks.** An operator controlling both the log and tree-head
  distribution can present divergent consistent histories to different auditors.
  Preventing this needs gossip or an external witness; neither is implemented.
- **Signing-key compromise.** An adversary holding the key can re-sign a rewritten
  history for any client that has not independently retained a tree head.
- **Archive destruction.** This yields **unverifiability, not tamper-evidence**: an
  auditor learns it cannot verify, not that data was altered.
- **Confidentiality.** The log provides integrity only.
- **Truthfulness of ingested records.** The log does not validate that a record
  reflects reality, and an adversary controlling log generation can omit events
  before they reach the ledger.
- **Write prevention.** Edits become detectable, not impossible.
- **Inference-time evasion**, poisoning, and concept drift.

## Privacy posture, stated honestly

Input-side pseudonymisation (CryptoPAn prefix-preserving IPs, salted SHA-256 user
ids) is a **compliance-and-utility design argument, not an empirical privacy
result.** The named residual adversary is real: injection and known-flow attacks
unravel prefix-preserving schemes. AIT is fully synthetic, so no claim about
protecting real subjects is made or supported.

Key handling: the signing key and salt are generated per deployment, live outside the
repository, are excluded from the dataset manifest, and are covered by a build-failing
test. A salt committed to the repo would let anyone re-derive every host and user hash.
