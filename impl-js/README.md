# ep1 — second implementation (JavaScript, model-free parts)

PRIVATE until the covering provisional application is filed.

A second codebase, in a second language, written from `spec/ep1.cddl` and `ARCHITECTURE.md`. Its purpose is to
prove that the specification is complete enough for another implementer to reproduce every byte, and to catch
ambiguities the reference implementation hides. It covers everything a verifier needs apart from running the model:

| Part | Function |
|---|---|
| Deterministic CBOR (RFC 8949 §4.2.1): encode, decode, shortest floats, bytewise key order | `encode`, `decode` |
| Canonical form v3 (code-point indexed, as in the record) and sentence segmentation, block digests | `canonicalize`, `segment`, `blockDigest` |
| Key hierarchy (HKDF-SHA256) and keyed selection (`bit`, content selection `wmDecide`) | `keys`, `bit`, `wmDecide` |
| Record token: ChaCha20-Poly1305 with the CBOR AAD | `openToken` |
| Commitment recomputation from a record copy | `commitment` |
| COSE_Sign1 with Ed25519 (alg −19 / −8), logged form of a statement | `verify1`, `entryOf` |
| RFC 9162 inclusion proofs, RFC 9942 receipts | `rootFromInclusion`, `verifyReceipt` |
| Exact binomial tail for record-free detection | `binomialTail` |
| HPKE (RFC 9180 base mode, X25519 / HKDF-SHA256 / ChaCha20-Poly1305 or AES-128-GCM), opening | `hpkeOpen` |
| Record delivered by public key: COSE-HPKE integrated encryption (draft-ietf-cose-hpke-27) | `openDelivered` |

Requirements: Node.js ≥ 20. No dependencies (Node's built-in `crypto` provides Ed25519, ChaCha20-Poly1305, HKDF,
HMAC and SHA-256; `String.prototype.normalize` provides NFC).

```bash
node impl-js/check-vectors.mjs      # 41/41 checks against spec/test-vectors/ep1-vectors.json
```

What writing it found in the specification: floating-point values in the record header could not be reproduced
from a JavaScript decode (0.0 and 0 are the same number there), so ep1 now carries no floating point on the wire
(tolerances in integer micro-units, fractions per-mille, p-values derived from counts).
Honest scope: same team, different language; it is a second implementation, not an independent one in the IETF
sense.
