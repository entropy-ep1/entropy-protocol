"""A minimal append-only transparency log: RFC 9162 Merkle tree (SHA-256) with RFC 9942 receipts.

For local runs, tests and interop checks; a production issuer registers with a SCITT Transparency Service
(RFC 9943) through SCRAPI instead.
- Entry: the registered Signed Statement with its unprotected header emptied (a TS clears the unprotected
  header before logging, RFC 9943 section 6.3). RFC 9942 leaves the entry encoding to the log.
- Leaf hash = SHA-256(0x00 || entry); interior = SHA-256(0x01 || left || right) (RFC 9162 section 2.1.1).
- Receipt of inclusion (RFC 9942 section 5.2.1): COSE_Sign1, protected {alg, vds: 1 (RFC9162_SHA256), kid},
  unprotected {vdp: {-1: [bstr .cbor [tree_size, leaf_index, [path]]]}}, detached payload = the tree root.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import cose

RFC9162_SHA256 = 1
INCLUSION = -1


def _h(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def leaf_hash(entry: bytes) -> bytes:
    return _h(b"\x00" + entry)


def _node(left: bytes, right: bytes) -> bytes:
    return _h(b"\x01" + left + right)


def _split(n: int) -> int:
    """Largest power of two smaller than n (n > 1)."""
    k = 1
    while k << 1 < n:
        k <<= 1
    return k


def mth(leaves: list[bytes]) -> bytes:
    """Merkle Tree Hash over leaf hashes (RFC 9162 section 2.1.1)."""
    n = len(leaves)
    if n == 0:
        return _h(b"")
    if n == 1:
        return leaves[0]
    k = _split(n)
    return _node(mth(leaves[:k]), mth(leaves[k:]))


def inclusion_path(m: int, leaves: list[bytes]) -> list[bytes]:
    """PATH(m, D[n]) of RFC 9162 section 2.1.3.1, over leaf hashes."""
    n = len(leaves)
    if n <= 1:
        return []
    k = _split(n)
    if m < k:
        return inclusion_path(m, leaves[:k]) + [mth(leaves[k:])]
    return inclusion_path(m - k, leaves[k:]) + [mth(leaves[:k])]


def root_from_inclusion(leaf: bytes, index: int, size: int, path: list[bytes]) -> bytes:
    """RFC 9162 section 2.1.3.2: the root a valid inclusion proof yields, or ValueError."""
    if index >= size:
        raise ValueError("leaf index outside the tree")
    fn, sn, r = index, size - 1, leaf
    for p in path:
        if sn == 0:
            raise ValueError("inclusion path longer than the tree")
        if fn & 1 or fn == sn:
            r = _node(p, r)
            while not (fn & 1) and fn != 0:
                fn >>= 1
                sn >>= 1
        else:
            r = _node(r, p)
        fn >>= 1
        sn >>= 1
    if sn != 0:
        raise ValueError("inclusion path shorter than the tree")
    return r


def entry_of(statement: bytes) -> bytes:
    """The logged form of a Signed Statement: same protected header, payload and signature; empty unprotected."""
    _, prot_bytes, _, payload, sig = cose.decode1(statement)
    return cose.dumps(cbor2.CBORTag(cose.COSE_SIGN1_TAG, [prot_bytes, {}, payload, sig]))


class TransparencyLog:
    def __init__(self, key: Ed25519PrivateKey, path: str | Path | None = None):
        self.key = key
        self.kid = cose.kid_of(key.public_key())
        self.path = Path(path) if path else None
        self.entries: list[bytes] = []
        if self.path and self.path.exists():
            self.entries = [bytes.fromhex(json.loads(line)["entry"]) for line in self.path.read_text().splitlines() if line]
        self._leaves = [leaf_hash(e) for e in self.entries]

    @property
    def size(self) -> int:
        return len(self.entries)

    def root(self, size: int | None = None) -> bytes:
        return mth(self._leaves[: self.size if size is None else size])

    def register(self, statement: bytes) -> tuple[int, bytes]:
        """Append a Signed Statement; return (leaf index, receipt at the new tree size)."""
        entry = entry_of(statement)
        self.entries.append(entry)
        self._leaves.append(leaf_hash(entry))
        if self.path:
            with self.path.open("a") as fh:
                fh.write(json.dumps({"entry": entry.hex()}) + "\n")
        return self.size - 1, self.receipt(self.size - 1)

    def receipt(self, index: int, size: int | None = None) -> bytes:
        size = self.size if size is None else size
        proof = [size, index, inclusion_path(index, self._leaves[:size])]
        protected = {cose.ALG: cose.ED25519, cose.KID: self.kid, cose.VDS: RFC9162_SHA256}
        unprotected = {cose.VDP: {INCLUSION: [cose.dumps(proof)]}}
        return cose.sign1(protected, self.root(size), self.key, unprotected=unprotected, detached=True)


def verify_receipt(receipt: bytes, statement: bytes, log_public_key: Ed25519PublicKey) -> dict:
    """Check that ``statement`` is included in the log the receipt's signature commits to."""
    prot, _, unprot, _, _ = cose.decode1(receipt)
    if prot.get(cose.VDS) != RFC9162_SHA256:
        raise cose.COSEError("unsupported verifiable data structure")
    proofs = unprot.get(cose.VDP, {}).get(INCLUSION)
    if not proofs:
        raise cose.COSEError("receipt carries no inclusion proof")
    size, index, path = cbor2.loads(proofs[0])
    root = root_from_inclusion(leaf_hash(entry_of(statement)), index, size, list(path))
    cose.verify1(receipt, log_public_key, payload=root)
    return {"tree_size": size, "leaf_index": index, "root": root.hex()}
