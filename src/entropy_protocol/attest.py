"""Evidence pack and signed verification attestation (spec [0126]-[0128]).

The pack carries the verdict, counts, divergence ranges, the model/tokenizer identity and the
decoding configuration, the verifier nonce and a digest of the log token — never the text.
Signed with Ed25519 over canonical JSON; checkable by anyone with the verifier's public key.
(RFC 3161 time-stamping and registry posting: stage 3.)
"""
from __future__ import annotations

import base64
import hashlib
import json
import time

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import ENGINE, crypto

_PACK_FIELDS = ("verdict", "reason", "score", "matched", "divergent", "indeterminate", "entries", "model_evidence",
                "divergence_ranges", "changed_ranges", "deletion_points", "digest_match", "model", "weights_digest",
                "tokenizer_digest", "decoding", "nonce", "doc_id", "candidate_digest")


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def new_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def public_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def key_id(key: Ed25519PublicKey) -> str:
    return hashlib.sha256(public_bytes(key)).hexdigest()[:16]


def evidence_pack(result, verifier_id: str, token: str) -> dict:
    pack = {f: getattr(result, f) for f in _PACK_FIELDS}
    pack.update({"type": "ep1/evidence-pack", "engine": ENGINE, "verifier": verifier_id,
                 "log_token_digest": crypto.sha256_hex(token), "time": int(time.time())})
    return pack


def sign(pack: dict, key: Ed25519PrivateKey) -> dict:
    sig = key.sign(canonical_json(pack))
    return {"pack": pack, "alg": "Ed25519", "kid": key_id(key.public_key()),
            "sig": base64.urlsafe_b64encode(sig).decode().rstrip("=")}


def verify_signed(signed: dict, key: Ed25519PublicKey) -> bool:
    try:
        sig = signed["sig"]
        key.verify(base64.urlsafe_b64decode(sig + "=" * (-len(sig) % 4)), canonical_json(signed["pack"]))
        return signed.get("kid") == key_id(key)
    except (InvalidSignature, KeyError, ValueError):
        return False
