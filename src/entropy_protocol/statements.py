"""Signed statements on the IETF building blocks (COSE_Sign1 + CWT claims + COSE Hash Envelope).

1. Generation statement — issued by the generating party when a text is sealed, registered before the text
   is released. COSE Hash Envelope (RFC 9995): the payload is SHA-256 of a small CBOR commitment
     {v, doc, rec: SHA-256(record plaintext), txt: SHA-256(salt || SHA-256(canonical text)), canon, wts, tok}
   so the log holds neither text nor record, and a short text cannot be confirmed by guessing (the salt
   lives inside the encrypted record). The record is hashed before encryption: every copy re-encrypted for a
   verifier opens to the same bytes, and the random salt inside makes the hash unguessable. Whoever can open
   a copy recomputes the commitment from it.
2. Verification statement — issued by a verifier (its own Issuer identity), subject = the document id:
   verdict, counts, canonical character ranges (no text), content hash of the candidate, model identity,
   decoding configuration, nonce, time, and a reference to the generation statement (SHA-256 of its logged
   form). With ``envelope`` the payload is a Hash Envelope over the same map, for public registration.
3. Detection statement — the same shape for the record-free (watermark) mode: detected, counts (the exact
   p-value follows from them), technique identifier: the signed, time-stamped detection result with content
   hash and detector identity that the EU Code of Practice on AI-generated content asks detectors to provide.
No wire format of ep1 carries floating point: tolerances are integer micro-units, fractions per-mille.
"""
from __future__ import annotations

import hashlib
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import ENGINE, cose, crypto
from .canon import CANON_ID, canonicalize
from .record import loads

TYP_GENERATION = "application/ep1-generation-statement+cose"
TYP_VERIFICATION = "application/ep1-verification-statement+cose"
TYP_DETECTION = "application/ep1-detection-statement+cose"
CT_COMMITMENT = "application/ep1-commitment+cbor"
CT_VERIFICATION = "application/ep1-verification+cbor"
CT_DETECTION = "application/ep1-detection+cbor"


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def commitment(plaintext: bytes, doc_id: str) -> dict:
    """The commitment for a record, from its decrypted CBOR bytes."""
    h = loads(plaintext).header
    if not h.salt:
        raise ValueError("record has no commitment salt")
    return {"v": 1, "doc": doc_id, "rec": sha256(plaintext),
            "txt": sha256(bytes.fromhex(h.salt) + bytes.fromhex(h.text_digest)),
            "canon": h.canon, "wts": bytes.fromhex(h.weights_digest), "tok": bytes.fromhex(h.tokenizer_digest)}


def _claims(iss: str, sub: str, iat: int | None) -> dict:
    return {cose.ISS: iss, cose.SUB: sub, cose.IAT: int(time.time()) if iat is None else int(iat)}


def generation_statement(key: bytes, token: str, signer: Ed25519PrivateKey, issuer: str,
                         iat: int | None = None, extra: dict | None = None) -> tuple[bytes, dict]:
    """(Signed Statement, commitment). ``key`` opens the record: the custody key K_log or a verifier key.
    ``extra``: further protected header parameters - in an attested environment the key's certificate (x5chain)
    and the digest of the attestation result (ARCHITECTURE §7d)."""
    doc_id = crypto.token_doc_id(token)
    c = commitment(crypto.decrypt_token(key, token), doc_id)
    protected = {cose.ALG: cose.ED25519, cose.KID: cose.kid_of(signer.public_key()),
                 cose.CWT_CLAIMS: _claims(issuer, doc_id, iat), cose.TYP: TYP_GENERATION,
                 cose.PAYLOAD_HASH_ALG: cose.SHA256, cose.PREIMAGE_CONTENT_TYPE: CT_COMMITMENT, **(extra or {})}
    return cose.sign1(protected, sha256(cose.dumps(c)), signer), c


def check_generation_statement(statement: bytes, issuer_key: Ed25519PublicKey, key: bytes, token: str) -> dict:
    """Signature valid, subject = the token's document, and the commitment recomputed from the record matches."""
    prot, _, payload = cose.verify1(statement, issuer_key)
    if prot.get(cose.PAYLOAD_HASH_ALG) != cose.SHA256 or prot.get(cose.PREIMAGE_CONTENT_TYPE) != CT_COMMITMENT:
        raise cose.COSEError("not a generation statement")
    doc_id = crypto.token_doc_id(token)
    if prot.get(cose.CWT_CLAIMS, {}).get(cose.SUB) != doc_id:
        raise cose.COSEError("statement subject is another document")
    c = commitment(crypto.decrypt_token(key, token), doc_id)
    if sha256(cose.dumps(c)) != payload:
        raise cose.COSEError("commitment does not match the record")
    return {"issuer": prot[cose.CWT_CLAIMS][cose.ISS], "iat": prot[cose.CWT_CLAIMS][cose.IAT], "commitment": c}


def statement_ref(statement: bytes) -> bytes:
    """Reference to a statement: SHA-256 of its logged form (unprotected header emptied)."""
    from .tlog import entry_of
    return sha256(entry_of(statement))


def _sign_result(body: dict, signer: Ed25519PrivateKey, issuer: str, sub: str, typ: str, ct: str,
                 envelope: bool, iat: int | None, extra: dict | None = None) -> bytes:
    protected = {cose.ALG: cose.ED25519, cose.KID: cose.kid_of(signer.public_key()),
                 cose.CWT_CLAIMS: _claims(issuer, sub, iat), cose.TYP: typ, **(extra or {})}
    if envelope:
        protected.update({cose.PAYLOAD_HASH_ALG: cose.SHA256, cose.PREIMAGE_CONTENT_TYPE: ct})
        return cose.sign1(protected, sha256(cose.dumps(body)), signer)
    return cose.sign1(protected, cose.dumps(body), signer)


def verification_statement(result, candidate: str, signer: Ed25519PrivateKey, verifier_id: str, *,
                           generation: bytes | None = None, envelope: bool = False,
                           iat: int | None = None, extra: dict | None = None) -> tuple[bytes, dict]:
    """(Signed Statement, payload map) for a ``verify.Result``. No text, no per-position ranks or probabilities."""
    d = result.decoding or {}
    body = {"v": 1, "verdict": result.verdict, "doc": result.doc_id,
            "cand": sha256(canonicalize(candidate, result.decoding.get("canon", CANON_ID)).encode()),
            "gen": statement_ref(generation) if generation else None,
            "counts": [result.matched, result.divergent, result.indeterminate, result.entries],
            "ranges": {"changed": result.changed_ranges, "deleted": result.deletion_points,
                       "divergent": result.divergence_ranges},
            "wts": bytes.fromhex(result.weights_digest) if result.weights_digest else b"",
            "tok": bytes.fromhex(result.tokenizer_digest) if result.tokenizer_digest else b"",
            "dec": {"class": d.get("class", ""), "dtype": d.get("dtype", ""), "space": d.get("tol_space", ""),
                    "tol_u": int(round(float(d.get("tol", 0.0)) * 1_000_000)), "canon": d.get("canon", CANON_ID),
                    "calibrated": result.calibrated},
            "nonce": result.nonce, "engine": result.engine}
    return _sign_result(body, signer, verifier_id, result.doc_id or "", TYP_VERIFICATION, CT_VERIFICATION,
                        envelope, iat, extra), body


def detection_statement(detection, candidate: str, backend, signer: Ed25519PrivateKey, detector_id: str, *,
                        envelope: bool = False, iat: int | None = None,
                        extra: dict | None = None) -> tuple[bytes, dict]:
    """(Signed Statement, payload map) for a ``detect.Detection``."""
    canon = getattr(detection, "canon", CANON_ID)
    cand = sha256(canonicalize(candidate, canon).encode())
    # no floating point on the wire: the p-value is exactly binomial_tail(checkpoints, matched); the threshold
    # travels as its decimal exponent (alpha = 10^-alpha_exp)
    import math
    body = {"v": 1, "detected": detection.detected, "alpha_exp": int(round(-math.log10(detection.alpha))),
            "counts": [detection.matched, detection.checkpoints, detection.excluded, detection.tokens],
            "technique": f"ep1-wm/k{detection.k}-i{detection.interval}-c{detection.context}"
                         + (f"-g{round(detection.gate * 1000)}" if detection.gate else ""), "cand": cand,
            "wts": bytes.fromhex(backend.weights_digest), "tok": bytes.fromhex(backend.tokenizer_digest),
            "engine": ENGINE, "canon": canon}
    return _sign_result(body, signer, detector_id, cand.hex(), TYP_DETECTION, CT_DETECTION, envelope, iat,
                        extra), body


def read_statement(statement: bytes, issuer_key: Ed25519PublicKey, body: dict | None = None) -> dict:
    """Verify a verification or detection statement. For an envelope, pass the payload map to check it."""
    prot, _, payload = cose.verify1(statement, issuer_key)
    if prot.get(cose.PAYLOAD_HASH_ALG) == cose.SHA256:
        if body is None:
            return {"claims": prot[cose.CWT_CLAIMS], "typ": prot.get(cose.TYP), "payload": None}
        if sha256(cose.dumps(body)) != payload:
            raise cose.COSEError("payload map does not match the envelope")
        return {"claims": prot[cose.CWT_CLAIMS], "typ": prot.get(cose.TYP), "payload": body}
    return {"claims": prot[cose.CWT_CLAIMS], "typ": prot.get(cose.TYP), "payload": cose.loads(payload)}
