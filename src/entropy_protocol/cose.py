"""Minimal COSE_Sign1 (RFC 9052) with Ed25519, and the header parameters the statements use.

Labels are the IANA COSE registry values (checked 12.09.2026):
  1 alg · 4 kid · 15 CWT Claims (RFC 9597) · 16 typ (RFC 9596)
  258 payload-hash-alg · 259 preimage-content-type · 260 payload-location (RFC 9995, COSE Hash Envelope)
  269 3161-ttc (RFC 9921) · 394 receipts · 395 vds · 396 vdp (RFC 9942) · 3 content type
  33 x5chain (RFC 9360) · "ep1-ear" (text label, ep1: SHA-256 of the attestation result the signing key was certified
  on; attested/service.py)
Algorithms: Ed25519 = -19 (RFC 9864, fully specified; -8 "EdDSA" accepted on verification) · SHA-256 = -16.
COSE-HPKE (draft-ietf-cose-hpke-27, suggested code points, not yet IANA): HPKE-3 = 41, HPKE-4 = 42, "ek" = -4.
CWT claims: 1 iss · 2 sub · 6 iat.

Encoding: CBOR with deterministic encoding (cbor2 canonical: shortest forms, sorted keys). For the maps defined
here — small integer keys, text keys shorter than 24 bytes, no mixing of negative keys with keys of longer
encodings — that ordering equals the bytewise order of RFC 8949 section 4.2.1 (checked in the tests).
"""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

import cbor2
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

ALG, KID, CWT_CLAIMS, TYP = 1, 4, 15, 16
PAYLOAD_HASH_ALG, PREIMAGE_CONTENT_TYPE, PAYLOAD_LOCATION = 258, 259, 260
TST_TTC, RECEIPTS, VDS, VDP = 269, 394, 395, 396
ED25519, EDDSA, SHA256 = -19, -8, -16
ISS, SUB, IAT = 1, 2, 6
COSE_SIGN1_TAG = 18


class COSEError(Exception):
    """Malformed COSE object or failed signature."""


def dumps(obj) -> bytes:
    return cbor2.dumps(obj, canonical=True)


# The ep1 CBOR profile, enforced before decoding untrusted bytes, the same in both implementations (impl-js decode):
# definite lengths only, each within the bytes that are left; integers and lengths in their shortest form; tags 16
# (COSE_Encrypt0) and 18 (COSE_Sign1) only; false, true and null as the only simple values; no floating point (ep1
# writes none); map keys integers, text or byte strings, never twice in one map; at most 64 levels of nesting.
# Without it the two decoders read different things into the same bytes (differential fuzzing, 27.10.2026: tags 3
# and 5 - cbor2 interprets them, the second implementation accepted any tag), and nesting limits differed (cbor2 400,
# the JavaScript stack about 10^5).
PROFILE_TAGS = frozenset({16, 18})
MAX_DEPTH = 64


class ProfileError(COSEError, ValueError):
    """Bytes outside the ep1 CBOR profile: a COSEError where COSE is parsed, a ValueError where a record is."""


def check_profile(data: bytes) -> None:
    n = len(data)

    def arg(i: int, ai: int) -> tuple[int, int]:
        if ai < 24:
            return ai, i
        size = {24: 1, 25: 2, 26: 4, 27: 8}.get(ai)
        if size is None:
            raise ProfileError("indefinite length or reserved additional information")
        if i + size > n:
            raise ProfileError("truncated")
        v = int.from_bytes(data[i:i + size], "big")
        if v < {24: 24, 25: 0x100, 26: 0x10000, 27: 0x100000000}[ai]:
            raise ProfileError("integer or length not in its shortest form")
        return v, i + size

    def item(i: int, depth: int) -> int:
        if depth > MAX_DEPTH:
            raise ProfileError("nested too deeply")
        if i >= n:
            raise ProfileError("truncated")
        major, ai = data[i] >> 5, data[i] & 31
        i += 1
        if major == 7:
            if ai in (20, 21, 22):
                return i
            raise ProfileError("simple value or floating point outside the profile")
        v, i = arg(i, ai)
        if major in (0, 1):
            return i
        if major in (2, 3):
            if v > n - i:
                raise ProfileError("length beyond the input")
            return i + v
        if major == 4:
            if v > n - i:
                raise ProfileError("length beyond the input")
            for _ in range(v):
                i = item(i, depth + 1)
            return i
        if major == 5:
            if v > n - i:
                raise ProfileError("length beyond the input")
            seen = set()
            for _ in range(v):
                if i >= n or data[i] >> 5 > 3:
                    raise ProfileError("map key that is not an integer, text or byte string")
                j = item(i, depth + 1)
                if data[i:j] in seen:
                    raise ProfileError("duplicate map key")
                seen.add(data[i:j])
                i = item(j, depth + 1)
            return i
        if v not in PROFILE_TAGS:
            raise ProfileError(f"tag {v} outside the profile")
        return item(i, depth + 1)

    if item(0, 0) != n:
        raise ProfileError("trailing bytes")


def loads(data: bytes):
    """Decode untrusted CBOR within the ep1 profile; ProfileError (a COSEError) for anything else."""
    check_profile(data)
    try:
        return cbor2.loads(data)
    except Exception as exc:  # noqa: BLE001 - e.g. ill-formed UTF-8, which the scan does not judge
        raise ProfileError(f"not decodable: {exc}") from exc


def kid_of(public_key: Ed25519PublicKey) -> bytes:
    """Key identifier: SHA-256 of the raw public key, first 16 bytes."""
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return hashlib.sha256(raw).digest()[:16]


def _tbs(protected: bytes, payload: bytes, external_aad: bytes = b"") -> bytes:
    return dumps(["Signature1", protected, external_aad, payload])


def sign1(protected: dict, payload: bytes | None, key: Ed25519PrivateKey, *, unprotected: dict | None = None,
          detached: bool = False, external_aad: bytes = b"") -> bytes:
    """Tagged COSE_Sign1. With ``detached`` the payload is signed but carried as nil."""
    if payload is None:
        raise COSEError("a payload is required to sign")
    prot = dumps(protected) if protected else b""
    sig = key.sign(_tbs(prot, payload, external_aad))
    return dumps(cbor2.CBORTag(COSE_SIGN1_TAG, [prot, unprotected or {}, None if detached else payload, sig]))


def decode1(data: bytes) -> tuple[dict, bytes, dict, bytes | None, bytes]:
    """(protected map, protected bytes, unprotected map, payload or None, signature)."""
    obj = loads(data)
    if isinstance(obj, cbor2.CBORTag):
        if obj.tag != COSE_SIGN1_TAG:
            raise COSEError(f"unexpected CBOR tag {obj.tag}")
        obj = obj.value
    # cbor2 returns tuples and frozendicts for items inside a tag
    if not (isinstance(obj, (list, tuple)) and len(obj) == 4 and isinstance(obj[0], bytes)
            and isinstance(obj[1], Mapping) and (obj[2] is None or isinstance(obj[2], bytes))
            and isinstance(obj[3], bytes)):
        raise COSEError("not a COSE_Sign1 structure")
    prot = loads(obj[0]) if obj[0] else {}
    if not isinstance(prot, Mapping):
        raise COSEError("protected header is not a map")
    return _thaw(prot), obj[0], _thaw(obj[1]), obj[2], obj[3]


def _thaw(v):
    if isinstance(v, Mapping):
        return {k: _thaw(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [_thaw(x) for x in v]
    return v


# ----------------------------------------------------------------------------------- COSE-HPKE (encryption)
# draft-ietf-cose-hpke-27 (12.09.2026), integrated encryption: COSE_Encrypt0, no recipients. The code points are
# not yet assigned by IANA; these are the draft's suggested values and change with it:
#   HPKE-3 = 41 (DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, AES-128-GCM), HPKE-4 = 42 (... ChaCha20-Poly1305),
#   "ek" (encapsulated key) = -4, unprotected. HPKE info = empty; AAD = Enc_structure ("Encrypt0").
HPKE_3, HPKE_4 = 41, 42
EK = -4
CONTENT_TYPE = 3
COSE_ENCRYPT0_TAG = 16
KTY, KTY_OKP, CRV, CRV_X25519, OKP_X, OKP_D = 1, 1, -1, 4, -2, -4


def _hpke_aead(alg: int) -> int:
    from . import hpke
    return {HPKE_3: hpke.AEAD_AES_128_GCM, HPKE_4: hpke.AEAD_CHACHA20_POLY1305}[alg]


def _enc_structure(protected: bytes, external_aad: bytes) -> bytes:
    return dumps(["Encrypt0", protected, external_aad])


def x25519_cose_key(public: bytes, private: bytes | None = None) -> bytes:
    """COSE_Key (RFC 9053 OKP, crv X25519) of an HPKE recipient; with ``private`` the private key too."""
    key = {KTY: KTY_OKP, CRV: CRV_X25519, OKP_X: public}
    if private is not None:
        key[OKP_D] = private
    return dumps(key)


def x25519_from_cose_key(data: bytes) -> tuple[bytes, bytes | None]:
    key = loads(data)
    if not isinstance(key, Mapping) or key.get(KTY) != KTY_OKP or key.get(CRV) != CRV_X25519:
        raise COSEError("not an X25519 COSE_Key")
    return key[OKP_X], key.get(OKP_D)


def encrypt0_hpke(protected: dict, plaintext: bytes, recipient_public: bytes, *, alg: int = HPKE_4,
                  external_aad: bytes = b"", sk_e: bytes | None = None) -> bytes:
    """Tagged COSE_Encrypt0 with COSE-HPKE integrated encryption to one recipient's X25519 public key.
    ``sk_e`` fixes the ephemeral key (test vectors only)."""
    from . import hpke
    prot = dumps({**protected, ALG: alg})
    enc, ct = hpke.seal(recipient_public, b"", _enc_structure(prot, external_aad), plaintext, _hpke_aead(alg),
                        sk_e=sk_e)
    return dumps(cbor2.CBORTag(COSE_ENCRYPT0_TAG, [prot, {EK: enc}, ct]))


def decrypt0_hpke(data: bytes, recipient_private: bytes, *, external_aad: bytes = b"") -> tuple[dict, bytes]:
    """(protected header, plaintext) of a COSE-HPKE COSE_Encrypt0; COSEError if it is not one or does not open."""
    from cryptography.exceptions import InvalidTag

    from . import hpke
    obj = loads(data)
    if isinstance(obj, cbor2.CBORTag):
        if obj.tag != COSE_ENCRYPT0_TAG:
            raise COSEError(f"unexpected CBOR tag {obj.tag}")
        obj = obj.value
    if not (isinstance(obj, (list, tuple)) and len(obj) == 3 and isinstance(obj[0], bytes)
            and isinstance(obj[1], Mapping) and isinstance(obj[2], bytes)):
        raise COSEError("not a COSE_Encrypt0 structure")
    prot = _thaw(loads(obj[0])) if obj[0] else {}
    if not isinstance(prot, dict) or prot.get(ALG) not in (HPKE_3, HPKE_4) or not isinstance(obj[1].get(EK), bytes):
        raise COSEError("not COSE-HPKE integrated encryption")
    try:
        plaintext = hpke.open_(obj[1][EK], recipient_private, b"", _enc_structure(obj[0], external_aad), obj[2],
                               _hpke_aead(prot[ALG]))
    except (InvalidTag, ValueError) as exc:
        raise COSEError("decryption failed (wrong recipient key or altered object)") from exc
    return prot, plaintext


def verify1(data: bytes, public_key: Ed25519PublicKey, *, payload: bytes | None = None,
            external_aad: bytes = b"") -> tuple[dict, dict, bytes]:
    """Verify a COSE_Sign1; ``payload`` supplies a detached payload. Returns (protected, unprotected, payload)."""
    prot, prot_bytes, unprot, carried, sig = decode1(data)
    if prot.get(ALG) not in (ED25519, EDDSA):
        raise COSEError(f"unsupported algorithm {prot.get(ALG)!r}")
    body = carried if carried is not None else payload
    if body is None:
        raise COSEError("detached payload not supplied")
    if carried is not None and payload is not None and payload != carried:
        raise COSEError("supplied payload differs from the carried payload")
    try:
        public_key.verify(sig, _tbs(prot_bytes, body, external_aad))
    except InvalidSignature as exc:
        raise COSEError("signature does not verify") from exc
    return prot, unprot, body
