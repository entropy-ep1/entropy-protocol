"""HPKE (RFC 9180), base mode, single-shot: DHKEM(X25519, HKDF-SHA256), HKDF-SHA256, and AES-128-GCM or
ChaCha20-Poly1305. Primitives from `cryptography`; the KEM derivation and key schedule follow RFC 9180 §4.1,
§5.1 and §7.1.3 and are checked against the RFC's test vectors A.1.1 and A.2.1.

Used to encrypt a record to one verifier's public key (COSE-HPKE integrated encryption, cose.py), so that the
issuer and the verifier share no symmetric secret.
"""
from __future__ import annotations

import hashlib
import hmac
import os

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

KEM_X25519_SHA256 = 0x0020
KDF_HKDF_SHA256 = 0x0001
AEAD_AES_128_GCM = 0x0001
AEAD_CHACHA20_POLY1305 = 0x0003
_AEAD = {AEAD_AES_128_GCM: (16, 12, AESGCM), AEAD_CHACHA20_POLY1305: (32, 12, ChaCha20Poly1305)}   # Nk, Nn, cipher
_KEM_SUITE = b"KEM" + KEM_X25519_SHA256.to_bytes(2, "big")


def _extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt or b"\x00" * 32, ikm, hashlib.sha256).digest()


def _expand(prk: bytes, info: bytes, length: int) -> bytes:
    out, block, i = b"", b"", 1
    while len(out) < length:
        block = hmac.new(prk, block + info + bytes([i]), hashlib.sha256).digest()
        out += block
        i += 1
    return out[:length]


def _labeled_extract(suite: bytes, salt: bytes, label: bytes, ikm: bytes) -> bytes:
    return _extract(salt, b"HPKE-v1" + suite + label + ikm)


def _labeled_expand(suite: bytes, prk: bytes, label: bytes, info: bytes, length: int) -> bytes:
    return _expand(prk, length.to_bytes(2, "big") + b"HPKE-v1" + suite + label + info, length)


def _raw_public(sk: X25519PrivateKey) -> bytes:
    return sk.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def derive_keypair(ikm: bytes) -> tuple[bytes, bytes]:
    """DeriveKeyPair (RFC 9180 §7.1.3) -> (private key bytes, public key bytes)."""
    dkp_prk = _labeled_extract(_KEM_SUITE, b"", b"dkp_prk", ikm)
    sk = _labeled_expand(_KEM_SUITE, dkp_prk, b"sk", b"", 32)
    return sk, _raw_public(X25519PrivateKey.from_private_bytes(sk))


def generate_keypair() -> tuple[bytes, bytes]:
    return derive_keypair(os.urandom(32))


def _shared_secret(dh: bytes, kem_context: bytes) -> bytes:
    eae_prk = _labeled_extract(_KEM_SUITE, b"", b"eae_prk", dh)
    return _labeled_expand(_KEM_SUITE, eae_prk, b"shared_secret", kem_context, 32)


def encap(pk_r: bytes, sk_e: bytes | None = None) -> tuple[bytes, bytes]:
    """(shared_secret, enc). ``sk_e`` fixes the ephemeral key (test vectors only)."""
    e = X25519PrivateKey.from_private_bytes(sk_e) if sk_e is not None else X25519PrivateKey.generate()
    enc = _raw_public(e)
    dh = e.exchange(X25519PublicKey.from_public_bytes(pk_r))
    return _shared_secret(dh, enc + pk_r), enc


def decap(enc: bytes, sk_r: bytes) -> bytes:
    r = X25519PrivateKey.from_private_bytes(sk_r)
    dh = r.exchange(X25519PublicKey.from_public_bytes(enc))
    return _shared_secret(dh, enc + _raw_public(r))


def key_schedule(shared_secret: bytes, info: bytes, aead_id: int) -> dict:
    """KeySchedule for mode_base (no PSK): key, base_nonce, exporter_secret and the intermediate values."""
    nk, nn, _ = _AEAD[aead_id]
    suite = b"HPKE" + b"".join(x.to_bytes(2, "big") for x in (KEM_X25519_SHA256, KDF_HKDF_SHA256, aead_id))
    context = b"\x00" + _labeled_extract(suite, b"", b"psk_id_hash", b"") + _labeled_extract(suite, b"", b"info_hash", info)
    secret = _labeled_extract(suite, shared_secret, b"secret", b"")
    return {"key_schedule_context": context, "secret": secret,
            "key": _labeled_expand(suite, secret, b"key", context, nk),
            "base_nonce": _labeled_expand(suite, secret, b"base_nonce", context, nn),
            "exporter_secret": _labeled_expand(suite, secret, b"exp", context, 32)}


def seal(pk_r: bytes, info: bytes, aad: bytes, plaintext: bytes, aead_id: int = AEAD_CHACHA20_POLY1305,
         sk_e: bytes | None = None) -> tuple[bytes, bytes]:
    """Single-shot SealBase -> (enc, ciphertext)."""
    shared, enc = encap(pk_r, sk_e)
    ks = key_schedule(shared, info, aead_id)
    return enc, _AEAD[aead_id][2](ks["key"]).encrypt(ks["base_nonce"], plaintext, aad)


def open_(enc: bytes, sk_r: bytes, info: bytes, aad: bytes, ciphertext: bytes,
          aead_id: int = AEAD_CHACHA20_POLY1305) -> bytes:
    """Single-shot OpenBase; raises cryptography.exceptions.InvalidTag on any mismatch."""
    ks = key_schedule(decap(enc, sk_r), info, aead_id)
    return _AEAD[aead_id][2](ks["key"]).decrypt(ks["base_nonce"], ciphertext, aad)
