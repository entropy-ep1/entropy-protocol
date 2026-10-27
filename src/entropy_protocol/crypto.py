"""Keys, keyed bitstream and the encrypted log token (ARCHITECTURE §3-4).

Key hierarchy (HKDF-SHA256, domain-separated):
    K                      master secret of the generating party (32 bytes)
    K_bits  = KDF(K, "ep1/bits")                  governs token selection; never leaves the generator
    K_doc   = KDF(K, "ep1/doc|" + doc_id)
    K_log   = KDF(K_doc, "ep1/log")               custody key of one document's record
    K_ver   = KDF(K_doc, "ep1/verifier|" + vid)   verifier-scoped key: decrypts one document's record
                                                  for one verifier; K, K_doc, K_bits not derivable.
Log token: "ep1." + base64url(CBOR{v, doc, n, ct}), ChaCha20-Poly1305, AAD = CBOR{v, doc}.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

import cbor2
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import cose

VERSION = 1
PREFIX = "ep1."


class TokenError(Exception):
    """The log token is malformed, was altered, or the key does not open it."""


def new_master_key() -> bytes:
    return os.urandom(32)


def new_doc_id() -> str:
    return base64.urlsafe_b64encode(os.urandom(12)).decode().rstrip("=")


def hkdf(key: bytes, info: str, length: int = 32) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info.encode()).derive(key)


def bits_key(master: bytes) -> bytes:
    return hkdf(master, "ep1/bits")


def doc_key(master: bytes, doc_id: str) -> bytes:
    return hkdf(master, f"ep1/doc|{doc_id}")


def log_key(master: bytes, doc_id: str) -> bytes:
    return hkdf(doc_key(master, doc_id), "ep1/log")


def verifier_key(master: bytes, doc_id: str, verifier_id: str) -> bytes:
    return hkdf(doc_key(master, doc_id), f"ep1/verifier|{verifier_id}")


def bit(k_bits: bytes, doc_id: str, position: int, low_share: float = 0.5) -> int:
    """Keyed bitstream: one bit per generation position, distinct per document.
    0 (first predicate, LOW) with probability ``low_share``; the default 0.5 is an unbiased bit."""
    d = hmac.new(k_bits, f"{doc_id}:{position}".encode(), hashlib.sha256).digest()
    if low_share == 0.5:
        return d[0] & 1
    return 0 if int.from_bytes(d[1:3], "big") < round(low_share * 65536) else 1


def wm_key(master: bytes) -> bytes:
    """Detection key of the record-free (watermark) mode. Holding it lets a party detect and, as with every
    keyed watermark, imitate the pattern; it opens no record and derives no other key."""
    return hkdf(master, "ep1/wm")


def wm_decide(k_wm: bytes, context: tuple[int, ...] | list[int], interval: int) -> int | None:
    """Content-keyed checkpoint: one PRF call on the preceding token ids decides whether this position is a
    checkpoint (probability 1/interval) and which predicate it carries (LOW or HIGH, probability 1/2 each).
    A detector recomputes it from the text alone, so edits desynchronise only the positions whose context
    they touch."""
    msg = b"ep1/wm|" + b"".join(int(t).to_bytes(4, "big") for t in context)
    d = hmac.new(k_wm, msg, hashlib.sha256).digest()
    if int.from_bytes(d[:4], "big") % interval:
        return None
    return d[4] & 1          # 0 = LOW, 1 = HIGH (record.LOW / record.HIGH)


def _aad(doc_id: str) -> bytes:
    return cbor2.dumps({"v": VERSION, "doc": doc_id}, canonical=True)


def encrypt_token(key: bytes, doc_id: str, plaintext: bytes, nonce: bytes | None = None) -> str:
    """``nonce`` (12 bytes) is random unless given; a fixed nonce is for test vectors only."""
    nonce = nonce if nonce is not None else os.urandom(12)
    if len(nonce) != 12:
        raise ValueError("nonce must be 12 bytes")
    ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, _aad(doc_id))
    env = cbor2.dumps({"v": VERSION, "doc": doc_id, "n": nonce, "ct": ct}, canonical=True)
    return PREFIX + base64.urlsafe_b64encode(env).decode().rstrip("=")


def _envelope(token: str) -> dict:
    if not isinstance(token, str) or not token.startswith(PREFIX):
        raise TokenError("not an ep1 log token")
    body = token[len(PREFIX):]
    try:
        env = cose.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except Exception as exc:  # noqa: BLE001
        raise TokenError("malformed log token") from exc
    if not isinstance(env, dict) or env.get("v") != VERSION or not {"doc", "n", "ct"} <= env.keys():
        raise TokenError("unsupported log token")
    return env


def token_doc_id(token: str) -> str:
    return _envelope(token)["doc"]


def decrypt_token(key: bytes, token: str) -> bytes:
    env = _envelope(token)
    try:
        return ChaCha20Poly1305(key).decrypt(env["n"], env["ct"], _aad(env["doc"]))
    except (InvalidTag, ValueError) as exc:
        raise TokenError("log token could not be opened with this key") from exc


def reissue_for_verifier(master: bytes, custody_token: str, verifier_id: str) -> tuple[str, bytes]:
    """Re-encrypt a document's record for one verifier; return (token, verifier-scoped key)."""
    doc_id = token_doc_id(custody_token)
    plaintext = decrypt_token(log_key(master, doc_id), custody_token)
    k_ver = verifier_key(master, doc_id, verifier_id)
    return encrypt_token(k_ver, doc_id, plaintext), k_ver


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()
