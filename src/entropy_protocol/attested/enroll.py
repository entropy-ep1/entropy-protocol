"""Attestation-based enrollment messages and the authority's handle book (ARCHITECTURE §7d).

Message shapes follow draft-novak-rats-tacra's wire schema (CBOR maps, text keys):
  initiation  {freshness_kind: "present-nonce", handle: bstr(32), rrp_id, expires_in, mode: "enroll",
               acceptable_csk: ["ed25519"], acceptable_evidence: [profile]}
  request     {freshness_kind, target, credential_type, handle, evidence: bstr, profile,
               binding: {method: "binding-input", hash: "sha512"}, csr: bstr (PKCS#10 DER)}
  evidence    (profile "ep1/azure-cgpu/v1", CBOR) {hcl: bstr, thim: {vcekCert, certificateChain, tcbm},
               quote: {msg, sig, pcrs: bstr, sel: [uint]}, eventlog: bstr, ? ima: bstr, ? akcert: bstr,
               gpus: [{report, chain, uuid, vbios, driver, cc, ready}],
               claims: {weights: bstr, tokenizer: bstr, code: bstr, class: tstr, params: bstr}}
The handle book is the authority's: a handle is issued once, expires, and is spent by the first enrollment that
presents it, whatever the outcome - a refused request cannot be replayed with corrections under the same handle.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cbor2

from .. import cose
from .binding import HANDLE_LEN

PROFILE_AZURE_CGPU = "ep1/azure-cgpu/v1"
CRED_STATEMENT = "ep1-statement-key"
CRED_DETECTION = "ep1-detection-key"
CREDENTIAL_TYPES = (CRED_STATEMENT, CRED_DETECTION)


class EnrollError(ValueError):
    """A malformed message, or a handle that cannot be used."""


def initiation(handle: bytes, rrp_id: str, expires_in: int) -> bytes:
    return cose.dumps({"freshness_kind": "present-nonce", "handle": handle, "rrp_id": rrp_id,
                       "expires_in": int(expires_in), "mode": "enroll", "acceptable_csk": ["ed25519"],
                       "acceptable_evidence": [PROFILE_AZURE_CGPU]})


def _map(data: bytes) -> dict:
    try:
        m = cose._thaw(cbor2.loads(data))
    except Exception as exc:  # noqa: BLE001 - every decoding failure of untrusted input is the same refusal
        raise EnrollError(f"not CBOR ({type(exc).__name__})") from exc
    if not isinstance(m, dict):
        raise EnrollError("not a CBOR map")
    return m


def read_initiation(data: bytes) -> dict:
    m = _map(data)
    if m.get("freshness_kind") != "present-nonce" or m.get("mode") != "enroll":
        raise EnrollError("not an enrollment initiation with a nonce")
    if not isinstance(m.get("handle"), bytes) or len(m["handle"]) != HANDLE_LEN:
        raise EnrollError("handle missing or not 32 octets")
    return m


def request(*, target: str, credential_type: str, handle: bytes, evidence: bytes, csr: bytes,
            profile: str = PROFILE_AZURE_CGPU) -> bytes:
    if credential_type not in CREDENTIAL_TYPES:
        raise EnrollError(f"unknown credential type {credential_type}")
    return cose.dumps({"freshness_kind": "present-nonce", "target": target, "credential_type": credential_type,
                       "handle": handle, "evidence": evidence, "profile": profile,
                       "binding": {"method": "binding-input", "hash": "sha512"}, "csr": csr})


def read_request(data: bytes) -> dict:
    m = _map(data)
    need = {"freshness_kind": str, "target": str, "credential_type": str, "handle": bytes, "evidence": bytes,
            "profile": str, "binding": dict, "csr": bytes}
    for k, t in need.items():
        if not isinstance(m.get(k), t):
            raise EnrollError(f"request field {k} missing or of the wrong type")
    if m["binding"] != {"method": "binding-input", "hash": "sha512"}:
        raise EnrollError("unsupported binding method")
    if m["credential_type"] not in CREDENTIAL_TYPES:
        raise EnrollError(f"unknown credential type {m['credential_type']}")
    return m


class HandleBook:
    """Issued handles in a JSON file: {hex: {"issued": t, "expires": t, "spent": t | null}}."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def _save(self, book: dict) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(book, indent=1, sort_keys=True))
        os.replace(tmp, self.path)

    def issue(self, ttl: int, now: float | None = None) -> bytes:
        now = time.time() if now is None else now
        h = os.urandom(HANDLE_LEN)
        book = self._load()
        book[h.hex()] = {"issued": int(now), "expires": int(now + ttl), "spent": None}
        self._save(book)
        return h

    def spend(self, handle: bytes, now: float | None = None) -> dict:
        """Mark the handle spent; EnrollError if it was never issued, has expired or was already spent."""
        now = time.time() if now is None else now
        book = self._load()
        e = book.get(handle.hex())
        if e is None:
            raise EnrollError("handle was not issued by this authority")
        if e["spent"] is not None:
            raise EnrollError("handle already used")
        e["spent"] = int(now)
        self._save(book)
        if now > e["expires"]:
            raise EnrollError("handle expired")
        return e
