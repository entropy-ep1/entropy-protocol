"""The Azure paravisor's report (vTPM NV 0x01400001): HCL header, SNP report, request data, runtime claims.

Layout (microsoft/openvmm `vm/devices/tpm/tpm_guest_tests/src/report.rs`, `openhcl_attestation_protocol/src/
igvm_attest/get.rs`): header {signature "HCLA" (0x414c4348), version 2, report_size, request_type, status, 3 reserved}
32 octets; the TEE report in a 0x4a0-octet slot; request data {data_size, version, report_type (2 = SNP),
report_data_hash_type (1 SHA-256, 2 SHA-384, 3 SHA-512), variable_data_size} 20 octets; then the runtime claims, a
JSON object {keys: [HCLAkPub, HCLEkPub as RSA JWKs], vm-configuration: {...}, user-data: hex}. The hash of the
runtime claims, in the stated type, is the start of REPORT_DATA; the rest of REPORT_DATA is zero.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass

from . import amd

SIGNATURE = 0x414C4348
HEADER = 32
SLOT = 0x4A0
REQUEST_DATA = 20
HASHES = {1: hashlib.sha256, 2: hashlib.sha384, 3: hashlib.sha512}
SNP_VM_REPORT = 2


class HCLError(ValueError):
    """Not a paravisor report, or its parts do not hold together."""


@dataclass
class HCLReport:
    raw: bytes
    version: int
    report_size: int
    request_type: int
    status: int
    snp: amd.Report
    report_type: int
    hash_type: int
    claims_raw: bytes
    claims: dict

    def jwk(self, kid: str) -> dict | None:
        return next((k for k in self.claims.get("keys", []) if k.get("kid") == kid), None)

    @property
    def user_data(self) -> bytes:
        return bytes.fromhex(self.claims.get("user-data", ""))


def _u32(b: bytes, off: int) -> int:
    return int.from_bytes(b[off:off + 4], "little")


def parse(raw: bytes) -> HCLReport:
    if len(raw) < HEADER + SLOT + REQUEST_DATA or _u32(raw, 0) != SIGNATURE:
        raise HCLError("not an HCL report")
    version, size, rtype, status = (_u32(raw, o) for o in (4, 8, 12, 16))
    rd = HEADER + SLOT
    data_size, _rd_version, report_type, hash_type, var_size = (_u32(raw, rd + o) for o in (0, 4, 8, 12, 16))
    start = rd + REQUEST_DATA
    if data_size != REQUEST_DATA + var_size or start + var_size != size or size > len(raw):
        raise HCLError("sizes in the report do not agree")
    claims_raw = raw[start:start + var_size]
    try:
        claims = json.loads(claims_raw)
    except ValueError as exc:
        raise HCLError("runtime claims are not JSON") from exc
    return HCLReport(raw=raw[:size], version=version, report_size=size, request_type=rtype, status=status,
                     snp=amd.parse_report(raw[HEADER:HEADER + SLOT]), report_type=report_type, hash_type=hash_type,
                     claims_raw=claims_raw, claims=claims)


def report_data_binds_claims(h: HCLReport) -> bool:
    """REPORT_DATA = hash(runtime claims) in the stated type, zero after it."""
    fn = HASHES.get(h.hash_type)
    if fn is None:
        return False
    d = fn(h.claims_raw).digest()
    return h.snp.report_data[:len(d)] == d and not any(h.snp.report_data[len(d):])


def rsa_from_jwk(jwk: dict):
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers

    def b64(s: str) -> int:
        return int.from_bytes(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)), "big")
    if jwk.get("kty") != "RSA":
        raise HCLError("not an RSA JWK")
    return RSAPublicNumbers(b64(jwk["e"]), b64(jwk["n"])).public_key()
