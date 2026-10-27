"""NVIDIA confidential GPU (Hopper): the SPDM measurement report and its certificate chain.

What NVML returns (NVIDIA/nvtrust local GPU verifier, `attestation/`): the SPDM 1.1 GET_MEASUREMENTS request (37
octets: version, code 0xE0, param1, param2, nonce 32, slot) followed by the MEASUREMENTS response: version, code
0x60, param1, param2, number of blocks, record length (3, little-endian), the measurement blocks {index,
specification (1 = DMTF), size (2), DMTF {value type, value size (2), value}}, the nonce (32), opaque length (2),
opaque data (TLVs: type 2, size 2, value), and the signature: ECDSA P-384, r || s, over SHA-384 of the request
followed by the response without its signature.
The certificate chain (PEM, leaf first) ends at NVIDIA's Device Identity CA, pinned here by SHA-256 of its DER (from
nvtrust tag 2026.06.04.001, `certs/verifier_device_root.pem`); reference integrity manifests are signed under
NVIDIA's RIM root, pinned the same way (`certs/verifier_RIM_root.pem`).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from .pki import UNREADABLE


DEVICE_ROOT_SHA256 = "102bf659d5419614c9d8e6aecebc80454eb26b1df6a769ac720b9a690b167b48"
RIM_ROOT_SHA256 = "12977b5115acb0381179279fffeb5a8c4d264971ebb32298023a465fa41df5d1"
REQUEST_LEN = 37
SIG_LEN = 96
OPAQUE = {1: "cert_issuer_name", 2: "cert_authority_key_identifier", 3: "driver_version", 4: "gpu_info", 5: "sku",
          6: "vbios_version", 7: "manufacturer_id", 8: "tamper_detection", 9: "smc", 10: "vpr", 11: "nvdec0_status",
          12: "msrscnt", 13: "cprinfo", 14: "board_id", 15: "chip_sku", 16: "chip_sku_mod", 17: "project",
          18: "project_sku", 19: "project_sku_mod", 20: "fwid", 21: "protected_pcie_status", 22: "switch_pdi",
          23: "floorswept_ports", 24: "position_id", 25: "lock_switch_status", 32: "gpu_link_conn",
          33: "sys_enable_status", 34: "opaque_data_version", 35: "chip_info", 36: "feature_flag"}


class GPUError(ValueError):
    """The GPU report or its chain does not hold."""


@dataclass
class GPUReport:
    request: bytes
    response: bytes
    nonce: bytes            # from the request
    response_nonce: bytes   # the responder's own nonce
    measurements: dict      # index -> value
    opaque: dict            # name -> raw value
    signature: bytes


def parse(report: bytes) -> GPUReport:
    """GPUError for anything that is not a well-formed GET_MEASUREMENTS exchange (every length is checked against the
    bytes there are)."""
    try:
        return _parse(report)
    except IndexError as exc:
        raise GPUError("GPU report truncated inside a field") from exc


def _parse(report: bytes) -> GPUReport:
    if len(report) <= REQUEST_LEN + SIG_LEN:
        raise GPUError("report too short")
    req, resp = report[:REQUEST_LEN], report[REQUEST_LEN:]
    if req[1] != 0xE0 or resp[1] != 0x60:
        raise GPUError("not a GET_MEASUREMENTS exchange")
    nblocks = resp[4]
    rec_len = int.from_bytes(resp[5:8], "little")
    if 8 + rec_len > len(resp):
        raise GPUError("measurement record longer than the response")
    rec, off, meas = resp[8:8 + rec_len], 0, {}
    for _ in range(nblocks):
        if off + 4 > rec_len:
            raise GPUError("measurement block header beyond the record")
        idx, spec, size = rec[off], rec[off + 1], int.from_bytes(rec[off + 2:off + 4], "little")
        if off + 4 + size > rec_len or size < 3:
            raise GPUError("measurement block beyond the record")
        body = rec[off + 4:off + 4 + size]
        if spec != 1:
            raise GPUError("measurement block not in DMTF format")
        vsize = int.from_bytes(body[1:3], "little")
        if 3 + vsize > size:
            raise GPUError("measurement value beyond its block")
        meas[idx] = body[3:3 + vsize]
        off += 4 + size
    if off != rec_len:
        raise GPUError("measurement record length does not match its blocks")
    p = 8 + rec_len
    if p + 34 > len(resp):
        raise GPUError("response ends before its nonce and opaque length")
    rnonce = resp[p:p + 32]
    olen = int.from_bytes(resp[p + 32:p + 34], "little")
    odata = resp[p + 34:p + 34 + olen]
    sig = resp[p + 34 + olen:]
    if len(sig) != SIG_LEN:
        raise GPUError(f"signature is {len(sig)} octets, expected {SIG_LEN}")
    opaque, o = {}, 0
    while o + 4 <= len(odata):
        t, n = int.from_bytes(odata[o:o + 2], "little"), int.from_bytes(odata[o + 2:o + 4], "little")
        opaque[OPAQUE.get(t, f"type_{t}")] = odata[o + 4:o + 4 + n]
        o += 4 + n
    return GPUReport(request=req, response=resp, nonce=req[4:36], response_nonce=rnonce, measurements=meas,
                     opaque=opaque, signature=sig)


def text(v: bytes) -> str:
    return v.split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()


def vbios_version(raw: bytes) -> str:
    """The opaque VBIOS field as NVIDIA prints it (nvtrust `format_vbios_version`): read little-endian, then the upper
    half followed by the byte just below it, dotted - 009f009604000000 -> 96.00.9f.00.04."""
    v = raw[::-1].hex()
    n = len(v) // 2
    t = v[n:] + v[n - 2:n]
    return ".".join(t[i:i + 2] for i in range(0, len(t), 2))


TCG_DICE_TCBINFO = "2.23.133.5.4.1"


def certificate_fwid(leaf: x509.Certificate) -> bytes:
    """The firmware ID the device certificate carries: the last FWID digest (SHA-384, 48 octets) of its TCG DICE
    TcbInfo extension - what NVIDIA's verifier reads (the extension's last 48 octets), with the OCTET STRING header
    in front of them checked as well."""
    for ext in leaf.extensions:
        if ext.oid.dotted_string in (TCG_DICE_TCBINFO, TCG_DICE_TCBINFO + ".1"):
            raw = ext.value.value
            if len(raw) >= 50 and raw[-50:-48] == b"\x04\x30":
                return raw[-48:]
            raise GPUError("TcbInfo extension does not end with a 48-octet FWID")
    raise GPUError("device certificate has no TcbInfo extension")


def verify_report(rep: GPUReport, leaf: x509.Certificate) -> None:
    try:
        key = leaf.public_key()
    except UNREADABLE as exc:
        raise GPUError(f"the attestation key cannot be read: {exc}") from exc
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp384r1":
        raise GPUError("the attestation key is not ECDSA P-384")
    r, s = int.from_bytes(rep.signature[:48], "big"), int.from_bytes(rep.signature[48:], "big")
    try:
        key.verify(encode_dss_signature(r, s), rep.request + rep.response[:-SIG_LEN], ec.ECDSA(hashes.SHA384()))
    except InvalidSignature as exc:
        raise GPUError("GPU report signature does not verify") from exc


def load_chain(pem: bytes) -> list[x509.Certificate]:
    return x509.load_pem_x509_certificates(pem)


def verify_chain(chain: list[x509.Certificate], root_sha256: str, at) -> None:
    """Leaf first; every certificate signed by the next (ECDSA), the last the pinned root, all valid at ``at``."""
    if not chain:
        raise GPUError("empty certificate chain")
    root = chain[-1]
    if hashlib.sha256(root.public_bytes(serialization.Encoding.DER)).hexdigest() != root_sha256:
        raise GPUError("chain does not end at NVIDIA's pinned root")
    for child, parent in zip(chain, chain[1:] + [root]):
        try:
            child.verify_directly_issued_by(parent)
        except (*UNREADABLE, InvalidSignature) as exc:                     # an unreadable key included
            raise GPUError(f"{child.subject.rfc4514_string()} is not signed by its issuer") from exc
        if not child.not_valid_before_utc <= at <= child.not_valid_after_utc:
            raise GPUError(f"{child.subject.rfc4514_string()} not valid at {at.isoformat()}")
