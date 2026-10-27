"""AMD SEV-SNP: the attestation report and its certificate chain, checked from the specifications.

Report (SEV-SNP Firmware ABI, AMD 56860; offsets also in our library, attestation/platforms.md): 1184 octets, signed
body [0x000:0x2A0], ECDSA P-384 with SHA-384, r and s little-endian in 72-octet slots at 0x2A0 and 0x2E8.
Chain: VCEK (or VLEK) <- ASK (or ASVK) <- ARK, all RSASSA-PSS with SHA-384. The certificates encode the PSS trailer
field at its default value, which DER forbids: cryptography 42.0.8 refuses to load them, 43.0.3 and later load and
verify them (checked 27.10.2026 against AMD KDS's Genoa chain; hence the dependency floor). ARKs are pinned by SHA-256
of their DER, taken from AMD KDS (snapshot archived 30.09.2026 in the library; ARK-Milan matches the fingerprint
recorded there).
VCEK extensions (AMD 57230 rev 1.05, tables 10-11): structVersion .1.1, productName .1.2, TCB_VERSION elements
.1.3.x (blSPL 1, teeSPL 2, snpSPL 3, ucodeSPL 8, fmcSPL 9), hwID .1.4 (64 octets on Milan/Genoa, 8 from Turin).
"""
from __future__ import annotations

import datetime as dt
import hashlib
from dataclasses import dataclass

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from .pki import UNREADABLE

ARK_SHA256 = {
    "Milan": "69d063b45344d26a2e94e1f4210de49ef555308287d4c174445c95639a540bcd",
    "Genoa": "4c6598d19c18719c5dfd4a7d335f674e5bfe1d8f800cea2cf270c10d103db2f1",
    "Turin": "1f084161a44bb6d93778a904877d4819cafa5d05ef4193b2ded9dd9c73dd3f6a",
    "Venice": "46ef2eb7664175c7e9d3923ea0d275e572366d755556a30e229634592dce5903",
}
OID_RSASSA_PSS = "1.2.840.113549.1.1.10"
OID_ECDSA_SHA384 = "1.2.840.10045.4.3.3"
VCEK_OID = "1.3.6.1.4.1.3704.1."
REPORT_SIZE = 0x4A0
SIGNED = 0x2A0


class SNPError(ValueError):
    """The report or its chain does not hold."""


def verify_chain(leaf_der: bytes, ask_der: bytes, ark_der: bytes, product: str, at: dt.datetime) -> dict:
    """The leaf (VCEK or VLEK) to the pinned ARK of ``product``. Returns the leaf's AMD extensions; SNPError otherwise."""
    pin = ARK_SHA256.get(product)
    if pin is None:
        raise SNPError(f"no pinned ARK for {product}")
    if hashlib.sha256(ark_der).hexdigest() != pin:
        raise SNPError(f"ARK is not AMD's pinned ARK-{product}")
    try:
        leaf, ask, ark = (x509.load_der_x509_certificate(x) for x in (leaf_der, ask_der, ark_der))
    except UNREADABLE as exc:
        raise SNPError(f"not a DER certificate: {exc}") from exc
    for child, parent in ((ark, ark), (ask, ark), (leaf, ask)):
        if child.signature_algorithm_oid.dotted_string != OID_RSASSA_PSS:
            raise SNPError(f"unexpected signature algorithm {child.signature_algorithm_oid.dotted_string}")
        try:
            child.verify_directly_issued_by(parent)
        except (*UNREADABLE, InvalidSignature) as exc:                     # an unreadable key included
            raise SNPError(f"{child.subject.rfc4514_string()} is not signed by its issuer") from exc
    for c in (leaf, ask, ark):
        if not c.not_valid_before_utc <= at <= c.not_valid_after_utc:
            raise SNPError(f"certificate not valid at {at.isoformat()}")
    return vcek_extensions(leaf)


def _der_value(raw: bytes) -> tuple[int, bytes]:
    """(tag, contents) of one short DER element - the INTEGER, IA5String or OCTET STRING inside an AMD extension."""
    tag, ln, off = raw[0], raw[1], 2
    if ln & 0x80:
        n = ln & 0x7F
        ln, off = int.from_bytes(raw[2:2 + n], "big"), 2 + n
    return tag, raw[off:off + ln]


def vcek_extensions(leaf: x509.Certificate) -> dict:
    out: dict = {}
    names = {"1": "structVersion", "3.1": "blSPL", "3.2": "teeSPL", "3.3": "snpSPL", "3.8": "ucodeSPL",
             "3.9": "fmcSPL"}
    for ext in leaf.extensions:
        oid = ext.oid.dotted_string
        if not oid.startswith(VCEK_OID) or not isinstance(ext.value, x509.UnrecognizedExtension):
            continue
        tail = oid[len(VCEK_OID):]
        tag, val = _der_value(ext.value.value)
        if tail in names and tag == 0x02:
            out[names[tail]] = int.from_bytes(val, "big", signed=True)
        elif tail == "2" and tag == 0x16:
            out["productName"] = val.decode("ascii")
        elif tail == "4":
            out["hwID"] = val if tag == 0x04 else ext.value.value
    return out


@dataclass
class Report:
    raw: bytes
    version: int
    guest_svn: int
    policy: int
    vmpl: int
    signature_algo: int
    current_tcb: int
    platform_info: int
    flags: int
    report_data: bytes
    measurement: bytes
    host_data: bytes
    id_key_digest: bytes
    report_id: bytes
    reported_tcb: int
    chip_id: bytes
    committed_tcb: int
    launch_tcb: int
    family_id: bytes           # from the ID block, when one was given at launch
    image_id: bytes
    author_key_digest: bytes
    launch_mit_vector: int     # report version 5: 0x1F8
    current_mit_vector: int    # report version 5: 0x200
    r: int
    s: int

    @property
    def id_block(self) -> bool:
        """An ID block was given at launch: the firmware then refused to launch unless the block's expected launch
        digest equalled the measurement and the block's signature verified under the ID key whose SHA-384 digest the
        report carries (AMD 56860 rev. 1.59 §8.19.2, SNP_LAUNCH_FINISH)."""
        return any(self.id_key_digest)

    @property
    def alias_check_complete(self) -> bool:
        """PLATFORM_INFO bit 5: the platform checked DIMM aliasing (the BadRAM fix, AMD-SB-3015)."""
        return bool(self.platform_info >> 5 & 1)

    @property
    def debug(self) -> bool:
        return bool(self.policy >> 19 & 1)

    @property
    def signing_key(self) -> str:
        return {0: "vcek", 1: "vlek", 7: "none"}.get(self.flags >> 2 & 7, "reserved")


def _u(b: bytes, off: int, n: int) -> int:
    return int.from_bytes(b[off:off + n], "little")


def parse_report(raw: bytes) -> Report:
    if len(raw) < REPORT_SIZE:
        raise SNPError(f"report is {len(raw)} octets, expected {REPORT_SIZE}")
    raw = raw[:REPORT_SIZE]
    return Report(raw=raw, version=_u(raw, 0, 4), guest_svn=_u(raw, 4, 4), policy=_u(raw, 8, 8), vmpl=_u(raw, 0x30, 4),
                  signature_algo=_u(raw, 0x34, 4), current_tcb=_u(raw, 0x38, 8), platform_info=_u(raw, 0x40, 8),
                  flags=_u(raw, 0x48, 4), report_data=raw[0x50:0x90], measurement=raw[0x90:0xC0],
                  host_data=raw[0xC0:0xE0], id_key_digest=raw[0xE0:0x110], report_id=raw[0x140:0x160],
                  reported_tcb=_u(raw, 0x180, 8), chip_id=raw[0x1A0:0x1E0], committed_tcb=_u(raw, 0x1E0, 8),
                  launch_tcb=_u(raw, 0x1F0, 8), family_id=raw[0x10:0x20], image_id=raw[0x20:0x30],
                  author_key_digest=raw[0x110:0x140], launch_mit_vector=_u(raw, 0x1F8, 8) if _u(raw, 0, 4) >= 5 else 0,
                  current_mit_vector=_u(raw, 0x200, 8) if _u(raw, 0, 4) >= 5 else 0,
                  r=_u(raw, 0x2A0, 72), s=_u(raw, 0x2E8, 72))


def verify_report(rep: Report, leaf_der: bytes) -> None:
    """ECDSA P-384 / SHA-384 over the signed body, under the VCEK or VLEK."""
    if rep.signature_algo != 1:
        raise SNPError(f"signature algorithm {rep.signature_algo}, expected 1 (ECDSA P-384 with SHA-384)")
    try:
        key = x509.load_der_x509_certificate(leaf_der).public_key()
    except UNREADABLE as exc:
        raise SNPError(f"the report key cannot be read: {exc}") from exc
    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp384r1":
        raise SNPError("the report key is not ECDSA P-384")
    try:
        key.verify(encode_dss_signature(rep.r, rep.s), rep.raw[:SIGNED], ec.ECDSA(hashes.SHA384()))
    except InvalidSignature as exc:
        raise SNPError("report signature does not verify") from exc


def tcb_parts(tcb: int, product: str) -> dict:
    """TCB_VERSION decomposed by product layout (Milan/Genoa vs Turin and later)."""
    if product in ("Milan", "Genoa"):
        return {"blSPL": tcb & 0xFF, "teeSPL": tcb >> 8 & 0xFF, "snpSPL": tcb >> 48 & 0xFF, "ucodeSPL": tcb >> 56 & 0xFF}
    return {"fmcSPL": tcb & 0xFF, "blSPL": tcb >> 8 & 0xFF, "teeSPL": tcb >> 16 & 0xFF, "snpSPL": tcb >> 24 & 0xFF,
            "ucodeSPL": tcb >> 56 & 0xFF}


def tcb_at_least(tcb: dict, minimum: dict) -> bool:
    """Every component at or above the minimum: TCB components are compared one by one, never as one integer."""
    return all(tcb.get(k, -1) >= v for k, v in minimum.items())


KDS = "https://kdsintf.amd.com"


def crl_url(product: str, signer: str = "vcek") -> str:
    return f"{KDS}/{signer}/v1/{product}/crl"


def fetch_crl(product: str, signer: str = "vcek", timeout: int = 20) -> bytes:
    """AMD's CRL for the product's ASK (or ASVK), signed by the ARK. KDS rate-limits: fetch once per appraisal."""
    import urllib.request
    with urllib.request.urlopen(crl_url(product, signer), timeout=timeout) as r:  # noqa: S310 - fixed https URL
        return r.read()


def check_crl(crl_der: bytes, ask_der: bytes, ark_der: bytes, at: dt.datetime,
              skew: dt.timedelta = dt.timedelta(minutes=5)) -> None:
    """The CRL is the ARK's, current at ``at``, and does not list the ASK; SNPError otherwise."""
    try:
        crl = x509.load_der_x509_crl(crl_der)
        ask, ark = x509.load_der_x509_certificate(ask_der), x509.load_der_x509_certificate(ark_der)
        signed = crl.issuer == ark.subject and crl.is_signature_valid(ark.public_key())
    except UNREADABLE as exc:
        raise SNPError(f"CRL or certificates cannot be read: {exc}") from exc
    if not signed:
        raise SNPError("CRL not signed by the ARK")
    if crl.last_update_utc > at + skew or (crl.next_update_utc is not None and at > crl.next_update_utc + skew):
        raise SNPError(f"CRL not current at {at.isoformat()}")
    if crl.get_revoked_certificate_by_serial_number(ask.serial_number) is not None:
        raise SNPError("the ASK is revoked")
