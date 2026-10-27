"""OCSP for NVIDIA's certificate chains (RFC 6960), as NVIDIA's local verifier asks it (nvtrust `cc_admin_utils.py`):
one request per certificate below the root (for a GPU chain, from the second certificate: the leaf is the per-boot
attestation key), CertID over SHA-384, a critical nonce extension; responder https://ocsp.ndis.nvidia.com/.
A response counts when it is successful, signed by a responder whose certificate chains to the pinned root, echoes
the nonce (NVIDIA puts it in responseExtensions), names the asked certificate, and its validity window covers the
time of the check. The raw responses are kept: the offline check re-reads them at the time of the original appraisal.
"""
from __future__ import annotations

import datetime as dt
import os
import urllib.request

from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509 import ocsp

from .nvidia import GPUError, verify_chain

OCSP_URL = "https://ocsp.ndis.nvidia.com/"
SKEW = dt.timedelta(seconds=300)    # tolerated clock difference with the responder (as `openssl ocsp -validity_period`)


def request(cert: x509.Certificate, issuer: x509.Certificate, nonce: bytes) -> bytes:
    b = ocsp.OCSPRequestBuilder().add_certificate(cert, issuer, hashes.SHA384())
    return b.add_extension(x509.OCSPNonce(nonce), critical=True).build().public_bytes(serialization.Encoding.DER)


def post(req: bytes, url: str = OCSP_URL, timeout: int = 20) -> bytes:
    r = urllib.request.Request(url, data=req, headers={"Content-Type": "application/ocsp-request"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        return resp.read()


def _responder_chain(given: list[x509.Certificate], known: list[x509.Certificate]) -> list[x509.Certificate]:
    """The responder's certificates as the response gives them, completed upward from ``known`` (the chain being
    checked, which ends at the pinned root) by issuer name."""
    chain = list(given)
    while chain[-1].issuer != chain[-1].subject:
        up = next((c for c in known if c.subject == chain[-1].issuer), None)
        if up is None or up in chain:
            break
        chain.append(up)
    return chain


def check(resp_der: bytes, cert: x509.Certificate, issuer: x509.Certificate, nonce: bytes, root_sha256: str,
          at: dt.datetime, known: list[x509.Certificate]) -> str:
    """"good", or GPUError naming what failed ("revoked: <reason>" is an error too)."""
    r = ocsp.load_der_ocsp_response(resp_der)
    if r.response_status != ocsp.OCSPResponseStatus.SUCCESSFUL:
        raise GPUError(f"OCSP response status {r.response_status.name}")
    if not r.certificates:
        raise GPUError("OCSP response carries no responder certificate")
    responder = r.certificates[0]
    verify_chain(_responder_chain(list(r.certificates), known), root_sha256, at)
    try:
        responder.public_key().verify(r.signature, r.tbs_response_bytes, ec.ECDSA(r.signature_hash_algorithm))
    except InvalidSignature as exc:
        raise GPUError("OCSP response signature does not verify") from exc
    try:
        echoed = r.extensions.get_extension_for_class(x509.OCSPNonce).value.nonce
    except x509.ExtensionNotFound as exc:
        raise GPUError("OCSP response does not echo the nonce") from exc
    if echoed != nonce:
        raise GPUError("OCSP nonce mismatch")
    asked = ocsp.load_der_ocsp_request(request(cert, issuer, nonce))
    if (r.serial_number, r.issuer_key_hash, r.issuer_name_hash) != (asked.serial_number, asked.issuer_key_hash,
                                                                     asked.issuer_name_hash):
        raise GPUError("OCSP response is about another certificate (CertID differs)")
    upd, nxt = r.this_update_utc, r.next_update_utc
    if upd > at + SKEW or (nxt is not None and at > nxt + SKEW):
        raise GPUError(f"OCSP response not current at {at.isoformat()} (this update {upd}, next {nxt})")
    if r.certificate_status == ocsp.OCSPCertStatus.GOOD:
        return "good"
    if r.certificate_status == ocsp.OCSPCertStatus.REVOKED:
        raise GPUError(f"revoked: {r.revocation_reason.name if r.revocation_reason else 'unspecified'}")
    raise GPUError("certificate status unknown to the responder")


def fresh_nonce() -> bytes:
    return os.urandom(32)
