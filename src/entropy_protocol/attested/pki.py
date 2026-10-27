"""The authority's certificates: a root and statement-key certificates (Ed25519).

The root signs the attestation results itself (COSE) and issues one end-entity certificate per certified statement
key (subject = the target, validity = the result's lifetime, extended key usage id-kp-documentSigning, RFC 9336).
Results are not signed under an end-entity certificate: a sealer holds one, and a result that verified under any
leaf of the root could then be forged by a sealer for itself. A relying party pins the root; statement certificates
arrive in x5chain headers (RFC 9360, label 33) and are checked against it.
"""
from __future__ import annotations

import datetime as dt

from cryptography import x509
from cryptography.exceptions import InvalidSignature, UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.x509.oid import NameOID

STATEMENT_EKU = x509.ObjectIdentifier("1.3.6.1.5.5.7.3.36")    # id-kp-documentSigning (RFC 9336)


# What reading a certificate, CSR or key out of evidence can raise: besides ValueError, an unknown key algorithm and an
# invalid X.509 version, neither of which is a ValueError. Every verifier here turns them into its own refusal.
UNREADABLE = (ValueError, TypeError, UnsupportedAlgorithm, x509.InvalidVersion)

class PKIError(ValueError):
    """A certificate does not chain to the root, or is not valid for its use."""


def der(cert: x509.Certificate) -> bytes:
    return cert.public_bytes(serialization.Encoding.DER)


def _name(cn: str, org: str | None = None) -> x509.Name:
    attrs = [x509.NameAttribute(NameOID.COMMON_NAME, cn)]
    if org:
        attrs.append(x509.NameAttribute(NameOID.ORGANIZATION_NAME, org))
    return x509.Name(attrs)


def make_root(key: Ed25519PrivateKey, cn: str, org: str, years: int = 10) -> x509.Certificate:
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    name = _name(cn, org)
    return (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now)
            .not_valid_after(now + dt.timedelta(days=365 * years))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), critical=True)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
            .sign(key, None))


def issue(root_key: Ed25519PrivateKey, root: x509.Certificate, public_key: Ed25519PublicKey, cn: str, *,
          lifetime: int, eku: x509.ObjectIdentifier, not_before: dt.datetime | None = None) -> x509.Certificate:
    nb = (not_before or dt.datetime.now(dt.timezone.utc)).replace(microsecond=0)
    return (x509.CertificateBuilder().subject_name(_name(cn)).issuer_name(root.subject).public_key(public_key)
            .serial_number(x509.random_serial_number()).not_valid_before(nb)
            .not_valid_after(nb + dt.timedelta(seconds=int(lifetime)))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), critical=True)
            .add_extension(x509.ExtendedKeyUsage([eku]), critical=False)
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(public_key), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root.public_key()), critical=False)
            .sign(root_key, None))


def check_leaf(leaf_der: bytes, root: x509.Certificate, at: dt.datetime, eku: x509.ObjectIdentifier) -> Ed25519PublicKey:
    """The leaf is signed by the root, valid at ``at``, an end entity with ``eku``; returns its Ed25519 key."""
    try:
        leaf = x509.load_der_x509_certificate(leaf_der)
    except UNREADABLE as exc:
        raise PKIError(f"leaf is not a certificate: {exc}") from exc
    if leaf.issuer != root.subject:
        raise PKIError("leaf not issued by this root")
    try:
        leaf.verify_directly_issued_by(root)
    except (*UNREADABLE, InvalidSignature) as exc:
        raise PKIError("leaf signature does not verify under the root") from exc
    if not leaf.not_valid_before_utc <= at <= leaf.not_valid_after_utc:
        raise PKIError(f"leaf not valid at {at.isoformat()}")
    try:
        bc = leaf.extensions.get_extension_for_class(x509.BasicConstraints).value
        ekus = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        key = leaf.public_key()
    except (*UNREADABLE, x509.ExtensionNotFound) as exc:
        raise PKIError(f"leaf lacks what an end-entity certificate of this authority carries: {exc}") from exc
    if bc.ca:
        raise PKIError("leaf is a CA certificate")
    if eku not in ekus:
        raise PKIError("leaf not issued for this use")
    if not isinstance(key, Ed25519PublicKey):
        raise PKIError("leaf key is not Ed25519")
    return key


def make_csr(key: Ed25519PrivateKey, target: str) -> bytes:
    return der_csr(x509.CertificateSigningRequestBuilder().subject_name(_name(target)).sign(key, None))


def der_csr(csr: x509.CertificateSigningRequest) -> bytes:
    return csr.public_bytes(serialization.Encoding.DER)


def read_csr(csr_der: bytes) -> tuple[Ed25519PublicKey, str]:
    """(key, subject CN) of a CSR whose self-signature verifies (proof of possession)."""
    try:
        csr = x509.load_der_x509_csr(csr_der)
        valid = csr.is_signature_valid
        key = csr.public_key()
    except UNREADABLE as exc:
        raise PKIError(f"not a readable CSR: {exc}") from exc
    if not valid:
        raise PKIError("CSR signature does not verify (no proof of possession)")
    if not isinstance(key, Ed25519PublicKey):
        raise PKIError("CSR key is not Ed25519")
    cn = csr.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return key, (cn[0].value if cn else "")


def spki(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
