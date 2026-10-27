"""The binding value an enrollment puts into the hardware evidence (ARCHITECTURE §7d).

SHA-512 over length-prefixed fields, each a 4-octet big-endian length followed by the octets, in this order:
  "ep1/attested-binding/v1", the freshness handle, rrp_id, target, SHA-256 of the CSR (DER), the statement key
  (SubjectPublicKeyInfo DER), the weights digest, the tokenizer digest, the code identity, the sealing class,
  SHA-256 of the sealing parameters (canonical CBOR).
The 64 octets go where the platform lets the guest choose them: SEV-SNP REPORT_DATA, TDX REPORTDATA, or the Azure
vTPM guest input that the paravisor copies into its report's runtime claims. rrp_id comes from the sealer's own
configuration, never from the wire: only an identity the attester holds ties its evidence to one authority.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .. import cose

LABEL = b"ep1/attested-binding/v1"
HANDLE_LEN = 32


def lp(parts: list[bytes]) -> bytes:
    """Length-prefixed concatenation: 4-octet big-endian length, then the octets, for every part."""
    return b"".join(len(p).to_bytes(4, "big") + p for p in parts)


@dataclass(frozen=True)
class BindingInput:
    handle: bytes          # the authority's nonce, 32 octets
    rrp_id: str            # the authority, from the sealer's configuration
    target: str            # the sealing service being certified
    csr_digest: bytes      # SHA-256 of the certificate signing request, DER
    statement_key: bytes   # SubjectPublicKeyInfo DER of the key being certified
    weights: bytes         # weights digest, 32 octets
    tokenizer: bytes       # tokenizer digest, 32 octets
    code: bytes            # code identity, 32 octets (code_identity)
    sealing_class: str     # numerical class of the sealing pass
    params: bytes          # SHA-256 of the sealing parameters (params_digest)

    def fields(self) -> list[bytes]:
        if len(self.handle) != HANDLE_LEN:
            raise ValueError(f"handle must be {HANDLE_LEN} octets, got {len(self.handle)}")
        for name in ("csr_digest", "weights", "tokenizer", "code", "params"):
            if len(getattr(self, name)) != 32:
                raise ValueError(f"{name} must be 32 octets")
        return [LABEL, self.handle, self.rrp_id.encode(), self.target.encode(), self.csr_digest, self.statement_key,
                self.weights, self.tokenizer, self.code, self.sealing_class.encode(), self.params]

    def value(self) -> bytes:
        return hashlib.sha512(lp(self.fields())).digest()


def params_digest(params: dict) -> bytes:
    """SHA-256 of the sealing parameters as canonical CBOR (integers and text only: no floating point on the wire)."""
    return hashlib.sha256(cose.dumps(params)).digest()


def package_files(root: Path | None = None) -> list[list]:
    """[[relative path, SHA-256], ...] of every source file of the installed package, sorted by path."""
    root = root or Path(__file__).resolve().parents[1]
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".cddl"):
            out.append([p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).digest()])
    return out


def code_identity(root: Path | None = None) -> bytes:
    """SHA-256 of the canonical CBOR list of package files: the identity of the code that seals."""
    return hashlib.sha256(cose.dumps(package_files(root))).digest()
