"""Record delivery to a verifier by public key (COSE-HPKE integrated encryption, cose.py).

The issuer opens the record with its per-document log key and encrypts the record plaintext to the verifier's
X25519 public key. The verifier needs no secret shared with the issuer, and the issuer can deliver one record
to many verifiers without deriving or distributing symmetric keys. The document id travels in the protected
header (CWT claim sub), so it is authenticated with the record.
"""
from __future__ import annotations

from . import cose, crypto

CT_RECORD = "application/ep1-record+cbor"


def deliver(key: bytes, token: str, verifier_public: bytes, *, sk_e: bytes | None = None) -> bytes:
    """COSE_Encrypt0 of the record in ``token`` (opened with the log key ``key``) to ``verifier_public``."""
    doc_id = crypto.token_doc_id(token)
    plaintext = crypto.decrypt_token(key, token)
    return cose.encrypt0_hpke({cose.CONTENT_TYPE: CT_RECORD, cose.CWT_CLAIMS: {cose.SUB: doc_id}}, plaintext,
                              verifier_public, sk_e=sk_e)


def open_delivered(data: bytes, verifier_private: bytes) -> tuple[str, bytes]:
    """(document id, record plaintext); cose.COSEError if not a record for this key."""
    prot, plaintext = cose.decrypt0_hpke(data, verifier_private)
    if prot.get(cose.CONTENT_TYPE) != CT_RECORD:
        raise cose.COSEError("not an ep1 record")
    doc_id = (prot.get(cose.CWT_CLAIMS) or {}).get(cose.SUB)
    if not isinstance(doc_id, str) or not doc_id:
        raise cose.COSEError("the delivered record names no document")
    return doc_id, plaintext
