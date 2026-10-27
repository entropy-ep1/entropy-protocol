"""RFC 3161 time stamps over a statement: "this statement existed before that minute", from an authority that is
not us. The token travels in the statement's unprotected header (RFC 9921 `3161-ttc`, label 269) or beside it in an
evidence package.

Scope, honestly: this module builds the request, reads the token's own claims (imprint, generation time, serial,
policy) and checks that the imprint is the digest we asked to stamp. It does NOT validate the authority's CMS
signature or its certificate chain - that needs an X.509 path and a trust list, which belong to the reader, not to
us. `entropy evidence --check` says so and prints the `openssl ts -verify` command for it.
"""
from __future__ import annotations

import os

SHA256_OID = bytes.fromhex("608648016503040201")           # 2.16.840.1.101.3.4.2.1
TSTINFO_OID = bytes.fromhex("2a864886f70d0109100104")      # 1.2.840.113549.1.9.16.1.4 (id-ct-TSTInfo)
CONTENT_TYPE = "application/timestamp-query"
REPLY_TYPE = "application/timestamp-reply"
COSE_3161_TTC = 269                                        # RFC 9921 header label for a 3161 token


def _len(n: int) -> bytes:
    if n < 0x80:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def _tlv(tag: int, body: bytes) -> bytes:
    return bytes([tag]) + _len(len(body)) + body


def _int(n: int) -> bytes:
    b = n.to_bytes(max(1, (n.bit_length() + 8) // 8), "big", signed=False)
    return _tlv(0x02, b)


def request(digest: bytes, *, nonce: bytes | None = None, cert_req: bool = True) -> bytes:
    """DER TimeStampReq for a SHA-256 digest (RFC 3161 §2.4.1). The nonce ties one reply to one request."""
    if len(digest) != 32:
        raise ValueError("a SHA-256 digest is 32 bytes")
    nonce = os.urandom(16) if nonce is None else nonce
    algorithm = _tlv(0x30, _tlv(0x06, SHA256_OID) + _tlv(0x05, b""))          # AlgorithmIdentifier, NULL params
    imprint = _tlv(0x30, algorithm + _tlv(0x04, digest))
    body = _int(1) + imprint + _int(int.from_bytes(nonce, "big"))
    if cert_req:
        body += _tlv(0x01, b"\xff")                                            # certReq TRUE: keep the cert
    return _tlv(0x30, body)


def _elements(der: bytes, start: int = 0, end: int | None = None):
    """(tag, element_start, body_start, body_end) of every element at this level."""
    end = len(der) if end is None else end
    i = start
    while i < end:
        head, tag = i, der[i]
        n = der[i + 1]
        i += 2
        if n & 0x80:
            k = n & 0x7F
            n = int.from_bytes(der[i:i + k], "big")
            i += k
        yield tag, head, i, i + n
        i += n


def _walk(der: bytes, start: int = 0, end: int | None = None):
    """(tag, body_start, body_end) of every element at this level."""
    for tag, _, s, e in _elements(der, start, end):
        yield tag, s, e


STATUS = {0: "granted", 1: "grantedWithMods", 2: "rejection", 3: "waiting", 4: "revocationWarning",
          5: "revocationNotification"}


def token_from_response(reply: bytes) -> bytes:
    """The TimeStampToken out of a TimeStampResp (RFC 3161 §2.4.2), refusing any status but granted. ValueError for
    anything that is not one."""
    try:
        return _token_from_response(reply)
    except ValueError:
        raise
    except (IndexError, TypeError) as exc:
        raise ValueError(f"malformed TimeStampResp ({type(exc).__name__})") from exc


def _token_from_response(reply: bytes) -> bytes:
    top = next(iter(_elements(reply)), None)
    if top is None or top[0] != 0x30:
        raise ValueError("not a DER TimeStampResp")
    parts = list(_elements(reply, top[2], top[3]))
    if not parts or parts[0][0] != 0x30:
        raise ValueError("TimeStampResp without PKIStatusInfo")
    status_int = next((reply[s:e] for t, _, s, e in _elements(reply, parts[0][2], parts[0][3]) if t == 0x02), None)
    status = int.from_bytes(status_int, "big") if status_int else -1
    if status not in (0, 1):
        text = next((reply[s:e].decode("utf-8", "replace") for t, _, s, e in
                     _elements(reply, parts[0][2], parts[0][3]) if t == 0x0C), "")
        raise ValueError(f"authority refused: status {status} ({STATUS.get(status, 'unknown')}) {text}".strip())
    if len(parts) < 2:
        raise ValueError("granted, but the reply carries no token")
    return reply[parts[1][1]:parts[1][3]]


def _find(der: bytes, oid: bytes, start: int = 0, end: int | None = None) -> int | None:
    """Offset just after the OBJECT IDENTIFIER with this value, anywhere in the structure."""
    end = len(der) if end is None else end
    for tag, s, e in _walk(der, start, end):
        if tag == 0x06 and der[s:e] == oid:
            return e
        if tag & 0x20 or tag in (0x30, 0x31):                                  # constructed: look inside
            hit = _find(der, oid, s, e)
            if hit is not None:
                return hit
    return None


def _first(der: bytes, tag_wanted: int, start: int, end: int) -> tuple[int, int] | None:
    for tag, s, e in _walk(der, start, end):
        if tag == tag_wanted:
            return s, e
        if tag & 0x20 or tag in (0x30, 0x31):
            hit = _first(der, tag_wanted, s, e)
            if hit is not None:
                return hit
    return None


def tst_info(token: bytes) -> bytes:
    """The DER TSTInfo carried by a TimeStampToken (CMS SignedData, eContent OCTET STRING), or the TSTInfo itself."""
    after = _find(token, TSTINFO_OID)
    if after is not None:
        oct_ = _first(token, 0x04, after, len(token))                          # [0] EXPLICIT eContent OCTET STRING
        if oct_:
            inner = token[oct_[0]:oct_[1]]
            seq = _first(inner, 0x30, 0, len(inner))
            if seq and seq[0] <= 4:                                            # the OCTET STRING wraps the SEQUENCE
                return inner
    return token


def token_info(token: bytes) -> dict:
    """What the token says about itself. TSTInfo fields are positional (RFC 3161 §2.4.2): version, policy,
    messageImprint, serialNumber, genTime. No signature check (see the module docstring). ValueError for anything
    that is not a time-stamp token."""
    try:
        return _token_info(token)
    except ValueError:
        raise
    except (IndexError, TypeError) as exc:
        raise ValueError(f"malformed time-stamp token ({type(exc).__name__})") from exc


def _token_info(token: bytes) -> dict:
    info = tst_info(token)
    seq = _first(info, 0x30, 0, len(info))
    if not seq:
        raise ValueError("not a DER structure")
    f = list(_walk(info, *seq))
    if len(f) < 5 or [t for t, _, _ in f[:5]] != [0x02, 0x06, 0x30, 0x02, 0x18]:
        raise ValueError("not a TSTInfo (version, policy, imprint, serial, genTime expected)")
    imp_alg = _first(info, 0x06, f[2][1], f[2][2])
    imp_val = _first(info, 0x04, f[2][1], f[2][2])
    return {"version": int.from_bytes(info[f[0][1]:f[0][2]], "big"),
            "policy": ".".join(str(x) for x in _oid(info[f[1][1]:f[1][2]])),
            "hash": "sha-256" if imp_alg and info[imp_alg[0]:imp_alg[1]] == SHA256_OID
                    else ".".join(str(x) for x in _oid(info[imp_alg[0]:imp_alg[1]])) if imp_alg else "",
            "imprint": info[imp_val[0]:imp_val[1]].hex() if imp_val else "",
            "serial": int.from_bytes(info[f[3][1]:f[3][2]], "big"),
            "gen_time": info[f[4][1]:f[4][2]].decode("ascii")}


def _oid(b: bytes) -> list[int]:
    parts = [b[0] // 40, b[0] % 40]
    v = 0
    for byte in b[1:]:
        v = (v << 7) | (byte & 0x7F)
        if not byte & 0x80:
            parts.append(v)
            v = 0
    return parts


def check(token: bytes, digest: bytes) -> dict:
    """The token's imprint against the digest we stamped; the authority's signature is the reader's to verify."""
    info = token_info(token)
    return {**info, "imprint_matches": info.get("imprint") == digest.hex(),
            "signature_checked": False,
            "how_to_verify": "openssl ts -verify -in <token.tsr> -data <stamped bytes> -CAfile <authority CA>"}
