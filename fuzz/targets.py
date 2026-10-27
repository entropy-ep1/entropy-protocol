"""Fuzz targets: every parser of untrusted input in the package, each with the errors it is allowed to raise.

A target takes arbitrary bytes and must return, or raise one of its declared errors. Any other exception, a hang or
unbounded memory is a finding. Targets that check a property (canonical form idempotent, the two implementations
equal) raise PropertyFailure, which is never allowed. Seeds are real data (fuzz/make_corpus.py). Run with atheris on
Linux (fuzz/run.py), or replay a corpus anywhere (tests/test_fuzz.py, which also drives every target with Hypothesis).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from entropy_protocol import canon, cose, crypto, delivery, pack, plain, record, statements, tlog, ts3161
from entropy_protocol.attested import amd, appraise, ear, enroll, hcl, nvidia, rim, tpm

ROOT = Path(__file__).resolve().parents[1]
TARGETS: dict[str, tuple] = {}

# fixed keys: a fuzzer explores parsing and checking, not key search
SIGNER = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
PUB = SIGNER.public_key()
X25519 = X25519PrivateKey.from_private_bytes(bytes(range(1, 33)))
LOG = Ed25519PrivateKey.from_private_bytes(bytes(range(2, 34)))
STATEMENT = cose.sign1({cose.ALG: cose.ED25519}, b"payload", SIGNER)
MASTER = bytes(32)
AT = dt.datetime(2026, 10, 27, 17, 0, tzinfo=dt.timezone.utc)


class PropertyFailure(AssertionError):
    """A property the target checks does not hold: always a finding."""


def target(name: str, *errors: type):
    def wrap(fn):
        TARGETS[name] = (fn, errors)
        return fn
    return wrap


def text_of(data: bytes) -> str:
    """A string from bytes, weighted toward what canonical forms treat specially: line ends, BOM, zero-widths, soft
    hyphen, variation selectors, no-break and other spaces, typographic quotes and dashes, combining marks, scripts
    written without spaces, emoji sequences - and lone surrogates, which JSON can carry and UTF-8 cannot encode."""
    pool = ("\r", "\n", "\r\n", "﻿", "​", "‌", "‍", "⁠", "­", "️", "︎",
            "\U000e0100", " ", " ", " ", "　", "\t", "‘", "’", "“", "”",
            "«", "»", "„", "‐", "–", "—", "…", "́", "̈", "é",
            "é", "й", "Й", "中", "ก", "่", "ا", "\U0001f44d\U0001f3fd",
            "\U0001f1fa\U0001f1e6", "\U0001f468‍\U0001f469", ". ", "! ", "? ", "。", "।", "  ", "- ",
            "* ", "1. ", "[ ] ", "#", "`", "|", "\ud800", "\udfff", "a", "b", " ")
    out, i = [], 0
    while i + 1 < len(data):
        b0, b1 = data[i], data[i + 1]
        if b0 < 0xC0:
            out.append(pool[b1 % len(pool)])
        else:
            cp = ((b0 & 0x3F) << 8 | b1) + (0x10000 if b0 & 0x20 else 0)
            out.append(chr(cp))
        i += 2
    return "".join(out)


@target("record", ValueError)
def t_record(data: bytes) -> None:
    record.loads(data)


@target("token", crypto.TokenError)
def t_token(data: bytes) -> None:
    tok = data.decode("utf-8", "replace")
    crypto.token_doc_id(tok)
    crypto.decrypt_token(MASTER, tok)


@target("cose_sign1", cose.COSEError)
def t_cose_sign1(data: bytes) -> None:
    cose.verify1(data, PUB)


@target("cose_encrypt0", cose.COSEError)
def t_cose_encrypt0(data: bytes) -> None:
    cose.decrypt0_hpke(data, X25519.private_bytes_raw())


@target("delivered", cose.COSEError)
def t_delivered(data: bytes) -> None:
    delivery.open_delivered(data, X25519.private_bytes_raw())


@target("statement", cose.COSEError)
def t_statement(data: bytes) -> None:
    statements.read_statement(data, PUB)


@target("receipt", cose.COSEError, ValueError)
def t_receipt(data: bytes) -> None:
    tlog.verify_receipt(data, STATEMENT, LOG.public_key())


@target("timestamp", ValueError)
def t_timestamp(data: bytes) -> None:
    ts3161.token_info(data)
    ts3161.check(data, hashlib.sha256(data).digest())


@target("timestamp_reply", ValueError)
def t_timestamp_reply(data: bytes) -> None:
    ts3161.token_from_response(data)


@target("pack")
def t_pack(data: bytes) -> None:
    """An offline check of an untrusted package reports what failed; it never raises."""
    pack.check(data)


@target("hcl", hcl.HCLError, amd.SNPError)
def t_hcl(data: bytes) -> None:
    h = hcl.parse(data)
    hcl.report_data_binds_claims(h)
    _ = h.user_data
    k = h.jwk("HCLAkPub")
    if k is not None:
        hcl.rsa_from_jwk(k)


@target("snp_report", amd.SNPError)
def t_snp(data: bytes) -> None:
    r = amd.parse_report(data)
    for product in ("Genoa", "Turin"):
        amd.tcb_parts(r.reported_tcb, product)


@target("tpm_quote", tpm.TPMError)
def t_tpm_quote(data: bytes) -> None:
    q = tpm.parse_attest(data)
    tpm.split_values(bytes(32 * 16), q)


@target("tpm_signature", tpm.TPMError)
def t_tpm_signature(data: bytes) -> None:
    tpm.parse_signature(data)


@target("event_log", tpm.TPMError)
def t_event_log(data: bytes) -> None:
    log = tpm.parse_event_log(data)
    tpm.replay(log)
    tpm.uki_sections(log)


@target("systemd_log", tpm.TPMError)
def t_systemd_log(data: bytes) -> None:
    tpm.parse_systemd_log(data)


@target("gpu_report", nvidia.GPUError)
def t_gpu_report(data: bytes) -> None:
    rep = nvidia.parse(data)
    nvidia.vbios_version(rep.opaque.get("vbios_version", b""))
    appraise.rim_ids(rep)


@target("rim", rim.RIMError)
def t_rim(data: bytes) -> None:
    rim.verify(data, AT)


@target("rim_payload", rim.RIMError)
def t_rim_payload(data: bytes) -> None:
    """The golden-value reader on a parsed tree (what follows a verified signature)."""
    from lxml import etree
    try:
        tree = etree.fromstring(data, etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False))
    except etree.XMLSyntaxError:
        return
    rim.golden(tree)


@target("enroll", enroll.EnrollError)
def t_enroll(data: bytes) -> None:
    for read in (enroll.read_request, enroll.read_initiation):
        try:
            read(data)
        except enroll.EnrollError:
            pass


@target("ear", cose.COSEError)
def t_ear(data: bytes) -> None:
    ear.read(data, PUB)


@target("appraise")
def t_appraise(data: bytes) -> None:
    """The authority appraises untrusted evidence: every failure is a named check, never an exception."""
    req = {"freshness_kind": "present-nonce", "target": "t", "credential_type": enroll.CRED_STATEMENT,
           "handle": bytes(32), "evidence": data, "profile": enroll.PROFILE_AZURE_CGPU,
           "binding": {"method": "binding-input", "hash": "sha512"}, "csr": b""}
    appraise.appraise(req, rrp_id="urn:x", policy={}, at=AT, collateral={})


@target("canon")
def t_canon(data: bytes) -> None:
    """Every canonical form runs on any string; v5, the reference, is idempotent and its index map points inside the
    canonical text (v3 and v4 are not idempotent - why v5 exists - and stay only to verify their records)."""
    s = text_of(data)
    for version in canon.SUPPORTED:
        c, cmap = canon.canonicalize_with_map(s, version)
        if version == canon.CANON_V5:
            if canon.canonicalize(c, version) != c:
                raise PropertyFailure(f"{version} not idempotent on {s!r}")
            if any(not 0 <= ci < max(len(c), 1) for ci in cmap.values()):
                raise PropertyFailure(f"index map points outside the canonical text on {s!r}")
        for a, b in canon.segment(c, version):
            if not 0 <= a <= b <= len(c):
                raise PropertyFailure(f"segment ({a}, {b}) outside the text")


@target("plain")
def t_plain(data: bytes) -> None:
    s = text_of(data)
    plain.normalize(s)
    plain.segment_visible(s)


# ------------------------------------------------------------------ the second implementation, on the same inputs
class _JS:
    """One Node process for the session: JSON lines in, JSON lines out (fuzz/js_server.mjs)."""
    proc = None

    @classmethod
    def call(cls, fn: str, *args):
        if cls.proc is None:
            cls.proc = subprocess.Popen(["node", str(ROOT / "fuzz/js_server.mjs")], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="surrogatepass")
        cls.proc.stdin.write(json.dumps({"fn": fn, "args": args}) + "\n")
        cls.proc.stdin.flush()
        return json.loads(cls.proc.stdout.readline())


@target("canon_diff")
def t_canon_diff(data: bytes) -> None:
    """Python and JavaScript give the same canonical form, blocks and visible text for the same string."""
    s = text_of(data).encode("utf-16", "surrogatepass").decode("utf-16", "replace")   # JSON carries no lone surrogate
    from entropy_protocol._unicode14 import assigned
    newer = any(not assigned(ord(ch)) for ch in s)
    for version in canon.SUPPORTED:
        if newer and version != canon.CANON_V5:
            continue          # v3 and v4 normalize with the library's Unicode version: equal only below 14.0's set
        py = canon.canonicalize(s, version)
        js = _JS.call("canonicalize", s, version)
        if js.get("ok") != py:
            raise PropertyFailure(f"canonical form {version} differs on {s!r}: python {py!r}, js {js}")
        segs = [list(x) for x in canon.segment(py, version)]
        if _JS.call("segment", py, version).get("ok") != segs:
            raise PropertyFailure(f"blocks {version} differ on {py!r}")
    vis = plain.normalize(s)
    if _JS.call("plainNormalize", s).get("ok") != vis:
        raise PropertyFailure(f"visible text differs on {s!r}")


@target("cbor_diff")
def t_cbor_diff(data: bytes) -> None:
    """Under the ep1 CBOR profile (cose.check_profile) both decoders accept exactly the same inputs and decode them to
    the same value."""
    try:
        py = cose.loads(data)
        py_ok = True
    except cose.COSEError:
        py_ok, py = False, None
    js = _JS.call("decode", data.hex())
    if py_ok != ("ok" in js):
        raise PropertyFailure(f"{'only the reference' if py_ok else 'only the second implementation'} accepts "
                              f"{data.hex()} ({js.get('error', '')})")
    if py_ok and _plain(py) != js["ok"]:
        raise PropertyFailure(f"decoders disagree on {data.hex()}: python {_plain(py)!r}, js {js['ok']!r}")


def _plain(v):
    """A decoded value in the shape js_server.mjs gives. JavaScript has one number type, so an integral float within
    2^53 reads as the integer (ep1 writes no floating point, so this hides nothing ep1 relies on); NaN payloads fold."""
    import math
    import struct
    if v is cbor2.undefined:
        return {"$undefined": True}
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return {"$int": str(v)}
    if isinstance(v, float):
        if math.isnan(v):
            return {"$float": "nan"}
        if v.is_integer() and abs(v) < 2 ** 53 and not (v == 0 and math.copysign(1, v) < 0):
            return {"$int": str(int(v))}
        return {"$float": struct.pack(">d", v).hex()}
    if isinstance(v, bytes):
        return {"$bytes": v.hex()}
    if isinstance(v, Mapping):                       # cbor2 gives frozendicts inside tags
        return {"$map": [[_plain(k), _plain(x)] for k, x in v.items()]}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, cbor2.CBORTag):
        return {"$tag": str(v.tag), "value": _plain(v.value)}
    return {"$other": type(v).__name__}


def run(name: str, data: bytes) -> None:
    """Call a target; let only its declared errors through as non-findings."""
    fn, errors = TARGETS[name]
    try:
        fn(data)
    except PropertyFailure:
        raise
    except errors:
        pass
