"""The attestation part of an evidence package (`ep1-pack/v2`, ARCHITECTURE §7d) and its offline check.

`att` = {"root": the authority's root certificate (DER), "authority": {"rrp_id", "name"},
         "gen" | "det": {"ear": the attestation result, "request": the enrollment request (evidence + CSR),
                         "collateral": what the authority fetched (CBOR), "policy": the policy it applied (JSON)}}

The check needs no secret and no network. Per attested statement it establishes: the result is signed by the root;
the statement carries that result's digest and a certificate of the root for the key it is signed under; the
result affirms exactly that key, for that kind of statement; the result's nonce is the request's handle; and the
evidence in the request, appraised again here with the same code, the same collateral and the same policy at the
result's issue time, affirms - from AMD's and NVIDIA's pinned roots, through the hardware reports and the binding
value, to the certified key. The reader still decides whether the root is the authority's (compare its fingerprint
with the one the authority publishes) - the package cannot vouch for itself.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import cbor2
from cryptography import x509

from .. import cose
from . import appraise, ear, enroll, pki

EAR_REF = "ep1-ear"
X5CHAIN = 33
CREDENTIAL_OF = {"gen": enroll.CRED_STATEMENT, "det": enroll.CRED_DETECTION}
TITLE = {"gen": "generation", "det": "detection"}


def section(root_der: bytes, authority: dict, roles: dict) -> dict:
    """``roles``: {"gen" | "det": issued directory of the authority (request.cbor, collateral.cbor, ear.cose) +
    the policy text}."""
    out = {"root": root_der, "authority": {"rrp_id": authority["rrp_id"], "name": authority["name"]}}
    for role, d in roles.items():
        out[role] = {"ear": (d["dir"] / "ear.cose").read_bytes(), "request": (d["dir"] / "request.cbor").read_bytes(),
                     "collateral": (d["dir"] / "collateral.cbor").read_bytes(), "policy": d["policy"]}
    return out


def check(st: dict, att: dict, note) -> dict:
    """Run every attestation check, reporting through ``note(name, ok, detail)``. Returns {role: Ed25519 key} for the
    statements whose key the attestation established, and the attested identities per role."""
    keys, ids = {}, {}
    try:
        root = x509.load_der_x509_certificate(att["root"])
    except (KeyError, ValueError) as exc:
        note("attestation: authority root", False, f"{type(exc).__name__}: {exc}")
        return {"keys": keys, "identities": ids}
    note("attestation: authority root", None, f"{att.get('authority', {}).get('name', '?')}, SHA-256 "
         f"{hashlib.sha256(att['root']).hexdigest()} - compare with the fingerprint the authority publishes")
    for role in ("gen", "det"):
        if role not in st or role not in att:
            continue
        t = TITLE[role]
        a = att[role]
        try:
            c = ear.read(a["ear"], root.public_key())
        except Exception as exc:  # noqa: BLE001 - a failed check is a result
            note(f"attestation ({t}): result signed by the authority", False, f"{type(exc).__name__}: {exc}")
            continue
        iat = dt.datetime.fromtimestamp(c[ear.IAT], dt.timezone.utc)
        ext = c.get(ear.EP1, {})
        note(f"attestation ({t}): result signed by the authority", True,
             f"issued {iat.isoformat()}, status {c.get(ear.STATUS)}, submods {sorted(c.get(ear.SUBMODS, {}))}")
        note(f"attestation ({t}): result affirming", c.get(ear.STATUS) == ear.AFFIRMING and not ext.get("failed"),
             f"failed checks: {ext.get('failed') or 'none'}")
        note(f"attestation ({t}): result is for this kind of statement", ext.get("credential_type") ==
             CREDENTIAL_OF[role], str(ext.get("credential_type")))
        prot = cose.decode1(st[role])[0]
        note(f"attestation ({t}): statement names this result", prot.get(EAR_REF) == hashlib.sha256(a["ear"]).digest(),
             "protected header ep1-ear = SHA-256 of the result")
        leaf = prot.get(X5CHAIN)
        leaf = leaf[0] if isinstance(leaf, list) else leaf
        try:
            key = pki.check_leaf(leaf, root, iat, pki.STATEMENT_EKU)
            note(f"attestation ({t}): certificate of the authority for the signing key", True,
                 x509.load_der_x509_certificate(leaf).subject.rfc4514_string())
        except Exception as exc:  # noqa: BLE001
            note(f"attestation ({t}): certificate of the authority for the signing key", False,
                 f"{type(exc).__name__}: {exc}")
            continue
        note(f"attestation ({t}): result affirms this key", ext.get("key") == hashlib.sha256(pki.spki(key)).digest(),
             "ep1 key = SHA-256 of the certified key")
        try:
            req = enroll.read_request(a["request"])
            note(f"attestation ({t}): result's nonce is the request's handle", c.get(ear.NONCE) == req["handle"])
            again = appraise.appraise(req, rrp_id=att["authority"]["rrp_id"], policy=json.loads(a["policy"]), at=iat,
                                      collateral=cose._thaw(cbor2.loads(a["collateral"])))
            failed = [f"{x.name}: {x.detail}" for x in again.failed()]
            note(f"attestation ({t}): evidence appraised again offline", not failed and again.ok,
                 f"{len(again.checks)} checks, failed: {failed or 'none'}; not performed: "
                 f"{[x.name for x in again.checks if x.ok is None] or 'none'}")
            same = all(ext.get(k) == v for k, v in again.identities.items())
            note(f"attestation ({t}): the result's identities are the evidence's", same,
                 f"binding {again.identities.get('binding', b'').hex()[:16]}…, weights "
                 f"{again.identities.get('weights', b'').hex()[:16]}…, code {again.identities.get('code', b'').hex()[:16]}…")
            note(f"attestation ({t}): the request's key is the certified key", again.statement_key is not None and
                 pki.spki(again.statement_key) == pki.spki(key))
        except Exception as exc:  # noqa: BLE001
            note(f"attestation ({t}): evidence appraised again offline", False, f"{type(exc).__name__}: {exc}")
            continue
        keys[role] = key
        ids[role] = ext
    return {"keys": keys, "identities": ids}
