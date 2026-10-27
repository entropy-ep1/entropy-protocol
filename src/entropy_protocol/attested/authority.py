"""The credential authority (ARCHITECTURE §7d): issues handles, appraises enrollment requests, certifies keys.

One directory per authority:
  root.key        Ed25519 private key (PKCS#8 PEM, mode 0600); it signs certificates and attestation results
  root.der        the root certificate - what relying parties pin
  authority.json  {"rrp_id", "name", "verifier_id": [developer, build]}
  policy.json     the appraisal policy: reference values with their provenance (appraise.py)
  handles.json    the handle book (enroll.HandleBook)
  issued/NNNN/    per request: request.cbor, collateral.cbor, appraisal.json, ear.cose, and cert.der when affirmed
Collateral is fetched live at enrollment - AMD's CRL, NVIDIA's manifests and OCSP answers - and kept with the
request, so the same appraisal can be repeated offline at the result's issue time.

`python -m entropy_protocol.attested.authority init|initiate|enroll|show ...`
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .. import ENGINE, cose
from . import amd, appraise, ear, enroll, nvidia, ocsp, pki, rim

EAR_DEVELOPER = "https://github.com/entropy-ep1/entropy-protocol"
SKEW = 300


@dataclass
class Outcome:
    appraisal: appraise.Appraisal
    ear: bytes
    cert: bytes | None
    dir: Path
    status: int


class Authority:
    def __init__(self, path: Path):
        self.path = Path(path)
        meta = json.loads((self.path / "authority.json").read_text())
        self.rrp_id, self.name = meta["rrp_id"], meta["name"]
        self.verifier_id = tuple(meta.get("verifier_id", (EAR_DEVELOPER, ENGINE)))
        self.root = x509.load_der_x509_certificate((self.path / "root.der").read_bytes())
        self.handles = enroll.HandleBook(self.path / "handles.json")

    @property
    def policy(self) -> dict:
        return json.loads((self.path / "policy.json").read_text())

    def _key(self) -> Ed25519PrivateKey:
        return serialization.load_pem_private_key((self.path / "root.key").read_bytes(), None)

    def initiate(self, ttl: int = 600) -> bytes:
        return enroll.initiation(self.handles.issue(ttl), self.rrp_id, ttl)

    def enroll(self, request: bytes, *, now: dt.datetime | None = None, collateral: dict | None = None) -> Outcome:
        now = now or dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        out = self._next_dir()
        (out / "request.cbor").write_bytes(request)
        policy = self.policy
        try:
            req = enroll.read_request(request)
        except (enroll.EnrollError, ValueError) as exc:
            return self._refuse(out, f"malformed request: {exc}", now, policy)
        if collateral is None:
            collateral = fetch_collateral(req, policy)
        (out / "collateral.cbor").write_bytes(cose.dumps(collateral))
        a = appraise.appraise(req, rrp_id=self.rrp_id, policy=policy, at=now, collateral=collateral)
        try:
            self.handles.spend(req["handle"], now.timestamp())
            a.checks.insert(0, appraise.Check("binding.handle_issued_unexpired_unused", True))
        except enroll.EnrollError as exc:
            a.checks.insert(0, appraise.Check("binding.handle_issued_unexpired_unused", False, str(exc)))
        if req["profile"] != enroll.PROFILE_AZURE_CGPU:
            a.checks.append(appraise.Check("request.profile", False, req["profile"]))
        status = ear.worst(s[ear.STATUS] for s in a.submods.values()) if a.submods else ear.CONTRAINDICATED
        if not a.ok:
            status = ear.CONTRAINDICATED
        lifetime = int(policy.get("lifetime", 86400))
        ext = {"target": req["target"], "credential_type": req["credential_type"],
               "failed": [c.name for c in a.failed()], **a.identities}
        topo = {"cpu": [k for k in a.submods if k != "cpu"]} if "cpu" in a.submods else None
        c = ear.claims(a.submods, self.verifier_id, nonce=req["handle"], lifetime=lifetime, topology=topo, ep1=ext,
                       iat=int(now.timestamp()))
        c[ear.STATUS] = status
        signed = ear.sign(c, self._key(), [pki.der(self.root)])
        (out / "ear.cose").write_bytes(signed)
        cert = None
        if status == ear.AFFIRMING and a.statement_key is not None:
            # valid from a few minutes before issue: the sealer checks it against its own clock
            cert = pki.der(pki.issue(self._key(), self.root, a.statement_key, req["target"], lifetime=lifetime + SKEW,
                                     eku=pki.STATEMENT_EKU, not_before=now - dt.timedelta(seconds=SKEW)))
            (out / "cert.der").write_bytes(cert)
        _write_appraisal(out, a, status, now)
        return Outcome(a, signed, cert, out, status)

    def _refuse(self, out: Path, reason: str, now: dt.datetime, policy: dict) -> Outcome:
        a = appraise.Appraisal(checks=[appraise.Check("request.format", False, reason)])
        c = ear.claims({}, self.verifier_id, nonce=b"\x00" * 8, lifetime=int(policy.get("lifetime", 86400)),
                       iat=int(now.timestamp()))
        c[ear.STATUS] = ear.CONTRAINDICATED
        signed = ear.sign(c, self._key(), [pki.der(self.root)])
        (out / "ear.cose").write_bytes(signed)
        _write_appraisal(out, a, ear.CONTRAINDICATED, now)
        return Outcome(a, signed, None, out, ear.CONTRAINDICATED)

    def _next_dir(self) -> Path:
        issued = self.path / "issued"
        issued.mkdir(exist_ok=True)
        n = 1 + max((int(p.name) for p in issued.iterdir() if p.name.isdigit()), default=0)
        d = issued / f"{n:04d}"
        d.mkdir()
        return d


def _write_appraisal(out: Path, a: appraise.Appraisal, status: int, now: dt.datetime) -> None:
    (out / "appraisal.json").write_text(json.dumps({
        "time": now.isoformat(), "status": status, "ok": a.ok, "target": a.target,
        "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in a.checks],
        "identities": {k: (v.hex() if isinstance(v, bytes) else v) for k, v in a.identities.items()}}, indent=1))


def fetch_collateral(req: dict, policy: dict) -> dict:
    """AMD's CRL, NVIDIA's manifests for every GPU in the evidence, and OCSP answers for the GPU and manifest chains.
    What cannot be fetched is left out: the appraisal then fails the check that needed it, by name."""
    ev = cose._thaw(cose.loads(req["evidence"]))
    out: dict = {"rims": {}, "ocsp": []}
    product = policy.get("amd", {}).get("product", "Genoa")
    if policy.get("amd", {}).get("crl", "required") != "off":
        try:
            out["amd_crl"] = amd.fetch_crl(product)
        except OSError as exc:
            out["errors"] = [f"amd crl: {exc}"]
    asked: set = set()

    def ask(cert: x509.Certificate, issuer: x509.Certificate) -> None:
        key = (cert.serial_number, cert.issuer.rfc4514_string())
        if key in asked:
            return
        asked.add(key)
        n = ocsp.fresh_nonce()
        try:
            resp = ocsp.post(ocsp.request(cert, issuer, n))
        except OSError as exc:
            out.setdefault("errors", []).append(f"ocsp {cert.subject.rfc4514_string()}: {exc}")
            return
        out["ocsp"].append({"serial": format(cert.serial_number, "x"), "nonce": n, "response": resp,
                            "asked_at": dt.datetime.now(dt.timezone.utc).isoformat()})

    for g in ev.get("gpus", []):
        try:
            chain = nvidia.load_chain(g["chain"])
            rep = nvidia.parse(g["report"])
        except (nvidia.GPUError, KeyError, ValueError):
            continue
        for cert, issuer in zip(chain[1:-1], chain[2:]):
            ask(cert, issuer)
        for rid in appraise.rim_ids(rep).values():
            if rid in out["rims"]:
                continue
            try:
                xml = rim.fetch(rid)
            except OSError as exc:
                out.setdefault("errors", []).append(f"rim {rid}: {exc}")
                continue
            out["rims"][rid] = xml
            from lxml import etree
            try:
                rc = rim._chain(etree.fromstring(xml, etree.XMLParser(resolve_entities=False, no_network=True)))
            except (rim.RIMError, etree.XMLSyntaxError):
                continue
            for cert, issuer in zip(rc[:-1], rc[1:]):
                ask(cert, issuer)
    return out


def init(path: Path, rrp_id: str, name: str, policy: dict) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    key = Ed25519PrivateKey.generate()
    fd = os.open(path / "root.key", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))
    (path / "root.der").write_bytes(pki.der(pki.make_root(key, name, "Entropy Protocol reference authority")))
    (path / "authority.json").write_text(json.dumps({"rrp_id": rrp_id, "name": name,
                                                     "verifier_id": [EAR_DEVELOPER, ENGINE]}, indent=1))
    (path / "policy.json").write_text(json.dumps(policy, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="entropy-authority")
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init")
    i.add_argument("--dir", required=True)
    i.add_argument("--rrp-id", required=True)
    i.add_argument("--name", required=True)
    i.add_argument("--policy", required=True, help="policy JSON (reference values)")
    n = sub.add_parser("initiate")
    n.add_argument("--dir", required=True)
    n.add_argument("--ttl", type=int, default=600)
    n.add_argument("--out", required=True)
    e = sub.add_parser("enroll")
    e.add_argument("--dir", required=True)
    e.add_argument("--request", required=True)
    e.add_argument("--ear-out")
    e.add_argument("--cert-out")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        init(Path(a.dir), a.rrp_id, a.name, json.loads(Path(a.policy).read_text()))
        print(f"authority {a.name} ({a.rrp_id}) in {a.dir}; root sha256 "
              f"{hashlib.sha256((Path(a.dir) / 'root.der').read_bytes()).hexdigest()}")
        return 0
    auth = Authority(Path(a.dir))
    if a.cmd == "initiate":
        Path(a.out).write_bytes(auth.initiate(a.ttl))
        return 0
    o = auth.enroll(Path(a.request).read_bytes())
    if a.ear_out:
        Path(a.ear_out).write_bytes(o.ear)
    if o.cert and a.cert_out:
        Path(a.cert_out).write_bytes(o.cert)
    failed = [f"{c.name}: {c.detail}" for c in o.appraisal.failed()]
    print(json.dumps({"status": o.status, "certified": o.cert is not None, "dir": str(o.dir), "failed": failed,
                      "not_performed": [c.name for c in o.appraisal.checks if c.ok is None]}, indent=1))
    return 0 if o.cert else 4


if __name__ == "__main__":
    sys.exit(main())
