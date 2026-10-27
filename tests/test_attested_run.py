"""Attested sealing end to end (ARCHITECTURE §7d), checked on the run kept in bench/results/attested/run-*: Qwen2.5-7B-
Instruct sealing inside an Azure confidential VM with a confidential H100, its keys certified by an authority that
appraised the hardware evidence. Every refusal is tried on that run's own data, changed the way an attacker would."""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import cbor2
import pytest

pytest.importorskip("signxml")

from cryptography import x509  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from entropy_protocol import cose, pack, statements  # noqa: E402
from entropy_protocol.attested import appraise, authority, ear, enroll, evidence, pki  # noqa: E402

RUNS = sorted(p for p in (Path(__file__).resolve().parents[1] / "bench/results/attested").glob("run-*")
              if (p / "guest/out/manifest.json").exists())
pytestmark = pytest.mark.skipif(not RUNS, reason="no attested run kept")
RUN = RUNS[-1] if RUNS else Path(".")
AUTH = RUN / "authority"
OUT = RUN / "guest/out"


def issued(credential_type: str) -> Path:
    for d in sorted((AUTH / "issued").iterdir()):
        if (d / "cert.der").exists() and enroll.read_request((d / "request.cbor").read_bytes())["credential_type"] \
                == credential_type:
            return d
    raise LookupError(credential_type)


@pytest.fixture(scope="module")
def root():
    return x509.load_der_x509_certificate((AUTH / "root.der").read_bytes())


@pytest.fixture(scope="module")
def rrp_id():
    return json.loads((AUTH / "authority.json").read_text())["rrp_id"]


@pytest.fixture(scope="module")
def policy():
    return json.loads((AUTH / "policy.json").read_text())


def reappraise(d: Path, rrp_id: str, policy: dict, request: bytes | None = None):
    c = ear.read((d / "ear.cose").read_bytes(), x509.load_der_x509_certificate((AUTH / "root.der").read_bytes())
                 .public_key())
    req = enroll.read_request(request or (d / "request.cbor").read_bytes())
    at = dt.datetime.fromtimestamp(c[ear.IAT], dt.timezone.utc)
    return appraise.appraise(req, rrp_id=rrp_id, policy=policy, at=at,
                             collateral=cose._thaw(cbor2.loads((d / "collateral.cbor").read_bytes()))), c


@pytest.mark.parametrize("cred", [enroll.CRED_STATEMENT, enroll.CRED_DETECTION])
def test_both_keys_affirmed_and_the_appraisal_repeats_offline(cred, root, rrp_id, policy):
    d = issued(cred)
    a, c = reappraise(d, rrp_id, policy)
    assert a.ok and not a.failed(), [f"{x.name}: {x.detail}" for x in a.failed()]
    assert c[ear.STATUS] == ear.AFFIRMING and c[ear.EP1]["credential_type"] == cred and not c[ear.EP1]["failed"]
    assert set(c[ear.SUBMODS]) == {"cpu", "vtpm", "gpu0"} and c[ear.TOPOLOGY] == {"cpu": ["vtpm", "gpu0"]}
    assert all(c[ear.EP1][k] == v for k, v in a.identities.items())
    leaf = pki.check_leaf((d / "cert.der").read_bytes(), root, dt.datetime.fromtimestamp(c[ear.IAT], dt.timezone.utc),
                          pki.STATEMENT_EKU)
    assert pki.spki(leaf) == pki.spki(a.statement_key)
    assert c[ear.NONCE] == enroll.read_request((d / "request.cbor").read_bytes())["handle"]
    for name in ("weights", "tokenizer", "code", "params"):
        assert c[ear.EP1][name].hex() in policy["identities"][name]


def test_every_statement_signed_in_the_environment(root):
    seal_dir, det_dir = issued(enroll.CRED_STATEMENT), issued(enroll.CRED_DETECTION)
    seal_cert, det_cert = (seal_dir / "cert.der").read_bytes(), (det_dir / "cert.der").read_bytes()
    seal_key = x509.load_der_x509_certificate(seal_cert).public_key()
    det_key = x509.load_der_x509_certificate(det_cert).public_key()
    docs = [p for p in OUT.iterdir() if (p / "generation.cose").exists()]
    assert len(docs) == json.loads((OUT / "manifest.json").read_text())["sealed"].__len__() > 0
    for d in docs:
        gen, det = (d / "generation.cose").read_bytes(), (d / "detection.cose").read_bytes()
        prot = cose.decode1(gen)[0]
        assert prot[evidence.X5CHAIN] == seal_cert
        assert prot[evidence.EAR_REF] == hashlib.sha256((seal_dir / "ear.cose").read_bytes()).digest()
        cose.verify1(gen, seal_key)
        assert cose.decode1(det)[0][evidence.X5CHAIN] == det_cert
        statements.read_statement(det, det_key)
        with pytest.raises(cose.COSEError):
            cose.verify1(gen, det_key)                       # the detection key cannot have signed a generation


def test_verdicts_and_detections():
    rows = json.loads((OUT / "verification.json").read_text())
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    assert by["as-sealed"] and all(r["verdict"] == "VERIFIED" for r in by["as-sealed"])
    assert all(r["verdict"] == "VERIFIED" for r in by["transported"])
    assert all(r["verdict"] == "MODIFIED" and r["changed"] for r in by["one-word-changed"])
    m = json.loads((OUT / "manifest.json").read_text())
    controls = [x for x in m["detections"] if x["kind"] == "control"]
    assert controls and not any(x["detected"] for x in controls)         # unsealed text of the same model


def test_packages_check_offline():
    reports = sorted((RUN / "packs").glob("*.check.json"))
    assert reports
    for f in reports:
        r = json.loads(f.read_text())
        assert r["ok"], [c for c in r["checks"] if c["ok"] is False]
        names = {c["name"] for c in r["checks"]}
        assert "attestation (generation): evidence appraised again offline" in names
        assert "attestation (detection): evidence appraised again offline" in names


# ------------------------------------------------------------------ refusals, on this run's own data
def test_a_request_bound_to_another_key_is_refused(rrp_id, policy):
    d = issued(enroll.CRED_STATEMENT)
    req = enroll.read_request((d / "request.cbor").read_bytes())
    other = enroll.request(target=req["target"], credential_type=req["credential_type"], handle=req["handle"],
                           evidence=req["evidence"], csr=pki.make_csr(Ed25519PrivateKey.generate(), req["target"]))
    a, _ = reappraise(d, rrp_id, policy, other)
    failed = {x.name for x in a.failed()}
    assert {"vtpm.runtime_claims.user_data_is_binding", "vtpm.quote.qualifying_data_is_binding_digest"} <= failed


def test_evidence_moved_to_another_enrollment_is_refused(rrp_id, policy):
    s, t = issued(enroll.CRED_STATEMENT), issued(enroll.CRED_DETECTION)
    rs, rt = (enroll.read_request((x / "request.cbor").read_bytes()) for x in (s, t))
    moved = enroll.request(target=rs["target"], credential_type=rs["credential_type"], handle=rs["handle"],
                           evidence=rt["evidence"], csr=rs["csr"])            # the detection key's evidence
    a, _ = reappraise(s, rrp_id, policy, moved)
    failed = {x.name for x in a.failed()}
    assert "vtpm.runtime_claims.user_data_is_binding" in failed and "gpu0.report.nonce_is_handle" in failed


def test_claims_changed_after_the_fact_are_refused(rrp_id, policy):
    d = issued(enroll.CRED_STATEMENT)
    req = enroll.read_request((d / "request.cbor").read_bytes())
    ev = cose._thaw(cbor2.loads(req["evidence"]))
    ev["claims"]["weights"] = hashlib.sha256(b"another model").digest()
    forged = enroll.request(target=req["target"], credential_type=req["credential_type"], handle=req["handle"],
                            evidence=cose.dumps(ev), csr=req["csr"])
    a, _ = reappraise(d, rrp_id, policy, forged)
    failed = {x.name for x in a.failed()}
    assert {"vtpm.runtime_claims.user_data_is_binding", "identity.weights"} <= failed


def test_a_handle_is_spent_once(tmp_path, policy):
    copy_dir = tmp_path / "authority"
    shutil.copytree(AUTH, copy_dir, ignore=shutil.ignore_patterns("root.key"))
    (copy_dir / "root.key").write_bytes((AUTH / "root.key").read_bytes()) if (AUTH / "root.key").exists() else None
    if not (copy_dir / "root.key").exists():
        pytest.skip("the authority's key is not in this checkout")
    d = issued(enroll.CRED_STATEMENT)
    o = authority.Authority(copy_dir).enroll((d / "request.cbor").read_bytes(),
                                             collateral=cose._thaw(cbor2.loads((d / "collateral.cbor").read_bytes())))
    assert o.cert is None and o.status == ear.CONTRAINDICATED
    assert any(c.name == "binding.handle_issued_unexpired_unused" and c.ok is False for c in o.appraisal.checks)


def test_a_package_with_a_changed_text_or_a_foreign_result_fails():
    f = sorted((RUN / "packs").glob("*.cbor"))[0]
    p = cose._thaw(cbor2.loads(f.read_bytes()))
    edited = copy.deepcopy(p)
    edited["text"] = edited["text"].replace(" ", "  ", 1).replace("a", "o", 1)
    r = pack.check(cose.dumps(edited))
    assert not r["ok"] and any(c["name"] == "the verdict is about this text" and c["ok"] is False for c in r["checks"])
    foreign = copy.deepcopy(p)
    other_root = Ed25519PrivateKey.generate()
    foreign["att"]["root"] = pki.der(pki.make_root(other_root, "another authority", "x"))
    r = pack.check(cose.dumps(foreign))
    assert not r["ok"] and any(c["name"].endswith("result signed by the authority") and c["ok"] is False
                               for c in r["checks"])
