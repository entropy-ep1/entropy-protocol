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
from cryptography.hazmat.primitives import hashes  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import padding, rsa  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from entropy_protocol import cose, pack, statements  # noqa: E402
from entropy_protocol.attested import appraise, authority, ear, enroll, evidence, hcl, pki, tpm  # noqa: E402

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


# ------------------------------------------------------------------ the quote's selection is the guest's to choose
@pytest.fixture(scope="module")
def stand_in_ak():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def uki_only(policy):
    """No PCR reference values: the kernel image's sections alone stand for the boot chain."""
    p = copy.deepcopy(policy)
    p["vtpm"].pop("pcrs")
    return p


def evidence_of(d: Path) -> tuple[dict, dict]:
    req = enroll.read_request((d / "request.cbor").read_bytes())
    return req, cose._thaw(cbor2.loads(req["evidence"]))


def with_evidence(req: dict, ev: dict) -> bytes:
    return enroll.request(target=req["target"], credential_type=req["credential_type"], handle=req["handle"],
                          evidence=cose.dumps(ev), csr=req["csr"])


def quoted_values(d: Path) -> dict:
    q = evidence_of(d)[1]["quote"]
    return tpm.split_values(q["pcrs"], tpm.parse_attest(q["msg"]))


def requoted(d: Path, selection: list, values: dict, sign_hash, key, monkeypatch) -> bytes:
    """The run's request with its quote taken again over a selection of the guest's choosing: the same qualifying data
    and clock, the PCR digest under the signing scheme's hash, as a TPM computes it. The run cannot lend the
    attestation key - which signs any selection the guest asks for - so a key of our own stands in for it."""
    req, ev = evidence_of(d)
    q = tpm.parse_attest(ev["quote"]["msg"])
    be = lambda v, n: v.to_bytes(n, "big")  # noqa: E731
    blob = b"".join(values[(alg, i)] for alg, idx in selection for i in idx)
    msg = (be(tpm.TPM_GENERATED, 4) + be(tpm.ST_ATTEST_QUOTE, 2) + be(len(q.signer), 2) + q.signer + be(len(q.extra), 2)
           + q.extra + be(q.clock, 8) + be(q.reset_count, 4) + be(q.restart_count, 4) + be(q.safe, 1)
           + be(q.firmware, 8) + be(len(selection), 4))
    for alg, idx in selection:
        bits = bytearray(3)
        for i in idx:
            bits[i // 8] |= 1 << i % 8
        msg += be(alg, 2) + be(len(bits), 1) + bytes(bits)
    h = hashes.Hash(sign_hash())
    h.update(blob)
    digest = h.finalize()
    msg += be(len(digest), 2) + digest
    s = key.sign(msg, padding.PKCS1v15(), sign_hash())
    alg_id = {hashes.SHA256: tpm.ALG_SHA256, hashes.SHA384: tpm.ALG_SHA384}[sign_hash]
    ev["quote"] = {**ev["quote"], "msg": msg, "sig": be(tpm.ALG_RSASSA, 2) + be(alg_id, 2) + be(len(s), 2) + s,
                   "pcrs": blob, "sel": [i for _, idx in selection for i in idx]}
    monkeypatch.setattr(hcl, "rsa_from_jwk", lambda jwk: key.public_key())   # the stand-in, bound in the AK's place
    return with_evidence(req, ev)


def vtpm_failed(a) -> set:
    return {x.name for x in a.failed() if x.name.startswith("vtpm")}


def test_requoted_over_the_same_pcrs_nothing_changes(rrp_id, policy, stand_in_ak, monkeypatch):
    d = issued(enroll.CRED_STATEMENT)
    values = quoted_values(d)
    sel = [(tpm.ALG_SHA256, [i for _, i in sorted(values)])]
    a, _ = reappraise(d, rrp_id, policy, requoted(d, sel, values, hashes.SHA256, stand_in_ak, monkeypatch))
    assert a.ok and not a.failed(), [f"{x.name}: {x.detail}" for x in a.failed()]


def test_a_quote_that_leaves_out_pcr_11_is_refused(rrp_id, uki_only, stand_in_ak, monkeypatch):
    d = issued(enroll.CRED_STATEMENT)
    values = quoted_values(d)
    sel = [(tpm.ALG_SHA256, [i for _, i in sorted(values) if i != 11])]
    a, _ = reappraise(d, rrp_id, uki_only, requoted(d, sel, values, hashes.SHA256, stand_in_ak, monkeypatch))
    by = {x.name: x for x in a.checks}
    assert vtpm_failed(a) == {"vtpm.eventlog.replays_to_quoted_pcrs", "vtpm.uki_sections.bound_to_quote"}
    assert "not in the quote [11]" in by["vtpm.eventlog.replays_to_quoted_pcrs"].detail
    # everything else holds: the sections would have been read, unbound, from the log
    assert by["vtpm.quote.pcr_digest"].ok and by["vtpm.uki_sections.in_reference_set"].ok


@pytest.mark.parametrize("bank, sign_hash", [(tpm.ALG_SHA384, hashes.SHA384), (tpm.ALG_SHA384, hashes.SHA256),
                                             (tpm.ALG_SHA1, hashes.SHA256)])
def test_a_quote_over_another_bank_is_refused(bank, sign_hash, rrp_id, uki_only, stand_in_ak, monkeypatch):
    """Azure's vTPM keeps SHA-1, SHA-256 and SHA-384 banks; its AK signs RSASSA-SHA256 only (capture platform.txt,
    ak_public.tss). A TPM puts the PCR digest under the signing hash, so the digest check alone refuses Azure's
    other banks - by accident of the hash it recomputes with; an AK that may sign SHA-384 passes it."""
    d = issued(enroll.CRED_STATEMENT)
    values = {(bank, i): tpm.HASH[bank](v).digest() for (_, i), v in quoted_values(d).items()}
    sel = [(bank, [i for _, i in sorted(values)])]
    a, _ = reappraise(d, rrp_id, uki_only, requoted(d, sel, values, sign_hash, stand_in_ak, monkeypatch))
    by = {x.name: x for x in a.checks}
    assert {"vtpm.quote.bank_is_sha256", "vtpm.eventlog.replays_to_quoted_pcrs", "vtpm.uki_sections.bound_to_quote"} \
        <= vtpm_failed(a)
    assert by["vtpm.quote.pcr_digest"].ok is (tpm.CRYPTO_HASH.get(bank) is sign_hash)


def test_without_systemds_log_the_kernel_sections_are_not_read(rrp_id, uki_only):
    """The run's own evidence, nothing signed again: withheld, systemd's log leaves PCR 11 unreplayable, so the
    sections the firmware log names are no longer bound to the quote."""
    d = issued(enroll.CRED_STATEMENT)
    req, ev = evidence_of(d)
    assert ev.pop("systemd_tpm_log")
    a, _ = reappraise(d, rrp_id, uki_only, with_evidence(req, ev))
    by = {x.name: x for x in a.checks}
    assert by["vtpm.eventlog.replays_to_quoted_pcrs"].ok
    assert "not covered without systemd's log: [11]" in by["vtpm.eventlog.replays_to_quoted_pcrs"].detail
    assert vtpm_failed(a) == {"vtpm.uki_sections.bound_to_quote"}
