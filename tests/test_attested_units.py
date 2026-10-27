"""Attested sealing (ARCHITECTURE §7d): the parts that need no hardware - the binding value's encoding (a fixed
vector, so another implementation can match it), the handle book, the enrollment messages, the attestation result's
claims and tiers, the authority's certificates."""
from __future__ import annotations

import datetime as dt
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from entropy_protocol import cose
from entropy_protocol.attested import binding, ear, enroll, pki


def test_binding_value_vector():
    b = binding.BindingInput(handle=bytes(range(32)), rrp_id="urn:ep1:authority:example",
                             target="urn:ep1:sealer:example/seal", csr_digest=hashlib.sha256(b"csr").digest(),
                             statement_key=bytes.fromhex("302a300506032b6570032100") + bytes(32),
                             weights=hashlib.sha256(b"weights").digest(), tokenizer=hashlib.sha256(b"tok").digest(),
                             code=hashlib.sha256(b"code").digest(), sealing_class="fp16",
                             params=hashlib.sha256(b"params").digest())
    enc = binding.lp(b.fields())
    assert enc[:4] == len(binding.LABEL).to_bytes(4, "big") and enc[4:4 + len(binding.LABEL)] == binding.LABEL
    assert b.value() == hashlib.sha512(enc).digest() and len(b.value()) == 64
    assert b.value().hex() == hashlib.sha512(b"".join(len(f).to_bytes(4, "big") + f for f in b.fields())).hexdigest()
    with pytest.raises(ValueError):
        binding.BindingInput(**{**vars(b), "handle": bytes(31)}).value()


def test_handle_book(tmp_path):
    book = enroll.HandleBook(tmp_path / "handles.json")
    h = book.issue(60, now=1000)
    book.spend(h, now=1010)
    with pytest.raises(enroll.EnrollError, match="already used"):
        book.spend(h, now=1011)
    with pytest.raises(enroll.EnrollError, match="not issued"):
        book.spend(bytes(32), now=1011)
    late = book.issue(60, now=2000)
    with pytest.raises(enroll.EnrollError, match="expired"):
        book.spend(late, now=2061)
    with pytest.raises(enroll.EnrollError, match="already used"):    # spent even though it was refused
        book.spend(late, now=2062)


def test_enrollment_messages():
    ini = enroll.read_initiation(enroll.initiation(bytes(32), "urn:x", 600))
    assert ini["freshness_kind"] == "present-nonce" and ini["rrp_id"] == "urn:x"
    req = enroll.read_request(enroll.request(target="t", credential_type=enroll.CRED_STATEMENT, handle=bytes(32),
                                             evidence=b"e", csr=b"c"))
    assert req["binding"] == {"method": "binding-input", "hash": "sha512"}
    with pytest.raises(enroll.EnrollError):
        enroll.request(target="t", credential_type="something else", handle=bytes(32), evidence=b"e", csr=b"c")
    with pytest.raises(enroll.EnrollError):
        enroll.read_initiation(cose.dumps({"freshness_kind": "absent-none", "mode": "enroll", "handle": bytes(32)}))


def test_tiers_and_result_round_trip():
    assert [ear.tier_of(v) for v in (0, 1, -1, 2, 31, -2, -32, 32, 95, -33, -96, 96, 127, -97, -128)] == \
        [ear.NONE] * 3 + [ear.AFFIRMING] * 4 + [ear.WARNING] * 4 + [ear.CONTRAINDICATED] * 4
    key = Ed25519PrivateKey.generate()
    root = pki.make_root(key, "test authority", "test")
    sub = {"cpu": ear.appraisal({ear.HARDWARE: 2, ear.CONFIGURATION: 2}, ["p"]),
           "gpu0": ear.appraisal({ear.HARDWARE: 2, ear.EXECUTABLES: 33}, ["p"])}
    c = ear.claims(sub, ("dev", "build"), nonce=bytes(32), lifetime=60, topology={"cpu": ["gpu0"]}, ep1={"k": 1})
    assert c[ear.STATUS] == ear.WARNING                      # the worst submod
    signed = ear.sign(c, key, [pki.der(root)])
    back = ear.read(signed, root.public_key())
    assert back[ear.SUBMODS]["gpu0"][ear.STATUS] == ear.WARNING and back[ear.EP1] == {"k": 1}
    with pytest.raises(cose.COSEError):
        ear.read(signed, Ed25519PrivateKey.generate().public_key())


def test_certificates():
    key = Ed25519PrivateKey.generate()
    root = pki.make_root(key, "test authority", "test")
    sealer = Ed25519PrivateKey.generate()
    csr = pki.make_csr(sealer, "urn:ep1:sealer:x/seal")
    pub, cn = pki.read_csr(csr)
    leaf = pki.der(pki.issue(key, root, pub, cn, lifetime=600, eku=pki.STATEMENT_EKU))
    now = dt.datetime.now(dt.timezone.utc)
    assert pki.spki(pki.check_leaf(leaf, root, now, pki.STATEMENT_EKU)) == pki.spki(sealer.public_key())
    with pytest.raises(pki.PKIError):
        pki.check_leaf(leaf, root, now + dt.timedelta(hours=1), pki.STATEMENT_EKU)
    other = pki.make_root(Ed25519PrivateKey.generate(), "test authority", "test")
    with pytest.raises(pki.PKIError):
        pki.check_leaf(leaf, other, now, pki.STATEMENT_EKU)
