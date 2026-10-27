"""Attested sealing (ARCHITECTURE §7d): the parts that need no hardware - the binding value's encoding (a fixed
vector, so another implementation can match it), the handle book, the enrollment messages, the attestation result's
claims and tiers, the authority's certificates, and the evidence readers' refusals of what they cannot use."""
from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from entropy_protocol import cose
from entropy_protocol.attested import binding, ear, enroll, pki, rim, tpm


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


def _event_log(banks: list, events: list) -> bytes:
    """A crypto-agile TCG event log: the Spec ID event announcing ``banks`` [(alg, size)], then ``events``
    [(pcr, type, [(alg, digest)], data)]."""
    le = lambda v, n: v.to_bytes(n, "little")  # noqa: E731
    spec = (b"Spec ID Event03\x00" + le(0, 4) + bytes([0, 2, 0]) + le(2, 1) + le(len(banks), 4) +
            b"".join(le(a, 2) + le(n, 2) for a, n in banks) + b"\x00")
    out = le(0, 4) + le(tpm.EV_NO_ACTION, 4) + bytes(20) + le(len(spec), 4) + spec
    for pcr, etype, digests, data in events:
        out += le(pcr, 4) + le(etype, 4) + le(len(digests), 4) + b"".join(le(a, 2) + d for a, d in digests)
        out += le(len(data), 4) + data
    return out


def test_event_log_refuses_what_replay_cannot_use():
    s256, d256 = (tpm.ALG_SHA256, 32), (tpm.ALG_SHA256, bytes(32))
    log = tpm.parse_event_log(_event_log([s256], [(4, 0x0D, [d256], b"x")]))
    assert tpm.replay(log)[4] == hashlib.sha256(bytes(64)).digest()
    for blob, why in [(_event_log([s256, s256], []), "announced twice"),
                      (_event_log([(tpm.ALG_SHA256, 20)], []), "wrong digest size"),
                      (_event_log([s256, (tpm.ALG_SHA1, 20)], [(4, 0x0D, [d256], b"x")]), "lacks"),
                      (_event_log([s256], [(4, 0x0D, [d256, d256], b"x")]), "given twice"),
                      (_event_log([s256], [(0, tpm.EV_NO_ACTION, [d256], b"StartupLocality\x00")]), "17 octets")]:
        with pytest.raises(tpm.TPMError, match=why):
            tpm.parse_event_log(blob)
    sha1_only = tpm.parse_event_log(_event_log([(tpm.ALG_SHA1, 20)], [(11, 0x0D, [(tpm.ALG_SHA1, bytes(20))], b"x")]))
    with pytest.raises(tpm.TPMError, match="not in the log"):
        tpm.replay(sha1_only)
    with pytest.raises(tpm.TPMError, match="no SHA-256 bank"):
        tpm.uki_sections(sha1_only)


def test_quote_values_and_systemd_records_refused_when_malformed():
    q = tpm.Quote(b"", b"", 0, 0, 0, 1, 0, [(0x010B, [0, 1])], bytes(32))     # a bank no TPM spec defines
    with pytest.raises(tpm.TPMError, match="not supported"):
        tpm.split_values(bytes(64), q)
    with pytest.raises(tpm.TPMError, match="not supported"):
        tpm.pcr_digest(q, {})
    q = tpm.Quote(b"", b"", 0, 0, 0, 1, 0, [(tpm.ALG_SHA256, [0, 1])], bytes(32))
    with pytest.raises(tpm.TPMError, match="no value"):
        tpm.pcr_digest(q, {(tpm.ALG_SHA256, 0): bytes(32)})

    def rec(pcr, digest):
        return b"\x1e" + json.dumps({"pcr": pcr, "digests": [{"hashAlg": "sha256", "digest": digest}]}).encode()
    assert tpm.parse_systemd_log(rec(11, "00" * 32) + b"\n") == [(11, bytes(32))]
    for bad in (rec(True, "00" * 32), rec(24, "00" * 32), rec(11, "00" * 20)):
        with pytest.raises(tpm.TPMError):
            tpm.parse_systemd_log(bad)


def test_jwk_reader_takes_base64url_rsa_keys_only():
    from cryptography.hazmat.primitives.asymmetric import rsa

    from entropy_protocol.attested import hcl
    pub = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key().public_numbers()

    def b64u(v: int) -> str:
        return __import__("base64").urlsafe_b64encode(v.to_bytes((v.bit_length() + 7) // 8, "big")).decode().rstrip("=")
    good = {"kty": "RSA", "e": b64u(pub.e), "n": b64u(pub.n)}
    assert hcl.rsa_from_jwk(good).public_numbers() == pub
    for bad in ({**good, "kty": "EC"}, {**good, "n": good["n"] + "AAA"}, {**good, "n": "+" + good["n"][1:]},
                {**good, "e": 3.0}, {"kty": "RSA", "n": good["n"]}, {**good, "e": b64u(2)}, "RSA"):
        with pytest.raises(hcl.HCLError):
            hcl.rsa_from_jwk(bad)


def test_unreadable_certificates_and_keys_are_refusals():
    from pathlib import Path

    from cryptography import x509
    from cryptography.hazmat.primitives import serialization

    from entropy_protocol.attested import amd, nvidia
    run = Path(__file__).resolve().parents[1] / "bench/results/attested/run-20261027T172258Z"
    thim = cose._thaw(cose.loads(enroll.read_request((run / "conduit/request-seal.cbor").read_bytes())["evidence"]))["thim"]
    vcek = x509.load_pem_x509_certificate(thim["vcekCert"].encode())
    ask, ark = x509.load_pem_x509_certificates(thim["certificateChain"].encode())[:2]
    der = lambda c: c.public_bytes(serialization.Encoding.DER)  # noqa: E731
    at = dt.datetime(2026, 10, 27, 17, 0, tzinfo=dt.timezone.utc)
    assert amd.verify_chain(der(vcek), der(ask), der(ark), "Genoa", at)
    with pytest.raises(amd.SNPError, match="not a DER certificate"):               # was a bare ValueError
        amd.verify_chain(der(vcek), der(ask)[:-7], der(ark), "Genoa", at)
    from entropy_protocol.attested import hcl
    report = hcl.parse((run.parent / "capture-20261027T162455Z/r0_hcl_report.bin").read_bytes()).snp
    with pytest.raises(amd.SNPError, match="cannot be read"):                      # a real report, no VCEK
        amd.verify_report(report, b"not a certificate")
    # a GPU chain whose parent's key type is unknown (1.2.844.10045.2.1, ClusterFuzzLite's mutation): read leaf first
    root_key = Ed25519PrivateKey.generate()
    root = pki.make_root(root_key, "test authority", "test")
    with pytest.raises(pki.PKIError):
        pki.check_leaf(b"\x30\x03\x02\x01\x00", root, at, pki.STATEMENT_EKU)
    with pytest.raises(pki.PKIError):
        pki.read_csr(b"\x30\x03\x02\x01\x00")
    leaf = x509.load_der_x509_certificate(der(vcek))
    odd = x509.load_der_x509_certificate(der(ask).replace(bytes.fromhex("06092a864886f70d010101"),
                                                          bytes.fromhex("06092a864886f70d01017f")))
    with pytest.raises(nvidia.GPUError, match="not signed by its issuer"):
        nvidia.verify_chain([leaf, odd], hashlib.sha256(der(odd)).hexdigest(), at)


def test_paravisor_report_claims_refused_when_malformed():
    from pathlib import Path

    from entropy_protocol.attested import hcl
    raw = (Path(__file__).resolve().parents[1] / "bench/results/attested/capture-20261027T162455Z/r0_hcl_report.bin"
           ).read_bytes()
    real = hcl.parse(raw)
    assert len(real.user_data) == 64 and real.jwk("HCLAkPub")["kty"] == "RSA"

    def with_claims(claims: bytes) -> bytes:                  # the same report with other runtime claims, sizes kept true
        rd = hcl.HEADER + hcl.SLOT
        start = rd + hcl.REQUEST_DATA
        out = bytearray(raw[:start] + claims)
        out[8:12] = len(out).to_bytes(4, "little")
        out[rd:rd + 4] = (hcl.REQUEST_DATA + len(claims)).to_bytes(4, "little")
        out[rd + 16:rd + 20] = len(claims).to_bytes(4, "little")
        return bytes(out)
    assert hcl.parse(with_claims(real.claims_raw)).user_data == real.user_data
    for claims in (b"[]", b"7", b"[" * 5000 + b"]" * 5000):
        with pytest.raises(hcl.HCLError, match="JSON"):
            hcl.parse(with_claims(claims))
    for user_data in ('"' + "ab" * 31 + 'gg"', '"abc"', "12", '"ab cd"'):
        with pytest.raises(hcl.HCLError, match="user-data"):
            _ = hcl.parse(with_claims(b'{"user-data": ' + user_data.encode() + b"}")).user_data
    assert hcl.parse(with_claims(b'{"keys": 5}')).jwk("HCLAkPub") is None
    assert hcl.parse(with_claims(b'{"keys": [5, {"kid": "HCLAkPub", "kty": "RSA"}]}')).jwk("HCLAkPub")["kty"] == "RSA"


def test_rim_golden_refuses_malformed_measurements():
    etree = pytest.importorskip("lxml.etree")

    def tree(attrs):
        return etree.fromstring(f'<SoftwareIdentity xmlns:h="{rim.HASH_NS[1:-1]}"><Payload><Resource {attrs}/>'
                                f'</Payload></SoftwareIdentity>'.encode())
    assert rim.golden(tree('index="1" size="48" alternatives="1" active="True" h:Hash0="AB"')) == \
        {1: {"active": True, "size": 48, "name": "", "values": ["ab"]}}
    for attrs in ('size="48" alternatives="1" h:Hash0="ab"', 'index="x" size="48" alternatives="1" h:Hash0="ab"',
                  'index="1" size="48" alternatives="2" h:Hash0="ab"'):
        with pytest.raises(rim.RIMError, match="malformed"):
            rim.golden(tree(attrs))
