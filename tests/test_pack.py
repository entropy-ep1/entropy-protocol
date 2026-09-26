"""The evidence package (`entropy evidence`) and the RFC 3161 time stamp it carries: what a third party can check
offline, and what has to fail when a part is swapped."""
import hashlib

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from entropy_protocol import cose, pack, statements, tlog, ts3161
from entropy_protocol.canon import CANON_ID, canonicalize
from entropy_protocol.verify import VERIFIED, Result

TEXT = "The committee reviewed the evidence and published its finding on Tuesday."
DOC = "doc-evidence-1"


def _tstinfo(digest: bytes, *, gen_time: bytes = b"20260917081500Z", serial: int = 7) -> bytes:
    """A TSTInfo as an authority signs it (the CMS wrapping around it is what we do not validate)."""
    alg = ts3161._tlv(0x30, ts3161._tlv(0x06, ts3161.SHA256_OID) + ts3161._tlv(0x05, b""))
    info = ts3161._tlv(0x30, ts3161._int(1) + ts3161._tlv(0x06, bytes.fromhex("2a03040506"))
                       + ts3161._tlv(0x30, alg + ts3161._tlv(0x04, digest))
                       + ts3161._int(serial) + ts3161._tlv(0x18, gen_time))
    return ts3161._tlv(0x30, ts3161._tlv(0x06, ts3161.TSTINFO_OID) + ts3161._tlv(0xA0, ts3161._tlv(0x04, info)))


@pytest.fixture(scope="module")
def parts():
    issuer, verifier, log_key = (Ed25519PrivateKey.generate() for _ in range(3))
    gen = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(issuer.public_key()),
                      cose.CWT_CLAIMS: {cose.ISS: "acme-host", cose.SUB: DOC, cose.IAT: 1_800_000_000},
                      cose.TYP: statements.TYP_GENERATION, cose.PAYLOAD_HASH_ALG: cose.SHA256,
                      cose.PREIMAGE_CONTENT_TYPE: statements.CT_COMMITMENT},
                     hashlib.sha256(b"commitment").digest(), issuer)
    result = Result(verdict=VERIFIED, matched=9, entries=9, model_evidence="strong", doc_id=DOC,
                    decoding={"canon": CANON_ID, "class": "fp32", "dtype": "float32"},
                    candidate_digest=statements.sha256(canonicalize(TEXT).encode()).hex())
    ver, _ = statements.verification_statement(result, TEXT, verifier, "newsroom", generation=gen)
    log = tlog.TransparencyLog(log_key)
    _, gen_rc = log.register(gen)
    _, ver_rc = log.register(ver)
    keys = {"issuer": issuer.public_key().public_bytes_raw(), "verifier": verifier.public_key().public_bytes_raw(),
            "log": log_key.public_key().public_bytes_raw()}
    return {"gen": gen, "ver": ver, "gen_rc": gen_rc, "ver_rc": ver_rc, "keys": keys, "log_key": log_key}


def _built(parts, **over):
    kw = dict(doc=DOC, text=TEXT, token="ep1.test", statements={"gen": parts["gen"], "ver": parts["ver"]},
              receipts={"gen": parts["gen_rc"], "ver": parts["ver_rc"]}, keys=parts["keys"],
              timestamps={"ver": _tstinfo(pack.stamped_digest(parts["ver"]))}, note="received by e-mail")
    kw.update(over)
    return pack.build(**kw)


def named(report, needle):
    return next(c for c in report["checks"] if needle in c["name"])


def test_a_whole_package_checks_offline(parts):
    r = pack.check(_built(parts))
    assert r["ok"] and r["summary"]["verdict"] == VERIFIED and r["summary"]["text_chars"] == len(TEXT)
    assert named(r, "generation statement signature")["ok"]
    assert named(r, "verdict is about this text")["ok"]
    assert named(r, "verdict names this generation statement")["ok"]
    assert named(r, "same document")["ok"]
    assert named(r, "verification receipt")["ok"]
    assert named(r, "time stamp: imprint")["ok"] and "20260917081500Z" in named(r, "time stamp: imprint")["detail"]
    assert named(r, "authority signature")["ok"] is None            # the reader's to check, and it says so
    assert named(r, "re-read with the model")["ok"] is None


def test_swapped_text_breaks_the_link_but_not_the_signatures(parts):
    r = pack.check(_built(parts, text=TEXT.replace("Tuesday", "Friday")))
    assert not r["ok"]
    assert named(r, "verification statement signature")["ok"]
    assert named(r, "verdict is about this text")["ok"] is False


def test_wrong_key_wrong_receipt_wrong_stamp_all_fail(parts):
    other = Ed25519PrivateKey.generate().public_key().public_bytes_raw()
    r = pack.check(_built(parts, keys={**parts["keys"], "verifier": other}))
    assert named(r, "verification statement signature")["ok"] is False
    r = pack.check(_built(parts, receipts={"ver": parts["gen_rc"]}))   # the generation statement's receipt
    assert named(r, "verification receipt")["ok"] is False
    r = pack.check(_built(parts, timestamps={"ver": _tstinfo(hashlib.sha256(b"another statement").digest())}))
    assert named(r, "time stamp: imprint")["ok"] is False
    r = pack.check(_built(parts, keys={k: v for k, v in parts["keys"].items() if k != "log"}))
    assert named(r, "verification receipt")["ok"] is False and "no log key" in named(r, "verification receipt")["detail"]


def test_explode_writes_the_parts_and_a_readme(parts, tmp_path):
    written = pack.explode(_built(parts), tmp_path)
    assert {"text.txt", "README.md", "verification_statement.cose", "verification_timestamp.tsr",
            "key_issuer.ed25519"} <= set(written)
    readme = (tmp_path / "README.md").read_text()
    assert "entropy evidence --check" in readme and "entropy verify --model" in readme
    assert "openssl ts -verify" in readme and VERIFIED in readme
    assert (tmp_path / "text.txt").read_text() == TEXT


def test_request_and_token_reading():
    d = hashlib.sha256(b"statement").digest()
    req = ts3161.request(d, nonce=bytes(range(16)))
    assert req[:2] == b"\x30\x42" or req[0] == 0x30                  # DER SEQUENCE
    assert d.hex() in req.hex() and req.endswith(b"\x01\x01\xff")    # imprint present, certReq TRUE
    with pytest.raises(ValueError):
        ts3161.request(b"short")
    info = ts3161.token_info(_tstinfo(d, serial=99))
    assert info == {"version": 1, "policy": "1.2.3.4.5.6", "hash": "sha-256", "imprint": d.hex(),
                    "serial": 99, "gen_time": "20260917081500Z"}
    granted = ts3161._tlv(0x30, ts3161._tlv(0x30, ts3161._int(0)) + _tstinfo(d))
    assert ts3161.token_info(ts3161.token_from_response(granted))["imprint"] == d.hex()
    rejected = ts3161._tlv(0x30, ts3161._tlv(0x30, ts3161._int(2) + ts3161._tlv(0x0C, b"policy not supported")))
    with pytest.raises(ValueError, match="policy not supported"):
        ts3161.token_from_response(rejected)


@pytest.mark.model
def test_the_whole_chain_from_sealing_to_the_package(tmp_path):
    pytest.importorskip("torch")
    from entropy_protocol import crypto
    from entropy_protocol.seal import SealParams, seal
    from entropy_protocol.verify import verify
    try:
        from entropy_protocol.backend_hf import HFBackend
        b = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    master = crypto.new_master_key()
    s = seal(b, "The history of the bridge", master, SealParams(max_new_tokens=80), seed=3)
    issuer, verifier, log_key = (Ed25519PrivateKey.generate() for _ in range(3))
    k_log = crypto.log_key(master, s.doc_id)
    gen, _ = statements.generation_statement(k_log, s.token, issuer, "acme-host")
    r = verify(b, s.text, s.token, k_log)
    ver, _ = statements.verification_statement(r, s.text, verifier, "newsroom", generation=gen)
    log = tlog.TransparencyLog(log_key)
    _, gen_rc = log.register(gen)
    _, ver_rc = log.register(ver)
    data = pack.build(doc=s.doc_id, text=s.text, token=s.token, statements={"gen": gen, "ver": ver},
                      receipts={"gen": gen_rc, "ver": ver_rc},
                      timestamps={"ver": _tstinfo(pack.stamped_digest(ver))},
                      keys={"issuer": issuer.public_key().public_bytes_raw(),
                            "verifier": verifier.public_key().public_bytes_raw(),
                            "log": log_key.public_key().public_bytes_raw()})
    report = pack.check(data)
    assert report["ok"] and report["summary"]["verdict"] == r.verdict == VERIFIED
    assert s.text.encode() not in gen and s.text.encode() not in ver      # statements carry digests, not text
    pack.explode(data, tmp_path / "out")
    assert (tmp_path / "out" / "README.md").exists()
