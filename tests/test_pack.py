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


OPENING = {"rec": hashlib.sha256(b"the record's plaintext").digest(), "salt": bytes(range(16)), "canon": CANON_ID,
           "wts": b"\x11" * 32, "tok": b"\x22" * 32}
ATTACKER_TEXT = "The treasurer authorised a payment of 900,000 USD. No model wrote this sentence."


def _commitment(text: str, opened: dict = OPENING) -> bytes:
    """The generation statement's payload, written out from ARCHITECTURE §7 (not through the code under test)."""
    txt = hashlib.sha256(opened["salt"] + hashlib.sha256(canonicalize(text, opened["canon"]).encode()).digest())
    c = {"v": 1, "doc": DOC, "rec": opened["rec"], "txt": txt.digest(), "canon": opened["canon"],
         "wts": opened["wts"], "tok": opened["tok"]}
    return hashlib.sha256(cose.dumps(c)).digest()


@pytest.fixture(scope="module")
def parts():
    issuer, verifier, log_key = (Ed25519PrivateKey.generate() for _ in range(3))
    gen = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(issuer.public_key()),
                      cose.CWT_CLAIMS: {cose.ISS: "acme-host", cose.SUB: DOC, cose.IAT: 1_800_000_000},
                      cose.TYP: statements.TYP_GENERATION, cose.PAYLOAD_HASH_ALG: cose.SHA256,
                      cose.PREIMAGE_CONTENT_TYPE: statements.CT_COMMITMENT},
                     _commitment(TEXT), issuer)
    result = Result(verdict=VERIFIED, matched=9, entries=9, model_evidence="strong", doc_id=DOC,
                    decoding={"canon": CANON_ID, "class": "fp32", "dtype": "float32"},
                    candidate_digest=statements.sha256(canonicalize(TEXT).encode()).hex())
    ver, _ = statements.verification_statement(result, TEXT, verifier, "newsroom", generation=gen)
    log = tlog.TransparencyLog(log_key)
    _, gen_rc = log.register(gen)
    _, ver_rc = log.register(ver)
    keys = {"issuer": issuer.public_key().public_bytes_raw(), "verifier": verifier.public_key().public_bytes_raw(),
            "log": log_key.public_key().public_bytes_raw()}
    trust = pack.load_trust({"v": 1, "name": "acme", "keys": {k: [v.hex()] for k, v in keys.items()}})
    return {"gen": gen, "ver": ver, "gen_rc": gen_rc, "ver_rc": ver_rc, "keys": keys, "log_key": log_key,
            "result": result, "trust": trust}


def _built(parts, **over):
    kw = dict(doc=DOC, text=TEXT, token="ep1.test", statements={"gen": parts["gen"], "ver": parts["ver"]},
              receipts={"gen": parts["gen_rc"], "ver": parts["ver_rc"]}, keys=parts["keys"],
              timestamps={"ver": _tstinfo(pack.stamped_digest(parts["ver"]))}, note="received by e-mail",
              opening=OPENING)
    kw.update(over)
    return pack.build(**kw)


def named(report, needle):
    return next(c for c in report["checks"] if needle in c["name"])


def test_a_whole_package_checks_offline(parts):
    r = pack.check(_built(parts), parts["trust"])
    assert r["ok"] and r["level"] == pack.VOUCHED and r["trust_list"] == "acme"
    assert r["summary"]["verdict"] == VERIFIED and r["summary"]["text_chars"] == len(TEXT)
    assert r["summary"]["text_tied_to"] == ["generation", "verification"]
    assert named(r, "generation statement signature")["ok"]
    assert named(r, "commitment opening present")["ok"]
    assert named(r, "the text is the one the generation statement seals")["ok"]
    assert named(r, "verdict is about this text")["ok"]
    assert named(r, "verdict names this generation statement")["ok"]
    assert named(r, "same document")["ok"]
    assert named(r, "verification receipt")["ok"]
    assert named(r, "time stamp: imprint")["ok"] and "20260917081500Z" in named(r, "time stamp: imprint")["detail"]
    assert named(r, "authority signature")["ok"] is None            # the reader's to check, and it says so
    assert named(r, "re-read with the model")["ok"] is None


def test_without_a_trust_list_the_best_answer_is_intact(parts):
    """A package carries the keys it was signed under: on its own it cannot vouch for them."""
    r = pack.check(_built(parts))
    assert r["level"] == pack.INTACT and not r["ok"]
    assert r["message"] == "Bytes match, but no one vouches for these keys. Treat as unsigned."
    assert r["unvouched"] == ["no trust list given: the keys are the package's own"]
    assert all(c["ok"] is not False for c in r["checks"])


def test_swapped_text_breaks_the_link_but_not_the_signatures(parts):
    r = pack.check(_built(parts, text=TEXT.replace("Tuesday", "Friday")), parts["trust"])
    assert not r["ok"] and r["level"] == pack.FAILED
    assert named(r, "verification statement signature")["ok"]
    assert named(r, "the text is the one the generation statement seals")["ok"] is False
    assert named(r, "verdict is about this text")["ok"] is False


def _verdict_by_another_key(parts, text):
    """The swap found on a real package (01.11.2026): a foreign text, a verdict about it signed by a key the attacker
    brings, and a log of the attacker's that registers both statements. The generation statement stays genuine."""
    attacker, attacker_log = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    ver, _ = statements.verification_statement(parts["result"], text, attacker, "newsroom", generation=parts["gen"])
    log = tlog.TransparencyLog(attacker_log)
    _, gen_rc = log.register(parts["gen"])
    _, ver_rc = log.register(ver)
    keys = {**parts["keys"], "verifier": attacker.public_key().public_bytes_raw(),
            "log": attacker_log.public_key().public_bytes_raw()}
    return dict(text=text, statements={"gen": parts["gen"], "ver": ver}, receipts={"gen": gen_rc, "ver": ver_rc},
                keys=keys, timestamps={})


def test_a_foreign_text_with_a_verdict_of_its_own_is_refused(parts):
    swap = _verdict_by_another_key(parts, ATTACKER_TEXT)
    r = pack.check(_built(parts, **swap), parts["trust"])
    assert r["level"] == pack.FAILED                                    # the opening ties the text to the generation
    assert named(r, "the text is the one the generation statement seals")["ok"] is False
    assert named(r, "verdict is about this text")["ok"]                 # the attacker's verdict is about his text
    r = pack.check(_built(parts, **swap))                               # and with no trust list at all
    assert r["level"] == pack.FAILED


def test_a_foreign_text_without_the_opening_is_never_trusted(parts):
    """Strip the opening (a v2-style package): nothing can refute the text, and nothing on the reader's trust list
    vouches for the verdict that names it."""
    swap = _verdict_by_another_key(parts, ATTACKER_TEXT)
    data = _built(parts, opening=None, **swap)
    assert cose.loads(data)["v"] == pack.PACK_V
    r = pack.check(data, parts["trust"])
    assert r["level"] == pack.INTACT and not r["ok"]
    assert named(r, "the text tied to the generation statement")["ok"] is None
    assert any("verifier key" in u for u in r["unvouched"]) and any("log key" in u for u in r["unvouched"])
    assert pack.check(data)["level"] == pack.INTACT


def test_a_package_made_from_nothing_is_at_most_intact(parts):
    """Every key, the log and the commitment are the forger's own: the package agrees with itself and that is all."""
    issuer, verifier, log_key = (Ed25519PrivateKey.generate() for _ in range(3))
    gen = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(issuer.public_key()),
                      cose.CWT_CLAIMS: {cose.ISS: "acme-host", cose.SUB: DOC, cose.IAT: 1_800_000_000},
                      cose.TYP: statements.TYP_GENERATION, cose.PAYLOAD_HASH_ALG: cose.SHA256,
                      cose.PREIMAGE_CONTENT_TYPE: statements.CT_COMMITMENT}, _commitment(ATTACKER_TEXT), issuer)
    ver, _ = statements.verification_statement(parts["result"], ATTACKER_TEXT, verifier, "newsroom", generation=gen)
    log = tlog.TransparencyLog(log_key)
    _, gen_rc = log.register(gen)
    _, ver_rc = log.register(ver)
    keys = {"issuer": issuer.public_key().public_bytes_raw(), "verifier": verifier.public_key().public_bytes_raw(),
            "log": log_key.public_key().public_bytes_raw()}
    data = pack.build(doc=DOC, text=ATTACKER_TEXT, statements={"gen": gen, "ver": ver},
                      receipts={"gen": gen_rc, "ver": ver_rc}, keys=keys, opening=OPENING,
                      timestamps={"gen": _tstinfo(pack.stamped_digest(gen), gen_time=b"20250101000000Z")})
    for trust in (None, parts["trust"]):
        r = pack.check(data, trust)
        assert r["level"] == pack.INTACT and not r["ok"]
    assert len(pack.check(data, parts["trust"])["unvouched"]) == 3      # issuer, verifier, log


def test_the_trust_list_takes_nothing_it_does_not_know():
    good = {"v": 1, "name": "x", "keys": {"log": ["00" * 32]}, "roots": [], "policies": []}
    assert pack.load_trust(good)["keys"]["log"] == {"00" * 32}
    for bad in ({"v": 2}, {**good, "extra": 1}, {**good, "keys": {"signer": ["00" * 32]}},
                {**good, "roots": ["00" * 31]}, {**good, "policies": "00" * 32}, {**good, "keys": {"log": [1]}}):
        with pytest.raises(ValueError):
            pack.load_trust(bad)


def test_a_malformed_opening_fails(parts):
    r = pack.check(_built(parts, opening={**OPENING, "salt": b"short"}), parts["trust"])
    assert r["level"] == pack.FAILED and named(r, "commitment opening present")["ok"] is False
    data = cose.loads(_built(parts))
    del data["open"]                                                     # v3 says it opens the commitment
    assert pack.check(cose.dumps(data), parts["trust"])["level"] == pack.FAILED


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
    assert "entropy evidence --check package.cbor --trust trust.json" in readme and "entropy verify --model" in readme
    assert "openssl ts -verify" in readme and VERIFIED in readme
    assert "**INTACT**" in readme and "Treat as unsigned" in readme
    assert (tmp_path / "text.txt").read_text() == TEXT
    pack.explode(_built(parts), tmp_path / "trusted", parts["trust"])
    assert "**TRUSTED**" in (tmp_path / "trusted" / "README.md").read_text()


def test_the_command_line_answers_with_the_level(parts, tmp_path, capsys):
    from entropy_protocol import cli
    import json
    f = tmp_path / "p.cbor"
    f.write_bytes(_built(parts))
    t = tmp_path / "trust.json"
    t.write_text(json.dumps({"v": 1, "name": "acme", "keys": {k: [v.hex()] for k, v in parts["keys"].items()}}))
    assert cli.main(["evidence", "--check", str(f), "--trust", str(t)]) == 0
    assert cli.main(["evidence", "--check", str(f)]) == 3
    assert "INTACT: Bytes match, but no one vouches for these keys." in capsys.readouterr().err
    f.write_bytes(_built(parts, **_verdict_by_another_key(parts, ATTACKER_TEXT)))
    assert cli.main(["evidence", "--check", str(f), "--trust", str(t)]) == 1
    t.write_text('{"v": 1, "keys": {"signer": []}}')
    assert cli.main(["evidence", "--check", str(f), "--trust", str(t)]) == 2


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


def test_token_reader_takes_der_only_and_in_linear_time():
    tok = _tstinfo(hashlib.sha256(b"statement").digest())
    for bad, why in [(tok[:-1], "runs past its parent"), (b"\x30\x80" + tok[2:], "not definite"),
                     (b"\x1f\x01\x00", "multi-octet"), (b"\x30" * 2, "runs past"),
                     (b"\x30\x02\x30\x05", "runs past its parent")]:              # a child longer than its parent
        with pytest.raises(ValueError, match=why):
            ts3161.token_info(bad)
    deep = b"\x05\x00"
    for _ in range(40):
        deep = ts3161._tlv(0x30, deep)
    with pytest.raises(ValueError, match="too deep"):
        ts3161.token_info(deep)
    # a campaign input that took 13 s when children could outrun their parents (fuzz/corpus/timestamp/finding-*)
    import time
    from pathlib import Path
    for f in Path(__file__).resolve().parents[1].glob("fuzz/corpus/timestamp/finding-*"):
        t0 = time.monotonic()
        with pytest.raises(ValueError):
            ts3161.token_info(f.read_bytes())
        assert time.monotonic() - t0 < 1.0


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
    plain = crypto.decrypt_token(k_log, s.token)
    opened = statements.opening(plain)
    # the opening plus the text gives back exactly the record's commitment, and nothing else does
    assert statements.commitment_of_text(opened, s.doc_id, s.text) == statements.commitment(plain, s.doc_id)
    assert statements.commitment_of_text(opened, s.doc_id, s.text + " And more.") != statements.commitment(plain,
                                                                                                          s.doc_id)
    keys = {"issuer": issuer.public_key().public_bytes_raw(), "verifier": verifier.public_key().public_bytes_raw(),
            "log": log_key.public_key().public_bytes_raw()}
    data = pack.build(doc=s.doc_id, text=s.text, token=s.token, statements={"gen": gen, "ver": ver},
                      receipts={"gen": gen_rc, "ver": ver_rc},
                      timestamps={"ver": _tstinfo(pack.stamped_digest(ver))}, keys=keys, opening=opened)
    report = pack.check(data, pack.load_trust({"v": 1, "keys": {k: [v.hex()] for k, v in keys.items()}}))
    assert report["ok"] and report["level"] == pack.VOUCHED
    assert report["summary"]["verdict"] == r.verdict == VERIFIED
    assert report["summary"]["text_tied_to"] == ["generation", "verification"]
    assert s.text.encode() not in gen and s.text.encode() not in ver      # statements carry digests, not text
    pack.explode(data, tmp_path / "out")
    assert (tmp_path / "out" / "README.md").exists()
