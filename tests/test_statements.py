"""COSE_Sign1, COSE Hash Envelope statements, RFC 9162 Merkle log and RFC 9942 receipts."""
import cbor2
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from entropy_protocol import cose, tlog


def test_merkle_inclusion_all_indices_and_tamper():
    for n in range(1, 34):
        leaves = [tlog.leaf_hash(bytes([i, n])) for i in range(n)]
        root = tlog.mth(leaves)
        for i in range(n):
            path = tlog.inclusion_path(i, leaves)
            assert tlog.root_from_inclusion(leaves[i], i, n, path) == root
        if n > 1:
            path = tlog.inclusion_path(0, leaves)
            bad = [bytes(32)] + path[1:]
            assert tlog.root_from_inclusion(leaves[0], 0, n, bad) != root
            with pytest.raises(ValueError):
                tlog.root_from_inclusion(leaves[0], n, n, path)


def test_rfc9162_reference_values():
    # RFC 9162 section 2.1.1: MTH of the empty list is SHA-256 of the empty string; a single leaf is H(0x00 || d)
    import hashlib
    assert tlog.mth([]) == hashlib.sha256(b"").digest()
    assert tlog.leaf_hash(b"") == hashlib.sha256(b"\x00").digest()


def test_cose_sign1_roundtrip_detached_and_tamper():
    k = Ed25519PrivateKey.generate()
    prot = {cose.ALG: cose.ED25519, cose.KID: cose.kid_of(k.public_key()),
            cose.CWT_CLAIMS: {cose.ISS: "issuer-a", cose.SUB: "doc-1", cose.IAT: 1_800_000_000}}
    msg = cose.sign1(prot, b"payload", k)
    p, u, body = cose.verify1(msg, k.public_key())
    assert body == b"payload" and p[cose.CWT_CLAIMS][cose.SUB] == "doc-1"
    det = cose.sign1(prot, b"payload", k, detached=True)
    assert cose.decode1(det)[3] is None
    cose.verify1(det, k.public_key(), payload=b"payload")
    with pytest.raises(cose.COSEError):
        cose.verify1(det, k.public_key(), payload=b"other")
    parts = list(cbor2.loads(msg).value)
    parts[0] = cose.dumps({**prot, cose.KID: b"x" * 16})                     # altered protected header
    with pytest.raises(cose.COSEError):
        cose.verify1(cbor2.dumps(cbor2.CBORTag(18, parts)), k.public_key())
    with pytest.raises(cose.COSEError):
        cose.verify1(msg, Ed25519PrivateKey.generate().public_key())


def test_deterministic_map_order_is_rfc8949_bytewise():
    maps = [{1: -19, 4: b"k" * 16, 15: {1: "i", 2: "s", 6: 1}, 16: "application/x+cose", 258: -16, 259: "a/b"},
            {"v": 1, "doc": "d", "rec": b"r", "txt": b"t", "canon": "c", "wts": b"w", "tok": b"k"},
            {"v": 1, "verdict": "VERIFIED", "doc": "d", "cand": b"c", "gen": None, "counts": [1, 2, 3, 4],
             "ranges": {"changed": [], "deleted": [], "divergent": []}, "wts": b"", "tok": b"",
             "dec": {"class": "fp32", "dtype": "float32", "space": "logit", "tol_u": 0, "canon": "c",
                     "calibrated": None}, "nonce": "n", "engine": "e"},
            {396: {-1: [b"proof"]}}]
    for m in maps:
        enc = cose.dumps(m)
        keys = list(cbor2.loads(enc).keys())
        assert keys == sorted(keys, key=lambda x: cbor2.dumps(x)), keys


def test_log_receipt_verifies_and_binds_the_statement():
    issuer, log_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    log = tlog.TransparencyLog(log_key)
    stmts = [cose.sign1({cose.ALG: cose.ED25519}, f"s{i}".encode(), issuer) for i in range(7)]
    receipts = [log.register(s)[1] for s in stmts]
    for s, r in zip(stmts, receipts):
        info = tlog.verify_receipt(r, s, log_key.public_key())
        assert info["tree_size"] >= 1
    fresh = log.receipt(2)                                                   # a receipt at the current size
    assert tlog.verify_receipt(fresh, stmts[2], log_key.public_key())["tree_size"] == 7
    with pytest.raises((cose.COSEError, ValueError)):
        tlog.verify_receipt(fresh, stmts[3], log_key.public_key())
    # the unprotected header of a registered statement does not change its entry
    parts = list(cbor2.loads(stmts[2]).value)
    parts[1] = {cose.RECEIPTS: [fresh]}
    assert tlog.entry_of(cbor2.dumps(cbor2.CBORTag(18, parts))) == tlog.entry_of(stmts[2])


def test_log_receipt_refuses_malformed_inclusion_proofs():
    issuer, log_key = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    log = tlog.TransparencyLog(log_key)
    stmts = [cose.sign1({cose.ALG: cose.ED25519}, f"s{i}".encode(), issuer) for i in range(3)]
    receipt = [log.register(s)[1] for s in stmts][-1]
    parts = list(cbor2.loads(receipt).value)

    def with_vdp(vdp):                       # the unprotected header is not signed: change it, keep the signature
        return cbor2.dumps(cbor2.CBORTag(18, [parts[0], {cose.VDP: vdp}, parts[2], parts[3]]))
    assert tlog.verify_receipt(with_vdp(parts[1][cose.VDP]), stmts[2], log_key.public_key())["leaf_index"] == 2
    size, index, path = cbor2.loads(parts[1][cose.VDP][tlog.INCLUSION][0])
    for vdp in (5, {tlog.INCLUSION: b"x"}, {tlog.INCLUSION: [5]}, {tlog.INCLUSION: [cbor2.dumps(5)]},   # the fuzz finding
                {tlog.INCLUSION: [cbor2.dumps([size, -1, list(path)])]},
                {tlog.INCLUSION: [cbor2.dumps([size, index, [b"x"]])]}):
        with pytest.raises(cose.COSEError, match="inclusion proof"):
            tlog.verify_receipt(with_vdp(vdp), stmts[2], log_key.public_key())


@pytest.mark.model
def test_statements_end_to_end():
    pytest.importorskip("torch")
    from entropy_protocol import crypto, statements
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.detect import detect
    from entropy_protocol.seal import SealParams, seal
    from entropy_protocol.verify import MODIFIED, VERIFIED, verify
    try:
        b = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    master = bytes(range(32))
    s = seal(b, "The history of the printing press shows", master, SealParams(max_new_tokens=160), seed=9)
    k_log = crypto.log_key(master, s.doc_id)
    issuer, verifier, log_key = (Ed25519PrivateKey.generate() for _ in range(3))
    gen_stmt, commitment = statements.generation_statement(k_log, s.token, issuer, "generator.example")
    assert b"printing" not in gen_stmt and s.text[:20].encode() not in gen_stmt     # no text in the statement
    log = tlog.TransparencyLog(log_key)
    _, receipt = log.register(gen_stmt)
    tlog.verify_receipt(receipt, gen_stmt, log_key.public_key())
    tok_v, k_v = crypto.reissue_for_verifier(master, s.token, "court-1")
    # the verifier recomputes the commitment from its own copy of the record
    assert statements.check_generation_statement(gen_stmt, issuer.public_key(), k_v, tok_v)["issuer"] == "generator.example"
    r = verify(b, s.text, tok_v, k_v)
    assert r.verdict == VERIFIED
    ver_stmt, body = statements.verification_statement(r, s.text, verifier, "court-1", generation=gen_stmt)
    got = statements.read_statement(ver_stmt, verifier.public_key())
    assert got["payload"]["verdict"] == VERIFIED and got["payload"]["gen"] == statements.statement_ref(gen_stmt)
    assert got["claims"][cose.SUB] == s.doc_id
    edited = s.text.replace(" the ", " a ", 1)
    r2 = verify(b, edited, tok_v, k_v)
    env_stmt, env_body = statements.verification_statement(r2, edited, verifier, "court-1", envelope=True)
    assert r2.verdict == MODIFIED
    assert statements.read_statement(env_stmt, verifier.public_key(), env_body)["payload"]["verdict"] == MODIFIED
    with pytest.raises(cose.COSEError):
        statements.read_statement(env_stmt, verifier.public_key(), {**env_body, "verdict": VERIFIED})
    d = detect(b, s.text, crypto.wm_key(master))
    det_stmt, det_body = statements.detection_statement(d, s.text, b, verifier, "detector-1")
    assert statements.read_statement(det_stmt, verifier.public_key())["payload"]["technique"] == "ep1-wm/k1-i5-c4"


def test_scrapi_service_register_resolve_and_policy():
    from entropy_protocol import scrapi
    issuer, stranger, log_key = (Ed25519PrivateKey.generate() for _ in range(3))
    ts = scrapi.TransparencyService(tlog.TransparencyLog(log_key), [issuer.public_key()])
    server = scrapi.serve(ts)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        stmt = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(issuer.public_key())}, b"hello", issuer)
        location, receipt = scrapi.register(url, stmt)
        assert location.startswith("/entries/")
        info = scrapi.verify_receipt_online(url, receipt, stmt)
        assert info["tree_size"] == 1
        # resolve again later: a fresh receipt at the current tree size
        stmt2 = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(issuer.public_key())}, b"again", issuer)
        scrapi.register(url, stmt2)
        _, _, again = scrapi._request(url + location)
        assert scrapi.verify_receipt_online(url, again, stmt)["tree_size"] == 2
        # registration policy: an unregistered issuer and a forged signature are refused with problem details
        foreign = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(stranger.public_key())}, b"x", stranger)
        with pytest.raises(scrapi.SCRAPIError) as err:
            scrapi.register(url, foreign)
        assert err.value.status == 400 and "policy" in err.value.title
        forged = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(issuer.public_key())}, b"x", stranger)
        with pytest.raises(scrapi.SCRAPIError) as err:
            scrapi.register(url, forged)
        assert err.value.status == 400
        with pytest.raises(scrapi.SCRAPIError) as err:
            scrapi._request(url + "/entries/0000")
        assert err.value.status == 404
        kid = cose.kid_of(log_key.public_key())
        _, _, one = scrapi._request(url + "/.well-known/scitt-keys/" + kid.hex(), accept="application/cbor")
        assert scrapi.public_key_from_cose(dict(cbor2.loads(one))) is not None
    finally:
        server.shutdown()
