"""Records delivered to a verifier by public key (COSE-HPKE integrated encryption, draft-ietf-cose-hpke-27)."""
import cbor2
import pytest

from entropy_protocol import cose, crypto, hpke
from entropy_protocol.delivery import CT_RECORD, deliver, open_delivered

MASTER = bytes(range(32))


def _token(doc_id="doc-delivery"):
    return crypto.encrypt_token(crypto.log_key(MASTER, doc_id), doc_id, b"\xa1\x01\x02", nonce=bytes(12))


def test_encrypt0_structure_and_round_trip():
    sk, pk = hpke.generate_keypair()
    data = cose.encrypt0_hpke({cose.CONTENT_TYPE: "text/plain"}, b"hello", pk)
    obj = cbor2.loads(data)
    assert obj.tag == cose.COSE_ENCRYPT0_TAG and len(obj.value) == 3
    prot = cbor2.loads(obj.value[0])
    assert prot[cose.ALG] == cose.HPKE_4 and len(obj.value[1][cose.EK]) == 32     # "ek" unprotected, raw X25519
    assert cose.decrypt0_hpke(data, sk) == ({cose.CONTENT_TYPE: "text/plain", cose.ALG: cose.HPKE_4}, b"hello")
    for alg in (cose.HPKE_3, cose.HPKE_4):
        assert cose.decrypt0_hpke(cose.encrypt0_hpke({}, b"x", pk, alg=alg), sk)[1] == b"x"


def test_delivery_opens_only_for_the_verifier_and_binds_the_document():
    sk, pk = hpke.generate_keypair()
    other_sk, _ = hpke.generate_keypair()
    token = _token()
    data = deliver(crypto.log_key(MASTER, "doc-delivery"), token, pk)
    doc, plaintext = open_delivered(data, sk)
    assert (doc, plaintext) == ("doc-delivery", crypto.decrypt_token(crypto.log_key(MASTER, "doc-delivery"), token))
    with pytest.raises(cose.COSEError):
        open_delivered(data, other_sk)
    # the document id is authenticated: rewrite it in the protected header -> the object no longer opens
    obj = cbor2.loads(data)
    prot = cbor2.loads(obj.value[0])
    prot[cose.CWT_CLAIMS] = {cose.SUB: "doc-other"}
    forged = cose.dumps(cbor2.CBORTag(cose.COSE_ENCRYPT0_TAG, [cose.dumps(prot), dict(obj.value[1]), obj.value[2]]))
    with pytest.raises(cose.COSEError):
        open_delivered(forged, sk)
    assert prot[cose.CONTENT_TYPE] == CT_RECORD


def test_cose_key_round_trip():
    sk, pk = hpke.generate_keypair()
    assert cose.x25519_from_cose_key(cose.x25519_cose_key(pk)) == (pk, None)
    assert cose.x25519_from_cose_key(cose.x25519_cose_key(pk, sk)) == (pk, sk)


@pytest.mark.model
def test_verify_with_a_delivered_record():
    pytest.importorskip("torch")
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.seal import SealParams, seal
    from entropy_protocol.verify import MODIFIED, NOT_VERIFIABLE, VERIFIED, verify_delivered
    try:
        b = HFBackend("distilgpt2")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"distilgpt2 not available offline: {exc}")
    s = seal(b, "The history of the printing press shows", MASTER, SealParams(max_new_tokens=120), doc_id="doc-hpke",
             seed=7)
    sk, pk = hpke.generate_keypair()
    other_sk, _ = hpke.generate_keypair()
    data = deliver(crypto.log_key(MASTER, s.doc_id), s.token, pk)
    r = verify_delivered(b, s.text, data, sk)
    assert r.verdict == VERIFIED and r.doc_id == "doc-hpke"
    assert verify_delivered(b, s.text + " Added.", data, sk).verdict == MODIFIED
    assert verify_delivered(b, s.text, data, other_sk).verdict == NOT_VERIFIABLE
