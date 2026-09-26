"""The published test vectors reproduce byte for byte, and they verify with the public algorithms."""
import json
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spec"))

from entropy_protocol import cose, crypto, statements, tlog  # noqa: E402

VECTORS = ROOT / "spec" / "test-vectors" / "ep1-vectors.json"


@pytest.fixture(scope="module")
def published():
    if not VECTORS.exists():
        pytest.skip("run spec/make_vectors.py first")
    return json.loads(VECTORS.read_text())


def test_vectors_reproduce_exactly(published):
    import make_vectors
    assert make_vectors.build() == published


def test_vectors_verify_with_public_algorithms(published):
    g = published["generation_statement"]
    signer = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(g["signer_seed"]))
    k_ver = bytes.fromhex(published["keys"]["k_ver"])
    info = statements.check_generation_statement(bytes.fromhex(g["cose_sign1"]), signer.public_key(), k_ver,
                                                 published["record"]["token_verifier"])
    assert info["iat"] == g["iat"]
    m = published["merkle"]
    log_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(m["log_seed"]))
    tlog.verify_receipt(bytes.fromhex(m["receipt_index6"]), bytes.fromhex(g["cose_sign1"]), log_key.public_key())
    vs = published["verification_statement"]
    ver = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(vs["signer_seed"]))
    got = statements.read_statement(bytes.fromhex(vs["cose_sign1"]), ver.public_key())
    assert got["payload"]["verdict"] == "MODIFIED"
    k_wm = bytes.fromhex(published["keys"]["k_wm"])
    cs = published["content_selection"]
    assert [crypto.wm_decide(k_wm, c, cs["interval"]) for c in cs["contexts"]] == cs["decisions"]
    assert cose.decode1(bytes.fromhex(g["cose_sign1"]))[0][cose.PAYLOAD_HASH_ALG] == cose.SHA256
