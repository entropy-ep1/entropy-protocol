"""HPKE against RFC 9180 test vectors A.1.1 (AES-128-GCM) and A.2.1 (ChaCha20-Poly1305), base mode."""
import pytest

from entropy_protocol import hpke

H = bytes.fromhex
INFO = H("4f6465206f6e2061204772656369616e2055726e")                     # "Ode on a Grecian Urn"
PT = H("4265617574792069732074727574682c20747275746820626561757479")    # "Beauty is truth, truth beauty"
AAD0 = H("436f756e742d30")                                               # "Count-0"
VECTORS = {
    hpke.AEAD_AES_128_GCM: dict(
        ikmE="7268600d403fce431561aef583ee1613527cff655c1343f29812e66706df3234",
        pkEm="37fda3567bdbd628e88668c3c8d7e97d1d1253b6d4ea6d44c150f741f1bf4431",
        skEm="52c4a758a802cd8b936eceea314432798d5baf2d7e9235dc084ab1b9cfa2f736",
        ikmR="6db9df30aa07dd42ee5e8181afdb977e538f5e1fec8a06223f33f7013e525037",
        pkRm="3948cfe0ad1ddb695d780e59077195da6c56506b027329794ab02bca80815c4d",
        skRm="4612c550263fc8ad58375df3f557aac531d26850903e55a9f23f21d8534e8ac8",
        shared_secret="fe0e18c9f024ce43799ae393c7e8fe8fce9d218875e8227b0187c04e7d2ea1fc",
        key_schedule_context="00725611c9d98c07c03f60095cd32d400d8347d45ed67097bbad50fc56da742d07cb6cffde367bb0565ba2"
                             "8bb02c90744a20f5ef37f30523526106f637abb05449",
        secret="12fff91991e93b48de37e7daddb52981084bd8aa64289c3788471d9a9712f397",
        key="4531685d41d65f03dc48f6b8302c05b0", base_nonce="56d890e5accaaf011cff4b7d",
        exporter_secret="45ff1c2e220db587171952c0592d5f5ebe103f1561a2614e38f2ffd47e99e3f8",
        ct0="f938558b5d72f1a23810b4be2ab4f84331acc02fc97babc53a52ae8218a355a96d8770ac83d07bea87e13c512a"),
    hpke.AEAD_CHACHA20_POLY1305: dict(
        ikmE="909a9b35d3dc4713a5e72a4da274b55d3d3821a37e5d099e74a647db583a904b",
        pkEm="1afa08d3dec047a643885163f1180476fa7ddb54c6a8029ea33f95796bf2ac4a",
        skEm="f4ec9b33b792c372c1d2c2063507b684ef925b8c75a42dbcbf57d63ccd381600",
        ikmR="1ac01f181fdf9f352797655161c58b75c656a6cc2716dcb66372da835542e1df",
        pkRm="4310ee97d88cc1f088a5576c77ab0cf5c3ac797f3d95139c6c84b5429c59662a",
        skRm="8057991eef8f1f1af18f4a9491d16a1ce333f695d4db8e38da75975c4478e0fb",
        shared_secret="0bbe78490412b4bbea4812666f7916932b828bba79942424abb65244930d69a7",
        key_schedule_context="00431df6cd95e11ff49d7013563baf7f11588c75a6611ee2a4404a49306ae4cfc5b69c5718a60cc5876c35"
                             "8d3f7fc31ddb598503f67be58ea1e798c0bb19eb9796",
        secret="5b9cd775e64b437a2335cf499361b2e0d5e444d5cb41a8a53336d8fe402282c6",
        key="ad2744de8e17f4ebba575b3f5f5a8fa1f69c2a07f6e7500bc60ca6e3e3ec1c91", base_nonce="5c4d98150661b848853b547f",
        exporter_secret="a3b010d4994890e2c6968a36f64470d3c824c8f5029942feb11e7a74b2921922",
        ct0="1c5250d8034ec2b784ba2cfd69dbdb8af406cfe3ff938e131f0def8c8b60b4db21993c62ce81883d2dd1b51a28"),
}


@pytest.mark.parametrize("aead_id", sorted(VECTORS))
def test_rfc9180_base_vectors(aead_id):
    v = {k: H(x) for k, x in VECTORS[aead_id].items()}
    assert hpke.derive_keypair(v["ikmE"]) == (v["skEm"], v["pkEm"])
    assert hpke.derive_keypair(v["ikmR"]) == (v["skRm"], v["pkRm"])
    shared, enc = hpke.encap(v["pkRm"], sk_e=v["skEm"])
    assert (shared, enc) == (v["shared_secret"], v["pkEm"])
    assert hpke.decap(enc, v["skRm"]) == v["shared_secret"]
    ks = hpke.key_schedule(shared, INFO, aead_id)
    for name in ("key_schedule_context", "secret", "key", "base_nonce", "exporter_secret"):
        assert ks[name] == v[name], name
    enc2, ct = hpke.seal(v["pkRm"], INFO, AAD0, PT, aead_id, sk_e=v["skEm"])
    assert (enc2, ct) == (v["pkEm"], v["ct0"])
    assert hpke.open_(enc2, v["skRm"], INFO, AAD0, ct, aead_id) == PT


def test_wrong_recipient_or_aad_fails():
    from cryptography.exceptions import InvalidTag
    sk, pk = hpke.generate_keypair()
    other_sk, _ = hpke.generate_keypair()
    enc, ct = hpke.seal(pk, b"", b"aad", b"record")
    assert hpke.open_(enc, sk, b"", b"aad", ct) == b"record"
    for args in ((enc, other_sk, b"", b"aad", ct), (enc, sk, b"", b"other aad", ct), (enc, sk, b"x", b"aad", ct)):
        with pytest.raises(InvalidTag):
            hpke.open_(*args)
