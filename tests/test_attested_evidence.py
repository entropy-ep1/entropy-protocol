"""Attested sealing (ARCHITECTURE §7d): the appraisal code against evidence captured on a live Azure confidential H100
(bench/results/attested/capture-20261027T162455Z: Standard_NCC40ads_H100_v5, West Europe, Ubuntu 24.04 CVM,
kernel 7.0.0-1016-azure-fde, NVIDIA driver 595.91.07). Refusals are tested on the same evidence with one field
changed - real data altered the way an attacker would alter it, never a stand-in for a component."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import cbor2
import pytest

pytest.importorskip("signxml")

from entropy_protocol.attested import amd, hcl, nvidia, ocsp, rim, tpm  # noqa: E402
from entropy_protocol.attested.appraise import _thim_chain  # noqa: E402

CAP = Path(__file__).resolve().parents[1] / "bench/results/attested/capture-20261027T162455Z"
AT = dt.datetime(2026, 10, 27, 16, 35, 47, tzinfo=dt.timezone.utc)      # MANIFEST captured_at
DRIVER_RIM = "NV_GPU_DRIVER_GH100_595.91.07"
VBIOS_RIM = "NV_GPU_VBIOS_1010_0210_886_96009F0004"


def b(name: str) -> bytes:
    return (CAP / name).read_bytes()


@pytest.fixture(scope="module")
def chain():
    return _thim_chain(json.loads(b("thim.json")))


@pytest.fixture(scope="module")
def report():
    return hcl.parse(b("r0_hcl_report.bin"))


def flip(data: bytes, at: int) -> bytes:
    return data[:at] + bytes([data[at] ^ 1]) + data[at + 1:]


def test_capture_matches_its_manifest():
    m = json.loads(b("MANIFEST.json"))
    assert not m["errors"]
    for name, digest in m["files"].items():
        if not name.endswith("_seconds"):
            assert hashlib.sha256(b(name)).hexdigest() == digest, name


def test_amd_chain_to_pinned_ark_and_report_signature(chain, report):
    vcek, ask, ark = chain
    ext = amd.verify_chain(vcek, ask, ark, "Genoa", AT)
    assert ext["productName"] == "Genoa"
    amd.verify_report(report.snp, vcek)
    assert report.snp.signing_key == "vcek" and not report.snp.debug
    assert all(ext[k] == v for k, v in amd.tcb_parts(report.snp.reported_tcb, "Genoa").items())
    assert ext["hwID"] == report.snp.chip_id                    # Azure reports a real chip id


def test_vcek_serial_number_is_zero(chain):
    # AMD's VCEK has serial number 0, which RFC 5280 forbids; cryptography loads it with a deprecation warning and
    # announces an exception "in a future release". If this test fails, that release has come: AMD reports would no
    # longer verify, and the VCEK needs a reader of its own.
    from cryptography import x509
    from cryptography.utils import CryptographyDeprecationWarning
    with pytest.warns(CryptographyDeprecationWarning):
        assert x509.load_der_x509_certificate(chain[0]).serial_number == 0


def test_amd_refusals(chain, report):
    vcek, ask, ark = chain
    with pytest.raises(amd.SNPError):
        amd.verify_chain(vcek, ask, ark, "Milan", AT)            # the ARK is not the pinned ARK-Milan
    with pytest.raises(amd.SNPError):
        amd.verify_chain(vcek, ask, ark, "Genoa", dt.datetime(2050, 1, 1, tzinfo=dt.timezone.utc))
    with pytest.raises(amd.SNPError):                            # any change to the signed body
        amd.verify_report(amd.parse_report(flip(report.snp.raw, 0x90)), vcek)


def test_runtime_claims_carry_the_guest_input_and_are_bound(report):
    assert report.report_type == hcl.SNP_VM_REPORT and report.hash_type == 1
    assert report.user_data == b("r0_user_data.bin")
    assert hcl.report_data_binds_claims(report)
    forged = report.claims_raw.replace(report.claims["user-data"].encode(), b"00" * 64)
    assert not hcl.report_data_binds_claims(hcl.parse(report.raw.replace(report.claims_raw, forged)))


def test_paravisor_renews_its_report_at_most_every_two_seconds():
    # the three rounds wrote three inputs 0.1-0.3 s apart and read back one report: the attester must wait and check
    assert b("r1_hcl_report.bin") == b("r2_hcl_report.bin") == b("r0_hcl_report.bin")
    assert hcl.parse(b("r1_hcl_report.bin")).user_data != b("r1_user_data.bin")


@pytest.mark.parametrize("r", [0, 1, 2])
def test_quote_by_the_bound_attestation_key(report, r):
    ak = hcl.rsa_from_jwk(report.jwk("HCLAkPub"))
    msg, sig, pcrs = b(f"r{r}_quote_msg.bin"), b(f"r{r}_quote_sig.bin"), b(f"r{r}_quote_pcrs.bin")
    tpm.verify_quote(msg, sig, ak)
    q = tpm.parse_attest(msg)
    assert q.extra == hashlib.sha256(b(f"r{r}_user_data.bin")).digest()
    values = tpm.split_values(pcrs, q)
    assert tpm.pcr_digest(q, values) == q.pcr_digest
    with pytest.raises(tpm.TPMError):
        tpm.verify_quote(flip(msg, 40), sig, ak)
    assert tpm.pcr_digest(q, tpm.split_values(flip(pcrs, 7 * 32), q)) != q.pcr_digest


def test_event_log_replays_to_the_quoted_boot_pcrs():
    q = tpm.parse_attest(b("r0_quote_msg.bin"))
    values = tpm.split_values(b("r0_quote_pcrs.bin"), q)
    replayed = tpm.replay(tpm.parse_event_log(b("event_log.bin")))
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 9, 14):
        assert replayed[i] == values[(tpm.ALG_SHA256, i)], i
    assert replayed[11] != values[(tpm.ALG_SHA256, 11)]          # extended again after boot (systemd), outside this log


@pytest.mark.parametrize("r", [0, 1, 2])
def test_gpu_report_over_the_handle(r):
    g = cbor2.loads(b(f"r{r}_gpus.cbor"))[0]
    chain = nvidia.load_chain(g["chain"])
    nvidia.verify_chain(chain, nvidia.DEVICE_ROOT_SHA256, AT)
    rep = nvidia.parse(g["report"])
    nvidia.verify_report(rep, chain[0])
    assert rep.nonce == b(f"r{r}_handle.bin")
    assert rep.opaque["fwid"] == nvidia.certificate_fwid(chain[0])
    assert nvidia.text(rep.opaque["driver_version"]) == g["driver"] == "595.91.07"
    assert nvidia.vbios_version(rep.opaque["vbios_version"]) == g["vbios"].lower()
    assert g["cc"]["feature"] == 1 and g["cc"]["devtools"] == 0
    with pytest.raises(nvidia.GPUError):
        nvidia.verify_report(nvidia.parse(flip(g["report"], 10)), chain[0])     # a different nonce in the request
    with pytest.raises(nvidia.GPUError):
        nvidia.verify_chain(chain[:-1], nvidia.DEVICE_ROOT_SHA256, AT)          # not ending at NVIDIA's root


@pytest.fixture(scope="module")
def golden():
    return {k: rim.golden(rim.verify(b(f"collateral/{r}.swidtag"), AT)) for k, r in
            (("driver", DRIVER_RIM), ("vbios", VBIOS_RIM))}


def test_measurements_match_nvidias_signed_manifests(golden):
    for r in range(3):
        rep = nvidia.parse(cbor2.loads(b(f"r{r}_gpus.cbor"))[0]["report"])
        assert rim.compare(rep.measurements, golden["driver"], golden["vbios"], rep.opaque["nvdec0_status"][0]) == []
    rep = nvidia.parse(cbor2.loads(b("r0_gpus.cbor"))[0]["report"])
    altered = dict(rep.measurements)
    altered[8] = bytes(48)                                        # golden index 7 (driver), block 8
    assert rim.compare(altered, golden["driver"], golden["vbios"], 0x55) == [(7, "driver")]


def test_rim_signature_refusals():
    xml = b(f"collateral/{DRIVER_RIM}.swidtag")
    i = xml.index(b'index="7"')
    with pytest.raises(rim.RIMError):
        rim.verify(xml[:i] + b'index="8"' + xml[i + 9:], AT)   # a signed manifest changed by one attribute


def test_kept_ocsp_answers_verify_offline():
    kept = cbor2.loads(b("collateral/ocsp.cbor"))
    g = cbor2.loads(b("r0_gpus.cbor"))[0]
    chain = nvidia.load_chain(g["chain"])
    gpu = [k for k in kept if k["chain"] == "gpu0"]
    assert len(gpu) == 3
    for k, cert, issuer in zip(gpu, chain[1:-1], chain[2:]):
        at = dt.datetime.fromisoformat(k["asked_at"])
        assert ocsp.check(k["response"], cert, issuer, k["nonce"], nvidia.DEVICE_ROOT_SHA256, at, chain) == "good"
        with pytest.raises(nvidia.GPUError):
            ocsp.check(k["response"], cert, issuer, bytes(32), nvidia.DEVICE_ROOT_SHA256, at, chain)
    with pytest.raises(nvidia.GPUError):                                      # an answer about another certificate
        ocsp.check(gpu[0]["response"], chain[2], chain[3], gpu[0]["nonce"], nvidia.DEVICE_ROOT_SHA256,
                   dt.datetime.fromisoformat(gpu[0]["asked_at"]), chain)


REFUSED = CAP.parent / "run-20261027T170502Z-refused-launch-measurement"


def test_another_paravisor_build_is_endorsed_by_the_same_id_key():
    """Two hosts an hour apart ran two builds of Azure's paravisor. A policy pinning the first build's measurement
    refused the second (the authority did, live); pinning the ID key that signed both builds' ID blocks affirms it,
    because the firmware launches only a measurement that ID key signed (AMD 56860 rev. 1.59 §8.19.2)."""
    from entropy_protocol import cose
    from entropy_protocol.attested import appraise, enroll
    d = REFUSED / "authority/issued/0001"
    policy = json.loads((REFUSED / "policy.json").read_text())
    at = dt.datetime.fromisoformat(json.loads((d / "appraisal.json").read_text())["time"])
    req = enroll.read_request((d / "request.cbor").read_bytes())
    collateral = cose._thaw(cbor2.loads((d / "collateral.cbor").read_bytes()))
    rrp_id = json.loads((REFUSED / "authority/authority.json").read_text())["rrp_id"]
    a = appraise.appraise(req, rrp_id=rrp_id, policy=policy, at=at, collateral=collateral)
    assert [c.name for c in a.failed()] == ["cpu.report.launch_endorsed"]
    first, second = hcl.parse(b("r0_hcl_report.bin")).snp, hcl.parse(cose._thaw(cbor2.loads(req["evidence"]))["hcl"]).snp
    assert first.measurement != second.measurement and first.id_key_digest == second.id_key_digest and first.id_block
    pinned = {**policy, "amd": {**policy["amd"], "id_key_digests": [first.id_key_digest.hex()]}}
    a = appraise.appraise(req, rrp_id=rrp_id, policy=pinned, at=at, collateral=collateral)
    assert a.ok and not a.failed()
