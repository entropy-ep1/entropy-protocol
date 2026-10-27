"""Evidence from inside an Azure confidential VM with confidential H100s (Standard_NCC*ads_H100_v5): attester side.

What the guest can obtain, and how (ARCHITECTURE §7d; sources: microsoft/openvmm vTPM and IGVM attestation code,
NVIDIA/nvtrust local GPU verifier):
- the paravisor's report: 64 octets written to vTPM NV 0x01400002 (guest attestation input) come back in the runtime
  claims (`user-data`, hex) of a fresh report at NV 0x01400001 - HCL header, SNP report, runtime claims JSON;
- the vTPM attestation key (persistent handle 0x81000003; its public part is also `HCLAkPub` in the runtime claims),
  Microsoft's certificate for it (NV 0x01C101D0), a TPM2_Quote over PCRs with chosen qualifying data, the TCG event
  log and, when the kernel keeps one, the IMA log;
- the VCEK and its chain from the instance metadata service (THIM);
- each GPU's SPDM measurement report over a 32-octet nonce, and its certificate chain, through NVML.
Linux only. The vTPM steps need root and tpm2-tools; the GPU steps need NVIDIA's driver in confidential-computing
mode and nvidia-ml-py. Nothing here judges the evidence: appraisal is the authority's (`appraise`).
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

NV_GUEST_INPUT = 0x01400002
NV_REPORT = 0x01400001
NV_AK_CERT = 0x01C101D0
AK_HANDLE = 0x81000003
GUEST_INPUT_SIZE = 64
THIM_URL = "http://169.254.169.254/metadata/THIM/amd/certification"
EVENT_LOG = Path("/sys/kernel/security/tpm0/binary_bios_measurements")
IMA_LOG = Path("/sys/kernel/security/ima/binary_runtime_measurements")
QUOTE_PCRS = tuple(range(16))
GPU_NONCE_SIZE = 32


class EvidenceError(RuntimeError):
    """The platform did not produce a piece of evidence."""


def _run(args: list[str], stdin: bytes | None = None) -> bytes:
    p = subprocess.run(args, input=stdin, capture_output=True, check=False)
    if p.returncode != 0:
        raise EvidenceError(f"{' '.join(args)}: rc {p.returncode}: {p.stderr.decode(errors='replace').strip()}")
    return p.stdout


def nv_size(index: int) -> int | None:
    """Data size of an NV index, or None when it is not defined."""
    try:
        out = _run(["tpm2_nvreadpublic", f"0x{index:08x}"]).decode()
    except EvidenceError:
        return None
    m = re.search(r"^\s*size:\s*(\d+)", out, re.M)
    return int(m.group(1)) if m else None


def nv_read(index: int) -> bytes:
    size = nv_size(index)
    if size is None:
        raise EvidenceError(f"NV index 0x{index:08x} is not defined")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "nv.bin"
        _run(["tpm2_nvread", "-C", "o", "-s", str(size), "-o", str(out), f"0x{index:08x}"])
        return out.read_bytes()


def guest_report(user_data: bytes) -> bytes:
    """The paravisor's report with ``user_data`` (64 octets) in its runtime claims; defines the input index (owner
    read/write, empty authorization, as the paravisor's own tests do) when the guest has not defined it yet."""
    if len(user_data) != GUEST_INPUT_SIZE:
        raise ValueError(f"guest input is {GUEST_INPUT_SIZE} octets, got {len(user_data)}")
    if nv_size(NV_GUEST_INPUT) is None:
        _run(["tpm2_nvdefine", "-C", "o", "-s", str(GUEST_INPUT_SIZE), "-a",
              "ownerread|ownerwrite|authread|authwrite", f"0x{NV_GUEST_INPUT:08x}"])
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "in.bin"
        src.write_bytes(user_data)
        _run(["tpm2_nvwrite", "-C", "o", "-i", str(src), f"0x{NV_GUEST_INPUT:08x}"])
    return nv_read(NV_REPORT)


def ak_public() -> dict:
    """The attestation key's public area (TPM2B_PUBLIC as tpm2-tools writes it) and its SubjectPublicKeyInfo (DER)."""
    with tempfile.TemporaryDirectory() as d:
        tss, der = Path(d) / "ak.tss", Path(d) / "ak.der"
        _run(["tpm2_readpublic", "-c", f"0x{AK_HANDLE:08x}", "-o", str(tss)])
        _run(["tpm2_readpublic", "-c", f"0x{AK_HANDLE:08x}", "-f", "der", "-o", str(der)])
        return {"tss": tss.read_bytes(), "spki": der.read_bytes()}


def ak_cert() -> bytes | None:
    """Microsoft's certificate for the attestation key, when the platform provisioned one."""
    try:
        return nv_read(NV_AK_CERT)
    except EvidenceError:
        return None


def quote(qualifying: bytes, pcrs: tuple[int, ...] = QUOTE_PCRS) -> dict:
    """TPM2_Quote by the attestation key over the SHA-256 bank: TPMS_ATTEST as signed, TPMT_SIGNATURE, and the PCR
    values in selection order."""
    sel = "sha256:" + ",".join(str(i) for i in pcrs)
    with tempfile.TemporaryDirectory() as d:
        msg, sig, vals = Path(d) / "msg", Path(d) / "sig", Path(d) / "pcrs"
        _run(["tpm2_quote", "-c", f"0x{AK_HANDLE:08x}", "-l", sel, "-q", qualifying.hex(), "-g", "sha256",
              "-m", str(msg), "-s", str(sig), "-o", str(vals), "-F", "values"])
        return {"msg": msg.read_bytes(), "sig": sig.read_bytes(), "pcrs": vals.read_bytes(), "sel": list(pcrs)}


def event_log() -> bytes:
    return EVENT_LOG.read_bytes()


def ima_log() -> bytes | None:
    return IMA_LOG.read_bytes() if IMA_LOG.exists() else None


def thim() -> dict:
    """VCEK, its chain and TCB from the instance metadata service: {"vcekCert", "certificateChain", "tcbm"} (PEM)."""
    req = urllib.request.Request(THIM_URL, headers={"Metadata": "true"})
    with urllib.request.urlopen(req, timeout=10) as r:  # noqa: S310 - link-local metadata endpoint
        return json.loads(r.read())


def gpus(nonce: bytes) -> list[dict]:
    """Every GPU's SPDM measurement report over ``nonce`` (32 octets) and its certificate chain, with the identifiers
    NVML gives beside them (UUID, VBIOS and driver versions, confidential-computing state)."""
    if len(nonce) != GPU_NONCE_SIZE:
        raise ValueError(f"GPU nonce is {GPU_NONCE_SIZE} octets, got {len(nonce)}")
    import pynvml as nv

    nv.nvmlInit()
    try:
        state = nv.nvmlSystemGetConfComputeState()
        common = {"driver": _text(nv.nvmlSystemGetDriverVersion()),
                  "cc": {"environment": int(state.environment), "feature": int(state.ccFeature),
                         "devtools": int(state.devToolsMode)},
                  "ready": int(nv.nvmlSystemGetConfComputeGpusReadyState())}
        out = []
        for i in range(nv.nvmlDeviceGetCount()):
            h = nv.nvmlDeviceGetHandleByIndex(i)
            rep = nv.nvmlDeviceGetConfComputeGpuAttestationReport(h, nonce)
            cert = nv.nvmlDeviceGetConfComputeGpuCertificate(h)
            out.append({"index": i, "uuid": _text(nv.nvmlDeviceGetUUID(h)),
                        "vbios": _text(nv.nvmlDeviceGetVbiosVersion(h)),
                        "report": bytes(rep.attestationReport[:rep.attestationReportSize]),
                        "cec": bytes(rep.cecAttestationReport[:rep.cecAttestationReportSize])
                        if getattr(rep, "isCecAttestationReportPresent", 0) else b"",
                        "chain": bytes(cert.attestationCertChain[:cert.attestationCertChainSize]),
                        "cert_chain": bytes(cert.certChain[:cert.certChainSize]), **common})
        return out
    finally:
        nv.nvmlShutdown()


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)
