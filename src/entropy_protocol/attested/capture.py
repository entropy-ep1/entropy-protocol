"""Capture raw evidence on an Azure confidential H100 guest, for the appraisal code and its tests (ARCHITECTURE §7d).

`sudo python3 -m entropy_protocol.attested.capture --out DIR [--rounds 3]` writes, per round, a fresh 32-octet
handle, the 64-octet guest input derived from it (SHA-512 of a label and the handle - the shape of a binding value
without an enrollment behind it), the paravisor's report, a quote whose qualifying data is SHA-256 of that input and
every GPU's report over the handle; once: the attestation key, its Microsoft certificate, the THIM certificates, the
TCG event log, the IMA log when the kernel keeps one, and what the platform says about itself (os-release, kernel,
command line, TPM properties, nvidia-smi). MANIFEST.json lists every file with its SHA-256 and every step that
failed, with the error: a capture that stops at the first failure would hide the rest of the platform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import cbor2

from . import azure_cgpu as az

LABEL = b"ep1/capture/v1"


def _sh(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return f"$ {' '.join(cmd)}\nrc={p.returncode}\n{p.stdout}{p.stderr}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="entropy-attested-capture")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    errors: dict[str, str] = {}

    def put(name: str, data: bytes) -> None:
        (out / name).write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()

    def step(name: str, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and the capture goes on
            errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    for r in range(a.rounds):
        handle = os.urandom(az.GPU_NONCE_SIZE)
        user = hashlib.sha512(LABEL + handle).digest()
        put(f"r{r}_handle.bin", handle)
        put(f"r{r}_user_data.bin", user)
        t0 = time.time()
        rep = step(f"r{r}_report", lambda: az.guest_report(user))
        if rep is not None:
            put(f"r{r}_hcl_report.bin", rep)
        q = step(f"r{r}_quote", lambda: az.quote(hashlib.sha256(user).digest()))
        if q is not None:
            put(f"r{r}_quote_msg.bin", q["msg"])
            put(f"r{r}_quote_sig.bin", q["sig"])
            put(f"r{r}_quote_pcrs.bin", q["pcrs"])
        g = step(f"r{r}_gpus", lambda: az.gpus(handle))
        if g is not None:
            put(f"r{r}_gpus.cbor", cbor2.dumps(g, canonical=True))
        files[f"r{r}_seconds"] = f"{time.time() - t0:.3f}"

    ak = step("ak_public", az.ak_public)
    if ak is not None:
        put("ak_public.tss", ak["tss"])
        put("ak_public.der", ak["spki"])
    c = step("ak_cert", az.ak_cert)
    if c:
        put("ak_cert.der", c)
    t = step("thim", az.thim)
    if t is not None:
        put("thim.json", json.dumps(t, indent=1, sort_keys=True).encode())
    ev = step("event_log", az.event_log)
    if ev is not None:
        put("event_log.bin", ev)
    ima = step("ima_log", az.ima_log)
    if ima:
        put("ima_log.bin", ima)
    sd = step("systemd_tpm_log", az.systemd_tpm_log)
    if sd:
        put("systemd_tpm2_measure.log", sd)
    about = "\n".join(_sh(c) for c in (
        ["cat", "/etc/os-release"], ["uname", "-a"], ["cat", "/proc/cmdline"],
        ["tpm2_getcap", "properties-fixed"], ["tpm2_getcap", "pcrs"], ["tpm2_getcap", "algorithms"],
        ["tpm2_getcap", "handles-nv-index"], ["tpm2_getcap", "handles-persistent"],
        ["nvidia-smi", "-q"], ["nvidia-smi", "conf-compute", "-q"],
        ["sh", "-c", "grep -E '^CONFIG_(IMA|INTEGRITY|SECURITY_LOCKDOWN|MODULE_SIG)' /boot/config-$(uname -r)"],
        ["sh", "-c", "dmesg | grep -i -E 'sev|snp|tpm|ima|lockdown|nvidia|cc ' | tail -80"]))
    put("platform.txt", about.encode())
    manifest = {"captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": platform.node(),
                "rounds": a.rounds, "files": files, "errors": errors}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({"files": len(files), "errors": errors}, indent=1))
    return 0 if not errors else 3


if __name__ == "__main__":
    raise SystemExit(main())
