"""Seed corpora for the fuzz targets, from real data: the test vectors, the live capture on a confidential H100 and
the attested run (records, tokens, statements, receipts, time stamps, packages, hardware reports, manifests,
enrollment messages, attestation results). A fuzzer that starts from real structures reaches the checks behind
the first length test.

    python fuzz/make_corpus.py            # writes fuzz/corpus/<target>/seed-*
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cbor2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fuzz/corpus"
CAP = ROOT / "bench/results/attested/capture-20261027T162455Z"
RUN = ROOT / "bench/results/attested/run-20261027T172258Z"


def put(target: str, data: bytes) -> None:
    d = OUT / target
    d.mkdir(parents=True, exist_ok=True)
    (d / f"seed-{hashlib.sha256(data).hexdigest()[:16]}").write_bytes(data)


def text_bytes(s: str) -> bytes:
    """Bytes that fuzz/targets.py:text_of turns back into ``s`` (code points written two bytes at a time)."""
    out = bytearray()
    for ch in s:
        cp = ord(ch)
        if cp < 0x4000:
            out += bytes([0xC0 | (cp >> 8), cp & 0xFF])
        elif 0x10000 <= cp < 0x14000:
            cp -= 0x10000
            out += bytes([0xE0 | (cp >> 8), cp & 0xFF])
    return bytes(out)


def main() -> int:
    v = json.loads((ROOT / "spec/test-vectors/ep1-vectors.json").read_text())
    for key in ("record", "record_v4_gated", "record_v4_visible"):
        if key in v:
            put("record", bytes.fromhex(v[key]["cbor"]))
            put("cbor_diff", bytes.fromhex(v[key]["cbor"]))
            put("token", v[key]["token_log"].encode())
    for key, val in v.items():
        if isinstance(val, dict):
            for field in ("cose_sign1", "cose", "statement", "receipt", "envelope"):
                if isinstance(val.get(field), str):
                    put("cose_sign1", bytes.fromhex(val[field]))
                    put("statement", bytes.fromhex(val[field]))
    for c in v.get("canonicalization_v5", {}).get("cases", []) + v.get("canonicalization_v4", {}).get("cases", []):
        put("canon", text_bytes(c["input"]))
        put("canon_diff", text_bytes(c["input"]))
    for c in v.get("visible_text_v2", {}).get("cases", []):
        put("plain", text_bytes(c["input"]))
    # the live capture
    put("hcl", (CAP / "r0_hcl_report.bin").read_bytes())
    put("snp_report", (CAP / "r0_hcl_report.bin").read_bytes()[32:32 + 0x4A0])
    put("tpm_quote", (CAP / "r0_quote_msg.bin").read_bytes())
    put("tpm_signature", (CAP / "r0_quote_sig.bin").read_bytes())
    put("event_log", (CAP / "event_log.bin").read_bytes())
    put("gpu_report", cbor2.loads((CAP / "r0_gpus.cbor").read_bytes())[0]["report"])
    for f in (CAP / "collateral").glob("*.swidtag"):
        put("rim", f.read_bytes())
        put("rim_payload", f.read_bytes())
    # the attested run
    for f in sorted((RUN / "conduit").glob("*.cbor")):
        put("enroll", f.read_bytes())
        req = cbor2.loads(f.read_bytes())
        if isinstance(req, dict) and "evidence" in req:
            put("appraise", req["evidence"])
            ev = cbor2.loads(req["evidence"])
            if ev.get("systemd_tpm_log"):
                put("systemd_log", ev["systemd_tpm_log"])
    for f in sorted((RUN / "conduit").glob("ear-*.cose")):
        put("ear", f.read_bytes())
        put("cose_sign1", f.read_bytes())
    docs = sorted(p for p in (RUN / "guest/out").iterdir() if (p / "generation.cose").exists())[:4]
    for d in docs:
        for name in ("generation.cose", "detection.cose", "verification.as-sealed.cose"):
            put("statement", (d / name).read_bytes())
        put("cose_encrypt0", (d / "record.delivered.cose").read_bytes())
        put("delivered", (d / "record.delivered.cose").read_bytes())
    for f in sorted((RUN / "packs").glob("*.cbor"))[:3]:
        put("pack", f.read_bytes())
        p = cbor2.loads(f.read_bytes())
        for part, rc in p.get("rc", {}).items():
            put("receipt", rc)
        for part, ts in p.get("ts", {}).items():
            put("timestamp", ts)
    print({d.name: len(list(d.iterdir())) for d in sorted(OUT.iterdir())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
