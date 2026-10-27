"""Evidence packages (`ep1-pack/v2`) for every text of an attested run, and their offline checks.

    python bench/attested/build_packs.py bench/results/attested/run-<stamp> [--tsa URL]

Per sealed text: the text, the generation statement (signed in the confidential VM under the certified statement
key), the verifier's statement on the text as sealed, the detection statement (signed in the VM under the
certified detection key); RFC 3161 time stamps from an outside authority over each statement's logged form;
receipts of a transparency log (RFC 9162 tree, RFC 9942 receipts) kept for this run; and the attestation section:
the authority's root, the attestation results, the enrollment requests with their evidence, the collateral the
authority fetched and the policy it applied. Every package is then checked offline (`pack.check`) and the reports
written beside it. The log's private key stays with the operator (packs/log.key, ignored by git).
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from entropy_protocol import pack, ts3161
from entropy_protocol.attested import enroll, evidence
from entropy_protocol.tlog import TransparencyLog

TSA = "http://timestamp.digicert.com"


def stamp(statement: bytes, tsa: str) -> bytes:
    req = ts3161.request(pack.stamped_digest(statement))
    rq = urllib.request.Request(tsa, data=req, headers={"Content-Type": ts3161.CONTENT_TYPE,
                                                        "Accept": ts3161.REPLY_TYPE})
    with urllib.request.urlopen(rq, timeout=30) as resp:  # noqa: S310 - the TSA is the operator's choice
        return ts3161.token_from_response(resp.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tsa", default=TSA)
    a = ap.parse_args()
    run = Path(a.run)
    out, auth = run / "guest/out", run / "authority"
    packs = run / "packs"
    packs.mkdir(exist_ok=True)
    roles = {}
    policy = (auth / "policy.json").read_text()
    for d in sorted((auth / "issued").iterdir()):
        if (d / "cert.der").exists():
            req = enroll.read_request((d / "request.cbor").read_bytes())
            roles["gen" if req["credential_type"] == enroll.CRED_STATEMENT else "det"] = {"dir": d, "policy": policy}
    att = evidence.section((auth / "root.der").read_bytes(), json.loads((auth / "authority.json").read_text()), roles)
    key_file = packs / "log.key"
    if key_file.exists():
        log_key = serialization.load_pem_private_key(key_file.read_bytes(), None)
    else:
        log_key = Ed25519PrivateKey.generate()
        key_file.write_bytes(log_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                   serialization.NoEncryption()))
        key_file.chmod(0o600)
    log = TransparencyLog(log_key, packs / "log.json")
    verifier_pub = bytes.fromhex((out / "verifier.pub").read_text().strip())
    summary = []
    for d in sorted(p for p in out.iterdir() if (p / "generation.cose").exists()):
        st = {"gen": (d / "generation.cose").read_bytes(), "ver": (d / "verification.as-sealed.cose").read_bytes(),
              "det": (d / "detection.cose").read_bytes()}
        rc, ts = {}, {}
        for part, s in st.items():
            idx, _ = log.register(s)
            rc[part] = log.receipt(idx)
            ts[part] = stamp(s, a.tsa)
        data = pack.build(doc=d.name, text=(d / "text.txt").read_text(), statements=st, receipts=rc, timestamps=ts,
                          keys={"verifier": verifier_pub,
                                "log": log_key.public_key().public_bytes_raw()},
                          note="sealed in an Azure confidential VM with a confidential H100; see att", att=att)
        (packs / f"{d.name}.cbor").write_bytes(data)
        report = pack.check(data)
        (packs / f"{d.name}.check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
        failed = [c["name"] for c in report["checks"] if c["ok"] is False]
        summary.append({"doc": d.name, "ok": report["ok"], "checks": len(report["checks"]), "failed": failed,
                        "verdict": report["summary"].get("verdict"), "detected": report["summary"].get("detected")})
        print(d.name, "ok" if report["ok"] else f"FAILED {failed}", flush=True)
    (packs / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({"packages": len(summary), "all_ok": all(s["ok"] for s in summary)}, indent=1))
    return 0 if summary and all(s["ok"] for s in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
