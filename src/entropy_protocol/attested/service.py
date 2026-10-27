"""The sealing service inside the confidential VM (ARCHITECTURE §7d): enroll, then seal and detect; and the
verifier's pass, run as a separate process.

`seal` - one process holds every secret in memory and writes none of them:
  1. load the model; generate the master key K (and so K_wm) here;
  2. for the statement key, then the detection key: wait for the authority's initiation in the conduit directory,
     enroll (attester.prepare: a fresh Ed25519 key, its CSR, the binding into the paravisor's report, the quote and
     every GPU's report), write the request, wait for the certificate and the attestation result, and check both
     against the key before using it;
  3. seal every prompt; deliver each record by HPKE to the verifier's public key; sign the generation statement under
     the certified key, with the certificate (x5chain, label 33) and SHA-256 of the result ("ep1-ear") protected;
  4. detect every sealed text and every control text with K_wm; sign each detection statement the same way.
`verify` - the verifier, with its own keys (an X25519 key the records were delivered to, an Ed25519 key it signs
under): every sealed text as delivered, with one word changed, and after ordinary transport.
The conduit directory is an untrusted transport; the authority runs elsewhere (authority.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from .. import ENGINE, crypto, statements
from . import attester, ear, enroll, pki

ROLES = {"seal": enroll.CRED_STATEMENT, "detect": enroll.CRED_DETECTION}
EAR_REF = "ep1-ear"
X5CHAIN = 33


def wait_for(path: Path, timeout: float, poll: float = 1.0) -> bytes:
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and path.stat().st_size:
            time.sleep(0.2)                      # a copy that is still being written
            return path.read_bytes()
        time.sleep(poll)
    raise TimeoutError(f"{path.name} did not arrive within {timeout:.0f} s")


def params_map(params) -> dict:
    """The sealing parameters as canonical-CBOR-safe values: no floating point (micro-units), as on every ep1 wire."""
    out = {}
    for k, v in sorted(vars(params).items()):
        if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
            out[k] = v
        elif isinstance(v, float):
            out[k + "_u"] = round(v * 1_000_000)
        elif isinstance(v, dict):
            out[k] = {kk: round(vv * 1_000_000) for kk, vv in sorted(v.items())}
        else:
            raise TypeError(f"parameter {k} of type {type(v).__name__}")
    return out


def sealing_params(cal_path: str | None, max_new_tokens: int, profile: str, weights: str):
    from ..profiles import seal_overrides
    from ..seal import SealParams, table_covers
    prof = seal_overrides(profile)
    if not cal_path:
        return SealParams(max_new_tokens=max_new_tokens, **prof)
    cal = json.loads(Path(cal_path).read_text())
    if cal["weights_digest"] != weights:
        raise SystemExit("calibration file belongs to a different model")
    params = SealParams(max_new_tokens=max_new_tokens, tol_space=cal["space"], cal=cal["table"], **prof)
    problem = table_covers(cal, params)
    if problem:
        raise SystemExit(problem)
    return params


def check_credential(cert_der: bytes, ear_bytes: bytes, root_der: bytes, root_sha256: str, key,
                     credential_type: str) -> dict:
    """The root is the one this sealer is configured with (the conduit could carry another); the certificate is the
    root's, for this key; the result is the root's, affirming, about this key."""
    from cryptography import x509
    import datetime as dt
    if hashlib.sha256(root_der).hexdigest() != root_sha256:
        raise SystemExit("the conduit carries another authority's root")
    root = x509.load_der_x509_certificate(root_der)
    leaf_key = pki.check_leaf(cert_der, root, dt.datetime.now(dt.timezone.utc), pki.STATEMENT_EKU)
    if pki.spki(leaf_key) != pki.spki(key.public_key()):
        raise SystemExit("the certificate is for another key")
    c = ear.read(ear_bytes, root.public_key())
    ext = c.get(ear.EP1, {})
    if c.get(ear.STATUS) != ear.AFFIRMING or ext.get("credential_type") != credential_type or \
            ext.get("key") != hashlib.sha256(pki.spki(key.public_key())).digest():
        raise SystemExit(f"the attestation result does not affirm this key: status {c.get(ear.STATUS)}, "
                         f"failed {ext.get('failed')}")
    return c


def enroll_role(io: Path, role: str, rrp_id: str, root_sha256: str, target: str, backend, sealing_class: str,
                params: dict, timeout: float) -> tuple:
    ini = wait_for(io / f"initiation-{role}.cbor", timeout)
    t0 = time.time()
    e = attester.prepare(initiation=ini, rrp_id=rrp_id, target=target, weights=bytes.fromhex(backend.weights_digest),
                         tokenizer=bytes.fromhex(backend.tokenizer_digest), sealing_class=sealing_class,
                         params=params, credential_type=ROLES[role])
    (io / f"request-{role}.cbor").write_bytes(e.request)
    evidence_s = time.time() - t0
    cert = wait_for(io / f"cert-{role}.der", timeout)
    result = wait_for(io / f"ear-{role}.cose", timeout)
    check_credential(cert, result, (io / "root.der").read_bytes(), root_sha256, e.key, ROLES[role])
    headers = {X5CHAIN: cert, EAR_REF: hashlib.sha256(result).digest()}
    return e.key, headers, {"evidence_seconds": round(evidence_s, 3), "binding": e.binding.value().hex()}


def cmd_seal(a) -> int:
    import torch

    from ..backend_hf import HFBackend
    from ..delivery import deliver
    from ..detect import detect
    from ..profiles import detect_gate_for
    from ..seal import seal
    io, out = Path(a.io), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    backend = HFBackend(a.model, device="cuda", dtype=torch.float16)
    params = sealing_params(a.cal, a.max_new_tokens, a.profile, backend.weights_digest)
    pmap = params_map(params)
    (io / "ready.json").write_text(json.dumps({
        "model": backend.model_ref, "weights": backend.weights_digest, "tokenizer": backend.tokenizer_digest,
        "class": backend.config_class, "params": pmap, "load_seconds": round(time.time() - t_load, 1),
        "env": backend.environment}, indent=1))
    master = crypto.new_master_key()                 # generated here, never written
    k_wm = crypto.wm_key(master)
    keys, headers, info = {}, {}, {}
    for role in ("seal", "detect"):
        keys[role], headers[role], info[role] = enroll_role(io, role, a.rrp_id, a.root_sha256, f"{a.target}/{role}",
                                                            backend, backend.config_class, pmap, a.timeout)
    verifier_pub = bytes.fromhex(Path(a.verifier_pub).read_text().strip())
    prompts = [json.loads(x) for x in Path(a.prompts).read_text().splitlines() if x.strip()][: a.n]
    sealed = []
    for p in prompts:
        t0 = time.time()
        r = seal(backend, p["prompt"], master, params)
        log_key = crypto.log_key(master, r.doc_id)
        gen, _ = statements.generation_statement(log_key, r.token, keys["seal"], f"{a.target}/seal",
                                                 extra=headers["seal"])
        d = out / r.doc_id
        d.mkdir()
        (d / "text.txt").write_text(r.text)
        (d / "record.delivered.cose").write_bytes(deliver(log_key, r.token, verifier_pub))
        (d / "generation.cose").write_bytes(gen)
        sealed.append({"doc": r.doc_id, "prompt_id": p["id"], "tokens": r.tokens, "entries": r.entries,
                       "seconds": round(time.time() - t0, 2)})
    controls = [json.loads(x) for x in Path(a.controls).read_text().splitlines() if x.strip()][: a.n_controls] \
        if a.controls else []
    gate = detect_gate_for(params)
    detections = []
    for item in [{"doc": s["doc"], "text": (out / s["doc"] / "text.txt").read_text(), "kind": "sealed"} for s in sealed] + \
            [{"doc": f"control-{c['id']}", "text": c["text"], "kind": "control"} for c in controls]:
        det = detect(backend, item["text"], k_wm, interval=params.interval, k=params.k, context=params.context,
                     gate=gate, alpha=a.alpha)
        st, body = statements.detection_statement(det, item["text"], backend, keys["detect"], f"{a.target}/detect",
                                                  extra=headers["detect"])
        d = out / ("detections" if item["kind"] == "control" else item["doc"])
        d.mkdir(exist_ok=True)
        name = f"{item['doc']}.detection.cose" if item["kind"] == "control" else "detection.cose"
        (d / name).write_bytes(st)
        if item["kind"] == "control":
            (d / f"{item['doc']}.txt").write_text(item["text"])
        detections.append({"doc": item["doc"], "kind": item["kind"], "detected": det.detected,
                           "matched": det.matched, "checkpoints": det.checkpoints, "p": det.p_value})
    manifest = {"engine": ENGINE, "model": backend.model_ref, "weights": backend.weights_digest,
                "tokenizer": backend.tokenizer_digest, "class": backend.config_class, "profile": a.profile,
                "params": pmap, "enrollment": info, "sealed": sealed, "detections": detections,
                "detect_gate": gate, "alpha": a.alpha}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (io / "done.json").write_text(json.dumps({"sealed": len(sealed), "detections": len(detections)}))
    return 0


def cmd_verify(a) -> int:
    import torch
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from ..backend_hf import HFBackend
    from ..verify import verify_delivered
    out = Path(a.out)
    backend = HFBackend(a.model, device="cuda", dtype=torch.float16)
    priv = bytes.fromhex(Path(a.verifier_key).read_text().strip())
    signer = Ed25519PrivateKey.generate()
    (out / "verifier.pub").write_text(signer.public_key().public_bytes_raw().hex())
    rows = []
    for d in sorted(p for p in out.iterdir() if (p / "generation.cose").exists()):
        text = (d / "text.txt").read_text()
        delivered = (d / "record.delivered.cose").read_bytes()
        words = text.split(" ")
        mid = len(words) // 2
        edited = " ".join(words[:mid] + ["remarkably"] + words[mid + 1:])
        transported = text.replace("\n", "\r\n").replace(" ", " ", 3) + "\n\n"
        for kind, cand in (("as-sealed", text), ("one-word-changed", edited), ("transported", transported)):
            res = verify_delivered(backend, cand, delivered, priv)
            st, _ = statements.verification_statement(res, cand, signer, a.verifier_id,
                                                      generation=(d / "generation.cose").read_bytes())
            (d / f"verification.{kind}.cose").write_bytes(st)
            if kind != "as-sealed":
                (d / f"candidate.{kind}.txt").write_text(cand)
            rows.append({"doc": d.name, "kind": kind, "verdict": res.verdict, "matched": res.matched,
                         "divergent": res.divergent, "indeterminate": res.indeterminate, "entries": res.entries,
                         "changed": res.changed_ranges})
    (out / "verification.json").write_text(json.dumps(rows, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="entropy-attested-service")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seal")
    s.add_argument("--io", required=True, help="conduit directory shared with the authority's side")
    s.add_argument("--out", required=True)
    s.add_argument("--model", required=True)
    s.add_argument("--cal")
    s.add_argument("--profile", default="quality")
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--prompts", required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--controls")
    s.add_argument("--n-controls", type=int, default=10)
    s.add_argument("--alpha", type=float, default=1e-6)
    s.add_argument("--verifier-pub", required=True, help="file with the verifier's X25519 public key (hex)")
    s.add_argument("--rrp-id", required=True, help="the authority this sealer is configured for")
    s.add_argument("--root-sha256", required=True, help="SHA-256 of that authority's root certificate (DER)")
    s.add_argument("--target", required=True)
    s.add_argument("--timeout", type=float, default=1800)
    v = sub.add_parser("verify")
    v.add_argument("--out", required=True)
    v.add_argument("--model", required=True)
    v.add_argument("--verifier-key", required=True, help="file with the verifier's X25519 private key (hex)")
    v.add_argument("--verifier-id", required=True)
    a = ap.parse_args(argv)
    return cmd_seal(a) if a.cmd == "seal" else cmd_verify(a)


if __name__ == "__main__":
    sys.exit(main())
