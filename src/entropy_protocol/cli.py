"""Command line: entropy keygen | signkey | wmkey | hpkekey | calibrate | seal | serve | commit | register |
serve-log | submit | issue | deliver | verify | detect | mcp | timestamp | evidence."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

CALIBRATION_PROMPTS = (
    "The city council met on Tuesday to discuss", "Here is a short explanation of how vaccines work:",
    "The quarterly report shows that", "Dear customer, thank you for", "In the early morning the harbor",
    "The committee reviewed the evidence and", "A simple recipe for bread starts with", "The history of the bridge",
)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _signer(path: str):
    from cryptography.hazmat.primitives import serialization
    return serialization.load_pem_private_key(Path(path).read_bytes(), password=None)


def _pub_raw(path: str) -> bytes:
    """An Ed25519 public key as 32 raw bytes, from a PEM private key, a PEM public key, raw bytes or hex."""
    from cryptography.hazmat.primitives import serialization
    data = Path(path).read_bytes()
    if b"PRIVATE KEY" in data:
        return serialization.load_pem_private_key(data, password=None).public_key().public_bytes_raw()
    if b"PUBLIC KEY" in data:
        return serialization.load_pem_public_key(data).public_bytes_raw()
    if len(data) == 32:
        return data
    raw = bytes.fromhex(data.decode().strip())
    if len(raw) != 32:
        raise ValueError(f"{path}: not an Ed25519 key (PEM, 32 raw bytes or hex)")
    return raw


def _backend(model: str, device: str = "cpu", dtype: str = "float32"):
    try:
        import torch

        from .backend_hf import HFBackend
    except ImportError as exc:           # the model stack is an extra; checking packages and statements needs none of it
        raise SystemExit(f"this command reads a language model ({exc.name} is missing): "
                         "pip install 'entropy-protocol[model]'") from exc
    return HFBackend(model, device=device, dtype=getattr(torch, dtype))


def build_parser() -> argparse.ArgumentParser:
    from .profiles import DEFAULT, NAMES
    point = (f"operating point (default {DEFAULT}): quality = guard 0.9 at sealing, detector gate 0.9; strong = the "
             f"protocol's reference point (profiles.py)")
    ap = argparse.ArgumentParser(prog="entropy", description="Seal AI-generated text; verify it anywhere.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen", help="create a master key file")
    k.add_argument("--out", required=True)
    c = sub.add_parser("calibrate", help="measure the verification tolerance per configuration class")
    c.add_argument("--model", required=True)
    c.add_argument("--configs", required=True, help="device:dtype list to measure, e.g. mps:float32,mps:float16")
    c.add_argument("--corpus", nargs="*", default=[], help="text files (default: generate with the model)")
    c.add_argument("--positions", type=int, default=4000, help="token positions to generate when no corpus is given")
    c.add_argument("--certify", default="", help="classes to list in the table, e.g. fp32,fp16 (default: all)")
    c.add_argument("--space", default="logit", choices=["logit", "prob"])
    c.add_argument("--c", type=float, default=3.0, help="safety factor")
    c.add_argument("--k", type=int, default=None, help="HIGH rank threshold of the records (default: reference)")
    c.add_argument("--out", required=True)
    s = sub.add_parser("seal", help="generate sealed text")
    s.add_argument("--model", required=True)
    s.add_argument("--key", required=True, help="master key file")
    s.add_argument("--prompt", required=True)
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--cal", help="calibration file from `entropy calibrate` (default: exact process, tol = 0)")
    s.add_argument("--device", default="cpu")
    s.add_argument("--dtype", default="float32")
    s.add_argument("--out", help="write {text, token, doc_id} JSON here")
    s.add_argument("--profile", choices=NAMES, default=DEFAULT, help=point)
    gw = sub.add_parser("serve", help="sealing gateway with an OpenAI-compatible API (internal tool)")
    gw.add_argument("--model", required=True)
    gw.add_argument("--key", required=True, help="master key file")
    gw.add_argument("--cal", help="calibration file from `entropy calibrate` (default: exact process, tol = 0)")
    gw.add_argument("--device", default="cpu")
    gw.add_argument("--dtype", default="float32")
    gw.add_argument("--host", default="127.0.0.1")
    gw.add_argument("--port", type=int, default=8080)
    gw.add_argument("--api-key", help="require 'Authorization: Bearer <key>' on every /v1 request")
    gw.add_argument("--gate", type=float, default=None, help="confidence gate for checkpoints (overrides the profile's)")
    gw.add_argument("--guard", type=float, default=None, help="generation guard (overrides the profile's)")
    gw.add_argument("--max-tokens", type=int, default=1024, help="largest max_tokens a request may ask for")
    gw.add_argument("--profile", choices=NAMES, default=DEFAULT, help=point)
    gw.add_argument("--max-footprint-gb", type=float, default=None,
                    help="memory budget (macOS): over it after an answer, the gateway restarts itself to give the "
                         "memory back (default: 80%% of the machine's memory; 0 = off)")
    i = sub.add_parser("issue", help="re-encrypt a record for one verifier")
    i.add_argument("--key", required=True, help="master key file")
    i.add_argument("--token", required=True)
    i.add_argument("--verifier", required=True)
    hk = sub.add_parser("hpkekey", help="create a verifier's X25519 key pair (records delivered by public key)")
    hk.add_argument("--out", required=True, help="prefix: writes <out>.pub.cose and <out>.priv.cose")
    dl = sub.add_parser("deliver", help="encrypt a record to one verifier's public key (COSE-HPKE)")
    dl.add_argument("--key", required=True, help="master key file")
    dl.add_argument("--token", required=True)
    dl.add_argument("--to", required=True, help="verifier's public COSE_Key file (from `entropy hpkekey`)")
    dl.add_argument("--out", required=True)
    v = sub.add_parser("verify", help="verify a text against its record")
    v.add_argument("--model", required=True)
    v.add_argument("--text-file", required=True)
    v.add_argument("--token", help="custody or verifier token (with --key / --verifier-key)")
    v.add_argument("--device", default="cpu")
    v.add_argument("--dtype", default="float32")
    g = v.add_mutually_exclusive_group(required=True)
    g.add_argument("--key", help="master key file (custody verification)")
    g.add_argument("--verifier-key", help="verifier-scoped key (base64url)")
    g.add_argument("--delivered", help="record delivered to this verifier (COSE-HPKE file, from `entropy deliver`)")
    v.add_argument("--hpke-key", help="this verifier's private COSE_Key file (with --delivered)")
    v.add_argument("--sign", help="verifier's Ed25519 private key (PEM): also emit a signed verification statement")
    v.add_argument("--verifier-id", default="verifier")
    v.add_argument("--generation", help="generation statement file, referenced from the verification statement")
    v.add_argument("--statement-out", help="write the verification statement (COSE) here")
    sk = sub.add_parser("signkey", help="create an Ed25519 signing key (PEM) for statements or a log")
    sk.add_argument("--out", required=True)
    wk = sub.add_parser("wmkey", help="export the detection key of the record-free mode")
    wk.add_argument("--key", required=True, help="master key file")
    wk.add_argument("--out", required=True)
    cm = sub.add_parser("commit", help="generation statement (COSE Hash Envelope) for a sealed text")
    cm.add_argument("--key", required=True, help="master key file")
    cm.add_argument("--token", required=True)
    cm.add_argument("--sign", required=True, help="issuer's Ed25519 private key (PEM)")
    cm.add_argument("--issuer", required=True)
    cm.add_argument("--out", required=True)
    rg = sub.add_parser("register", help="append a statement to a local transparency log; write its receipt")
    rg.add_argument("--log", required=True, help="log file (JSON lines)")
    rg.add_argument("--log-key", required=True, help="log's Ed25519 private key (PEM)")
    rg.add_argument("--statement", required=True)
    rg.add_argument("--receipt-out", required=True)
    sv = sub.add_parser("serve-log", help="run a SCRAPI Transparency Service over a local log")
    sv.add_argument("--log", required=True)
    sv.add_argument("--log-key", required=True, help="log's Ed25519 private key (PEM)")
    sv.add_argument("--issuer-key", action="append", required=True, help="accepted issuer key (PEM); repeatable")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8090)
    sm = sub.add_parser("submit", help="register a statement with a SCRAPI Transparency Service")
    sm.add_argument("--url", required=True)
    sm.add_argument("--statement", required=True)
    sm.add_argument("--receipt-out", required=True)
    tsp = sub.add_parser("timestamp", help="RFC 3161 time stamp over a statement (ask an authority, or check one)")
    tsp.add_argument("--statement", required=True, help="the statement stamped (its logged form is what is stamped)")
    tsp.add_argument("--tsa", help="authority URL (POST application/timestamp-query); without it only the request "
                                   "is written, to send by hand")
    tsp.add_argument("--request-out", help="write the DER TimeStampReq here")
    tsp.add_argument("--out", help="write the token here (needs --tsa)")
    tsp.add_argument("--check", help="check this token against the statement instead of asking for one")
    tsp.add_argument("--timeout", type=float, default=20.0)
    ev = sub.add_parser("evidence", help="evidence package: build one, or check one offline")
    ev.add_argument("--check", help="package to check offline instead of building one")
    ev.add_argument("--out-dir", help="with --check: write the parts and a README.md here")
    ev.add_argument("--out", help="package to write")
    ev.add_argument("--doc", default="", help="document id (default: read from the verification statement)")
    ev.add_argument("--token", default="", help="custody token, so a key holder can re-verify from scratch")
    ev.add_argument("--text-file", help="the text this package is about")
    ev.add_argument("--note", default="", help="one line on how the text was obtained")
    for part, title in (("generation", "issuer"), ("verification", "verifier"), ("detection", "detector")):
        ev.add_argument(f"--{part}", help=f"{part} statement (COSE)")
        ev.add_argument(f"--{part[:3]}-receipt", help=f"transparency-log receipt for the {part} statement")
        ev.add_argument(f"--{part[:3]}-timestamp", help=f"RFC 3161 token over the {part} statement")
        ev.add_argument(f"--{title}-pub", help=f"{title}'s Ed25519 key (PEM, raw or hex) the statement verifies under")
    ev.add_argument("--log-pub", help="transparency log's Ed25519 key the receipts verify under")
    mc = sub.add_parser("mcp", help="MCP server over stdio: verify and detect as tools for an assistant")
    mc.add_argument("--model", required=True)
    mc.add_argument("--key", help="master key file: serves both tools")
    mc.add_argument("--wm-key", help="detection key only (from `entropy wmkey`): serves `detect`")
    mc.add_argument("--device", default="cpu")
    mc.add_argument("--dtype", default="float32")
    mc.add_argument("--profile", choices=NAMES, default=DEFAULT, help="operating point the texts were sealed at")
    mc.add_argument("--gate", type=float, default=None, help="detector's confidence gate (default: the profile's)")
    dt = sub.add_parser("detect", help="record-free detection: is the text sealed under this key?")
    dt.add_argument("--model", required=True)
    dt.add_argument("--text-file", required=True)
    gk = dt.add_mutually_exclusive_group(required=True)
    gk.add_argument("--key", help="master key file")
    gk.add_argument("--wm-key", help="detection key file (from `entropy wmkey`)")
    dt.add_argument("--alpha", type=float, default=1e-6)
    dt.add_argument("--gate", type=float, default=None, help="detector's confidence gate (default: the profile's)")
    dt.add_argument("--profile", choices=NAMES, default=DEFAULT, help="operating point the text was sealed at "
                    f"(default {DEFAULT}): sets the detector's gate unless --gate is given")
    dt.add_argument("--device", default="cpu")
    dt.add_argument("--dtype", default="float32")
    dt.add_argument("--sign", help="detector's Ed25519 private key (PEM): also emit a signed detection statement")
    dt.add_argument("--detector-id", default="detector")
    dt.add_argument("--statement-out")
    return ap


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(raw)

    from . import crypto
    if a.cmd == "keygen":
        Path(a.out).write_bytes(crypto.new_master_key())
        print(f"master key written to {a.out} — keep it secret")
        return 0
    if a.cmd == "signkey":
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        pem = Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                         serialization.NoEncryption())
        Path(a.out).write_bytes(pem)
        print(f"Ed25519 key written to {a.out}")
        return 0
    if a.cmd == "wmkey":
        Path(a.out).write_bytes(crypto.wm_key(Path(a.key).read_bytes()))
        print(f"detection key written to {a.out} — whoever holds it can detect, and imitate, the mark")
        return 0
    if a.cmd == "commit":
        from .statements import generation_statement
        master = Path(a.key).read_bytes()
        stmt, c = generation_statement(crypto.log_key(master, crypto.token_doc_id(a.token)), a.token,
                                       _signer(a.sign), a.issuer)
        Path(a.out).write_bytes(stmt)
        from .cose import dumps as cbor
        from .statements import sha256
        print(json.dumps({"statement": a.out, "bytes": len(stmt), "doc": c["doc"],
                          "commitment_sha256": sha256(cbor(c)).hex()}, indent=1))
        return 0
    if a.cmd == "register":
        from .tlog import TransparencyLog
        log = TransparencyLog(_signer(a.log_key), a.log)
        index, receipt = log.register(Path(a.statement).read_bytes())
        Path(a.receipt_out).write_bytes(receipt)
        print(json.dumps({"leaf_index": index, "tree_size": log.size, "root": log.root().hex(),
                          "receipt": a.receipt_out}, indent=1))
        return 0
    if a.cmd == "serve-log":
        from . import scrapi
        from .tlog import TransparencyLog
        ts = scrapi.TransparencyService(TransparencyLog(_signer(a.log_key), a.log),
                                        [scrapi.load_public(k) for k in a.issuer_key])
        server = scrapi.serve(ts, a.host, a.port)
        print(f"SCRAPI Transparency Service on http://{a.host}:{server.server_address[1]} "
              f"({ts.log.size} entries); Ctrl+C to stop")
        try:
            import threading
            threading.Event().wait()
        except KeyboardInterrupt:
            server.shutdown()
        return 0
    if a.cmd == "submit":
        from . import scrapi
        stmt = Path(a.statement).read_bytes()
        location, receipt = scrapi.register(a.url, stmt)
        Path(a.receipt_out).write_bytes(receipt)
        print(json.dumps({"location": location, **scrapi.verify_receipt_online(a.url, receipt, stmt)}, indent=1))
        return 0
    if a.cmd == "issue":
        token_v, key_v = crypto.reissue_for_verifier(Path(a.key).read_bytes(), a.token, a.verifier)
        print(json.dumps({"token": token_v, "verifier_key": _b64(key_v), "verifier": a.verifier}, indent=1))
        return 0
    if a.cmd == "hpkekey":
        from . import cose, hpke
        priv, pub = hpke.generate_keypair()
        Path(a.out + ".pub.cose").write_bytes(cose.x25519_cose_key(pub))
        Path(a.out + ".priv.cose").write_bytes(cose.x25519_cose_key(pub, priv))
        print(f"verifier key pair: {a.out}.pub.cose (give to issuers), {a.out}.priv.cose (keep secret)")
        return 0
    if a.cmd == "deliver":
        from . import cose
        from .delivery import deliver
        pub, _ = cose.x25519_from_cose_key(Path(a.to).read_bytes())
        master = Path(a.key).read_bytes()
        data = deliver(crypto.log_key(master, crypto.token_doc_id(a.token)), a.token, pub)
        Path(a.out).write_bytes(data)
        print(json.dumps({"delivered": a.out, "bytes": len(data), "doc": crypto.token_doc_id(a.token)}, indent=1))
        return 0

    if a.cmd == "calibrate":
        from .calibrate import calibrate
        from .canon import canonicalize
        from .seal import SealParams, first_checkpoint, seal
        ref = _backend(a.model)
        k = a.k if a.k is not None else SealParams().k
        if a.corpus:
            corpus = [ref.canonical_ids(canonicalize(Path(p).read_text(encoding="utf-8"))) for p in a.corpus]
        else:
            # sealed generations, until the corpus has enough positions: forced tokens put the model in
            # contexts where numerical noise is larger (Qwen2.5 fp16: 0.36 on sealed vs 0.19 on plain text)
            corpus, n, throwaway = [], 0, crypto.new_master_key()
            while sum(map(len, corpus)) < a.positions and n < 50 * len(CALIBRATION_PROMPTS):
                s = seal(ref, CALIBRATION_PROMPTS[n % len(CALIBRATION_PROMPTS)], throwaway,
                         SealParams(k=k, max_new_tokens=200), seed=1_000 + n)
                n += 1
                text = canonicalize(s.text)
                if len(text) >= 40:
                    corpus.append(ref.canonical_ids(text))
        corpus = [ids for ids in corpus if ids]
        configs = {x: (lambda x=x: _backend(a.model, *x.split(":"))) for x in a.configs.split(",") if x}
        cal = calibrate(ref, configs, corpus, k=k, space=a.space, c=a.c,
                        from_position=first_checkpoint(SealParams(k=k)))
        out = {"model": ref.model_ref, "weights_digest": ref.weights_digest, "tokenizer_digest": ref.tokenizer_digest,
               "space": cal.space, "c": cal.c, "k": cal.k, "reference": cal.reference, "texts": len(corpus),
               "from_position": cal.from_position,
               "positions": cal.positions, "delta": cal.delta, "errors": cal.errors,
               "table": cal.table([x for x in a.certify.split(",") if x] or None)}
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return 0

    if a.cmd == "serve":
        from .gateway import Gateway, phys_footprint, serve
        from .seal import SealParams, table_covers
        from .profiles import seal_overrides
        backend = _backend(a.model, a.device, a.dtype)
        kw = seal_overrides(a.profile)
        kw.update({k: v for k, v in (("gate", a.gate), ("guard", a.guard)) if v is not None})
        if a.cal:
            cal = json.loads(Path(a.cal).read_text())
            if cal["weights_digest"] != backend.weights_digest or cal.get("k", SealParams().k) != SealParams().k:
                print("calibration file belongs to a different model or k", file=sys.stderr)
                return 2
            kw.update(tol_space=cal["space"], cal=cal["table"])
        params = SealParams(**kw)
        if a.cal and table_covers(cal, params):
            print(table_covers(cal, params), file=sys.stderr)
            return 2
        budget = a.max_footprint_gb
        if budget is None:                                  # 80% of the machine where the footprint can be read
            budget = 0.8 * os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30 \
                if phys_footprint() is not None else 0.0
        gateway = Gateway(backend, Path(a.key).read_bytes(), params, api_key=a.api_key, max_tokens=a.max_tokens,
                          max_footprint=int(budget * 2 ** 30) if budget > 0 else None)
        server = serve(gateway, a.host, a.port)
        print(f"entropy gateway: http://{a.host}:{server.server_port}/ ({a.model}); /v1/chat/completions, "
              f"/v1/completions, /v1/verify, /v1/detect; operating point {a.profile} (gate {params.gate}, guard "
              f"{params.guard}; detector gate {gateway.detect_gate}); memory budget "
              f"{f'{budget:.1f} GB' if budget > 0 else 'off'}", flush=True)
        fp = phys_footprint()
        if budget > 0 and fp is not None and fp > budget * 2 ** 30:
            print(f"warning: the model alone takes {fp / 2 ** 30:.1f} GB, over the budget: every answer will restart "
                  f"the gateway (raise --max-footprint-gb)", file=sys.stderr, flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        server.server_close()
        if server.restart:
            with gateway.lock:                              # nothing is running on the model any more
                print(f"entropy gateway: restarting ({server.restart})", flush=True)
                os.execv(sys.executable, [sys.executable, "-m", "entropy_protocol.cli", *raw])
        return 0
    if a.cmd == "seal":
        from .seal import SealParams, seal, table_covers
        from .profiles import seal_overrides
        backend = _backend(a.model, a.device, a.dtype)
        prof = seal_overrides(a.profile)
        params = SealParams(max_new_tokens=a.max_new_tokens, **prof)
        if a.cal:
            cal = json.loads(Path(a.cal).read_text())
            if cal["weights_digest"] != backend.weights_digest:
                print("calibration file belongs to a different model", file=sys.stderr)
                return 2
            if cal.get("k", params.k) != params.k:
                print(f"calibration was measured for k = {cal['k']}, records use k = {params.k}", file=sys.stderr)
                return 2
            params = SealParams(max_new_tokens=a.max_new_tokens, tol_space=cal["space"], cal=cal["table"], **prof)
            if table_covers(cal, params):
                print(table_covers(cal, params), file=sys.stderr)
                return 2
        res = seal(backend, a.prompt, Path(a.key).read_bytes(), params)
        out = {"text": res.text, "token": res.token, "doc_id": res.doc_id, "entries": res.entries}
        if a.out:
            Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "verify":
        from .verify import verify, verify_delivered
        if a.delivered and not a.hpke_key or not a.delivered and not a.token:
            print("--delivered needs --hpke-key; --key / --verifier-key need --token", file=sys.stderr)
            return 2
        backend = _backend(a.model, a.device, a.dtype)
        text = Path(a.text_file).read_text(encoding="utf-8")
        if a.delivered:
            from . import cose
            _, priv = cose.x25519_from_cose_key(Path(a.hpke_key).read_bytes())
            result = verify_delivered(backend, text, Path(a.delivered).read_bytes(), priv)
        else:
            if a.key:
                key = crypto.log_key(Path(a.key).read_bytes(), crypto.token_doc_id(a.token))
            else:
                key = _unb64(a.verifier_key)
            result = verify(backend, text, a.token, key)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=1))
        if a.sign:
            from .statements import verification_statement
            gen = Path(a.generation).read_bytes() if a.generation else None
            stmt, _ = verification_statement(result, text, _signer(a.sign), a.verifier_id, generation=gen)
            if a.statement_out:
                Path(a.statement_out).write_bytes(stmt)
        return 0
    if a.cmd == "mcp":
        from .mcp import Verifier, serve
        master = Path(a.key).read_bytes() if a.key else None
        wm = crypto.wm_key(master) if master else (Path(a.wm_key).read_bytes() if a.wm_key else None)
        if master is None and wm is None:
            print("give --key (both tools) or --wm-key (detection only)", file=sys.stderr)
            return 2
        serve(Verifier(a.model, device=a.device, dtype=a.dtype, master=master, wm_key=wm, profile=a.profile,
                       gate=a.gate))
        return 0
    if a.cmd == "timestamp":
        from . import ts3161
        from .pack import stamped_digest
        digest = stamped_digest(Path(a.statement).read_bytes())
        if a.check:
            print(json.dumps(ts3161.check(Path(a.check).read_bytes(), digest), indent=1))
            return 0
        req = ts3161.request(digest)
        if a.request_out:
            Path(a.request_out).write_bytes(req)
        if not a.tsa:
            print(json.dumps({"stamped_digest": digest.hex(), "request": a.request_out, "bytes": len(req),
                              "content_type": ts3161.CONTENT_TYPE}, indent=1))
            return 0
        import urllib.request
        rq = urllib.request.Request(a.tsa, data=req, headers={"Content-Type": ts3161.CONTENT_TYPE,
                                                              "Accept": ts3161.REPLY_TYPE})
        with urllib.request.urlopen(rq, timeout=a.timeout) as resp:     # noqa: S310 - the URL is the operator's
            token = ts3161.token_from_response(resp.read())
        if a.out:
            Path(a.out).write_bytes(token)
        print(json.dumps({"token": a.out, "bytes": len(token), **ts3161.check(token, digest)}, indent=1))
        return 0
    if a.cmd == "evidence":
        from .pack import build, check as check_pack, explode
        if a.check:
            data = Path(a.check).read_bytes()
            report = check_pack(data)
            if a.out_dir:
                report["written"] = explode(data, a.out_dir)
            print(json.dumps(report, ensure_ascii=False, indent=1))
            return 0 if report["ok"] else 1
        if not a.out:
            print("give --out to build a package or --check to read one", file=sys.stderr)
            return 2
        files = (("gen", a.generation, a.gen_receipt, a.gen_timestamp),
                 ("ver", a.verification, a.ver_receipt, a.ver_timestamp),
                 ("det", a.detection, a.det_receipt, a.det_timestamp))
        st = {k: Path(v).read_bytes() for k, v, _, _ in files if v}
        rc = {k: Path(v).read_bytes() for k, _, v, _ in files if v}
        ts = {k: Path(v).read_bytes() for k, _, _, v in files if v}
        keys = {name: _pub_raw(path) for name, path in (("issuer", a.issuer_pub), ("verifier", a.verifier_pub),
                                                        ("detector", a.detector_pub), ("log", a.log_pub)) if path}
        doc = a.doc
        if not doc and "ver" in st:
            from . import cose as _cose
            import cbor2 as _cbor2
            doc = (_cbor2.loads(_cose.decode1(st["ver"])[3]) or {}).get("doc", "")   # (prot, prot_bytes, unprot, payload, sig)
        text = Path(a.text_file).read_text(encoding="utf-8") if a.text_file else None
        pack = build(doc=doc, text=text, token=a.token, statements=st, receipts=rc, timestamps=ts, keys=keys,
                     note=a.note)
        Path(a.out).write_bytes(pack)
        report = check_pack(pack)
        print(json.dumps({"package": a.out, "bytes": len(pack), "ok": report["ok"],
                          "summary": report["summary"]}, ensure_ascii=False, indent=1))
        return 0
    if a.cmd == "detect":
        from .detect import detect
        backend = _backend(a.model, a.device, a.dtype)
        text = Path(a.text_file).read_text(encoding="utf-8")
        k_wm = crypto.wm_key(Path(a.key).read_bytes()) if a.key else Path(a.wm_key).read_bytes()
        from .profiles import detect_gate
        gate = a.gate if a.gate is not None else detect_gate(a.profile)
        d = detect(backend, text, k_wm, alpha=a.alpha, gate=gate)
        print(json.dumps(d.to_dict(), indent=1))
        if a.sign:
            from .statements import detection_statement
            stmt, _ = detection_statement(d, text, backend, _signer(a.sign), a.detector_id)
            if a.statement_out:
                Path(a.statement_out).write_bytes(stmt)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
