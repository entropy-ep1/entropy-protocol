"""Evidence package (`entropy evidence`): one file a third party can check without us and without any secret.

Inside: the text as disclosed, the signed statements about it (generation by the issuer, verification by the
verifier, detection when there is one), their transparency-log receipts, the public keys those signatures are made
under, and any RFC 3161 time-stamp token. The record never goes in - it is the issuer's secret, and the statements
speak about it only through digests (ARCHITECTURE §7).

`check` re-runs, offline, everything that needs no model: every signature, the digest that ties the disclosed text
to the verdict, the reference that ties the verdict to the generation statement, inclusion in the log, and the
time-stamp imprint. Two things it cannot do and says so: read the text with the model again (that needs the model
and a key - `explode` writes the command), and validate a time-stamping authority's certificate chain.
Trust stays with the reader: the package carries the keys it was signed under, and the reader compares them with
the issuer's published key.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import cbor2
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import cose, ts3161
from .canon import CANON_ID, canonicalize
from .statements import read_statement, sha256, statement_ref
from .tlog import entry_of, verify_receipt

PACK_V = "ep1-pack/v1"
PACK_V2 = "ep1-pack/v2"     # + "att": attestation of the keys that signed (attested/evidence.py, ARCHITECTURE §7d)
PARTS = ("gen", "ver", "det")
_TITLE = {"gen": "generation", "ver": "verification", "det": "detection"}
_KEY_OF = {"gen": "issuer", "ver": "verifier", "det": "detector"}


def build(*, doc: str, text: str | None = None, token: str = "", statements: dict | None = None,
          receipts: dict | None = None, timestamps: dict | None = None, keys: dict | None = None,
          note: str = "", att: dict | None = None) -> bytes:
    """CBOR package. ``statements``/``receipts``/``timestamps`` are keyed "gen" / "ver" / "det"; ``keys`` holds raw
    32-byte Ed25519 public keys under "issuer" / "verifier" / "detector" / "log". With ``att`` (attested/evidence.py)
    the issuer's and the detector's keys come from certificates the attestation establishes."""
    pack = {"v": PACK_V2 if att else PACK_V, "doc": doc, "token": token,
            "st": {k: v for k, v in (statements or {}).items() if v},
            "rc": {k: v for k, v in (receipts or {}).items() if v},
            "ts": {k: v for k, v in (timestamps or {}).items() if v},
            "keys": {k: v for k, v in (keys or {}).items() if v}, "note": note}
    if text is not None:
        pack["text"] = text
    if att:
        pack["att"] = att
    return cose.dumps(pack)


def stamped_digest(statement: bytes) -> bytes:
    """What a time stamp is taken over: the statement's logged form, so a receipt can still be added afterwards."""
    return statement_ref(statement)


def _pub(raw: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(raw)


def check(pack: bytes) -> dict:
    """Every offline check, as a report: {"ok", "checks": [{"name", "ok", "detail"}], "summary"}."""
    p = cbor2.loads(pack)
    checks: list[dict] = []

    def note(name: str, ok: bool | None, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    if p.get("v") not in (PACK_V, PACK_V2):
        return {"ok": False, "checks": [{"name": "version", "ok": False, "detail": str(p.get("v"))}], "summary": {}}
    st, rc, ts, keys = p.get("st", {}), p.get("rc", {}), p.get("ts", {}), p.get("keys", {})
    if not st:
        note("statements present", False, "a package without a statement proves nothing")
    attested = {"keys": {}, "identities": {}}
    if p.get("att"):
        from .attested import evidence
        attested = evidence.check(st, p["att"], note)
    signer = {part: attested["keys"].get(part) or (_pub(keys[_KEY_OF[part]]) if keys.get(_KEY_OF[part]) else None)
              for part in PARTS}
    bodies: dict[str, dict] = {}
    for part in PARTS:
        if part not in st:
            continue
        key = signer[part]
        if key is None:
            note(f"{_TITLE[part]} statement signature", False, f"no {_KEY_OF[part]} key in the package")
            continue
        try:
            read = read_statement(st[part], key)
            bodies[part] = read.get("payload") or {}
            claims = read.get("claims") or {}
            note(f"{_TITLE[part]} statement signature", True,
                 f"issuer {claims.get(cose.ISS)!r}, subject {claims.get(cose.SUB)!r}, "
                 f"key {key.public_bytes_raw().hex()[:16]}…"
                 + (" (certified on attested hardware)" if part in attested["keys"] else ""))
        except Exception as exc:  # noqa: BLE001 - a failed check is a result, not a crash
            note(f"{_TITLE[part]} statement signature", False, f"{type(exc).__name__}: {exc}")
    text = p.get("text")
    ver = bodies.get("ver") or {}
    if text is not None and ver:
        canon = (ver.get("dec") or {}).get("canon") or CANON_ID
        want = sha256(canonicalize(text, canon).encode())
        note("the verdict is about this text", want == ver.get("cand"),
             f"canonical form {canon}, digest {want.hex()[:16]}…")
    if text is not None and (bodies.get("det") or {}):
        det = bodies["det"]
        want = sha256(canonicalize(text, CANON_ID).encode())
        note("the detection is about this text", want == det.get("cand"), det.get("technique", ""))
    if "gen" in st and ver:
        note("the verdict names this generation statement", ver.get("gen") == statement_ref(st["gen"]),
             "reference = SHA-256 of the logged statement")
    if "gen" in bodies and ver:
        doc_claim = ((read_statement(st["gen"], signer["gen"]).get("claims")) or {}).get(cose.SUB)
        note("both statements speak about the same document", doc_claim == ver.get("doc"), str(doc_claim))
    for part, body in (("gen", ver), ("det", bodies.get("det") or {})):
        ids = attested["identities"].get(part)
        if ids and body:
            note(f"the {'verifier' if part == 'gen' else 'detector'} read with the attested model",
                 body.get("wts") == ids.get("weights") and body.get("tok") == ids.get("tokenizer"),
                 f"weights {body.get('wts', b'').hex()[:16]}…, tokenizer {body.get('tok', b'').hex()[:16]}…")
    log_key = keys.get("log")
    for part in PARTS:
        if part not in rc:
            continue
        if not log_key:
            note(f"{_TITLE[part]} receipt", False, "no log key in the package")
            continue
        try:
            r = verify_receipt(rc[part], st[part], _pub(log_key))
            note(f"{_TITLE[part]} receipt: included in the log", True,
                 f"leaf {r['leaf_index']} of {r['tree_size']}, root {r['root'][:16]}…")
        except Exception as exc:  # noqa: BLE001
            note(f"{_TITLE[part]} receipt: included in the log", False, f"{type(exc).__name__}: {exc}")
    for part in PARTS:
        if part not in ts:
            continue
        try:
            info = ts3161.check(ts[part], stamped_digest(st[part]))
            note(f"{_TITLE[part]} time stamp: imprint", info["imprint_matches"],
                 f"{info.get('gen_time', '?')}, serial {info.get('serial')}, policy {info.get('policy')}")
            note(f"{_TITLE[part]} time stamp: authority signature", None, info["how_to_verify"])
        except Exception as exc:  # noqa: BLE001
            note(f"{_TITLE[part]} time stamp: imprint", False, f"{type(exc).__name__}: {exc}")
    note("the text re-read with the model", None,
         "not part of an offline check: run `entropy verify` with the model and the record (see README.md)")
    summary = {"doc": p.get("doc"), "verdict": ver.get("verdict"), "counts": ver.get("counts"),
               "detected": (bodies.get("det") or {}).get("detected"),
               "statements": sorted(st), "receipts": sorted(rc), "time_stamps": sorted(ts),
               "text_chars": len(text) if text is not None else None, "note": p.get("note", "")}
    return {"ok": all(c["ok"] for c in checks if c["ok"] is not None), "checks": checks, "summary": summary}


def explode(pack: bytes, out_dir: str | Path) -> list[str]:
    """Write the parts as files with a README that says what each one is and how to check it."""
    p = cbor2.loads(pack)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def put(name: str, data: bytes | str) -> None:
        f = d / name
        f.write_bytes(data.encode() if isinstance(data, str) else data)
        written.append(name)

    for part in PARTS:
        if part in p.get("st", {}):
            put(f"{_TITLE[part]}_statement.cose", p["st"][part])
        if part in p.get("rc", {}):
            put(f"{_TITLE[part]}_receipt.cose", p["rc"][part])
        if part in p.get("ts", {}):
            put(f"{_TITLE[part]}_timestamp.tsr", p["ts"][part])
    for name, raw in (p.get("keys") or {}).items():
        put(f"key_{name}.ed25519", raw)
    if p.get("text") is not None:
        put("text.txt", p["text"])
    report = check(pack)
    lines = [f"# Evidence package for document {p.get('doc')}", "",
             f"Verdict of record: **{report['summary'].get('verdict')}**"
             + (f" · detection: {report['summary'].get('detected')}" if "det" in p.get("st", {}) else ""), ""]
    if p.get("note"):
        lines += [f"Note from whoever built this package: {p['note']}", ""]
    lines += ["## What is inside", "",
              "- `text.txt` — the text this package is about, exactly as disclosed.",
              "- `*_statement.cose` — signed statements (COSE_Sign1, Ed25519): the issuer's generation statement, "
              "the verifier's verdict, a detector's result. They carry digests, never the text or the record.",
              "- `*_receipt.cose` — transparency-log receipts (RFC 9942): proof that a statement was registered.",
              "- `*_timestamp.tsr` — RFC 3161 tokens over the statement's logged form: it existed before that "
              "minute, according to an authority that is not us.",
              "- `key_*.ed25519` — the raw public keys those signatures verify under. Compare them with the keys "
              "the issuer publishes; this package cannot vouch for itself.", "",
              "## How to check it", "",
              "1. Everything that needs no model:", "", "   ```", "   entropy evidence --check package.cbor",
              "   ```", "",
              "2. The text re-read with the model (only the key holder can): ", "", "   ```",
              f"   entropy verify --model <model> --text-file text.txt --token {p.get('token') or '<token>'} "
              "--key <master key>", "   ```", "",
              "3. The time-stamping authority's own signature and certificate chain:", "", "   ```",
              "   openssl ts -verify -in verification_timestamp.tsr -data <stamped bytes> -CAfile <authority CA>",
              "   ```", "",
              "## What this package checked", ""]
    for c in report["checks"]:
        mark = "ok" if c["ok"] else ("—" if c["ok"] is None else "FAILED")
        lines.append(f"- [{mark}] {c['name']}" + (f" — {c['detail']}" if c["detail"] else ""))
    put("README.md", "\n".join(lines) + "\n")
    return written
