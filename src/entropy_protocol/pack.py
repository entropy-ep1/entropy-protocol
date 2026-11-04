"""Evidence package (`entropy evidence`): one file a third party can check without us and without any secret.

Inside: the text as disclosed, the signed statements about it (generation by the issuer, verification by the
verifier, detection when there is one), their transparency-log receipts, the public keys those signatures are made
under, any RFC 3161 time-stamp token and - from `ep1-pack/v3` - the opening of the generation statement's
commitment: the record's digest and the commitment's salt, never the record. The record is the issuer's secret; the
statements speak about it only through digests (ARCHITECTURE §7).

`check` re-runs, offline, everything that needs no model: every signature; the commitment recomputed from the
disclosed text and the opening, which must be the one the generation statement signs; the digest that ties the text
to the verdict and the reference that ties the verdict to the generation statement; inclusion in the log; and the
time-stamp imprint. Two things it cannot do and says so: read the text with the model again (that needs the model
and a key - `explode` writes the command), and validate a time-stamping authority's certificate chain.

A package carries the keys it was signed under, so it cannot vouch for itself. The answer has three levels:
FAILED - a check failed; INTACT - everything agrees under the keys the package brings, but no one vouches for those
keys, so treat it as unsigned; TRUSTED - it is intact and every key (for an attested key: the authority's root and
the policy it applied) is on a trust list the reader holds, as the issuer publishes it (`load_trust`).
"""
from __future__ import annotations

import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from . import cose, ts3161
from .canon import CANON_ID, CANON_V4, canonicalize
from .statements import (CT_COMMITMENT, commitment_of_text, read_statement, sha256, statement_ref)
from .tlog import verify_receipt

PACK_V = "ep1-pack/v1"
PACK_V2 = "ep1-pack/v2"     # + "att": attestation of the keys that signed (attested/evidence.py, ARCHITECTURE §7d)
PACK_V3 = "ep1-pack/v3"     # + "open": the opening of the generation statement's commitment (statements.opening)
VERSIONS = (PACK_V, PACK_V2, PACK_V3)
PARTS = ("gen", "ver", "det")
_TITLE = {"gen": "generation", "ver": "verification", "det": "detection"}
_KEY_OF = {"gen": "issuer", "ver": "verifier", "det": "detector"}
_OPEN = {"rec": bytes, "salt": bytes, "canon": str, "wts": bytes, "tok": bytes}

TRUSTED, INTACT, FAILED = "TRUSTED", "INTACT", "FAILED"
MESSAGES = {TRUSTED: "Intact, and signed under keys on your trust list.",
            INTACT: "Bytes match, but no one vouches for these keys. Treat as unsigned.",
            FAILED: "A check failed. Do not rely on this package."}
EXIT_CODES = {TRUSTED: 0, FAILED: 1, INTACT: 3}     # 2 stays the usage error of the command line


def build(*, doc: str, text: str | None = None, token: str = "", statements: dict | None = None,
          receipts: dict | None = None, timestamps: dict | None = None, keys: dict | None = None,
          note: str = "", att: dict | None = None, opening: dict | None = None) -> bytes:
    """CBOR package. ``statements``/``receipts``/``timestamps`` are keyed "gen" / "ver" / "det"; ``keys`` holds raw
    32-byte Ed25519 public keys under "issuer" / "verifier" / "detector" / "log". With ``att`` (attested/evidence.py)
    the issuer's and the detector's keys come from certificates the attestation establishes. ``opening``
    (`statements.opening` of the record) lets anyone tie the text to the generation statement: give it whenever the
    package holds a generation statement and its text."""
    pack = {"v": PACK_V3 if opening else (PACK_V2 if att else PACK_V), "doc": doc, "token": token,
            "st": {k: v for k, v in (statements or {}).items() if v},
            "rc": {k: v for k, v in (receipts or {}).items() if v},
            "ts": {k: v for k, v in (timestamps or {}).items() if v},
            "keys": {k: v for k, v in (keys or {}).items() if v}, "note": note}
    if text is not None:
        pack["text"] = text
    if att:
        pack["att"] = att
    if opening:
        pack["open"] = {k: opening[k] for k in _OPEN}
    return cose.dumps(pack)


def load_trust(source: str | Path | dict) -> dict:
    """A trust list as an issuer publishes it (JSON): {"v": 1, "name": "...", "keys": {"issuer" | "verifier" |
    "detector" | "log": [hex of a raw Ed25519 public key]}, "roots": [SHA-256 hex of an authority's root
    certificate, DER], "policies": [SHA-256 hex of an appraisal policy's text]}. Anything else is refused, so a typo
    cannot widen what is trusted."""
    data = source if isinstance(source, dict) else json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("v") != 1:
        raise ValueError("not a trust list: {\"v\": 1, ...} expected")
    unknown = set(data) - {"v", "name", "keys", "roots", "policies"}
    if unknown:
        raise ValueError(f"unknown fields in the trust list: {sorted(unknown)}")
    keys = data.get("keys") or {}
    if not isinstance(keys, dict) or set(keys) - set(_KEY_OF.values()) - {"log"}:
        raise ValueError("trust list keys: only issuer, verifier, detector and log")

    def digests(values, what: str) -> set[str]:
        if not isinstance(values or [], list):
            raise ValueError(f"trust list {what}: a list of hex strings expected")
        out = set()
        for v in values or []:
            raw = bytes.fromhex(v) if isinstance(v, str) else b""
            if len(raw) != 32:
                raise ValueError(f"trust list {what}: 32 bytes in hex expected, got {v!r}")
            out.add(raw.hex())
        return out

    return {"name": str(data.get("name", "")), "keys": {k: digests(v, k) for k, v in keys.items()},
            "roots": digests(data.get("roots"), "roots"), "policies": digests(data.get("policies"), "policies")}


def stamped_digest(statement: bytes) -> bytes:
    """What a time stamp is taken over: the statement's logged form, so a receipt can still be added afterwards."""
    return statement_ref(statement)


def _pub(raw: bytes) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(raw)


def _report(checks: list[dict], summary: dict, trust: dict | None, unvouched: list[str]) -> dict:
    if any(c["ok"] is False for c in checks):
        level = FAILED
    elif trust is None or unvouched:
        level = INTACT
    else:
        level = TRUSTED
    report = {"ok": level == TRUSTED, "level": level, "message": MESSAGES[level], "checks": checks,
              "summary": summary}
    if level == INTACT:
        report["unvouched"] = unvouched
    if trust is not None:
        report["trust_list"] = trust.get("name", "")
    return report


def check(pack: bytes, trust: dict | None = None) -> dict:
    """Every offline check, as a report: {"ok", "level", "message", "checks": [{"name", "ok", "detail"}],
    "summary"} plus "unvouched" (why it is only INTACT). ``trust`` is a loaded trust list (`load_trust`); without one
    the best answer is INTACT. ``ok`` is true only for TRUSTED. The package is untrusted input: whatever it holds,
    the answer is a report, and a package that cannot be read fails."""
    try:
        return _check(pack, trust)
    except Exception as exc:  # noqa: BLE001 - fails closed: an unreadable part can only make the package fail
        return _report([{"name": "package format", "ok": False, "detail": f"{type(exc).__name__}: {exc}"}], {},
                       trust, [])


def _opening_problem(opened) -> str | None:
    if not isinstance(opened, dict) or set(opened) != set(_OPEN):
        return f"fields {sorted(_OPEN)} expected"
    for k, t in _OPEN.items():
        if not isinstance(opened[k], t):
            return f"{k}: {t.__name__} expected"
    if len(opened["rec"]) != 32 or len(opened["salt"]) < 16:
        return "rec: 32 bytes; salt: 16 bytes or more"
    return None


def _check(pack: bytes, trust: dict | None) -> dict:
    p = cose.loads(pack)
    if not isinstance(p, dict):
        raise ValueError("not an evidence package (a CBOR map is expected)")
    checks: list[dict] = []

    def note(name: str, ok: bool | None, detail: str = "") -> None:
        checks.append({"name": name, "ok": ok, "detail": detail})

    if p.get("v") not in VERSIONS:
        return _report([{"name": "version", "ok": False, "detail": str(p.get("v"))}], {}, trust, [])
    st, rc, ts, keys = p.get("st", {}), p.get("rc", {}), p.get("ts", {}), p.get("keys", {})
    if not st:
        note("statements present", False, "a package without a statement proves nothing")
    attested = {"keys": {}, "identities": {}, "root": None, "policies": {}}
    if p.get("att"):
        from .attested import evidence
        attested = evidence.check(st, p["att"], note)
    signer = {part: attested["keys"].get(part) or (_pub(keys[_KEY_OF[part]]) if keys.get(_KEY_OF[part]) else None)
              for part in PARTS}
    bodies: dict[str, dict] = {}
    claims_of: dict[str, dict] = {}
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
            claims = claims_of[part] = read.get("claims") or {}
            note(f"{_TITLE[part]} statement signature", True,
                 f"issuer {claims.get(cose.ISS)!r}, subject {claims.get(cose.SUB)!r}, "
                 f"key {key.public_bytes_raw().hex()[:16]}…"
                 + (" (certified on attested hardware)" if part in attested["keys"] else ""))
        except Exception as exc:  # noqa: BLE001 - a failed check is a result, not a crash
            note(f"{_TITLE[part]} statement signature", False, f"{type(exc).__name__}: {exc}")
    text = p.get("text")
    tied: list[str] = []            # the signed statements this text is tied to
    opened = p.get("open")
    if p.get("v") == PACK_V3 or opened is not None:
        problem = _opening_problem(opened)
        note("commitment opening present and well formed", problem is None, problem or "record digest and salt")
        if problem is not None:
            opened = None
    if text is not None and "gen" in claims_of:
        if opened is not None:
            prot, _, _, payload, _ = cose.decode1(st["gen"])
            is_commitment = (prot.get(cose.PAYLOAD_HASH_ALG) == cose.SHA256
                             and prot.get(cose.PREIMAGE_CONTENT_TYPE) == CT_COMMITMENT)
            again = commitment_of_text(opened, claims_of["gen"].get(cose.SUB), text)
            same = is_commitment and sha256(cose.dumps(again)) == payload
            note("the text is the one the generation statement seals", same,
                 f"commitment recomputed from this text and the opening (canonical form {opened['canon']}, "
                 f"record {opened['rec'].hex()[:16]}…)" if is_commitment else "not a commitment (Hash Envelope)")
            if same:
                tied.append("gen")
            ids = attested["identities"].get("gen")
            if ids:
                note("the sealed text came from the attested model",
                     opened["wts"] == ids.get("weights") and opened["tok"] == ids.get("tokenizer"),
                     f"weights {opened['wts'].hex()[:16]}…, tokenizer {opened['tok'].hex()[:16]}…")
        else:
            note("the text tied to the generation statement", None,
                 "this package does not open the commitment: only a verdict can tie the text to the generation")
    ver = bodies.get("ver") or {}
    if text is not None and ver:
        canon = (ver.get("dec") or {}).get("canon") or CANON_ID
        want = sha256(canonicalize(text, canon).encode())
        about = want == ver.get("cand")
        note("the verdict is about this text", about, f"canonical form {canon}, digest {want.hex()[:16]}…")
        if about:
            tied.append("ver")
    if text is not None and (bodies.get("det") or {}):
        det = bodies["det"]
        # a detection names its canonical form; one that does not was made when v4 was the reference
        want = sha256(canonicalize(text, det.get("canon") or CANON_V4).encode())
        about = want == det.get("cand")
        note("the detection is about this text", about, det.get("technique", ""))
        if about:
            tied.append("det")
    if "gen" in st and ver:
        note("the verdict names this generation statement", ver.get("gen") == statement_ref(st["gen"]),
             "reference = SHA-256 of the logged statement")
    if "gen" in claims_of and ver:
        doc_claim = claims_of["gen"].get(cose.SUB)
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
    unvouched = _unvouched(trust, claims_of, signer, attested, rc, log_key)
    if text is not None and st and not tied:
        unvouched.append("nothing in the package ties this text to a signed statement")
    summary = {"doc": p.get("doc"), "verdict": ver.get("verdict"), "counts": ver.get("counts"),
               "detected": (bodies.get("det") or {}).get("detected"), "text_tied_to": [_TITLE[x] for x in tied],
               "statements": sorted(st), "receipts": sorted(rc), "time_stamps": sorted(ts),
               "text_chars": len(text) if text is not None else None, "note": p.get("note", "")}
    return _report(checks, summary, trust, unvouched)


def _unvouched(trust: dict | None, signed: dict, signer: dict, attested: dict, rc: dict, log_key) -> list[str]:
    """Why the reader's trust list does not cover this package (empty: it does)."""
    if trust is None:
        return ["no trust list given: the keys are the package's own"]
    out = []
    for part in PARTS:
        if part not in signed:
            continue
        if part in attested["keys"]:
            if attested["root"] not in trust["roots"]:
                out.append(f"{_TITLE[part]}: authority root {str(attested['root'])[:16]}… is not on the trust list")
            policy = attested["policies"].get(part)
            if policy not in trust["policies"]:
                out.append(f"{_TITLE[part]}: appraisal policy {str(policy)[:16]}… is not on the trust list")
        else:
            raw = signer[part].public_bytes_raw().hex()
            if raw not in trust["keys"].get(_KEY_OF[part], set()):
                out.append(f"{_TITLE[part]}: {_KEY_OF[part]} key {raw[:16]}… is not on the trust list")
    if rc and log_key and bytes(log_key).hex() not in trust["keys"].get("log", set()):
        out.append(f"log key {bytes(log_key).hex()[:16]}… is not on the trust list")
    return out


def explode(pack: bytes, out_dir: str | Path, trust: dict | None = None) -> list[str]:
    """Write the parts as files with a README that says what each one is and how to check it."""
    p = cose.loads(pack)
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
    report = check(pack, trust)
    lines = [f"# Evidence package for document {p.get('doc')}", "",
             f"**{report['level']}** — {report['message']}", "",
             f"Verdict claimed by the verifier: {report['summary'].get('verdict')}"
             + (f" · detection: {report['summary'].get('detected')}" if "det" in p.get("st", {}) else ""), ""]
    if report.get("unvouched"):
        lines += ["Not vouched for:", ""] + [f"- {u}" for u in report["unvouched"]] + [""]
    if p.get("note"):
        lines += [f"Note from whoever built this package: {p['note']}", ""]
    lines += ["## What is inside", "",
              "- `text.txt` — the text this package is about, exactly as disclosed.",
              "- `*_statement.cose` — signed statements (COSE_Sign1, Ed25519): the issuer's generation statement, "
              "the verifier's verdict, a detector's result. They carry digests, never the text or the record.",
              "- `*_receipt.cose` — transparency-log receipts (RFC 9942): proof that a statement was registered.",
              "- `*_timestamp.tsr` — RFC 3161 tokens over the statement's logged form: it existed before that "
              "minute, according to an authority that is not us.",
              "- `key_*.ed25519` — the raw public keys those signatures verify under. The package cannot vouch for "
              "them: a trust list from the issuer can.", "",
              "## How to check it", "",
              "1. Everything that needs no model, against the issuer's published trust list:", "", "   ```",
              "   entropy evidence --check package.cbor --trust trust.json", "   ```", "",
              "   Without `--trust` the best answer is INTACT: the bytes agree, but the keys are the package's own.", "",
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
