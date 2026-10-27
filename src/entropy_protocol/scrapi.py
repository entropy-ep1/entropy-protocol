"""SCRAPI (draft-ietf-scitt-scrapi-11) for the local transparency log: a minimal Transparency Service and a client.

Server (standard library only):
  GET  /.well-known/scitt-keys          COSE Key Set of the log's keys (application/cbor)
  GET  /.well-known/scitt-keys/{kid}    one COSE Key (kid as base64url or hex)
  POST /entries                         body: COSE_Sign1 (application/cose) -> 201 Created, Location, Receipt
  GET  /entries/{entry_id}              -> 200 Receipt at the current tree size, or 404
Errors: 4xx with application/concise-problem-details+cbor (RFC 9290: -1 title, -2 detail).
Registration policy (applied before anything else): the statement must verify under a registered issuer key,
matched by kid; its unprotected header is emptied before logging (tlog.entry_of).
"""
from __future__ import annotations

import base64
import hashlib
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cbor2
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from . import cose, tlog

CT_COSE = "application/cose"
CT_CBOR = "application/cbor"
CT_PROBLEM = "application/concise-problem-details+cbor"
OKP, CRV_ED25519 = 1, 6


def cose_key(public_key: Ed25519PublicKey) -> dict:
    """COSE_Key (RFC 9052 section 7) for an Ed25519 public key."""
    raw = public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {1: OKP, 2: cose.kid_of(public_key), 3: cose.ED25519, -1: CRV_ED25519, -2: raw}


def public_key_from_cose(key: dict) -> Ed25519PublicKey:
    if key.get(1) != OKP or key.get(-1) != CRV_ED25519:
        raise cose.COSEError("not an Ed25519 COSE key")
    return Ed25519PublicKey.from_public_bytes(key[-2])


class TransparencyService:
    """Registration policy + log. ``issuers``: public keys whose statements are accepted."""

    def __init__(self, log: tlog.TransparencyLog, issuers: list[Ed25519PublicKey]):
        self.log = log
        self.issuers = {cose.kid_of(k): k for k in issuers}
        self.ids: dict[str, int] = {}
        for i, e in enumerate(log.entries):
            self.ids[self.entry_id(e)] = i
        self.lock = threading.Lock()

    @staticmethod
    def entry_id(entry: bytes) -> str:
        return hashlib.sha256(entry).hexdigest()[:32]

    def register(self, statement: bytes) -> tuple[str, bytes]:
        prot, _, _, payload, _ = cose.decode1(statement)
        issuer = self.issuers.get(prot.get(cose.KID))
        if issuer is None:
            raise PermissionError("the statement's key is not registered with this Transparency Service")
        if payload is None:
            raise ValueError("detached payloads are not accepted")
        cose.verify1(statement, issuer)                     # raises COSEError
        with self.lock:
            index, receipt = self.log.register(statement)
            eid = self.entry_id(self.log.entries[index])
            self.ids[eid] = index
        return eid, receipt

    def receipt(self, eid: str) -> bytes | None:
        with self.lock:
            index = self.ids.get(eid)
            return None if index is None else self.log.receipt(index)


def _handler(ts: TransparencyService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            return

        def _send(self, code: int, body: bytes = b"", ctype: str | None = None, headers: dict | None = None):
            self.send_response(code)
            if ctype:
                self.send_header("Content-Type", ctype)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _problem(self, code: int, title: str, detail: str):
            self._send(code, cbor2.dumps({-1: title, -2: detail}), CT_PROBLEM)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/.well-known/scitt-keys":
                return self._send(200, cbor2.dumps([cose_key(ts.log.key.public_key())]), CT_CBOR)
            if path.startswith("/.well-known/scitt-keys/"):
                want = path.rsplit("/", 1)[1]
                k = cose_key(ts.log.key.public_key())
                forms = {k[2].hex(), base64.urlsafe_b64encode(k[2]).decode().rstrip("=")}
                if want in forms:
                    return self._send(200, cbor2.dumps(k), CT_CBOR)
                return self._problem(404, "Not Found", "no key with this identifier")
            if path.startswith("/entries/"):
                r = ts.receipt(path.rsplit("/", 1)[1])
                if r is None:
                    return self._problem(404, "Not Found", "no entry with this identifier")
                return self._send(200, r, CT_COSE)          # no header echoes the request (Location: on POST only)
            return self._problem(404, "Not Found", "unknown resource")

        def do_POST(self):  # noqa: N802
            if self.path.split("?")[0] != "/entries":
                return self._problem(404, "Not Found", "unknown resource")
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != CT_COSE:
                return self._problem(415, "Unsupported Media Type", f"expected {CT_COSE}")
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                eid, receipt = ts.register(body)
            except PermissionError as exc:
                return self._problem(400, "Rejected by registration policy", str(exc))
            except (cose.COSEError, ValueError) as exc:
                return self._problem(400, "Bad Signed Statement", str(exc))
            return self._send(201, receipt, CT_COSE, {"Location": f"/entries/{eid}"})

    return Handler


def serve(ts: TransparencyService, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    """Start the service in a background thread; ``server.server_address`` gives the bound port."""
    server = ThreadingHTTPServer((host, port), _handler(ts))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ------------------------------------------------------------------ client
class SCRAPIError(Exception):
    def __init__(self, status: int, title: str, detail: str):
        super().__init__(f"{status} {title}: {detail}")
        self.status, self.title, self.detail = status, title, detail


def _request(url: str, data: bytes | None = None, ctype: str | None = None, accept: str = CT_COSE):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    req.add_header("Accept", accept)
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        try:
            prob = cose.loads(body)
            raise SCRAPIError(exc.code, prob.get(-1, ""), prob.get(-2, "")) from None
        except (ValueError, AttributeError, cbor2.CBORDecodeError):
            raise SCRAPIError(exc.code, exc.reason, body[:200].decode(errors="replace")) from None


def register(base_url: str, statement: bytes) -> tuple[str, bytes]:
    """POST a Signed Statement; returns (Location, Receipt). Follows 202 by polling the Location."""
    status, headers, body = _request(base_url.rstrip("/") + "/entries", statement, CT_COSE)
    location = headers.get("Location", "")
    if status == 201:
        return location, body
    if status == 202:
        import time
        for _ in range(60):
            time.sleep(float(headers.get("Retry-After", 1)))
            status, headers, body = _request(base_url.rstrip("/") + location)
            if status == 200:
                return location, body
    raise SCRAPIError(status, "unexpected status", "")


def service_keys(base_url: str) -> list[Ed25519PublicKey]:
    _, _, body = _request(base_url.rstrip("/") + "/.well-known/scitt-keys", accept=CT_CBOR)
    return [public_key_from_cose(dict(k)) for k in cose.loads(body)]


def verify_receipt_online(base_url: str, receipt: bytes, statement: bytes) -> dict:
    """Verify a receipt with the service's published key whose kid matches the receipt's."""
    kid = cose.decode1(receipt)[0].get(cose.KID)
    for key in service_keys(base_url):
        if cose.kid_of(key) == kid:
            return tlog.verify_receipt(receipt, statement, key)
    raise cose.COSEError("no published key matches the receipt")


def load_public(path: str) -> Ed25519PublicKey:
    data = open(path, "rb").read()
    try:
        return serialization.load_pem_public_key(data)
    except ValueError:
        return serialization.load_pem_private_key(data, password=None).public_key()


def new_log(key: Ed25519PrivateKey, path=None) -> tlog.TransparencyLog:
    return tlog.TransparencyLog(key, path)
