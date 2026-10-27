"""Real RFC 3161 replies (TimeStampResp) for the timestamp_reply fuzz target's seeds, from the time-stamp authority
the attested run used. Three requests over the SHA-256 of real statements of that run (two asking for the
certificate, one not) and one request the authority must refuse (a SHA-256 imprint one octet short). Each request,
reply and the time are kept as evidence; fuzz/make_corpus.py turns the replies into seeds.

    python fuzz/fetch_tsa_replies.py   ->  bench/results/fuzz/tsa-replies-<stamp>/
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from entropy_protocol import ts3161  # noqa: E402

TSA = "http://timestamp.digicert.com"
RUN = ROOT / "bench/results/attested/run-20261027T172258Z/guest/out"


def ask(body: bytes) -> tuple[int, bytes]:
    req = urllib.request.Request(TSA, data=body, headers={"Content-Type": ts3161.CONTENT_TYPE,
                                                          "Accept": ts3161.REPLY_TYPE})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 - fixed authority URL
            return r.status, r.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / f"bench/results/fuzz/tsa-replies-{stamp}"
    out.mkdir(parents=True)
    docs = sorted(p for p in RUN.iterdir() if (p / "generation.cose").exists())[:3]
    asks = [(f"{d.name}-generation", ts3161.request(hashlib.sha256((d / "generation.cose").read_bytes()).digest(),
                                                    cert_req=k != 2)) for k, d in enumerate(docs)]
    short = hashlib.sha256(b"refused").digest()[:31]          # not a SHA-256 imprint: the authority must refuse it
    algorithm = ts3161._tlv(0x30, ts3161._tlv(0x06, ts3161.SHA256_OID) + ts3161._tlv(0x05, b""))
    asks.append(("short-imprint", ts3161._tlv(0x30, ts3161._int(1) + ts3161._tlv(0x30, algorithm + ts3161._tlv(0x04, short))
                                              + ts3161._int(7))))
    log = []
    for k, (name, body) in enumerate(asks):
        status, reply = ask(body)
        (out / f"request-{k}.tsq").write_bytes(body)
        (out / f"reply-{k}.tsr").write_bytes(reply)
        try:
            outcome = ts3161.token_info(ts3161.token_from_response(reply))
        except ValueError as exc:
            outcome = {"refused": str(exc)}
        log.append({"k": k, "name": name, "http": status, "request_sha256": hashlib.sha256(body).hexdigest(),
                    "reply_sha256": hashlib.sha256(reply).hexdigest(), "reply_bytes": len(reply), "outcome": outcome})
    (out / "MANIFEST.json").write_text(json.dumps({"authority": TSA, "fetched_at": stamp, "replies": log}, indent=1))
    print(json.dumps(log, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
