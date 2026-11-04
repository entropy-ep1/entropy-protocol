"""The forgeries an evidence check must not accept, run on every re-issued package of an attested run.

    python bench/attested/forgeries.py bench/results/attested/run-<stamp>

Each package of <run>/packs-v3 is checked against the run's trust list (<run>/packs-v3/trust.json) after one change:

  foreign-text          a foreign text, a verdict about it signed by a key the forger brings, and a log of the
                        forger's that registers both statements (the swap found on 01.11.2026; v0.1.0 accepted it)
  foreign-text-v2       the same with the commitment opening stripped, as a v2-style package
  text-only             only the text changed
  weak-policy           the attestation policy replaced by one with the AMD, vTPM and GPU requirements removed
  from-nothing          a package with the same text whose keys, log and commitment are all the forger's own

Expected: foreign-text and text-only FAILED; foreign-text-v2, weak-policy and from-nothing no better than INTACT.
The result is written to <run>/packs-v3/forgeries.json.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from entropy_protocol import cose, pack, statements, tlog
from entropy_protocol.canon import canonicalize

FOREIGN = "The treasurer authorised a payment of 900,000 USD. No model wrote this sentence."
EXPECT = {"foreign-text": {pack.FAILED}, "foreign-text-v2": {pack.FAILED, pack.INTACT},
          "text-only": {pack.FAILED}, "weak-policy": {pack.FAILED, pack.INTACT},
          "from-nothing": {pack.FAILED, pack.INTACT}}


def _verdict(old_ver: bytes, text: str, key: Ed25519PrivateKey, gen: bytes | None = None) -> bytes:
    """The genuine verdict's body, re-pointed at ``text`` (and at ``gen``, if given) and signed by ``key``."""
    prot, _, _, payload, _ = cose.decode1(old_ver)
    body = dict(cose.loads(payload))
    body["cand"] = statements.sha256(canonicalize(text, body["dec"]["canon"]).encode())
    if gen is not None:
        body["gen"] = statements.statement_ref(gen)
    claims = prot[cose.CWT_CLAIMS]
    return cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(key.public_key()),
                       cose.CWT_CLAIMS: {cose.ISS: claims[cose.ISS], cose.SUB: body["doc"], cose.IAT: claims[cose.IAT]},
                       cose.TYP: statements.TYP_VERIFICATION}, cose.dumps(body), key)


def foreign_text(p: dict, opening: bool) -> bytes:
    forger, forger_log = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    ver = _verdict(p["st"]["ver"], FOREIGN, forger)
    log = tlog.TransparencyLog(forger_log)
    _, rg = log.register(p["st"]["gen"])
    _, rv = log.register(ver)
    return pack.build(doc=p["doc"], text=FOREIGN, statements={"gen": p["st"]["gen"], "ver": ver},
                      receipts={"gen": rg, "ver": rv}, timestamps={"gen": p["ts"]["gen"]},
                      keys={"verifier": forger.public_key().public_bytes_raw(),
                            "log": forger_log.public_key().public_bytes_raw()},
                      note=p.get("note", ""), att=p["att"], opening=p["open"] if opening else None)


def text_only(p: dict) -> bytes:
    q = copy.deepcopy(p)
    q["text"] = FOREIGN
    return cose.dumps(q)


def weak_policy(p: dict) -> bytes:
    q = copy.deepcopy(p)
    pol = json.loads(q["att"]["gen"]["policy"])
    pol.setdefault("amd", {}).update({"min_tcb": {}, "measurements": [], "id_key_digests": [],
                                      "alias_check_complete": False, "mitigation_vector_bits": []})
    pol["vtpm"], pol["gpu"] = {}, {"count": 1, "rim": "off", "ocsp": "off"}
    for role in ("gen", "det"):
        if role in q["att"]:
            q["att"][role]["policy"] = json.dumps(pol)
    return cose.dumps(q)


def from_nothing(p: dict) -> bytes:
    issuer, verifier, log_key = (Ed25519PrivateKey.generate() for _ in range(3))
    opened = {**p["open"], "rec": statements.sha256(b"a record nobody sealed")}
    payload = statements.sha256(cose.dumps(statements.commitment_of_text(opened, p["doc"], p["text"])))
    gen = cose.sign1({cose.ALG: cose.ED25519, cose.KID: cose.kid_of(issuer.public_key()),
                      cose.CWT_CLAIMS: {cose.ISS: "forger", cose.SUB: p["doc"], cose.IAT: 1_793_000_000},
                      cose.TYP: statements.TYP_GENERATION, cose.PAYLOAD_HASH_ALG: cose.SHA256,
                      cose.PREIMAGE_CONTENT_TYPE: statements.CT_COMMITMENT}, payload, issuer)
    ver = _verdict(p["st"]["ver"], p["text"], verifier, gen)
    log = tlog.TransparencyLog(log_key)
    _, rg = log.register(gen)
    _, rv = log.register(ver)
    return pack.build(doc=p["doc"], text=p["text"], statements={"gen": gen, "ver": ver},
                      receipts={"gen": rg, "ver": rv},
                      keys={"issuer": issuer.public_key().public_bytes_raw(),
                            "verifier": verifier.public_key().public_bytes_raw(),
                            "log": log_key.public_key().public_bytes_raw()}, opening=opened)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    out = Path(a.run) / "packs-v3"
    trust = pack.load_trust(out / "trust.json")
    makers = {"foreign-text": lambda p: foreign_text(p, True), "foreign-text-v2": lambda p: foreign_text(p, False),
              "text-only": text_only, "weak-policy": weak_policy, "from-nothing": from_nothing}
    levels = {k: collections.Counter() for k in makers}
    reasons = {k: collections.Counter() for k in makers}
    files = sorted(out.glob("*.cbor"))
    for f in files:
        p = cose._thaw(cose.loads(f.read_bytes()))
        for name, make in makers.items():
            r = pack.check(make(p), trust)
            levels[name][r["level"]] += 1
            for c in r["checks"]:
                if c["ok"] is False:
                    reasons[name][c["name"]] += 1
            for u in r.get("unvouched", []):
                reasons[name]["not on the trust list: " + (u.split(":")[0] if ":" in u else u.split(" ")[0] + " key")] += 1
    result = {"packages": len(files), "trust_list": trust["name"],
              "forgeries": {k: {"levels": dict(levels[k]), "expected": sorted(EXPECT[k]),
                                "held": set(levels[k]) <= EXPECT[k], "why": dict(reasons[k].most_common(4))}
                            for k in makers}}
    (out / "forgeries.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))
    return 0 if files and all(v["held"] for v in result["forgeries"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
