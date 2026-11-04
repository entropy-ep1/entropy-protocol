"""Re-issue an attested run's evidence packages as `ep1-pack/v3`, with the opening of each generation statement's
commitment, and write the run's trust list.

    python bench/attested/reissue_packs.py bench/results/attested/run-<stamp>

The statements, receipts and time stamps stay exactly as issued; only the opening is added - the record's digest and
the commitment's salt, read from the record delivered to the verifier (its private key stays with the operator:
<run>/verifier/verifier.key, ignored by git). With the opening, `entropy evidence --check` recomputes the commitment
from the disclosed text and compares it with what the attested generation statement signs: a foreign text fails,
whoever signed a verdict about it.

The trust list (<run>/packs-v3/trust.json) names what this run vouches for: the authority's root, the policy it
applied, the verifier's key and the log's key. A reader takes it from where the operator publishes it, not from a
package. Every package is checked against it and the reports are written beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from entropy_protocol import cose, pack, statements
from entropy_protocol.delivery import open_delivered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--verifier-key", help="the verifier's private key (default <run>/verifier/verifier.key)")
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "packs-v3"
    out.mkdir(exist_ok=True)
    verifier_key = bytes.fromhex(Path(a.verifier_key or run / "verifier/verifier.key").read_text().strip())
    olds = sorted((run / "packs").glob("*.cbor"))
    first = cose._thaw(cose.loads(olds[0].read_bytes()))
    trust = {"v": 1, "name": f"ep1 attested run {run.name}",
             "keys": {"verifier": [first["keys"]["verifier"].hex()], "log": [first["keys"]["log"].hex()]},
             "roots": [hashlib.sha256(first["att"]["root"]).hexdigest()],
             "policies": sorted({hashlib.sha256(first["att"][r]["policy"].encode()).hexdigest()
                                 for r in ("gen", "det") if r in first["att"]})}
    (out / "trust.json").write_text(json.dumps(trust, indent=1) + "\n")
    loaded = pack.load_trust(trust)
    summary = []
    for f in olds:
        p = cose._thaw(cose.loads(f.read_bytes()))
        delivered = run / "guest/out" / p["doc"] / "record.delivered.cose"
        doc, plaintext = open_delivered(delivered.read_bytes(), verifier_key)
        if doc != p["doc"]:
            raise SystemExit(f"{delivered}: record of {doc}, package of {p['doc']}")
        data = pack.build(doc=p["doc"], text=p["text"], token=p.get("token", ""), statements=p["st"],
                          receipts=p["rc"], timestamps=p["ts"], keys=p["keys"], note=p.get("note", ""),
                          att=p["att"], opening=statements.opening(plaintext))
        (out / f.name).write_bytes(data)
        report = pack.check(data, loaded)
        (out / f"{f.stem}.check.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
        failed = [c["name"] for c in report["checks"] if c["ok"] is False]
        summary.append({"doc": p["doc"], "level": report["level"], "checks": len(report["checks"]),
                        "failed": failed, "text_tied_to": report["summary"]["text_tied_to"],
                        "verdict": report["summary"].get("verdict")})
        print(p["doc"], report["level"], failed or "", flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    vouched = sum(s["level"] == pack.VOUCHED for s in summary)
    print(json.dumps({"packages": len(summary), "levels": {lv: sum(s["level"] == lv for s in summary)
                                                         for lv in {s["level"] for s in summary}}}, indent=1))
    return 0 if summary and vouched == len(summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
