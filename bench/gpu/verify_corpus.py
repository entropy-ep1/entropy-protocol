#!/usr/bin/env python3
"""Records at scale (G1-G3, G6) and one seal verified on another configuration (G4): runs a sealed corpus
through the batched verifier on this machine's configuration (GPU, precision, attention kernel, batch size).

Per text: unchanged; 7 transports; 5 edits with the edit position (localization); an unrelated text (the next
text of the corpus) under this text's record; privacy scan of the decrypted record. Records carry their own
calibration table; the verifier applies its configuration class (calibrated = False outside the table).

Output: runs/<model>/verify_<corpus>_<config>.jsonl + .summary.json
Usage: python3 bench/gpu/verify_corpus.py --model M --corpus c4_sealed_L200 --dtype float16 [--batch 32] [--attn eager]
"""
from __future__ import annotations

import argparse
import re
import time

from common import (RUNS, Store, add_backend_args, chunks, load_backend, log, program_keys, provenance, read_jsonl,
                    slug, write_json)
from entropy_protocol import crypto
from entropy_protocol.batch import verify_batch
from entropy_protocol.canon import canonicalize
from entropy_protocol.record import loads
from entropy_protocol.verify import INDETERMINATE, MODIFIED, VERIFIED
from gates import TRANSPORTS, _cpos, _header_words, edits


def localized(r, cp: int) -> bool:
    spans = r.changed_ranges + r.divergence_ranges
    return any(a - 2 <= cp <= b + 2 for a, b in spans) or any(abs(p - cp) <= 2 for p in r.deletion_points)


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap, dtype="float16")
    ap.add_argument("--corpus", required=True, help="runs/<model>/<corpus>.jsonl written by seal_corpus.py")
    ap.add_argument("--batch", type=int, default=32, help="texts per verification batch (the configuration's batch)")
    ap.add_argument("--n", type=int, default=0, help="first n texts only (0 = all)")
    ap.add_argument("--only-same", action="store_true", help="G4: unchanged text and transports only")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    backend = load_backend(a.model, a.device, a.dtype, a.attn)
    cfg = f"{backend.config_class}_{a.attn or 'default'}_b{a.batch}" + (f"_{a.tag}" if a.tag else "")
    rows_in = [r for r in read_jsonl(RUNS / slug(a.model) / f"{a.corpus}.jsonl") if r.get("mode") == "sealed"]
    if a.n:
        rows_in = rows_in[:a.n]
    out = RUNS / slug(a.model) / f"verify_{a.corpus}_{cfg}.jsonl"
    st = Store(out)
    masters = program_keys()
    todo = [r for r in rows_in if r["id"] not in st.done]
    log(f"verify {a.corpus} on {cfg}: {len(todo)} of {len(rows_in)} to do")
    nxt = {r["id"]: rows_in[(i + 1) % len(rows_in)]["text"] for i, r in enumerate(rows_in)}
    t_start = time.time()
    for group in chunks(todo, a.batch):
        keys = [crypto.log_key(masters[r["key"]], r["doc_id"]) for r in group]
        jobs = []                                           # (row index, kind, name, candidate, edit position)
        for i, r in enumerate(group):
            jobs.append((i, "same", "same", r["text"], None))
            for name, f in TRANSPORTS.items():
                jobs.append((i, "transport", name, f(r["text"]), None))
            if not a.only_same:
                base = canonicalize(r["text"])
                for name, (txt, pos) in edits(r["text"]).items():
                    if canonicalize(txt) == base:           # the "edit" left the canonical text as it was
                        jobs.append((i, "noop", name, txt, None))
                        continue
                    jobs.append((i, "edit", name, txt, _cpos(r["text"], pos)))
                jobs.append((i, "unrelated", "next_text", nxt[r["id"]], None))
        res = []
        for part in chunks(jobs, a.batch):                  # the configuration's batch size, whatever the job mix
            res += verify_batch(backend, [j[3] for j in part], [group[j[0]]["token"] for j in part],
                                [keys[j[0]] for j in part])
        per = {i: {"id": r["id"], "transport": {}, "edits": {}} for i, r in enumerate(group)}
        for (i, kind, name, _, cp), v in zip(jobs, res):
            o = per[i]
            if kind == "same":
                o.update(same=v.verdict, calibrated=v.calibrated, matched=v.matched, divergent=v.divergent,
                         indeterminate=v.indeterminate, entries=v.entries, reason=v.reason)
            elif kind == "transport":
                o["transport"][name] = v.verdict
            elif kind == "edit":
                o["edits"][name] = {"verdict": v.verdict, "localized": localized(v, cp),
                                    "model_layer": v.divergent > 0}
            elif kind == "noop":
                o.setdefault("noop_edits", {})[name] = v.verdict   # must be VERIFIED: nothing changed
            else:
                o["unrelated"] = v.verdict
        for i, r in enumerate(group):
            if not a.only_same:                              # G6: the decrypted record holds no text
                rec = loads(crypto.decrypt_token(keys[i], r["token"]))
                allowed = " ".join(_header_words(rec.header.__dict__)).lower()
                blob = crypto.decrypt_token(keys[i], r["token"]).decode("latin-1").lower()
                per[i]["leaks"] = [w for w in set(re.findall(r"[a-z]{5,}", r["text"].lower()))
                                   if w in blob and w not in allowed]
            st.write(per[i])
        log(f"  {len(st.done)}/{len(rows_in)}  {time.time() - t_start:.0f}s")

    rows = [r for r in st.rows() if r["id"] in {x["id"] for x in rows_in}]
    dec = [r for r in rows if r["same"] != INDETERMINATE]

    def pct(x, n):
        return round(100 * x / n, 3) if n else None
    s = {"corpus": a.corpus, "config": cfg, "texts": len(rows), "decisive": len(dec),
         "same_VERIFIED_%": pct(sum(r["same"] == VERIFIED for r in dec), len(dec)),
         "same_wrong_MODIFIED": sum(r["same"] == MODIFIED for r in rows),
         "same_INDETERMINATE": sum(r["same"] == INDETERMINATE for r in rows),
         "calibrated": sorted({str(r.get("calibrated")) for r in rows}),
         "transport_wrong_MODIFIED": {n: sum(r["transport"][n] == MODIFIED for r in rows) for n in TRANSPORTS},
         "transport_VERIFIED_%": {n: pct(sum(r["transport"][n] == VERIFIED for r in dec), len(dec)) for n in TRANSPORTS}}
    if not a.only_same:
        s["unrelated_false_VERIFIED"] = sum(r.get("unrelated") == VERIFIED for r in rows)
        s["edits"] = {}
        for name in ("replace_word", "one_char", "delete_sentence", "insert_sentence", "truncate_tail"):
            es = [r["edits"][name] for r in dec if name in r["edits"]]
            det = [e for e in es if e["verdict"] == MODIFIED]
            s["edits"][name] = {"n": len(es), "noop_skipped": sum(name in r.get("noop_edits", {}) for r in dec),
                                "MODIFIED_%": pct(len(det), len(es)),
                                "false_VERIFIED": sum(e["verdict"] == VERIFIED for e in es),
                                "localized_%": pct(sum(e["localized"] for e in det), len(det)),
                                "model_layer_%": pct(sum(e["model_layer"] for e in es), len(es))}
        s["noop_edits_not_VERIFIED"] = sum(v != VERIFIED for r in rows for v in r.get("noop_edits", {}).values())
        s["privacy_leaks"] = sum(bool(r.get("leaks")) for r in rows)
    write_json(out.with_suffix(".summary.json"), {**provenance(backend, batch=a.batch), "summary": s})
    log(str(s))


if __name__ == "__main__":
    main()
