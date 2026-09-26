#!/usr/bin/env python3
"""G10 at scale — record-free detection: power by length, the empirical null over >= 10^6 (text, key) pairs,
other-key nulls, attacks, localization in mixed documents.

One verification pass per text (batched). Detection under many keys needs no further model work: the
distinct-context positions, the rank of each token and the near-tie exclusion do not depend on the key, so a
key only decides which positions are checkpoints and which predicate each carries (one HMAC per position).
Prefixes: the pass over a text also scores each of its prefixes (causal model), so the lengths 50...1000 are
evaluated on the first L canonical tokens of the same texts.

Usage: python3 bench/gpu/detect_corpus.py --model M --dtype float16 --positives c4_sealed_L1000 \
         --lengths 50,100,200,400,1000 --nulls data:c4:human,data:wiki_en:text,runs:c4_plain:text \
         --null-keys 50 --attacks runs:attacks_c4_sealed_L400 --cal cal.json --name main
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from common import (DATA, RUNS, Store, add_backend_args, load_backend, log, null_keys, program_keys, provenance,
                    read_jsonl, slug, write_json)
from entropy_protocol import crypto
from entropy_protocol.batch import detect_from_scores, fold_all, reading_scores, score_batch
from entropy_protocol.detect import binomial_tail, detection_tokens, readings_of, unsatisfiable

ALPHAS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def _source(spec: str, model: str) -> tuple[str, list[dict], str]:
    """'data:<file>:<field>' or 'runs:<corpus>:<field>' -> (label, rows, field)."""
    kind, name, field = spec.split(":")
    rows = read_jsonl((DATA / f"{name}.jsonl") if kind == "data" else (RUNS / slug(model) / f"{name}.jsonl"))
    return f"{kind}:{name}", [r for r in rows if r.get(field)], field


def _position_data(ids, sc, k: int, tol: float, context: int, gate: float = 0.0, backend=None):
    """Key-independent part of detection: distinct-context positions with (context, first choice?, excluded?).
    Excluded = below the tolerance (near tie), above the confidence gate, or where canonical generation could not
    have emitted the first choice (detect.unsatisfiable) — all read from the text and the model, not the key."""
    seen, pos = set(), []
    for t in range(context, len(ids)):
        ctx = tuple(ids[t - context:t])
        if ctx not in seen:
            seen.add(ctx)
            pos.append(t)
    unsat = (unsatisfiable(backend, ids, pos, sc.first[pos].tolist())
             if backend is not None and sc.first is not None else None) or [False] * len(pos)
    out = []
    for t, u in zip(pos, unsat):
        ctx = tuple(ids[t - context:t])
        rank0 = int(sc.rank0[t])
        excluded = False
        if tol > 0:                                   # k = 1: LOW and HIGH margins coincide (key-independent)
            top = sc.top[t]
            boundary = float(top[1]) if rank0 == 0 else float(top[0])
            excluded = abs(float(sc.l_tok[t]) - boundary) < tol
        if gate and float(sc.top_p[t][0]) >= gate:
            excluded = True
        out.append((ctx, rank0 == 0, excluded or u))
    return out


def _pvals_under_keys(job):
    """Worker: p-values of one text under every key (exact Binomial(n, 1/2) tail)."""
    positions, keys, interval = job
    res = []
    for kw in keys:
        n = m = 0
        for ctx, first, excluded in positions:
            prop = crypto.wm_decide(kw, ctx, interval)
            if prop is None or excluded:
                continue
            n += 1
            m += int(first if prop == 0 else not first)     # LOW holds iff first choice; HIGH iff not (k = 1)
        res.append((n, m, binomial_tail(n, m)))
    return res


def rates(ps) -> dict:
    return {f"{a:g}": {"rate": round(sum(p <= a for p in ps) / max(1, len(ps)), 6), "count": sum(p <= a for p in ps)}
            for a in ALPHAS}


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap, dtype="float16")
    ap.add_argument("--positives", default="", help="sealed corpus (runs/<model>/<name>.jsonl)")
    ap.add_argument("--lengths", default="50,100,200,400,1000")
    ap.add_argument("--nulls", default="", help="comma list of data:<file>:<field> / runs:<corpus>:<field>")
    ap.add_argument("--null-keys", type=int, default=50)
    ap.add_argument("--max-null-texts", type=int, default=20000)
    ap.add_argument("--other-key", type=int, default=7, help="program key index used as the wrong key (corpora "
                    "rotate over keys 0-4)")
    ap.add_argument("--attacks", default="", help="comma list of runs:<attack file> (rows: id, attack, text)")
    ap.add_argument("--cal", default="", help="calibration JSON: the detector uses its class's tolerance")
    ap.add_argument("--interval", type=int, default=5)
    ap.add_argument("--context", type=int, default=4)
    ap.add_argument("--gate", type=float, default=0.0, help="confidence gate of the sealed corpus (0 = none)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--restarts", type=int, default=32, help="attacks: paragraph-start readings per text (0 = off)")
    ap.add_argument("--restart-attacks", default="embed_10,embed_25,embed_3spans_25,long_doc",
                    help="attacks that get the paragraph-start readings (empty = all; they cost the most: "
                         "on 15.09 long_doc alone took 12 of the 26 minutes of phase A detection)")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    t0 = time.time()
    backend = load_backend(a.model, a.device, a.dtype, a.attn)
    cal = json.loads(open(a.cal).read()) if a.cal else {}
    tol = float(cal.get("table", {}).get(backend.config_class, max(cal.get("table", {"x": 0.0}).values()))) if cal else 0.0
    if cal and cal.get("from_position", 0) > a.context:   # the detector counts positions from a.context on
        raise SystemExit(f"calibration measured positions from {cal['from_position']} on; the detector reads from "
                         f"{a.context} on")
    k = 1
    out_dir = RUNS / slug(a.model)
    report = {**provenance(backend), "name": a.name, "tol": tol, "interval": a.interval, "context": a.context,
              "alphas": list(ALPHAS), "positives": {}, "nulls": {}, "attacks": {}}
    masters = program_keys()

    def scored(texts):
        toks = [detection_tokens(backend, t) for t in texts]
        todo = [i for i, (_, _, n) in enumerate(toks) if n > a.context]
        return toks, dict(zip(todo, score_batch(backend, [toks[i][0] for i in todo], k, a.token_budget)))

    # ---------------------------------------------------------------- positives by length, other key
    if a.positives:
        rows = [r for r in read_jsonl(out_dir / f"{a.positives}.jsonl") if r.get("mode") == "sealed"]
        toks, scores = scored([r["text"] for r in rows])
        per = Store(out_dir / f"detect_{a.name}_positives.jsonl")
        for L in (int(x) for x in a.lengths.split(",")):
            ps, ns, ms, full = [], [], [], 0
            for key_i in sorted({r["key"] for r in rows}):
                idx = [i for i, r in enumerate(rows) if r["key"] == key_i]
                cut = [(toks[i][0][:L], toks[i][1][:L], min(toks[i][2], L)) for i in idx]
                sub = {j: scores[i] for j, i in enumerate(idx) if i in scores}
                dets = detect_from_scores(cut, sub, crypto.wm_key(masters[key_i]), interval=a.interval, k=k,
                                          context=a.context, tol=tol, alpha=1e-6, localize=False, min_run=8, t0=time.time(),
                                          gate=a.gate, backend=backend)
                for i, d in zip(idx, dets):
                    ps.append(d.p_value); ns.append(d.checkpoints); ms.append(d.matched)
                    full += toks[i][2] >= L
                    rid = f"{rows[i]['id']}@{L}"
                    if rid not in per.done:
                        per.write({"id": rid, "L": L, "p": d.p_value, "n": d.checkpoints, "m": d.matched,
                                   "tokens": min(toks[i][2], L)})
            report["positives"][str(L)] = {"texts": len(ps), "texts_reaching_L": full, "detected_at": rates(ps),
                                           "median_p": sorted(ps)[len(ps) // 2] if ps else None,
                                           "mean_checkpoints": round(sum(ns) / max(1, len(ns)), 1),
                                           "match_rate": round(sum(ms) / max(1, sum(ns)), 4)}
            log(f"  positives L={L}: {report['positives'][str(L)]['detected_at']}")
        other = crypto.wm_key(masters[a.other_key])
        dets = detect_from_scores(toks, scores, other, interval=a.interval, k=k, context=a.context, tol=tol,
                                  alpha=1e-6, localize=False, min_run=8, t0=time.time(), gate=a.gate, backend=backend)
        wrong = [d.p_value for d, r in zip(dets, rows) if r["key"] != a.other_key]
        report["nulls"]["same_model_other_key"] = {"pairs": len(wrong), "false_positive_at": rates(wrong),
                                                   "min_p": min(wrong) if wrong else None}

    # ---------------------------------------------------------------- empirical null over many keys
    keys = null_keys(a.null_keys)
    for spec in (s for s in a.nulls.split(",") if s):
        label, rows, field = _source(spec, a.model)
        rows = rows[:a.max_null_texts]
        texts = [r[field] for r in rows]
        tb = time.time()
        toks, scores = scored(texts)
        jobs = [(_position_data(toks[i][0], scores[i], k, tol, a.context, a.gate, backend), keys, a.interval)
                for i in scores]
        with ProcessPoolExecutor(a.workers) as pool:
            res = list(pool.map(_pvals_under_keys, jobs, chunksize=16))
        ps = [p for text in res for (_, _, p) in text]
        ns = [n for text in res for (n, _, _) in text]
        report["nulls"][label] = {"texts": len(res), "keys": len(keys), "pairs": len(ps),
                                  "false_positive_at": rates(ps), "min_p": min(ps) if ps else None,
                                  "mean_checkpoints": round(sum(ns) / max(1, len(ns)), 1),
                                  "seconds": round(time.time() - tb, 1)}
        log(f"  null {label}: {len(ps)} pairs, {report['nulls'][label]['false_positive_at']}")

    # ---------------------------------------------------------------- attacks (true key of each text)
    for spec in (s for s in a.attacks.split(",") if s):
        rows = read_jsonl(out_dir / f"{spec.split(':', 1)[1]}.jsonl")
        for attack in sorted({r["attack"] for r in rows}):
            sub = [r for r in rows if r["attack"] == attack]
            rs = a.restarts if (not a.restart_attacks or attack in a.restart_attacks.split(",")) else 0
            toks, scores = scored([r["text"] for r in sub])
            # a pasted passage is also read from each paragraph start (as detect.detect does); key-independent pass
            readings = [readings_of(backend, r["text"], limit=rs) if rs else [] for r in sub]
            r_scores = reading_scores(backend, readings, k, a.token_budget)
            ps, segs = [], []
            for key_i in sorted({r["key"] for r in sub}):
                idx = [i for i, r in enumerate(sub) if r["key"] == key_i]
                kw_ = crypto.wm_key(masters[key_i])
                dets = detect_from_scores([toks[i] for i in idx], {j: scores[i] for j, i in enumerate(idx) if i in scores},
                                          kw_, interval=a.interval, k=k, context=a.context,
                                          tol=tol, alpha=1e-6, localize=True, min_run=8, t0=time.time(), gate=a.gate,
                                          backend=backend)
                fold_all(dets, [readings[i] for i in idx], [r_scores[i] for i in idx], kw_, interval=a.interval, k=k,
                         context=a.context, tol=tol, alpha=1e-6, min_run=8, gate=a.gate, backend=backend)
                ps += [d.p_value for d in dets]
                segs += [d.segment_p_corrected for d in dets if d.segment_p_corrected is not None]
            report["attacks"][attack] = {"texts": len(ps), "detected_at": rates(ps),
                                         "median_p": sorted(ps)[len(ps) // 2] if ps else None,
                                         "segment_detected_at_1e-6": round(sum(x <= 1e-6 for x in segs) / max(1, len(ps)), 4)}
            log(f"  attack {attack}: {report['attacks'][attack]['detected_at']['0.01']}")
    report["minutes"] = round((time.time() - t0) / 60, 1)
    write_json(out_dir / f"detect_{a.name}.summary.json", report)
    log(f"written detect_{a.name}.summary.json")


if __name__ == "__main__":
    main()
