#!/usr/bin/env python3
"""Baselines on the same model, prompts, lengths and attacks: KGW and SynthID-Text detection.

  kgw      transformers WatermarkDetector (z-test over green tokens, repeated n-grams ignored)
  synthid  g-values of transformers' SynthID processor; two scores from the paper's non-learned detectors:
           mean (exact null: under H0 every counted g-value is Bernoulli(1/2), so the sum is Binomial) and
           weighted mean (depth weights 10..1; empirical null). The paper's learned Bayesian detector is not
           reproduced here (it needs training on the provider's data), so SynthID is if anything understated.

Thresholds for the TPR at a target FPR are taken from the empirical nulls (human texts, unwatermarked texts of
the same model), identically for all schemes; exact p-values are reported where the scheme has one.
Usage: python3 bench/gpu/baselines.py --model M --scheme kgw --positives c4_kgw_L400 --lengths 100,200,400 \
         --nulls data:c4:human,runs:c4_plain_L400:text --attacks attacks_c4_kgw_L400 --name main
"""
from __future__ import annotations

import argparse
import math

import numpy as np

import torch

from common import (DATA, RUNS, add_backend_args, chunks, load_backend, log, provenance, read_jsonl, slug,
                    write_json)
from entropy_protocol.canon import canonicalize
from entropy_protocol.detect import binomial_tail
from seal_corpus import SYNTHID_KEYS

FPRS = (1e-2, 1e-3, 1e-4)


def kgw_scores(backend, id_lists):
    from transformers import WatermarkDetector, WatermarkingConfig
    det = WatermarkDetector(backend.model.config, backend.device, WatermarkingConfig(), ignore_repeated_ngrams=True)
    out = []
    for ids in id_lists:
        if len(ids) < 3:
            out.append((0.0, 1.0))
            continue
        r = det(torch.tensor([ids], device=backend.device), return_dict=True)
        out.append((float(r.z_score[0]), float(r.p_value[0])))
    return out                                           # (score, p-value); larger score = more watermarked


def tail_half(n: int, m: int) -> float:
    """P(X >= m) for X ~ Binomial(n, 1/2): exact integers up to n = 1000 (binomial_tail), log space above, where
    SynthID's g-value counts live (n = tokens x depth, 30 000 for 1000 tokens: the big-integer sum took minutes per
    text on 15.09.2026). Double precision, the same distribution."""
    if n <= 1000 or m <= 0 or m > n:
        return binomial_tail(n, m)
    i = np.arange(m, n, dtype=np.float64)
    head = math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1)       # log C(n, m)
    logc = np.concatenate(([head], head + np.cumsum(np.log(n - i) - np.log(i + 1))))
    top = logc.max()
    return float(np.exp(top + np.log(np.exp(logc - top).sum()) - n * math.log(2)))


def repetition_mask(proc, ids: torch.Tensor) -> torch.Tensor:
    """proc.compute_context_repetition_mask without its Python loop over positions, same output: position i is
    repeated when the hash of its context equals one of the previous context_history_size context hashes, or is 0
    while the history still holds its initial zeros. (15.09.2026, phase A: the loop kept the SynthID baseline on one
    CPU core with the GPU idle; the scoring checks this function against the original on its first batch.)"""
    b = ids.shape[0]
    ctx = ids[:, :-1].unfold(1, proc.ngram_len - 1, 1)                  # [b, n, ngram_len - 1]
    n = ctx.shape[1]
    h = proc.accumulate_hash(torch.ones(b, n, dtype=torch.long, device=ids.device), ctx).cpu().numpy()
    idx, H = np.arange(n), proc.context_history_size
    rep = np.zeros((b, n), dtype=bool)
    for r in range(b):
        order = np.lexsort((idx, h[r]))                                  # equal hashes together, in text order
        hs, js = h[r][order], idx[order]
        same = hs[1:] == hs[:-1]
        prev = np.full(n, -1)
        prev[js[1:][same]] = js[:-1][same]                               # the latest earlier position, same hash
        rep[r] = ((prev >= 0) & (idx - prev <= H)) | ((h[r] == 0) & (idx < H))
    return torch.from_numpy(~rep).to(ids.device)


def synthid_scores(backend, id_lists, batch: int = 16):
    from transformers.generation.logits_process import SynthIDTextWatermarkLogitsProcessor
    proc = SynthIDTextWatermarkLogitsProcessor(ngram_len=5, keys=SYNTHID_KEYS, sampling_table_size=2 ** 16,
                                               sampling_table_seed=0, context_history_size=1024,
                                               device=backend.device)
    depth = len(SYNTHID_KEYS)
    w = torch.linspace(10, 1, depth, device=backend.device)
    out = []
    for part in chunks(id_lists, batch):
        L = max(len(x) for x in part)
        ids = torch.full((len(part), L), backend.pad_id, dtype=torch.long, device=backend.device)
        for i, x in enumerate(part):
            ids[i, :len(x)] = torch.tensor(x, device=backend.device)
        g = proc.compute_g_values(ids)                                  # [b, L - 4, depth]
        mask = repetition_mask(proc, ids)                               # [b, L - 4]
        if not out:                                                     # the original on a prefix of the first batch
            head = ids[:, :min(L, 160)]
            assert torch.equal(repetition_mask(proc, head), proc.compute_context_repetition_mask(head).bool())
        valid = torch.zeros_like(mask, dtype=torch.bool)
        for i, x in enumerate(part):
            valid[i, :max(0, len(x) - 4)] = True
        mask = mask.bool() & valid
        for i in range(len(part)):
            gi, mi = g[i][mask[i]], int(mask[i].sum())
            if mi == 0:
                out.append((0.0, 1.0, 0.0))
                continue
            s = int(gi.sum())
            n = mi * depth
            wmean = float((gi.float() * w).sum() / (w.sum() * mi))
            out.append((s / n, tail_half(n, s), wmean))
    return out                                           # (mean, exact p of mean, weighted mean)


def exact_rate(ps, f):
    """Share of p-values <= f, a missing p-value counting as 1. A p-value of exactly 0.0 is the strongest detection,
    not a missing one: transformers' KGW detector returns 0.0 above z ~ 7.5 (its normal tail underflows), and the
    earlier `(p or 1) <= f` counted those as misses - phase A's KGW 'fell' to 75 % and 34 % at 200 and 400 tokens
    where the detector found 97 % and 100 % (26.09.2026).
    >>> exact_rate([0.0, 0.5, None], 0.01)
    0.3333
    """
    return round(sum((1.0 if p is None else p) <= f for p in ps) / max(1, len(ps)), 4)


def tpr_at_fpr(pos, neg, fpr):
    """Threshold = the (1 - fpr) empirical quantile of the null scores (larger = watermarked)."""
    if not neg:
        return None
    neg = sorted(neg)
    k = min(len(neg) - 1, math.ceil((1 - fpr) * len(neg)))
    thr = neg[k]
    return {"tpr": round(sum(p > thr for p in pos) / max(1, len(pos)), 4), "threshold": thr,
            "null_n": len(neg), "resolvable": len(neg) * fpr >= 1}


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap, dtype="float16")
    ap.add_argument("--scheme", required=True, choices=["kgw", "synthid"])
    ap.add_argument("--positives", required=True)
    ap.add_argument("--lengths", default="100,200,400")
    ap.add_argument("--nulls", default="")
    ap.add_argument("--max-null-texts", type=int, default=10000)
    ap.add_argument("--attacks", default="")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    b = load_backend(a.model, a.device, a.dtype, a.attn)
    out_dir = RUNS / slug(a.model)

    def all_scores(texts, L=None):
        ids = [b.canonical_ids(canonicalize(t))[: (L or 100000)] for t in texts]
        if a.scheme == "kgw":
            return {"score": kgw_scores(b, ids)}
        s = synthid_scores(b, ids)
        return {"mean": [(x[0], x[1]) for x in s], "weighted": [(x[2], None) for x in s]}

    report = {**provenance(b), "scheme": a.scheme, "positives": {}, "nulls": {}, "attacks": {}}
    # the null is scored at the SAME length as the positives: a threshold from full-length human documents flatters
    # a score whose spread shrinks with length (15.09.2026, phase A: nulls were full documents, positives cut to L)
    null_texts = []
    for spec in (s for s in a.nulls.split(",") if s):
        kind, name, field = spec.split(":")
        rows = read_jsonl((DATA / f"{name}.jsonl") if kind == "data" else (out_dir / f"{name}.jsonl"))
        null_texts.append((spec, [r[field] for r in rows if r.get(field)][:a.max_null_texts]))
    pos = [r["text"] for r in read_jsonl(out_dir / f"{a.positives}.jsonl")]
    full_null: dict[str, list] = {}
    for L in (int(x) for x in a.lengths.split(",")):
        null_scores: dict[str, list] = {}
        for spec, texts in null_texts:
            for k, v in all_scores(texts, L).items():
                null_scores.setdefault(k, []).extend(x[0] for x in v)
                ps = [x[1] for x in v if x[1] is not None]
                report["nulls"][f"L{L}:{spec}:{k}"] = {
                    "texts": len(v), "cut_to_tokens": L,
                    "exact_fpr_at": {f"{f:g}": round(sum(p <= f for p in ps) / max(1, len(ps)), 5)
                                     for f in FPRS} if ps else None}
        sc = all_scores(pos, L)
        report["positives"][str(L)] = {k: {"texts": len(v), **{f"tpr_at_fpr_{f:g}": tpr_at_fpr([x[0] for x in v],
                                                                                              null_scores.get(k, []), f)
                                                               for f in FPRS},
                                           "exact_tpr_at": {f"{f:g}": exact_rate([x[1] for x in v], f)
                                                            for f in FPRS} if v and v[0][1] is not None else None}
                                       for k, v in sc.items()}
        log(f"  {a.scheme} L={L}: " + str({k: x["tpr_at_fpr_0.01"] for k, x in report["positives"][str(L)].items()}))
        if not full_null:                                   # attacks keep their own length: nulls at full length
            for spec, texts in null_texts:
                for k, v in all_scores(texts).items():
                    full_null.setdefault(k, []).extend(x[0] for x in v)
                    ps = [x[1] for x in v if x[1] is not None]
                    report["nulls"][f"full:{spec}:{k}"] = {
                        "texts": len(v), "cut_to_tokens": None,
                        "exact_fpr_at": {f"{f:g}": round(sum(p <= f for p in ps) / max(1, len(ps)), 5)
                                         for f in FPRS} if ps else None}
    null_scores = full_null
    if a.attacks:
        rows = read_jsonl(out_dir / f"{a.attacks}.jsonl")
        for attack in sorted({r["attack"] for r in rows}):
            sc = all_scores([r["text"] for r in rows if r["attack"] == attack])
            report["attacks"][attack] = {k: {f"tpr_at_fpr_{f:g}": tpr_at_fpr([x[0] for x in v], null_scores.get(k, []), f)
                                             for f in FPRS} for k, v in sc.items()}
    write_json(out_dir / f"baseline_{a.scheme}_{a.name}.summary.json", report)
    log(f"written baseline_{a.scheme}_{a.name}.summary.json")


if __name__ == "__main__":
    main()
