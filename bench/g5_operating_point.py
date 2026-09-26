#!/usr/bin/env python3
"""G5b — operating point: quality price vs checkpoints kept, per sealing configuration.

For each prompt and seed: one unsealed generation and one sealed generation per variant (interval, k,
HIGH window, share of LOW checkpoints). Every variant generates at least --min-new-tokens tokens (end of text
unavailable before that), so lengths match. Quality = perplexity under an independent oracle over the common
prefix of all variants for that prompt; reported as the paired mean log-ratio to the unsealed text with a 95 %
bootstrap interval, plus distinct-2 and repeated-trigram rate on the common word prefix. Seal strength = checkpoints kept per text and per 100 tokens under the
given calibration table (the margin filter of seal.py), and the share of texts below min_entries.

Usage: python3 bench/g5_operating_point.py --model distilgpt2 --oracle gpt2-medium --n 40 \
         --space logit --cal fp32=0.1,fp16=0.5 [--device mps]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import torch  # noqa: E402

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.backend_hf import HFBackend  # noqa: E402
from entropy_protocol.canon import canonicalize  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from _meta import code_digest  # noqa: E402
from gates import PROMPTS  # noqa: E402

VARIANTS = {                     # name: SealParams overrides
    "i5_k10_w40": dict(interval=5, k=10, high_window=40, selection="interval"),
    "i5_k3_w5": dict(interval=5, k=3, high_window=5, selection="interval"),
    "i10_k10_w40": dict(interval=10, k=10, high_window=40, selection="interval"),
    "i10_k3_w5": dict(interval=10, k=3, high_window=5, selection="interval"),
    "i5_k10_w40_L75": dict(interval=5, k=10, high_window=40, low_share=0.75, selection="interval"),
    "i5_k3_w5_L75": dict(interval=5, k=3, high_window=5, low_share=0.75, selection="interval"),
    "i10_k10_w1": dict(interval=10, k=10, high_window=1, selection="interval"),   # claim 4: most probable HIGH token
    "i5_k1_w3": dict(interval=5, k=1, high_window=3, selection="interval"),       # HIGH = not the first choice
    "i10_k1_w3": dict(interval=10, k=1, high_window=3, selection="interval"),
    "c5_k1_w3_prompt": dict(interval=5, k=1, high_window=3, selection="content", align="prompt"),
    "c5_k1_w3_free": dict(interval=5, k=1, high_window=3, selection="content", align="free"),
    "c5_free_hm0": dict(interval=5, k=1, high_window=3, selection="content", align="free", high_margin=0.0),
    # "record strength" profile: the HIGH margin the seal's filter needs (high_margin=None, automatic); since the
    # final defaults (high_margin=0.0) "c5_k1_w3_free" is the same as "c5_free_hm0"
    "c5_record_strength": dict(interval=5, k=1, high_window=3, selection="content", align="free", high_margin=None),
    "c5_hybrid32": dict(interval=5, k=1, high_window=3, selection="content", align="free", free_after=32),
    "c5_hybrid32_hm0": dict(interval=5, k=1, high_window=3, selection="content", align="free", free_after=32,
                            high_margin=0.0),
}
MIN_PREFIX = 30


@torch.no_grad()
def oracle_ppl(oracle: HFBackend, ids: list[int]) -> float:
    p = oracle.probs_for_ids(ids)
    nll = -torch.log(p[torch.arange(len(ids)), torch.tensor(ids)].clamp_min(1e-12)).mean()
    return math.exp(float(nll))


def distinct2(words):
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / max(1, len(bigrams))


def rep3(words):
    tri = list(zip(words, words[1:], words[2:]))
    return 1 - len(set(tri)) / max(1, len(tri))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--oracle", default="gpt2-medium")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", help="float32 | float16 | bfloat16 (models too large for fp32 memory)")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--min-new-tokens", type=int, default=120)
    ap.add_argument("--space", default="logit")
    ap.add_argument("--cal", default="")
    ap.add_argument("--variants", default="i5_k1_w3,c5_k1_w3_prompt,c5_k1_w3_free")
    ap.add_argument("--self-ppl", action="store_true", help="also score each prefix with the generator itself")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cal = {c: float(v) for c, v in (x.split("=") for x in a.cal.split(",") if x)}
    names = a.variants.split(",")
    t0 = time.time()
    gen = HFBackend(a.model, device=a.device, dtype=getattr(torch, a.dtype))
    oracle = HFBackend(a.oracle, device=a.device, dtype=getattr(torch, a.dtype))
    master = crypto.new_master_key()
    rows = []
    for i in range(a.n):
        prompt = PROMPTS[i % len(PROMPTS)]
        seed = 90_000 + i
        ids, _ = gen.generate(prompt, lambda j, gen: None, a.max_new_tokens, 0.8, 0.9, 10, 40, seed=seed,
                              min_new_tokens=a.min_new_tokens)
        texts = {"unsealed": gen.tok.decode(ids, clean_up_tokenization_spaces=False)}
        row = {"i": i, "prompt": prompt, "seal": {}}
        for name in names:
            s = seal(gen, prompt, master, SealParams(max_new_tokens=a.max_new_tokens, min_new_tokens=a.min_new_tokens,
                                                     tol_space=a.space, cal=cal, **VARIANTS[name]), seed=seed)
            texts[name] = s.text
            row["seal"][name] = {"tokens": s.tokens, "enforced": s.enforced, "entries": s.entries,
                                 "dropped_margin": s.dropped_margin, "dropped_rank": s.dropped_rank}
        oids = {name: oracle.canonical_ids(canonicalize(t))[:1000] for name, t in texts.items()}
        gids = {name: gen.canonical_ids(canonicalize(t))[:1000] for name, t in texts.items()} if a.self_ppl else {}
        TG = min(len(v) for v in gids.values()) if gids else 0
        T = min(len(v) for v in oids.values())
        W = min(len(t.split()) for t in texts.values())
        row["prefix_tokens"] = T
        row["quality"] = {}
        for name, t in texts.items():
            words = t.split()[:W]
            row["quality"][name] = {"ppl_prefix": oracle_ppl(oracle, oids[name][:T]) if T >= MIN_PREFIX else None,
                                    "self_ppl_prefix": oracle_ppl(gen, gids[name][:TG]) if gids and TG >= MIN_PREFIX else None,
                                    "distinct2": distinct2(words), "rep3": rep3(words), "words": len(t.split()),
                                    "text": t[:500]}
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{a.n}  {time.time() - t0:.0f}s", flush=True)

    usable = [r for r in rows if r["prefix_tokens"] >= MIN_PREFIX]
    summary = {"model": a.model, "code": code_digest(), "variant_params": {n: VARIANTS[n] for n in names}, "oracle": a.oracle, "n": a.n, "usable_for_ppl": len(usable), "space": a.space,
               "min_new_tokens": a.min_new_tokens, "max_new_tokens": a.max_new_tokens,
               "cal": cal, "minutes": None, "variants": {}}
    base = {r["i"]: r["quality"]["unsealed"]["ppl_prefix"] for r in usable}
    summary["variants"]["unsealed"] = {
        "median_ppl_prefix": round(statistics.median(base.values()), 2),
        "mean_distinct2": round(statistics.mean(r["quality"]["unsealed"]["distinct2"] for r in rows), 3),
        "mean_rep3": round(statistics.mean(r["quality"]["unsealed"]["rep3"] for r in rows), 3),
        "mean_words": round(statistics.mean(r["quality"]["unsealed"]["words"] for r in rows), 1)}
    rng = random.Random(0)
    for name in names:
        ratios = [r["quality"][name]["ppl_prefix"] / base[r["i"]] for r in usable]
        logs = [math.log(x) for x in ratios]
        boots = sorted(statistics.mean(rng.choices(logs, k=len(logs))) for _ in range(2000))
        seals = [r["seal"][name] for r in rows]
        self_ratios = [r["quality"][name]["self_ppl_prefix"] / r["quality"]["unsealed"]["self_ppl_prefix"] for r in rows
                       if r["quality"][name].get("self_ppl_prefix") and r["quality"]["unsealed"].get("self_ppl_prefix")]
        self_geo = round(math.exp(statistics.mean(math.log(x) for x in self_ratios)), 3) if self_ratios else None
        summary["variants"][name] = {
            "self_geo_mean_ppl_ratio": self_geo, "self_pairs": len(self_ratios),
            "median_ppl_prefix": round(statistics.median(r["quality"][name]["ppl_prefix"] for r in usable), 2),
            "median_ppl_ratio": round(statistics.median(ratios), 3),
            "geo_mean_ppl_ratio": round(math.exp(statistics.mean(logs)), 3),
            "ci95": [round(math.exp(boots[50]), 3), round(math.exp(boots[1949]), 3)],
            "share_ratio_gt_1.25": round(sum(x > 1.25 for x in ratios) / len(ratios), 3),
            "mean_distinct2": round(statistics.mean(r["quality"][name]["distinct2"] for r in rows), 3),
            "mean_rep3": round(statistics.mean(r["quality"][name]["rep3"] for r in rows), 3),
            "mean_words": round(statistics.mean(r["quality"][name]["words"] for r in rows), 1),
            "mean_entries": round(statistics.mean(s["entries"] for s in seals), 1),
            "entries_per_100_tokens": round(100 * sum(s["entries"] for s in seals) / max(1, sum(s["tokens"] for s in seals)), 2),
            "survival_%": round(100 * sum(s["entries"] for s in seals) / max(1, sum(s["enforced"] for s in seals)), 1),
            "texts_below_8_entries_%": round(100 * sum(s["entries"] < 8 for s in seals) / len(seals), 1)}
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    tag = f"{a.model.replace('/', '_')}_{a.space}_min{a.min_new_tokens}_N{a.n}" + (f"_{a.tag}" if a.tag else "")
    out = ROOT / "bench" / "results" / f"g5b_{tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
