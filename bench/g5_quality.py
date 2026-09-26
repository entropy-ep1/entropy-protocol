#!/usr/bin/env python3
"""G5 — the price of the seal for text quality.

Same prompts and seeds, three generations each: unsealed (plain nucleus sampling), sealed with checkpoints
every 5 tokens, sealed every 10 tokens. Quality proxies: perplexity under an independent oracle model
(lower = more fluent), distinct-2 (diversity) and repeated-trigram rate. Samples are saved for human reading.

Usage: python3 bench/g5_quality.py --model distilgpt2 --oracle gpt2-medium --n 60 [--device mps]
"""
from __future__ import annotations

import argparse
import json
import math
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
from entropy_protocol.record import HIGH, LOW  # noqa: E402
from gates import PROMPTS  # noqa: E402


@torch.no_grad()
def oracle_ppl(oracle: HFBackend, text: str) -> float:
    ids = oracle.canonical_ids(canonicalize(text))[:1000]
    if len(ids) < 5:
        return float("nan")
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
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=160)
    a = ap.parse_args()
    t0 = time.time()
    gen = HFBackend(a.model, device=a.device)
    oracle = HFBackend(a.oracle, device=a.device)
    kb = crypto.bits_key(crypto.new_master_key())
    variants = {"unsealed": None, "sealed_every_5": 5, "sealed_every_10": 10}
    rows = []
    for i in range(a.n):
        doc = crypto.new_doc_id()
        row = {"i": i, "prompt": PROMPTS[i % len(PROMPTS)]}
        for name, interval in variants.items():
            def decide(j, gen=None, interval=interval):
                if interval is None or j == 0 or j % interval:
                    return None
                return LOW if crypto.bit(kb, doc, j) == 0 else HIGH
            ids, _ = gen.generate(row["prompt"], decide, a.max_new_tokens, 0.8, 0.9, 10, 40, seed=70_000 + i)
            text = gen.tok.decode(ids, clean_up_tokenization_spaces=False)
            words = text.split()
            row[name] = {"ppl": oracle_ppl(oracle, text), "distinct2": distinct2(words), "rep3": rep3(words),
                         "words": len(words), "text": text[:600]}
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{a.n}  {time.time() - t0:.0f}s", flush=True)
    summary = {"model": a.model, "oracle": a.oracle, "n": a.n, "minutes": round((time.time() - t0) / 60, 1)}
    for name in variants:
        ppl = [r[name]["ppl"] for r in rows if not math.isnan(r[name]["ppl"])]
        summary[name] = {"median_ppl": round(statistics.median(ppl), 2), "mean_distinct2": round(statistics.mean(r[name]["distinct2"] for r in rows), 3),
                         "mean_rep3": round(statistics.mean(r[name]["rep3"] for r in rows), 3),
                         "mean_words": round(statistics.mean(r[name]["words"] for r in rows), 1)}
    for name in ("sealed_every_5", "sealed_every_10"):
        ratios = [r[name]["ppl"] / r["unsealed"]["ppl"] for r in rows
                  if not math.isnan(r[name]["ppl"]) and not math.isnan(r["unsealed"]["ppl"])]
        summary[name]["median_ppl_ratio_vs_unsealed"] = round(statistics.median(ratios), 3)
    out = ROOT / "bench" / "results" / f"g5_{a.model.replace('/', '_')}_N{a.n}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
