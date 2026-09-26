#!/usr/bin/env python3
"""G7 on accelerators: sealing overhead against plain generation with the same sampler and batch, and the
throughput of the verification pass. Run alone on the machine.

  generation  tokens/s of plain generate_batch vs seal_batch (both contexts, measurement pass, records) at
              batch sizes 1 / 8 / 32, fixed length (end of text disabled)
  verify      tokens/s of the verification pass (score_batch) at batch 1 and 32 on 1000-token texts
  memory      peak allocated GPU memory per case

Usage: python3 bench/gpu/speed.py --model M --dtype bfloat16 [--batches 1,8,32] [--tokens 256]
"""
from __future__ import annotations

import argparse
import re
import time

import torch

from common import RUNS, add_backend_args, load_backend, log, program_keys, provenance, slug, write_json
from entropy_protocol.batch import generate_batch, score_batch, seal_batch
from entropy_protocol.seal import SealParams
from gates import PROMPTS, UNRELATED


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def timed(dev, fn):
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    sync(dev)
    t0 = time.time()
    r = fn()
    sync(dev)
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.startswith("cuda") else None
    return r, time.time() - t0, mem


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--verify-tokens", type=int, default=1000)
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    b = load_backend(a.model, a.device, a.dtype, a.attn)
    dev = str(b.device)
    master = program_keys()[0]
    rep = {**provenance(b), "generation": {}, "verify": {}}
    params = SealParams(max_new_tokens=a.tokens, min_new_tokens=a.tokens)
    seal_batch(b, PROMPTS[:2], master, SealParams(max_new_tokens=16, min_new_tokens=16), seed=0)   # warm-up
    for B in (int(x) for x in a.batches.split(",")):
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(B)]
        plain, sealed = [], []
        for r in range(a.repeats):
            _, tp, mp = timed(dev, lambda: generate_batch(b, prompts, [lambda j, g: None] * B, a.tokens, 0.8, 0.9, 1, 3,
                                                          seed=r, min_new_tokens=a.tokens))
            _, ts, ms = timed(dev, lambda: seal_batch(b, prompts, master, params, seed=r))
            plain.append(tp); sealed.append(ts)
        tp, ts = min(plain), min(sealed)
        rep["generation"][str(B)] = {"plain_tok_s": round(B * a.tokens / tp, 1), "sealed_tok_s": round(B * a.tokens / ts, 1),
                                     "overhead_%": round(100 * (ts / tp - 1), 1), "peak_gb_plain": mp, "peak_gb_sealed": ms}
        log(f"  batch {B}: {rep['generation'][str(B)]}")
    ids = b.canonical_ids(" ".join(UNRELATED * 60))[: a.verify_tokens]
    for B in (1, 32):
        (_, t, m) = timed(dev, lambda: score_batch(b, [ids] * B, 1, token_budget=B * (a.verify_tokens + 1)))
        rep["verify"][str(B)] = {"tok_s": round(B * len(ids) / t, 1), "seconds_per_text": round(t / B, 4), "peak_gb": m}
        log(f"  verify batch {B}: {rep['verify'][str(B)]}")
    hw = re.sub(r"[^A-Za-z0-9]+", "-", b.environment["hardware"]).strip("-")
    write_json(RUNS / slug(a.model) / f"speed_{b.config_class}_{hw}.json", rep)


if __name__ == "__main__":
    main()
