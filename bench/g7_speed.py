#!/usr/bin/env python3
"""G7 — speed. Run on a quiet machine: timings taken while other benchmarks run are not valid.

1. Verification time vs candidate length (one forward pass over BOS + text, plus the content layer):
   median of 5 runs after one warm-up, per configuration.
2. Sealing overhead: seal() at the reference operating point vs plain generation of the same length (no
   checkpoints, nucleus sampling): forcing, the verification-context cache when enabled, decoding with spans,
   canonicalization, the measurement pass and encryption, all included.

Usage: python3 bench/g7_speed.py --model distilgpt2 --configs cpu:float32,mps:float32,mps:float16
"""
from __future__ import annotations

import argparse
import json
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
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from entropy_protocol.verify import verify  # noqa: E402
from _meta import code_digest  # noqa: E402
from gates import PROMPTS, UNRELATED  # noqa: E402


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--configs", default="cpu:float32,mps:float32,mps:float16")
    ap.add_argument("--lengths", default="128,512,1000")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seal-tokens", type=int, default=160)
    a = ap.parse_args()
    master = crypto.new_master_key()
    report = {"model": a.model, "code": code_digest(), "verify": {}, "seal": {}}
    long_text = " ".join(UNRELATED * 40)
    for cfg in a.configs.split(","):
        dev, dt = cfg.split(":")
        b = HFBackend(a.model, device=dev, dtype=getattr(torch, dt))
        # the timed record must have checkpoints: with none, verification never runs the model (a small model can
        # fall into a blank-line loop and leave a record without checkpoints) and the timing would be meaningless
        for seed in range(1, 21):
            s = seal(b, PROMPTS[(seed - 1) % len(PROMPTS)], master, SealParams(max_new_tokens=a.seal_tokens), seed=seed)
            if s.entries >= SealParams().min_entries:
                break
        key = crypto.log_key(master, s.doc_id)
        ids = b.canonical_ids(long_text)
        rows = {}
        for L in (int(x) for x in a.lengths.split(",")):
            L = min(L, b.max_positions - 1, len(ids))
            cand = b.tok.decode(ids[:L])
            verify(b, cand, s.token, key)                      # warm-up
            ts = []
            for _ in range(a.repeats):
                sync(dev)
                t0 = time.perf_counter()
                verify(b, cand, s.token, key)
                sync(dev)
                ts.append(time.perf_counter() - t0)
            med = statistics.median(ts)
            rows[L] = {"seconds": round(med, 4), "tokens_per_s": round(L / med, 1)}
        report["verify"][cfg] = {**rows, "record_entries": s.entries}
        # sealing overhead: plain generation vs seal() at the reference operating point, same length
        ref = SealParams(max_new_tokens=a.seal_tokens, min_new_tokens=a.seal_tokens)
        gen_t, seal_t, toks = [], [], []
        for r in range(3):
            sync(dev)
            t0 = time.perf_counter()
            b.generate(PROMPTS[r], lambda j, gen: None, a.seal_tokens, ref.temperature, ref.top_p, ref.k,
                       ref.high_window, seed=100 + r, min_new_tokens=a.seal_tokens)
            sync(dev)
            gen_t.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            ss = seal(b, PROMPTS[r], master, ref, seed=100 + r)
            sync(dev)
            seal_t.append(time.perf_counter() - t0)
            toks.append(ss.tokens)
        report["seal"][cfg] = {"align": ref.align, "free_after": ref.free_after, "tokens": toks,
                               "generate_s": round(statistics.median(gen_t), 3),
                               "seal_s": round(statistics.median(seal_t), 3),
                               "overhead_%": round(100 * (statistics.median(seal_t) / statistics.median(gen_t) - 1), 1)}
        del b
    out = ROOT / "bench" / "results" / f"g7_{a.model.replace('/', '_')}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
