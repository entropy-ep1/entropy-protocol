#!/usr/bin/env python3
"""G4b — in which space should the tolerance live, and which configurations can share one?

Candidate checkpoints = every checkpoint that holds on the reference (CPU float32) at tol = 0.
For each checkpoint the signed predicate margin is measured in two spaces:
  prob  : p_tok - p_boundary                      (the probability space)
  logit : l_tok - l_boundary  (= log p_tok - log p_boundary, a log-probability ratio)
Texts are split: the first half calibrates delta (max |margin_cfg - margin_ref| over checkpoints),
the second half tests tol = c * delta:
  survival    = share of test checkpoints with reference margin >= 2*tol (what the seal would keep)
  unabsorbed  = kept checkpoints whose outcome still flips on the configuration (must be 0)
  indet       = kept checkpoints whose configuration margin falls below tol (reported INDETERMINATE)
Certified sets: tol = c * max delta over the set, evaluated on every member.

Usage: python3 bench/g4b_margins.py --model distilgpt2 --n 60
"""
from __future__ import annotations

import argparse
import gc
import json
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
from entropy_protocol.predicates import boundary, holds  # noqa: E402
from entropy_protocol.record import loads  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from gates import PROMPTS  # noqa: E402

CONFIGS = ["mps:float32", "mps:float16", "mps:bfloat16", "cpu:bfloat16"]
SETS = {"fp32": ["mps:float32"], "fp32+fp16": ["mps:float32", "mps:float16"],
        "all": ["mps:float32", "mps:float16", "mps:bfloat16", "cpu:bfloat16"],
        "bf16_only": ["mps:bfloat16", "cpu:bfloat16"]}
K = 10


@torch.no_grad()
def margins(backend: HFBackend, ids: list[int], points: list[tuple[int, int]]):
    inp = torch.tensor([[backend.start_id] + ids], device=backend.device)
    logits = backend.model(inp).logits[0, :-1].float().cpu()
    out = []
    for t, prop in points:
        row = logits[t]
        tok = ids[t]
        rank0 = int((row > row[tok]).sum())
        top_l = torch.topk(row, K + 1).values.tolist()
        p = torch.softmax(row, 0)
        top_p = torch.topk(p, K + 1).values.tolist()
        out.append((holds(prop, rank0, K),
                    float(p[tok]) - boundary(prop, rank0, top_p, K),
                    float(row[tok]) - boundary(prop, rank0, top_l, K),
                    prop))
    return out


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--c", type=float, default=3.0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    a = ap.parse_args()
    t0 = time.time()
    ref = HFBackend(a.model)
    master = crypto.new_master_key()
    texts = []
    for i in range(a.n):
        s = seal(ref, PROMPTS[(i + 5) % len(PROMPTS)], master, SealParams(max_new_tokens=a.max_new_tokens),
                 seed=80_000 + i)
        rec = loads(crypto.decrypt_token(crypto.log_key(master, s.doc_id), s.token))
        ctext = canonicalize(s.text)
        enc = ref.tok(ctext, return_offsets_mapping=True, add_special_tokens=False)
        cover = {}
        for t, (x, y) in enumerate(enc["offset_mapping"]):
            for c in range(x, y):
                cover.setdefault(c, t)
        texts.append((enc["input_ids"], [(cover[c], p) for c, p in rec.entries if c in cover]))
    half = a.n // 2
    ref_m = [margins(ref, ids, pts) for ids, pts in texts]
    cfg_m = {}
    for cfg in CONFIGS:
        dev, dt = cfg.split(":")
        b = HFBackend(a.model, device=dev, dtype=getattr(torch, dt))
        cfg_m[cfg] = [margins(b, ids, pts) for ids, pts in texts]
        del b
        gc.collect()

    def deltas(cfg, space, lo, hi):
        j = 1 if space == "prob" else 2
        return [abs(m[j] - r[j]) for rm, mm in zip(ref_m[lo:hi], cfg_m[cfg][lo:hi]) for r, m in zip(rm, mm)]

    report = {"model": a.model, "n_texts": a.n, "calib_texts": half, "test_texts": a.n - half, "c": a.c,
              "candidate_checkpoints_test": sum(len(m) for m in ref_m[half:]), "per_config": {}, "certified_sets": {}}
    for cfg in CONFIGS:
        row = {"flips_all_texts": sum(int(m[0] != r[0]) for rm, mm in zip(ref_m, cfg_m[cfg]) for r, m in zip(rm, mm))}
        for space in ("prob", "logit"):
            d = deltas(cfg, space, 0, half)
            row[space] = {"delta_max": round(max(d), 6), "delta_q999": round(quantile(d, 0.999), 6),
                          "delta_median": round(quantile(d, 0.5), 7)}
        report["per_config"][cfg] = row

    for set_name, members in SETS.items():
        entry = {}
        for space in ("prob", "logit"):
            j = 1 if space == "prob" else 2
            tol = a.c * max(max(deltas(cfg, space, 0, half)) for cfg in members)
            kept = [(ti, ci) for ti in range(half, a.n) for ci, r in enumerate(ref_m[ti]) if abs(r[j]) >= 2 * tol]
            total = sum(len(ref_m[ti]) for ti in range(half, a.n))
            by_prop = {0: [0, 0], 1: [0, 0]}
            for ti in range(half, a.n):
                for r in ref_m[ti]:
                    by_prop[r[3]][1] += 1
                    by_prop[r[3]][0] += int(abs(r[j]) >= 2 * tol)
            res = {"tol": round(tol, 6), "survival_%": round(100 * len(kept) / max(1, total), 1),
                   "survival_LOW_%": round(100 * by_prop[0][0] / max(1, by_prop[0][1]), 1),
                   "survival_HIGH_%": round(100 * by_prop[1][0] / max(1, by_prop[1][1]), 1),
                   "kept_per_text": round(len(kept) / max(1, a.n - half), 1), "members": {}}
            for cfg in CONFIGS:
                unabsorbed = sum(int(cfg_m[cfg][ti][ci][0] != ref_m[ti][ci][0]) for ti, ci in kept)
                indet = sum(int(abs(cfg_m[cfg][ti][ci][j]) < tol) for ti, ci in kept)
                res["members" if cfg in members else "outside"] = res.get("members" if cfg in members else "outside", {})
                res["members" if cfg in members else "outside"][cfg] = {"unabsorbed_flips": unabsorbed, "indeterminate": indet}
            entry[space] = res
        report["certified_sets"][set_name] = entry
    report["minutes"] = round((time.time() - t0) / 60, 1)
    out = ROOT / "bench" / "results" / f"g4b_{a.model.replace('/', '_')}_N{a.n}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
