#!/usr/bin/env python3
"""Calibration across machines, step 1: this machine's margins on the calibration corpus.

Every configuration (GPU x precision x attention kernel x batch size) re-reads the same sealed calibration texts
and saves the signed LOW / HIGH margin at every position (logit space, k = 1). calib_table.py then compares the
files of all machines with the reference configuration and writes the tolerance table.

--batch 1 uses the single-text pass of the reference implementation (calibrate.margin_matrix); larger batches
use the batched pass (batch.score_batch), itself a configuration to calibrate.

Output: runs/<model>/calib/<corpus>__<gpu>_<dtype>_<attn>_b<batch>.npz
Usage: python3 bench/gpu/calib_dump.py --model M --corpus cal_sealed --dtype float16 --batch 32 [--attn eager]
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import torch

from common import RUNS, add_backend_args, chunks, load_backend, log, provenance, read_jsonl, slug, write_json
from entropy_protocol.batch import score_batch
from entropy_protocol.calibrate import margin_matrix
from entropy_protocol.canon import canonicalize


def margins_from_scores(sc, k: int) -> np.ndarray:
    top, rank0, v = sc.top, sc.rank0, sc.l_tok
    low_b = torch.where(rank0 == 0, top[:, 1], top[:, 0])
    high_b = torch.where(rank0 < k, top[:, k], top[:, k - 1])
    return torch.stack([v - low_b, v - high_b], 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap, dtype="float16")
    ap.add_argument("--corpus", required=True, help="sealed calibration corpus (runs/<model>/<corpus>.jsonl)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    b = load_backend(a.model, a.device, a.dtype, a.attn)
    rows = [r for r in read_jsonl(RUNS / slug(a.model) / f"{a.corpus}.jsonl") if r.get("text")]
    if not rows:     # 15.09.2026: with the corpus missing, four dumps "succeeded" on 0 texts and the table failed later
        raise SystemExit(f"no sealed texts in {a.corpus}: nothing to measure")
    ids =[b.canonical_ids(canonicalize(r["text"]))[: b.max_positions - 1] for r in rows]
    gpu = re.sub(r"[^A-Za-z0-9]+", "-", b.environment["hardware"]).strip("-")
    cfg = f"{gpu}_{a.dtype}_{a.attn or 'default'}_b{a.batch}" + (f"_{a.tag}" if a.tag else "")
    mats = []
    if a.batch == 1:
        for x in ids:
            mats.append(margin_matrix(b, x, a.k, "logit").numpy())
    else:
        for part in chunks(list(range(len(ids))), a.batch):
            for sc in score_batch(b, [ids[i] for i in part], a.k, token_budget=a.batch * 1100):
                mats.append(margins_from_scores(sc, a.k))
    m = np.concatenate(mats) if mats else np.zeros((0, 2))
    bounds = np.cumsum([0] + [len(x) for x in mats])
    out = RUNS / slug(a.model) / "calib" / f"{a.corpus}__{cfg}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, margins=m.astype(np.float32), bounds=bounds, finite=np.isfinite(m).all())
    write_json(out.with_suffix(".json"), {**provenance(b, batch=a.batch), "config": cfg, "texts": len(ids),
                                          "positions": int(len(m)), "finite": bool(np.isfinite(m).all())})
    log(f"{cfg}: {len(ids)} texts, {len(m)} positions -> {out.name}")


if __name__ == "__main__":
    main()
