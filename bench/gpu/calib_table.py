#!/usr/bin/env python3
"""Calibration across machines, step 2: tolerance table and the numerical-noise attack (class 11).

For each configuration dump (calib_dump.py) against the reference configuration:
  delta        max |margin_config - margin_reference| over the positions that can hold a checkpoint (index >=
               --from-position, default 4 = seal.first_checkpoint of the default parameters) and both predicates
               (spec [0117]; 15.09.2026: over all positions, one position-3 outlier made the fp16 class 14x wider)
  p99.9/p99.99 the same distribution's upper quantiles (how much the max is carried by rare positions)
  flips        positions where the predicate's truth differs from the reference (k = 1: the sign of the margin,
               the same for LOW and HIGH): at all positions, and at positions a record would keep
               (|margin_reference| >= seal_margin_factor x tolerance) -> must be 0
A class (fp32 / fp16 / bf16) takes the largest delta of its members; tolerance = c x delta.

Usage: python3 bench/gpu/calib_table.py --model M --corpus cal_sealed --reference NVIDIA-L40S_float16_default_b1 \
         [--certify fp32,fp16] [--c 3] [--out calibrations/<model>_gpu.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import RUNS, log, slug, write_json

CLASS = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def load(path: Path):
    z = np.load(path)
    return z["margins"], z["bounds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--reference", required=True, help="configuration name of the reference dump")
    ap.add_argument("--certify", default="fp32,fp16", help="classes the table certifies")
    ap.add_argument("--c", type=float, default=3.0)
    ap.add_argument("--factor", type=float, default=2.0, help="seal margin factor (records keep >= factor x tol)")
    ap.add_argument("--from-position", type=int, default=4,
                    help="first token index that can hold a checkpoint (seal.first_checkpoint); earlier ones "
                         "are not measured")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = RUNS / slug(a.model) / "calib"
    ref_m, ref_b = load(d / f"{a.corpus}__{a.reference}.npz")
    pos = np.concatenate([np.arange(ref_b[i + 1] - ref_b[i]) for i in range(len(ref_b) - 1)]) if len(ref_b) > 1 \
        else np.zeros(0, dtype=int)
    use = pos >= a.from_position                     # rows that can carry a record entry or a detector count
    meta_ref = json.loads((d / f"{a.corpus}__{a.reference}.json").read_text())
    per_cfg, errors = {}, {}
    for p in sorted(d.glob(f"{a.corpus}__*.npz")):
        cfg = p.stem.split("__", 1)[1]
        if cfg == a.reference:
            continue
        m, b = load(p)
        if m.shape != ref_m.shape or not np.array_equal(b, ref_b):
            errors[cfg] = "different positions (tokenization or corpus differs)"
            continue
        if not np.isfinite(m).all():
            errors[cfg] = "non-finite values"
            continue
        whole = np.abs(m - ref_m)
        diff = whole[use]
        flips = (m[:, 0] >= 0) != (ref_m[:, 0] >= 0)
        per_cfg[cfg] = {"delta": float(diff.max()), "p99.9": float(np.quantile(diff, 0.999)),
                        "p99.99": float(np.quantile(diff, 0.9999)), "mean": float(diff.mean()),
                        "flips_all": int((flips & use).sum()), "positions": int(use.sum()),
                        "delta_incl_unused_positions": float(whole.max()),
                        "flips_unused_positions": int((flips & ~use).sum())}
    by_class: dict[str, float] = {}
    for cfg, s in per_cfg.items():
        cls = next((CLASS[k] for k in CLASS if f"_{k}_" in f"_{cfg}_"), "other")
        by_class[cls] = max(by_class.get(cls, 0.0), s["delta"])
    ref_cls = next(CLASS[k] for k in CLASS if f"_{k}_" in f"_{a.reference}_")
    by_class.setdefault(ref_cls, 0.0)
    certify = [c for c in a.certify.split(",") if c]
    table = {c: round(a.c * by_class[c], 6) for c in certify if c in by_class}
    need = a.factor * max(table.values()) if table else 0.0
    kept = (np.abs(ref_m[:, 0]) >= need) & use       # k = 1: the LOW and HIGH columns coincide
    for cfg, s in per_cfg.items():                  # class 11: flips a record could suffer
        m, _ = load(d / f"{a.corpus}__{cfg}.npz")
        s["flips_at_kept_positions"] = int((((m[:, 0] >= 0) != (ref_m[:, 0] >= 0)) & kept).sum())
        s["kept_positions"] = int(kept.sum())
    result = {"model": meta_ref["model"], "weights_digest": meta_ref["weights"], "space": "logit", "c": a.c, "k": 1,
              "reference": a.reference, "reference_env": meta_ref["env"], "texts": meta_ref["texts"],
              "from_position": a.from_position, "positions": int(use.sum()),
              "positions_incl_unused": int(ref_m.shape[0]),
              "delta": {c: s["delta"] for c, s in per_cfg.items()},
              "per_config": per_cfg, "by_class": by_class, "errors": errors, "table": table,
              "record_margin_needed": need, "kept_share": float(kept.sum() / use.sum()) if use.any() else 0.0}
    out = Path(a.out) if a.out else RUNS / slug(a.model) / f"calibration_{a.corpus}.json"
    write_json(out, result)
    log(f"table {table}; kept share {result['kept_share']:.3f}; "
        f"flips at kept positions: {sum(s['flips_at_kept_positions'] for s in per_cfg.values())} -> {out}")


if __name__ == "__main__":
    main()
