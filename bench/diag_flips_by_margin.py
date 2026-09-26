#!/usr/bin/env python3
"""Phase B: how often does a verifier on other hardware or precision disagree with the sealer at a checkpoint, as a
function of the checkpoint's margin?

A record entry can only be misjudged where the verifier's signed margin has the opposite sign to the reference's
(k = 1: the token present is, or is not, the first choice). This counts those sign flips over every usable position of
the shared calibration corpus, for each dumped configuration against the reference dump, at a ladder of reference
margins - the record margin a sealer would require. It also reports the deviation quantiles, which show that the
maximum deviation (the old "3 x max |delta|" tolerance) is set by a handful of positions and keeps growing with the
corpus, while the flips at a practical record margin stay at or near zero.

Usage: python3 bench/diag_flips_by_margin.py <calib dir> --reference NVIDIA-L40S_float16_default_b1 [--corpus B_cal]
         [--out flips_by_margin.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

LADDER = (0.5, 0.84, 1.0, 2.0, 3.0, 5.0)


def load(path: Path):
    z = np.load(path)
    return z["margins"][:, 0].astype(np.float64), z["bounds"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("calib_dir")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--corpus", default="B_cal")
    ap.add_argument("--from-position", type=int, default=4, help="first token index that can hold a checkpoint")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = Path(a.calib_dir)
    ref, bounds = load(d / f"{a.corpus}__{a.reference}.npz")
    pos = np.concatenate([np.arange(bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)])
    use = pos >= a.from_position
    r = np.abs(ref[use])
    report = {"corpus": a.corpus, "reference": a.reference, "texts": int(len(bounds) - 1), "positions": int(use.sum()),
              "share_at_or_above": {f"{t:g}": round(float((r >= t).mean()), 4) for t in LADDER}, "configs": {}}
    print(f"{a.corpus}: {report['texts']} texts, {report['positions']} usable positions, reference {a.reference}")
    print("share of positions with |reference margin| >= t: "
          + "  ".join(f"{k}: {v}" for k, v in report["share_at_or_above"].items()))
    for p in sorted(d.glob(f"{a.corpus}__*.npz")):
        cfg = p.stem.split("__", 1)[1]
        if cfg == a.reference:
            continue
        m, b = load(p)
        if m.shape != ref.shape or not np.array_equal(b, bounds) or not np.isfinite(m).all():
            report["configs"][cfg] = {"skipped": "different positions or non-finite values"}
            continue
        diff = np.abs(m - ref)[use]
        flip = ((m >= 0) != (ref >= 0))[use]
        c = {"max_delta": round(float(diff.max()), 3), "p99.99_delta": round(float(np.quantile(diff, 0.9999)), 3),
             "p99.9_delta": round(float(np.quantile(diff, 0.999)), 3),
             "flips_at_or_above": {f"{t:g}": [int((flip & (r >= t)).sum()), int((r >= t).sum())] for t in LADDER}}
        report["configs"][cfg] = c
        print(f"{cfg:<36} max {c['max_delta']:7.3f}  p99.99 {c['p99.99_delta']:6.3f}  p99.9 {c['p99.9_delta']:6.3f} | "
              + "  ".join(f"{t}: {f}/{n}" for t, (f, n) in c["flips_at_or_above"].items()))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"written {a.out}")


if __name__ == "__main__":
    main()
