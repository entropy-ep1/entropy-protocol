"""Two questions the phase A table left open, answered on the dumps already downloaded (no GPU):
1. How fast does the certified tolerance grow with the calibration corpus? (max over positions is a heavy-tailed
   statistic: on the Mac the same model gave 0.35 on one corpus and 1.20 on another.) Subsamples of texts, so
   phase B picks a corpus size where the max has stopped moving.
2. What if a verifier of an uncertified class drops the checkpoints inside its own tolerance instead of the record
   carrying the widest margin of all classes? Coverage (how many record points it can still judge) and soundness
   (any sign flip among those) decide whether fp32 can be served without thinning the record.
Usage: python3 bench/diag_calib_corpus_and_classes.py runs_gpu/a1 [--model Qwen/Qwen2.5-7B-Instruct] [--corpus A_cal]
"""
import argparse
import json
from pathlib import Path

import numpy as np

C = 3.0          # tolerance = C x max |delta| (calibrate.py)
FACTOR = 2.0     # record keeps points with |margin| >= FACTOR x widest certified tolerance (SealParams)
FIRST = 4        # first position that can hold a checkpoint (seal.first_checkpoint)


def load(p: Path):
    z = np.load(p)
    return z["margins"], z["bounds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase_dir")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--corpus", default="A_cal")
    ap.add_argument("--reference", default="")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    d = Path(a.phase_dir) / a.model.replace("/", "_") / "calib"
    ref_name = a.reference or next(p.stem.split("__", 1)[1] for p in sorted(d.glob(f"{a.corpus}__*_float16_default_b1.npz")))
    ref_m, bounds = load(d / f"{a.corpus}__{ref_name}.npz")
    n_texts = len(bounds) - 1
    pos = np.concatenate([np.arange(bounds[i + 1] - bounds[i]) for i in range(n_texts)])
    use = pos >= FIRST
    rows, per_text = {}, {}
    for p in sorted(d.glob(f"{a.corpus}__*.npz")):
        cfg = p.stem.split("__", 1)[1]
        if cfg == ref_name:
            continue
        m, _ = load(p)
        diff = np.abs(m - ref_m).max(1)                                  # worst of LOW / HIGH per position
        rows[cfg] = diff
        per_text[cfg] = np.array([diff[bounds[i]:bounds[i + 1]][use[bounds[i]:bounds[i + 1]]].max()
                                  for i in range(n_texts)])              # each text's worst position
    rng = np.random.default_rng(0)
    report = {"model": a.model, "reference": ref_name, "texts": n_texts, "positions_used": int(use.sum()),
              "growth": {}, "classes": {}}
    print(f"reference {ref_name}: {n_texts} texts, {int(use.sum())} usable positions\n")
    print("1) tolerance (3 x max |delta|) against corpus size — median [min, max] over subsamples")
    for cfg, per in per_text.items():
        line, g = [], {}
        for k in (4, 8, 16, 32, 48, 64):
            if k > n_texts:
                continue
            draws = np.array([per[rng.choice(n_texts, k, replace=False)].max() for _ in range(a.trials)])
            g[k] = {"median": round(float(C * np.median(draws)), 4), "min": round(float(C * draws.min()), 4),
                    "max": round(float(C * draws.max()), 4)}
            line.append(f"{k}:{g[k]['median']:.3f}[{g[k]['min']:.3f}-{g[k]['max']:.3f}]")
        report["growth"][cfg] = g
        full = C * per.max()
        print(f"  {cfg:42s} full corpus {full:.3f} | " + " ".join(line))
    print("\n2) a verifier outside the table, dropping points inside its own tolerance")
    fp16 = next(c for c in rows if "_float16_" in f"_{c}_")        # not bfloat16: its name contains float16
    need16 = FACTOR * C * per_text[fp16].max()                           # record margin with fp16 alone certified
    kept = (np.abs(ref_m[:, 0]) >= need16) & use
    print(f"  record margin with fp16 certified alone: {need16:.3f}; points kept: {int(kept.sum())} "
          f"of {int(use.sum())} ({100 * kept.sum() / use.sum():.1f}%)")
    for cfg, diff in rows.items():
        tol = C * per_text[cfg].max()
        own = np.abs(np.load(d / f"{a.corpus}__{cfg}.npz")["margins"][:, 0])
        judged = kept & (own >= tol)                                     # this verifier can decide these
        flips = judged & ((np.load(d / f"{a.corpus}__{cfg}.npz")["margins"][:, 0] >= 0) != (ref_m[:, 0] >= 0))
        report["classes"][cfg] = {"own_tolerance": round(float(tol), 4), "judged": int(judged.sum()),
                                  "judged_share_of_kept": round(float(judged.sum() / max(1, kept.sum())), 4),
                                  "sign_flips_among_judged": int(flips.sum())}
        print(f"  {cfg:42s} tol {tol:6.3f} | judges {judged.sum():5d} of {int(kept.sum())} kept "
              f"({100 * judged.sum() / max(1, kept.sum()):5.1f}%) | sign flips among judged: {int(flips.sum())}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
