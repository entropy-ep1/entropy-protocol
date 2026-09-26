"""Where does the calibration delta come from? Per-position |margin_cfg - margin_ref| on a fresh corpus built the way
`entropy calibrate` builds it (sealed generations of CALIBRATION_PROMPTS, throwaway key), reported over all
positions and from the first checkpoint on (15.09.2026: on the L40S the fp16 class was carried by one position-3
outlier; this checks the Mac bridge tables, which were measured over all positions).
Backends are loaded one at a time (one heavy job on the Mac).
Usage: python3 bench/diag_calib_positions.py --model Qwen/Qwen2.5-1.5B-Instruct [--gen mps:float32]
         [--reference cpu:float32] [--configs mps:float16,mps:bfloat16] [--positions 3200]"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.backend_hf import HFBackend  # noqa: E402
from entropy_protocol.calibrate import margin_matrix  # noqa: E402
from entropy_protocol.canon import canonicalize  # noqa: E402
from entropy_protocol.cli import CALIBRATION_PROMPTS  # noqa: E402
from entropy_protocol.seal import SealParams, first_checkpoint, seal  # noqa: E402

DT = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def load(model: str, spec: str) -> HFBackend:
    dev, dt = spec.split(":")
    return HFBackend(model, device=dev, dtype=DT[dt])


def free(b) -> None:
    del b
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen", default="mps:float32")
    ap.add_argument("--reference", default="cpu:float32")
    ap.add_argument("--configs", default="mps:float16")
    ap.add_argument("--positions", type=int, default=3200)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    k, first = SealParams().k, first_checkpoint(SealParams())
    t0 = time.time()
    g = load(a.model, a.gen)
    corpus, n, throwaway = [], 0, crypto.new_master_key()
    while sum(map(len, corpus)) < a.positions and n < 50 * len(CALIBRATION_PROMPTS):
        s = seal(g, CALIBRATION_PROMPTS[n % len(CALIBRATION_PROMPTS)], throwaway, SealParams(k=k, max_new_tokens=200),
                 seed=1_000 + n)
        n += 1
        text = canonicalize(s.text)
        if len(text) >= 40:
            corpus.append(g.canonical_ids(text))
    free(g)
    print(f"corpus: {len(corpus)} texts, {sum(map(len, corpus))} positions ({time.time() - t0:.0f}s)", flush=True)
    r = load(a.model, a.reference)
    ref = [margin_matrix(r, ids, k, "logit") for ids in corpus]
    free(r)
    report = {"model": a.model, "reference": a.reference, "gen": a.gen, "first_checkpoint": first,
              "texts": len(corpus), "positions": sum(map(len, corpus)), "configs": {}}
    for spec in (x for x in a.configs.split(",") if x):
        b = load(a.model, spec)
        rows = []                                           # (|delta|, position in text, text index)
        for ti, (ids, rm) in enumerate(zip(corpus, ref)):
            d = (margin_matrix(b, ids, k, "logit") - rm).abs().max(1).values
            rows += [(float(v), p, ti) for p, v in enumerate(d.tolist())]
        free(b)
        allv = torch.tensor([x[0] for x in rows])
        used = torch.tensor([x[0] for x in rows if x[1] >= first])
        top = sorted(rows, reverse=True)[:8]
        report["configs"][spec] = {"delta_all_positions": round(float(allv.max()), 4),
                                   "delta_from_first_checkpoint": round(float(used.max()), 4),
                                   "p99.99_from_first_checkpoint": round(float(torch.quantile(used, 0.9999)), 4),
                                   "top_outliers_(delta,position,text)": [(round(v, 3), p, t) for v, p, t in top]}
        print(spec, json.dumps(report["configs"][spec]), flush=True)
    report["seconds"] = round(time.time() - t0)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
