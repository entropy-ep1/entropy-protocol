"""Does the KGW baseline find fewer texts as they get longer? (Phase A reported 81 / 75 / 34 % at p <= 1e-2 for
100 / 200 / 400 tokens - an artifact of counting p = 0.0 as a miss, see baselines.exact_rate; on CUDA this finds
81 / 97 / 100 %.) The detector needs no model weights - only the tokenizer and the vocabulary size - but it
must run on the device type the corpus was generated on (CUDA for the GPU phases; see --device). Reports, per length:
the z-score, how many tokens the detector scored (it ignores repeated bigrams), the green fraction, and how
repetitive the text is.
Usage: python3 bench/diag_kgw_length.py runs_gpu/a1 [--model Qwen/Qwen2.5-7B-Instruct] [--tokenizer Qwen/Qwen2.5-0.5B-Instruct]
         [--vocab 152064] [--corpus A_c4_kgw] [--device cuda]"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entropy_protocol.canon import canonicalize  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase_dir")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B-Instruct", help="same tokenizer across Qwen2.5 sizes")
    ap.add_argument("--vocab", type=int, default=0, help="vocabulary of the model that generated (0 = the config's "
                                                         "own; the greenlist depends on it, so a guess reads as noise)")
    ap.add_argument("--corpus", default="A_c4_kgw")
    ap.add_argument("--lengths", default="100,200,400")
    # the greenlist is torch.randperm under a torch.Generator on the detector's device, and CPU and CUDA generators
    # draw different permutations from the same seed: a CPU detector reads a CUDA-generated corpus as noise (green
    # fraction = gamma; 26.09.2026). Detect on the device type the corpus was generated on.
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from transformers import AutoConfig, AutoTokenizer, WatermarkDetector, WatermarkingConfig
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    cfg = AutoConfig.from_pretrained(a.tokenizer)
    if a.vocab:
        cfg.vocab_size = a.vocab
    det = WatermarkDetector(cfg, a.device, WatermarkingConfig(), ignore_repeated_ngrams=True)
    rows = [json.loads(x) for x in (Path(a.phase_dir) / a.model.replace("/", "_") / f"{a.corpus}.jsonl").read_text().splitlines() if x.strip()]
    ids = [tok(canonicalize(r["text"]), add_special_tokens=False)["input_ids"] for r in rows]
    print(f"{len(rows)} texts, vocabulary {cfg.vocab_size}, tokenizer {a.tokenizer}")
    report = {"model": a.model, "vocab": cfg.vocab_size, "tokenizer": a.tokenizer, "texts": len(rows), "lengths": {}}
    for L in (int(x) for x in a.lengths.split(",")):
        z, p, scored, green, uniq = [], [], [], [], []
        for x in ids:
            cut = x[:L]
            if len(cut) < 3:
                continue
            r = det(torch.tensor([cut], device=a.device), return_dict=True)
            z.append(float(r.z_score[0])); p.append(float(r.p_value[0]))
            scored.append(int(r.num_tokens_scored[0])); green.append(float(r.green_fraction[0]))
            grams = list(zip(cut, cut[1:]))
            uniq.append(len(set(grams)) / max(1, len(grams)))
        d = {"texts": len(z), "median_z": round(st.median(z), 2), "share_p_le_1e-2": round(sum(x <= 1e-2 for x in p) / len(p), 4),
             "share_p_le_1e-4": round(sum(x <= 1e-4 for x in p) / len(p), 4),
             "median_scored_tokens": st.median(scored), "median_green_fraction": round(st.median(green), 4),
             "median_unique_bigrams": round(st.median(uniq), 3)}
        report["lengths"][str(L)] = d
        print(f"  L={L:4d}: median z {d['median_z']:6.2f} | p<=1e-2 {100 * d['share_p_le_1e-2']:5.1f}% | "
              f"p<=1e-4 {100 * d['share_p_le_1e-4']:5.1f}% | scored {d['median_scored_tokens']:6.1f} | "
              f"green {d['median_green_fraction']:.3f} | unique bigrams {d['median_unique_bigrams']:.3f}")
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"written {a.out}")


if __name__ == "__main__":
    main()
