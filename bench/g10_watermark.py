#!/usr/bin/env python3
"""G10 — record-free detection (watermark mode): power by length, empirical false positives, robustness.

Positives: texts sealed under key K (content selection) at fixed lengths. Nulls, each scored with K's detection
key: (a) human prose (paragraphs of the Python standard library's docstrings, written before LLMs), (b) the same
model's plain generations (no key), (c) the same model sealed under another key. Attacks on 200-token positives:
10% of words replaced, one sentence deleted, embedding into human text (25% sealed), truncation to half.
Reported: detection rate at alpha = 1e-2 / 1e-3 / 1e-4 / 1e-6 per bucket, empirical FPR on every null set,
checkpoints n and match rate m/n. Under the null the p-value is exact (Binomial(n, 1/2)); the empirical FPR
checks the implementation, not the statistics.

Usage: python3 bench/g10_watermark.py --model distilgpt2 --n 40 [--align prompt|free] [--tag x]
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
import sys
import sysconfig
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import torch  # noqa: E402

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.backend_hf import HFBackend  # noqa: E402
from entropy_protocol.detect import detect  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from _meta import code_digest  # noqa: E402
from gates import PROMPTS  # noqa: E402

ALPHAS = (1e-2, 1e-3, 1e-4, 1e-6)


def human_paragraphs(limit: int, min_words: int = 40) -> list[str]:
    """Prose paragraphs from docstrings of the Python standard library (human-written)."""
    out, seen = [], set()
    for path in sorted(Path(sysconfig.get_paths()["stdlib"]).glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if not doc:
                    continue
                for para in re.split(r"\n\s*\n", doc):
                    text = " ".join(para.split())
                    words = text.split()
                    prose = sum(w.strip(".,;:!?'\"()").isalpha() for w in words) / max(1, len(words))
                    if (len(words) >= min_words and prose >= 0.85 and text not in seen
                            and not re.search(r">>>|::|[=#{}\[\]]|\w\(|_\w", text)):
                        seen.add(text)
                        out.append(text)
    random.Random(0).shuffle(out)
    return out[:limit]


def rates(ps: list[float]) -> dict:
    return {f"{a:g}": round(sum(p <= a for p in ps) / max(1, len(ps)), 4) for a in ALPHAS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--lengths", default="50,100,200,400")
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32", help="float32 | float16 | bfloat16 (models too large for fp32 memory)")
    ap.add_argument("--align", default=None, help="enforcement context: prompt | free (default: SealParams)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    b = HFBackend(a.model, device=a.device, dtype=getattr(torch, a.dtype))
    master, other = crypto.new_master_key(), crypto.new_master_key()
    kw = crypto.wm_key(master)
    base = SealParams() if a.align is None else SealParams(align=a.align)
    report = {"model": a.model, "code": code_digest(), "align": base.align, "interval": base.interval, "k": base.k,
              "context": base.context, "alphas": list(ALPHAS), "positives": {}, "nulls": {}, "attacks": {}}

    def det(text):
        return detect(b, text, kw, interval=base.interval, k=base.k, context=base.context)

    sealed_200 = []
    for L in (int(x) for x in a.lengths.split(",")):
        ps, ns, ms = [], [], []
        for i in range(a.n):
            params = SealParams(max_new_tokens=L, min_new_tokens=L, align=base.align)
            s = seal(b, PROMPTS[i % len(PROMPTS)], master, params, seed=30_000 + 97 * L + i)
            d = det(s.text)
            ps.append(d.p_value); ns.append(d.checkpoints); ms.append(d.matched)
            if L == 200:
                sealed_200.append(s.text)
        report["positives"][str(L)] = {"n_texts": a.n, "detected_at": rates(ps),
                                        "median_p": statistics.median(ps),
                                        "mean_checkpoints": round(statistics.mean(ns), 1),
                                        "match_rate": round(sum(ms) / max(1, sum(ns)), 3)}
        print(f"  L={L}: {report['positives'][str(L)]}  {time.time() - t0:.0f}s", flush=True)

    human = human_paragraphs(a.nulls)
    null_sets = {"human_docstrings": human}
    plain = []
    for i in range(min(a.nulls, 120)):
        ids, _ = b.generate(PROMPTS[i % len(PROMPTS)], lambda j, gen: None, 200, 0.8, 0.9, 1, 3,
                            seed=40_000 + i, min_new_tokens=150)
        plain.append(b.tok.decode(ids, clean_up_tokenization_spaces=False))
    null_sets["same_model_no_key"] = plain
    other_key = []
    for i in range(min(a.nulls, 60)):
        s = seal(b, PROMPTS[i % len(PROMPTS)], other, SealParams(max_new_tokens=200, min_new_tokens=150,
                                                                align=base.align), seed=41_000 + i)
        other_key.append(s.text)
    null_sets["same_model_other_key"] = other_key
    for name, texts in null_sets.items():
        ds = [det(t) for t in texts]
        ps = [d.p_value for d in ds]
        report["nulls"][name] = {"n_texts": len(texts), "false_positive_at": rates(ps),
                                 "mean_checkpoints": round(statistics.mean(d.checkpoints for d in ds), 1),
                                 "match_rate": round(sum(d.matched for d in ds) / max(1, sum(d.checkpoints for d in ds)), 3),
                                 "min_p": min(ps)}
        print(f"  null {name}: {report['nulls'][name]}", flush=True)

    rng = random.Random(1)

    def replace_words(t, frac):
        w = t.split(" ")
        idx = rng.sample(range(len(w)), max(1, int(frac * len(w))))
        pool = [x for x in w if x.isalpha()] or ["the"]
        for i in idx:
            w[i] = rng.choice(pool)
        return " ".join(w)

    def delete_sentence(t):
        ends = [m.end() for m in re.finditer(r"[.!?](?=\s)", t)]
        if len(ends) < 3:
            return t
        a_, b_ = ends[len(ends) // 2 - 1] + 1, ends[len(ends) // 2] + 1
        return t[:a_] + t[b_:]

    def embed(t, h):
        return h + "\n\n" + t + "\n\n" + h

    attacks = {"replace_10pct_words": lambda t, h: replace_words(t, 0.10),
               "replace_25pct_words": lambda t, h: replace_words(t, 0.25),
               "delete_sentence": lambda t, h: delete_sentence(t),
               "embed_in_human_text": embed,
               "truncate_half": lambda t, h: t[: len(t) // 2]}
    for name, f in attacks.items():
        ps = [det(f(t, human[i % len(human)] + " " + human[(i + 1) % len(human)])).p_value
              for i, t in enumerate(sealed_200)]
        report["attacks"][name] = {"n_texts": len(ps), "detected_at": rates(ps), "median_p": statistics.median(ps)}
        print(f"  attack {name}: {report['attacks'][name]}", flush=True)
    report["minutes"] = round((time.time() - t0) / 60, 1)
    out = ROOT / "bench" / "results" / f"g10_{a.model.replace('/', '_')}_{base.align}{('_' + a.tag) if a.tag else ''}.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
