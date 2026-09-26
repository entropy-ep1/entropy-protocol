#!/usr/bin/env python3
"""Generate a corpus in one of four modes, batched, resumable:

  sealed   Entropy Protocol reference operating point (record + record-free watermark), batch.seal_batch
  plain    the same sampler without checkpoints (unsealed baseline; same-model null texts)
  kgw      Kirchenbauer et al. watermark, transformers WatermarkingConfig (gamma 0.25, delta 2, left hash)
  synthid  SynthID-Text, transformers SynthIDTextWatermarkingConfig (tournament depth 30, 5-gram context)

Output: runs/<model>/<name>.jsonl (one line per text) and <name>.meta.json (provenance, arguments).

Usage: python3 bench/gpu/seal_corpus.py --model Qwen/Qwen2.5-7B-Instruct --data c4 --field prompt --n 1000 \
         --max-new 1000 --min-new 1000 --mode sealed --name c4_sealed_L1000 [--chat] [--cal cal.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import unicodedata

import torch

from common import (DATA, RUNS, Store, add_backend_args, chunks, load_backend, log, program_keys, prompt_input,
                    provenance, read_jsonl, slug, write_json)
from entropy_protocol.batch import generate_batch, seal_batch
from entropy_protocol.seal import SealParams, table_covers

SYNTHID_KEYS = [int.from_bytes(hashlib.sha256(f"ep-gpu-eval/synthid/{i}".encode()).digest()[:4], "big") % 1000
                for i in range(30)]


def seal_params(a, cal: dict) -> SealParams:
    kw = dict(max_new_tokens=a.max_new, min_new_tokens=a.min_new, temperature=a.temperature, top_p=a.top_p)
    if cal:
        kw.update(tol_space=cal["space"], cal=cal["table"])
    for name in ("interval", "k", "high_window", "context", "free_after"):
        if getattr(a, name) is not None:
            kw[name] = getattr(a, name)
    if a.align:
        kw["align"] = a.align
    if a.selection:
        kw["selection"] = a.selection
    if a.gate:
        kw["gate"] = a.gate
    if a.guard:
        kw["guard"] = a.guard
    if a.canon_gen:
        kw["canon_gen"] = True
    if a.high_margin is not None:
        kw["high_margin"] = None if a.high_margin < 0 else a.high_margin
    params = SealParams(**kw)
    if cal and table_covers(cal, params):           # the table must measure every position a checkpoint can take
        raise SystemExit(table_covers(cal, params))
    return params


def _left_pad(backend, inputs: list) -> tuple[torch.Tensor, torch.Tensor]:
    enc = [x if isinstance(x, list) else backend.tok(x)["input_ids"] for x in inputs]
    L = max(len(x) for x in enc)
    ids = torch.full((len(enc), L), backend.pad_id, dtype=torch.long)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, x in enumerate(enc):
        ids[i, L - len(x):] = torch.tensor(x)
        mask[i, L - len(x):] = 1
    return ids.to(backend.device), mask.to(backend.device)


@torch.no_grad()
def hf_watermarked(backend, inputs: list, a, cfg, seed: int) -> list[list[int]]:
    """Baselines through transformers' own generate() with a watermarking configuration, sampler settings as for
    sealed and plain texts. transformers applies the watermark after temperature and top-p, so with top-p < 1 the
    KGW bias acts inside the nucleus only (the original KGW adds it before truncation): corpora with --top-p 1.0
    give the canonical setting."""
    ids, mask = _left_pad(backend, inputs)
    torch.manual_seed(seed)
    out = backend.model.generate(input_ids=ids, attention_mask=mask, do_sample=a.temperature > 0,
                                 temperature=a.temperature if a.temperature > 0 else None,
                                 top_p=a.top_p if a.temperature > 0 else None, top_k=0,
                                 max_new_tokens=a.max_new, min_new_tokens=a.min_new, watermarking_config=cfg,
                                 repetition_penalty=1.0,          # the model's own generation_config must not differ
                                 pad_token_id=backend.pad_id, eos_token_id=sorted(backend.stop_ids))
    res = []
    for row in out[:, ids.shape[1]:].tolist():
        cut = next((i for i, t in enumerate(row) if t in backend.stop_ids), len(row))
        res.append(row[:cut])
    return res


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--measure-dtype", default="", help="make the record's measurement pass in this precision")
    ap.add_argument("--data", required=True, help="data/<name>.jsonl")
    ap.add_argument("--field", default="prompt")
    ap.add_argument("--chat", action="store_true", help="wrap the field in the model's chat template")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--min-new", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--mode", default="sealed", choices=["sealed", "plain", "kgw", "synthid"])
    ap.add_argument("--keys", type=int, default=5, help="sealed: texts rotate over this many program keys (the "
                    "protocol asks for >= 5 keys; on small models text quality can depend on the key)")
    ap.add_argument("--cal", default="", help="calibration JSON (bench/gpu/calib_table.py) for the record")
    for name in ("interval", "k", "high_window", "context", "free_after"):
        ap.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    ap.add_argument("--high-margin", type=float, default=None, help="logits; negative = automatic (record strength)")
    ap.add_argument("--align", default="")
    ap.add_argument("--gate", type=float, default=0.0, help="confidence gate (verifier-context first-choice probability)")
    ap.add_argument("--guard", type=float, default=0.0, help="generation-side guard (generation-context probability)")
    ap.add_argument("--canon-gen", action="store_true", help="canonical generation (no characters v4 would change)")
    ap.add_argument("--selection", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()

    items = read_jsonl(DATA / f"{a.data}.jsonl")[a.offset:a.offset + a.n]
    backend = load_backend(a.model, a.device, a.dtype, a.attn)
    meter = load_backend(a.model, a.device, a.measure_dtype, a.attn) if a.measure_dtype and a.measure_dtype != a.dtype else None
    cal = json.loads(open(a.cal).read()) if a.cal else {}
    params = seal_params(a, cal) if a.mode == "sealed" else None
    out_dir = RUNS / slug(a.model)
    st = Store(out_dir / f"{a.name}.jsonl")
    write_json(out_dir / f"{a.name}.meta.json", {**provenance(backend), "args": vars(a),
                                                  "params": params.__dict__ if params else None,
                                                  "measure_env": meter.environment if meter else None})
    masters = program_keys(max(8, a.keys))

    def key_of(item_id: str) -> int:
        return int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:4], "big") % a.keys
    cfg = None
    if a.mode == "kgw":
        from transformers import WatermarkingConfig
        cfg = WatermarkingConfig()                      # gamma 0.25, delta 2.0, left hash over 1 token
    elif a.mode == "synthid":
        from transformers import SynthIDTextWatermarkingConfig
        cfg = SynthIDTextWatermarkingConfig(keys=SYNTHID_KEYS, ngram_len=5)
    todo = [it for it in items if it["id"] not in st.done]
    log(f"{a.model} {a.mode} {a.name}: {len(todo)} of {len(items)} to do, batch {a.batch}")
    t_all, n_tok = time.time(), 0
    for b_i, group in enumerate(chunks(todo, a.batch)):
        t0 = time.time()
        inputs = [prompt_input(backend, it, a.field, a.chat) for it in group]
        seed = a.seed * 1_000_003 + a.offset + b_i
        if a.mode == "sealed":
            keys = [key_of(it["id"]) for it in group]                    # mixed keys in one generation batch
            res = seal_batch(backend, inputs, [masters[k] for k in keys], params,
                             doc_ids=[f"{a.name}-{it['id']}" for it in group], seed=seed, meter=meter)
            rows = [{"id": it["id"], "mode": "sealed", "key": k, "doc_id": s.doc_id, "text": s.text, "token": s.token,
                     "tokens": s.tokens, "enforced": s.enforced, "entries": s.entries,
                     "dropped": [s.dropped_anchor, s.dropped_rank, s.dropped_margin]}
                    for it, k, s in zip(group, keys, res)]
        else:
            if a.mode == "plain":
                outs = [ids for ids, _ in generate_batch(backend, inputs, [lambda j, g: None] * len(group), a.max_new,
                                                          a.temperature, a.top_p, 1, 3, seed=seed,
                                                          min_new_tokens=a.min_new)]
            else:
                outs = hf_watermarked(backend, inputs, a, cfg, seed)
            rows = [{"id": it["id"], "mode": a.mode,
                     "text": unicodedata.normalize("NFC", backend.tok.decode(ids, clean_up_tokenization_spaces=False)),
                     "tokens": len(ids)} for it, ids in zip(group, outs)]
        dt = time.time() - t0
        for r in rows:
            r["seconds"] = round(dt / len(rows), 3)
            r["batch"] = len(rows)
            st.write(r)
            n_tok += r["tokens"]
        log(f"  batch {b_i + 1}: {len(rows)} texts, {sum(r['tokens'] for r in rows)} tokens, {dt:.1f}s "
            f"({sum(r['tokens'] for r in rows) / dt:.0f} tok/s)")
    log(f"done: {n_tok} tokens in {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
