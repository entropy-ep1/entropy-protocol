#!/usr/bin/env python3
"""G4 — verification across inference configurations (spec [0117]), with a per-class calibration table.

1. Calibrate (entropy_protocol.calibrate): on a corpus of reference generations, every configuration
   (device x precision) re-reads each text; delta = max |margin_cfg - margin_ref| over all positions and both
   predicates; class tolerance = c * max delta of its members. Reference = CPU float32.
2. Seal N texts on the reference with the table of the certified classes (default fp32, fp16) and, for
   contrast, with tol = 0 (uncalibrated).
3. Verify unchanged, transported and edited texts on every configuration, including classes outside the table.
Gate: no wrong MODIFIED on unchanged/transported text and no VERIFIED edited text on any configuration;
unchanged text VERIFIED on every configuration of a certified class.

Usage: python3 bench/g4_backends.py --model distilgpt2 --n 40 --calib 30 [--space logit] [--certify fp32,fp16]
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
from entropy_protocol.calibrate import calibrate  # noqa: E402
from entropy_protocol.canon import canonicalize  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from entropy_protocol.verify import MODIFIED, VERIFIED, verify  # noqa: E402
from _meta import code_digest  # noqa: E402
from gates import PROMPTS, TRANSPORTS, edits  # noqa: E402

DEFAULT_CONFIGS = "cpu:float32,mps:float32,mps:float16,mps:bfloat16,cpu:bfloat16"


def load(model, dev, dt):
    return HFBackend(model, device=dev, dtype=getattr(torch, dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--calib", type=int, default=30)
    ap.add_argument("--c", type=float, default=3.0)
    ap.add_argument("--space", default="logit")
    ap.add_argument("--certify", default="fp32,fp16")
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--tag", default="", help="suffix for the result file name")
    a = ap.parse_args()
    t0 = time.time()
    configs = [tuple(x.split(":")) for x in a.configs.split(",")]
    ref_cfg = configs[0]
    ref = load(a.model, *ref_cfg)
    master = crypto.new_master_key()

    # ---- 1. calibration corpus (reference generations) -> per-class tolerance table
    corpus = []
    for i in range(a.calib):
        s = seal(ref, PROMPTS[(i + 17) % len(PROMPTS)], master, SealParams(max_new_tokens=a.max_new_tokens), seed=50_000 + i)
        ids = ref.canonical_ids(canonicalize(s.text))
        if ids:                                  # a text can end at its first token: nothing to calibrate on
            corpus.append(ids)
    cal = calibrate(ref, {f"{d}:{t}": (lambda d=d, t=t: load(a.model, d, t)) for d, t in configs[1:]},
                    corpus, k=SealParams().k, space=a.space, c=a.c)
    gc.collect()
    table = cal.table(a.certify.split(","))

    # ---- 2. seal with the table (calibrated) and with tol = 0 (uncalibrated contrast)
    sealed = {"calibrated": [], "uncalibrated": []}
    for i in range(a.n):
        for name, params in (("calibrated", SealParams(max_new_tokens=a.max_new_tokens, tol_space=a.space, cal=table)),
                             ("uncalibrated", SealParams(max_new_tokens=a.max_new_tokens))):
            sealed[name].append(seal(ref, PROMPTS[i % len(PROMPTS)], master, params, seed=60_000 + i))
    survival = {name: round(sum(s.entries for s in ss) / max(1, sum(s.enforced for s in ss)) * 100, 2) for name, ss in sealed.items()}
    entries = {name: round(sum(s.entries for s in ss) / len(ss), 1) for name, ss in sealed.items()}
    weak = {name: sum(s.entries < 8 for s in ss) for name, ss in sealed.items()}

    # ---- 3. verify on every configuration
    table_out, flags = {}, {}
    for dev, dt in configs:
        key_cfg = f"{dev}:{dt}"
        if key_cfg in cal.errors:
            continue
        b = ref if (dev, dt) == ref_cfg else load(a.model, dev, dt)
        for name, ss in sealed.items():
            counts = {"unchanged": {}, "transported": {}, "edited": {}, "no_edit_possible": 0}
            for s in ss:
                k = crypto.log_key(master, s.doc_id)
                # the first edit that changes the canonical text (a text of blank lines only has none: the
                # "truncation" of blank lines leaves it as it was, and VERIFIED would be the right verdict)
                base = canonicalize(s.text)
                edited = next((t for t, _ in edits(s.text).values() if canonicalize(t) != base), None)
                if edited is None:
                    counts["no_edit_possible"] += 1
                for label, txt in (("unchanged", s.text), ("transported", TRANSPORTS["combined"](s.text)), ("edited", edited)):
                    if txt is None:
                        continue
                    r = verify(b, txt, s.token, k)
                    counts[label][r.verdict] = counts[label].get(r.verdict, 0) + 1
                    if name == "calibrated":
                        flags[key_cfg] = r.calibrated
            table_out[f"{key_cfg}|{name}"] = counts
        if b is not ref:
            del b
            gc.collect()

    wrong = sum(c["unchanged"].get(MODIFIED, 0) + c["transported"].get(MODIFIED, 0) for c in table_out.values())
    missed = sum(c["edited"].get(VERIFIED, 0) for c in table_out.values())
    certified_ok = all(c["unchanged"].get(VERIFIED, 0) == a.n and c["transported"].get(VERIFIED, 0) == a.n
                       for key, c in table_out.items() if key.endswith("|calibrated") and flags.get(key.split("|")[0]))
    summary = {"model": a.model, "code": code_digest(), "n": a.n, "calib_texts": a.calib, "calib_positions": cal.positions, "space": a.space,
               "c": a.c, "reference": cal.reference, "delta": cal.delta, "calibration_errors": cal.errors,
               "certified": a.certify.split(","), "table": table, "calibrated_flag_by_config": flags,
               "survival_%": survival, "mean_entries": entries, "texts_below_8_entries": weak, "verdicts": table_out,
               "wrong_MODIFIED_on_unchanged_any_config": wrong, "edited_reported_VERIFIED_any_config": missed,
               "certified_configs_all_VERIFIED": certified_ok, "minutes": round((time.time() - t0) / 60, 1)}
    ref_point = SealParams()
    summary["operating_point"] = {"interval": ref_point.interval, "k": ref_point.k, "window": ref_point.high_window}
    out = ROOT / "bench" / "results" / f"g4v2_{a.model.replace('/', '_')}_{a.space}_N{a.n}{('_' + a.tag) if a.tag else ''}.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
