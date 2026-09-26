#!/usr/bin/env python3
"""Phase B, the practical question: do records sealed on one machine verify on the others?

Reads every runs_gpu/<machine>/<model>/verify_*.summary.json (the phase A records verified again on each phase B card)
and prints one Markdown table: card, corpus, verifier configuration, texts, VERIFIED on the unchanged text, false
MODIFIED, INDETERMINATE, and whether the verifier's class was in the record's calibration table. Plus the throughput
probe of each card. Numpy-free, standard library only.

Usage: python3 phase_b_verify_table.py [runs_gpu dir] [model slug] > table.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[3] / "runs_gpu"
SLUG = sys.argv[2] if len(sys.argv) > 2 else "Qwen_Qwen2.5-7B-Instruct"


def main() -> None:
    rows, speed = [], []
    for machine in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        d = machine / SLUG
        if not d.is_dir():
            continue
        for f in sorted(d.glob("verify_*.summary.json")):
            j = json.loads(f.read_text())
            s, env = j["summary"], j.get("env", {})
            rows.append((machine.name, env.get("hardware", "?"), s["corpus"], s["config"], s["texts"],
                         s.get("same_VERIFIED_%"), s.get("same_wrong_MODIFIED"), s.get("same_INDETERMINATE"),
                         ",".join(s.get("calibrated", [])), j.get("time", "")[:10]))
        for f in sorted(d.glob("speed*.json")):
            j = json.loads(f.read_text())
            speed.append((machine.name, j.get("env", {}).get("hardware", "?"), f.name, j))
    print("| Machine | Card | Corpus | Verifier | Texts | VERIFIED % | False MODIFIED | INDETERMINATE | In table | Date |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    false_mod = sum(r[6] or 0 for r in rows)
    texts = sum(r[4] or 0 for r in rows)
    print(f"\n{len(rows)} verifications, {texts} texts, false MODIFIED in total: {false_mod}")
    if speed:
        print("\n| Machine | Card | Batch | Generation tok/s plain | sealed | Sealing overhead % | Verification tok/s | Peak GB sealed |")
        print("|---|---|---|---|---|---|---|---|")
        for m, hw, _, j in speed:
            gen, ver = j.get("generation", {}), j.get("verify", {})
            for b in sorted(set(gen) | set(ver), key=int):
                g, v = gen.get(b, {}), ver.get(b, {})
                peak = g.get("peak_gb_sealed")
                print(f"| {m} | {hw} | {b} | {g.get('plain_tok_s', '')} | {g.get('sealed_tok_s', '')} | "
                      f"{g.get('overhead_%', '')} | {v.get('tok_s', '')} | {'' if peak is None else round(peak, 2)} |")


if __name__ == "__main__":
    main()
