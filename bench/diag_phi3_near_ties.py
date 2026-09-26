"""Are the two checkpoints Phi-3 missed in the alignment audit near ties? (13.09.2026)

bench/results/align_microsoft_phi-3-mini-4k-instruct.json (canonical generation, MPS fp16): 733 of 735 checkpoints
matched, one miss in text #4 and one in #6, each where the record also dropped one entry by rank, with no token
difference. The audit keeps 240 characters of each text, so the texts are generated again exactly as the audit made
them (same prompt, key, document id and seed) and checked to be the same (record counts, detection counts). Then the
detector reads them with a tolerance: a checkpoint whose margin is below it is excluded, not counted. If the misses
are near ties of half precision, a small tolerance removes them and nothing else.

Usage: PYTHONPATH=src python3 bench/diag_phi3_near_ties.py [--rows 4,6]
"""
import argparse
import json
import sys

sys.path.insert(0, "bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
    ap.add_argument("--rows", default="4,6")
    ap.add_argument("--audit", default="bench/results/align_microsoft_phi-3-mini-4k-instruct.json")
    a = ap.parse_args()
    import torch
    from align_audit import BASE, QUOTES_CHAT
    from entropy_protocol import crypto
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.detect import detect
    from entropy_protocol.seal import SealParams, decider, finish, margins, measure
    audit = {(r["i"], r["canon_gen"]): r for r in json.load(open(a.audit))["rows"]}
    b = HFBackend(a.model, device="mps", dtype=torch.float16)
    prompts = ["Write a paragraph that begins: " + p for p in BASE] + QUOTES_CHAT
    for i in (int(x) for x in a.rows.split(",")):
        prompt = b.tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False,
                                           add_generation_prompt=True)
        master, doc = bytes([i]) * 32, f"audit-{i}"
        params = SealParams(max_new_tokens=200, min_new_tokens=160, canon_gen=True)
        _, _, hm = margins(params)
        ids, enf = b.generate(prompt, decider(params, master, doc), params.max_new_tokens, params.temperature,
                              params.top_p, params.k, params.high_window, 100 + i, params.min_new_tokens, hm,
                              params.align, params.free_after, gate=params.gate, guard=params.guard, canon_gen=True)
        m = measure(b, ids, enf)
        s = finish(b, params, m, b.position_stats(m.ctext, m.anchors, params.k) if m.anchors else [], master, doc)
        ref = audit[(i, True)]
        same = (s.enforced, s.entries, [s.dropped_anchor, s.dropped_rank, s.dropped_margin, s.dropped_gate],
                m.raw[:240]) == (ref["enforced"], ref["entries"], ref["dropped"], ref["text"])
        print(f"#{i}: reproduced {'the audit text' if same else 'a DIFFERENT text'}: enforced {s.enforced}, "
              f"entries {s.entries}, dropped {[s.dropped_anchor, s.dropped_rank, s.dropped_margin, s.dropped_gate]}",
              flush=True)
        for tol in (0.0, 0.05, 0.1, 0.25, 0.5):
            d = detect(b, m.raw, crypto.wm_key(master), tol=tol)
            print(f"   tol {tol:4.2f} logits: checkpoints {d.checkpoints}, matched {d.matched}, excluded {d.excluded}, "
                  f"p {d.p_value:.1e}", flush=True)


if __name__ == "__main__":
    main()
