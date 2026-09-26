"""distilgpt2: do stage-2 seals degenerate into blank-line runs, and which part of the operating point causes it?"""
import json, statistics, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import torch
torch.set_num_threads(1)
from entropy_protocol import crypto
from entropy_protocol.backend_hf import HFBackend
from entropy_protocol.seal import SealParams, seal
from entropy_protocol.detect import detect
from gates import PROMPTS
b = HFBackend("distilgpt2")
M = bytes(range(32))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
variants = {"reference_content_free": {}, "content_prompt": {"align": "prompt"},
            "interval_prompt": {"selection": "interval", "align": "prompt"}, "plain": None}
out = {}
for name, kw in variants.items():
    rows = []
    for i in range(N):
        p = PROMPTS[i % len(PROMPTS)]
        if kw is None:
            ids, _ = b.generate(p, lambda j, g: None, 200, 0.8, 0.9, 1, 3, seed=10_000 + i)
            text, entries, enforced = b.tok.decode(ids, clean_up_tokenization_spaces=False), None, None
        else:
            s = seal(b, p, M, SealParams(**kw), doc_id=f"d{i}", seed=10_000 + i)
            text, entries, enforced = s.text, s.entries, s.enforced
        cut = int(len(text) * 0.75)
        d = detect(b, text, crypto.wm_key(M)) if kw and kw.get("selection", "content") == "content" else None
        rows.append({"tail_blank": text[cut:].strip() == "" and len(text) > 0, "nl_share": text.count("\n") / max(1, len(text)),
                     "entries": entries, "enforced": enforced, "chars": len(text),
                     "wm_checkpoints": d.checkpoints if d else None, "wm_p": d.p_value if d else None})
    s = {"texts": N, "tail_blank_%": round(100 * sum(r["tail_blank"] for r in rows) / N, 1),
         "mean_newline_share": round(statistics.mean(r["nl_share"] for r in rows), 3),
         "mean_chars": round(statistics.mean(r["chars"] for r in rows), 1)}
    if kw is not None:
        s.update({"mean_entries": round(statistics.mean(r["entries"] for r in rows), 1),
                  "below_8_entries_%": round(100 * sum(r["entries"] < 8 for r in rows) / N, 1)})
        if rows[0]["wm_checkpoints"] is not None:
            s.update({"mean_wm_checkpoints": round(statistics.mean(r["wm_checkpoints"] for r in rows), 1),
                      "wm_detected_at_1%": round(sum(r["wm_p"] <= 0.01 for r in rows) / N, 3)})
    out[name] = s
    print(name, s, flush=True)
json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "degen.json", "w"), indent=1)
