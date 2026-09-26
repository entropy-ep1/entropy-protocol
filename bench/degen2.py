"""Is the blank-line degeneration key-dependent (a LOW checkpoint on a context whose first choice repeats it)?"""
import json, os, statistics, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))
import torch
torch.set_num_threads(1)
from entropy_protocol import crypto
from entropy_protocol.backend_hf import HFBackend
from entropy_protocol.seal import SealParams, seal
from gates import PROMPTS
b = HFBackend("distilgpt2")
NL, NN = 198, 628                                    # GPT-2 tokens "\n" and "\n\n"
traps = {"nl4": (NL,) * 4, "nn4": (NN,) * 4, "nl_nn": (NL, NN, NL, NN)}
N = int(sys.argv[1]); K = int(sys.argv[2])
out = []
for kk in range(K):
    master = hashlib_key = __import__("hashlib").sha256(f"degen2/{kk}".encode()).digest()
    kw = crypto.wm_key(master)
    trap = {name: crypto.wm_decide(kw, ctx, 5) for name, ctx in traps.items()}
    rows = []
    for i in range(N):
        s = seal(b, PROMPTS[i % len(PROMPTS)], master, SealParams(), doc_id=f"d{i}", seed=10_000 + i)
        t = s.text
        rows.append({"tail_blank": len(t) > 0 and t[int(len(t) * 0.75):].strip() == "", "entries": s.entries})
    r = {"key": kk, "trap_decisions": trap, "tail_blank_%": round(100 * sum(x["tail_blank"] for x in rows) / N, 1),
         "mean_entries": round(statistics.mean(x["entries"] for x in rows), 1)}
    out.append(r)
    print(r, flush=True)
json.dump(out, open("degen2.json", "w"), indent=1)
