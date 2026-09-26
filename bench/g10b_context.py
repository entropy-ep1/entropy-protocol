#!/usr/bin/env python3
"""G10b — context length of the content-keyed selection vs robustness to word replacement (distilgpt2)."""
from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.backend_hf import HFBackend  # noqa: E402
from entropy_protocol.detect import detect  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from _meta import code_digest  # noqa: E402
from g10_watermark import human_paragraphs  # noqa: E402
from gates import PROMPTS  # noqa: E402


def main():
    model, device, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    b = HFBackend(model, device=device)
    rng = random.Random(3)
    human = human_paragraphs(40)
    out = {"model": model, "code": code_digest(), "n": n, "contexts": {}}

    def replace(t, frac):
        w = t.split(" ")
        pool = [x for x in w if x.isalpha()] or ["the"]
        for i in rng.sample(range(len(w)), max(1, int(frac * len(w)))):
            w[i] = rng.choice(pool)
        return " ".join(w)

    for h in (2, 3, 4):
        master = crypto.new_master_key()
        kw = crypto.wm_key(master)
        res = {"clean": [], "replace_10": [], "replace_25": [], "embed": []}
        for i in range(n):
            s = seal(b, PROMPTS[i % len(PROMPTS)], master,
                     SealParams(max_new_tokens=200, min_new_tokens=200, context=h, align="free"), seed=77_000 + i)
            det = lambda t: detect(b, t, kw, context=h).p_value  # noqa: E731
            res["clean"].append(det(s.text))
            res["replace_10"].append(det(replace(s.text, 0.10)))
            res["replace_25"].append(det(replace(s.text, 0.25)))
            res["embed"].append(det(human[i % 40] + "\n\n" + s.text + "\n\n" + human[(i + 1) % 40]))
        out["contexts"][h] = {k: {"detected_at_1e-2": round(sum(p <= 1e-2 for p in v) / n, 3),
                                  "detected_at_1e-4": round(sum(p <= 1e-4 for p in v) / n, 3),
                                  "median_p": statistics.median(v)} for k, v in res.items()}
        print(h, json.dumps(out["contexts"][h]), flush=True)
    (ROOT / "bench" / "results" / f"g10b_context_{model.replace('/', '_')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
