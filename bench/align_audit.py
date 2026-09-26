"""Alignment audit: do the sealed tokens and the verifier's re-read of the canonical text coincide?

For each prompt, seal with canonical generation off and on (same seed, same key) and record:
- residual canonical changes (canonical text != raw.strip()) by class: typographic, space before a line break, NFC,
  the replacement character
- a leading-whitespace first token (the canonical form strips it)
- token edit regions between the verifier's view of the generated ids and the tokenization of the canonical text
- the record: enforced, entries, dropped (anchor / rank / margin / gate)
- record-free detection: checkpoints, matched
- transport: the text through a word processor's autocorrect (smart quotes, dashes, ellipsis) and CRLF line ends:
  the verify verdict, and whether canonical form v3 would still have seen the same text

Usage: PYTHONPATH=src python3 bench/align_audit.py --model distilgpt2 --out bench/results/align_distilgpt2.json
       PYTHONPATH=src python3 bench/align_audit.py --model Qwen/Qwen2.5-0.5B-Instruct --chat --out ...
       python3 bench/align_audit.py --summary bench/results/align_distilgpt2.json
"""
import argparse
import difflib
import json
import re
import sys
import time
import unicodedata
from collections import Counter

LDQ, RDQ, LSQ, RSQ, MDASH, NDASH, ELL = (chr(c) for c in (0x201C, 0x201D, 0x2018, 0x2019, 0x2014, 0x2013, 0x2026))
TYPO = set(chr(c) for c in (0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB, 0x2018, 0x2019, 0x201A, 0x201B, 0x02BC,
                            0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2015, 0x2212, 0x2026, 0x00A0, 0x202F, 0x2007))
BASE = [
    "The city council met on Tuesday to discuss", "In the early morning, the research team",
    "A new study on sleep and memory suggests that", "The history of the printing press shows",
    "When the engineers opened the server room,", "Our quarterly report highlights three trends:",
    "The old lighthouse keeper wrote in his journal", "Scientists have long debated whether",
    "Dear customer, we are writing to inform you that", "The court found that the defendant",
    "In a statement released on Monday, the ministry", "Students returning to campus this fall will notice",
    "The startup announced that it had raised", "Farmers in the region are worried that",
]
QUOTES_RAW = [
    'The mayor told reporters: "', "She wrote in her diary that it wasn't", 'In the interview, the author explained: "We',
    "The report " + MDASH + " published on Monday " + MDASH + " says", "He paused... then said",
    "It" + RSQ + "s been a long year, the minister" + RSQ + "s spokesperson said",
]
QUOTES_CHAT = [
    "Write a short news paragraph that quotes the mayor directly, with quotation marks.",
    "Write two sentences of dialogue between a teacher and a student, with quotation marks.",
    "Explain in one paragraph why the printing press mattered. Use an em dash somewhere.",
    "Write a short product review that uses contractions like don't and it's.",
    "Summarize a fictional court ruling in one paragraph, quoting the judge.",
    "Write a brief letter to a customer apologizing for a delay... and promising a refund.",
]


def word_autocorrect(t: str) -> str:
    """What Word / Google Docs do to typed text: smart quotes, dashes, ellipsis (and Windows line endings)."""
    out = []
    for i, ch in enumerate(t):
        prev = t[i - 1] if i else " "
        if ch == '"':
            out.append(LDQ if (prev.isspace() or prev in "([{-") else RDQ)
        elif ch == "'":
            out.append(LSQ if (prev.isspace() or prev in "([{\"") else RSQ)
        else:
            out.append(ch)
    t = "".join(out)
    t = re.sub(r"(?<=\w)--(?=\w)", MDASH, t)
    t = t.replace(" -- ", " " + MDASH + " ").replace(" - ", " " + NDASH + " ").replace("...", ELL)
    return t.replace("\n", "\r\n")


def audit(a) -> None:
    from entropy_protocol import crypto
    from entropy_protocol.backend_hf import HFBackend
    from entropy_protocol.canon import CANON_V3, canonicalize
    from entropy_protocol.detect import detect
    from entropy_protocol.seal import SealParams, decider, finish, margins, measure
    from entropy_protocol.verify import verify
    sys.path.insert(0, "bench")
    from _meta import code_digest

    import torch
    b = HFBackend(a.model, device=a.device, dtype=getattr(torch, a.dtype))
    if a.chat:
        prompts = ["Write a paragraph that begins: " + p for p in BASE] + QUOTES_CHAT
        prompts = [b.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                   for p in prompts]
    else:
        prompts = BASE + QUOTES_RAW
    rows, t0 = [], time.time()
    for i, prompt in enumerate(prompts):
        master = bytes([i]) * 32
        for cg in (False, True):
            params = SealParams(max_new_tokens=a.tokens, min_new_tokens=int(a.tokens * 0.8), canon_gen=cg)
            doc = f"audit-{i}"
            _, _, hm = margins(params)
            ids, enf = b.generate(prompt, decider(params, master, doc), params.max_new_tokens, params.temperature,
                                  params.top_p, params.k, params.high_window, 100 + i, params.min_new_tokens, hm,
                                  params.align, params.free_after, gate=params.gate, guard=params.guard, canon_gen=cg)
            m = measure(b, ids, enf)
            s = finish(b, params, m, b.position_stats(m.ctext, m.anchors, params.k) if m.anchors else [], master, doc)
            raw, ct = m.raw, m.ctext
            rids = b.tok(ct, add_special_tokens=False)["input_ids"]
            view = ([int(b.canon_guard.lead[ids[0]])] + ids[1:]) if (cg and ids) else ids
            ops = [o for o in difflib.SequenceMatcher(None, view, rids, autojunk=False).get_opcodes() if o[0] != "equal"]
            first_piece = b.tok.decode(ids[:1]) if ids else ""
            d = detect(b, raw, crypto.wm_key(master))
            w = word_autocorrect(raw)
            rows.append({
                "i": i, "canon_gen": cg, "tokens": len(ids), "enforced": s.enforced, "entries": s.entries,
                "dropped": [s.dropped_anchor, s.dropped_rank, s.dropped_margin, s.dropped_gate],
                "residual": ct != raw.strip(), "typo_in_raw": sorted({hex(ord(c)) for c in raw if c in TYPO}),
                "space_before_nl": bool(re.search(r"[ \t]+\n", raw)),
                "nfc": unicodedata.normalize("NFC", raw) != raw, "fffd": chr(0xFFFD) in raw,
                "lead_ws": first_piece[:1].isspace(), "lead_pure_ws": bool(first_piece) and not first_piece.strip(),
                "regions": [{"tag": o[0], "at": o[1], "from_end": len(view) - o[2],
                             "gen": b.tok.convert_ids_to_tokens(view[o[1]:o[2]])[:8],
                             "retok": b.tok.convert_ids_to_tokens(rids[o[3]:o[4]])[:8]} for o in ops],
                "det_cps": d.checkpoints, "det_matched": d.matched, "det_unsat": d.unsat, "det_p": d.p_value,
                "word_changed": w.replace("\r\n", "\n") != raw,
                "word_verdict": verify(b, w, s.token, crypto.log_key(master, doc)).verdict,
                "v3_would_differ": canonicalize(w, CANON_V3) != canonicalize(raw, CANON_V3),
                "text": raw[:240],
            })
            r = rows[-1]
            print(f"[{time.time() - t0:6.0f}s] #{i:2d} cg={int(cg)} enf={r['enforced']} ent={r['entries']} "
                  f"det={r['det_matched']}/{r['det_cps']} regions={len(r['regions'])} word={r['word_verdict']}", flush=True)
            with open(a.out, "w") as fh:
                json.dump({"model": a.model, "chat": a.chat, "code": code_digest(), "rows": rows}, fh, indent=1)


def summary(path: str) -> None:
    d = json.load(open(path))
    print(f"{d['model']} chat={d['chat']} code={d.get('code')}")
    for cg in (False, True):
        R = [r for r in d["rows"] if r["canon_gen"] == cg]
        if not R:
            continue
        cps, mt = sum(r["det_cps"] for r in R), sum(r["det_matched"] for r in R)
        where = Counter("START" if g["at"] == 0 else "END" if g["from_end"] == 0 else "MID"
                        for r in R for g in r["regions"])
        mids = [(r["i"], g["gen"], g["retok"]) for r in R for g in r["regions"] if g["at"] > 0 and g["from_end"] > 0]
        print(f"canon_gen={cg}: texts={len(R)} enforced={sum(r['enforced'] for r in R)} "
              f"entries={sum(r['entries'] for r in R)} dropped={[sum(r['dropped'][k] for r in R) for k in range(4)]}")
        print(f"   detection matched {mt}/{cps} = {mt / max(1, cps):.4f}; texts with a mismatch "
              f"{sum(r['det_matched'] < r['det_cps'] for r in R)}; unsat skipped {sum(r.get('det_unsat', 0) for r in R)}")
        print(f"   residual {sum(r['residual'] for r in R)}; typographic {sum(bool(r['typo_in_raw']) for r in R)}; "
              f"space-before-newline {sum(r['space_before_nl'] for r in R)}; NFC {sum(r['nfc'] for r in R)}; "
              f"U+FFFD {sum(r['fffd'] for r in R)}; lead ws {sum(r['lead_ws'] for r in R)}")
        print(f"   regions {dict(where)}; MID {mids[:6]}")
        print(f"   word transport: changed {sum(r['word_changed'] for r in R)}; verdicts "
              f"{dict(Counter(r['word_verdict'] for r in R))}; v3 would differ {sum(r['v3_would_differ'] for r in R)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--out")
    ap.add_argument("--summary")
    a = ap.parse_args()
    if a.summary:
        summary(a.summary)
    else:
        audit(a)


if __name__ == "__main__":
    main()
