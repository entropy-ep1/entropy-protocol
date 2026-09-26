#!/usr/bin/env python3
"""Quality gates G1 (transport), G2 (unrelated text), G3 (edits + localization), G6 (privacy),
G10-lite (record-free detection of every sealed text; human texts as the null).

Usage:  python3 bench/gates.py --model distilgpt2 --n 200 [--device cpu] [--dtype float32]
Writes bench/results/gates_<model>_N<n>.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import torch  # noqa: E402

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.backend_hf import HFBackend  # noqa: E402
from entropy_protocol.canon import canonicalize_with_map  # noqa: E402
from entropy_protocol.detect import detect  # noqa: E402
from entropy_protocol.record import loads  # noqa: E402
from entropy_protocol.seal import SealParams, seal  # noqa: E402
from entropy_protocol.verify import INDETERMINATE, MODIFIED, VERIFIED, verify  # noqa: E402
from _meta import code_digest  # noqa: E402

PROMPTS = [
    "The city council met on Tuesday to discuss", "In the early morning, the research team",
    "A new study on sleep and memory suggests that", "The history of the printing press shows",
    "When the engineers opened the server room,", "Our quarterly report highlights three trends:",
    "The old lighthouse keeper wrote in his journal", "Scientists have long debated whether",
    "The recipe calls for fresh tomatoes and", "Before the election, voters in the district",
    "Dear customer, we are writing to inform you that", "The court found that the defendant",
    "In a statement released on Monday, the ministry", "The insurance policy covers damage caused by",
    "Students returning to campus this fall will notice", "The museum's new exhibition explores",
    "According to the latest figures from the bureau,", "The startup announced that it had raised",
    "Doctors recommend that patients with high blood pressure", "The train was delayed for two hours because",
    "Our team reviewed the contract and found", "The mayor's office confirmed that the bridge",
    "Farmers in the region are worried that", "The new software update changes the way",
    "It was the coldest winter anyone could remember,", "The committee will vote next week on",
    "Researchers at the university discovered that", "The company's chief executive said that",
    "Residents were asked to leave their homes after", "The article argues that modern cities",
    "Please find attached the minutes of", "The spacecraft entered orbit around",
    "Local businesses have struggled since", "The treaty was signed in", "Our records indicate that your account",
    "The festival returns this summer with", "The hospital announced a new policy on",
    "Investigators said the fire started in", "The school board approved a plan to", "Historians disagree about why",
]
UNRELATED = [
    "I walked to the market this morning because the fridge was empty again. The man who sells apples asked about my "
    "sister, and we talked for a while about the rain. Some weeks are like that: you start many things and finish few.",
    "The committee has reviewed your application and regrets to inform you that it cannot be approved at this time. "
    "You may reapply after the next funding cycle opens. Please retain this letter for your records.",
    "Mix the flour and the salt, then add the water slowly while stirring. Leave the dough to rest for an hour under a "
    "damp cloth. Bake at a high temperature until the crust sounds hollow when you tap it.",
    "Our tenants' association meets every second Thursday in the community room. Agenda items this month include the "
    "broken elevator, the new recycling schedule, and a proposal to repaint the hallway on the third floor.",
]

def _c2pa_wrapper(payload: bytes) -> str:
    """U+FEFF followed by one variation selector per byte (C2PA 2.4 A.8 wrapper layout)."""
    data = b"C2PATXT\x00\x01" + len(payload).to_bytes(4, "big") + payload
    return "\ufeff" + "".join(chr(0xFE00 + b) if b < 16 else chr(0xE0100 + b - 16) for b in data)


TRANSPORTS = {
    "crlf": lambda t: t.replace("\n", "\r\n"),
    "email": lambda t: "\n" + t.replace("\n", "  \n") + "\n\n",
    "nbsp": lambda t: t.replace(" ", "\u00a0", 3),
    "combined": lambda t: "\n" + t.replace("\n", "  \r\n").replace(" ", "\u00a0", 1) + "\n\n",
    "bom_nnbsp": lambda t: "\ufeff" + t.replace(" ", "\u202f", 2),      # Notepad BOM + narrow no-break spaces
    "c2pa_manifest": lambda t: t + _c2pa_wrapper(bytes(range(64))),     # C2PA 2.4 A.8 text manifest appended
    "invisible": lambda t: t.replace(" ", "\u200b ", 2).replace("e", "e\u00ad", 1),   # zero-width space, soft hyphen
}
INSERT = " This sentence was added by someone else."


def _g10(rows, backend, k_wm, params, tol, gate: float = 0.0) -> dict:
    """Record-free detection on the sealed texts; the four hand-written texts as the null."""
    ps = [r["wm_p"] for r in rows if "wm_p" in r]
    if not ps:
        return {}
    null = [detect(backend, t, k_wm, interval=params.interval, k=params.k, context=params.context, tol=tol,
                   gate=gate).p_value for t in UNRELATED]
    return {"texts": len(ps), **{f"detected_at_{a:g}": round(sum(p <= a for p in ps) / len(ps), 4)
                                 for a in (1e-2, 1e-3, 1e-4, 1e-6)},
            "median_p": statistics.median(ps), "null_min_p": min(null)}


def _header_words(d) -> list[str]:
    """Header keys and string values, nested dicts included (they are metadata, not text)."""
    out = []
    for k, v in d.items():
        out.append(str(k))
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            out.extend(_header_words(v))
    return out


def _cpos(raw: str, pos: int) -> int:
    _, m = canonicalize_with_map(raw)
    later = [m[r] for r in range(pos, len(raw)) if r in m]
    return later[0] if later else len(canonicalize_with_map(raw)[0])


def edits(raw: str):
    out = {}
    words = list(re.finditer(r"[A-Za-z]{3,}", raw))
    if words:
        w = words[len(words) // 2]
        out["replace_word"] = (raw[:w.start()] + ("banana" if w.group(0).lower() != "banana" else "orange") + raw[w.end():], w.start())
        mid = w.start() + len(w.group(0)) // 2
        ch = "x" if raw[mid] != "x" else "z"
        out["one_char"] = (raw[:mid] + ch + raw[mid + 1:], mid)
    sents = [m.end() for m in re.finditer(r"[.!?](?=\s)", raw)]
    if len(sents) >= 3:
        a, b = sents[len(sents) // 2 - 1] + 1, sents[len(sents) // 2] + 1
        out["delete_sentence"] = (raw[:a] + raw[b:], a)
        out["insert_sentence"] = (raw[:b - 1] + INSERT + raw[b - 1:], b - 1)
    cut = int(len(raw) * 0.75)
    out["truncate_tail"] = (raw[:cut], cut)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ref = SealParams()
    ap.add_argument("--interval", type=int, default=ref.interval)
    ap.add_argument("--k", type=int, default=ref.k)
    ap.add_argument("--window", type=int, default=ref.high_window)
    ap.add_argument("--cal", help="calibration file from `entropy calibrate` (default: exact process, tol = 0)")
    ap.add_argument("--tag", default="", help="suffix for the result file name")
    ap.add_argument("--align", default=ref.align, choices=["prompt", "free"])
    ap.add_argument("--selection", default=ref.selection, choices=["content", "interval"])
    ap.add_argument("--profile", choices=["quality", "strong"], help="operating point (profiles.py): sealing guard/gate "
                    "and the detector's gate")
    a = ap.parse_args()
    t0 = time.time()
    backend = HFBackend(a.model, device=a.device, dtype=getattr(torch, a.dtype))
    extra = {}
    if a.cal:
        cal = json.loads(Path(a.cal).read_text())
        if cal.get("k") != a.k:
            raise SystemExit(f"calibration was measured for k = {cal.get('k')}, records use k = {a.k}")
        extra = {"tol_space": cal["space"], "cal": cal["table"]}
    if a.profile:
        from entropy_protocol.profiles import detect_gate, seal_overrides
        extra.update(seal_overrides(a.profile))
    det_gate = detect_gate(a.profile) if a.profile else 0.0
    params = SealParams(max_new_tokens=a.max_new_tokens, interval=a.interval, k=a.k, high_window=a.window,
                        align=a.align, selection=a.selection, **extra)
    master = crypto.new_master_key()
    k_wm = crypto.wm_key(master)
    wm_tol = float(extra.get("cal", {}).get(backend.config_class, 0.0)) if extra else 0.0
    localization_misses = []
    rows, prev = [], None
    for i in range(a.n):
        s = seal(backend, PROMPTS[i % len(PROMPTS)], master, params, seed=10_000 + i)
        key = crypto.log_key(master, s.doc_id)
        row = {"i": i, "tokens": s.tokens, "enforced": s.enforced, "entries": s.entries, "seal_s": s.seconds,
               "dropped": [s.dropped_anchor, s.dropped_rank, s.dropped_margin]}
        same = verify(backend, s.text, s.token, key)
        row["same"] = same.verdict
        row["verify_s"] = same.seconds
        row["transport"] = {name: verify(backend, f(s.text), s.token, key).verdict for name, f in TRANSPORTS.items()}
        if same.verdict != VERIFIED or any(v != VERIFIED for v in row["transport"].values()):
            row["text"] = s.text                               # kept for diagnosis of a failed gate
        row["edits"] = {}
        for name, (txt, pos) in edits(s.text).items():
            r = verify(backend, txt, s.token, key)
            cp = _cpos(s.text, pos)
            spans = r.changed_ranges + r.divergence_ranges
            hit = any(a - 2 <= cp <= b + 2 for a, b in spans) or any(abs(p - cp) <= 2 for p in r.deletion_points)
            row["edits"][name] = {"verdict": r.verdict, "first_div": r.first_divergence, "edit_at": cp,
                                  "localized": hit, "model_layer_detected": r.divergent > 0}
            if r.verdict == MODIFIED and not hit and len(localization_misses) < 30:
                ct = canonicalize_with_map(s.text)[0]
                localization_misses.append({"i": i, "edit": name, "edit_at": cp, "changed": r.changed_ranges,
                                            "deleted": r.deletion_points, "divergent": r.divergence_ranges[:5],
                                            "original_around": ct[max(0, cp - 80): cp + 80]})
        if params.selection == "content":
            row["wm_p"] = detect(backend, s.text, k_wm, interval=params.interval, k=params.k, context=params.context,
                                 tol=wm_tol, gate=det_gate).p_value
        row["unrelated"] = [verify(backend, UNRELATED[i % len(UNRELATED)], s.token, key).verdict]
        if prev is not None:
            row["unrelated"].append(verify(backend, prev, s.token, key).verdict)
        prev = s.text
        # G6 privacy: the decrypted record holds offsets/properties and digests only
        plain = crypto.decrypt_token(key, s.token)
        rec = loads(plain)
        allowed = " ".join(_header_words(rec.header.__dict__)).lower()
        blob = plain.decode("latin-1").lower()
        leaked = [w for w in set(re.findall(r"[a-z]{5,}", s.text.lower())) if w in blob and w not in allowed]
        row["privacy_leaks"] = leaked
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{a.n}  {time.time() - t0:.0f}s", flush=True)

    decisive = [r for r in rows if r["same"] != INDETERMINATE]
    def pct(x, n):
        return round(100 * x / n, 2) if n else None
    g1 = {"decisive_texts": len(decisive), "indeterminate_texts": len(rows) - len(decisive),
          "same_VERIFIED_%": pct(sum(r["same"] == VERIFIED for r in decisive), len(decisive)),
          "same_wrong_MODIFIED": sum(r["same"] == MODIFIED for r in rows)}
    for name in TRANSPORTS:
        g1[f"{name}_VERIFIED_%"] = pct(sum(r["transport"][name] == VERIFIED for r in decisive), len(decisive))
        g1[f"{name}_wrong_MODIFIED"] = sum(r["transport"][name] == MODIFIED for r in rows)
    unrelated = [v for r in rows for v in r["unrelated"]]
    g2 = {"checks": len(unrelated), "false_VERIFIED": sum(v == VERIFIED for v in unrelated),
          "MODIFIED_%": pct(sum(v == MODIFIED for v in unrelated), len(unrelated))}
    g3 = {}
    for name in ("replace_word", "one_char", "delete_sentence", "insert_sentence", "truncate_tail"):
        es = [r["edits"][name] for r in decisive if name in r["edits"]]
        det = [e for e in es if e["verdict"] == MODIFIED]
        g3[name] = {"n": len(es), "MODIFIED_%": pct(len(det), len(es)),
                    "false_VERIFIED": sum(e["verdict"] == VERIFIED for e in es),
                    "localized_%": pct(sum(e["localized"] for e in det), len(det)),
                    "model_layer_alone_%": pct(sum(e["model_layer_detected"] for e in es), len(es))}
    summary = {"model": a.model, "device": a.device, "dtype": a.dtype, "n": a.n, "minutes": round((time.time() - t0) / 60, 1),
               "code": code_digest(),
               "operating_point": {"interval": a.interval, "k": a.k, "window": a.window, **extra},
               "texts_below_8_entries": sum(r["entries"] < 8 for r in rows),
               "mean_tokens": round(statistics.mean(r["tokens"] for r in rows), 1),
               "mean_entries": round(statistics.mean(r["entries"] for r in rows), 1),
               "survival_%": pct(sum(r["entries"] for r in rows), sum(r["enforced"] for r in rows)),
               "mean_seal_s": round(statistics.mean(r["seal_s"] for r in rows), 2),
               "mean_verify_s": round(statistics.mean(r["verify_s"] for r in rows), 3),
               "G1": g1, "G2": g2, "G3": g3, "G6_privacy_leaks": sum(bool(r["privacy_leaks"]) for r in rows),
               "G10_lite": _g10(rows, backend, k_wm, params, wm_tol, det_gate), "localization_misses": localization_misses,
               "profile": a.profile or "", "detect_gate": det_gate}
    out = ROOT / "bench" / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"gates_{a.model.replace('/', '_')}_{a.device}_{a.dtype}_N{a.n}{('_' + a.tag) if a.tag else ''}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("written", path)


if __name__ == "__main__":
    main()
