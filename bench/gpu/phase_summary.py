"""One Markdown page from a phase directory synced down from an evaluation machine (runs_gpu/<name>/): step times,
throughput, the calibration table, record density, verification, detection, baselines, quality and speed. Files that
are missing are reported as missing, so the page also works on a phase that is still running.
Usage: python3 bench/gpu/phase_summary.py runs_gpu/a1 [--model Qwen/Qwen2.5-7B-Instruct] [--prefix A]"""
import argparse
import json
import re
import statistics
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def steps(log: Path) -> list[str]:
    if not log.exists():
        return ["(no phase log)"]
    start, out = {}, []
    for line in log.read_text().splitlines():
        m = re.match(r"(\d\d:\d\d:\d\d) (START|END|SKIP) (\S+)(?: rc=(\d+))?", line)
        if not m:
            if "PHASE A" in line:
                out.append(f"- {line}")
            continue
        t, kind, name, rc = m.groups()
        h, mi, s = map(int, t.split(":"))
        sec = h * 3600 + mi * 60 + s
        if kind == "START":
            start[name] = sec
        elif kind == "SKIP":
            out.append(f"| {name} | skipped (done earlier) | |")
        else:
            dur = (sec - start.get(name, sec)) % 86400
            out.append(f"| {name} | {'ok' if rc == '0' else 'FAILED rc=' + rc} | {dur // 60}:{dur % 60:02d} |")
    running = [n for n in start if not any(f"| {n} |" in o for o in out)]
    out += [f"| {n} | running | |" for n in running]
    return ["| step | result | min:s |", "|---|---|---|"] + out


def corpus_line(d: Path, name: str) -> str | None:
    rs = rows(d / f"{name}.jsonl")
    if not rs:
        return None
    toks = [r["tokens"] for r in rs]
    secs = sum(r.get("seconds", 0) for r in rs)
    batches = {r.get("batch") for r in rs}
    line = (f"| {name} | {len(rs)} | {statistics.mean(toks):.0f} | "
            f"{sum(toks) / secs:.0f} | {','.join(map(str, sorted(b for b in batches if b)))} |")
    if rs[0].get("mode") == "sealed":
        ent = [len(r.get("entries") or []) if isinstance(r.get("entries"), list) else r.get("entries", 0) for r in rs]
        enf = [r.get("enforced", 0) for r in rs]
        line += f" {statistics.mean(enf):.1f} | {statistics.mean(ent):.1f} | {sum(e == 0 for e in ent)} |"
    else:
        line += " | | |"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase_dir")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prefix", default="A")
    a = ap.parse_args()
    root = Path(a.phase_dir)
    d = root / a.model.replace("/", "_")
    P = a.prefix
    out = [f"# Phase {P} — {a.model} ({root.name})", ""]

    meta = next((load(p) for p in sorted(d.glob(f"{P}_*.meta.json"))), None)
    if meta:
        env = meta.get("env", {})
        out += [f"- code `{meta.get('code')}`, weights `{str(meta.get('weights'))[:16]}`",
                f"- hardware: {env.get('hardware')}; torch {env.get('torch')}; attention {env.get('attn')}", ""]

    out += ["## Steps", ""] + steps(root / "logs" / "phase_a.log") + [""]

    out += ["## Corpora", "", "| corpus | texts | mean tokens | tok/s | batch | enforced/text | record points/text | "
            "texts with 0 points |", "|---|---|---|---|---|---|---|---|"]
    for p in sorted(d.glob(f"{P}_*.jsonl")):
        if "detect_" in p.name or p.name.startswith("attacks_"):
            continue
        line = corpus_line(d, p.stem)
        if line:
            out.append(line)
    out.append("")

    cal = load(d / f"calibration_{P}_cal.json")
    out += ["## Calibration (logit space, k = 1)", ""]
    if cal:
        out += [f"- reference `{cal['reference']}`, {cal['texts']} texts, {cal['positions']} positions",
                f"- certified table: {cal['table']}; record margin needed {cal['record_margin_needed']:.4f}; "
                f"positions kept {pct(cal['kept_share'])}", "",
                "| configuration | max |Δ| (delta) | p99.99 | flips (all) | flips at kept positions |", "|---|---|---|---|---|"]
        for cfg, s in cal["per_config"].items():
            out.append(f"| {cfg} | {s.get('delta', 0):.4f} | {s.get('p99.99', 0):.4f} | {s.get('flips_all')} | "
                       f"{s.get('flips_at_kept_positions')} |")
    else:
        out.append("(missing)")
    out.append("")

    out += ["## Verification by record", ""]
    for p in sorted(d.glob("*.summary.json")):
        if p.name.startswith(("detect_", "baseline_")):
            continue
        s = load(p)["summary"]
        out.append(f"- **{p.name.replace('.summary.json', '')}**: {s['texts']} texts, decisive {s['decisive']}, "
                   f"same → VERIFIED {s['same_VERIFIED_%']}%, wrong MODIFIED {s['same_wrong_MODIFIED']}, "
                   f"INDETERMINATE {s['same_INDETERMINATE']}; transport wrong MODIFIED "
                   f"{sum(s['transport_wrong_MODIFIED'].values())}")
        if "edits" in s:
            e = s["edits"]
            out.append("  - edits: " + "; ".join(f"{k} {v['MODIFIED_%']}% (localized {v['localized_%']}%, "
                                                  f"false VERIFIED {v['false_VERIFIED']})" for k, v in e.items()))
            out.append(f"  - unrelated false VERIFIED {s.get('unrelated_false_VERIFIED')}, privacy leaks "
                       f"{s.get('privacy_leaks')}")
    out.append("")

    out += ["## Detection without a record", ""]
    for p in sorted(d.glob("detect_*.summary.json")):
        r = load(p)
        out += [f"**{p.name.replace('.summary.json', '')}** (tol {r['tol']:.4f}, {r.get('minutes')} min)", "",
                "| length | texts | at 1e-6 | at 1e-4 | at 1e-2 | checkpoints | match |", "|---|---|---|---|---|---|---|"]
        for L, x in r["positives"].items():
            da = x["detected_at"]
            out.append(f"| {L} | {x['texts']} ({x['texts_reaching_L']} full) | {pct(da['1e-06']['rate'])} | "
                       f"{pct(da['0.0001']['rate'])} | {pct(da['0.01']['rate'])} | {x['mean_checkpoints']} | "
                       f"{pct(x['match_rate'])} |")
        out += ["", "| null | pairs | false positives at 1e-4 | at 1e-2 | min p |", "|---|---|---|---|---|"]
        for lab, x in r["nulls"].items():
            fp = x["false_positive_at"]
            out.append(f"| {lab} | {x.get('pairs')} | {fp['0.0001']['count']} | {fp['0.01']['count']} "
                       f"({pct(fp['0.01']['rate'])}) | {x.get('min_p')} |")
        if r["attacks"]:
            out += ["", "| attack | texts | at 1e-6 | at 1e-2 | segment at 1e-6 |", "|---|---|---|---|---|"]
            for att, x in r["attacks"].items():
                out.append(f"| {att} | {x['texts']} | {pct(x['detected_at']['1e-06']['rate'])} | "
                           f"{pct(x['detected_at']['0.01']['rate'])} | {pct(x['segment_detected_at_1e-6'])} |")
        out.append("")

    out += ["## Baselines (same model, same prompts)", ""]
    for p in sorted(d.glob("baseline_*.summary.json")):
        r = load(p)
        for L, per in r["positives"].items():
            for k, x in per.items():
                ex = x.get("exact_tpr_at") or {}
                out.append(f"- {r['scheme']} {k} L={L}: TPR at empirical FPR 1e-2 {x.get('tpr_at_fpr_0.01')}, "
                           f"1e-3 {x.get('tpr_at_fpr_0.001')}; exact p ≤ 1e-4 {ex.get('0.0001')}")
    out.append("")

    out += ["## Quality", ""]
    for p in sorted(d.glob("quality_*.json")):
        out.append(f"- {p.stem}: {json.dumps(load(p), ensure_ascii=False)}")
    out.append("")

    out += ["## Speed", ""]
    for p in sorted(d.glob("speed_*.json")):
        r = load(p)
        for B, x in r["generation"].items():
            out.append(f"- batch {B}: plain {x['plain_tok_s']} tok/s, sealed {x['sealed_tok_s']} tok/s "
                       f"(overhead {x['overhead_%']}%), peak {x['peak_gb_sealed']} GB")
        for B, x in r["verify"].items():
            out.append(f"- verify batch {B}: {x['tok_s']} tok/s, {x['seconds_per_text']} s/text")
    print("\n".join(out))


if __name__ == "__main__":
    main()
