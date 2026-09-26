"""Tables for the report from a bridge part's result files (bench/run_bridge_part7.sh, part8.sh): no hand copying.

Usage: python3 bench/bridge_summary.py Qwen_Qwen2.5-1.5B-Instruct [--runs bench/results/runs]
Prints Markdown: GSM8K (accuracy against plain, detection without the record), prose (perplexity, diversity,
detection), the record's density per corpus, the batch-size check, and the gates files of the model if present.
"""
import argparse
import glob
import json
import os
import statistics


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def rows(path):
    return [json.loads(line) for line in open(path) if line.strip()] if os.path.exists(path) else []


def pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def det(summary, length, alpha):
    try:
        return summary["positives"][str(length)]["detected_at"][alpha]["rate"]
    except (KeyError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--runs", default="bench/results/runs")
    a = ap.parse_args()
    d = os.path.join(a.runs, a.slug)
    print(f"### {a.slug}\n")

    plain = rows(os.path.join(d, "gsm_plain.jsonl"))
    if plain:
        print(f"**GSM8K** ({len(plain)} problems, greedy; detection without the record: rate at 1e-6 (at 1e-2), "
              f"match rate)\n")
        print("| Variant | Accuracy | Only this / only plain correct | 100 tok. | 200 tok. | 400 tok. | Match | "
              "Entries (median) |")
        print("|---|---|---|---|---|---|---|---|")
        acc_plain = None
        for v in ("quality", "strong"):
            q = load(os.path.join(d, f"quality_gsm8k_gsm_{v}_vs_gsm_plain.json"))
            if q and acc_plain is None:
                acc_plain = q["accuracy_b"]
        print(f"| plain | {pct(acc_plain)} | — | — | — | — | — | — |")
        for v in ("quality", "strong"):
            q = load(os.path.join(d, f"quality_gsm8k_gsm_{v}_vs_gsm_plain.json"))
            s = load(os.path.join(d, f"detect_gsm_{v}.summary.json"))
            r = rows(os.path.join(d, f"gsm_{v}.jsonl"))
            cells = [f"{pct(det(s, L, '1e-06'))} ({pct(det(s, L, '0.01'))})" for L in (100, 200, 400)]
            match = s["positives"].get("400", s["positives"].get("200", {})).get("match_rate") if s else None
            ent = statistics.median([x["entries"] for x in r]) if r else None
            print(f"| {v} | {pct(q['accuracy_a']) if q else '—'} | "
                  f"{(str(q['only_a_correct']) + ' / ' + str(q['only_b_correct'])) if q else '—'} | "
                  + " | ".join(cells) + f" | {pct(match)} | {ent if ent is not None else '—'} |")
        print()

    pp = rows(os.path.join(d, "prose_plain.jsonl"))
    if pp:
        print(f"**Prose** ({len(pp)} prompts, T 0.7; perplexity of the gpt2-medium oracle against plain, 95% CI; "
              f"detection without the record)\n")
        print("| Variant | Perplexity | 100 tok. 1e-6 | 200 tok. 1e-6 | 200 tok. 1e-4 | 200 tok. 1e-2 | Match | "
              "distinct-2 | self-BLEU | Entries (median) |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        div0 = None
        for v in ("quality", "strong"):
            q = load(os.path.join(d, f"quality_ppl_prose_{v}_vs_prose_plain.json"))
            s = load(os.path.join(d, f"detect_prose_{v}.summary.json"))
            dv = load(os.path.join(d, f"quality_diversity_prose_{v}_vs_prose_plain.json"))
            r = rows(os.path.join(d, f"prose_{v}.jsonl"))
            if dv:
                div0 = dv["prose_plain"]
            ppl = f"x{q['geo_mean_ppl_ratio']:.2f} [{q['ci95'][0]:.2f}-{q['ci95'][1]:.2f}]" if q else "—"
            match = s["positives"].get("200", {}).get("match_rate") if s else None
            ent = statistics.median([x["entries"] for x in r]) if r else None
            dd = dv[f"prose_{v}"] if dv else None
            print(f"| {v} | {ppl} | {pct(det(s, 100, '1e-06'))} | {pct(det(s, 200, '1e-06'))} | "
                  f"{pct(det(s, 200, '0.0001'))} | {pct(det(s, 200, '0.01'))} | {pct(match)} | "
                  f"{dd['distinct_2'] if dd else '—'} | {dd['self_bleu4'] if dd else '—'} | "
                  f"{ent if ent is not None else '—'} |")
        if div0:
            print(f"| plain | x1 | — | — | — | — | — | {div0['distinct_2']} | {div0['self_bleu4']} | — |")
        print()

    b4 = os.path.join(os.path.dirname(a.runs.rstrip("/")), "runs_batch4", a.slug, "gsm_quality.jsonl")
    if os.path.exists(b4):
        x = {r["id"]: r for r in rows(b4)}
        y = {r["id"]: r for r in rows(os.path.join(d, "gsm_quality.jsonl"))}
        common = sorted(set(x) & set(y))
        same = sum(x[i]["text"] == y[i]["text"] for i in common)
        print(f"**Batch size:** {same} of {len(common)} GSM8K texts identical at batch 4 and batch 16.\n")

    for path in sorted(glob.glob(f"bench/results/gates_{a.slug}_*_p7*.json")):
        s = json.load(open(path))["summary"]
        g1 = s.get("G1", {})
        print(f"**Gates** `{os.path.basename(path)}`: texts {s['n']}, mean entries {s.get('mean_entries')}, "
              f"survival {s.get('survival_%')}%; G1 wrong MODIFIED "
              f"{sum(v for k, v in g1.items() if k.endswith('wrong_MODIFIED'))}; G2 false VERIFIED "
              f"{s.get('G2', {}).get('false_VERIFIED')}; G3 MODIFIED "
              f"{[round(v.get('MODIFIED_%', 0)) for v in s.get('G3', {}).values() if isinstance(v, dict)]}; "
              f"G6 leaks {s.get('G6_privacy_leaks')}; G10-lite {s.get('G10_lite')}\n")


if __name__ == "__main__":
    main()
