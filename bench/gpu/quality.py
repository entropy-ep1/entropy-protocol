#!/usr/bin/env python3
"""Quality price of sealing, paired against the same model's unsealed texts for the same prompts.

  ppl        oracle perplexity over the common prefix of each (sealed, plain) pair, prompt-conditioned for raw
             continuations; geometric-mean ratio with a 95 % bootstrap interval (as G5b)
  judge      pairwise preference by an LLM judge, both presentation orders (vLLM if installed)
  gsm8k      exact-answer accuracy, sealed vs plain; discordant pairs (McNemar)
  humaneval  pass@1 by running the unit tests (subprocess, timeout; disposable evaluation machine only)
  diversity  distinct-1/2/3 and Self-BLEU-4

Usage: python3 bench/gpu/quality.py ppl --model M --oracle Qwen/Qwen2.5-32B-Instruct --a c4_sealed_L400 \
         --b c4_plain_L400 --data c4 --field prompt
"""
from __future__ import annotations

import argparse
import math
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter

import torch

from common import DATA, RUNS, add_backend_args, chunks, load_backend, log, read_jsonl, slug, write_json
from entropy_protocol.canon import canonicalize


def pairs(model: str, a: str, b: str) -> list[tuple[dict, dict]]:
    A = {r["id"]: r for r in read_jsonl(RUNS / slug(model) / f"{a}.jsonl")}
    B = {r["id"]: r for r in read_jsonl(RUNS / slug(model) / f"{b}.jsonl")}
    return [(A[i], B[i]) for i in sorted(A) if i in B]


def boot_ci(logs: list[float], n: int = 2000) -> list[float]:
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(logs, k=len(logs))) for _ in range(n))
    return [round(math.exp(bs[int(0.025 * n)]), 3), round(math.exp(bs[int(0.975 * n) - 1]), 3)]


@torch.no_grad()
def nll(oracle, prefix_ids: list[list[int]], text_ids: list[list[int]], batch: int = 8) -> list[float]:
    """Mean negative log-likelihood of each text given its prefix (right-padded batches)."""
    out = []
    for part in chunks(list(range(len(text_ids))), batch):
        seqs = [prefix_ids[i] + text_ids[i] for i in part]
        L = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), L), oracle.pad_id, dtype=torch.long)
        for r, s in enumerate(seqs):
            inp[r, :len(s)] = torch.tensor(s)
        logp = torch.log_softmax(oracle.model(input_ids=inp.to(oracle.device)).logits.float(), -1)
        for r, i in enumerate(part):
            p0, n = len(prefix_ids[i]), len(text_ids[i])
            tgt = torch.tensor(text_ids[i], device=oracle.device)
            rows = logp[r, p0 - 1:p0 - 1 + n]
            out.append(float(-rows.gather(1, tgt[:, None]).mean()))
    return out


def cmd_ppl(a):
    oracle = load_backend(a.oracle, a.device, a.dtype, a.attn)
    items = {r["id"]: r for r in read_jsonl(DATA / f"{a.data}.jsonl")} if a.data else {}
    ps = pairs(a.model, a.a, a.b)[: a.n or None]
    pre, xa, xb = [], [], []
    for ra, rb in ps:
        ia = oracle.canonical_ids(canonicalize(ra["text"]))
        ib = oracle.canonical_ids(canonicalize(rb["text"]))
        T = min(len(ia), len(ib), a.max_tokens)
        if T < 30:
            continue
        p = [oracle.start_id] + (oracle.canonical_ids(items[ra["id"]][a.field]) if a.field and ra["id"] in items else [])
        pre.append(p); xa.append(ia[:T]); xb.append(ib[:T])
    na, nb = nll(oracle, pre, xa), nll(oracle, pre, xb)
    logs = [x - y for x, y in zip(na, nb)]                   # log(ppl_a / ppl_b)
    res = {"pairs": len(logs), "oracle": a.oracle, "geo_mean_ppl_ratio": round(math.exp(statistics.mean(logs)), 3),
           "ci95": boot_ci(logs), "median_ppl_a": round(math.exp(statistics.median(na)), 2),
           "median_ppl_b": round(math.exp(statistics.median(nb)), 2), "conditioned_on_prompt": bool(a.field)}
    write_json(RUNS / slug(a.model) / f"quality_ppl_{a.a}_vs_{a.b}.json", res)
    log(str(res))


JUDGE = ("You compare two responses to the same instruction. Decide which response is better overall: helpful, "
         "correct, fluent and coherent. Reply with exactly one letter: A, B, or T for a tie.\n\n"
         "Instruction:\n{q}\n\nResponse A:\n{x}\n\nResponse B:\n{y}\n\nWhich is better? Reply A, B or T.")


def cmd_judge(a):
    from attacks_llm import Chat
    chat = Chat(a.judge, 4)
    items = {r["id"]: r for r in read_jsonl(DATA / f"{a.data}.jsonl")}
    ps = [(x, y) for x, y in pairs(a.model, a.a, a.b) if x["id"] in items][: a.n or None]
    prompts = []
    for x, y in ps:
        q = items[x["id"]][a.field]
        prompts.append(JUDGE.format(q=q, x=x["text"], y=y["text"]))      # sealed first
        prompts.append(JUDGE.format(q=q, x=y["text"], y=x["text"]))      # sealed second
    answers = []
    for part in chunks(prompts, 64):
        answers += [(t.strip()[:1] or "T").upper() for t in chat(part)]
    wins = ties = losses = consistent_a = consistent_b = 0
    for i in range(len(ps)):
        first, second = answers[2 * i], answers[2 * i + 1]
        va = {"A": 1, "B": -1}.get(first, 0)                             # +1 = sealed preferred
        vb = {"B": 1, "A": -1}.get(second, 0)
        for v in (va, vb):
            wins += v == 1; losses += v == -1; ties += v == 0
        consistent_a += va == vb == 1
        consistent_b += va == vb == -1
    n = 2 * len(ps)
    res = {"judge": a.judge, "pairs": len(ps), "sealed_win_rate": round((wins + 0.5 * ties) / max(1, n), 4),
           "wins": wins, "ties": ties, "losses": losses, "consistent_sealed": consistent_a,
           "consistent_plain": consistent_b}
    write_json(RUNS / slug(a.model) / f"quality_judge_{slug(a.judge)}_{a.a}_vs_{a.b}.json", res)
    log(str(res))


def last_number(t: str) -> str | None:
    nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def cmd_gsm8k(a):
    items = {r["id"]: r for r in read_jsonl(DATA / "gsm8k.jsonl")}
    ps = pairs(a.model, a.a, a.b)
    ok = lambda r: last_number(r["text"]) is not None and float(last_number(r["text"])) == float(items[r["id"]]["answer"])  # noqa: E731
    ca = [ok(x) for x, _ in ps]
    cb = [ok(y) for _, y in ps]
    res = {"items": len(ps), "accuracy_a": round(sum(ca) / max(1, len(ps)), 4), "accuracy_b": round(sum(cb) / max(1, len(ps)), 4),
           "only_a_correct": sum(x and not y for x, y in zip(ca, cb)), "only_b_correct": sum(y and not x for x, y in zip(ca, cb))}
    write_json(RUNS / slug(a.model) / f"quality_gsm8k_{a.a}_vs_{a.b}.json", res)
    log(str(res))


def run_tests(code: str, test: str, entry: str, timeout: int = 10) -> bool:
    prog = code + "\n\n" + test + f"\n\ncheck({entry})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(prog)
    try:
        return subprocess.run([sys.executable, fh.name], capture_output=True, timeout=timeout).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def extract_code(t: str, prompt: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", t, re.S)
    code = m.group(1) if m else t
    return code if "def " in code else prompt + code


def cmd_humaneval(a):
    items = {r["id"]: r for r in read_jsonl(DATA / "humaneval.jsonl")}
    ps = pairs(a.model, a.a, a.b)
    pa = [run_tests(extract_code(x["text"], items[x["id"]]["prompt"]), items[x["id"]]["test"], items[x["id"]]["entry_point"]) for x, _ in ps]
    pb = [run_tests(extract_code(y["text"], items[y["id"]]["prompt"]), items[y["id"]]["test"], items[y["id"]]["entry_point"]) for _, y in ps]
    res = {"items": len(ps), "pass@1_a": round(sum(pa) / max(1, len(ps)), 4), "pass@1_b": round(sum(pb) / max(1, len(ps)), 4),
           "only_a": sum(x and not y for x, y in zip(pa, pb)), "only_b": sum(y and not x for x, y in zip(pa, pb))}
    write_json(RUNS / slug(a.model) / f"quality_humaneval_{a.a}_vs_{a.b}.json", res)
    log(str(res))


def _ngrams(w, n):
    return Counter(tuple(w[i:i + n]) for i in range(len(w) - n + 1))


def self_bleu(texts: list[str], n_max: int = 4, sample: int = 200) -> float:
    rng = random.Random(0)
    toks = [t.split() for t in texts]
    idx = rng.sample(range(len(toks)), min(sample, len(toks)))
    scores = []
    for i in idx:
        refs = [toks[j] for j in idx if j != i]
        logp = 0.0
        for n in range(1, n_max + 1):
            hyp = _ngrams(toks[i], n)
            ref_max: Counter = Counter()
            for r in refs:
                ref_max |= _ngrams(r, n)
            match = sum(min(c, ref_max[g]) for g, c in hyp.items())
            logp += math.log(max(match, 1e-9) / max(1, sum(hyp.values()))) / n_max
        scores.append(math.exp(logp))
    return round(statistics.mean(scores), 4) if scores else 0.0


def cmd_diversity(a):
    res = {}
    for name in (a.a, a.b):
        texts = [r["text"] for r in read_jsonl(RUNS / slug(a.model) / f"{name}.jsonl")]
        words = [t.split() for t in texts]
        res[name] = {f"distinct_{n}": round(statistics.mean(len(set(_ngrams(w, n))) / max(1, len(w) - n + 1) for w in words), 4)
                     for n in (1, 2, 3)}
        res[name]["self_bleu4"] = self_bleu(texts)
    write_json(RUNS / slug(a.model) / f"quality_diversity_{a.a}_vs_{a.b}.json", res)
    log(str(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ppl", "judge", "gsm8k", "humaneval", "diversity"])
    add_backend_args(ap, dtype="bfloat16")
    ap.add_argument("--oracle", default="")
    ap.add_argument("--judge", default="Qwen/Qwen2.5-32B-Instruct")
    ap.add_argument("--a", required=True, help="corpus A (sealed)")
    ap.add_argument("--b", required=True, help="corpus B (plain)")
    ap.add_argument("--data", default="")
    ap.add_argument("--field", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1000)
    a = ap.parse_args()
    {"ppl": cmd_ppl, "judge": cmd_judge, "gsm8k": cmd_gsm8k, "humaneval": cmd_humaneval, "diversity": cmd_diversity}[a.cmd](a)


if __name__ == "__main__":
    main()
