#!/usr/bin/env python3
"""Evaluation data (licences allow commercial use): writes data/*.jsonl.

  c4.jsonl          news prompts (first ~100 words) + the human continuation (~300 words)   allenai/c4 realnewslike, ODC-BY
  wiki_<lang>.jsonl human passages (~250 words; ~600 characters for zh) + a prompt prefix     wikimedia/wikipedia, CC BY-SA
  dolly.jsonl       open-ended instructions (creative writing, open/general QA, brainstorming) databricks-dolly-15k, CC BY-SA
  gsm8k.jsonl       grade-school maths, test split (1319)                                      openai/gsm8k, MIT
  humaneval.jsonl   code completion with unit tests (164)                                      openai/openai_humaneval, MIT

--tiny builds a small offline set from the Python standard library's docstrings (for tests without network).
Usage: python3 bench/gpu/prep_data.py [--c4 20000] [--wiki 5000] [--langs en,uk,es,zh,ar] [--tiny]
"""
from __future__ import annotations

import argparse
import os
import sys

from common import DATA, Store, log


def _words(text: str) -> list[str]:
    return text.split()


def c4(n: int, prompt_words: int = 100, human_words: int = 300) -> None:
    from datasets import load_dataset
    import itertools
    st = Store(DATA / "c4.jsonl")
    # the validation split (13.9k documents, the usual watermark benchmark source) first, then the train split
    val = ((f"c4-v{i:06d}", r) for i, r in enumerate(load_dataset("allenai/c4", "realnewslike", split="validation",
                                                                  streaming=True)))
    train = ((f"c4-t{i:07d}", r) for i, r in enumerate(load_dataset("allenai/c4", "realnewslike", split="train",
                                                                    streaming=True)))
    k = len(st.done)
    for rid, row in itertools.chain(val, train):
        if k >= n:
            break
        w = _words(row["text"])
        if len(w) < prompt_words + human_words:
            continue
        if rid in st.done:
            continue
        st.write({"id": rid, "prompt": " ".join(w[:prompt_words]), "human": " ".join(w[prompt_words:prompt_words + human_words]),
                  "url": row.get("url", "")})
        k += 1
    log(f"c4: {k} items")


def wiki(lang: str, n: int) -> None:
    from datasets import load_dataset
    st = Store(DATA / f"wiki_{lang}.jsonl")
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    k = len(st.done)
    cjk = lang in ("zh", "ja")
    for i, row in enumerate(ds):
        if k >= n:
            break
        paras = [p.strip() for p in row["text"].split("\n") if len(p.strip()) > (200 if cjk else 300)]
        for j, p in enumerate(paras[:3]):
            rid = f"wiki-{lang}-{i:07d}-{j}"
            if rid in st.done or k >= n:
                continue
            if cjk:
                text = p[:600]
                prompt = text[:150]
            else:
                w = _words(p)
                if len(w) < 120:
                    continue
                text = " ".join(w[:250])
                prompt = " ".join(w[:60])
            st.write({"id": rid, "lang": lang, "text": text, "prompt": prompt, "title": row.get("title", "")})
            k += 1
    log(f"wiki {lang}: {k} items")


def dolly(n: int) -> None:
    from datasets import load_dataset
    st = Store(DATA / "dolly.jsonl")
    keep = {"creative_writing", "open_qa", "general_qa", "brainstorming"}
    k = len(st.done)
    for i, row in enumerate(load_dataset("databricks/databricks-dolly-15k", split="train")):
        if k >= n:
            break
        if row["category"] not in keep or len(row["instruction"]) < 20:
            continue
        rid = f"dolly-{i:05d}"
        if rid in st.done:
            continue
        instr = row["instruction"] + (("\n\n" + row["context"]) if row.get("context") else "")
        st.write({"id": rid, "instruction": instr, "category": row["category"], "human": row["response"]})
        k += 1
    log(f"dolly: {k} items")


def gsm8k() -> None:
    from datasets import load_dataset
    st = Store(DATA / "gsm8k.jsonl")
    for i, row in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
        rid = f"gsm8k-{i:04d}"
        if rid not in st.done:
            answer = row["answer"].split("####")[-1].strip().replace(",", "")
            st.write({"id": rid, "question": row["question"] + "\nThink step by step, then give the final answer "
                      "as a number on the last line.", "answer": answer})
    log(f"gsm8k: {len(st.done)} items")


def humaneval() -> None:
    from datasets import load_dataset
    st = Store(DATA / "humaneval.jsonl")
    for row in load_dataset("openai/openai_humaneval", split="test"):
        rid = "humaneval-" + row["task_id"].split("/")[-1]
        if rid not in st.done:
            st.write({"id": rid, "prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"],
                      "instruction": "Complete the following Python function. Reply with the full function in one "
                                     "Python code block.\n\n" + row["prompt"]})
    log(f"humaneval: {len(st.done)} items")


def tiny() -> None:
    """Offline smoke-test data from the standard library's docstrings (human-written prose)."""
    from g10_watermark import human_paragraphs
    from gates import PROMPTS
    paras = human_paragraphs(80, min_words=40)
    st = Store(DATA / "c4.jsonl")
    for i, p in enumerate(paras[:40]):
        rid = f"tiny-c4-{i:03d}"
        if rid not in st.done:
            st.write({"id": rid, "prompt": PROMPTS[i % len(PROMPTS)], "human": p})
    st = Store(DATA / "wiki_en.jsonl")
    for i, p in enumerate(paras[40:]):
        rid = f"tiny-wiki-{i:03d}"
        if rid not in st.done:
            st.write({"id": rid, "lang": "en", "text": p, "prompt": " ".join(p.split()[:20])})
    st = Store(DATA / "dolly.jsonl")
    for i, q in enumerate(["Write a short story about a lighthouse keeper.", "Explain how vaccines work.",
                           "Give me ideas for a rainy weekend.", "Why is the sky blue?"]):
        if f"tiny-dolly-{i}" not in st.done:
            st.write({"id": f"tiny-dolly-{i}", "instruction": q, "category": "tiny", "human": ""})
    st = Store(DATA / "gsm8k.jsonl")
    if "tiny-gsm8k-0" not in st.done:
        st.write({"id": "tiny-gsm8k-0", "question": "Tom has 3 apples and buys 4 more. How many apples does he have?\n"
                  "Think step by step, then give the final answer as a number on the last line.", "answer": "7"})
    log(f"tiny data written to {DATA}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c4", type=int, default=20000)
    ap.add_argument("--wiki", type=int, default=5000)
    ap.add_argument("--langs", default="en,uk,es,zh,ar")
    ap.add_argument("--dolly", type=int, default=2000)
    ap.add_argument("--only", default="", help="comma list of: c4,wiki,dolly,gsm8k,humaneval")
    ap.add_argument("--tiny", action="store_true")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if a.tiny:
        return tiny()
    only = set(a.only.split(",")) if a.only else {"c4", "wiki", "dolly", "gsm8k", "humaneval"}
    if "c4" in only:
        c4(a.c4)
    if "wiki" in only:
        for lang in a.langs.split(","):
            wiki(lang, a.wiki)
    if "dolly" in only:
        dolly(a.dolly)
    if "gsm8k" in only:
        gsm8k()
    if "humaneval" in only:
        humaneval()


if __name__ == "__main__":
    main()
    # Every row is already on disk: Store writes, flushes and fsyncs each line. Leave without interpreter
    # finalization - the streaming loaders of `datasets` keep background threads whose teardown aborts CPython at
    # exit ("PyGILState_Release: thread state ... must be current", SIGABRT, rc 134), which marked a complete
    # download as failed (phase B on the L40S, 26.09.2026). An exception in main() still exits non-zero.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
