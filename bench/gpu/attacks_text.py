#!/usr/bin/env python3
"""Text-level attacks on a sealed corpus (protocol section C, classes 1-3), no model needed.

  1 benign transformations   nfkc, collapse_whitespace, crlf, smart_quotes (copy-paste from a word processor),
                             markdown_rendered (emphasis/bullets stripped), zero_width (inserted)
  2 edits                    words_5/10/25/50 (replaced by other words of the text), chars_5/10 (insert/delete/
                             substitute), homoglyphs_5 (Latin -> Cyrillic look-alikes)
  3 dilution                 embed_10 / embed_25 (sealed share of a human document, 1 span), embed_3spans_25,
                             long_doc (a sealed passage inside ~12k tokens of human text)
  +                          truncate_half, delete_sentence (stage-2 set)

Output: runs/<model>/attacks_<corpus>.jsonl rows {id, src, attack, key, doc_id, text, edited_share}
Usage: python3 bench/gpu/attacks_text.py --model M --corpus c4_sealed_L400 --human data:c4:human
"""
from __future__ import annotations

import argparse
import random
import re
import unicodedata

from common import DATA, RUNS, Store, log, read_jsonl, slug

HOMOGLYPHS = {"a": "\u0430", "c": "\u0441", "e": "\u0435", "o": "\u043e", "p": "\u0440", "x": "\u0445",
              "y": "\u0443", "i": "\u0456", "A": "\u0410", "B": "\u0412", "C": "\u0421", "E": "\u0415",
              "H": "\u041d", "K": "\u041a", "M": "\u041c", "O": "\u041e", "P": "\u0420", "T": "\u0422"}


def words_replaced(t: str, frac: float, rng: random.Random) -> str:
    w = t.split(" ")
    pool = [x for x in w if x.isalpha()] or ["the"]
    for i in rng.sample(range(len(w)), max(1, int(frac * len(w)))):
        w[i] = rng.choice(pool)
    return " ".join(w)


def chars_edited(t: str, frac: float, rng: random.Random) -> str:
    s = list(t)
    for _ in range(max(1, int(frac * len(s)))):
        i = rng.randrange(len(s))
        op = rng.randrange(3)
        if op == 0:
            s.insert(i, rng.choice("abcdefghijklmnopqrstuvwxyz"))
        elif op == 1 and len(s) > 1:
            del s[i]
        else:
            s[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(s)


def homoglyphs(t: str, frac: float, rng: random.Random) -> str:
    idx = [i for i, ch in enumerate(t) if ch in HOMOGLYPHS]
    s = list(t)
    for i in rng.sample(idx, min(len(idx), max(1, int(frac * len(t))))):
        s[i] = HOMOGLYPHS[s[i]]
    return "".join(s)


def smart_quotes(t: str) -> str:
    t = re.sub(r'"(\S)', "\u201c\\1", t)
    t = t.replace('"', "\u201d")
    t = re.sub(r"(\w)'(\w)", "\\1\u2019\\2", t)
    return t.replace("'", "\u2019").replace("--", "\u2014").replace("...", "\u2026")


def markdown_rendered(t: str) -> str:
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)            # headings
    t = re.sub(r"^\s*[-*+]\s+", "", t, flags=re.M)                 # bullets
    t = re.sub(r"(\*\*|__|\*|_|`)(.+?)\1", r"\2", t)               # emphasis, code spans
    return t


def zero_width(t: str, rng: random.Random) -> str:
    s = t.split(" ")
    for i in rng.sample(range(len(s)), max(1, len(s) // 10)):
        s[i] = s[i] + "\u200b"
    return " ".join(s)


def delete_sentence(t: str) -> str:
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s)", t)]
    if len(ends) < 3:
        return t
    a, b = ends[len(ends) // 2 - 1] + 1, ends[len(ends) // 2] + 1
    return t[:a] + t[b:]


def embed(t: str, human: list[str], share: float, spans: int, rng: random.Random) -> str:
    """The sealed text cut into ``spans`` pieces placed inside human text so the sealed part is ``share`` of all."""
    need = int(len(t) * (1 - share) / share)
    filler, i = [], 0
    while sum(len(x) for x in filler) < need:
        filler.append(human[(rng.randrange(len(human)) + i) % len(human)])
        i += 1
    pieces = [t[len(t) * j // spans: len(t) * (j + 1) // spans] for j in range(spans)]
    doc = []
    for j, f in enumerate(filler):
        doc.append(f)
        if j < spans:
            doc.append(pieces[j])
    return "\n\n".join(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--human", default="data:c4:human", help="data:<file>:<field> for dilution")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rows = [r for r in read_jsonl(RUNS / slug(a.model) / f"{a.corpus}.jsonl") if r.get("text")]   # any mode
    rows = rows[:a.n] if a.n else rows
    _, name, field = a.human.split(":")
    human = [r[field] for r in read_jsonl(DATA / f"{name}.jsonl") if r.get(field)]
    st = Store(RUNS / slug(a.model) / f"attacks_{a.corpus}.jsonl")
    rng = random.Random(a.seed)
    for r in rows:
        t = r["text"]
        att = {
            "nfkc": unicodedata.normalize("NFKC", t),
            "collapse_whitespace": re.sub(r"[ \t]+", " ", re.sub(r"\n{2,}", "\n", t)).strip(),
            "crlf": t.replace("\n", "\r\n"),
            "smart_quotes": smart_quotes(t),
            "markdown_rendered": markdown_rendered(t),
            "zero_width": zero_width(t, rng),
            "words_5": words_replaced(t, 0.05, rng), "words_10": words_replaced(t, 0.10, rng),
            "words_25": words_replaced(t, 0.25, rng), "words_50": words_replaced(t, 0.50, rng),
            "chars_5": chars_edited(t, 0.05, rng), "chars_10": chars_edited(t, 0.10, rng),
            "homoglyphs_5": homoglyphs(t, 0.05, rng),
            "truncate_half": t[: len(t) // 2], "delete_sentence": delete_sentence(t),
            "embed_10": embed(t, human, 0.10, 1, rng), "embed_25": embed(t, human, 0.25, 1, rng),
            "embed_3spans_25": embed(t, human, 0.25, 3, rng),
            "long_doc": embed(t[: len(t) // 2], human, 0.02, 1, rng),
        }
        for name_, txt in att.items():
            rid = f"{r['id']}|{name_}"
            if rid not in st.done:
                st.write({"id": rid, "src": r["id"], "attack": name_, "key": r.get("key", 0),
                          "doc_id": r.get("doc_id", ""), "text": txt})
    log(f"attacks for {len(rows)} texts -> {st.path}")


if __name__ == "__main__":
    main()
