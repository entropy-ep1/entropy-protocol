#!/usr/bin/env python3
"""Model-based attacks on a sealed corpus (protocol section C, classes 4-6).

  paraphrase    an open instruction model rewrites the text (1..R recursive rounds)            class 4
  dipper        DIPPER (T5-XXL) paraphraser, lexical diversity 20/40/60, order 0/20, 3-sentence windows
  translate     round trip EN -> <lang> -> EN through the instruction model                    class 5
  strip_emoji   a corpus sealed under an "emoji after every word" instruction, emojis removed  class 6

Instruction models run through vLLM when it is installed (fast), else through transformers.
Output rows are appended to runs/<model>/attacks_<corpus>.jsonl as in attacks_text.py.

Usage: python3 bench/gpu/attacks_llm.py --model M --corpus c4_sealed_L400 --mode paraphrase \
         --attacker Qwen/Qwen2.5-32B-Instruct-AWQ --rounds 3
"""
from __future__ import annotations

import argparse
import re

from common import RUNS, Store, chunks, log, read_jsonl, slug

EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]")
PROMPTS = {
    "paraphrase": "Paraphrase the following text. Keep its meaning and roughly its length. Output only the "
                  "paraphrase.\n\n{text}",
    "to": "Translate the following text into {lang}. Output only the translation.\n\n{text}",
    "back": "Translate the following text into English. Output only the translation.\n\n{text}",
}
LANGS = {"uk": "Ukrainian", "zh": "Chinese", "es": "Spanish"}


class Chat:
    """Batched chat completion with an instruction model: vLLM if available, else transformers."""

    def __init__(self, model: str, max_tokens: int):
        self.max_tokens = max_tokens
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model, max_model_len=8192, gpu_memory_utilization=0.9)
            self.sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens)
            self.kind = "vllm"
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            self.tok = AutoTokenizer.from_pretrained(model)
            self.tok.padding_side = "left"
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, device_map="cuda")
            self.kind = "hf"
        log(f"attacker {model} via {self.kind}")

    def __call__(self, prompts: list[str]) -> list[str]:
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        if self.kind == "vllm":
            outs = self.llm.chat(msgs, self.sp, use_tqdm=False)
            return [o.outputs[0].text.strip() for o in outs]
        import torch
        texts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = self.model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.95, max_new_tokens=self.max_tokens,
                                      pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]


def sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]


class Dipper:
    """DIPPER (Krishna et al., NeurIPS 2023): lexical/order codes, 3-sentence windows, previous output as prefix;
    windows of many texts are paraphrased in one batch."""

    def __init__(self, model: str = "kalpeshk2011/dipper-paraphraser-xxl"):
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.tok = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self.model = T5ForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda").eval()

    def __call__(self, texts: list[str], lex: int, order: int, batch: int = 32) -> list[str]:
        import torch
        wins = [[" ".join(s[i:i + 3]) for i in range(0, len(s), 3)] for s in (sentences(t) for t in texts)]
        prefix = [""] * len(texts)
        out = [""] * len(texts)
        for w in range(max(len(x) for x in wins)):
            idx = [i for i, x in enumerate(wins) if w < len(x)]
            for part in chunks(idx, batch):
                inp = [f"lexical = {100 - lex}, order = {100 - order}" + (f" {prefix[i]}" if prefix[i] else "")
                       + f" <sent> {wins[i][w]} </sent>" for i in part]
                enc = self.tok(inp, return_tensors="pt", padding=True, truncation=True, max_length=1024).to("cuda")
                with torch.no_grad():
                    gen = self.model.generate(**enc, do_sample=True, top_p=0.75, top_k=None, max_length=512)
                for i, g in zip(part, self.tok.batch_decode(gen, skip_special_tokens=True)):
                    prefix[i] = (prefix[i] + " " + g).strip()
                    out[i] = (out[i] + " " + g).strip()
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="the sealing model (whose corpus is attacked)")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--mode", required=True, choices=["paraphrase", "dipper", "translate", "strip_emoji"])
    ap.add_argument("--attacker", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--langs", default="uk,zh")
    ap.add_argument("--dipper", default="20:0,40:0,60:20", help="lex:order settings")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    rows = [r for r in read_jsonl(RUNS / slug(a.model) / f"{a.corpus}.jsonl") if r.get("text")]   # any mode
    rows = rows[:a.n] if a.n else rows
    st = Store(RUNS / slug(a.model) / f"attacks_{a.corpus}.jsonl")

    def emit(r, name, text):
        rid = f"{r['id']}|{name}"
        if rid not in st.done:
            st.write({"id": rid, "src": r["id"], "attack": name, "key": r.get("key", 0), "doc_id": r.get("doc_id", ""),
                      "text": text})

    if a.mode == "strip_emoji":
        for r in rows:
            emit(r, "emoji_stripped", re.sub(r"[ \t]{2,}", " ", EMOJI.sub("", r["text"])))
        return log(f"strip_emoji: {len(rows)} texts")
    if a.mode == "dipper":
        dp = Dipper()
        for setting in a.dipper.split(","):
            lex, order = (int(x) for x in setting.split(":"))
            name = f"dipper_lex{lex}_ord{order}"
            todo = [r for r in rows if f"{r['id']}|{name}" not in st.done]
            for part in chunks(todo, 256):
                for r, t in zip(part, dp([r["text"] for r in part], lex, order)):
                    emit(r, name, t)
            log(f"{name}: {len(todo)} texts")
        return
    chat = Chat(a.attacker, a.max_tokens)
    if a.mode == "paraphrase":
        cur = {r["id"]: r["text"] for r in rows}
        for k in range(1, a.rounds + 1):
            name = f"paraphrase_r{k}"
            done = {r["src"]: r["text"] for r in st.rows() if r["attack"] == name}
            todo = [r for r in rows if r["id"] not in done]
            for part in chunks(todo, a.batch):
                for r, t in zip(part, chat([PROMPTS["paraphrase"].format(text=cur[r["id"]]) for r in part])):
                    emit(r, name, t)
                    done[r["id"]] = t
            cur = {rid: done.get(rid, cur[rid]) for rid in cur}
            log(f"{name}: done")
    else:
        for lang in a.langs.split(","):
            name = f"translate_{lang}_en"
            todo = [r for r in rows if f"{r['id']}|{name}" not in st.done]
            for part in chunks(todo, a.batch):
                mid = chat([PROMPTS["to"].format(lang=LANGS[lang], text=r["text"]) for r in part])
                back = chat([PROMPTS["back"].format(text=t) for t in mid])
                for r, t in zip(part, back):
                    emit(r, name, t)
            log(f"{name}: {len(todo)} texts")


if __name__ == "__main__":
    main()
