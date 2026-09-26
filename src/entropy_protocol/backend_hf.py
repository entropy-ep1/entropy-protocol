"""Hugging Face transformers backend: constrained generation and the canonical scoring pass.

Verification context ("bos+text"): the model re-reads the canonical candidate text alone,
preceded by the tokenizer's BOS (or EOS) token — no prompt, so the prompt can stay confidential.
"""
from __future__ import annotations

import glob
import hashlib
import os
import unicodedata
from dataclasses import dataclass
from functools import cached_property

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from .record import HIGH, LOW

hf_logging.set_verbosity_error()


def weight_files(root: str) -> list[str]:
    """The checkpoint files a transformers loader reads from ``root``: the shards named by the index, else the
    single-file checkpoint; safetensors before pickle."""
    import json
    for index, single in (("model.safetensors.index.json", "model.safetensors"),
                          ("pytorch_model.bin.index.json", "pytorch_model.bin")):
        path = os.path.join(root, index)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return [os.path.join(root, f) for f in sorted(set(json.load(fh)["weight_map"].values()))]
        if os.path.exists(os.path.join(root, single)):
            return [os.path.join(root, single)]
    return sorted(glob.glob(os.path.join(root, "*.safetensors"))) or sorted(glob.glob(os.path.join(root, "*.bin")))


def first_choice_prob(logits: torch.Tensor) -> torch.Tensor:
    """Probability of the first choice along the last axis, exp(max - logsumexp): one formula for the generator's
    gate, the record and the detector, and no full softmax over the vocabulary."""
    return torch.exp(logits.max(-1).values - torch.logsumexp(logits, -1))


@dataclass(frozen=True)
class PositionStat:
    token_index: int
    char_range: tuple[int, int]
    p_tok: float
    rank0: int
    top_sorted: list[float]                 # k+1 largest probabilities, descending
    l_tok: float = 0.0                      # logit of the token present
    top_logits: tuple[float, ...] = ()      # k+1 largest logits, descending

    def values(self, space: str) -> tuple[float, list[float]]:
        """(value of the token present, top-(k+1) values) in probability or logit space."""
        return (self.l_tok, list(self.top_logits)) if space == "logit" else (self.p_tok, self.top_sorted)


class HFBackend:
    def __init__(self, model_id: str, device: str = "cpu", dtype: torch.dtype = torch.float32,
                 local_files_only: bool = True, attn: str | None = None):
        """``attn``: attention kernel ("sdpa", "eager", "flash_attention_2"); default: the library's choice."""
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only, use_fast=True)
        if not self.tok.is_fast:
            raise ValueError("a fast tokenizer with offset mapping is required")
        kw = {"attn_implementation": attn} if attn else {}
        if str(device).startswith("cuda"):
            # straight to the accelerator: a 14B checkpoint does not fit twice in a GPU host's memory
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=local_files_only, torch_dtype=dtype, device_map=device, **kw).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, local_files_only=local_files_only, torch_dtype=dtype, **kw).to(device).eval()
        start = self.tok.bos_token_id if self.tok.bos_token_id is not None else self.tok.eos_token_id
        if start is None:
            raise ValueError("tokenizer has neither a BOS nor an EOS token")
        self.start_id = start
        self.max_positions = int(getattr(self.model.config, "max_position_embeddings", 0) or
                                 getattr(self.model.config, "n_positions", 0) or 2048)
        # tokens that end a text: every end-of-sequence id the model declares, and every special token
        # (a special token is never text; e.g. Qwen2.5-Instruct ends on <|im_end|> and on <|endoftext|>)
        gen_cfg = getattr(self.model, "generation_config", None)
        eos_cfg = getattr(gen_cfg, "eos_token_id", None)
        stops = set([eos_cfg] if isinstance(eos_cfg, int) else list(eos_cfg or []))
        if self.tok.eos_token_id is not None:
            stops.add(self.tok.eos_token_id)
        stops |= set(self.tok.all_special_ids)
        stops |= {i for i, t in self.tok.added_tokens_decoder.items() if getattr(t, "special", False)}
        self.stop_ids = frozenset(stops)
        # filler for padded batch rows (never attended to): the pad token, else the end-of-text token
        self.pad_id = next(i for i in (self.tok.pad_token_id, self.tok.eos_token_id, self.start_id) if i is not None)
        # MPS keeps freed blocks cached up to its low watermark (1.4 x the recommended working set: 16.6 GB on a 16 GB
        # Mac), and a generation allocates new sizes at every step (its cache grows by one position), so that cache
        # fills. 13.09.2026, fp16, 4 GSM8K questions sealed: Qwen2.5-1.5B 2.9 GB of live tensors, 11.7 GB held, a 14 GB
        # footprint; Phi-3-mini (no grouped attention: about 1 GB of cache reallocated per step) past 16 GB, the
        # system in swap. After every forward pass of the model the cache goes back to the system once it exceeds the
        # live tensors by MPS_SLACK (see tidy): Phi-3 then holds at most 9.1 GB (7.8 GB live), same tokens.
        self._mps = str(device).startswith("mps")
        if self._mps:
            body = getattr(self.model, "base_model", None) or self.model
            body.register_forward_hook(lambda *_: self.tidy())

    MPS_SLACK = 1 << 30     # bytes the MPS allocator may keep cached beyond the live tensors

    def tidy(self) -> None:
        """Return the MPS allocator's cache to the system once it holds more than MPS_SLACK beyond the live tensors
        (nothing on other devices). Only unused blocks are freed: no result changes."""
        if self._mps and torch.mps.driver_allocated_memory() - torch.mps.current_allocated_memory() > self.MPS_SLACK:
            torch.mps.empty_cache()

    # ---------------------------------------------------------------- identity
    @property
    def model_ref(self) -> str:
        rev = getattr(self.model.config, "_commit_hash", None) or "local"
        return f"{self.model_id}@{rev}"

    @cached_property
    def weights_digest(self) -> str:
        """sha256 over the checkpoint files the loader reads, as distributed (independent of runtime dtype/device).
        Only those files: a repository may carry the same weights in a second format (Mistral ships
        consolidated.safetensors next to the shards), and a verifier that downloaded a different set of files
        must still obtain the same digest."""
        if os.path.isdir(self.model_id):
            root = self.model_id
        else:
            from huggingface_hub import snapshot_download
            root = snapshot_download(self.model_id, local_files_only=self.local_files_only)
        files = weight_files(root)
        if not files:
            raise FileNotFoundError(f"no weight files found for {self.model_id}")
        h = hashlib.sha256()
        for path in files:
            h.update(os.path.basename(path).encode())
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
        return h.hexdigest()

    @cached_property
    def canon_guard(self):
        """Ban masks for canonical generation (canongen.py), built once per backend."""
        from .canongen import CanonGuard
        return CanonGuard(self.tok, int(self.model.get_output_embeddings().weight.shape[0]), device=self.device)

    @cached_property
    def tokenizer_digest(self) -> str:
        return hashlib.sha256(self.tok.backend_tokenizer.to_str().encode()).hexdigest()

    @property
    def config_class(self) -> str:
        """Calibration class of this inference configuration: the numeric precision of the scoring pass."""
        return {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}.get(self.dtype, str(self.dtype))

    @cached_property
    def components(self) -> dict:
        """Measured components of the model (RFC 10013 style: name -> sha-256 hex). Weights and tokenizer decide
        verification; chat template and generation defaults shape generation only and are recorded as provenance."""
        import json
        out = {"weights": self.weights_digest, "tokenizer": self.tokenizer_digest}
        template = getattr(self.tok, "chat_template", None)
        if template:
            out["chat_template"] = hashlib.sha256(template.encode()).hexdigest()
        gen_cfg = getattr(self.model, "generation_config", None)
        if gen_cfg is not None:
            d = {k: v for k, v in gen_cfg.to_dict().items() if k != "transformers_version"}
            out["generation_config"] = hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
        return out

    @property
    def environment(self) -> dict:
        """The inference environment of this backend (the determinism envelope a record or a result reports)."""
        import platform

        import transformers
        dev = str(self.device)
        if dev.startswith("cuda") and torch.cuda.is_available():
            hw = torch.cuda.get_device_name(torch.device(dev))
        else:
            hw = platform.machine()
        return {"device": dev, "hardware": hw, "dtype": str(self.dtype).replace("torch.", ""),
                "class": self.config_class, "torch": torch.__version__, "transformers": transformers.__version__,
                "attn": str(getattr(self.model.config, "_attn_implementation", "")), "batch": 1}

    # -------------------------------------------------------------- generation
    @staticmethod
    def _nucleus(scores: torch.Tensor, temperature: float, top_p: float, g: torch.Generator) -> int:
        if temperature <= 0:                               # greedy decoding
            return int(torch.argmax(scores))
        p = torch.softmax(scores / temperature, 0)
        sp, si = torch.sort(p, descending=True)
        keep = (torch.cumsum(sp, 0) - sp) < top_p
        sp, si = sp[keep], si[keep]
        return int(si[torch.multinomial(sp / sp.sum(), 1, generator=g)])

    @torch.no_grad()
    def generate(self, prompt: str, decide, max_new_tokens: int, temperature: float, top_p: float,
                 k: int, high_window: int, seed: int | None = None,
                 min_new_tokens: int = 0, high_margin: float = 0.0,
                 align: str = "prompt", free_after: int = 0,
                 gate: float = 0.0, guard: float = 0.0,
                 canon_gen: bool = False) -> tuple[list[int], dict[int, int]]:
        """Generate with keyed entropy predicates. ``decide(j, gen)`` returns None, LOW or HIGH for position j
        (``gen`` = the token ids generated so far, read-only).
        End-of-text tokens are unavailable before ``min_new_tokens`` tokens. A HIGH position never selects
        a special token; where the model's first choice is to end the text, the text ends.
        ``high_margin`` (logits): a HIGH position selects only among tokens whose logit lies at least that far
        below the k-th largest logit, so the checkpoint keeps a margin the seal's tolerance filter accepts.
        ``align``: the context in which a predicate is enforced. "prompt" = the generation context (prompt +
        text). "free" = the verification context (start token + text, no prompt), kept in a second cache: the
        predicate then holds exactly where a verifier or detector evaluates it, and the token is still chosen
        by the prompt-conditioned distribution among those that satisfy it. With ``free_after`` = W, positions
        before W are enforced in the generation context (early in a text the prompt-free view knows too little
        of the topic) and positions from W on in the verification context.
        ``gate`` (confidence gate, needs align="free"): a position is a checkpoint only if the verification
        context's first choice has probability below ``gate``, so the model is never pushed off a choice it is
        sure of (code, arithmetic, names). The detector sees the same probability, and the gate does not depend
        on the key, so the exact null of record-free detection holds on the gated positions.
        ``guard``: no checkpoint where the generation context (with the prompt) is at least this sure of its first
        choice — what the question implies but the text has not said yet. A detector cannot see this, so such
        positions only dilute record-free detection (they behave like unsealed text); the null is unaffected.
        ``canon_gen``: never emit a character the canonical form would change (canongen.py), so the text is
        canonical as generated and a verifier re-reads exactly the enforced tokens. Predicates keep their
        meaning in the unmasked verification context: a LOW position whose first choice is banned is left free.
        The verifier's view (its cache, and the token contexts ``decide`` reads) takes the first token as the
        verifier will read it once the canonical form has stripped its leading whitespace."""
        if align not in ("prompt", "free"):
            raise ValueError(f"unknown enforcement context {align!r}")
        if gate and align != "free":
            raise ValueError("the confidence gate is read in the verification context: it needs align='free'")
        g = torch.Generator(device="cpu")
        g.manual_seed(seed if seed is not None else int.from_bytes(os.urandom(8), "little"))
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.model(ids, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]
        past_f = logits_f = None
        if align == "free":
            out_f = self.model(torch.tensor([[self.start_id]], device=self.device), use_cache=True)
            past_f, logits_f = out_f.past_key_values, out_f.logits[0, -1]
        gen: list[int] = []
        view: list[int] = []                               # the verifier's view of gen (see canon_gen)
        enforced: dict[int, int] = {}
        stops = self.stop_ids
        stop_list = sorted(stops)
        cg = self.canon_guard if canon_gen else None
        cstate, spaced = 0, False                          # canonical generation: unfinished character, trailing space
        for j in range(max_new_tokens):
            scores = logits.float().cpu()
            if j < min_new_tokens and stop_list:
                scores[stop_list] = float("-inf")
            banned = None
            if cg is not None:
                banned = cg.masks_cpu[cstate].clone()
                if j == 0:
                    banned |= cg.lead_ban.cpu()            # the verifier's first token; and never an empty text
                    if stop_list:
                        banned[stop_list] = True
                if spaced:
                    banned |= cg.nl_start_cpu
                if cstate == 0 and bool(banned[int(torch.argmax(logits_f if logits_f is not None else logits))]):
                    banned |= cg.partial_cpu               # reaching for a banned character: no half characters
                scores[banned] = float("-inf")
            prop = decide(j, view)
            if prop is not None and gate and float(first_choice_prob(logits_f.float())) >= gate:
                prop = None                                # the verifier's view is confident here: no checkpoint
            if prop is not None and guard and float(first_choice_prob(scores)) >= guard:
                prop = None                                # the generation context is confident: leave it alone
            rejected: list[int] = []                       # joins the verifier would re-tokenize differently
            first_try = None                               # the model's own choice, kept if nothing joins
            for _ in range(self.JOIN_TRIES):
                top = int(torch.argmax(scores))
                if prop is not None and top in stops:
                    prop, nxt = None, top                  # the model's first choice is to end: the text ends
                elif prop is None:
                    nxt = self._nucleus(scores, temperature, top_p, g)
                elif align == "prompt" or j < free_after:
                    nxt = top if prop == LOW else self._high_choice(
                        scores, scores, k, high_window, high_margin if align == "prompt" else 0.0, temperature, stops,
                        g, exclude=(cg.high_exclusion(torch.topk(scores, k).indices.tolist()) + rejected)
                        if cg is not None else None)
                else:
                    ref = logits_f.float().cpu()           # the verifier's view of this position
                    if prop == LOW:
                        nxt = int(torch.argmax(ref))
                        if nxt in stops or (banned is not None and bool(banned[nxt])) or nxt in rejected:
                            prop, nxt = None, self._nucleus(scores, temperature, top_p, g)   # cannot be satisfied
                    else:
                        nxt = self._high_choice(ref, scores, k, high_window, high_margin, temperature, stops, g,
                                                exclude=(cg.high_exclusion(torch.topk(ref, k).indices.tolist())
                                                         + rejected) if cg is not None else None)
                if first_try is None:
                    first_try = (prop, nxt)
                if cg is None or nxt in stops or j == 0 or cg.joinable(view, nxt):
                    break
                rejected.append(nxt)
                scores[nxt] = float("-inf")
            else:                                          # every choice rejected: the best joinable one
                avoid = set()
                if prop == HIGH and logits_f is not None and not (align == "prompt" or j < free_after):
                    top_ref = torch.topk(logits_f.float().cpu(), k).indices.tolist()
                    avoid = set(top_ref) | set(cg.high_exclusion(top_ref))
                elif prop == HIGH:
                    top_gen = torch.topk(scores, k).indices.tolist()
                    avoid = set(top_gen) | set(cg.high_exclusion(top_gen))
                elif prop == LOW:
                    prop = None                            # the first choice itself was rejected
                pick = cg.first_joinable(view, scores, avoid, stops)
                if pick is not None:
                    nxt = pick
                else:                                      # nothing joins: a local break beats a far-down choice
                    prop, nxt = first_try
            if nxt in stops:
                break
            gen.append(nxt)
            view.append(int(cg.lead[nxt]) if cg is not None and j == 0 else nxt)
            if cg is not None:
                cstate, spaced = cg.state_after(cstate, nxt), cg.ends_in_space(nxt)
            if prop is not None:
                enforced[len(gen) - 1] = prop
            step = torch.tensor([[nxt]], device=self.device)
            out = self.model(step, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[0, -1]
            if align == "free":
                step_f = step if view[-1] == nxt else torch.tensor([[view[-1]]], device=self.device)
                out_f = self.model(step_f, past_key_values=past_f, use_cache=True)
                past_f, logits_f = out_f.past_key_values, out_f.logits[0, -1]
        return gen, enforced

    JOIN_TRIES = 8      # canonical generation: choices tried at one position before a rejected join is kept

    @staticmethod
    def _high_choice(ref: torch.Tensor, scores: torch.Tensor, k: int, window: int, high_margin: float,
                     temperature: float, stops, g: torch.Generator, exclude: list[int] | None = None) -> int:
        """A token outside the top-k of ``ref`` (and at least ``high_margin`` below its k-th logit), chosen among
        the ``window`` most probable such tokens under ``scores`` (the generation distribution). ``exclude``:
        tokens never chosen (canonical generation: the byte prefixes of the first choice)."""
        ranked = torch.argsort(ref, descending=True)
        allowed = torch.ones_like(ref, dtype=torch.bool)
        allowed[ranked[:k]] = False
        if stops:
            allowed[sorted(stops)] = False
        if exclude:
            allowed[exclude] = False
        if high_margin > 0:
            allowed &= (float(ref[ranked[k - 1]]) - ref) >= high_margin
        if not bool(allowed.any()):
            allowed[ranked[k:]] = True
            if stops:
                allowed[sorted(stops)] = False
            if exclude:
                allowed[exclude] = False
        masked = scores.masked_fill(~allowed, float("-inf"))
        cand = torch.topk(masked, min(window, int(allowed.sum()))).indices
        if temperature <= 0:                               # greedy: the most probable token that satisfies HIGH
            return int(cand[0])
        p = torch.softmax(masked[cand] / temperature, 0)
        return int(cand[torch.multinomial(p, 1, generator=g)])

    def decode_with_spans(self, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
        """Decoded (NFC) text and the character span each generated token contributed."""
        spans, prev = [], ""
        for i in range(len(ids)):
            cur = unicodedata.normalize("NFC", self.tok.decode(ids[: i + 1], clean_up_tokenization_spaces=False))
            spans.append((len(prev), len(cur)))
            prev = cur
        return prev, spans

    # ------------------------------------------------------------ scoring pass
    HEAD_ROWS = 512     # output-head rows computed at once in a scoring pass (a vocabulary-wide row each)

    @cached_property
    def split(self):
        """(base model, output head) when the model's logits are exactly head(last hidden state): a scoring pass then
        computes the vocabulary-wide rows only where they are needed, in chunks, instead of a [length x vocabulary]
        tensor (5 GB for 16k positions of a 152k vocabulary). Checked once on a short input; None (full forward) if a
        model transforms the logits after its head (soft-capping, scaling) or is not split this way."""
        head = self.model.get_output_embeddings()
        base = getattr(self.model, "base_model", None)
        if head is None or base is None or base is self.model:
            return None
        probe = torch.tensor([[self.start_id] + list(range(10, 17))], device=self.device)
        with torch.no_grad():
            full = self.model(probe, use_cache=False).logits[0].float()
            part = head(base(input_ids=probe, use_cache=False).last_hidden_state[0]).float()
        if full.shape != part.shape or float((full - part).abs().max()) > 1e-3 * max(1.0, float(full.abs().max())):
            return None
        return base, head

    @torch.no_grad()
    def logits_at(self, input_ids: torch.Tensor, rows: list[int]) -> torch.Tensor:
        """The pass's logits (float32, CPU) at the given positions of one sequence; row t predicts input_ids[t + 1]."""
        if self.split is None:
            return self.model(input_ids, use_cache=False).logits[0][rows].float().cpu()
        base, head = self.split
        h = base(input_ids=input_ids, use_cache=False).last_hidden_state[0]
        parts = [head(h[rows[i:i + self.HEAD_ROWS]]).float().cpu() for i in range(0, len(rows), self.HEAD_ROWS)]
        return torch.cat(parts) if parts else torch.zeros((0, head.out_features))

    @torch.no_grad()
    def probs_for_ids(self, ids: list[int]) -> torch.Tensor:
        """Full next-token distributions (float32, CPU) for BOS + ids; row t is the distribution for ids[t]."""
        inp = torch.tensor([[self.start_id] + ids], device=self.device)
        return torch.softmax(self.model(inp).logits[0, :-1].float(), dim=-1).cpu()

    def canonical_ids(self, ctext: str) -> list[int]:
        return self.tok(ctext, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def position_stats(self, ctext: str, char_offsets: list[int], k: int) -> list[PositionStat | None]:
        """For each canonical char offset: the token covering it and its rank/probabilities under bos+text."""
        enc = self.tok(ctext, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        if len(ids) + 1 > self.max_positions:
            raise ValueError(f"text of {len(ids)} tokens exceeds the model context of {self.max_positions}")
        cover: dict[int, int] = {}
        for t, (a, b) in enumerate(offs):
            for c in range(a, b):
                cover.setdefault(c, t)
        need = sorted({cover[c] for c in char_offsets if c in cover})
        by_t: dict[int, PositionStat] = {}
        if need:
            inp = torch.tensor([[self.start_id] + ids], device=self.device)
            rows = self.logits_at(inp, need)                           # row t predicts ids[t]
            tgt = torch.tensor([ids[t] for t in need])
            l_tok = rows.gather(1, tgt[:, None])
            rank0 = (rows > l_tok).sum(1)
            lse = torch.logsumexp(rows, -1, keepdim=True)             # probabilities without the full softmax
            p_tok = torch.exp(l_tok - lse).squeeze(1)
            top_l = torch.topk(rows, k + 1, dim=-1).values
            top = torch.exp(top_l - lse)
            for i, t in enumerate(need):
                by_t[t] = PositionStat(t, (int(offs[t][0]), int(offs[t][1])), float(p_tok[i]), int(rank0[i]),
                                       [float(x) for x in top[i]], float(l_tok[i, 0]),
                                       tuple(float(x) for x in top_l[i]))
        return [by_t.get(cover[c]) if c in cover else None for c in char_offsets]
