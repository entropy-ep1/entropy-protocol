"""Batched sealing, verification and detection (accelerators; the evaluation program at scale).

The algorithms are those of seal.seal, verify.verify and detect.detect; only the model work is batched.

Generation: one forward pass per step over 2B rows: the B generation contexts (prompt + text, left-padded) and,
with align="free", the B verification contexts (start token + text), so the weights are read once per step for
both views. Rows that end leave the batch (the cache is compacted). Sampling runs on the device.

Scoring: the verifier's pass (start token + canonical ids) over right-padded batches with no attention mask, as
the single pass; causal attention keeps a real position from seeing the padding. Per position only what a
verdict or a detection needs leaves the device: the logit and probability of the token present, its rank,
the k+1 largest logits and probabilities.

Batching changes the order of floating-point reductions, so the logits of a batched pass differ from the
single pass at the level of numerical noise. That is a configuration effect like a change of hardware: the
environment of a batched seal or verification reports its batch size, and calibration measures it.
"""
from __future__ import annotations

import inspect
import os
import time
from dataclasses import dataclass

import torch

from . import crypto
from .backend_hf import PositionStat, first_choice_prob
from .detect import (Detection, checkpoints, decide_counts, detection_tokens, fold_readings, readings_of,
                     unsatisfiable)
from .record import HIGH, LOW
from .seal import SealParams, Sealed, check_params, decider, finish, margins, measure
from .verify import Result, judge, open_record


# ------------------------------------------------------------------------------------------------ sampling
def _generator(device: str, seed: int | None) -> torch.Generator:
    seed = seed if seed is not None else int.from_bytes(os.urandom(8), "little")
    try:
        g = torch.Generator(device=device)
    except RuntimeError:
        g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _multinomial(p: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """One draw per row of the (unnormalized, non-negative) weights p [n, m] -> [n].
    The device TYPE decides where the draw happens: a generator made for "cuda" reports device cuda, its tensors
    cuda:0, and torch accepts the pair (15.09.2026: comparing whole devices sent CUDA draws to the CPU branch with a
    CUDA generator, and every sampled generation failed on the first GPU run)."""
    if g.device.type != p.device.type:
        return torch.multinomial(p.to(g.device), 1, generator=g).squeeze(1).to(p.device)
    return torch.multinomial(p, 1, generator=g).squeeze(1)


def nucleus_batch(scores: torch.Tensor, temperature: float, top_p: float, g: torch.Generator) -> torch.Tensor:
    """Top-p sampling per row of scores [n, V] (as HFBackend._nucleus); temperature 0 = greedy."""
    if temperature <= 0:
        return scores.argmax(-1)
    p = torch.softmax(scores / temperature, -1)
    sp, si = torch.sort(p, descending=True, dim=-1)
    sp = sp * ((torch.cumsum(sp, -1) - sp) < top_p)
    return si.gather(1, _multinomial(sp, g)[:, None]).squeeze(1)


def high_batch(ref: torch.Tensor, scores: torch.Tensor, k: int, window: int, high_margin: float,
               temperature: float, stop_idx: torch.Tensor, g: torch.Generator, cg=None) -> torch.Tensor:
    """Per row, a token outside the top-k of ``ref`` (at least ``high_margin`` below its k-th value where that
    leaves a choice), drawn among the ``window`` most probable such tokens under ``scores``
    (as HFBackend._high_choice; ``cg``: canonical generation, whose rows never choose a prefix of a top-k token)."""
    topv, topi = torch.topk(ref, k, dim=-1)
    allowed = torch.ones_like(ref, dtype=torch.bool)
    allowed.scatter_(1, topi, False)
    if stop_idx.numel():
        allowed[:, stop_idx] = False
    if cg is not None:
        for r, first in enumerate(topi.tolist()):
            ex = cg.high_exclusion(first)
            if ex:
                allowed[r, ex] = False
    if high_margin > 0:
        strict = allowed & ((topv[:, k - 1:k] - ref) >= high_margin)
        allowed = torch.where(strict.any(-1, keepdim=True), strict, allowed)
    masked = scores.masked_fill(~allowed, float("-inf"))
    cv, ci = torch.topk(masked, window, dim=-1)
    if temperature <= 0:
        return ci[:, 0]
    return ci.gather(1, _multinomial(torch.softmax(cv / temperature, -1), g)[:, None]).squeeze(1)


def _accepts(model, name: str) -> bool:
    return name in inspect.signature(model.forward).parameters


# ---------------------------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_batch(backend, prompts: list, decides: list, max_new_tokens: int, temperature: float, top_p: float,
                   k: int, high_window: int, seed: int | None = None, min_new_tokens: int = 0,
                   high_margin: float = 0.0, align: str = "prompt", free_after: int = 0,
                   compact_frac: float = 0.25, gate: float = 0.0, guard: float = 0.0,
                   canon_gen: bool = False) -> list[tuple[list[int], dict[int, int]]]:
    """HFBackend.generate over a batch. ``prompts``: strings (tokenized as the single path does) or token id
    lists (e.g. a chat template applied by the caller). ``decides[i](j, gen)`` as in HFBackend.generate."""
    if align not in ("prompt", "free"):
        raise ValueError(f"unknown enforcement context {align!r}")
    if gate and align != "free":
        raise ValueError("the confidence gate is read in the verification context: it needs align='free'")
    dev = backend.device
    B = len(prompts)
    enc = [list(p) if not isinstance(p, str) else backend.tok(p)["input_ids"] for p in prompts]
    free = align == "free"
    seqs = enc + ([[backend.start_id]] * B if free else [])       # generation rows, then verification rows
    Lp = max(len(x) for x in seqs)
    model = backend.model
    extra = {"logits_to_keep": 1} if _accepts(model, "logits_to_keep") else {}
    # the prompt pass row by row, without padding (as the single path): in a left-padded pass the padding positions
    # attend to nothing, which in half precision can turn them into NaN (seen on MPS, sdpa and eager alike), and
    # 0 x NaN carries it into the row's real positions. The rows' caches are then left-padded with zeros, which the
    # attention mask leaves at weight exactly 0.
    layers, lasts, legacy = None, [], True
    for x in seqs:
        o = model(input_ids=torch.tensor([x], device=dev), use_cache=True, **extra)
        lasts.append(o.logits[0, -1])
        legacy = not hasattr(o.past_key_values, "to_legacy_cache")
        kv = o.past_key_values if legacy else o.past_key_values.to_legacy_cache()
        if layers is None:
            layers = [([], []) for _ in kv]
        for (ks, vs), (k_, v_) in zip(layers, kv):
            w = Lp - k_.shape[-2]
            ks.append(torch.nn.functional.pad(k_, (0, 0, w, 0)))
            vs.append(torch.nn.functional.pad(v_, (0, 0, w, 0)))
        del o, kv
    kv = tuple((torch.cat(ks, 0), torch.cat(vs, 0)) for ks, vs in layers)
    if legacy:
        past = kv
    else:
        from transformers import DynamicCache
        past = DynamicCache.from_legacy_cache(kv)
    del layers, kv
    last = torch.stack(lasts)
    if not bool(torch.isfinite(last).all()):
        raise FloatingPointError(f"non-finite logits after the prompt pass ({backend.environment['class']}, "
                                 f"attention {backend.environment['attn']}): precision overflow")
    mask = torch.tensor([[0] * (Lp - len(x)) + [1] * len(x) for x in seqs], dtype=torch.long, device=dev)
    next_pos = torch.tensor([len(x) for x in seqs], dtype=torch.long, device=dev)
    can_compact = hasattr(past, "batch_select_indices")

    g = _generator(str(dev), seed)
    stops = backend.stop_ids
    stop_idx = torch.tensor(sorted(stops), dtype=torch.long, device=dev)
    gens: list[list[int]] = [[] for _ in range(B)]
    views: list[list[int]] = [[] for _ in range(B)]   # the verifier's view of each text (see HFBackend.generate)
    enforced: list[dict[int, int]] = [{} for _ in range(B)]
    cg = backend.canon_guard if canon_gen else None
    cstates = [0] * B                          # canonical generation: unfinished character per document
    spaced = [False] * B                       # canonical generation: the text ends in a space or tab
    active = list(range(B))                    # document index of each generation row in the batch
    alive = [True] * B                         # per row: the text has not ended (ended rows wait for compaction)
    for j in range(max_new_tokens):
        nb = len(active)
        props = [decides[d](j, views[d]) if alive[r] else None for r, d in enumerate(active)]
        if gate and any(p is not None for p in props):   # confidence gate, read in the verification context
            p1 = first_choice_prob(last[nb:].float()).tolist()
            props = [None if p is not None and p1[r] >= gate else p for r, p in enumerate(props)]
        scores = last[:nb].float()
        if j < min_new_tokens and stop_idx.numel():
            scores[:, stop_idx] = float("-inf")
        if guard and any(p is not None for p in props):  # the generation context is confident: leave it alone
            pg = first_choice_prob(scores).tolist()
            props = [None if p is not None and pg[r] >= guard else p for r, p in enumerate(props)]
        bm = None
        if cg is not None:                         # canonical generation (as HFBackend.generate)
            bm = cg.masks[torch.tensor([cstates[d] for d in active], device=cg.masks.device)]
            if j == 0:
                bm = bm | cg.lead_ban[None, :]         # the verifier's first token; and never an empty text
                if stop_idx.numel():
                    bm[:, stop_idx.to(bm.device)] = True
            sp_rows = [r for r, d in enumerate(active) if spaced[d]]
            if sp_rows:
                bm[sp_rows] |= cg.nl_start
            first = (last[nb:] if free else last[:nb]).argmax(-1).to(bm.device)
            reach = bm.gather(1, first[:, None]).squeeze(1)
            reach &= torch.tensor([cstates[d] == 0 for d in active], device=bm.device)
            if bool(reach.any()):
                bm[reach] |= cg.partial            # reaching for a banned character: no half characters
            scores = scores.masked_fill(bm.to(scores.device), float("-inf"))
        top = scores.argmax(-1)
        samp = nucleus_batch(scores, temperature, top_p, g)
        in_free = free and j >= free_after
        ref = last[nb:].float() if in_free else None
        lowf = ref.argmax(-1) if in_free else None
        high_rows = [r for r in range(nb) if props[r] == HIGH]
        high: dict[int, int] = {}
        if high_rows:
            sel = torch.tensor(high_rows, device=dev)
            if in_free:
                h = high_batch(ref[sel], scores[sel], k, high_window, high_margin, temperature, stop_idx, g, cg=cg)
            else:
                h = high_batch(scores[sel], scores[sel], k, high_window, high_margin if align == "prompt" else 0.0,
                               temperature, stop_idx, g, cg=cg)
            high = dict(zip(high_rows, h.tolist()))
        top_l, samp_l = top.tolist(), samp.tolist()
        lowf_l = lowf.tolist() if lowf is not None else None
        low_banned = (bm.to(lowf.device).gather(1, lowf[:, None]).squeeze(1).tolist()
                      if bm is not None and lowf is not None else None)

        def choose(r: int, rejected=()) -> tuple[int | None, int]:
            """(predicate kept, token) for row r; a re-choice after ``rejected`` joins works on the row alone."""
            prop = props[r]
            if not rejected:
                t = top_l[r]
                if prop is not None and t in stops:
                    return None, t                     # the model's first choice is to end: the text ends
                if prop is None:
                    return None, samp_l[r]
                if not in_free:
                    return prop, (t if prop == LOW else high[r])
                if prop == LOW:
                    f = lowf_l[r]
                    if f in stops or (low_banned is not None and low_banned[r]):     # cannot be satisfied
                        return None, samp_l[r]
                    return prop, f
                return prop, high[r]
            row = scores[r:r + 1]
            t = int(row.argmax())
            if prop is not None and t in stops:
                return None, t
            if prop is None:
                return None, int(nucleus_batch(row, temperature, top_p, g)[0])
            if not in_free:
                return prop, (t if prop == LOW else int(high_batch(
                    row, row, k, high_window, high_margin if align == "prompt" else 0.0, temperature, stop_idx, g,
                    cg=cg)[0]))
            if prop == LOW:
                f = lowf_l[r]
                if f in stops or (low_banned is not None and low_banned[r]) or f in rejected:
                    return None, int(nucleus_batch(row, temperature, top_p, g)[0])
                return prop, f
            return prop, int(high_batch(ref[r:r + 1], row, k, high_window, high_margin, temperature, stop_idx, g,
                                        cg=cg)[0])

        choice = {r: choose(r) for r in range(nb) if alive[r]}
        first_try = dict(choice)                   # the model's own choices, kept where nothing joins
        if cg is not None and j > 0:               # joins the verifier would re-tokenize: banned, chosen again
            rejected: dict[int, list[int]] = {}
            todo = [r for r, (_, t) in choice.items() if t not in stops]
            for attempt in range(backend.JOIN_TRIES):
                bad = [r for r in todo if not cg.joinable(views[active[r]], choice[r][1])]
                if not bad:
                    break
                for r in bad:
                    rejected.setdefault(r, []).append(choice[r][1])
                    scores[r, choice[r][1]] = float("-inf")
                    props[r] = choice[r][0]
                    if attempt < backend.JOIN_TRIES - 1:
                        choice[r] = choose(r, rejected[r])
                        continue
                    prop_r, avoid = props[r], set()    # every choice rejected: the best joinable one
                    if prop_r == HIGH:
                        top_ref = torch.topk(ref[r] if in_free else scores[r], k).indices.tolist()
                        avoid = set(top_ref) | set(cg.high_exclusion(top_ref))
                    elif prop_r == LOW:
                        prop_r = None
                    pick = cg.first_joinable(views[active[r]], scores[r].float().cpu(), avoid, stops)
                    choice[r] = (prop_r, pick) if pick is not None else first_try[r]
                todo = [r for r in bad if choice[r][1] not in stops]
        step: list[int] = []
        fstep: list[int] = []                          # the verifier's view of this step
        keep: list[int] = []
        for r, d in enumerate(active):
            if not alive[r]:
                step.append(backend.pad_id)
                fstep.append(backend.pad_id)
                continue
            prop, nxt = choice[r]
            if nxt in stops:
                alive[r] = False
                step.append(backend.pad_id)
                fstep.append(backend.pad_id)
                continue
            gens[d].append(nxt)
            views[d].append(int(cg.lead[nxt]) if cg is not None and j == 0 else nxt)
            fstep.append(views[d][-1])
            if cg is not None:
                cstates[d], spaced[d] = cg.state_after(cstates[d], nxt), cg.ends_in_space(nxt)
            if prop is not None:
                enforced[d][len(gens[d]) - 1] = prop
            step.append(nxt)
            keep.append(r)
        if not keep or j == max_new_tokens - 1:
            break
        tokens = torch.tensor(step, device=dev)
        ftokens = torch.tensor(fstep, device=dev)
        if can_compact and nb - len(keep) >= max(1, int(compact_frac * nb)):
            sel = keep + [nb + r for r in keep] if free else keep
            sel_t = torch.tensor(sel, device=dev)
            past.batch_select_indices(sel_t)
            mask, next_pos = mask[sel_t], next_pos[sel_t]
            keep_t = torch.tensor(keep, device=dev)
            tokens, ftokens = tokens[keep_t], ftokens[keep_t]
            active = [active[r] for r in keep]
            alive = [True] * len(active)
        step_ids = torch.cat([tokens, ftokens]) if free else tokens
        mask = torch.cat([mask, torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=dev)], 1)
        out = model(input_ids=step_ids[:, None], attention_mask=mask, position_ids=next_pos[:, None],
                    past_key_values=past, use_cache=True)
        past, last = out.past_key_values, out.logits[:, -1, :]
        del out
        next_pos = next_pos + 1
    return list(zip(gens, enforced))


# ------------------------------------------------------------------------------------------------- scoring
@dataclass
class Scores:
    """The verifier's pass over one text (start token + ids), per position t (row t scores ids[t])."""
    l_tok: torch.Tensor      # [T] logit of the token present
    rank0: torch.Tensor      # [T] number of tokens with a strictly larger logit
    top: torch.Tensor        # [T, k+1] largest logits, descending
    p_tok: torch.Tensor      # [T] probability of the token present
    top_p: torch.Tensor      # [T, k+1] largest probabilities, descending
    first: torch.Tensor | None = None   # [T] the first choice (token id)


@torch.no_grad()
def score_batch(backend, id_lists: list[list[int]], k: int, token_budget: int = 16384) -> list[Scores]:
    """The verification pass for many texts, batched by length under a token budget."""
    for ids in id_lists:
        if len(ids) + 1 > backend.max_positions:
            raise ValueError(f"text of {len(ids)} tokens exceeds the model context of {backend.max_positions}")
    dev = backend.device
    out: list[Scores | None] = [None] * len(id_lists)
    order = sorted(range(len(id_lists)), key=lambda i: -len(id_lists[i]))
    i = 0
    while i < len(order):
        L = len(id_lists[order[i]]) + 1
        n = max(1, min(len(order) - i, token_budget // max(1, L)))
        group = order[i:i + n]
        i += n
        inp = torch.full((len(group), L), backend.pad_id, dtype=torch.long)
        for r, d in enumerate(group):
            seq = [backend.start_id] + id_lists[d]
            inp[r, :len(seq)] = torch.tensor(seq)
        split = getattr(backend, "split", None)            # head over the hidden state, chunk by chunk (backend_hf)
        if split is not None:
            hidden, logits, step = split[0](input_ids=inp.to(dev), use_cache=False).last_hidden_state, None, backend.HEAD_ROWS
        else:
            hidden, logits, step = None, backend.model(input_ids=inp.to(dev), use_cache=False).logits, 2048
        bad = 0
        for r, d in enumerate(group):
            T = len(id_lists[d])
            parts: list[tuple] = []
            tgt_all = torch.tensor(id_lists[d], device=dev)[:, None]
            for a in range(0, T, step):                    # bounded memory: one chunk of vocabulary-wide rows at a time
                rows = (split[1](hidden[r, a:min(T, a + step)]) if split is not None
                        else logits[r, a:min(T, a + step)]).float()
                # probabilities as the single pass computes them (logsumexp, no full softmax); a non-finite logit
                # (+inf or NaN in the logsumexp, -inf or NaN in the minimum) is a precision overflow
                lse = torch.logsumexp(rows, -1, keepdim=True)
                if not bool((torch.isfinite(lse) & torch.isfinite(rows.amin(-1, keepdim=True))).all()):
                    bad += 1
                    break
                tgt = tgt_all[a:a + rows.shape[0]]
                l_tok = rows.gather(1, tgt)
                top_l = torch.topk(rows, k + 1, dim=-1).values
                parts.append((l_tok.squeeze(1).cpu(), (rows > l_tok).sum(1).cpu(), top_l.cpu(),
                              torch.exp(l_tok - lse).squeeze(1).cpu(), torch.exp(top_l - lse).cpu(),
                              rows.argmax(-1).cpu()))
                del rows, lse
            if bad:
                raise FloatingPointError(f"non-finite logits in the verification pass ({backend.environment['class']}): "
                                         f"precision overflow")
            if not parts:
                z = torch.zeros(0)
                out[d] = Scores(z, z.long(), torch.zeros(0, k + 1), z, torch.zeros(0, k + 1), z.long())
                continue
            out[d] = Scores(*(torch.cat([p[i] for p in parts]) for i in range(6)))
        del logits, hidden
        tidy = getattr(backend, "tidy", None)              # the group's vocabulary-wide rows back to the system (MPS)
        if tidy is not None:
            tidy()
    return out


def stats_at(offs, char_offsets: list[int], sc: Scores) -> list[PositionStat | None]:
    """HFBackend.position_stats from a batched pass: for each canonical char offset, the token covering it."""
    cover: dict[int, int] = {}
    for t, (a, b) in enumerate(offs):
        for c in range(a, b):
            cover.setdefault(c, t)
    res: list[PositionStat | None] = []
    for c in char_offsets:
        t = cover.get(c)
        if t is None:
            res.append(None)
            continue
        res.append(PositionStat(t, (int(offs[t][0]), int(offs[t][1])), float(sc.p_tok[t]), int(sc.rank0[t]),
                                [float(x) for x in sc.top_p[t]], float(sc.l_tok[t]),
                                tuple(float(x) for x in sc.top[t])))
    return res


def _batch_env(backend, n: int) -> dict:
    return {**dict(getattr(backend, "environment", {})), "batch": n}


# ------------------------------------------------------------------------------------------ seal / verify
def seal_batch(backend, prompts: list, master, params: SealParams = SealParams(),
               doc_ids: list[str] | None = None, seed: int | None = None, token_budget: int = 16384,
               meter=None) -> list[Sealed]:
    """seal.seal for many prompts at once (one generation batch, one batched verification pass).
    ``master``: one master key, or one per prompt (documents of different issuers in one batch).
    ``meter``: a second backend of the same model that makes the record's measurement pass (e.g. generate in
    bf16, measure in fp16); the record then reports the meter's environment. Default: ``backend``."""
    t0 = time.time()
    check_params(params)
    meter = meter or backend
    if (meter.weights_digest, meter.tokenizer_digest) != (backend.weights_digest, backend.tokenizer_digest):
        raise ValueError("the measuring backend must run the generating model")
    doc_ids = doc_ids or [crypto.new_doc_id() for _ in prompts]
    masters = list(master) if isinstance(master, (list, tuple)) else [master] * len(prompts)
    _, _, high_margin = margins(params)
    outs = generate_batch(backend, prompts, [decider(params, m, d) for m, d in zip(masters, doc_ids)],
                          params.max_new_tokens,
                          params.temperature, params.top_p, params.k, params.high_window, seed,
                          params.min_new_tokens, high_margin, params.align, params.free_after, gate=params.gate,
                          guard=params.guard, canon_gen=params.canon_gen)
    ms = [measure(backend, ids, enf) for ids, enf in outs]
    encs = [meter.tok(m.ctext, return_offsets_mapping=True, add_special_tokens=False) for m in ms]
    scores = score_batch(meter, [e["input_ids"] for e in encs], params.k, token_budget)
    env = _batch_env(meter, len(prompts))
    per = (time.time() - t0) / max(1, len(prompts))
    sealed = [finish(meter, params, m, stats_at(e["offset_mapping"], m.anchors, sc), mk, d, env=env)
              for m, e, sc, mk, d in zip(ms, encs, scores, masters, doc_ids)]
    return [Sealed(**{**s.__dict__, "seconds": round(per, 3)}) for s in sealed]


def verify_batch(backend, candidates: list[str], tokens: list[str], keys: list[bytes],
                 token_budget: int = 16384) -> list[Result]:
    """verify.verify for many (candidate, token, key) triples; one batched verification pass."""
    t0 = time.time()
    env = _batch_env(backend, len(candidates))
    opened = [open_record(backend, c, t, k, env=env, t0=t0) for c, t, k in zip(candidates, tokens, keys)]
    todo = [i for i, o in enumerate(opened) if not isinstance(o, Result) and o.offsets]
    encs = {i: backend.tok(opened[i].ctext, return_offsets_mapping=True, add_special_tokens=False) for i in todo}
    ks = {opened[i].rec.header.k for i in todo}
    scores: dict[int, Scores] = {}
    for kk in ks:                                  # records of one batch normally share k
        grp = [i for i in todo if opened[i].rec.header.k == kk]
        for i, sc in zip(grp, score_batch(backend, [encs[i]["input_ids"] for i in grp], kk, token_budget)):
            scores[i] = sc
    res: list[Result] = []
    for i, o in enumerate(opened):
        if isinstance(o, Result):
            res.append(o)
        elif not o.offsets:
            res.append(judge(o, [], t0))
        else:
            res.append(judge(o, stats_at(encs[i]["offset_mapping"], o.offsets, scores[i]), t0))
    per = round((time.time() - t0) / max(1, len(res)), 3)
    for r in res:
        r.seconds = per                            # amortized over the batch
    return res


def detect_batch(backend, candidates: list[str], k_wm: bytes, *, interval: int = 5, k: int = 1, context: int = 4,
                 tol: float = 0.0, alpha: float = 1e-6, localize: bool = False, min_run: int = 8,
                 token_budget: int = 16384, gate: float = 0.0, canon: str | None = None,
                 restarts: int = 32) -> list[Detection]:
    """detect.detect for many candidates under one detection key; one batched pass (with ``localize``, one more
    over the paragraph-start readings of all candidates, as detect.detect reads them)."""
    from .canon import CANON_ID
    t0 = time.time()
    toks = [detection_tokens(backend, c, canon or CANON_ID) for c in candidates]
    todo = [i for i, (_, _, n) in enumerate(toks) if n > context]
    scores = dict(zip(todo, score_batch(backend, [toks[i][0] for i in todo], k, token_budget)))
    kw = dict(interval=interval, k=k, context=context, tol=tol, alpha=alpha, min_run=min_run, gate=gate)
    dets = detect_from_scores(toks, scores, k_wm, localize=localize, t0=t0, backend=backend, **kw)
    if localize and restarts:
        readings = [readings_of(backend, c, canon or CANON_ID, restarts) if toks[i][2] > context else []
                    for i, c in enumerate(candidates)]
        fold_all(dets, readings, reading_scores(backend, readings, k, token_budget), k_wm, backend=backend, **kw)
        for d in dets:
            d.seconds = round((time.time() - t0) / max(1, len(dets)), 3)
    return dets


def reading_scores(backend, readings: list, k: int, token_budget: int = 16384) -> list[list]:
    """The verifier's pass over every paragraph-start reading (see detect.readings_of), batched; per candidate,
    per reading: Scores, or None for a reading too short to hold a checkpoint."""
    flat = [(i, j) for i, rs in enumerate(readings) for j, (_, ids, _, _) in enumerate(rs) if len(ids) > 4]
    got = dict(zip(flat, score_batch(backend, [readings[i][j][1] for i, j in flat], k, token_budget)))
    return [[got.get((i, j)) for j in range(len(rs))] for i, rs in enumerate(readings)]


def fold_all(dets: list, readings: list, scores: list, k_wm: bytes, *, interval: int, k: int, context: int,
             tol: float, alpha: float, min_run: int, gate: float = 0.0, backend=None) -> list:
    """Fold each candidate's paragraph-start readings into its localized detection under the key ``k_wm``
    (the readings' pass is key-independent: made once, folded under many keys)."""
    for d, rs, scs in zip(dets, readings, scores):
        views = []
        for (shift, ids, offs, n), sc in zip(rs, scs):
            if sc is None:
                continue
            dv = detect_from_scores([(ids, offs, n)], {0: sc}, k_wm, interval=interval, k=k, context=context,
                                    tol=tol, alpha=alpha, localize=True, min_run=min_run, t0=time.time(), gate=gate,
                                    backend=backend)[0]
            views.append((shift, dv))
        fold_readings(d, views, alpha)
    return dets


def detect_from_scores(toks, scores: dict, k_wm: bytes, *, interval: int, k: int, context: int, tol: float,
                       alpha: float, localize: bool, min_run: int, t0: float, gate: float = 0.0,
                       backend=None) -> list[Detection]:
    """Detection decisions from a pass already made (the same pass can be scored under many keys: the
    empirical null over many keys costs one model pass per text). ``backend``: skip the positions where a
    canonically generating sealer could not have emitted the first choice (detect.unsatisfiable)."""
    out: list[Detection] = []
    for i, (ids, offs, n_tok) in enumerate(toks):
        if i not in scores:
            out.append(Detection(False, 1.0, 0, 0, 0, n_tok, alpha, k, interval, context, 0.0, gate=gate))
            continue
        sc = scores[i]
        cps = checkpoints(ids, k_wm, interval, context)
        at = [t for t, _ in cps]
        unsat = (unsatisfiable(backend, ids, at, sc.first[at].tolist())
                 if backend is not None and sc.first is not None else None)
        out.append(decide_counts(cps, offs, n_tok, sc.rank0[at].tolist(), sc.l_tok[at].tolist(),
                                 sc.top[at].tolist(), interval=interval, k=k, context=context, tol=tol,
                                 alpha=alpha, localize=localize, min_run=min_run, t0=t0,
                                 p1s=sc.top_p[at, 0].tolist() if gate else None, gate=gate, unsat=unsat))
    per = round((time.time() - t0) / max(1, len(out)), 3)
    for d in out:
        d.seconds = per                            # amortized over the batch
    return out
