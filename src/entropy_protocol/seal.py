"""Sealing = constrained generation + canonical measurement.

The record is measured the way the verifier will measure it: on the canonical form of the final
text, re-read by the same model with no prompt, checkpoints anchored by character offset. Only
checkpoints whose enforced property holds there (with at least twice the calibrated tolerance of
margin) enter the record. With a calibration table, the margin must cover the largest tolerance
among the configuration classes the record is certified for.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace

from . import ENGINE, crypto, plain
from .canon import CANON_ID, block_digest, canonicalize_with_map, segment
from .predicates import holds, stat_margin
from .record import HIGH, LOW, Header, Record, dumps


@dataclass(frozen=True)
class SealParams:
    # Reference operating point (measured on Qwen2.5-0.5B, 40-60 prompts, calibrated fp32+fp16):
    # one checkpoint in 5 tokens chosen by the detection key from the preceding 4 tokens; LOW = the first choice,
    # HIGH = not the first choice (k = 1, among ranks 1-3); enforced in the verifier's context (align="free")
    # without a HIGH margin. Perplexity under an independent judge x1.12 [0.97-1.29] (enforcing in the
    # generation context: x1.17), record-free detection at 200 tokens: 90 % at FPR 1e-6 (generation context:
    # 33 %). Profiles: high_margin=None ("record strength": 75 % of checkpoints survive the fp16 tolerance,
    # x1.22); selection="interval", align="prompt" (the claim-1 embodiment). k = 10 costs x1.84.
    k: int = 1
    interval: int = 5
    high_window: int = 3
    temperature: float = 0.8
    top_p: float = 0.9
    max_new_tokens: int = 200
    tol: float = 0.0
    seal_margin_factor: float = 2.0
    min_entries: int = 8
    indet_frac: float = 0.2
    tol_space: str = "logit"
    cal: dict = field(default_factory=dict)      # configuration class -> tolerance, in tol_space units
    low_share: float = 0.5                       # share of checkpoints assigned the first predicate (LOW)
    min_new_tokens: int = 0
    # HIGH selects only tokens this far (logits) below the k-th; None = automatic: the seal's required margin
    # when predicates are enforced in the verifier's context, else 0
    high_margin: float | None = 0.0
    # "content": checkpoints and predicates chosen by a PRF of the preceding `context` token ids under the
    # detection key -> the text can also be detected without its record (detect.py); "interval": every
    # `interval`-th generated position, predicate from the per-document keyed bitstream
    selection: str = "content"
    context: int = 4
    # where predicates are enforced: "prompt" (generation context) or "free" (the verifier's context:
    # start token + text, no prompt) — see backend_hf.generate
    align: str = "free"
    free_after: int = 0      # with align="free": positions before this index are enforced in the generation context
    # confidence gate (align="free"): checkpoints only where the verifier-context first choice has probability
    # below this; 0 = off. Key-independent, so record-free detection keeps its exact null on gated positions
    gate: float = 0.0
    # generation-side guard: no checkpoint where the generation context (with the prompt) is at least this sure;
    # invisible to a detector (such positions only dilute record-free detection), 0 = off
    guard: float = 0.0
    # canonical generation: never emit a character the canonical form would change (canongen.py), so verifier and
    # detector re-read exactly the enforced tokens; a word processor's substitutions are folded back on paste
    canon_gen: bool = True
    # visible-text layer (plain.py): the record also carries the digest and blocks of the text as a reader sees it
    # rendered, so a copy out of a chat window verifies as VERIFIED_TEXT; written when markdown-it-py is present
    plain_layer: bool = True


@dataclass(frozen=True)
class Sealed:
    text: str
    token: str          # custody token, encrypted under K_log of this document
    doc_id: str
    tokens: int
    enforced: int
    entries: int
    dropped_anchor: int
    dropped_rank: int
    dropped_margin: int
    seconds: float
    dropped_gate: int = 0   # enforced below the gate at generation, at or above it in the verifier's pass


@dataclass(frozen=True)
class _Measured:
    """A generated text on its way into a record: canonical form and the anchors of its enforced positions."""
    raw: str
    ctext: str
    anchors: list
    props: list
    dropped_anchor: int
    tokens: int
    enforced: int


def check_params(params: SealParams) -> None:
    if params.selection not in ("content", "interval"):
        raise ValueError(f"unknown checkpoint selection {params.selection!r}")
    if params.selection == "content" and params.low_share != 0.5:
        raise ValueError("content selection needs unbiased predicates (low_share = 0.5): the detector's exact "
                         "null distribution relies on it")
    if params.gate and (params.align != "free" or not 0 < params.gate < 1):
        raise ValueError("the confidence gate is a probability in (0, 1), read in the verifier's context (align='free')")


def first_checkpoint(params: SealParams) -> int:
    """The first token index at which these parameters can put a checkpoint (``decider``; the detector reads the
    same positions): a calibration table must cover every position from here on."""
    return params.context if params.selection == "content" else params.interval


def table_covers(cal: dict, params: SealParams) -> str | None:
    """None if a calibration file (``entropy calibrate``) measured every position these parameters can use,
    else the reason (a table without ``from_position`` measured all positions)."""
    first = cal.get("from_position", 0)
    if first > first_checkpoint(params):
        return f"calibration measured positions from {first} on; these parameters put checkpoints from " \
               f"{first_checkpoint(params)} on"
    return None


def decider(params: SealParams, master: bytes, doc_id: str):
    """``decide(j, gen)`` for generation: None, LOW or HIGH at generated position j."""
    k_bits = crypto.bits_key(master)
    k_wm = crypto.wm_key(master)

    def decide(j: int, gen):
        if params.selection == "content":
            if j < params.context:          # the context would reach into the prompt, invisible to a detector
                return None
            return crypto.wm_decide(k_wm, gen[j - params.context:j], params.interval)
        if j == 0 or j % params.interval:
            return None
        return LOW if crypto.bit(k_bits, doc_id, j, params.low_share) == 0 else HIGH
    return decide


def margins(params: SealParams) -> tuple[dict, float, float]:
    """(calibration table, margin a checkpoint needs to enter the record, HIGH margin used at generation)."""
    cal = {str(c): float(v) for c, v in params.cal.items()}
    need = params.seal_margin_factor * (max(cal.values()) if cal else params.tol)
    # enforced in the verifier's own context, a HIGH token can be chosen with the margin the filter below
    # will demand, so HIGH checkpoints are not lost to the tolerance
    high_margin = params.high_margin if params.high_margin is not None else (need if params.align == "free" else 0.0)
    return cal, need, high_margin


def measure(backend, ids: list[int], enforced: dict[int, int]) -> _Measured:
    """Canonical text of the generated ids and the canonical character anchor of each enforced position."""
    raw, spans = backend.decode_with_spans(ids)
    ctext, cmap = canonicalize_with_map(raw)
    anchors: list[int] = []
    props: list[int] = []
    dropped_anchor = 0
    for j, prop in sorted(enforced.items()):
        a, b = spans[j]
        c = next((cmap[r] for r in range(a, b) if r in cmap and not raw[r].isspace()), None)
        if c is None:
            c = next((cmap[r] for r in range(a, b) if r in cmap), None)
        if c is None:
            dropped_anchor += 1
            continue
        anchors.append(c)
        props.append(prop)
    return _Measured(raw, ctext, anchors, props, dropped_anchor, len(ids), len(enforced))


def finish(backend, params: SealParams, m: _Measured, stats, master: bytes, doc_id: str,
           salt: bytes | None = None, nonce: bytes | None = None, env: dict | None = None,
           t0: float | None = None) -> Sealed:
    """Keep the checkpoints that hold in the verifier's pass with the required margin (and, with a confidence gate,
    lie below the gate in that pass, where the detector reads it); write the record."""
    cal, need, _ = margins(params)
    entries: list[tuple[int, int]] = []
    seen: set[int] = set()
    dropped_anchor = m.dropped_anchor
    dropped_rank = dropped_margin = dropped_gate = 0
    for c, prop, st in zip(m.anchors, m.props, stats):
        if st is None or st.token_index in seen:
            dropped_anchor += 1
            continue
        if params.gate and st.top_sorted[0] >= params.gate:
            dropped_gate += 1                    # the generation-time view differed (e.g. a stripped leading space)
            continue
        if not holds(prop, st.rank0, params.k):
            dropped_rank += 1
            continue
        if need > 0 and stat_margin(prop, st, params.k, params.tol_space) < need:
            dropped_margin += 1
            continue
        seen.add(st.token_index)
        entries.append((c, prop))

    header = Header(model=backend.model_ref, weights_digest=backend.weights_digest,
                    tokenizer_digest=backend.tokenizer_digest, canon=CANON_ID, context="bos+text",
                    k=params.k, interval=params.interval, tol=params.tol, min_entries=params.min_entries,
                    indet_frac=params.indet_frac, text_digest=crypto.sha256_hex(m.ctext), engine=ENGINE,
                    tol_space=params.tol_space, cal=cal,
                    components=dict(getattr(backend, "components", {})),
                    env=dict(env if env is not None else getattr(backend, "environment", {})),
                    selection=params.selection, wm_context=params.context if params.selection == "content" else 0,
                    salt=(salt or os.urandom(16)).hex(), wm_gate=params.gate)
    blocks = tuple((s, e, block_digest(m.ctext[s:e])) for s, e in segment(m.ctext))
    pblocks: tuple = ()
    if params.plain_layer and plain.available():
        ptext, pspans = plain.segment_source(m.ctext)
        header = replace(header, plain=plain.PLAIN_ID, plain_digest=crypto.sha256_hex(ptext))
        pblocks = tuple((s, e, block_digest(ptext[s:e])) for s, e in pspans)
    token = crypto.encrypt_token(crypto.log_key(master, doc_id), doc_id,
                                 dumps(Record(header, tuple(entries), blocks, pblocks)), nonce=nonce)
    return Sealed(text=m.raw, token=token, doc_id=doc_id, tokens=m.tokens, enforced=m.enforced,
                  entries=len(entries), dropped_anchor=dropped_anchor, dropped_rank=dropped_rank,
                  dropped_margin=dropped_margin, seconds=round(time.time() - t0, 3) if t0 is not None else 0.0,
                  dropped_gate=dropped_gate)


def seal(backend, prompt: str, master: bytes, params: SealParams = SealParams(),
         doc_id: str | None = None, seed: int | None = None, salt: bytes | None = None,
         nonce: bytes | None = None) -> Sealed:
    """``salt`` and ``nonce`` are random unless given (test vectors)."""
    t0 = time.time()
    doc_id = doc_id or crypto.new_doc_id()
    check_params(params)
    _, _, high_margin = margins(params)
    ids, enforced = backend.generate(prompt, decider(params, master, doc_id), params.max_new_tokens,
                                     params.temperature, params.top_p, params.k, params.high_window, seed,
                                     params.min_new_tokens, high_margin, params.align, params.free_after,
                                     gate=params.gate, guard=params.guard, canon_gen=params.canon_gen)
    m = measure(backend, ids, enforced)
    stats = backend.position_stats(m.ctext, m.anchors, params.k) if m.anchors else []
    return finish(backend, params, m, stats, master, doc_id, salt=salt, nonce=nonce, t0=t0)
