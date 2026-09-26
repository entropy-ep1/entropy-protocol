"""Record-free detection (watermark mode): is this text sealed under the key, without its record?

Sealing with content selection forces, at positions a PRF of the preceding token ids picks under the detection
key K_wm, the predicate the PRF also picks: LOW (the model's first choice) or HIGH (not in its top-k).
The detector re-reads the canonical text with the declared model (BOS + text, one pass, as verification does),
recomputes the PRF at every position and counts, over distinct contexts, how often the predicate holds.

Exact null. For text not generated under K_wm, the PRF's predicate is independent of the text, and exactly
one of LOW / HIGH holds at any position (k = 1: LOW = first choice, HIGH = not first choice). So each decisive
checkpoint matches with probability exactly 1/2, independently across distinct contexts, whatever the text or
model: matches ~ Binomial(n, 1/2), and the p-value below is exact, not an approximation or a calibration.
(For k > 1 a position can satisfy neither predicate; the null is then Binomial(n, 1/2) only as an upper bound
if at least one holds, so the reference profile uses k = 1.)
Positions whose margin is below the detector's calibrated tolerance are excluded before the bit is looked at:
for k = 1 the LOW and HIGH margins coincide, so the exclusion cannot depend on the key.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch

from . import crypto
from .canon import CANON_ID, canonicalize
from .predicates import holds, margin
from .stats import binomial_tail  # noqa: F401  (re-exported: detect.binomial_tail stays importable)


def first_choice_prob(logits):
    from .backend_hf import first_choice_prob as f        # torch-only helper; the backend module holds it
    return f(logits)


@dataclass
class Detection:
    detected: bool
    p_value: float
    matched: int
    checkpoints: int                 # decisive checkpoints over distinct contexts
    excluded: int                    # below the tolerance (near ties), not counted
    tokens: int
    alpha: float
    k: int
    interval: int
    context: int
    seconds: float = 0.0
    # localized search (mixed documents): the run of consecutive checkpoints with the smallest p-value,
    # Bonferroni-corrected by the number of runs examined; canonical character range of that run
    segment: list | None = None
    segment_p: float | None = None
    segment_p_corrected: float | None = None
    segment_detected: bool | None = None      # corrected p <= alpha; a separate decision from `detected`
    segment_windows: int | None = None        # windows examined (over every reading), the Bonferroni factor
    restarts: int = 0                         # readings from a paragraph start (see detect)
    gate: float = 0.0                         # confidence gate applied (0 = none)
    gated: int = 0                            # checkpoints skipped by the gate (the model was confident there)
    unsat: int = 0                            # skipped: canonical generation could not emit the first choice there
    canon: str = CANON_ID                     # canonical form the detector read the text in

    def to_dict(self) -> dict:
        return asdict(self)




def best_run(matches: list[int], min_len: int = 8) -> tuple[int, int, float, int]:
    """Dyadic scan: windows of min_len, 2*min_len, 4*min_len, ... checkpoints, each moved in half-window
    steps. Returns (first, last, p, windows examined) for the window with the smallest exact p-value; the
    caller corrects by the number of windows (Bonferroni), which stays small: about n / min_len."""
    n = len(matches)
    pre = [0]
    for x in matches:
        pre.append(pre[-1] + x)
    best = (0, n - 1, 1.0)
    windows = 0
    length = min_len
    while length <= n:
        step = max(1, length // 2)
        starts = list(range(0, n - length + 1, step))
        if starts[-1] != n - length:
            starts.append(n - length)                      # the last window ends at the last checkpoint
        for i in starts:
            windows += 1
            p = binomial_tail(length, pre[i + length] - pre[i])
            if p < best[2]:
                best = (i, i + length - 1, p)
        length *= 2
    return best[0], best[1], best[2], windows


def detection_tokens(backend, candidate: str, canon: str = CANON_ID) -> tuple[list[int], list, int]:
    """Canonical token ids (cut to the model context), their character offsets, and the uncut token count."""
    ctext = canonicalize(candidate, canon)
    enc = backend.tok(ctext, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    return ids[: backend.max_positions - 1], offs, len(ids)


def unsatisfiable(backend, ids: list[int], positions: list[int], first: list[int]) -> list[bool] | None:
    """Positions where a canonically generating sealer could not have emitted the verifier's first choice (see
    CanonGuard.unsatisfiable); None for a backend without canonical generation."""
    guard = getattr(backend, "canon_guard", None)
    if guard is None:
        return None
    return guard.unsatisfiable(ids, positions, first, backend.stop_ids)


def checkpoints(ids: list[int], k_wm: bytes, interval: int, context: int) -> list[tuple[int, int]]:
    """(token index, predicate) at every distinct context where the detection key selects a checkpoint."""
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, int]] = []
    for t in range(context, len(ids)):
        ctx = tuple(ids[t - context:t])
        if ctx in seen:
            continue
        seen.add(ctx)
        prop = crypto.wm_decide(k_wm, ctx, interval)
        if prop is not None:
            out.append((t, prop))
    return out


def decide_counts(cps, offs, n_tok: int, rank0s, ltoks, tops, *, interval: int, k: int, context: int, tol: float,
                  alpha: float, localize: bool, min_run: int, t0: float, p1s=None, gate: float = 0.0,
                  unsat=None) -> Detection:
    """Counts and the exact p-value from the scoring pass at the checkpoints ``cps`` (rank of the token present,
    its logit and the k+1 largest logits at each; ``p1s``: the first choice's probability there, for the gate;
    ``unsat``: per checkpoint, canonical generation could not have emitted the first choice there).
    All exclusions — the confidence gate, unsatisfiable positions and near ties — are read from the text and the
    model, never from the key, so the counted checkpoints keep the exact Binomial(n, 1/2) null."""
    n = m = excluded = gated = skipped = 0
    trail: list[tuple[int, int]] = []                            # (token index, match) of decisive checkpoints
    flags = unsat if unsat is not None else [False] * len(cps)
    for (t, prop), rank0, l_tok, top, p1, u in zip(cps, rank0s, ltoks, tops, p1s if gate else [0.0] * len(cps),
                                                  flags):
        if gate and p1 >= gate:
            gated += 1
            continue
        if u:
            skipped += 1
            continue
        if tol > 0 and margin(prop, rank0, l_tok, top, k) < tol:
            excluded += 1
            continue
        hit = int(holds(prop, rank0, k))
        n += 1
        m += hit
        trail.append((t, hit))
    p = binomial_tail(n, m)
    det = Detection(p <= alpha, p, m, n, excluded, n_tok, alpha, k, interval, context, 0.0, gate=gate, gated=gated,
                    unsat=skipped)
    if localize and len(trail) >= min_run:
        i, j, sp, runs = best_run([h for _, h in trail], min_run)
        det.segment = [int(offs[trail[i][0]][0]), int(offs[trail[j][0]][1])]
        det.segment_p = sp
        det.segment_p_corrected = min(1.0, sp * runs)
        det.segment_detected = det.segment_p_corrected <= alpha
        det.segment_windows = runs
    det.seconds = round(time.time() - t0, 3)
    return det


def _read(backend, ids: list[int], offs, n_tok: int, k_wm: bytes, *, interval: int, k: int, context: int,
          tol: float, alpha: float, localize: bool, min_run: int, gate: float, t0: float) -> Detection:
    """One reading: the verifier's pass over start token + ids, and the counts at the key's checkpoints."""
    if n_tok <= context:
        return Detection(False, 1.0, 0, 0, 0, n_tok, alpha, k, interval, context, round(time.time() - t0, 3),
                         gate=gate)
    inp = torch.tensor([[backend.start_id] + ids], device=backend.device)
    cps = checkpoints(ids, k_wm, interval, context)
    rows_at = backend.logits_at(inp, [t for t, _ in cps])     # only the checkpoints' rows; row t predicts ids[t]
    rank0s, ltoks, tops, p1s, first = [], [], [], [], []
    for (t, _), row in zip(cps, rows_at):
        rank0s.append(int((row > row[ids[t]]).sum()))
        ltoks.append(float(row[ids[t]]))
        tops.append(torch.topk(row, k + 1).values.tolist() if tol > 0 else [])
        p1s.append(float(first_choice_prob(row)) if gate else 0.0)
        first.append(int(row.argmax()))
    unsat = unsatisfiable(backend, ids, [t for t, _ in cps], first)
    return decide_counts(cps, offs, n_tok, rank0s, ltoks, tops, interval=interval, k=k, context=context, tol=tol,
                         alpha=alpha, localize=localize, min_run=min_run, t0=t0, p1s=p1s, gate=gate, unsat=unsat)


def paragraph_starts(ctext: str, limit: int) -> list[int]:
    """Character offsets where a paragraph (after a blank line) begins, at most ``limit`` of them."""
    return [m.end() for m in re.finditer(r"\n[ \t]*\n\s*", ctext)][:limit]


def readings_of(backend, candidate: str, canon: str = CANON_ID, limit: int = 32,
                span: int = 1024) -> list[tuple[int, list, list, int]]:
    """The readings of a text from each paragraph start, as texts of their own: (character offset of the reading
    in the canonical text, token ids, their offsets, the token count). A reading covers at most ``span`` tokens
    (a pasted passage lies within it), so a long document costs readings x span, not readings x its length."""
    ctext = canonicalize(candidate, canon)
    out = []
    for r in paragraph_starts(ctext, limit):
        sub = ctext[r:]
        view = canonicalize(sub, canon)
        if not view:
            continue
        enc = backend.tok(view, return_offsets_mapping=True, add_special_tokens=False)
        n = min(len(enc["input_ids"]), span, backend.max_positions - 1)
        out.append((r + sub.index(view[: min(len(view), 16)]), enc["input_ids"][:n], enc["offset_mapping"][:n], n))
    return out


def fold_readings(det: Detection, readings: list[tuple[int, Detection]], alpha: float) -> Detection:
    """Fold the localized detections of the paragraph-start readings (offset, detection) into ``det``: the best
    window over all readings, corrected by the number of windows examined in all of them."""
    best_p, best_seg, windows = det.segment_p, det.segment, det.segment_windows or 0
    for shift, d in readings:
        windows += d.segment_windows or 0
        if d.segment_p is not None and (best_p is None or d.segment_p < best_p):
            best_p, best_seg = d.segment_p, [d.segment[0] + shift, d.segment[1] + shift]
    if best_p is not None:
        det.segment, det.segment_p = best_seg, best_p
        det.segment_p_corrected = min(1.0, best_p * max(1, windows))
        det.segment_detected = det.segment_p_corrected <= alpha
    det.segment_windows, det.restarts = windows, len(readings)
    return det


@torch.no_grad()
def detect(backend, candidate: str, k_wm: bytes, *, interval: int = 5, k: int = 1, context: int = 4,
           tol: float = 0.0, alpha: float = 1e-6, localize: bool = False, min_run: int = 8,
           gate: float = 0.0, canon: str = CANON_ID, restarts: int = 32) -> Detection:
    """``canon``: the canonical form of the sealing deployment (a detector has no record to read it from).
    With ``localize``, the search also reads the text from each paragraph start (up to ``restarts`` of them) as a
    text of its own, start token first: a sealed passage pasted into a longer document is then read in the context
    it was sealed in, not after the document's other paragraphs. The best window over all readings is corrected
    by the number of windows examined in all of them (Bonferroni holds under any dependence between readings)."""
    t0 = time.time()
    ids, offs, n_tok = detection_tokens(backend, candidate, canon)
    kw = dict(interval=interval, k=k, context=context, tol=tol, alpha=alpha, min_run=min_run, gate=gate, t0=t0)
    det = _read(backend, ids, offs, n_tok, k_wm, localize=localize, **kw)
    det.canon = canon
    if not localize or not restarts or n_tok <= context:
        return det
    views = [(shift, _read(backend, v_ids, v_offs, n, k_wm, localize=True, **kw))
             for shift, v_ids, v_offs, n in readings_of(backend, candidate, canon, restarts)]
    fold_readings(det, views, alpha)
    det.seconds = round(time.time() - t0, 3)
    return det
