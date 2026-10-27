"""Verification: open the record, canonicalize the candidate, compare the content layer, re-read the
text with the same model and check each checkpoint of the model layer. Determinations, not probabilities.

Two layers in one record:
  content layer  canonical-text digest + per-block digests  -> exact integrity and block localization
  model layer    checkpoints (char offset, ordinal property) -> binds the text to the declared model,
                 localizes changes at character level; carries no text
Verdicts (ARCHITECTURE §6):
  NOT_VERIFIABLE  the log token cannot be opened with the key (wrong key, altered token)
  VERIFIED        canonical text identical to the sealed text AND every decisive checkpoint matches
  MODIFIED        canonical text differs from the sealed text; changed blocks, deletion points and
                  diverging checkpoints are reported as canonical character ranges
  INDETERMINATE   the verifier's model/tokenizer differs, or the text is identical but the model
                  evaluation disagrees (environment outside the calibrated tolerance).
                  An identical text can never be reported as MODIFIED.
  VERIFIED_TEXT   the canonical text differs, but its visible text (plain.py: a rendered copy, e.g. out of a chat
                  window) is identical to the sealed text's; formatting is not covered and the model layer is not
                  evaluated on it. Only records with the visible-text layer; the candidate is taken literally.
Tolerance (ARCHITECTURE §6a): with a calibration table in the header the verifier applies the tolerance of its
own configuration class; a class absent from the table is evaluated with the largest listed tolerance and
reported as calibrated = False.
"""
from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

from . import ENGINE, crypto
from . import plain
from .canon import CANON_ID, SUPPORTED, block_digest, canonicalize, segment
from .predicates import holds, stat_margin
from .record import loads

VERIFIED, MODIFIED, INDETERMINATE, NOT_VERIFIABLE = "VERIFIED", "MODIFIED", "INDETERMINATE", "NOT_VERIFIABLE"
VERIFIED_TEXT = "VERIFIED_TEXT"


@dataclass
class Result:
    verdict: str
    reason: str = ""
    score: float | None = None               # model layer: matched / decisive checkpoints
    matched: int = 0
    divergent: int = 0
    indeterminate: int = 0
    entries: int = 0
    model_evidence: str = ""                  # "strong" | "weak" (fewer than min_entries matched) | ""
    divergence_ranges: list = field(default_factory=list)   # model layer, canonical char ranges
    changed_ranges: list = field(default_factory=list)      # content layer: candidate blocks not in the seal
    deletion_points: list = field(default_factory=list)     # content layer: where sealed blocks went missing
    first_divergence: int | None = None
    digest_match: bool | None = None
    calibrated: bool | None = None            # None: no calibration table; False: class outside the table
    candidate_digest: str = ""
    model: str = ""
    weights_digest: str = ""
    tokenizer_digest: str = ""
    decoding: dict = field(default_factory=dict)
    text_match: bool | None = None            # visible-text layer: None = not in the record; True = VERIFIED_TEXT
    text_view: dict | None = None             # MODIFIED with that layer: {text, changed_ranges, deletion_points}
    doc_id: str = ""
    nonce: str = ""
    engine: str = ENGINE
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _content_diff(ctext: str, sealed_blocks, canon: str = CANON_ID) -> tuple[list[list[int]], list[int]]:
    """Ordered alignment of the sealed block digests with the candidate's (longest matching runs first,
    like diff), so a repeated sentence is attributed to the copy that actually changed.
    Returns candidate ranges of changed or inserted blocks and candidate offsets where sealed blocks
    were deleted (the end of the preceding kept block, 0 at the start)."""
    return _diff_spans(ctext, segment(ctext, canon), sealed_blocks)


def _diff_spans(text: str, spans, sealed_blocks) -> tuple[list[list[int]], list[int]]:
    cand = [(s, e, block_digest(text[s:e])) for s, e in spans]
    sealed = [d for _, _, d in sealed_blocks]
    changed: list[list[int]] = []
    points: list[int] = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(a=sealed, b=[d for _, _, d in cand], autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            changed.extend([cand[j][0], cand[j][1]] for j in range(j1, j2))
        elif tag == "delete":
            points.append(cand[j1 - 1][1] if j1 > 0 else 0)
    return changed, sorted(set(points))


@dataclass
class Opened:
    """A candidate paired with its opened record, ready for the model layer: ``offsets`` are the canonical
    character offsets the verifier's pass must score (empty: no model evaluation needed)."""
    rec: object
    ctext: str
    tol: float
    base: dict
    offsets: list


def open_record(backend, candidate: str, token: str, key: bytes, nonce: str | None = None,
                env: dict | None = None, t0: float | None = None) -> Result | Opened:
    """Everything up to the model layer: open the record, canonicalize, choose the tolerance of this
    verifier's configuration class. Returns a final Result when no model evaluation is needed."""
    t0 = time.time() if t0 is None else t0
    nonce = nonce or os.urandom(8).hex()
    try:
        doc_id = crypto.token_doc_id(token)
        rec = loads(crypto.decrypt_token(key, token))
    except (crypto.TokenError, ValueError, KeyError, TypeError) as exc:
        return Result(verdict=NOT_VERIFIABLE, reason=str(exc), nonce=nonce, seconds=round(time.time() - t0, 3))
    return _prepare(backend, candidate, rec, doc_id, nonce, env)


def open_delivered_record(backend, candidate: str, delivered: bytes, verifier_private: bytes,
                          nonce: str | None = None, env: dict | None = None, t0: float | None = None) -> Result | Opened:
    """As open_record, for a record delivered to this verifier by public key (COSE-HPKE, delivery.py)."""
    from .cose import COSEError
    from .delivery import open_delivered
    t0 = time.time() if t0 is None else t0
    nonce = nonce or os.urandom(8).hex()
    try:
        doc_id, plaintext = open_delivered(delivered, verifier_private)
        rec = loads(plaintext)
    except (COSEError, ValueError, KeyError, TypeError) as exc:
        return Result(verdict=NOT_VERIFIABLE, reason=str(exc), nonce=nonce, seconds=round(time.time() - t0, 3))
    return _prepare(backend, candidate, rec, doc_id, nonce, env)


def _prepare(backend, candidate: str, rec, doc_id: str, nonce: str, env: dict | None) -> Result | Opened:
    h = rec.header
    if h.canon not in SUPPORTED or h.context != "bos+text":
        return Result(verdict=INDETERMINATE, reason="unsupported canonicalization or context", doc_id=doc_id,
                      nonce=nonce, model=h.model, entries=len(rec.entries))
    ctext = canonicalize(candidate, h.canon)
    cdigest = crypto.sha256_hex(ctext)
    cls = backend.config_class
    if h.cal:
        calibrated = cls in h.cal
        tol = float(h.cal[cls]) if calibrated else max(float(v) for v in h.cal.values())
    else:
        calibrated, tol = None, float(h.tol)
    base = dict(model=h.model, weights_digest=h.weights_digest, tokenizer_digest=h.tokenizer_digest, doc_id=doc_id,
                nonce=nonce, candidate_digest=cdigest, entries=len(rec.entries), digest_match=(cdigest == h.text_digest),
                calibrated=calibrated,
                decoding={"device": str(backend.device), "dtype": str(backend.dtype).replace("torch.", ""),
                          "class": cls, "canon": h.canon, "context": h.context, "k": h.k,
                          "tol_space": h.tol_space, "tol": tol,
                          "env": dict(env if env is not None else getattr(backend, "environment", {})),
                          "sealing_env": dict(h.env)})
    if h.weights_digest != backend.weights_digest or h.tokenizer_digest != backend.tokenizer_digest:
        return Result(verdict=INDETERMINATE, reason="the verifier's model or tokenizer differs from the sealing model", **base)
    return Opened(rec, ctext, tol, base, [c for c, _ in rec.entries])


def verify(backend, candidate: str, token: str, key: bytes, nonce: str | None = None) -> Result:
    t0 = time.time()
    return _finish(backend, open_record(backend, candidate, token, key, nonce, t0=t0), t0)


def verify_delivered(backend, candidate: str, delivered: bytes, verifier_private: bytes,
                     nonce: str | None = None) -> Result:
    """Verification with a record delivered by public key (COSE-HPKE): no secret shared with the issuer."""
    t0 = time.time()
    return _finish(backend, open_delivered_record(backend, candidate, delivered, verifier_private, nonce, t0=t0), t0)


def _finish(backend, o, t0: float) -> Result:
    if isinstance(o, Result):
        return o
    stats = backend.position_stats(o.ctext, o.offsets, o.rec.header.k) if o.offsets else []
    return judge(o, stats, t0)


def judge(o: Opened, stats, t0: float) -> Result:
    """The model layer and the verdict, from the verifier's pass statistics at the record's checkpoints."""
    rec, ctext, tol, base = o.rec, o.ctext, o.tol, o.base
    h = rec.header
    cls = base["decoding"]["class"]
    calibrated = base["calibrated"]
    matched = divergent = indet = 0
    ranges: list[list[int]] = []
    for (c, prop), st in zip(rec.entries, stats):
        if st is None:                          # cannot be aligned to the candidate -> divergence
            divergent += 1
            ranges.append([c, c + 1])
            continue
        if tol > 0 and stat_margin(prop, st, h.k, h.tol_space) < tol:
            indet += 1
            continue
        if holds(prop, st.rank0, h.k):
            matched += 1
        else:
            divergent += 1
            ranges.append(list(st.char_range))
    ranges.sort()
    decisive = matched + divergent
    evidence = ("strong" if matched >= h.min_entries else "weak") if rec.entries else ""
    common = dict(score=(matched / decisive if decisive else None), matched=matched, divergent=divergent,
                  indeterminate=indet, model_evidence=evidence, divergence_ranges=ranges)

    if base["digest_match"]:
        n = len(rec.entries)
        if n and indet > h.indet_frac * n:
            verdict, reason = INDETERMINATE, "text identical; too many checkpoints within the calibrated tolerance"
        elif divergent > 0:
            verdict, reason = INDETERMINATE, "text identical; the model evaluation differs from the sealing environment"
            if calibrated is False:
                reason += f" (configuration class {cls} is outside the calibration table)"
        else:
            verdict, reason = VERIFIED, ""
        return Result(verdict=verdict, reason=reason, first_divergence=None, seconds=round(time.time() - t0, 3),
                      **common, **base)

    text_match = text_view = None
    if h.plain in plain.SUPPORTED_PLAIN and h.plain_digest:
        ptext, pspans = plain.segment_visible(ctext, h.plain)
        if crypto.sha256_hex(ptext) == h.plain_digest:
            return Result(verdict=VERIFIED_TEXT, text_match=True, seconds=round(time.time() - t0, 3), **base,
                          reason="the visible text is identical to the sealed text (a rendered copy); formatting "
                                 "(Markdown markup, line breaks, list numbering) is not covered and the model layer "
                                 "is not evaluated on it")
        pchanged, ppoints = _diff_spans(ptext, pspans, rec.plain_blocks)
        text_match, text_view = False, {"text": ptext, "changed_ranges": pchanged, "deletion_points": ppoints}
    changed, points = _content_diff(ctext, rec.blocks, h.canon)
    starts = [r[0] for r in ranges] + [r[0] for r in changed] + points
    return Result(verdict=MODIFIED, reason="the canonical text differs from the sealed text", changed_ranges=changed,
                  deletion_points=points, first_divergence=(min(starts) if starts else None), text_match=text_match,
                  text_view=text_view, seconds=round(time.time() - t0, 3), **common, **base)
