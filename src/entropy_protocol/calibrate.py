"""Calibration of the verification tolerance (spec [0117]) — measured, not chosen.

Every configuration re-reads each calibration text (BOS + text, one pass). At every position and for both
predicates the signed margin (value of the token present minus the boundary value) is compared with the
reference configuration:
    delta(config) = max over positions and predicates of |margin_config - margin_reference|
The positions are those that can hold a checkpoint (index >= from_position; seal.first_checkpoint): the first
tokens after BOS never carry a record entry or a detector count, and their noise is not the text's (15.09.2026,
Qwen2.5-7B on an L40S, fp16 batch 32 against batch 1: 1.99 logits at position 3, at most 0.141 from position 4 on;
the table was 14 times too wide and records kept 1.6% of positions).
A configuration class (numeric precision of the scoring pass) takes the largest delta of its members, and its
tolerance is c * delta. The table goes into the record header; the seal keeps a checkpoint only if its margin
is at least seal_margin_factor times the largest tolerance of the table.

Space "logit": values are logits, so a margin is a log-probability ratio (log p_tok - log p_boundary); numerical
noise is then of the same size for the most probable token and for tail tokens, and one tolerance fits both
predicates. Space "prob": values are probabilities (the embodiment of [0115] as filed).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@torch.no_grad()
def margin_matrix(backend, ids: list[int], k: int, space: str) -> torch.Tensor:
    """Signed margins [T, 2] (column 0: LOW, column 1: HIGH) at every position of BOS + ids."""
    inp = torch.tensor([[backend.start_id] + ids], device=backend.device)
    logits = backend.model(inp).logits[0, :-1].float().cpu()
    vals = logits if space == "logit" else torch.softmax(logits, -1)
    tgt = torch.tensor(ids)[:, None]
    rank0 = (logits > logits.gather(1, tgt)).sum(1)
    v_tok = vals.gather(1, tgt).squeeze(1)
    top = torch.topk(vals, k + 1, dim=-1).values
    low_b = torch.where(rank0 == 0, top[:, 1], top[:, 0])
    high_b = torch.where(rank0 < k, top[:, k], top[:, k - 1])
    return torch.stack([v_tok - low_b, v_tok - high_b], 1)


@dataclass
class Calibration:
    space: str
    c: float
    k: int
    reference: str
    positions: int
    delta: dict = field(default_factory=dict)          # "device:dtype" -> max |delta margin|
    errors: dict = field(default_factory=dict)         # "device:dtype" -> reason (e.g. non-finite logits)
    from_position: int = 0                             # positions before this index are not measured

    def table(self, classes: list[str] | None = None) -> dict:
        """Configuration class -> tolerance, for the classes requested (default: every measured class)."""
        by_class: dict[str, float] = {}
        for cfg, d in self.delta.items():
            cls = _class_of(cfg.split(":")[1])
            by_class[cls] = max(by_class.get(cls, 0.0), d)
        ref_cls = _class_of(self.reference.split(":")[1])
        by_class.setdefault(ref_cls, 0.0)
        wanted = classes or sorted(by_class)
        missing = [c for c in wanted if c not in by_class]
        if missing:
            raise ValueError(f"no measurement for configuration classes {missing}")
        return {cls: round(self.c * by_class[cls], 6) for cls in wanted}


def _class_of(dtype_name: str) -> str:
    return {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(dtype_name, dtype_name)


def calibrate(reference, configs: dict, corpus: list[list[int]], k: int = 10, space: str = "logit",
              c: float = 3.0, from_position: int = 0) -> Calibration:
    """``reference`` is a backend; ``configs`` maps "device:dtype" to a backend of the same model or to a
    zero-argument loader (configurations are then loaded and released one at a time). ``from_position``: the
    first checkpoint position of the parameters the table will serve (seal.first_checkpoint); 0 measures all."""
    ref_name = f"{reference.device}:{str(reference.dtype).replace('torch.', '')}"
    corpus = [ids for ids in corpus if ids]          # a text may end at its first token: nothing to compare
    ref_m = [margin_matrix(reference, ids, k, space) for ids in corpus]
    cal = Calibration(space=space, c=c, k=k, reference=ref_name, from_position=from_position,
                      positions=sum(max(0, len(ids) - from_position) for ids in corpus))
    for name, b in configs.items():
        try:
            backend = b() if callable(b) else b
        except Exception as exc:  # noqa: BLE001 - an unavailable device is reported, not fatal
            cal.errors[name] = f"unavailable: {str(exc)[:160]}"
            continue
        d = 0.0
        for ids, rm in zip(corpus, ref_m):
            m = margin_matrix(backend, ids, k, space)
            if not torch.isfinite(m).all():
                cal.errors[name] = "non-finite values in the scoring pass"
                break
            diff = (m - rm)[from_position:]
            if diff.numel():
                d = max(d, float(diff.abs().max()))
        else:
            cal.delta[name] = d
        del backend
    return cal
