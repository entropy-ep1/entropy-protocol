"""Operating points: what a deployment seals with, and the gate its record-free detector reads with.

Measured on the Mac (scale bridge, 13.09.2026; Qwen2.5-0.5B and 1.5B Instruct, MPS fp16):

quality  guard 0.9 at sealing: no checkpoint where the prompt-conditioned model is at least 90% sure of its next
         token (a number or a name copied from the question, code). GSM8K accuracy as unsealed (0.5B: 38% = 38%),
         prose perplexity as unsealed (x1.01). The detector skips the positions where its own, prompt-free view is at
         least 90% sure (read from the text and the model, so the exact null holds): prose, 200 tokens, 0.5B: 90% at
         a false-positive rate of 1e-6 and 100% at 1e-4. Maths keeps almost no mark (low entropy: any watermark).
strong   the protocol's reference point: every selected checkpoint is enforced. Prose, 200 tokens: 100% at 1e-6, at
         a perplexity of x1.15-1.25; but a checkpoint may overrule a confident model: GSM8K 38% -> 3% (0.5B) and
         64% -> 11% (1.5B).
The record (verification against it) is exact under either: its content layer does not depend on enforcement.

Default: quality (Serhii, 13.09.2026), on the product's surfaces: `entropy seal / detect / serve` and gateway.Gateway.
SealParams() itself stays the protocol's reference parameters (strong): the specification and its test vectors are
written against them, and a record carries no guard (only a sealing gate, wm_gate_pm), so records of either point
have the same form.
"""
from __future__ import annotations

SEAL = {"quality": {"guard": 0.9}, "strong": {}}
DETECT_GATE = {"quality": 0.9, "strong": 0.0}
NAMES = tuple(SEAL)
DEFAULT = "quality"


def detect_gate_for(params) -> float:
    """The detector's gate for texts sealed with ``params``: positions the sealing left alone because the model was
    confident (the gate reads the verifier's view, the guard the view with the prompt) are not counted."""
    return max(params.gate, params.guard)


def seal_overrides(name: str) -> dict:
    """SealParams fields of the operating point ``name``."""
    if name not in SEAL:
        raise ValueError(f"unknown operating point {name!r}; known: {', '.join(NAMES)}")
    return dict(SEAL[name])


def detect_gate(name: str) -> float:
    """The detector's confidence gate for texts sealed at the operating point ``name``."""
    if name not in DETECT_GATE:
        raise ValueError(f"unknown operating point {name!r}; known: {', '.join(NAMES)}")
    return DETECT_GATE[name]
