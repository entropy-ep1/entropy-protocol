"""The sealed record: a header and entries (canonical character offset, expected property).

Entries carry no token identifier and no text — only where a checkpoint is and which property
the model's choice there must satisfy. Serialized as canonical CBOR, then encrypted (crypto.py).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cbor2

LOW, HIGH = 0, 1
PROP_NAMES = {LOW: "LOW", HIGH: "HIGH"}


@dataclass(frozen=True)
class Header:
    model: str                 # model identifier (repo id @ revision)
    weights_digest: str        # sha256 over the model's parameters
    tokenizer_digest: str      # sha256 of the serialized tokenizer
    canon: str                 # canonicalization identifier
    context: str               # how the verifier builds context, e.g. "bos+text"
    k: int                     # HIGH holds iff rank >= k (0-based), i.e. outside the top-k
    interval: int              # checkpoint spacing at generation (informative)
    tol: float                 # tolerance when no calibration table is given (0 = exact process)
    min_entries: int           # fewer decisive entries -> INDETERMINATE
    indet_frac: float          # more indeterminate entries than this fraction -> INDETERMINATE
    text_digest: str           # sha256 of the canonical text (binds the record to one text)
    engine: str                # sealing engine version
    extra: dict = field(default_factory=dict)
    tol_space: str = "prob"    # "prob": |p - p_boundary| (spec [0115]); "logit": |log p - log p_boundary|
    cal: dict = field(default_factory=dict)   # calibration table: configuration class -> tolerance ([0117])
    components: dict = field(default_factory=dict)   # measured model components: name -> sha-256 hex
    env: dict = field(default_factory=dict)          # sealing environment (determinism envelope)
    selection: str = "interval"  # "interval" | "content" (content-keyed checkpoints, detectable without the record)
    wm_context: int = 0          # content selection: number of preceding token ids the PRF reads
    salt: str = ""               # 16 random bytes (hex) for the public commitment SHA-256(salt || text_digest)
    wm_gate: float = 0.0         # confidence gate: checkpoints only where the verifier-context first choice has
                                 # probability below this (0 = off); on the wire only when set (per-mille)
    plain: str = ""              # visible-text layer (plain.py): its form id; on the wire only when set
    plain_digest: str = ""       # SHA-256 of the visible text


@dataclass(frozen=True)
class Record:
    header: Header
    entries: tuple[tuple[int, int], ...]            # model layer: (canonical char offset, LOW|HIGH)
    blocks: tuple[tuple[int, int, str], ...] = ()   # content layer: (start, end, 64-bit digest) per block
    plain_blocks: tuple[tuple[int, int, str], ...] = ()   # visible-text layer: blocks of the visible text


MICRO = 1_000_000          # tolerances travel as integers in micro-units of the tolerance space
PERMILLE = 1_000          # fractions travel as integers in per-mille


def _wire_header(h: Header) -> dict:
    """Header as it travels: no floating point anywhere (integer micro-units and per-mille)."""
    d = asdict(h)
    d["tol_u"] = int(round(d.pop("tol") * MICRO))
    d["cal_u"] = {str(c): int(round(float(v) * MICRO)) for c, v in d.pop("cal").items()}
    d["indet_pm"] = int(round(d.pop("indet_frac") * PERMILLE))
    gate = d.pop("wm_gate")
    if gate:                                    # optional: records without a gate keep their exact bytes
        d["wm_gate_pm"] = int(round(gate * PERMILLE))
    if not d["plain"]:                          # optional: records without the visible-text layer keep their bytes
        del d["plain"], d["plain_digest"]
    for key in ("extra", "components", "env"):
        if any(isinstance(v, float) for v in d[key].values()):
            raise ValueError(f"floating point value in header field {key!r}")
    return d


def _header_from_wire(d: dict) -> Header:
    d = dict(d)
    if any(isinstance(v, float) for v in d.values()):
        raise ValueError("floating point value in a record header")
    d["tol"] = d.pop("tol_u") / MICRO
    d["cal"] = {c: v / MICRO for c, v in d.pop("cal_u").items()}
    d["indet_frac"] = d.pop("indet_pm") / PERMILLE
    d["wm_gate"] = d.pop("wm_gate_pm", 0) / PERMILLE
    return Header(**d)


def dumps(rec: Record) -> bytes:
    body = {"h": _wire_header(rec.header), "e": [[int(c), int(p)] for c, p in rec.entries],
            "b": [[int(s), int(e), str(d)] for s, e, d in rec.blocks]}
    if rec.plain_blocks:
        body["p"] = [[int(s), int(e), str(d)] for s, e, d in rec.plain_blocks]
    return cbor2.dumps(body, canonical=True)


def loads(data: bytes) -> Record:
    body = cbor2.loads(data)
    header = _header_from_wire(body["h"])
    entries = tuple((int(c), int(p)) for c, p in body["e"])
    blocks = tuple((int(s), int(e), str(d)) for s, e, d in body.get("b", []))
    plain_blocks = tuple((int(s), int(e), str(d)) for s, e, d in body.get("p", []))
    for c, p in entries:
        if p not in PROP_NAMES or c < 0:
            raise ValueError("invalid record entry")
    if header.tol_space not in ("prob", "logit") or any(float(v) < 0 for v in header.cal.values()):
        raise ValueError("invalid tolerance declaration")
    if not 0 <= header.wm_gate < 1:
        raise ValueError("invalid confidence gate")
    return Record(header=header, entries=entries, blocks=blocks, plain_blocks=plain_blocks)
