"""Provenance of a benchmark result: which code produced it (digest taken when the benchmark starts)."""
from __future__ import annotations

import hashlib
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "entropy_protocol"


def _compute() -> str:
    h = hashlib.sha256()
    for p in sorted(SRC.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


_DIGEST = _compute()


def code_digest() -> str:
    """sha256 over the reference implementation's source files (sorted by name) at import time, 16 hex digits."""
    return _DIGEST
