"""Exact statistics of record-free detection, with no machine-learning dependency.

Kept apart from detect.py (which needs torch to run a model) so that everything a third party needs to check
formats, statements, test vectors and evidence packages imports without torch: a second implementation and a
conformance run should not have to install a model stack to compare p-values.
"""
from __future__ import annotations

from functools import lru_cache
from math import comb


@lru_cache(maxsize=65536)
def binomial_tail(n: int, m: int) -> float:
    """P(X >= m) for X ~ Binomial(n, 1/2), exact (integer arithmetic)."""
    if m <= 0:
        return 1.0
    if m > n:
        return 0.0
    return sum(comb(n, i) for i in range(m, n + 1)) / (1 << n)
