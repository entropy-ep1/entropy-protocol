"""Ordinal (rank) predicates and their boundary margins (spec [0114]-[0115]).

rank0 = number of vocabulary tokens with strictly greater probability than the token present.
  LOW  holds iff rank0 == 0          (the model's single most probable choice)
  HIGH holds iff rank0 >= k          (outside the model's top-k)
Boundary probability = the probability at which satisfaction would flip:
  LOW  -> the largest probability among the other tokens
  HIGH -> the k-th largest probability among the other tokens (1-based)
margin = |p(token) - boundary|. If margin < tolerance the outcome is INDETERMINATE.
"""
from __future__ import annotations

from .record import HIGH, LOW


def holds(prop: int, rank0: int, k: int) -> bool:
    return rank0 == 0 if prop == LOW else rank0 >= k


def boundary(prop: int, rank0: int, top_sorted: list[float], k: int) -> float:
    """``top_sorted`` = the k+1 largest probabilities of the distribution, descending."""
    if prop == LOW:
        return top_sorted[1] if rank0 == 0 else top_sorted[0]
    if prop == HIGH:
        return top_sorted[k] if rank0 < k else top_sorted[k - 1]
    raise ValueError(f"unknown property {prop}")


def margin(prop: int, rank0: int, p_tok: float, top_sorted: list[float], k: int) -> float:
    """Works in either space: pass probabilities, or logits (then the margin is a log-probability ratio,
    because log p_i - log p_j = logit_i - logit_j)."""
    return abs(p_tok - boundary(prop, rank0, top_sorted, k))


def stat_margin(prop: int, st, k: int, space: str) -> float:
    value, top = st.values(space)
    return margin(prop, st.rank0, value, top, k)
