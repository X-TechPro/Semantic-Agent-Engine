"""Reciprocal Rank Fusion (6.7), with weights and an ablatable `k`.

score(d) = Σ_i w_i / (rrf_k + rank_i(d))

Ranks are 1-based. Unit-tested on hand-computed toy ranks (12.1).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FusedHit:
    unit_id: str
    score: float
    parts: dict[str, float]


def rrf(rankings: dict[str, list[str]], *, k: int = 60,
        weights: dict[str, float] | None = None) -> list[FusedHit]:
    """rankings: source name -> ordered unit ids (best first)."""
    weights = weights or {}
    acc: dict[str, float] = {}
    parts: dict[str, dict[str, float]] = {}
    for name, order in rankings.items():
        w = weights.get(name, 1.0)
        for rank, uid in enumerate(order, start=1):
            contrib = w / (k + rank)
            acc[uid] = acc.get(uid, 0.0) + contrib
            parts.setdefault(uid, {})[f"rrf_{name}"] = contrib
    out = [FusedHit(uid, s, parts[uid]) for uid, s in acc.items()]
    out.sort(key=lambda h: (-h.score, h.unit_id))  # deterministic tie-break
    return out


def normalised_scores(results: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max to [0,1] for reporting only — fusion never uses these."""
    if not results:
        return {}
    vals = [s for _, s in results]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {uid: 1.0 for uid, _ in results}
    return {uid: (s - lo) / (hi - lo) for uid, s in results}
