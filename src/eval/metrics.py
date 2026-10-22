"""Metric definitions (BUILD_GUIDE 9.1), each unit-tested on hand-computed
toy inputs.

The instrument is written *before* the comparison, and deliberately reports
diagnostics that can falsify its own headline: `gold_in_pool` says whether the
correct unit was even retrieved. If that is low, temporal-correctness@1 cannot
detect a relation-layer gain — a null result would be an instrument problem,
not a thesis problem (see docs/LIMITS.md).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def recall_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if not gold:
        return float("nan")
    return len(gold & set(ranked[:k])) / len(gold)


def precision_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    if k <= 0:
        return float("nan")
    return len(gold & set(ranked[:k])) / k


def hit_at_k(gold: set[str], ranked: list[str], k: int) -> float:
    return 1.0 if gold & set(ranked[:k]) else 0.0


def reciprocal_rank(gold: set[str], ranked: list[str], cutoff: int = 10) -> float:
    for i, uid in enumerate(ranked[:cutoff], start=1):
        if uid in gold:
            return 1.0 / i
    return 0.0


def mrr_at_10(gold: set[str], ranked: list[str]) -> float:
    return reciprocal_rank(gold, ranked, 10)


def ndcg_at_k(gold: set[str], ranked: list[str], k: int = 10) -> float:
    gains = [1.0 if uid in gold else 0.0 for uid in ranked[:k]]
    dcg = sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))
    ideal = sorted(gains, reverse=True)
    n = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n + 1))
    return dcg / idcg if idcg else 0.0


def temporal_correctness_at_1(current_gold: set[str], ranked: list[str],
                              superseded: set[str]) -> float | None:
    """Headline metric (9.1): rank-1 is the in-force unit, not superseded,
    not a hypothesis. Returns None when the question has no gold (unanswerable).
    """
    if not current_gold:
        return None
    if not ranked:
        return 0.0
    top = ranked[0]
    if top in superseded:
        return 0.0
    return 1.0 if top in current_gold else 0.0


def stale_rate_at_k(ranked: list[str], superseded: set[str], k: int = 10) -> float:
    pool = ranked[:k]
    if not pool:
        return float("nan")
    return len([u for u in pool if u in superseded]) / len(pool)


def hypothesis_leak(ranked: list[str], soft_unit_ids: set[str],
                    has_confirmed: bool) -> float | None:
    """1.0 when top-1 is a proposal/hypothesis while a confirmed answer exists."""
    if not has_confirmed or not ranked:
        return None
    return 1.0 if ranked[0] in soft_unit_ids else 0.0


def gold_in_pool(gold: set[str], ranked: list[str]) -> float:
    """Diagnostic ceiling: was the gold retrieved at all, at any rank?"""
    if not gold:
        return float("nan")
    return 1.0 if gold & set(ranked) else 0.0


def pairwise_accuracy(positive_rank: int, negative_rank: int) -> float:
    """1.0 when the positive outranks its minimal-pair negative."""
    return 1.0 if positive_rank < negative_rank else 0.0


def macro_f1(gold_labels: list[str], pred_labels: list[str]
             ) -> tuple[float, dict[str, dict[str, float]]]:
    """Returns (macro-F1, per-class {p,r,f1,support})."""
    labels = sorted(set(gold_labels) | set(pred_labels))
    per: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for lab in labels:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold_labels, pred_labels) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold_labels, pred_labels) if g == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per[lab] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": tp + fn}
        f1s.append(f1)
    return (round(sum(f1s) / len(f1s), 4) if f1s else 0.0), per


def precision_of(relation: str, edges: list[tuple[str, str, str]],
                 gold: list[tuple[str, str, str]]) -> float:
    """P/R for one relation over (from, to, relation) triples."""
    pred = {(a, b) for a, b, r in edges if r == relation}
    tr = {(a, b) for a, b, r in gold if r == relation}
    prec = len(pred & tr) / len(pred) if pred else 0.0
    rec = len(pred & tr) / len(tr) if tr else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4)}.get("precision", 0.0)


def relation_prf(edges, gold) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    rels = {r for _, _, r in gold} | {r for _, _, r in edges}
    pred_s = set(edges)
    gold_s = set(gold)
    for rel in sorted(rels):
        p = {(a, b) for a, b, r in pred_s if r == rel}
        t = {(a, b) for a, b, r in gold_s if r == rel}
        prec = len(p & t) / len(p) if p else 0.0
        rec = len(p & t) / len(t) if t else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[rel] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "pred_n": len(p), "gold_n": len(t)}
    return out


def expected_calibration_error(probs: list[float], labels: list[int],
                               n_bins: int = 10) -> float:
    """ECE over (confidence, correctness) pairs — 6.2.3 calibration gate."""
    if not probs:
        return float("nan")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    total = sum(len(b) for b in bins)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / total) * abs(conf - acc)
    return round(ece, 4)


def temperature_scale(logits: list[list[float]], labels: list[int],
                      iters: int = 40) -> float:
    """1-D temperature fit by golden-section on NLL (no torch dependency)."""
    import math as _m

    def nll(t: float) -> float:
        s = 0.0
        for row, y in zip(logits, labels):
            z = [v / t for v in row]
            mx = max(z)
            denom = mx + _m.log(sum(_m.exp(v - mx) for v in z))
            s += denom - z[y]
        return s / max(1, len(logits))

    lo, hi = 0.05, 10.0
    phi = (1 + 5 ** 0.5) / 2
    a, b = lo, hi
    c, d = b - (b - a) / phi, a + (b - a) / phi
    for _ in range(iters):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - (b - a) / phi, a + (b - a) / phi
        if abs(b - a) < 1e-4:
            break
    return round((a + b) / 2, 4)


def _finite(v) -> bool:
    return v is not None and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def bootstrap_ci(values: list[float], groups: list[str], *, seed: int = 42,
                 n: int = 1000, alpha: float = 0.05) -> tuple[float, float, float]:
    """9.6: bootstrap over *groups* (projects), not over sentences."""
    if not values:
        return (float("nan"),) * 3
    by_group: dict[str, list[float]] = {}
    for v, g in zip(values, groups):
        if not _finite(v):
            continue
        by_group.setdefault(g, []).append(float(v))
    keys = sorted(by_group)
    if not keys:
        return (float("nan"),) * 3
    point = sum(sum(v) / len(v) for v in by_group.values()) / len(keys)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(n):
        sample = [by_group[rng.choice(keys)] for _ in keys]
        means.append(sum(sum(s) / len(s) for s in sample) / len(sample))
    means.sort()
    lo = means[max(0, int((alpha / 2) * n))]
    hi = means[min(n - 1, int((1 - alpha / 2) * n))]
    return round(point, 4), round(lo, 4), round(hi, 4)


def paired_bootstrap_diff(a: list[float], b: list[float], groups: list[str], *,
                          seed: int = 42, n: int = 1000
                          ) -> dict[str, float]:
    """Paired difference B-A over projects; wins only if the CI excludes 0.

    Non-finite pairs (a question type that is absent from a project) are dropped
    *pairwise* — dropping one side only would bias the difference, and letting a
    NaN through used to poison the whole statistic (observed as delta=NaN).
    """
    by: dict[str, list[tuple[float, float]]] = {}
    for va, vb, g in zip(a, b, groups):
        if not (_finite(va) and _finite(vb)):
            continue
        by.setdefault(g, []).append((float(va), float(vb)))
    if not by:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "significantly_better": 0.0, "n_projects": 0}
    keys = sorted(by)
    # B - A: a positive delta means the relation layer (System B) is ahead.
    # The sign here once read A - B while the docstring promised B - A, which
    # made `significant_win` fire when the *baseline* won.
    dvals = [sum(y - x for x, y in by[g]) / len(by[g]) for g in keys]
    delta = sum(dvals) / len(dvals)
    rng = random.Random(seed)
    res: list[float] = []
    for _ in range(n):
        pick = [by[rng.choice(keys)] for _ in keys]
        res.append(sum(sum(y - x for x, y in s) / len(s) for s in pick) / len(pick))
    res.sort()
    lo, hi = res[int(0.025 * n)], res[min(n - 1, int(0.975 * n))]
    return {"delta": round(delta, 4), "lo": round(lo, 4), "hi": round(hi, 4),
            "significantly_better": 1.0 if lo > 0 else 0.0,
            "significantly_worse": 1.0 if hi < 0 else 0.0,
            "n_projects": len(keys)}
