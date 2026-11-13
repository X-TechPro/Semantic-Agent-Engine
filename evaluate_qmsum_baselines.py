#!/usr/bin/env python3
"""Baselines on the converted QMSum test split, same protocol as evaluate_qmsum.py.

Methods (all frozen, no training):
- bm25        : lexical BM25 over the meeting's turns
- bge_only    : BGE-M3 dense (CLS), one vector per turn
- colbert_v2  : BGE-M3 late-chunked ColBERTv2-style index (frozen projection)

Meeting-scoped: each query searches only its own meeting. Metrics match evaluate_qmsum.py.
Results go to artifacts/qmsum_baselines_eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import memory_engine as me
from hybrid_retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parent


def metrics(ranked_by_query: list[tuple[list[str], set[str]]]) -> dict[str, float]:
    rec1, any1, rec5, any5, mrr, best = [], [], [], [], [], []
    for order, gold in ranked_by_query:
        pos = [order.index(c) + 1 for c in gold if c in order]
        best.append(min(pos) if pos else 10**6)
        top1, top5 = set(order[:1]) & gold, set(order[:5]) & gold
        rec1.append(len(top1) / len(gold)); any1.append(float(bool(top1)))
        rec5.append(len(top5) / len(gold)); any5.append(float(bool(top5)))
        mrr.append(1.0 / min(pos) if pos and min(pos) <= 10 else 0.0)
    return {
        "Recall@1": float(np.mean(rec1)), "AnyR@1": float(np.mean(any1)),
        "Recall@5": float(np.mean(rec5)), "AnyR@5": float(np.mean(any5)),
        "MRR@10": float(np.mean(mrr)), "median_best_gold_rank": float(np.median(best)),
    }


def main() -> None:
    e = me.MeetingMemoryEngine(device="cuda")
    ds = e._get_dataset("qmsum")
    qs = [q for q in ds["queries"] if q.get("split") == "test"]
    hr = HybridRetriever(e, "qmsum")

    results: dict[str, dict] = {}
    for method in ("bm25", "bge_only", "colbert_v2"):
        ranked = []
        for q in qs:
            mid = q["meeting_id"]
            cands = ds["by_meeting"][mid]
            gold = set(q["target_chunk_ids"])
            if method == "bm25":
                scores = ds["bm25"].scores(q["query"], cands)
                order = [cands[i] for i in np.argsort(-scores)]
            else:
                out = hr.search(q["query"], top_k=len(cands), meeting_id=mid, method=method)
                order = [x["chunk_id"] for x in out["results"]]
            ranked.append((order, gold))
        results[method] = metrics(ranked)
        print(method, json.dumps(results[method]), flush=True)

    payload = {"dataset": "qmsum-test", "scope": "meeting", "n_queries": len(qs),
               "n_meetings": len(ds["meetings"]), "methods": results}
    (ROOT / "artifacts" / "qmsum_baselines_eval.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
