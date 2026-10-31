#!/usr/bin/env python3
"""End-to-end retrieval evaluation: in-distribution test split + hand-written OOD set.

Methods:
  bm25 / dense / feature_aware  - existing single-vector baselines
  colbert                       - BGE-M3 late interaction (ColBERT-style MaxSim)
  graphrag                      - ColBERT + BM25 + differentiable graph propagation
                                  + temporal subspace, deterministic operator plan
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

import memory_engine as me
import hybrid_retrieval as hr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
METHODS = ["bm25", "dense", "feature_aware", "colbert", "graphrag"]


def rank_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {}
    arr = np.asarray(ranks, dtype=np.float32)
    return {
        "Recall@1": float(np.mean(arr <= 1)),
        "Recall@3": float(np.mean(arr <= 3)),
        "Recall@5": float(np.mean(arr <= 5)),
        "Recall@10": float(np.mean(arr <= 10)),
        "MRR@10": float(np.mean(np.where(arr <= 10, 1.0 / arr, 0.0))),
        "n": len(ranks),
    }


def evaluate(engine, retriever, queries, split_label):
    results: dict[str, dict[str, list[int]]] = {m: defaultdict(list) for m in METHODS}
    per_query = []
    for q in queries:
        target = q["target_chunk_id"]
        row = {"query_id": q["query_id"], "query": q["query"], "target": target,
               "category": q.get("category", q.get("query_type", "test")), "ranks": {}}
        for method in METHODS:
            if method in ("colbert", "graphrag"):
                res = retriever.search(q["query"], top_k=10, meeting_id=q.get("meeting_id"), method=method)
            else:
                res = engine.search("synthetic", q["query"], method, top_k=10, meeting_id=q.get("meeting_id"))
            ids = [x["chunk_id"] for x in res["results"]]
            rank = ids.index(target) + 1 if target in ids else 11
            row["ranks"][method] = rank
            results[method][row["category"]].append(rank)
            results[method]["__all__"].append(rank)
        per_query.append(row)
    summary = {m: {cat: rank_metrics(v) for cat, v in cats.items()} for m, cats in results.items()}
    return {"split": split_label, "queries": len(queries), "summary": summary, "per_query": per_query}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--train-fusion", action="store_true")
    args = ap.parse_args()

    engine = me.MeetingMemoryEngine(device=args.device, batch_size=args.batch_size)
    retriever = hr.get_retriever(engine, "synthetic")

    fusion = None
    if args.train_fusion:
        train_queries = [q for q in engine._get_dataset("synthetic")["queries"] if q.get("split") == "train"]
        t = time.time()
        fusion = retriever.train_fusion(train_queries)
        print("fusion trained on", fusion["queries"], "queries in", round(time.time() - t, 1), "s",
              "weights", [round(w, 3) for w in fusion["weights"]])
        retriever.weights = __import__("torch").tensor(fusion["weights"])

    ood = json.loads((DATA / "ood_queries.json").read_text(encoding="utf-8"))
    test = [q for q in engine._get_dataset("synthetic")["queries"] if q.get("split") == "test"]

    t = time.time()
    in_dist = evaluate(engine, retriever, test, "test")
    print("in-distribution evaluated in", round(time.time() - t, 1), "s")
    t = time.time()
    out_dist = evaluate(engine, retriever, ood, "ood")
    print("ood evaluated in", round(time.time() - t, 1), "s")

    payload = {"protocol": {
        "in_distribution": "252 templated test queries, meeting-scoped",
        "ood": "18 hand-written paraphrases and multi-constraint queries, meeting-scoped",
        "colbert": "BAAI/bge-m3 token embeddings, MaxSim",
        "graphrag": "learned fusion of ColBERT rank, BM25 rank, graph PPR, temporal subspace, act match",
        "fusion": fusion,
        "weights": retriever.weights.tolist(),
    }, "in_distribution": in_dist, "ood": out_dist}
    ART.mkdir(exist_ok=True)
    (ART / "ood_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for label, block in (("IN-DISTRIBUTION (test, 252)", in_dist), ("OOD (18 hand-written)", out_dist)):
        print("\n" + label)
        print(f"{'method':14} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR@10':>7}")
        for method in METHODS:
            m = block["summary"][method].get("__all__", {})
            print(f"{method:14} {m.get('Recall@1',0):6.3f} {m.get('Recall@3',0):6.3f} "
                  f"{m.get('Recall@5',0):6.3f} {m.get('MRR@10',0):7.3f}")
    print("\nOOD by category (graphrag)")
    for cat, m in out_dist["summary"]["graphrag"].items():
        if cat != "__all__":
            print(f"  {cat:18} n={m['n']:2d} R@1={m['Recall@1']:.2f} MRR={m['MRR@10']:.3f}")
    print("\nSaved", ART / "ood_results.json")


if __name__ == "__main__":
    main()
