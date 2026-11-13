#!/usr/bin/env python3
"""Evaluate Rexu v3 on the converted QMSum test split (meeting-scoped retrieval).

QMSum queries are per-meeting, so each query searches only inside its own meeting.
Metrics: Recall@k credits the fraction of gold turns found; AnyR@k credits a hit on
at least one gold turn. Results go to artifacts/pplx_qmsum_eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import memory_engine as me
from pplx_rexu import RexuV3Retriever

ROOT = Path(__file__).resolve().parent


def main() -> None:
    e = me.MeetingMemoryEngine(device="cuda")
    ds = e._get_dataset("qmsum")
    r = RexuV3Retriever(e, "qmsum")
    r._ensure()
    qs = [q for q in ds["queries"] if q.get("split") == "test"]

    ranks, rec1, any1, rec5, any5, mrr = [], [], [], [], [], []
    for q in qs:
        gold = set(q["target_chunk_ids"])
        n_in_meeting = len(ds["by_meeting"][q["meeting_id"]])
        out = r.search(q["query"], top_k=n_in_meeting,
                       meeting_id=q["meeting_id"], method="rexu_v3")
        order = [x["chunk_id"] for x in out["results"]]
        pos = [order.index(c) + 1 for c in gold if c in order]
        ranks.append(min(pos) if pos else 10**6)
        top1, top5 = set(order[:1]) & gold, set(order[:5]) & gold
        rec1.append(len(top1) / len(gold)); any1.append(float(bool(top1)))
        rec5.append(len(top5) / len(gold)); any5.append(float(bool(top5)))
        mrr.append(1.0 / min(pos) if pos and min(pos) <= 10 else 0.0)

    res = {
        "dataset": "qmsum-test", "scope": "meeting", "n_queries": len(qs),
        "n_meetings": len(ds["meetings"]),
        "Recall@1": float(np.mean(rec1)), "AnyR@1": float(np.mean(any1)),
        "Recall@5": float(np.mean(rec5)), "AnyR@5": float(np.mean(any5)),
        "MRR@10": float(np.mean(mrr)),
        "median_best_gold_rank": float(np.median(ranks)),
    }
    (ROOT / "artifacts" / "pplx_qmsum_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
