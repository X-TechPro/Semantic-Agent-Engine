#!/usr/bin/env python3
"""v8 multi-hop evaluation: Rexu v2.2 and GraphRAG."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

import memory_engine as me
import run_french_lab as lab
from hybrid_retrieval import MultiVectorIndex, GraphMemory
from bge_colbert_v2 import CompressedColBERTIndex

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def main():
    data = json.loads((ROOT / "data/v8_meetings.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "data/v8_queries.json").read_text(encoding="utf-8"))
    records = {}
    for meeting in data["meetings"]:
        for m in meeting["messages"]:
            cid = f"{meeting['meeting_id']}:{m['message_id']:04d}"
            records[cid] = {
                "chunk_id": cid, "message_id": m["message_id"],
                "meeting_id": meeting["meeting_id"], "meeting_date": meeting["meeting_date"], "username": m["username"],
                "timestamp": m["timestamp"], "message_text": m["message"],
                "encoder_text": f"Réunion : {meeting['title']}. Intervenant : {m['username']}. Intervention : {m['message']}",
            }
    ids = list(records); pos = {c: i for i, c in enumerate(ids)}
    meetings = {m["meeting_id"]: m for m in data["meetings"]}
    ds = {"records": records, "meetings": meetings, "relations": data.get("relations", [])}
    graph = GraphMemory(ds); graph.build_torch(device="cuda")
    bm25 = lab.BM25(records)

    def pct(x):
        return np.argsort(np.argsort(x)) / max(1, len(x) - 1)

    def evaluate(score_fn, name):
        ranks, ev, allh, details = [], [], [], []
        for q in queries:
            score = score_fn(q)
            order = [ids[i] for i in np.argsort(-score)]
            gold = set(q["evidence_chunk_ids"])
            p = [order.index(c) + 1 for c in order if c in gold]
            rank = min(p) if p else len(order) + 1
            found = {c for c in order[:5] if c in gold}
            ranks.append(rank); ev.append(len(found) / len(gold))
            allh.append(1.0 if found == gold else 0.0)
            details.append({"query": q["query"], "answer": q["answer"], "rank": rank,
                            "evidence_recall@5": ev[-1], "all_hops@5": allh[-1], "top5": order[:5]})
        a = np.asarray(ranks)
        return {"R@1": float((a <= 1).mean()), "R@5": float((a <= 5).mean()),
                "MRR": float(np.where(a <= 10, 1 / a, 0).mean()),
                "EvidenceRecall@5": float(np.mean(ev)), "AllHops@5": float(np.mean(allh)),
                "details": details}

    raw = MultiVectorIndex(records, "v8-raw", device="cuda"); raw.build()
    normal = CompressedColBERTIndex(raw, device="cuda")
    def normal_base(q):
        full = normal.score(q["query"]); return np.asarray([full[pos[c]] for c in ids], np.float32)
    normal_out = evaluate(normal_base, "normal")
    def graph_from(base_fn):
        def fn(q):
            col = base_fn(q); bm = np.asarray(bm25.scores(q["query"], ids), np.float32)
            seed = 0.65 * pct(col) + 0.35 * pct(bm)
            seed = np.where(seed >= np.partition(seed, -30)[-30], seed, 0.0)
            gp = graph.ppr(seed)
            return 0.5 * pct(col) + 0.2 * pct(bm) + 0.3 * pct(gp)
        return fn
    graph_normal = evaluate(graph_from(normal_base), "graph_normal")
    del normal, raw; gc.collect(); torch.cuda.empty_cache()

    ft = MultiVectorIndex(records, "v8-ft", device="cuda",
                          projection_path=ART / "bge_colbert_projection_mixed.pt"); ft.build()
    rexu = CompressedColBERTIndex(ft, device="cuda")
    def rexu_base(q):
        full = rexu.score(q["query"]); return np.asarray([full[pos[c]] for c in ids], np.float32)
    rexu_out = evaluate(rexu_base, "rexu22")
    graph_rexu = evaluate(graph_from(rexu_base), "graph_rexu22")
    del rexu, ft; gc.collect(); torch.cuda.empty_cache()
    out = {"normal_bge_colbertv2": normal_out, "rexu_v2_2": rexu_out,
           "graphrag_normal": graph_normal, "graphrag_rexu_v2_2": graph_rexu,
           "meta": {"meetings": len(meetings), "messages": len(records),
                    "queries": len(queries), "relations": len(data.get("relations", []))}}
    print(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    (ART / "v8_benchmark.json").write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
