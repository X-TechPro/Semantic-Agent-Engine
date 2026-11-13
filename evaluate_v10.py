#!/usr/bin/env python3
"""Evaluate Rexu v3 on the hand-written v10 test split."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import memory_engine as me
from pplx_rexu import RexuV3Retriever

e = me.MeetingMemoryEngine(device="cuda")
ds = e._get_dataset("v10")
r = RexuV3Retriever(e, "v10"); r._ensure()
qs = [q for q in ds["queries"] if q.get("split") == "test"]
ranks, ev, ah = [], [], []
for q in qs:
    out = r.search(q["query"], top_k=len(ds["records"]), meeting_id=None, method="rexu_v3")
    order = [x["chunk_id"] for x in out["results"]]; gold = set(q["target_chunk_ids"])
    p = [order.index(c) + 1 for c in gold if c in order]
    ranks.append(min(p) if p else len(order) + 1)
    found = set(order[:5]) & gold; ev.append(len(found) / len(gold)); ah.append(float(found == gold))
a = np.asarray(ranks)
out = {"n": len(qs), "R@1": float((a <= 1).mean()), "R@5": float((a <= 5).mean()),
       "MRR": float(np.where(a <= 10, 1/a, 0).mean()),
       "EvidenceRecall@5": float(np.mean(ev)), "AllHops@5": float(np.mean(ah))}
(Path(__file__).parent / "artifacts/pplx_v10_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
