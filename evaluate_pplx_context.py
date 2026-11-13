#!/usr/bin/env python3
"""Evaluate pplx-embed-context-v1-0.6b in its intended dense-chunk mode.

This model is a 1024-d contextual dense embedder, not a late-interaction
encoder. It therefore cannot be plugged into Rexu v2.2's ColBERT MaxSim path
without a different projection/training setup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
HF = ART / "hf_cache"
MODEL = "perplexity-ai/pplx-embed-context-v1-0.6b"


class PplxContext:
    def __init__(self, device: str = "cuda"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            MODEL, trust_remote_code=True, cache_dir=str(HF)).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        # Standalone query mode: each query is a one-chunk document.
        rows = []
        for start in range(0, len(texts), 8):
            batch = texts[start:start + 8]
            out = self.model.encode([[t] for t in batch])
            for x in out:
                v = torch.as_tensor(np.asarray(x[0]), dtype=torch.float32)
                rows.append(F.normalize(v, dim=-1).cpu())
        return torch.stack(rows)

    @torch.inference_mode()
    def encode_document(self, chunks: list[str]):
        # Context mode: all chunks from one document in order, one call.
        out = self.model.encode([chunks])
        return [F.normalize(torch.as_tensor(np.asarray(x), dtype=torch.float32), dim=-1).cpu()
                for x in out[0]]


def metrics(ranks):
    a = np.asarray(ranks, dtype=np.float32)
    return {"R@1": float((a <= 1).mean()), "R@5": float((a <= 5).mean()),
            "MRR": float(np.where(a <= 10, 1 / a, 0).mean()), "n": len(a)}


def main():
    import json
    import memory_engine as me
    eng = me.MeetingMemoryEngine(device="cuda")
    ds = eng._get_dataset("synthetic")
    enc = PplxContext()
    # Correct contextual mode: all chunks of a meeting in chronological order.
    meeting_chunks = {
        mid: [ds["records"][c]["encoder_text"] for c in chunks]
        for mid, chunks in ds["by_meeting"].items()
    }
    doc_vectors = {}
    for mid, chunks in meeting_chunks.items():
        doc_vectors[mid] = enc.encode_document(chunks)
        print(f"  context meeting {mid}: {len(chunks)} chunks", flush=True)
    # v7 test queries, candidate pool is the meeting.
    queries = [q for q in json.loads((ROOT / "data/french_retrieval_queries.json").read_text())
               if q["split"] == "test"]
    ranks = []
    by = {}
    for q in queries:
        qv = enc.encode([q["query"]])[0]
        cand = ds["by_meeting"][q["meeting_id"]]
        dvecs = doc_vectors[q["meeting_id"]]
        scores = np.asarray([float(qv @ dvecs[i]) for i, c in enumerate(cand)])
        order = [cand[i] for i in np.argsort(-scores)]
        gold = set(q["target_chunk_ids"])
        p = [order.index(c) + 1 for c in gold if c in order]
        r = min(p) if p else len(order) + 1
        ranks.append(r); by.setdefault(q["query_type"], []).append(r)
    out = {"v7_test": metrics(ranks), "v7_by_type": {k: metrics(v)["R@1"] for k, v in by.items()}}
    # v8 held-out multi-hop, same records constructed as the existing eval.
    d8 = json.loads((ROOT / "data/v8_meetings.json").read_text())
    q8 = json.loads((ROOT / "data/v8_queries.json").read_text())
    rec = {}
    for meeting in d8["meetings"]:
        for m in meeting["messages"]:
            cid = f"{meeting['meeting_id']}:{m['message_id']:04d}"
            rec[cid] = (f"Réunion : {meeting['title']}. Intervenant : {m['username']}. "
                        f"Intervention : {m['message']}")
    rids = list(rec); rpos = {c: i for i, c in enumerate(rids)}
    meeting_vecs = {}
    for meeting in d8["meetings"]:
        mcs = [f"Réunion : {meeting['title']}. Intervenant : {m['username']}. Intervention : {m['message']}" for m in meeting["messages"]]
        meeting_vecs[meeting["meeting_id"]] = enc.encode_document(mcs)
    ridx = {}
    for meeting in d8["meetings"]:
        for k, m in enumerate(meeting["messages"]):
            ridx[f"{meeting['meeting_id']}:{m['message_id']:04d}"] = (meeting["meeting_id"], k)
    ranks8, ev, allh = [], [], []
    for q in q8:
        qv = enc.encode([q["query"]])[0]
        sc = np.asarray([float(qv @ meeting_vecs[ridx[c][0]][ridx[c][1]]) for c in rids])
        order = [rids[i] for i in np.argsort(-sc)]
        gold = set(q["evidence_chunk_ids"])
        p = [order.index(c) + 1 for c in gold if c in order]
        ranks8.append(min(p) if p else len(order) + 1)
        found = {c for c in order[:5] if c in gold}
        ev.append(len(found) / len(gold)); allh.append(float(found == gold))
    out["v8_test"] = {**metrics(ranks8), "EvidenceRecall@5": float(np.mean(ev)),
                      "AllHops@5": float(np.mean(allh))}
    out["meta"] = {"model": MODEL, "mode": "dense contextual, mean-pooled 1024d",
                   "note": "not late interaction; no MaxSim / ColBERTv2 path"}
    (ART / "pplx_context_eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
