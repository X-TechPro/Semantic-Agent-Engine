#!/usr/bin/env python3
"""Corrected ColBERT comparison.

- bge_raw: BGE-M3 last_hidden_state MaxSim (the v1 path).
- bge_proj: BGE-M3 + official colbert_linear projection MaxSim.
- colbert_xm: antoinelouis/colbert-xm, multilingual 128-d ColBERT.
- bge_dense / bge_late: single-vector BGE-M3, per-chunk vs late-chunked
  (the late-chunking paper's actual setting: mean-pool a long-context token
  sequence into chunk vectors).

Writes artifacts/proper_colbert_eval.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import memory_engine as me
import hybrid_retrieval as hr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
HF = ART / "hf_cache"


def gold_ids(q):
    return [x for x in (q.get("target_chunk_ids") or [q.get("target_chunk_id")]) if x]


def metrics(ranks, ndcgs):
    arr = np.asarray(ranks, dtype=np.float32)
    return {"Recall@1": float((arr <= 1).mean()), "Recall@5": float((arr <= 5).mean()),
            "Recall@10": float((arr <= 10).mean()), "MRR@10": float(np.where(arr <= 10, 1 / arr, 0).mean()),
            "nDCG@5": float(np.mean(ndcgs)), "n": int(len(arr))}


def ndcg(order, gold, k=5):
    dcg = sum((1.0 if cid in gold else 0.0) / math.log2(i + 2) for i, cid in enumerate(order[:k]))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


class TokenScorer:
    def __init__(self, ids: list[str], tokens: torch.Tensor, offsets: list[int]):
        self.ids = ids
        self.pos = {cid: i for i, cid in enumerate(ids)}
        self.tokens = tokens
        self.doc_index = torch.repeat_interleave(
            torch.arange(len(ids)), torch.tensor(offsets[1:]) - torch.tensor(offsets[:-1]))

    def to(self, device):
        self.tokens = self.tokens.to(device)
        self.doc_index = self.doc_index.to(device)
        return self

    @torch.inference_mode()
    def score(self, q: torch.Tensor) -> np.ndarray:
        q = q.to(dtype=self.tokens.dtype, device=self.tokens.device)
        sims = q @ self.tokens.T
        maxima = torch.full((q.size(0), len(self.ids)), -1e4, device=self.tokens.device, dtype=torch.float16)
        maxima.scatter_reduce_(1, self.doc_index.unsqueeze(0).expand(q.size(0), -1), sims,
                               reduce="amax", include_self=True)
        return maxima.sum(dim=0).float().cpu().numpy()


def load_bge(device):
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-m3", cache_dir=str(HF))
    model = AutoModel.from_pretrained("BAAI/bge-m3", cache_dir=str(HF), dtype=torch.float16).to(device).eval()
    path = hf_hub_download("BAAI/bge-m3", "colbert_linear.pt", cache_dir=str(HF))
    linear = torch.nn.Linear(1024, 1024).to(device)
    linear.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    linear.eval()
    return tok, model, linear


def encode_bge_query(tok, model, linear, device, text: str, projected: bool) -> torch.Tensor:
    with torch.inference_mode():
        enc = tok([text], truncation=True, max_length=64, return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state.float()
        if projected:
            hidden = linear(hidden)[:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
        else:
            mask = enc["attention_mask"].bool()
        return F.normalize(hidden[0][mask[0]], dim=-1).to(torch.float16)


class BgeQueryScorer:
    def __init__(self, token_scorer, tok, model, linear, device, projected):
        self.token_scorer = token_scorer
        self.tok, self.model, self.linear = tok, model, linear
        self.device, self.projected = device, projected
        self.pos = token_scorer.pos

    def score(self, text: str) -> np.ndarray:
        q = encode_bge_query(self.tok, self.model, self.linear, self.device, text, self.projected)
        return self.token_scorer.score(q)


class XmQueryScorer:
    def __init__(self, token_scorer, encoder):
        self.token_scorer = token_scorer
        self.encoder = encoder
        self.pos = token_scorer.pos

    def score(self, text: str) -> np.ndarray:
        q = self.encoder.encode_query([text])[0].to(torch.float16).to(self.token_scorer.tokens.device)
        return self.token_scorer.score(q)


def build_bge_tokens(records, ids, tok, model, linear, device, projected: bool):
    rows, offsets = [], [0]
    with torch.inference_mode():
        for start in range(0, len(ids), 32):
            batch = [records[cid]["encoder_text"] for cid in ids[start:start + 32]]
            enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state.float()
            if projected:
                hidden = linear(hidden)[:, 1:]
                mask = enc["attention_mask"][:, 1:].bool()
            else:
                mask = enc["attention_mask"].bool()
            hidden = F.normalize(hidden, dim=-1)
            for i in range(hidden.size(0)):
                vec = hidden[i][mask[i]].to(torch.float16).cpu()
                rows.append(vec)
                offsets.append(offsets[-1] + vec.size(0))
            if start % 512 == 0:
                print(f"  bge {'proj' if projected else 'raw'}: {min(start + 32, len(ids))}/{len(ids)}", flush=True)
    return TokenScorer(ids, torch.cat(rows), offsets)


def build_colbert_xm(records, ids, device):
    from sentence_transformers import MultiVectorEncoder
    encoder = MultiVectorEncoder("antoinelouis/colbert-xm", cache_folder=str(HF))
    rows, offsets = [], [0]
    for start in range(0, len(ids), 16):
        batch = [records[cid]["encoder_text"] for cid in ids[start:start + 16]]
        vecs = encoder.encode_document(batch)
        for vec in vecs:
            vec = vec.to(torch.float16).cpu()
            rows.append(vec)
            offsets.append(offsets[-1] + vec.size(0))
        if start % 256 == 0:
            print(f"  colbert-xm: {min(start + 16, len(ids))}/{len(ids)}", flush=True)
    return TokenScorer(ids, torch.cat(rows), offsets), encoder


def build_bge_dense(records, meetings, ids, tok, model, device, late: bool):
    if not late:
        vecs = []
        with torch.inference_mode():
            for start in range(0, len(ids), 64):
                batch = [records[cid]["encoder_text"] for cid in ids[start:start + 64]]
                enc = tok(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
                hidden = model(**enc).last_hidden_state.float()
                vecs.append(F.normalize(hidden[:, 0], dim=-1).cpu())
        return torch.cat(vecs, dim=0)
    by_id = {}
    with torch.inference_mode():
        for index, (mid, meeting) in enumerate(meetings.items(), 1):
            prefix = f"Réunion : {meeting.get('title', mid)}.\n"
            parts = [prefix]
            spans, pos = {}, len(prefix)
            for cid in meeting["messages"]:
                seg = f"{records[cid]['username']} : {records[cid]['message_text']}\n"
                spans[cid] = (pos, pos + len(seg))
                parts.append(seg)
                pos += len(seg)
            doc = "".join(parts)
            enc = tok(doc, return_offsets_mapping=True, truncation=True, max_length=8192,
                      return_tensors="pt")
            offsets = enc["offset_mapping"][0].tolist()
            enc = {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            hidden = F.normalize(model(**enc).last_hidden_state[0].float(), dim=-1)
            for cid in meeting["messages"]:
                start, end = spans[cid]
                idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
                by_id[cid] = F.normalize(hidden[idx].mean(0), dim=-1).cpu() if idx else torch.zeros(hidden.size(1))
            if index % 10 == 0:
                print(f"  bge late: {index}/{len(meetings)} meetings", flush=True)
    return torch.stack([by_id[cid] for cid in ids])


def evaluate(engine, ds, scorer, queries, dense=False):
    ranks, ndcgs = [], []
    for q in queries:
        mid = q.get("meeting_id")
        candidates = ds["by_meeting"][mid]
        if dense:
            qv = scorer(q["query"])
            mapping = {cid: i for i, cid in enumerate(ds["_dense_ids"])}
            scores = np.asarray([float(qv @ ds["_dense"][mapping[cid]]) for cid in candidates])
        else:
            scores = scorer.score(q["query"])
            mapping = getattr(scorer, "pos", None) or getattr(scorer, "id_to_pos")
            scores = np.asarray([scores[mapping[cid]] for cid in candidates])
        order = [candidates[i] for i in np.argsort(-scores)]
        gold = gold_ids(q)
        positions = [order.index(cid) + 1 for cid in gold if cid in order]
        ranks.append(min(positions) if positions else len(order) + 1)
        ndcgs.append(ndcg(order, gold))
    return metrics(ranks, ndcgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dataset", default="transcribed")
    ap.add_argument("--models", default="bge_raw,bge_proj,colbert_xm,bge_dense,bge_late")
    args = ap.parse_args()
    device = args.device

    engine = me.MeetingMemoryEngine(device=device)
    ds = engine._get_dataset(args.dataset)
    ids = list(ds["records"])
    queries = ds["queries"] if args.dataset != "synthetic" else [
        q for q in ds["queries"] if q.get("split") == "test"]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    output = {}

    tok, bge_model, linear = load_bge(device)
    if "bge_raw" in models:
        raw = hr.MultiVectorIndex(ds["records"], args.dataset, device=device)
        raw.build()
        output["bge_raw"] = evaluate(engine, ds, raw, queries)
        print("bge_raw", output["bge_raw"], flush=True)
    if "bge_proj" in models:
        proj = build_bge_tokens(ds["records"], ids, tok, bge_model, linear, device, True).to(device)
        output["bge_proj"] = evaluate(engine, ds, BgeQueryScorer(proj, tok, bge_model, linear, device, True), queries)
        del proj
        torch.cuda.empty_cache()
        print("bge_proj", output["bge_proj"], flush=True)
    if "colbert_xm" in models:
        xm, encoder = build_colbert_xm(ds["records"], ids, device)
        xm.to(device)
        output["colbert_xm"] = evaluate(engine, ds, XmQueryScorer(xm, encoder), queries)
        del xm, encoder
        torch.cuda.empty_cache()
        print("colbert_xm", output["colbert_xm"], flush=True)

    def embed_bge(texts):
        vecs = []
        with torch.inference_mode():
            for start in range(0, len(texts), 64):
                enc = tok(texts[start:start + 64], padding=True, truncation=True, max_length=128,
                          return_tensors="pt").to(device)
                h = bge_model(**enc).last_hidden_state.float()
                vecs.append(F.normalize(h[:, 0], dim=-1).cpu())
        return torch.cat(vecs, dim=0)

    if "bge_dense" in models or "bge_late" in models:
        ds["_dense_ids"] = ids
        if "bge_dense" in models:
            ds["_dense"] = embed_bge([ds["records"][cid]["encoder_text"] for cid in ids])
            output["bge_dense"] = evaluate(engine, ds, lambda q: embed_bge([q])[0], queries, dense=True)
            print("bge_dense", output["bge_dense"], flush=True)
        if "bge_late" in models:
            ds["_dense"] = build_bge_dense(ds["records"], ds["meetings"], ids, tok, bge_model, device, True)
            output["bge_late"] = evaluate(engine, ds, lambda q: embed_bge([q])[0], queries, dense=True)
            print("bge_late", output["bge_late"], flush=True)

    ART.mkdir(exist_ok=True)
    path = ART / "proper_colbert_eval.json"
    combined = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    combined[args.dataset] = output
    path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", path)


if __name__ == "__main__":
    main()
