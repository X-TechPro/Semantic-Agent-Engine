#!/usr/bin/env python3
"""Fine-tune a ColBERT projection on frozen BGE-M3 token states for v7.

The base BGE-M3 model is frozen; only the 1024->1024 projection is trained with
a listwise MaxSim loss over one positive + five v7 hard negatives. This is the
efficient way to adapt the BGE ColBERT signal without full-model training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

import memory_engine as me

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
HF = ART / "hf_cache"
MODEL = "BAAI/bge-m3"


def gids(q):
    return [x for x in (q.get("target_chunk_ids") or [q.get("target_chunk_id")]) if x]


def metrics(ranks):
    a = np.asarray(ranks, dtype=np.float32)
    return {"Recall@1": float((a <= 1).mean()), "Recall@5": float((a <= 5).mean()),
            "Recall@10": float((a <= 10).mean()), "MRR@10": float(np.where(a <= 10, 1 / a, 0).mean()),
            "n": int(len(a))}


def load_bge(device):
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(HF))
    model = AutoModel.from_pretrained(MODEL, cache_dir=str(HF), dtype=torch.float16).to(device).eval()
    path = next((Path(HF) / "models--BAAI--bge-m3").rglob("colbert_linear.pt"))
    proj = torch.nn.Linear(1024, 1024).to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    proj.load_state_dict(state)
    return tok, model, proj


@torch.inference_mode()
def encode_base(tok, model, texts, max_len, device):
    rows = []
    for start in range(0, len(texts), 32):
        enc = tok(texts[start:start + 32], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        hidden = model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].bool()
        for i in range(hidden.size(0)):
            rows.append(hidden[i][mask[i]].to(torch.float16).cpu())
    return rows


def project(proj, tokens, device):
    return F.normalize(proj(tokens.to(device).float()), dim=-1)


def score_pairs(qs, ds_, group=6):
    scores = []
    for i in range(qs.size(0)):
        row = []
        for j in range(i * group, (i + 1) * group):
            row.append((qs[i] @ ds_[j].T).max(dim=1).values.sum())
        scores.append(torch.stack(row))
    return torch.stack(scores)


def evaluate(proj, query_tokens, doc_tokens, queries, ds, split, device):
    ids = list(doc_tokens)
    ranks = []
    for q in [x for x in queries if x.get("split") == split]:
        mid = q.get("meeting_id")
        candidates = ds["by_meeting"].get(mid, ids) if mid else ids
        qv = project(proj, query_tokens[q["query_id"]], device)
        scores = []
        for cid in candidates:
            d = project(proj, doc_tokens[cid], device)
            scores.append(float((qv @ d.T).max(dim=1).values.sum().cpu()))
        order = [candidates[i] for i in np.argsort(-np.asarray(scores))]
        gold = gids(q)
        pos = [order.index(c) + 1 for c in gold if c in order]
        ranks.append(min(pos) if pos else len(order) + 1)
    return metrics(ranks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    engine = me.MeetingMemoryEngine(device=args.device)
    ds = engine._get_dataset("synthetic")
    queries = json.loads((DATA / "french_retrieval_queries.json").read_text(encoding="utf-8"))
    train = [q for q in queries if q.get("split") == "train"]
    tok, model, proj = load_bge(args.device)
    ids = list(ds["records"])
    print(f"encoding {len(ids)} docs + {len(queries)} queries with frozen BGE-M3")
    doc_tokens = dict(zip(ids, encode_base(tok, model, [ds["records"][c]["encoder_text"] for c in ids], 128, args.device)))
    query_tokens = {q["query_id"]: encode_base(tok, model, [q["query"]], 64, args.device)[0] for q in queries}

    proj.train()
    opt = torch.optim.AdamW(proj.parameters(), lr=args.lr)
    best = None
    best_val = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        order = np.random.default_rng(epoch).permutation(len(train))
        losses = []
        started = time.time()
        for start in range(0, len(order), args.batch_size):
            batch = [train[i] for i in order[start:start + args.batch_size]]
            qs = [query_tokens[q["query_id"]] for q in batch]
            ds_ = []
            for q in batch:
                ds_.append(doc_tokens[q["target_chunk_id"]])
                ds_.extend(doc_tokens[c] for c in q.get("hard_negative_chunk_ids", [])[:5])
            qp = [project(proj, x, args.device) for x in qs]
            dp = [project(proj, x, args.device) for x in ds_]
            scores = []
            for i in range(len(qp)):
                scores.append(torch.stack([(qp[i] @ dp[j].T).max(dim=1).values.sum()
                                           for j in range(i * 6, (i + 1) * 6)]))
            scores = torch.stack(scores)
            loss = F.cross_entropy(scores, torch.zeros(len(qp), dtype=torch.long, device=args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        proj.eval()
        val = evaluate(proj, query_tokens, doc_tokens, queries, ds, "val", args.device)
        proj.train()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": val,
                        "seconds": round(time.time() - started, 1)})
        print(f"epoch {epoch}/{args.epochs} loss {np.mean(losses):.4f} val {val}", flush=True)
        if val["MRR@10"] > best_val:
            best_val = val["MRR@10"]
            best = proj.state_dict()
    if best:
        proj.load_state_dict(best)
    proj.eval()
    test = evaluate(proj, query_tokens, doc_tokens, queries, ds, "test", args.device)
    output = {"model": MODEL, "finetuned": "projection", "val": evaluate(proj, query_tokens, doc_tokens, queries, ds, "val", args.device),
              "test": test, "history": history}
    ART.mkdir(exist_ok=True)
    torch.save({"projection": proj.state_dict()}, ART / "bge_colbert_projection.pt")
    (ART / "bge_colbert_finetuned_eval.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test", test)
    print("Saved", ART / "bge_colbert_finetuned_eval.json")


if __name__ == "__main__":
    main()
