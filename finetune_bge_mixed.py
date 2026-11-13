#!/usr/bin/env python3
"""Style-balanced BGE-M3 ColBERT projection fine-tune.

v7 train + messy-transcription train (project-disjoint); 50/50 batches.
Evaluates on v7 test and held-out messy test.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import memory_engine as me
from finetune_bge_colbert import load_bge, encode_base, project, metrics, gids


def encode_colbert_base(tok, model, texts, max_len, device):
    # BGE-M3 multi-vector API drops the CLS state: colbert_linear(last_hidden_state[:, 1:]).
    return [row[1:] for row in encode_base(tok, model, texts, max_len, device)]

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"


def messy_records():
    meetings = json.loads((DATA / "archive_v6/transcribed_meetings.json").read_text(encoding="utf-8"))
    queries = json.loads((DATA / "archive_v6/transcribed_queries.json").read_text(encoding="utf-8"))
    records = {}
    by_meeting = defaultdict(list)
    project_of = {}
    for m in meetings:
        mid = str(m["meeting_id"])
        project_of[mid] = m.get("project_id") or m.get("project") or mid
        for msg in m["messages"]:
            cid = f"{mid}:{int(msg['message_id']):03d}"
            records[cid] = {
                "chunk_id": cid, "meeting_id": mid, "username": msg.get("username", ""),
                "message_text": msg.get("message", ""),
                "encoder_text": f"Réunion : {m.get('title', mid)}. Intervenant : {msg.get('username', '')}. Intervention : {msg.get('message', '')}",
            }
            by_meeting[mid].append(cid)
    projects = sorted(set(project_of.values()))
    split_of = {}
    for i, p in enumerate(projects):
        split_of[p] = "train" if i < max(1, int(len(projects) * 0.6)) else ("val" if i < len(projects) - 2 else "test")
    for q in queries:
        q["style"] = "messy"
        q["split"] = split_of.get(project_of.get(q["meeting_id"]), "train")
    return records, by_meeting, queries, split_of


def v7_records():
    engine = me.MeetingMemoryEngine(device="cuda")
    ds = engine._get_dataset("synthetic")
    queries = json.loads((DATA / "french_retrieval_queries.json").read_text(encoding="utf-8"))
    for q in queries:
        q["style"] = "v7"
    return ds["records"], ds["by_meeting"], queries


def fill_negs(q, by_meeting, rng, n=5):
    gold = set(gids(q))
    pool = [c for c in by_meeting.get(q.get("meeting_id"), []) if c not in gold]
    if len(pool) < n:
        allids = [c for mid in by_meeting.values() for c in mid if c not in gold]
        pool = pool + allids
    return rng.sample(pool, min(n, len(pool)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    rng = random.Random(42)
    v7r, v7by, v7q = v7_records()
    mr, mby, mq, split_of = messy_records()
    all_docs = {**v7r, **mr}
    all_queries = v7q + mq
    tok, model, linear = load_bge(args.device)
    # official projection init
    linear.train()
    print("encoding BGE base tokens", flush=True)
    doc_tokens = {}
    for name, records in (("v7", v7r), ("messy", mr)):
        ids = list(records)
        got = []
        for start in range(0, len(ids), 64):
            batch = ids[start:start + 64]
            got += encode_colbert_base(tok, model, [records[c]["encoder_text"] for c in batch], 128, args.device)
        doc_tokens.update(dict(zip(ids, got)))
        print(" docs", name, len(ids), flush=True)
    query_tokens = {}
    for start in range(0, len(all_queries), 32):
        batch = all_queries[start:start + 32]
        got = encode_colbert_base(tok, model, [q["query"] for q in batch], 64, args.device)
        query_tokens.update({q["query_id"]: g for q, g in zip(batch, got)})
    v7_train = [q for q in v7q if q["split"] == "train"]
    messy_train = [q for q in mq if q["split"] == "train"]
    print("train", len(v7_train), len(messy_train), "messy split", split_of, flush=True)
    opt = torch.optim.AdamW(linear.parameters(), lr=args.lr)
    best = None
    best_score = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        linear.train()
        steps = max(len(v7_train) // 4, len(messy_train) // 4, 20)
        losses = []
        started = time.time()
        for _ in range(steps):
            batch = rng.sample(v7_train, min(4, len(v7_train))) + rng.sample(messy_train, min(4, len(messy_train)))
            qs = [query_tokens[q["query_id"]] for q in batch]
            ds_ = []
            for q in batch:
                ds_.append(doc_tokens[q["target_chunk_id"]])
                ds_.extend(doc_tokens[c] for c in fill_negs(q, v7by if q["style"] == "v7" else mby, rng))
            qp = [project(linear, x, args.device) for x in qs]
            dp = [project(linear, x, args.device) for x in ds_]
            scores = []
            for i in range(len(qp)):
                scores.append(torch.stack([(qp[i] @ dp[j].T).max(dim=1).values.sum()
                                           for j in range(i * 6, (i + 1) * 6)]))
            scores = torch.stack(scores)
            loss = F.cross_entropy(scores, torch.zeros(len(qp), dtype=torch.long, device=args.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(linear.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        linear.eval()
        val = {}
        for style, by, qs_ in (("v7", v7by, v7q), ("messy", mby, mq)):
            ranks = []
            for q in [x for x in qs_ if x["split"] == "val"]:
                cand = by.get(q["meeting_id"], [])
                qv = project(linear, query_tokens[q["query_id"]], args.device)
                sc = [float((qv @ project(linear, doc_tokens[c], args.device).T).max(dim=1).values.sum().cpu()) for c in cand]
                order = [cand[i] for i in np.argsort(-np.asarray(sc))]
                gold = gids(q); p = [order.index(c) + 1 for c in gold if c in order]
                ranks.append(min(p) if p else len(order) + 1)
            val[style] = metrics(ranks)
        score = 0.5 * val["v7"]["MRR@10"] + 0.5 * val["messy"]["MRR@10"]
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val": val, "score": score,
                        "seconds": round(time.time() - started, 1)})
        print("epoch", epoch, "loss", round(float(np.mean(losses)), 4), "val", val, "score", round(score, 4), flush=True)
        if score > best_score:
            best_score = score
            best = {k: v.detach().cpu().clone() for k, v in linear.state_dict().items()}
    if best:
        linear.load_state_dict(best)
    linear.eval()
    test = {}
    for style, by, qs_ in (("v7", v7by, v7q), ("messy", mby, mq)):
        ranks = []
        for q in [x for x in qs_ if x["split"] == "test"]:
            cand = by.get(q["meeting_id"], [])
            qv = project(linear, query_tokens[q["query_id"]], args.device)
            sc = [float((qv @ project(linear, doc_tokens[c], args.device).T).max(dim=1).values.sum().cpu()) for c in cand]
            order = [cand[i] for i in np.argsort(-np.asarray(sc))]
            gold = gids(q); p = [order.index(c) + 1 for c in gold if c in order]
            ranks.append(min(p) if p else len(order) + 1)
        test[style] = metrics(ranks)
    out = {"model": "BAAI/bge-m3 projection", "training": "v7 + messy, 50/50 batches",
           "val": best_score, "test": test, "history": history,
           "messy_split": split_of}
    torch.save({"projection": linear.state_dict()}, ART / "bge_mixed_style_projection.pt")
    (ART / "bge_mixed_style_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("test", test, flush=True)
    print("Saved", ART / "bge_mixed_style_eval.json")


if __name__ == "__main__":
    main()
