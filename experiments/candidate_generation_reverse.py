"""Candidate generation in the deployed direction: incoming new -> antecedent old.

The old->new read was a stress test, not the query the architecture receives.
Here the query is the gold new turn; the target is the gold old turn.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings, load_pairs, pair_rows
from engine.link.thread import detect_topic
from engine.store.fts import tokenize

STOP = set("le la les un une des de du au aux et ou mais donc pour dans sur avec sans ce cette ces il elle ils elles on nous vous je tu se sa son ses leur leurs est sont a ai as ont ete etre avoir plus moins tres bien tout tous toute toutes".split())


def active_topics(turns):
    out, active = [], None
    for t in turns:
        topic = detect_topic(str(t.get("text") or ""))
        active = topic or active
        out.append(active)
    return out


def rrf(lists, k=20):
    score = Counter()
    for order in lists:
        for r, j in enumerate(order):
            score[j] += 1.0 / (60 + r + 1)
    return [j for j, _ in score.most_common(k)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/candidate_generation/reverse.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    cache = np.load(root / "runs/state_reader/turn_embeddings.npz")
    mids = sorted(pairs)
    V = {m: cache[m] for m in mids}
    from bm25s import BM25
    rows = []
    for mid, topic, old, new in pair_rows(pairs):
        turns = meetings[mid]
        texts = [str(t.get("text") or "") for t in turns]
        bm = BM25(); bm.index([tokenize(t) for t in texts])
        topics = active_topics(turns)
        qidx = new
        qctx = " ".join(texts[max(0, qidx - 2):qidx + 1])
        q_topic = topics[qidx]
        def dense(qv):
            s = V[mid] @ qv
            return [int(j) for j in np.argsort(-s) if int(j) != qidx]
        d_query = dense(V[mid][qidx])
        cv = V[mid][max(0, qidx - 2):qidx + 1].mean(axis=0)
        cv = cv / max(float(np.linalg.norm(cv)), 1e-12)
        d_ctx = dense(cv)
        def bm_order(q):
            idx, _ = bm.retrieve([tokenize(q)], k=min(len(texts) - 1, 80),
                                 sorted=True, show_progress=False)
            return [int(j) for j in idx[0].tolist() if int(j) != qidx]
        b_query = bm_order(texts[qidx])
        b_ctx = bm_order(qctx)
        tp = [j for j in range(len(texts)) if j != qidx and q_topic and topics[j] == q_topic]
        tp = sorted(tp, key=lambda j: -float(V[mid][j] @ V[mid][qidx]))
        docs = [set(tokenize(t)) - STOP for t in texts]
        df = Counter(t for d in docs for t in d)
        q_tokens = set(tokenize(qctx)) - STOP
        rare = {t for t in q_tokens if df.get(t, 0) <= max(3, len(texts) // 20)}
        ent_score = Counter()
        for j, d in enumerate(docs):
            if j == qidx:
                continue
            hit = d & rare
            if hit:
                ent_score[j] = sum(1.0 / math.log(2 + df[t]) for t in hit)
        ent = [j for j, _ in ent_score.most_common(80)]
        near = [j for j in range(max(0, qidx - 5), min(len(texts), qidx + 6))
                if j != qidx]
        source = {"dense_query": d_query, "bm25_query": b_query,
                  "dense_context": d_ctx, "bm25_context": b_ctx,
                  "topic": tp, "entity": ent, "near": near}
        rank_maps = {name: {j: i + 1 for i, j in enumerate(order)}
                     for name, order in source.items()}
        pool = set().union(*(order[:20] for order in source.values()))
        def best_rank(j):
            return min(rank_maps[name].get(j, 10**6) for name in source)
        union = sorted(pool, key=lambda j: (
            best_rank(j), rank_maps["entity"].get(j, 10**6),
            rank_maps["bm25_context"].get(j, 10**6), j))
        fused = rrf([d_query[:50], b_query[:50], tp[:50], ent[:50]], k=80)
        union_near = sorted(set(union) | set(near),
                            key=lambda j: (best_rank(j), abs(j - qidx), j))
        rec = {"meeting": mid, "old": old, "new": new, "query_turn": new,
               "target_old": old}
        for name, order in {**source, "fusion": fused, "source_union": union,
                            "source_union_near": union_near}.items():
            rec[name] = order.index(old) + 1 if old in order else None
            rec[f"{name}_in20"] = old in order[:20]
            rec[f"{name}_in50"] = old in order[:50]
        rows.append(rec)
    agg = {}
    for name in ("dense_query", "bm25_query", "dense_context", "bm25_context",
                 "topic", "entity", "near", "fusion", "source_union",
                 "source_union_near"):
        agg[name] = {"at_20": sum(r[f"{name}_in20"] for r in rows),
                     "at_50": sum(r[f"{name}_in50"] for r in rows),
                     "mrr_at_50": round(sum(1.0 / r[name] for r in rows
                                            if r[name] and r[name] <= 50) / len(rows), 4)}
    out = {"n": len(rows), "direction": "new query -> old antecedent",
           "variants": agg, "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"n": len(rows), "variants": agg}, indent=1))
    for r in rows:
        print(f"{r['meeting']} {r['old']:>3}<-{r['new']:<3} "
              f"dense={r['dense_query']} bm25={r['bm25_query']} "
              f"topic={r['topic']} ent={r['entity']} fuse={r['fusion']} "
              f"union={r['source_union']} near={r['source_union_near']}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
