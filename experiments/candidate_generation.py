"""Candidate-generation ablations on the 40 real pairs.

Goal: move candidate recall@20 before touching the reranker. Sources:
dense, BM25, active topic, rare-token/entity, RRF fusion, neighbour expansion.
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


def active_topics(turns: list[dict]) -> list[str | None]:
    out, active = [], None
    for t in turns:
        topic = detect_topic(str(t.get("text") or ""))
        if topic:
            active = topic
        out.append(active)
    return out


def rrf(lists: list[list[int]], k: int = 20) -> list[int]:
    score = Counter()
    for order in lists:
        for r, j in enumerate(order):
            score[j] += 1.0 / (60 + r + 1)
    return [j for j, _ in score.most_common(k)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/candidate_generation/result.json"))
    ap.add_argument("--k", type=int, default=20)
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
        qctx = " ".join(texts[max(0, old - 2):old + 1])
        q_topic = topics[old]
        # dense old / context
        def dense(qv):
            s = V[mid] @ qv
            return [int(j) for j in np.argsort(-s) if int(j) != old]
        d_old = dense(V[mid][old])
        ctxv = V[mid][max(0, old - 2):old + 1].mean(axis=0)
        ctxv = ctxv / max(float(np.linalg.norm(ctxv)), 1e-12)
        d_ctx = dense(ctxv)
        # Topic-expanded query: old wording plus the active topic's centroid.
        if q_topic:
            topic_idx = [j for j in range(len(texts)) if topics[j] == q_topic]
            tvec = V[mid][topic_idx].mean(axis=0) if topic_idx else V[mid][old]
            tvec = tvec / max(float(np.linalg.norm(tvec)), 1e-12)
            d_topic = dense(V[mid][old] + tvec)
        else:
            d_topic = d_old
        def bm_order(q):
            idx, _ = bm.retrieve([tokenize(q)], k=min(len(texts) - 1, 80),
                                 sorted=True, show_progress=False)
            return [int(j) for j in idx[0].tolist() if int(j) != old]
        b_old = bm_order(texts[old])
        b_ctx = bm_order(qctx)
        b_topic = bm_order(texts[old] + " " + (q_topic or ""))
        # topic index: turns carrying the active topic at old
        tp = [j for j in range(len(texts)) if j != old and q_topic and topics[j] == q_topic]
        tp = sorted(tp, key=lambda j: -float(V[mid][j] @ V[mid][old]))
        # rare-token/entity index
        docs = [set(tokenize(t)) - STOP for t in texts]
        df = Counter(t for d in docs for t in d)
        q_tokens = set(tokenize(qctx)) - STOP
        rare = {t for t in q_tokens if df.get(t, 0) <= max(3, len(texts) // 20)}
        ent_score = Counter()
        for j, d in enumerate(docs):
            if j == old:
                continue
            hit = d & rare
            if hit:
                ent_score[j] = sum(1.0 / math.log(2 + df[t]) for t in hit)
        ent = [j for j, _ in ent_score.most_common(80)]
        fused = rrf([d_ctx[:50], b_ctx[:50], tp[:50], ent[:50]], k=80)
        neigh = []
        for j in fused[:20]:
            for z in (j - 2, j - 1, j, j + 1, j + 2):
                if 0 <= z < len(texts) and z != old and z not in neigh:
                    neigh.append(z)
        source = {
            "dense_old": d_old, "bm25_old": b_old,
            "dense_context": d_ctx, "bm25_context": b_ctx,
            "dense_topic": d_topic, "bm25_topic": b_topic,
            "topic": tp, "entity": ent,
        }
        rank_maps = {name: {j: i + 1 for i, j in enumerate(order)}
                     for name, order in source.items()}
        pool = set().union(*(order[:20] for order in source.values()))
        def best_rank(j):
            return min(rank_maps[name].get(j, 10**6) for name in source)
        union_best = sorted(pool, key=lambda j: (best_rank(j), j))
        # A neighbour of the query is a valid candidate even when no source
        # ranks it; this is how "payer des influenceurs" reaches "on les invite".
        near = [j for j in range(max(0, old - 2), min(len(texts), old + 3))
                if j != old]
        union_near = sorted(set(union_best) | set(near),
                            key=lambda j: (best_rank(j), abs(j - old), j))
        variants = {
            **source,
            "fusion": fused, "fusion_neighbors": rrf([fused[:50], neigh[:50]], k=80),
            "source_union": union_best,
            "source_union_near": union_near,
        }
        rec = {"meeting": mid, "topic": topic, "old": old, "new": new,
               "active_topic": q_topic,
               "query_context": qctx[:400]}
        for name, order in variants.items():
            rec[name] = order.index(new) + 1 if new in order else None
            rec[f"{name}_in20"] = new in order[:args.k]
            rec[f"{name}_in50"] = new in order[:50]
        rows.append(rec)
    agg = {}
    for name in ("dense_old", "bm25_old", "dense_context", "bm25_context",
                 "dense_topic", "bm25_topic", "topic", "entity", "fusion",
                 "fusion_neighbors", "source_union", "source_union_near"):
        agg[name] = {
            "at_20": sum(r[f"{name}_in20"] for r in rows),
            "at_50": sum(r[f"{name}_in50"] for r in rows),
            "mrr_at_50": round(sum(1.0 / r[name] for r in rows
                                   if r[name] and r[name] <= 50) / len(rows), 4),
        }
    out = {"n": len(rows), "k": args.k, "variants": agg, "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"n": len(rows), "variants": agg}, indent=1))
    for r in rows:
        print(f"{r['meeting']} {r['old']:>3}->{r['new']:<3} "
              f"dense={r['dense_old']} bm25={r['bm25_old']} "
              f"topic={r['topic']} ent={r['entity']} dtopic={r['dense_topic']} "
              f"btopic={r['bm25_topic']} fuse={r['fusion']} "
              f"neigh={r['fusion_neighbors']} union={r['source_union']} "
              f"union_near={r['source_union_near']}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
