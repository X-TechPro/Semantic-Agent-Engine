"""Stage table in the deployed direction: incoming new -> antecedent old.

Candidate pool: dense/BM25/topic/entity/query-neighbour sources, union top20 by
best rank. Then the current relation scorer reranks that pool. Graph/current
are still measured old->new because the stored edge direction is temporal.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings, load_pairs, pair_rows
from engine.eval.relation_graph_real import (BUNDLE, add_hard_negatives,
                                             read_jsonl, synth_rows)
from engine.link.relation_scorer import PairMeta, RelationScorer, features
from engine.link.thread import detect_topic
from engine.memory.thread_graph import build_thread_graph
from engine.store.fts import tokenize

STOP = set("le la les un une des de du au aux et ou mais donc pour dans sur avec sans ce cette ces il elle ils elles on nous vous je tu se sa son ses leur leurs est sont a ai as ont ete etre avoir plus moins tres bien tout tous toute toutes".split())


def active_topics(turns):
    out, active = [], None
    for t in turns:
        active = detect_topic(str(t.get("text") or "")) or active
        out.append(active)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/continuity_stages_v2/result.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    mids = sorted(pairs)
    cache = np.load(root / "runs/state_reader/turn_embeddings.npz")
    V = {m: cache[m] for m in mids}
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    units_tr = read_jsonl(root / BUNDLE / "units" / "train.jsonl")
    units_va = read_jsonl(root / BUNDLE / "units" / "validation.jsonl")
    texts = [u["statement"] for u in units_tr + units_va]
    UV = enc.encode(texts, batch_size=64, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
    uvec = {u["unit_id"]: UV[i] for i, u in enumerate(units_tr + units_va)}
    tr = add_hard_negatives(synth_rows(root, "train", uvec), units_tr, uvec, n=500)
    va = synth_rows(root, "validation", uvec)
    scorer = RelationScorer().fit(tr)
    synth_f1 = float(f1_score([r["relation"] for r in va],
                              [scorer.score_one(r["x"]).relation for r in va],
                              average="macro", zero_division=0))
    graphs = {m: build_thread_graph(m, meetings[m], V[m], scorer,
                                    top_k=16, threshold=0.5) for m in mids}
    from bm25s import BM25
    rows = []
    for mid, topic, old, new in pair_rows(pairs):
        turns = meetings[mid]
        texts_m = [str(t.get("text") or "") for t in turns]
        bm = BM25(); bm.index([tokenize(t) for t in texts_m])
        topics = active_topics(turns)
        qidx, target = new, old
        qctx = " ".join(texts_m[max(0, qidx - 2):qidx + 1])
        q_topic = topics[qidx]
        def dense(qv):
            s = V[mid] @ qv
            return [int(j) for j in np.argsort(-s) if int(j) != qidx]
        d_q = dense(V[mid][qidx])
        cv = V[mid][max(0, qidx - 2):qidx + 1].mean(axis=0)
        cv = cv / max(float(np.linalg.norm(cv)), 1e-12)
        d_ctx = dense(cv)
        def bm_order(q):
            idx, _ = bm.retrieve([tokenize(q)], k=min(len(texts_m) - 1, 80),
                                 sorted=True, show_progress=False)
            return [int(j) for j in idx[0].tolist() if int(j) != qidx]
        b_q, b_ctx = bm_order(texts_m[qidx]), bm_order(qctx)
        tp = [j for j in range(len(texts_m)) if j != qidx and q_topic
              and topics[j] == q_topic]
        tp = sorted(tp, key=lambda j: -float(V[mid][j] @ V[mid][qidx]))
        docs = [set(tokenize(t)) - STOP for t in texts_m]
        df = Counter(t for d in docs for t in d)
        q_tokens = set(tokenize(qctx)) - STOP
        rare = {t for t in q_tokens if df.get(t, 0) <= max(3, len(texts_m) // 20)}
        ent_score = Counter()
        for j, d in enumerate(docs):
            if j == qidx:
                continue
            hit = d & rare
            if hit:
                ent_score[j] = sum(1.0 / math.log(2 + df[t]) for t in hit)
        ent = [j for j, _ in ent_score.most_common(80)]
        near = [j for j in range(max(0, qidx - 5), min(len(texts_m), qidx + 6))
                if j != qidx]
        source = {"dense": d_q, "bm25": b_q, "dense_context": d_ctx,
                  "bm25_context": b_ctx, "topic": tp, "entity": ent,
                  "near": near}
        rank_maps = {n: {j: i + 1 for i, j in enumerate(o)}
                     for n, o in source.items()}
        pool = set().union(*(o[:20] for o in source.values()))
        def best_rank(j):
            return min(rank_maps[n].get(j, 10**6) for n in source)
        cand = sorted(pool, key=lambda j: (
            best_rank(j), rank_maps["entity"].get(j, 10**6),
            rank_maps["bm25_context"].get(j, 10**6), abs(j - qidx), j))[:20]
        xs = [features(texts_m[qidx], texts_m[j], V[mid][qidx], V[mid][j],
                       PairMeta(mid, str(turns[qidx].get("speaker") or ""), qidx, ""),
                       PairMeta(mid, str(turns[j].get("speaker") or ""), j, ""))
              for j in cand]
        scores = scorer.score_many(xs)
        order = sorted(range(len(cand)),
                       key=lambda k: (-scores[k].p_same,
                                      -scores[k].probs.get("supersedes", 0.0)))
        rank_old = next((i + 1 for i, k in enumerate(order) if cand[k] == target), None)
        pair_x = features(texts_m[old], texts_m[new], V[mid][old], V[mid][new],
                          PairMeta(mid, str(turns[old].get("speaker") or ""), old, ""),
                          PairMeta(mid, str(turns[new].get("speaker") or ""), new, ""))
        rel = scorer.score_one(pair_x)
        g = graphs[mid]
        comp = {n.turn for n in g.component(f"{mid}:{old}")}
        cur = g.current(f"{mid}:{old}")
        connected = new in comp
        current_new = bool(cur and cur.turn == new)
        if target not in cand:
            stage = "candidate_missing"
        elif rank_old is None or rank_old > 5:
            stage = "rerank_low"
        elif not connected:
            stage = "link_wrong"
        elif not current_new:
            stage = "state_wrong"
        elif rel.relation != "supersedes":
            stage = "wrong_relation"
        else:
            stage = "correct"
        rows.append({"meeting": mid, "topic": topic, "old": old, "new": new,
                     "candidate_in_20": target in cand,
                     "candidate_rank": next((i + 1 for i, j in enumerate(cand)
                                             if j == target), None),
                     "candidate_turns": cand,
                     "rerank_rank": rank_old,
                     "relation_pred": rel.relation,
                     "relation_confidence": round(rel.p_same, 4),
                     "pair_connected": connected,
                     "current_turn": cur.turn if cur else None,
                     "current_is_new": current_new,
                     "failure_stage": stage})
    n = len(rows)
    out = {"n": n, "direction": "new query -> old antecedent",
           "synthetic_relation_macro_f1": round(synth_f1, 4),
           "candidate_recall_at_20": sum(r["candidate_in_20"] for r in rows),
           "rerank_at_1": sum(r["rerank_rank"] == 1 for r in rows),
           "rerank_at_5": sum(r["rerank_rank"] is not None
                              and r["rerank_rank"] <= 5 for r in rows),
           "pair_connected": sum(r["pair_connected"] for r in rows),
           "current_is_new": sum(r["current_is_new"] for r in rows),
           "stages": {k: sum(r["failure_stage"] == k for r in rows)
                      for k in sorted({r["failure_stage"] for r in rows})},
           "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        print(f"{r['meeting']} {r['old']:>3}<-{r['new']:<3} "
              f"cand={r['candidate_rank']} rerank={r['rerank_rank']} "
              f"edge={r['pair_connected']} cur={r['current_turn']} "
              f"stage={r['failure_stage']}")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
