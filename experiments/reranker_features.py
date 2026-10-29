"""LOMO reranker over candidate features for the 40 real pairs.

The candidate pool is fixed (v2). Each candidate is scored by source ranks,
dense/BM25, lexical/entity overlap, query-context cross-encoder and the learned
relation scorer. Train on five meetings, test the held-out meeting.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings
from relation_graph_real import (BUNDLE, add_hard_negatives,
                                             read_jsonl, synth_rows)
from engine.link.cross_encoder_linker import Candidate, CrossEncoderLinker
from engine.link.relation_scorer import PairMeta, RelationScorer, features
from engine.link.thread import detect_topic
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
    ap.add_argument("--out", default=str(repo_root() / "runs/reranker_features/result.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(root)
    stage = json.loads((root / "runs/continuity_stages_v2/result.json").read_text())
    mids = sorted({r["meeting"] for r in stage["rows"]})
    V = {m: np.load(root / "runs/state_reader/turn_embeddings.npz")[m] for m in mids}
    from bm25s import BM25
    bms = {}
    topics = {}
    for mid in mids:
        texts = [str(t.get("text") or "") for t in meetings[mid]]
        bm = BM25(); bm.index([tokenize(t) for t in texts])
        bms[mid] = bm; topics[mid] = active_topics(meetings[mid])

    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    units_tr = read_jsonl(root / BUNDLE / "units" / "train.jsonl")
    units_va = read_jsonl(root / BUNDLE / "units" / "validation.jsonl")
    txt = [u["statement"] for u in units_tr + units_va]
    UV = enc.encode(txt, batch_size=64, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
    uvec = {u["unit_id"]: UV[i] for i, u in enumerate(units_tr + units_va)}
    tr = add_hard_negatives(synth_rows(root, "train", uvec), units_tr, uvec, n=500)
    scorer = RelationScorer().fit(tr)
    ce = CrossEncoderLinker("BAAI/bge-reranker-v2-m3", device="cuda")

    def lex(a, b):
        qa = set(tokenize(a)) - STOP; qb = set(tokenize(b)) - STOP
        return (len(qa & qb) / max(1, len(qa | qb)),
                len(qa & qb) / max(1, min(len(qa), len(qb))),
                len(set(tokenize(a)) & set(tokenize(b))))

    rows = []
    for r in stage["rows"]:
        mid, old, new = r["meeting"], r["old"], r["new"]
        turns = meetings[mid]
        cand = r["candidate_turns"]
        if not cand:
            continue
        texts = [str(t.get("text") or "") for t in turns]
        qtext = texts[new]
        qctx = " || ".join(texts[max(0, new - 2):new + 1])
        q_topic = topics[mid][new]
        # source ranks over all turns
        d_scores = V[mid] @ V[mid][new]
        d_order = [int(j) for j in np.argsort(-d_scores) if int(j) != new]
        b_idx, _ = bms[mid].retrieve([tokenize(qctx)],
                                     k=min(len(texts) - 1, 80), sorted=True,
                                     show_progress=False)
        b_order = [int(j) for j in b_idx[0].tolist() if int(j) != new]
        t_order = [j for j in range(len(texts)) if j != new and q_topic
                   and topics[mid][j] == q_topic]
        docs = [set(tokenize(t)) - STOP for t in texts]
        df = Counter(t for d in docs for t in d)
        q_tokens = set(tokenize(qctx)) - STOP
        rare = {t for t in q_tokens if df.get(t, 0) <= max(3, len(texts) // 20)}
        ent_score = Counter()
        for j, d in enumerate(docs):
            if j == new:
                continue
            hit = d & rare
            if hit:
                ent_score[j] = sum(1.0 / math.log(2 + df[t]) for t in hit)
        e_order = [j for j, _ in ent_score.most_common(80)]
        maps = {"dense": {j: i + 1 for i, j in enumerate(d_order)},
                "bm25": {j: i + 1 for i, j in enumerate(b_order)},
                "topic": {j: i + 1 for i, j in enumerate(t_order)},
                "entity": {j: i + 1 for i, j in enumerate(e_order)}}
        cands = [Candidate(turn=j, text=texts[j],
                           speaker=str(turns[j].get("speaker") or ""),
                           timestamp=str(turns[j].get("timestamp") or ""))
                 for j in cand]
        ce_q = ce.scores(qctx, cands, turns, style="plain")
        ce_qctx = ce.scores(qctx, cands, turns, style="context")
        ce_meta = ce.scores(qctx, cands, turns, style="metadata_context")
        for c, s0, s1, s2 in zip(cands, ce_q, ce_qctx, ce_meta):
            j = c.turn
            x = features(qtext, texts[j], V[mid][new], V[mid][j],
                         PairMeta(mid, str(turns[new].get("speaker") or ""), new, ""),
                         PairMeta(mid, str(turns[j].get("speaker") or ""), j, ""))
            ps = scorer.score_one(x)
            jac, cov, shared = lex(qtext, texts[j])
            rows.append({"mid": mid, "old": old, "new": new, "turn": j,
                         "label": int(j == old), "cand_rank": cand.index(j) + 1,
                         "dense_rank": maps["dense"].get(j, 10**6),
                         "bm25_rank": maps["bm25"].get(j, 10**6),
                         "topic_rank": maps["topic"].get(j, 10**6),
                         "entity_rank": maps["entity"].get(j, 10**6),
                         "dist": abs(j - new), "same_speaker": int(
                             str(turns[j].get("speaker") or "") ==
                             str(turns[new].get("speaker") or "")),
                         "jaccard": jac, "coverage": cov, "shared_tokens": shared,
                         "rel_p": ps.p_same, "rel_sup": ps.probs.get("supersedes", 0.0),
                         "ce_q": s0, "ce_qctx": s1, "ce_meta": s2,
                         "ce_margin": s1 - s0})
    keys = ["cand_rank", "dense_rank", "bm25_rank", "topic_rank", "entity_rank",
            "dist", "same_speaker", "jaccard", "coverage", "shared_tokens",
            "rel_p", "rel_sup", "ce_q", "ce_qctx", "ce_meta", "ce_margin"]
    folds = []
    for held in sorted({r["mid"] for r in rows}):
        train = [r for r in rows if r["mid"] != held]
        test = [r for r in rows if r["mid"] == held]
        X = np.asarray([[r[k] for k in keys] for r in train], dtype=float)
        y = np.asarray([r["label"] for r in train], dtype=int)
        clf = make_pipeline(StandardScaler(), LogisticRegression(
            class_weight="balanced", max_iter=5000, C=0.5))
        clf.fit(X, y)
        scores = clf.predict_proba(np.asarray([[r[k] for k in keys] for r in test]))[:, 1]
        by_pair = {}
        for r, s in zip(test, scores):
            by_pair.setdefault((r["old"], r["new"]), []).append((s, r["turn"]))
        local = []
        for (old, new), items in by_pair.items():
            if not any(r["label"] for r in test if r["old"] == old and r["new"] == new):
                continue
            order = [j for _, j in sorted(items, key=lambda t: -t[0])]
            rank = order.index(old) + 1
            local.append({"old": old, "new": new, "rank": rank,
                          "top": order[:5]})
        folds.append({"held": held, "n": len(local),
                      "at_1": sum(x["rank"] == 1 for x in local),
                      "at_5": sum(x["rank"] <= 5 for x in local),
                      "mrr": round(sum(1.0 / x["rank"] for x in local) / len(local), 4)
                      if local else 0, "rows": local})
    agg = {"n": sum(f["n"] for f in folds),
           "at_1": sum(f["at_1"] for f in folds),
           "at_5": sum(f["at_5"] for f in folds),
           "mrr": round(sum(f["mrr"] * f["n"] for f in folds) /
                        max(1, sum(f["n"] for f in folds)), 4)}
    out = {"n_candidates": len(rows), "features": keys, "aggregate": agg,
           "folds": folds, "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1))
    for f in folds:
        print(f["held"], "n", f["n"], "at1", f["at_1"], "at5", f["at_5"],
              "mrr", f["mrr"])
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
