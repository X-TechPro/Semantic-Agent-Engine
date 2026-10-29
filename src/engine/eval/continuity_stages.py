"""Stage-by-stage error table for the 40 real supersedes pairs.

Stages:
  candidate recall (dense / BM25 / union)
  -> rerank rank (current RelationScorer)
  -> graph link
  -> graph current
  -> relation prediction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings, load_pairs, pair_rows
from engine.eval.relation_graph_real import (BUNDLE, add_hard_negatives,
                                             read_jsonl, synth_rows)
from engine.link.relation_scorer import PairMeta, RelationScorer, features
from engine.memory.thread_graph import build_thread_graph
from engine.store.fts import tokenize


def dense_vecs(root: Path, meetings: dict[str, list[dict]]):
    cache = root / "runs/state_reader/turn_embeddings.npz"
    if cache.exists():
        z = np.load(cache)
        if all(m in z.files for m in meetings):
            return {m: z[m] for m in meetings}
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    out = {}
    for mid, turns in meetings.items():
        out[mid] = enc.encode([t["text"] for t in turns], batch_size=64,
                              normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)
    return out


def bm25_index(turns: list[dict]):
    from bm25s import BM25

    corpus = [tokenize(t["text"]) for t in turns]
    bm = BM25()
    bm.index(corpus)
    return bm, corpus


def dense_rank(V: np.ndarray, old: int) -> tuple[list[int], dict[int, int]]:
    scores = V @ V[old]
    order = np.argsort(-scores)
    order = [int(j) for j in order if int(j) != old]
    return order, {int(j): i + 1 for i, j in enumerate(order)}


def bm25_rank(bm, turns: list[dict], old: int) -> tuple[list[int], dict[int, int]]:
    n = min(len(turns) - 1, 40)
    q = tokenize(turns[old]["text"])
    idx, _scores = bm.retrieve([q], k=n, sorted=True,
                               show_progress=False)
    order = [int(j) for j in idx[0].tolist() if int(j) != old]
    return order, {j: i + 1 for i, j in enumerate(order)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/continuity_stages/result.json"))
    ap.add_argument("--labels", default=str(repo_root() / "data/gold/summre_pair_stage_labels.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    mids = sorted(pairs)
    V = dense_vecs(root, meetings)

    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device="cuda")
    units_tr = read_jsonl(root / BUNDLE / "units" / "train.jsonl")
    units_va = read_jsonl(root / BUNDLE / "units" / "validation.jsonl")
    unit_texts = [u["statement"] for u in units_tr + units_va]
    UV = enc.encode(unit_texts, batch_size=64, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False)
    uvec = {u["unit_id"]: UV[i] for i, u in enumerate(units_tr + units_va)}
    tr = add_hard_negatives(synth_rows(root, "train", uvec), units_tr, uvec, n=500)
    va = synth_rows(root, "validation", uvec)
    scorer = RelationScorer().fit(tr)
    val_pred = [scorer.score_one(r["x"]).relation for r in va]
    val_gold = [r["relation"] for r in va]
    synth_f1 = float(f1_score(val_gold, val_pred, average="macro", zero_division=0))

    graphs = {mid: build_thread_graph(mid, meetings[mid], V[mid], scorer,
                                      top_k=16, threshold=0.5)
              for mid in mids}
    bms = {mid: bm25_index(meetings[mid]) for mid in mids}
    rows = []
    for mid, topic, old, new in pair_rows(pairs):
        turns = meetings[mid]
        d_order, d_rank = dense_rank(V[mid], old)
        bm, _ = bms[mid]
        b_order, b_rank = bm25_rank(bm, turns, old)
        d20 = set(d_order[:20]); b20 = set(b_order[:20]); union = d20 | b20
        pool = sorted(union)
        xs, cand = [], []
        for j in pool:
            xs.append(features(
                turns[old]["text"], turns[j]["text"], V[mid][old], V[mid][j],
                PairMeta(mid, str(turns[old].get("speaker") or ""), old, ""),
                PairMeta(mid, str(turns[j].get("speaker") or ""), j, "")))
            cand.append(j)
        scores = scorer.score_many(xs)
        rerank_order = sorted(range(len(cand)),
                              key=lambda k: (-scores[k].p_same,
                                             -scores[k].probs.get("supersedes", 0.0)))
        rank_new = next((i + 1 for i, k in enumerate(rerank_order)
                         if cand[k] == new), None)
        pair_x = features(
            turns[old]["text"], turns[new]["text"], V[mid][old], V[mid][new],
            PairMeta(mid, str(turns[old].get("speaker") or ""), old, ""),
            PairMeta(mid, str(turns[new].get("speaker") or ""), new, ""))
        pair_score = scorer.score_one(pair_x)
        g = graphs[mid]
        comp = {n.turn for n in g.component(f"{mid}:{old}")}
        cur = g.current(f"{mid}:{old}")
        connected = new in comp
        current_new = bool(cur and cur.turn == new)
        if new not in union:
            stage = "candidate_missing"
        elif rank_new is None or rank_new > 5:
            stage = "rerank_low"
        elif not connected:
            stage = "link_wrong"
        elif not current_new:
            stage = "state_wrong"
        elif pair_score.relation != "supersedes":
            stage = "wrong_relation"
        else:
            stage = "correct"
        rows.append({
            "meeting": mid, "topic": topic, "old": old, "new": new,
            "dense_rank": d_rank.get(new),
            "bm25_rank": b_rank.get(new),
            "candidate_in_top20_dense": new in d20,
            "candidate_in_top20_bm25": new in b20,
            "candidate_in_top20_union": new in union,
            "union_pool_size": len(union),
            "rerank_rank": rank_new,
            "relation_pred": pair_score.relation,
            "relation_gold": "supersedes",
            "relation_confidence": round(pair_score.p_same, 4),
            "pair_connected": connected,
            "current_turn": cur.turn if cur else None,
            "current_is_new": current_new,
            "state_correct": current_new,
            "failure_stage": stage,
            "evidence_turns": d_order[:5],
            "notes": "",
        })

    def agg(key):
        return sum(bool(r[key]) for r in rows)
    out = {
        "n": len(rows),
        "synthetic_relation_macro_f1": round(synth_f1, 4),
        "candidate_recall_at_20_dense": agg("candidate_in_top20_dense"),
        "candidate_recall_at_20_bm25": agg("candidate_in_top20_bm25"),
        "candidate_recall_at_20_union": agg("candidate_in_top20_union"),
        "rerank_at_1": sum(r["rerank_rank"] == 1 for r in rows),
        "rerank_at_5": sum(r["rerank_rank"] is not None and r["rerank_rank"] <= 5
                          for r in rows),
        "pair_connected": agg("pair_connected"),
        "current_is_new": agg("current_is_new"),
        "relation_supersedes": sum(r["relation_pred"] == "supersedes" for r in rows),
        "stages": {k: sum(r["failure_stage"] == k for r in rows)
                   for k in sorted({r["failure_stage"] for r in rows})},
        "rows": rows,
    }
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    labels = {"reviewed": False,
              "note": "auto-filled stage table; manual review pending",
              "rows": rows}
    lp = Path(args.labels); lp.parent.mkdir(parents=True, exist_ok=True)
    if not lp.exists():
        lp.write_text(json.dumps(labels, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    for r in rows:
        print(f"{r['meeting']} {r['old']:>3}->{r['new']:<3} "
              f"dense={r['dense_rank']} bm25={r['bm25_rank']} "
              f"rerank={r['rerank_rank']} edge={r['pair_connected']} "
              f"cur={r['current_turn']} stage={r['failure_stage']}")
    print("wrote", p, "and", lp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
