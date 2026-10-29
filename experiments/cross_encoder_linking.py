"""Compare the learned pair scorer with a pretrained cross-encoder on real pairs.

Same candidate pool for both: dense top20 union BM25 top20. Cross-encoder is a
relevance model, not a relation classifier; the rank read is the fair read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.config import repo_root
from continuity_stages import (bm25_index, bm25_rank, dense_rank,
                                           dense_vecs)
from engine.eval.real_pairs import load_meetings, load_pairs, pair_rows
from engine.link.cross_encoder_linker import Candidate, CrossEncoderLinker


def metrics(rows, key):
    n = len(rows)
    vals = [r.get(key) for r in rows]
    present = [v for v in vals if v is not None]
    return {
        "coverage": len(present),
        "at_1": sum(v == 1 for v in present),
        "at_5": sum(v is not None and v <= 5 for v in present),
        "mrr_all": round(sum(1.0 / v for v in present) / n, 4) if n else 0.0,
        "mrr_covered": round(sum(1.0 / v for v in present) / len(present), 4)
        if present else 0.0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default=str(repo_root() / "runs/cross_encoder/result.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    V = dense_vecs(root, meetings)
    bms = {mid: bm25_index(meetings[mid]) for mid in meetings}
    ce = CrossEncoderLinker(args.model, device="cuda")
    rows = []
    for mid, topic, old, new in pair_rows(pairs):
        turns = meetings[mid]
        d_order, _ = dense_rank(V[mid], old)
        bm, _ = bms[mid]
        b_order, _ = bm25_rank(bm, turns, old)
        pool = sorted(set(d_order[:20]) | set(b_order[:20]))
        old_t = turns[old]
        cands = [Candidate(turn=j, text=turns[j]["text"],
                           speaker=str(turns[j].get("speaker") or ""),
                           timestamp=str(turns[j].get("timestamp") or ""))
                 for j in pool]
        rec = {"meeting": mid, "topic": topic, "old": old, "new": new,
               "pool": pool, "new_in_pool": new in pool,
               "old_text": old_t["text"], "new_text": turns[new]["text"]}
        for style in ("plain", "metadata", "context", "metadata_context"):
            order = ce.rank(old_t["text"], cands, turns, style=style)
            rec[f"ce_{style}_rank"] = order.index(new) + 1 if new in order else None
        # current scorer rank from the stage table is loaded separately
        rows.append(rec)
    stage = json.loads((root / "runs/continuity_stages/result.json").read_text())
    by = {(r["meeting"], r["old"], r["new"]): r for r in stage["rows"]}
    for r in rows:
        s = by[(r["meeting"], r["old"], r["new"])]
        r["current_scorer_rank"] = s["rerank_rank"]
        r["candidate_missing"] = not r["new_in_pool"]
    out = {
        "model": args.model,
        "n": len(rows),
        "candidate_pool": "dense@20 union bm25@20",
        "new_in_pool": sum(r["new_in_pool"] for r in rows),
        "metrics": {
            "current_scorer": metrics(rows, "current_scorer_rank"),
            **{f"ce_{s}": metrics(rows, f"ce_{s}_rank")
               for s in ("plain", "metadata", "context", "metadata_context")},
            "ce_style_conditional": {
                s: {
                    "at_1": sum(r[f"ce_{s}_rank"] == 1 for r in rows
                                if r["new_in_pool"]),
                    "at_5": sum(r[f"ce_{s}_rank"] is not None
                                and r[f"ce_{s}_rank"] <= 5 for r in rows
                                if r["new_in_pool"]),
                    "n": sum(1 for r in rows if r["new_in_pool"]),
                } for s in ("plain", "metadata", "context", "metadata_context")
            },
        },
        "rows": rows,
    }
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
