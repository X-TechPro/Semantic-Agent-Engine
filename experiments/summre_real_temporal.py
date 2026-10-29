"""Real-corpus temporal read: SUMM-RE train+sel, top-1 current-vs-stale.

The sealed test split is not touched. For each hand-labelled supersedes pair the
query is the OLD statement's own text, mode=current; the system should return the
NEWER claim at rank 1 (or at least above the old one). Same shipped e5 +
cross-encoder as the ledger headline.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Query
from eval.frbench import build_meeting_source, summre_pair_scores

SPLITS = ("train", "sel")
GOLD = {"train": "summre_temporal_train.json", "sel": "summre_temporal_sel.json"}
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
TOP_K = 30


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--query", choices=("old", "topic"), default="topic")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = repo_root()
    if args.out is None:
        args.out = str(root / "runs/summre_real_temporal" /
                       f"result_{args.query}.json")
    raw = json.loads((root / "data/external/summre_test/meetings.json").read_text())
    meetings = {m: (v["segments"] if isinstance(v, dict) else v)
                for m, v in raw.items()}
    pairs: dict[str, list[tuple[str, int, int]]] = {}
    for split in SPLITS:
        doc = json.loads((root / "data/gold" / GOLD[split]).read_text())
        for mid, mm in doc["meetings"].items():
            if mid in SPENT:
                raise ValueError(f"{mid} is the spent external test")
            for th in mm["threads"]:
                for r in th["relations"]:
                    if r["relation"] == "supersedes":
                        pairs.setdefault(mid, []).append(
                            (str(th.get("topic") or "sujet"),
                             int(r["old"]), int(r["new"])))
    mids = sorted(pairs)
    cfg = load_config(overrides={
        "embed": {"backend": "sentence_transformers",
                  "model_id": "intfloat/multilingual-e5-base",
                  "device": "cuda"},
        "retrieve": {"rerank": {"backend": "cross_encoder"}},
        "store": {"path": "/tmp/summre_real_temporal.db"},
    })
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_meeting = {}
    with __import__("tempfile").TemporaryDirectory() as td:
        for mid in mids:
            eng = Engine(cfg, path=Path(td) / f"{mid}.db", linker_enabled=True)
            eng.ingest(build_meeting_source(mid, meetings[mid],
                                            on_date=date(2021, 10, 5)))
            uid = {int(u.turn_index): u.unit_id
                   for u in eng.store.units_of_source(f"summre:{mid}")
                   if u.turn_index is not None}
            local = []
            for topic, old, new in sorted(pairs[mid]):
                if old not in uid or new not in uid:
                    continue
                if args.query == "old":
                    qtext = meetings[mid][old]["text"]
                else:
                    qtext = f"Où en est {topic} pour le projet {mid} ?"
                hits = [h.unit.unit_id for h in eng.retrieve(
                    Query(text=qtext, project=mid, mode="current"))][: args.top_k]
                ro = hits.index(uid[old]) if uid[old] in hits else None
                rn = hits.index(uid[new]) if uid[new] in hits else None
                rec = {"meeting": mid, "topic": topic, "old": old, "new": new,
                       "query": qtext, "rank_old": ro, "rank_new": rn}
                local.append(rec); rows.append(rec)
            per_meeting[mid] = summre_pair_scores(local)
            eng.close()
    agg = summre_pair_scores(rows)
    new_at_1 = sum(1 for r in rows if r["rank_new"] == 0)
    old_at_1 = sum(1 for r in rows if r["rank_old"] == 0)
    out = {"corpus": "SUMM-RE train+sel hand-labelled supersedes pairs",
           "query": args.query,
           "n_pairs": len(rows), "top_k": args.top_k,
           "new_at_1": new_at_1, "new_at_1_rate": round(new_at_1 / len(rows), 4),
           "old_at_1": old_at_1, "old_at_1_rate": round(old_at_1 / len(rows), 4),
           "new_above_old": agg["new_above_old"],
           "stale_above_new": agg["stale_above_new"],
           "mrr_new_at_k": agg["mrr_new_at_k"],
           "both_in_pool": agg["both_in_pool"],
           "per_meeting": per_meeting, "pairs": rows}
    p = Path(args.out); p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("pairs", "per_meeting")}, indent=1))
    print("per_meeting", json.dumps(per_meeting, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
