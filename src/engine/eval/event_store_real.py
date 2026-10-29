"""Real SUMM-RE read for the mention/event JSON store.

For each gold pair the query is the old sentence (stress) or the hand thread
topic (natural). The returned event current is checked against the gold new
turn. `latest` follows your "newest wins" rule; `latest_confirmed` ignores
proposals/questions. Both are reported.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from engine.config import repo_root
from engine.memory.event_builder import mentions_from_meeting
from engine.memory.event_store import EventStore
from engine.threads.state_reader import StateReader

SPLITS = ("train", "sel")
GOLD = {"train": "summre_temporal_train.json", "sel": "summre_temporal_sel.json"}
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", choices=("old", "topic"), default="old")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = repo_root()
    if args.out is None:
        args.out = str(root / "runs/summre_event_store" /
                       f"result_{args.query}.json")
    raw = json.loads((root / "data/external/summre_test/meetings.json").read_text())
    meetings = {m: (v["segments"] if isinstance(v, dict) else v)
                for m, v in raw.items()}
    pairs = {}
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
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("intfloat/multilingual-e5-base", device="cuda")
    stores = {}
    mention_ids = {}
    for mid in sorted(pairs):
        mentions = mentions_from_meeting(mid, meetings[mid],
                                         on_date=date(2021, 10, 5))
        texts = list(dict.fromkeys(m.text for m in mentions))
        vecs = enc.encode(texts, batch_size=48, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=True)
        by_text = {t: vecs[i].tolist() for i, t in enumerate(texts)}
        store = EventStore()
        for m in mentions:
            store.add(m, embedding=by_text[m.text])
        stores[mid] = store
        mention_ids[mid] = {m.turn: m.mention_id for m in mentions}

    def run(mode: str) -> dict:
        rows = []
        for mid, plist in pairs.items():
            store = stores[mid]
            for topic, old, new in sorted(plist):
                qtext = (meetings[mid][old]["text"] if args.query == "old"
                         else f"Où en est {topic} pour le projet {mid} ?")
                qv = enc.encode([qtext], normalize_embeddings=True,
                                convert_to_numpy=True)[0]
                res = store.query(qtext, embedder=lambda _t: qv,
                                  project=mid, top_k=16, mode=mode)
                cur = res[0].current if res else None
                rows.append({"meeting": mid, "topic": topic, "old": old,
                             "new": new, "query": qtext,
                             "current_turn": cur.turn if cur else None,
                             "current_id": cur.mention_id if cur else None,
                             "new_at_1": bool(cur and cur.turn == new),
                             "old_at_1": bool(cur and cur.turn == old)})
        n = len(rows)
        return {"mode": mode, "n": n,
                "new_at_1": sum(r["new_at_1"] for r in rows),
                "new_at_1_rate": round(sum(r["new_at_1"] for r in rows) / n, 4),
                "old_at_1": sum(r["old_at_1"] for r in rows),
                "old_at_1_rate": round(sum(r["old_at_1"] for r in rows) / n, 4),
                "none": sum(r["current_turn"] is None for r in rows),
                "rows": rows}

    out = {"corpus": "SUMM-RE train+sel, hand-labelled pairs",
           "query": args.query, "modes": {m: run(m) for m in
                                           ("latest", "latest_confirmed")}}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m, r in out["modes"].items():
        print(args.query, m, "new@1", r["new_at_1"], "/", r["n"],
              "old@1", r["old_at_1"], "none", r["none"])
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
