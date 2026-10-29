"""Does the fixed v4 survive real meetings? Four arms, one pipeline.

    baseline   rules tags, per-turn embeddings
    laya       fine-tuned Laya heads for kind/status/intent
    late       late-chunked jina-v5 vectors, contextual query vector
    laya+late  both

Each meeting gets its own store and its own leave-one-meeting-out relation
checkpoint (`models/flaubert_v4/fold_<mid>`), so typed links and append-only
state are live, as in the fold arm of `engine.eval.v4_real`.

Measured: retrieval on the 40 annotated pairs and the 114 edges, reading
("this unit carries a state" against the 187 changes), and end-to-end answers.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings, load_pairs, pair_rows
from engine.eval.v4_data import citation_set, load_edges, texts_of
from engine.ingest_v4 import V4Ingestor
from engine.link.v4_linker import V4Linker
from engine.query.v4_search import V4Search
from engine.schemas_v4 import MessageRecord
from engine.store.v4 import V4Store

STATE = {"proposed", "confirmed", "cancelled", "completed"}


def build_store(meetings, tag_fn=None, linker=None,
                long_split: bool = False) -> V4Store:
    store = V4Store()
    ing = V4Ingestor(store, linker=linker, tag_fn=tag_fn,
                     long_split=long_split)
    for mid in sorted(meetings):
        for i, turn in enumerate(meetings[mid]):
            text = str(turn.get("text") or "")
            if not text.strip():
                continue
            ing.ingest(MessageRecord(
                message_id=f"{mid}:t{i}", source_id=mid, project_id=mid,
                speaker=str(turn.get("speaker") or ""),
                timestamp=str(turn.get("timestamp") or "2021-10-05"),
                text=text, source_type="meeting", origin="real"))
    return store


def laya_tag_fn(meetings, model_dir: str, device: str):
    from engine.read import laya_reader
    from engine.read.v4_tags import read_tags

    texts = [str(t.get("text") or "") for mid in meetings
             for t in meetings[mid] if str(t.get("text") or "").strip()]
    tags = laya_reader.tag_texts(texts, model_dir, device=device)
    table = dict(zip(texts, tags))

    def fn(text, speaker="", source_date=None, source_type="meeting"):
        return table.get(text) or read_tags(text, speaker=speaker,
                                            source_date=source_date,
                                            source_type=source_type)

    laya_reader._MODEL.clear()
    gc.collect()
    return fn


class LateArm:
    """Late-chunked vectors + a query encoder in the same space."""

    def __init__(self, meetings, stores: dict, device: str):
        from engine.eval.late_chunk import LateV5

        self.late = LateV5(device)
        self.vectors: dict[str, np.ndarray] = {}
        for mid in sorted(meetings):
            V = self.late.meeting(texts_of(meetings, mid))
            for i in range(len(V)):
                for u in stores[mid].units_for_message(f"{mid}:t{i}"):
                    self.vectors[u["unit_id"]] = V[i]

    def qvec_for(self, mid: str, turn: int):
        for i in range(8):
            v = self.vectors.get(f"{mid}:t{turn}:u{i}")
            if v is not None:
                return v
        return None

    def query_encoder(self, texts: list[str]) -> np.ndarray:
        import torch
        parts = []
        for i in range(0, len(texts), 32):
            out = self.late.m.encode(texts=list(texts[i:i + 32]),
                                     task="text-matching")
            parts.append(np.asarray(torch.as_tensor(out).float().cpu(),
                                    dtype=np.float32))
        v = np.concatenate(parts, axis=0)
        return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-9, None)


def reading(stores: dict, root: Path) -> dict:
    gold = json.loads((root / "data/gold/summre_change_labels.json").read_text())
    tp = fp = fn = 0
    for mid, doc in gold["meetings"].items():
        store = stores.get(mid)
        if store is None:
            continue
        for i, y in enumerate(doc["labels"]):
            units = store.units_for_message(f"{mid}:t{i}")
            if not units:
                continue
            pred = int(any(u.get("status") in STATE for u in units))
            tp += int(pred and y)
            fp += int(pred and not y)
            fn += int(not pred and y)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def retrieval(meetings, stores, searches, prows, edges, qvec_for=None) -> dict:
    at = {1: 0, 5: 0, 20: 0}
    for mid, _topic, old, new in prows:
        store = stores[mid]
        gold = store.units_for_message(f"{mid}:t{old}")
        gold_uid = gold[0]["unit_id"] if gold else None
        own = {u["unit_id"] for u in store.units_for_message(f"{mid}:t{new}")}
        hits = searches[mid].retrieve(
            str(meetings[mid][new].get("text") or ""), k=40, project=mid,
            mode="link", qvec=qvec_for(mid, new) if qvec_for else None)
        order = [h.unit_id for h in hits if h.unit_id not in own]
        rank = order.index(gold_uid) + 1 if gold_uid in order else None
        for cut in at:
            at[cut] += int(rank is not None and rank <= cut)

    hit = total = 0
    for mid in sorted(edges):
        store = stores.get(mid)
        if store is None:
            continue
        texts = texts_of(meetings, mid)
        for e in edges[mid]:
            valid = {u["unit_id"]
                     for i in citation_set(meetings, mid, e.old, e.new)
                     for u in store.units_for_message(f"{mid}:t{i}")}
            hits = searches[mid].retrieve(
                str(texts[e.new]), k=20, project=mid, mode="link",
                qvec=qvec_for(mid, e.new) if qvec_for else None)
            total += 1
            hit += int(any(h.unit_id in valid for h in hits))
    return {"pairs_n": len(prows), "at_1": at[1], "at_5": at[5],
            "at_20": at[20], "citation_recall_at_20": f"{hit}/{total}"}


def answers(meetings, stores, searches, prows, qvec_for=None) -> dict:
    from engine.answer.v4 import answer

    ok = gold_cited = 0
    for mid, _topic, old, new in prows:
        store = stores[mid]
        text = str(meetings[mid][new].get("text") or "")
        hits = searches[mid].retrieve(
            text, k=10, project=mid, mode="link",
            qvec=qvec_for(mid, new) if qvec_for else None)
        if not hits:
            continue
        a = answer(text, hits, store=store)
        ok += int(a.verified and not a.unknown)
        gold = store.units_for_message(f"{mid}:t{old}")
        gu = gold[0]["unit_id"] if gold else None
        gold_cited += int(any(c.unit_id == gu for c in a.citations))
    return {"n": len(prows), "verified": ok, "cites_gold": gold_cited}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("baseline", "laya", "late", "laya+late"),
                    default="laya+late")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--models",
                    default=str(repo_root() / "runs/laya_tags/multilingual_v2"))
    ap.add_argument("--fold-models", default=str(repo_root() / "models/flaubert_v4"))
    ap.add_argument("--no-linker", action="store_true")
    ap.add_argument("--long-split", action="store_true")
    args = ap.parse_args(argv)

    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    prows = list(pair_rows(pairs))
    edges = load_edges(root)

    tag_fn = None
    if "laya" in args.arm:
        print("tagging with laya…", flush=True)
        tag_fn = laya_tag_fn(meetings, args.models, args.device)

    stores: dict[str, V4Store] = {}
    n_linkers = 0
    for mid in sorted(meetings):
        linker = None
        ckpt = Path(args.fold_models) / f"fold_{mid}"
        if not args.no_linker and ckpt.exists():
            cand = V4Linker(checkpoint=ckpt, device=args.device)
            if cand.available:
                linker = cand
                n_linkers += 1
        stores[mid] = build_store({mid: meetings[mid]}, tag_fn=tag_fn,
                                  linker=linker,
                                  long_split=args.long_split)
    print(f"stores built: {len(stores)} meetings, {n_linkers} fold linkers",
          flush=True)

    late = None
    searches: dict[str, V4Search] = {}
    if "late" in args.arm:
        print("late-chunking with v5…", flush=True)
        late = LateArm(meetings, stores, args.device)
        for mid, st in stores.items():
            searches[mid] = V4Search(st, encoder=late.query_encoder,
                                     vectors=late.vectors)
    else:
        from engine.eval.v4_real import turn_encoder
        enc = turn_encoder(meetings, root)
        for mid, st in stores.items():
            searches[mid] = V4Search(st, encoder=enc)

    qvec_for = late.qvec_for if late is not None else None
    out = {
        "arm": args.arm,
        "linkers": n_linkers,
        "units": sum(len(s.all_units()) for s in stores.values()),
        "links": sum(len(s.links()) for s in stores.values()),
        "states": sum(len(s.state_events()) for s in stores.values()),
        "reading": reading(stores, root),
        "retrieval": retrieval(meetings, stores, searches, prows, edges, qvec_for),
        "answer": answers(meetings, stores, searches, prows, qvec_for),
    }
    suffix = "_longsplit" if args.long_split else ""
    path = Path(args.out or root /
                f"runs/v4/fixed_{args.arm.replace('+', '_')}{suffix}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "retrieval"}, indent=1))
    print(json.dumps(out["retrieval"], indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
