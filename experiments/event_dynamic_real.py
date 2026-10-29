"""Real LOMO read of the dynamic event writer on SUMM-RE train+sel pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from engine.config import repo_root
from engine.memory.dynamic import DynamicWriter, load_embeddings
from engine.memory.event_store import EventStore

SPLITS = ("train", "sel")
GOLD = {"train": "summre_temporal_train.json", "sel": "summre_temporal_sel.json"}
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
THRESHOLDS = (0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.50, 0.60, 0.72)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() /
                    "runs/event_dynamic/result.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    raw = json.loads((root / "data/external/summre_test/meetings.json").read_text())
    turns = {m: (v["segments"] if isinstance(v, dict) else v)
             for m, v in raw.items()}
    emb = load_embeddings(root / "runs/state_reader/turn_embeddings.npz")
    labels = json.loads((root / "data/gold/summre_change_labels.json")
                        .read_text())["meetings"]
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
    mids = sorted(pairs)

    def mention_store(writer: DynamicWriter, mid: str, threshold: float):
        store = EventStore()
        for m in writer.mentions(mid, turns[mid], emb[mid], threshold):
            store.add(m)
        return store

    def score(writer, eval_mids, threshold, mode):
        n = new = old = 0
        for mid in eval_mids:
            store = mention_store(writer, mid, threshold)
            for _topic, o, nw in pairs.get(mid, []):
                q = writer.project(emb[mid][o][None])[0]
                res = store.query("", embedder=lambda _t: q, project=mid,
                                  top_k=16, mode=mode)
                cur = res[0].current if res else None
                n += 1
                if cur is not None and cur.turn == nw:
                    new += 1
                if cur is not None and cur.turn == o:
                    old += 1
        return {"n": n, "new": new, "old": old,
                "new_rate": round(new / n, 4) if n else None,
                "old_rate": round(old / n, 4) if n else None}

    folds = []
    for held in mids:
        tr = [m for m in mids if m != held]
        writer = DynamicWriter().fit(
            tr, emb, pairs, {m: labels[m]["labels"] for m in mids},
            {m: [t["text"] for t in turns[m]] for m in mids})
        tune = {}
        for t in THRESHOLDS:
            tune[str(t)] = {mode: score(writer, tr, t, mode)
                            for mode in ("latest", "latest_confirmed")}
        best = max(THRESHOLDS, key=lambda t: tune[str(t)]["latest_confirmed"]["new_rate"] or 0)
        held_store = mention_store(writer, held, best)
        # score locally, avoiding a second writer fit
        local = {"latest": {"n": 0, "new": 0, "old": 0},
                 "latest_confirmed": {"n": 0, "new": 0, "old": 0}}
        for _topic, o, nw in pairs.get(held, []):
            q = writer.project(emb[held][o][None])[0]
            for mode in local:
                res = held_store.query("", embedder=lambda _t: q,
                                       project=held, top_k=16, mode=mode)
                cur = res[0].current if res else None
                local[mode]["n"] += 1
                local[mode]["new"] += int(cur is not None and cur.turn == nw)
                local[mode]["old"] += int(cur is not None and cur.turn == o)
        for mode, d in local.items():
            n = d["n"]
            d["new_rate"] = round(d["new"] / n, 4) if n else None
            d["old_rate"] = round(d["old"] / n, 4) if n else None
        folds.append({"held": held, "threshold": best, "train_tune": tune,
                      "held": held, "latest": local["latest"],
                      "latest_confirmed": local["latest_confirmed"]})
        print(held, "threshold", best, "latest", local["latest"],
              "confirmed", local["latest_confirmed"], flush=True)
    agg = {}
    for mode in ("latest", "latest_confirmed"):
        n = sum(f[mode]["n"] for f in folds)
        new = sum(f[mode]["new"] for f in folds)
        old = sum(f[mode]["old"] for f in folds)
        agg[mode] = {"n": n, "new": new, "old": old,
                     "new_rate": round(new / n, 4), "old_rate": round(old / n, 4)}
    out = {"corpus": "SUMM-RE train+sel dynamic event writer LOMO",
           "thresholds": THRESHOLDS, "folds": folds, "aggregate": agg}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(agg, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
