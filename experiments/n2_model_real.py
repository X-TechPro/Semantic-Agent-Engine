"""Real SUMM-RE check of the N2 bundle's contextual encoder.

Query is the hand thread topic; history is every turn through the gold new turn.
The model's contextual vector ranks that history. We measure whether the new
turn beats the old one. This is the only thing the N2 bundle can answer: a
history-conditioned vector, not event identity/status.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from engine.config import repo_root

BUNDLE = (repo_root() / "N2-model" / "piscine_summre_model_v2_bundle")
SPLITS = ("train", "sel")
GOLD = {"train": "summre_temporal_train.json", "sel": "summre_temporal_sel.json"}
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/n2_model_real/result.json"))
    ap.add_argument("--alpha", type=float, default=2.0)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(BUNDLE))
    from contextual_memory import embed_with_history, load_trained_model  # noqa

    root = repo_root()
    raw = json.loads((root / "data/external/summre_test/meetings.json").read_text())
    turns = {m: (v["segments"] if isinstance(v, dict) else v)
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
    model = load_trained_model(BUNDLE / "contextual_model_piscine.pt",
                               device="cpu")
    rows = []
    for mid, ps in sorted(pairs.items()):
        texts = [t["text"] for t in turns[mid]]
        for topic, old, new in sorted(ps):
            hist = texts[:new + 1]
            q = f"Où en est {topic} pour le projet {mid} ?"
            _vec, ranked = embed_with_history(model, q, hist, device="cpu")
            rank = {r["history_index"]: r["history_index"] for r in ranked}
            order = [r["history_index"] for r in ranked]
            r_old = order.index(old) + 1 if old in order else None
            r_new = order.index(new) + 1 if new in order else None
            rows.append({"meeting": mid, "topic": topic, "old": old,
                         "new": new, "old_rank": r_old, "new_rank": r_new,
                         "old_at_1": r_old == 1, "new_at_1": r_new == 1,
                         "old_at_5": bool(r_old and r_old <= 5),
                         "new_at_5": bool(r_new and r_new <= 5)})
    n = len(rows)
    out = {"corpus": "SUMM-RE train+sel, N2 bundle contextual encoder",
           "query": "hand thread topic", "alpha": args.alpha, "n": n,
           "old_at_1": sum(r["old_at_1"] for r in rows),
           "new_at_1": sum(r["new_at_1"] for r in rows),
           "old_at_5": sum(r["old_at_5"] for r in rows),
           "new_at_5": sum(r["new_at_5"] for r in rows),
           "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
