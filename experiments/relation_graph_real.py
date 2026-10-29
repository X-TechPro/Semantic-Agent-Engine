"""Train the continuity scorer on the synthetic relation graph, test on real.

This is the user's two-stage mechanism: embeddings retrieve candidates, a learned
pair scorer decides same-thread/typed relation, and the graph keeps edges. The
real 40 pairs are never used for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer

from engine.config import repo_root
from engine.link.relation_scorer import PairMeta, RelationScorer, features
from engine.memory.thread_graph import build_thread_graph

SPLITS = ("train", "sel")
GOLD = {"train": "summre_temporal_train.json", "sel": "summre_temporal_sel.json"}
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
BUNDLE = Path("N2-model/piscine_summre_synthetic")


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def synth_rows(root: Path, split: str, vec: dict[str, np.ndarray]) -> list[dict]:
    units = {u["unit_id"]: u for u in read_jsonl(root / BUNDLE / "units" / f"{split}.jsonl")}
    rows = []
    for r in read_jsonl(root / BUNDLE / "relations" / f"{split}.jsonl"):
        a, b = units.get(r["from_unit_id"]), units.get(r["to_unit_id"])
        if not a or not b:
            continue
        x = features(a["statement"], b["statement"], vec[a["unit_id"]],
                     vec[b["unit_id"]],
                     PairMeta(a.get("project_id", ""), a.get("speaker", ""), 0, ""),
                     PairMeta(b.get("project_id", ""), b.get("speaker", ""), 0, ""))
        rows.append({"x": x, "relation": r["relation"], "a": a, "b": b})
    return rows


def add_hard_negatives(rows: list[dict], units: list[dict], vec, n: int = 400,
                       seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_project = {}
    for u in units:
        by_project.setdefault(u.get("project_id", ""), []).append(u)
    out = list(rows)
    for _ in range(n):
        us = [u for u in units if u.get("project_id")]
        a = us[int(rng.integers(0, len(us)))]
        same = [u for u in by_project[a["project_id"]] if u["unit_id"] != a["unit_id"]]
        # 70% same-project hard negative, 30% cross-project.
        pool = same if (same and rng.random() < 0.7) else us
        b = pool[int(rng.integers(0, len(pool)))]
        if b.get("unit_id") == a.get("unit_id"):
            continue
        x = features(a["statement"], b["statement"], vec[a["unit_id"]],
                     vec[b["unit_id"]],
                     PairMeta(a.get("project_id", ""), a.get("speaker", ""), 0, ""),
                     PairMeta(b.get("project_id", ""), b.get("speaker", ""), 0, ""))
        out.append({"x": x, "relation": "unrelated", "a": a, "b": b})
    return out


def load_real(root: Path):
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
    return turns, pairs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default=str(repo_root() / "runs/relation_graph/result.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    enc = SentenceTransformer(args.model, device="cuda")

    def embed(texts, batch=64):
        return enc.encode(texts, batch_size=batch, normalize_embeddings=True,
                          convert_to_numpy=True, show_progress_bar=False)

    units_tr = read_jsonl(root / BUNDLE / "units" / "train.jsonl")
    units_va = read_jsonl(root / BUNDLE / "units" / "validation.jsonl")
    texts = [u["statement"] for u in units_tr + units_va]
    V = embed(texts)
    vec = {u["unit_id"]: V[i] for i, u in enumerate(units_tr + units_va)}
    tr = add_hard_negatives(synth_rows(root, "train", vec), units_tr, vec, n=500)
    va = synth_rows(root, "validation", vec)
    scorer = RelationScorer().fit(tr)
    pred = [scorer.score_one(r["x"]).relation for r in va]
    gold = [r["relation"] for r in va]
    synth = {
        "train_rows": len(tr), "val_rows": len(va),
        "relation_macro_f1": round(float(f1_score(gold, pred, average="macro",
                                                  zero_division=0)), 4),
        "relation_accuracy": round(float(np.mean([a == b for a, b in zip(gold, pred)])), 4),
    }
    print("synthetic", synth, flush=True)

    turns, pairs = load_real(root)
    mids = sorted(pairs)
    real_vec = {}
    for mid in mids:
        real_vec[mid] = embed([t["text"] for t in turns[mid]])
    rows, graph_rows = [], []
    for mid in mids:
        g = build_thread_graph(mid, turns[mid], real_vec[mid], scorer,
                               top_k=16, threshold=0.5)
        for topic, old, new in pairs[mid]:
            comp = g.component(f"{mid}:{old}")
            comp_turns = {n.turn for n in comp}
            cur = g.current(f"{mid}:{old}")
            # candidate ranking: exclude the old sentence itself
            xs, cand = [], []
            for j in range(len(turns[mid])):
                if j == old:
                    continue
                xs.append(features(
                    turns[mid][old]["text"], turns[mid][j]["text"],
                    real_vec[mid][old], real_vec[mid][j],
                    PairMeta(mid, str(turns[mid][old].get("speaker") or ""), old, ""),
                    PairMeta(mid, str(turns[mid][j].get("speaker") or ""), j, "")))
                cand.append(j)
            scores = scorer.score_many(xs)
            order = sorted(range(len(cand)),
                           key=lambda k: (-scores[k].p_same, -scores[k].probs.get("supersedes", 0)))
            rank_new = next((r + 1 for r, k in enumerate(order) if cand[k] == new), None)
            old_edge = [e for e in g.edges_of(f"{mid}:{old}") if e.target == f"{mid}:{new}"]
            rows.append({
                "meeting": mid, "topic": topic, "old": old, "new": new,
                "pair_connected": new in comp_turns,
                "current_turn": cur.turn if cur else None,
                "current_is_new": bool(cur and cur.turn == new),
                "new_rank": rank_new, "new_at_1": rank_new == 1,
                "new_at_5": bool(rank_new and rank_new <= 5),
                "direct_edge": old_edge[0].relation if old_edge else None,
                "direct_p": max((e.confidence for e in old_edge), default=0.0),
            })
    n = len(rows)
    real = {
        "n": n,
        "pair_connected": sum(r["pair_connected"] for r in rows),
        "current_is_new": sum(r["current_is_new"] for r in rows),
        "new_at_1": sum(r["new_at_1"] for r in rows),
        "new_at_5": sum(r["new_at_5"] for r in rows),
        "rows": rows,
    }
    print("real", {k: real[k] for k in ("n", "pair_connected", "current_is_new",
                                        "new_at_1", "new_at_5")}, flush=True)
    out = {"model": args.model, "synthetic": synth, "real": real}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
