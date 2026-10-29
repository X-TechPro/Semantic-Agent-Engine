"""Real-data v4 read: the §6 pipeline measured layer by layer.

Everything below runs the shipped implementation: `V4Ingestor` → `V4Store` →
`V4Search` → `V4Linker` → `answer`. Nothing constructs ad-hoc candidate lists
beside the pipeline.

Arms, all on the 40 hand-labelled `supersedes` pairs:

* **rules** — one store, no relation head; measures layers 1-3 (+ state from
  rule-read tags);
* **fold** — for each meeting, a store ingested with the relation head trained
  on the other five meetings: typed links + append-only state + rerank;
* **head** — the fold head scoring the fold arm's candidate list on its own,
  which isolates the reranker from the pipeline.

Every pair is counted. A pair with no candidate is a miss at every cutoff.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings, load_pairs, pair_rows
from engine.ingest_v4 import V4Ingestor
from engine.link.v4_linker import V4Linker
from engine.query.v4_search import V4Search
from engine.schemas_v4 import MessageRecord
from engine.store.v4 import V4Store


def build_store(meetings: dict[str, list[dict]], linker=None) -> V4Store:
    store = V4Store()
    ing = V4Ingestor(store, linker=linker)
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


def turn_encoder(meetings: dict[str, list[dict]], root: Path):
    """Turn text -> recorded Qwen3 turn embedding; zeros when absent."""
    path = root / "runs/state_reader/turn_embeddings.npz"
    if not path.exists():
        return None
    V = {k: np.asarray(v, dtype=np.float32) for k, v in
         np.load(path, allow_pickle=False).items()}
    table: dict[str, np.ndarray] = {}
    for mid in meetings:
        if mid not in V:
            continue
        for i, turn in enumerate(meetings[mid]):
            if i < len(V[mid]):
                table[str(turn.get("text") or "")] = V[mid][i]
    dim = next(iter(V.values())).shape[1] if V else 0

    def encode(texts: list[str]) -> np.ndarray:
        return np.stack([table.get(t, np.zeros(dim, dtype=np.float32))
                         for t in texts])

    return encode


def gold_unit(store: V4Store, mid: str, turn: int) -> str | None:
    units = store.units_for_message(f"{mid}:t{turn}")
    return units[0]["unit_id"] if units else None


def rank_of(order: list[str], target: str | None) -> int | None:
    if target is None:
        return None
    return order.index(target) + 1 if target in order else None


def _arm(ranks: list[int | None], n: int) -> dict:
    return {"at_1": sum(r == 1 for r in ranks if r),
            "at_5": sum(r is not None and r <= 5 for r in ranks if r),
            "covered": sum(r is not None for r in ranks),
            "mrr": round(sum(1.0 / r for r in ranks if r) / n, 4) if n else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/v4/real.json"))
    ap.add_argument("--models", default=str(repo_root() / "models/flaubert_v4"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    encoder = turn_encoder(meetings, root)
    store = build_store(meetings)
    search = V4Search(store, encoder=encoder)

    folds = sorted(pairs)
    fold_search: dict[str, V4Search] = {}
    fold_linkers: dict[str, V4Linker] = {}
    for mid in folds:
        ckpt = Path(args.models) / f"fold_{mid}"
        if not ckpt.exists():
            continue
        linker = V4Linker(checkpoint=ckpt, device=args.device)
        if not linker.available:
            continue
        fold_store = build_store({mid: meetings[mid]}, linker=linker)
        fold_search[mid] = V4Search(fold_store, linker=linker, encoder=encoder)
        fold_linkers[mid] = linker

    stages = {"entity": 0, "event": 0, "bm25": 0, "dense": 0,
              "rrf": 0, "graph": 0}
    rules_ranks: list[int | None] = []
    fused_ranks: list[int | None] = []
    head_ranks: list[int | None] = []
    state_ranks: list[int | None] = []
    guarded_ranks: list[int | None] = []
    rows = []

    for mid, topic, old, new in pair_rows(pairs):
        new_text = str(meetings[mid][new].get("text") or "")
        new_uid = gold_unit(store, mid, new)
        gold_uid = gold_unit(store, mid, old)
        diag = search.diagnose(new_text, project=mid)
        for name in ("entity", "event", "bm25", "dense"):
            stages[name] += int(gold_uid in diag["pools"].get(name, [])[:20])
        stages["rrf"] += int(gold_uid in diag["rrf"][:20])
        stages["graph"] += int(gold_uid in diag["graph"][:20])

        rules_hits = search.retrieve(new_text, k=40, project=mid, mode="link")
        rules_order = [h.unit_id for h in rules_hits if h.unit_id != new_uid]
        rules_ranks.append(rank_of(rules_order[:20], gold_uid))

        fused_rank = head_rank = state_rank = None
        if mid in fold_search:
            fs = fold_search[mid]
            f_hits = fs.retrieve(new_text, k=40, project=mid, mode="link")
            f_order = [h.unit_id for h in f_hits if h.unit_id != new_uid]
            fused_rank = rank_of(f_order[:20], gold_uid)

            linker = fold_linkers[mid]
            cand_texts = [fs.store.unit(u)["text"] for u in f_order]
            if cand_texts:
                probs = linker.model.relation_probs(
                    [new_text] * len(cand_texts), cand_texts)
                scores = [p.get("supersedes", 0.0) for p in probs]
                head_order = [f_order[i] for i in np.argsort(-np.asarray(scores))]
                head_rank = rank_of(head_order, gold_uid)
                # guarded read: the linker only lets the head decide among
                # candidates that pass the same-slot guard, so measure that too
                slot = diag["frame"].slot
                guarded = [u for u in f_order
                           if fs.store.unit(u).get("slot") == slot] if slot else []
                if gold_uid in guarded:
                    idx = [f_order.index(u) for u in guarded]
                    g_scores = [scores[i] for i in idx]
                    g_order = [guarded[i] for i in
                               np.argsort(-np.asarray(g_scores))]
                    guarded_rank = rank_of(g_order, gold_uid)
                    guarded_ranks.append(guarded_rank)

            s_hits = fs.retrieve(new_text, k=5, project=mid, mode="current")
            state_rank = rank_of([h.unit_id for h in s_hits
                                  if h.unit_id != new_uid], gold_uid)
        fused_ranks.append(fused_rank)
        head_ranks.append(head_rank)
        state_ranks.append(state_rank)
        rows.append({"meeting": mid, "topic": topic, "old": old, "new": new,
                     "gold_unit": gold_uid, "rules_rank": rules_ranks[-1],
                     "fused_rank": fused_rank, "head_rank": head_rank,
                     "state_rank": state_rank,
                     "query_slot": diag["frame"].slot})

    n = len(rows)
    out = {
        "corpus": "SUMM-RE six real meetings, 40 hand-labelled supersedes pairs",
        "direction": "new query -> old antecedent",
        "n": n,
        "candidate_recall_at_20": stages,
        "rules": _arm(rules_ranks, n),
        "fold": _arm(fused_ranks, n),
        "head_only": _arm(head_ranks, n),
        "head_guarded": _arm(guarded_ranks, n),
        "state_mode": _arm(state_ranks, n),
        "fold_meetings": sorted(fold_search),
        "rows": rows,
    }
    if not fold_search:
        out["note"] = ("no per-fold relation checkpoints: train with "
                       "`python -m engine.link.train_relation`; the fold arm "
                       "is empty rather than silently reusing the base model")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
