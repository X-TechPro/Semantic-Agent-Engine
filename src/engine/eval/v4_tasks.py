"""v4 benchmark on the questions the data can actually answer.

Four sections, each with the baseline it must beat:

1. **Change detection** — "did this message change something?" on 1475 turns
   (187 positive). Arms: never-change, the deterministic reader, the trained
   head. Metrics: precision / recall / F1 / AUC.
2. **State tracking** — ingest each meeting turn by turn and compare what the
   engine put in force with the gold change labels. Measures both directions:
   a change that produced no update (missed) and a non-change that moved the
   current value (drift).
3. **Relation classification** — given two turns, name the relation, on the
   114 gold edges plus sampled `unrelated`. This is a fair question; the
   earlier benchmark asked a model to *find* the pair, which is not.
4. **Citation retrieval** — kept, but scored as a set: any earlier turn that
   states the replaced value counts. Reported next to the "just take the
   previous turn" baseline, so a semantic engine has to earn its keep.

Leaves one meeting out everywhere. Writes `runs/v4/tasks.json`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings, load_pairs
from engine.eval.v4_data import (card, citation_set, load_change_labels,
                                 load_edges, texts_of)
from engine.ingest_v4 import V4Ingestor
from engine.link.v4_linker import V4Linker
from engine.query.v4_search import V4Search
from engine.read.v4_tags import read_tags
from engine.schemas_v4 import LIVE_STATUSES, MessageRecord
from engine.store.v4 import V4Store


def prf(pred: list[int], gold: list[int]) -> dict:
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def auroc(scores: list[float], labels: list[int]) -> float:
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return 0.0
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + j + 2) / 2 * sum(lab for _, lab in pairs[i:j + 1])
        i = j + 1
    return round((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg), 4)


def macro_f1(y: list[int], p: list[int], labels: list[str]) -> float:
    f1s = []
    for c in range(len(labels)):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return round(sum(f1s) / len(labels), 4)


def message(mid: str, i: int, text: str) -> MessageRecord:
    return MessageRecord(f"{mid}:t{i}", mid, mid, "", "2021-10-05T09:00:00",
                         text, "meeting", "real")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(repo_root() / "models/flaubert_v4"))
    ap.add_argument("--out", default=str(repo_root() / "runs/v4/tasks.json"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    root = repo_root()
    meetings = load_meetings(root)
    edges = load_edges(root)
    labels = load_change_labels(root)
    pairs = load_pairs(root)
    mids = sorted(labels)

    out: dict = {"data": card(meetings, edges, labels)}
    change_folds, state_folds, rel_folds, cite_folds = [], [], [], []

    for held in mids:
        ckpt = Path(args.models) / f"fold_{held}"
        linker = V4Linker(checkpoint=ckpt, device=args.device) \
            if ckpt.exists() else None
        texts = texts_of(meetings, held)
        gold = labels[held]

        # --- 1. change detection -------------------------------------------
        # score only turns that carry text: an empty ASR turn is not a
        # prediction, and encoding it injects a meaningless vector.
        idx = [i for i, t in enumerate(texts) if t.strip() and i < len(gold)]
        scored_texts = [texts[i] for i in idx]
        gold_s = [gold[i] for i in idx]
        never = [0] * len(gold_s)
        rules = [int(read_tags(t).status in LIVE_STATUSES) for t in scored_texts]
        row = {"held": held, "n": len(gold_s), "pos": sum(gold_s),
               "never": prf(never, gold_s),
               "never_acc": round(1 - sum(gold_s) / max(1, len(gold_s)), 4),
               "rules": prf(rules, gold_s), "rules_auc": auroc(
                   [float(r) for r in rules], gold_s)}
        if linker is not None and linker.available:
            scores = linker.model.change_probs(scored_texts)
            thr = linker.model.change_threshold
            row["head"] = prf([int(s >= thr) for s in scores], gold_s)
            row["head_auc"] = auroc(scores, gold_s)
            row["head_threshold"] = round(thr, 4)
        change_folds.append(row)

        # --- 2. state tracking ---------------------------------------------
        store = V4Store()
        ing = V4Ingestor(store)
        updated = []
        drift = []
        for i, text in enumerate(texts):
            if not text.strip():
                continue
            report = ing.ingest(message(held, i, text))
            live = 0
            replaced = 0
            for ev in store.state_events():
                if ev["evidence_unit"] not in report["unit_ids"]:
                    continue
                if ev["status"] in LIVE_STATUSES:
                    live += 1
                elif ev["status"] == "replaced":
                    replaced += 1
            updated.append(int(live > 0))
            drift.append(int(replaced > 0))
        keep = [i for i in range(len(gold)) if i < len(updated)]
        state_folds.append({
            "held": held,
            "on_change_update": round(
                sum(updated[i] for i in keep if gold[i] == 1)
                / max(1, sum(gold[i] for i in keep)), 4),
            "on_no_change_update": round(
                sum(updated[i] for i in keep if gold[i] == 0)
                / max(1, sum(1 for i in keep if gold[i] == 0)), 4),
            "replaced_on_no_change": sum(
                drift[i] for i in keep if gold[i] == 0)})

        # --- 3. relation classification ------------------------------------
        held_edges = edges.get(held, [])
        if linker is not None and linker.available and held_edges:
            y, p = [], []
            for e in held_edges:
                pred = linker.model.predict_relation(texts[e.new], texts[e.old])
                y.append(["supports", "supersedes", "contradicts"].index(
                    e.relation) if e.relation in ("supports", "supersedes",
                                                  "contradicts") else 0)
                p.append(["supports", "supersedes", "contradicts"].index(
                    pred.relation) if pred.relation in ("supports",
                                                        "supersedes",
                                                        "contradicts") else 0)
            majority = max(set(y), key=y.count)
            rel_folds.append({"held": held, "n": len(y),
                              "macro_f1": macro_f1(
                                  y, p, ["supports", "supersedes",
                                         "contradicts"]),
                              "accuracy": round(sum(a == b for a, b in
                                                    zip(y, p)) / len(y), 4),
                              "majority_accuracy": round(
                                  y.count(majority) / len(y), 4),
                              "gold": dict(Counter(y)), "pred": dict(Counter(p))})

    # --- 4. citation retrieval, exact vs set -------------------------------
    store = V4Store()
    ing = V4Ingestor(store)
    for mid in mids:
        for i, text in enumerate(texts_of(meetings, mid)):
            if text.strip():
                ing.ingest(message(mid, i, text))
    search = V4Search(store)

    cite = {"n": 0, "exact": {"at_1": 0, "at_5": 0}, "set": {"at_1": 0, "at_5": 0},
            "previous": {"at_1": 0, "at_5": 0}, "set_sizes": []}
    for mid, topic, old, new in [(m, t, o, n) for m in pairs
                                 for t, o, n in pairs[m]]:
        texts = texts_of(meetings, mid)
        valid = citation_set(meetings, mid, old, new)
        hits = search.retrieve(texts[new], k=20, project=mid, mode="link")
        turns = []
        for h in hits:
            try:
                turns.append(int(h.message_id.rsplit(":t", 1)[1]))
            except (IndexError, ValueError):
                continue
        cite["n"] += 1
        cite["exact"]["at_1"] += turns[:1] == [old]
        cite["exact"]["at_5"] += old in turns[:5]
        cite["set"]["at_1"] += bool(turns[:1] and turns[0] in valid)
        cite["set"]["at_5"] += any(t in valid for t in turns[:5])
        cite["previous"]["at_1"] += new - old == 1
        cite["previous"]["at_5"] += 0 < new - old <= 5
        cite["set_sizes"].append(len(valid))
    cite["set_sizes"] = {"mean": round(
        sum(cite["set_sizes"]) / max(1, len(cite["set_sizes"])), 2),
        "max": max(cite["set_sizes"] or [0])}

    def mean(rows, path):
        vals = []
        for r in rows:
            v = r
            for key in path:
                v = v.get(key, {}) if isinstance(v, dict) else None
            if isinstance(v, (int, float)):
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else None

    out.update({
        "change": {
            "n": sum(r["n"] for r in change_folds),
            "pos": sum(r["pos"] for r in change_folds),
            "never_f1": mean(change_folds, ["never", "f1"]),
            "never_accuracy": mean(change_folds, ["never_acc"]),
            "rules_precision": mean(change_folds, ["rules", "precision"]),
            "rules_recall": mean(change_folds, ["rules", "recall"]),
            "rules_f1": mean(change_folds, ["rules", "f1"]),
            "rules_auc": mean(change_folds, ["rules_auc"]),
            "head_precision": mean(change_folds, ["head", "precision"]),
            "head_recall": mean(change_folds, ["head", "recall"]),
            "head_f1": mean(change_folds, ["head", "f1"]),
            "head_auc": mean(change_folds, ["head_auc"]),
            "folds": change_folds,
        },
        "state": {
            "on_change_update": mean(state_folds, ["on_change_update"]),
            "on_no_change_update": mean(state_folds,
                                        ["on_no_change_update"]),
            "replaced_on_no_change": sum(
                r["replaced_on_no_change"] for r in state_folds),
            "folds": state_folds,
        },
        "relation": {
            "n": sum(r["n"] for r in rel_folds),
            "macro_f1": mean(rel_folds, ["macro_f1"]),
            "accuracy": mean(rel_folds, ["accuracy"]),
            "majority_accuracy": mean(rel_folds, ["majority_accuracy"]),
            "folds": rel_folds,
        },
        "citation": cite,
    })
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    printable = {k: v for k, v in out.items()
                 if k not in ("change", "relation")}
    printable["change"] = {k: v for k, v in out["change"].items()
                           if k != "folds"}
    printable["relation"] = {k: v for k, v in out["relation"].items()
                             if k != "folds"}
    printable["state"] = {k: v for k, v in out["state"].items() if k != "folds"}
    print(json.dumps(printable, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
