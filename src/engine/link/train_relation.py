"""Train the §4 heads on the gold annotations, leave-one-meeting-out.

One encoder, three heads, one pass per fold:

* **relation** (6 classes) on the 114 typed gold edges + sampled `unrelated`;
* **change** (2 classes) on 1475 turns of the change annotation (187 positive);
* **status** (4 classes) on reader-tagged turns, as an auxiliary signal.

Why the change head is the primary one: the project's own change definition
(`docs/CHANGE_DEFINITION.md`) says exact-pair scoring is a *citation metric,
never the primary target*. The primary target is "did this message change
something", and that annotation is 37x larger and unambiguous.

Per fold we report:

* relation macro-F1 and per-class counts on the held meeting's edges;
* change precision / recall / F1 / AUC on the held meeting's turns;
* status accuracy;
* the antecedent-ranking read, kept for continuity with earlier reports.

Checkpoints are written per fold (`fold_<meeting>`) so no evaluation reuses a
model that has seen the meeting it is scored on.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings, load_pairs
from engine.eval.v4_data import (change_rows, load_change_labels, load_edges)
from engine.link.flaubert_relation import (FlaubertRelationModel, MODEL_ID,
                                           RELATION_LABELS, STATUS_LABELS)
from engine.link.relation_data import build_gold_rows, status_rows
from engine.read.v4_tags import read_tags

STATUS_HEAD = ("proposed", "confirmed", "cancelled", "completed")


def _weights(labels: list[int], n: int) -> torch.Tensor:
    counts = Counter(labels)
    total = sum(counts.values()) or 1
    return torch.tensor([total / (n * counts.get(i, 1)) for i in range(n)],
                        dtype=torch.float32)


def _auroc(scores: list[float], labels: list[int]) -> float:
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
        avg_rank = (i + j + 2) / 2
        rank_sum += avg_rank * sum(lab for _, lab in pairs[i:j + 1])
        i = j + 1
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _prf(pred: list[int], gold: list[int]) -> dict:
    tp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 1)
    fp = sum(1 for p, g in zip(pred, gold) if p == 1 and g == 0)
    fn = sum(1 for p, g in zip(pred, gold) if p == 0 and g == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}


def _macro_f1(y: list[int], p: list[int], n: int) -> float:
    f1s = []
    for c in range(n):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(sum(f1s) / n) if f1s else 0.0


def train_fold(model: FlaubertRelationModel, rows: list, change_rows_: list,
               statuses: list, epochs: int, batch: int, lr_head: float,
               lr_encoder: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    rel_index = {r: i for i, r in enumerate(RELATION_LABELS)}
    st_index = {s: i for i, s in enumerate(STATUS_LABELS)}
    w_rel = _weights([rel_index[r.relation] for r in rows],
                     len(RELATION_LABELS)).to(model.device)
    ch_labels = [c.label for c in change_rows_]
    w_ch = _weights(ch_labels, 2).to(model.device)
    st_labels = [st_index[s] for _m, _t, s in statuses if s in st_index]
    w_st = _weights(st_labels, len(STATUS_LABELS)).to(model.device)

    opt = torch.optim.AdamW([
        {"params": model.heads.parameters(), "lr": lr_head},
        {"params": model.encoder.parameters(), "lr": lr_encoder},
    ], weight_decay=0.01)
    losses: list[float] = []
    for _ep in range(epochs):
        order = list(range(len(rows)))
        rng.shuffle(order)
        ch_order = list(range(len(change_rows_)))
        rng.shuffle(ch_order)
        model.encoder.train()
        model.heads.train()

        for s in range(0, len(order), batch):
            b = [rows[i] for i in order[s:s + batch]]
            feat = model.pair_features([r.query for r in b],
                                       [r.candidate for r in b])
            y = torch.tensor([rel_index[r.relation] for r in b],
                             device=model.device)
            loss = F.cross_entropy(model.heads.forward_pair(feat).float(), y,
                                   weight=w_rel)
            keep = [(r.candidate, st_index[r.candidate_status]) for r in b
                    if r.candidate_status in st_index]
            if keep:
                log_s = model.heads.forward_unit(
                    model.pooled([t for t, _ in keep])).float()
                y_s = torch.tensor([l for _, l in keep], device=model.device)
                loss = loss + 0.5 * F.cross_entropy(log_s, y_s, weight=w_st)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt.step(); losses.append(float(loss.detach()))

        for s in range(0, len(ch_order), batch * 2):
            b = [change_rows_[i] for i in ch_order[s:s + batch * 2]]
            logits = model.heads.forward_change(
                model.pooled([c.text for c in b])).float()
            y = torch.tensor([c.label for c in b], device=model.device)
            loss = F.cross_entropy(logits, y, weight=w_ch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
            opt.step(); losses.append(float(loss.detach()))

    model.heads.eval()
    # Prior matching: pick the cutoff that predicts the training positive rate.
    # A class-weighted 2-way head scores far above 0.5 on most turns, so a
    # fixed 0.5 labels everything "change" (measured: recall 1.0, precision
    # 0.13). This is calibrated on training rows only.
    if change_rows_:
        # read the head directly: `change_probs` refuses an untrained model and
        # the flag is only set once training returns.
        with torch.no_grad():
            p = torch.softmax(model.heads.forward_change(
                model.pooled([c.text for c in change_rows_])).float(), dim=-1)
        scores = p[:, 1].cpu().tolist()
        rate = sum(c.label for c in change_rows_) / len(change_rows_)
        model.change_threshold = float(np.quantile(scores, 1.0 - rate))
    return losses


def evaluate(model: FlaubertRelationModel, rows: list, change_held: list,
             statuses: list, meetings: dict, pairs: dict, held: str) -> dict:
    rel_index = {r: i for i, r in enumerate(RELATION_LABELS)}
    y, p = [], []
    for r in rows:
        pred = model.predict_relation(r.query, r.candidate)
        y.append(rel_index[r.relation])
        p.append(rel_index[pred.relation])

    st_ok = st_n = 0
    for _m, text, label in statuses:
        pred, _conf = model.predict_status(text)
        st_ok += int(pred == label)
        st_n += 1

    change_gold = [c.label for c in change_held]
    change_scores = model.change_probs([c.text for c in change_held]) \
        if change_held else []
    change_pred = [int(x >= model.change_threshold) for x in change_scores]

    # antecedent ranking, kept for continuity; not the primary target
    from engine.store.fts import tokenize

    def jac(a: str, b: str) -> float:
        A = set(tokenize(a)); B = set(tokenize(b))
        return len(A & B) / max(1, len(A | B))

    at1 = at5 = 0
    rr = []
    n = 0
    texts = [str(t.get("text") or "") for t in meetings.get(held, [])]
    tags = [read_tags(t) for t in texts]
    for _topic, old, new in pairs.get(held, []):
        if old >= len(texts) or new >= len(texts):
            continue
        slot = tags[new].slot
        confusable = [j for j in range(len(texts))
                      if j not in (old, new) and tags[j].slot == slot]
        confusable.sort(key=lambda j: -jac(texts[new], texts[j]))
        cands = [old] + confusable[:8]
        if len(cands) < 2:
            continue
        probs = model.relation_probs([texts[new]] * len(cands),
                                     [texts[j] for j in cands])
        scores = [row["supersedes"] for row in probs]
        order = [cands[i] for i in np.argsort(-np.asarray(scores))]
        rank = order.index(old) + 1
        n += 1
        at1 += rank == 1
        at5 += rank <= 5
        rr.append(1.0 / rank)

    return {"rel_acc": round(sum(a == b for a, b in zip(y, p)) / max(1, len(y)), 4),
            "rel_f1": round(_macro_f1(y, p, len(RELATION_LABELS)), 4),
            "status_acc": round(st_ok / st_n, 4) if st_n else None,
            "change": _prf(change_pred, change_gold),
            "change_auc": round(_auroc(change_scores, change_gold), 4),
            "change_n": len(change_gold),
            "change_pos": sum(change_gold),
            "gold_n": n, "at_1": at1, "at_5": at5,
            "mrr": round(float(np.mean(rr)), 4) if rr else 0.0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "models/flaubert_v4"))
    ap.add_argument("--runs", default=str(repo_root() / "runs/v4/linker.json"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    edges = load_edges(root)
    labels = load_change_labels(root)
    mids = sorted(labels)

    rel_rows = build_gold_rows(meetings, edges, seed=args.seed)
    ch_rows = change_rows(meetings, labels)
    st_rows = status_rows(meetings, mids=mids)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("relation rows", len(rel_rows),
          Counter(r.relation for r in rel_rows), flush=True)
    print("change rows", len(ch_rows),
          Counter(c.label for c in ch_rows), flush=True)
    print("status rows", len(st_rows), flush=True)

    folds = []
    for held in mids:
        train = [r for r in rel_rows if r.meeting != held]
        held_rel = [r for r in rel_rows if r.meeting == held]
        ch_train = [c for c in ch_rows if c.meeting != held]
        ch_held = [c for c in ch_rows if c.meeting == held]
        st_train = [s for s in st_rows if s[0] != held]
        st_held = [s for s in st_rows if s[0] == held]
        if not held_rel or not ch_held:
            continue
        model = FlaubertRelationModel(device=args.device, seed=args.seed)
        losses = train_fold(model, train, ch_train, st_train, args.epochs,
                            args.batch, args.lr_head, args.lr_encoder, args.seed)
        model.trained = True
        metrics = evaluate(model, held_rel, ch_held, st_held, meetings, pairs, held)
        metrics.update({"held": held, "n_train": len(train),
                        "loss": round(float(np.mean(losses)), 4)})
        model.save(out_dir / f"fold_{held}")
        folds.append(metrics)
        print(metrics, flush=True)
        del model
        torch.cuda.empty_cache()

    final = FlaubertRelationModel(device=args.device, seed=args.seed)
    losses = train_fold(final, rel_rows, ch_rows, st_rows, args.epochs,
                        args.batch, args.lr_head, args.lr_encoder, args.seed)
    final.trained = True
    final.save(out_dir)
    del final
    torch.cuda.empty_cache()

    def agg(key):
        return round(float(np.mean([f[key] for f in folds])), 4)

    result = {
        "model": MODEL_ID,
        "heads": {"relation": list(RELATION_LABELS),
                  "status": list(STATUS_LABELS),
                  "change": ["no_change", "change"]},
        "rows": {"relation": len(rel_rows), "change": len(ch_rows),
                 "status": len(st_rows)},
        "relation_counts": dict(Counter(r.relation for r in rel_rows)),
        "change_counts": dict(Counter(c.label for c in ch_rows)),
        "lr": {"head": args.lr_head, "encoder": args.lr_encoder},
        "epochs": args.epochs,
        "final_loss": round(float(np.mean(losses)), 4),
        "mean": {k: agg(k) for k in ("rel_acc", "rel_f1", "change_auc")},
        "change_f1_mean": round(float(np.mean([f["change"]["f1"] for f in folds])), 4),
        "change_precision_mean": round(
            float(np.mean([f["change"]["precision"] for f in folds])), 4),
        "change_recall_mean": round(
            float(np.mean([f["change"]["recall"] for f in folds])), 4),
        "ranking": {"at_1": sum(f["at_1"] for f in folds),
                    "at_5": sum(f["at_5"] for f in folds),
                    "gold_n": sum(f["gold_n"] for f in folds),
                    "mrr": round(float(np.mean([f["mrr"] for f in folds])), 4)},
        "folds": folds,
    }
    runs = Path(args.runs)
    runs.parent.mkdir(parents=True, exist_ok=True)
    runs.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=1))
    print("wrote", out_dir, "and", runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
