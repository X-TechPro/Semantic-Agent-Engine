"""SUPERSEDED — numbers from this script are not relation results.

`inokufu/flaubert-base-uncased-xnli-sts` is a bare `FlaubertModel`: it has no
`classifier` and no `sequence_summary` weights. `from_pretrained` therefore
builds a **randomly initialised** 2-logit head, so everything this script
measures is a random projection of the encoder, not a zero-shot NLI/STS score.
It is kept to reproduce the earlier artefacts; the working path is
`engine.link.flaubert_relation` + `engine.link.train_relation`.

Fine-tune FlauBERT XNLI/STS on real pairs with structural hard negatives.

Positive: gold old antecedent for an incoming new turn.
Negatives: same meeting / same active topic or same entity, but not the gold
antecedent and not another gold positive for the same query. No synthetic text.
Leave-one-meeting-out; candidate pool comes from the v2 real stage.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from engine.config import repo_root
from engine.eval.continuity_stages_v2 import active_topics
from engine.eval.real_pairs import load_meetings, load_pairs, pair_rows
from engine.link.thread import detect_topic
from engine.retrieve.entity_index import extract_entities
from engine.store.fts import tokenize

MODEL = "inokufu/flaubert-base-uncased-xnli-sts"


def pair_score(model, tok, queries, candidates, device, batch=16):
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(queries), batch):
            q = queries[start:start+batch]
            c = candidates[start:start+batch]
            enc = tok(q, c, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(device)
            lg = model(**enc).logits.float()
            # 2-logit head: class 1 minus class 0. Stable for ranking.
            out += (lg[:, 1] - lg[:, 0]).detach().cpu().tolist()
    return out


def make_training(mid, turns, pairs, gold_pairs):
    topics = active_topics(turns)
    texts = [str(t.get("text") or "") for t in turns]
    # gold old for a given new, used to avoid a true positive as a negative
    pos_for = {}
    for _topic, old, new in pairs:
        pos_for.setdefault(new, set()).add(old)
    rows = []
    for topic, old, new in pairs:
        qctx = " || ".join(texts[max(0, new-2):new+1])
        qents = set()
        for vals in extract_entities(texts[new]).values():
            qents.update(v.lower() for v in vals)
        scored = []
        for j in range(len(texts)):
            if j in (old, new) or j in pos_for.get(new, set()):
                continue
            same_topic = bool(topics[new] and topics[j] == topics[new])
            ents = set()
            for vals in extract_entities(texts[j]).values():
                ents.update(v.lower() for v in vals)
            same_entity = bool(qents & ents)
            if not (same_topic or same_entity):
                continue
            score = (3 if same_topic else 0) + (2 if same_entity else 0) + \
                    (0.5 if abs(j-new) <= 5 else 0)
            scored.append((score, j))
        scored.sort(reverse=True)
        negs = [texts[j] for _, j in scored[:8]]
        if len(negs) < 3:
            negs += [texts[j] for j in range(max(0, new-10), min(len(texts), new+11))
                     if j not in (old, new)][:3]
        rows.append({"mid": mid, "query": qctx, "positive": texts[old],
                     "negatives": negs})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/flaubert_finetune/real.json"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args(argv)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    root = repo_root()
    meetings = load_meetings(root)
    pairs = load_pairs(root)
    stage = json.loads((root / "runs/continuity_stages_v2/result.json").read_text())
    by_pair = {(r["meeting"], r["old"], r["new"]): r for r in stage["rows"]}
    mids = sorted(pairs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = []
    for held in mids:
        train_mids = [m for m in mids if m != held]
        train = []
        for mid in train_mids:
            gold = {(o, n) for _, o, n in pairs[mid]}
            train += make_training(mid, meetings[mid], pairs[mid], gold)
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        rng = random.Random(42 + len(held))
        for ep in range(args.epochs):
            rng.shuffle(train); model.train(); losses = []
            for start in range(0, len(train), args.batch):
                batch = train[start:start+args.batch]
                qs = [x["query"] for x in batch]
                ps = [x["positive"] for x in batch]
                ns = [rng.choice(x["negatives"]) for x in batch]
                ep_ = tok(qs, ps, padding=True, truncation=True, max_length=256,
                          return_tensors="pt").to(device)
                en = tok(qs, ns, padding=True, truncation=True, max_length=256,
                         return_tensors="pt").to(device)
                sp = model(**ep_).logits.float()
                sn = model(**en).logits.float()
                sp = sp[:, 1] - sp[:, 0]
                sn = sn[:, 1] - sn[:, 0]
                loss = F.softplus(0.5 + sn - sp).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); losses.append(float(loss.detach()))
        # held-out candidate pool from v2
        test = []
        for r in stage["rows"]:
            if r["meeting"] != held:
                continue
            turns = meetings[held]; texts = [str(t.get("text") or "") for t in turns]
            old, new = r["old"], r["new"]
            qctx = " || ".join(texts[max(0, new-2):new+1])
            cand = r["candidate_turns"]
            if old not in cand:
                continue
            scores = pair_score(model, tok, [qctx]*len(cand),
                                [texts[j] for j in cand], device)
            order = [cand[i] for i in np.argsort(-np.asarray(scores))]
            rank = order.index(old) + 1
            test.append(rank)
        folds.append({"held": held, "train_rows": len(train),
                      "n": len(test), "at_1": sum(r == 1 for r in test),
                      "at_5": sum(r <= 5 for r in test),
                      "mrr": round(sum(1.0/r for r in test)/len(test), 4) if test else 0,
                     "loss": round(float(np.mean(losses)), 4) if losses else None})
        del model
        torch.cuda.empty_cache()
        print(folds[-1], flush=True)
    out = {"model": MODEL, "real_only": True,
           "n": sum(f["n"] for f in folds),
           "at_1": sum(f["at_1"] for f in folds),
           "at_5": sum(f["at_5"] for f in folds),
           "mrr": round(sum(f["mrr"]*f["n"] for f in folds) /
                        max(1, sum(f["n"] for f in folds)), 4),
           "folds": folds}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
