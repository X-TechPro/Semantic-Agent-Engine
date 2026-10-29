"""FlauBERT triplet fine-tune: synthetic structural pairs -> real held-out test.

Training uses synthetic typed relations only. The real 40 pairs are never in
training. Negatives are same-project/same-field but not the typed positive.
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
from continuity_stages_v2 import active_topics
from engine.eval.real_pairs import load_meetings
from engine.store.fts import tokenize

BUNDLE = Path("N2-model/piscine_summre_synthetic")
MODEL = "inokufu/flaubert-base-uncased-xnli-sts"


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def score(model, tok, qs, cs, device, batch=16):
    out = []; model.eval()
    with torch.no_grad():
        for s in range(0, len(qs), batch):
            enc = tok(qs[s:s+batch], cs[s:s+batch], padding=True,
                      truncation=True, max_length=256, return_tensors="pt").to(device)
            lg = model(**enc).logits.float()
            out += (lg[:, 1] - lg[:, 0]).cpu().tolist()
    return out


def make_triplets(root):
    units = {}
    for split in ("train", "validation"):
        for u in read_jsonl(root / BUNDLE / "units" / f"{split}.jsonl"):
            units[u["unit_id"]] = u
    buckets = {}
    for u in units.values():
        buckets.setdefault((u.get("project_id"), u.get("field")), []).append(u)
    rng = random.Random(42)
    rows = []
    for split in ("train", "validation"):
        for r in read_jsonl(root / BUNDLE / "relations" / f"{split}.jsonl"):
            if r["relation"] != "supersedes":
                continue
            a, b = units.get(r["from_unit_id"]), units.get(r["to_unit_id"])
            if not a or not b:
                continue
            pool = [u for u in buckets.get((b.get("project_id"), b.get("field")), [])
                    if u["unit_id"] not in (a["unit_id"], b["unit_id"])]
            if not pool:
                continue
            negs = [u["statement"] for u in rng.sample(pool, min(4, len(pool)))]
            rows.append({"query": b["statement"], "positive": a["statement"],
                         "negatives": negs})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/flaubert_finetune/synth.json"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    root = repo_root()
    train = make_triplets(root)
    print("synthetic triplets", len(train), flush=True)
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    rng = random.Random(42)
    for ep in range(args.epochs):
        rng.shuffle(train); model.train(); losses = []
        for s in range(0, len(train), args.batch):
            batch = train[s:s+args.batch]
            qs = [x["query"] for x in batch]
            ps = [x["positive"] for x in batch]
            ns = [rng.choice(x["negatives"]) for x in batch]
            ep_ = tok(qs, ps, padding=True, truncation=True, max_length=256,
                      return_tensors="pt").to(dev)
            en = tok(qs, ns, padding=True, truncation=True, max_length=256,
                     return_tensors="pt").to(dev)
            sp = model(**ep_).logits.float(); sp = sp[:, 1] - sp[:, 0]
            sn = model(**en).logits.float(); sn = sn[:, 1] - sn[:, 0]
            loss = F.softplus(0.5 + sn - sp).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); losses.append(float(loss.detach()))
        print("epoch", ep, "loss", round(float(np.mean(losses)), 4), flush=True)
    stage = json.loads((root / "runs/continuity_stages_v2/result.json").read_text())
    meetings = load_meetings(root)
    rows = []
    for r in stage["rows"]:
        mid, old, new = r["meeting"], r["old"], r["new"]
        cand = r["candidate_turns"]
        if old not in cand:
            continue
        turns = meetings[mid]; texts = [str(t.get("text") or "") for t in turns]
        qctx = " || ".join(texts[max(0, new-2):new+1])
        sc = score(model, tok, [qctx]*len(cand), [texts[j] for j in cand], dev)
        order = [cand[i] for i in np.argsort(-np.asarray(sc))]
        rank = order.index(old)+1
        rows.append({"meeting": mid, "old": old, "new": new, "rank": rank})
    out = {"model": MODEL, "train_rows": len(train), "real_test": len(rows),
           "at_1": sum(r["rank"] == 1 for r in rows),
           "at_5": sum(r["rank"] <= 5 for r in rows),
           "mrr": round(sum(1.0/r["rank"] for r in rows)/len(rows), 4) if rows else 0,
           "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
