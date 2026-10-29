"""Fine-tune a cross-encoder on synthetic supersedes pairs, test real pool."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings
from engine.link.cross_encoder_linker import Candidate

BUNDLE = Path("N2-model/piscine_summre_synthetic")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/ce_finetune/result.json"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args(argv)
    root = repo_root()
    units = {}
    for split in ("train", "validation"):
        for u in read_jsonl(root / BUNDLE / "units" / f"{split}.jsonl"):
            units[u["unit_id"]] = u
    pos, neg = [], []
    rng = random.Random(42)
    for split in ("train", "validation"):
        for r in read_jsonl(root / BUNDLE / "relations" / f"{split}.jsonl"):
            a, b = units.get(r["from_unit_id"]), units.get(r["to_unit_id"])
            if not a or not b:
                continue
            if r["relation"] == "supersedes":
                pos.append((a["statement"], b["statement"]))
                pos.append((b["statement"], a["statement"]))
            elif r["relation"] == "unrelated":
                neg.append((a["statement"], b["statement"]))
    # Same-project hard negatives.
    by_proj = {}
    for u in units.values():
        by_proj.setdefault(u.get("project_id", ""), []).append(u)
    for _ in range(len(pos)):
        a = rng.choice(list(units.values()))
        pool = by_proj.get(a.get("project_id", ""), [])
        if len(pool) < 2:
            continue
        b = rng.choice(pool)
        if b["unit_id"] != a["unit_id"]:
            neg.append((a["statement"], b["statement"]))
    examples = [InputExample(texts=[a, b], label=1.0) for a, b in pos]
    examples += [InputExample(texts=[a, b], label=0.0) for a, b in neg]
    rng.shuffle(examples)
    print("train", len(examples), "pos", len(pos), "neg", len(neg), flush=True)
    model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
    loader = DataLoader(examples, batch_size=args.batch, shuffle=True)
    model.fit(loader, epochs=args.epochs, warmup_steps=10,
              optimizer_params={"lr": 2e-5}, use_amp=False,
              show_progress_bar=True)
    model.save(str(root / "runs/ce_finetune/model"))
    stage = json.loads((root / "runs/continuity_stages_v2/result.json").read_text())
    meetings = load_meetings(root)
    rows = []
    for r in stage["rows"]:
        mid, old, new = r["meeting"], r["old"], r["new"]
        cand = r["candidate_turns"]
        if not cand:
            continue
        turns = meetings[mid]
        qctx = " || ".join(str(turns[k].get("text") or "")
                           for k in range(max(0, new - 2), new + 1))
        cs = [Candidate(turn=j, text=str(turns[j].get("text") or ""),
                        speaker=str(turns[j].get("speaker") or ""),
                        timestamp=str(turns[j].get("timestamp") or ""))
              for j in cand]
        pairs = [(qctx, c.text) for c in cs]
        scores = model.predict(pairs, batch_size=32, show_progress_bar=False)
        order = [c.turn for _, c in sorted(zip(scores, cs), key=lambda x: -x[0])]
        rec = {"meeting": mid, "old": old, "new": new,
               "ce_ft_rank": order.index(old) + 1 if old in order else None}
        pairs_ctx = [(qctx, " || ".join(str(turns[k].get("text") or "")
                                       for k in range(max(0, c.turn - 1),
                                                      min(len(turns), c.turn + 2))))
                     for c in cs]
        scores_ctx = model.predict(pairs_ctx, batch_size=32,
                                   show_progress_bar=False)
        order_ctx = [c.turn for _, c in sorted(zip(scores_ctx, cs),
                                               key=lambda x: -x[0])]
        rec["ce_ft_context_rank"] = (order_ctx.index(old) + 1
                                     if old in order_ctx else None)
        rows.append(rec)
    def m(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"covered": len(vals), "at_1": sum(v == 1 for v in vals),
                "at_5": sum(v <= 5 for v in vals),
                "mrr": round(sum(1.0 / v for v in vals) / len(rows), 4)}
    out = {"train_examples": len(examples), "epochs": args.epochs,
           "metrics": {k: m(k) for k in ("ce_ft_rank", "ce_ft_context_rank")},
           "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
