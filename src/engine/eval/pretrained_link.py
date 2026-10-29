"""Zero-shot test: does a pretrained causal LM prefer the real continuation?

No fine-tuning, no adapter, no prompt engineering — the model's own next-token
probabilities, teacher-forced:

    score(B | A) = mean_i log p(b_i | A, b_<i)

Three questions:

1. the user's example, in English, as a direct check of the intuition:
   "the buffet is confirmed" -> "so we pay him then?" should score higher than
   the same sentence after an unrelated or same-words-different-meaning context;
2. the sanity control on the French corpus: does the real next turn beat a
   random turn from the same meeting (per token, so length cancels);
3. the decoys used throughout: same-point/different-value and
   different-thread/same-words.

A pretrained model has seen neither corpus nor labels, so there is no split:
the whole labelled set is evaluated once.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from engine.config import repo_root
from engine.eval.mlm_link import auc, decoys
from engine.eval.real_pairs import load_meetings
from engine.eval.v4_data import load_edges, texts_of

MODEL = "Qwen/Qwen3-0.6B"

#: The example from the conversation, plus two paraphrases of the same trap.
ENGLISH = [
    {"a": "the buffet is confirmed",
     "b": "so we pay him then?",
     "negated": "the buffet is not confirmed",
     "cancelled": "the buffet is cancelled",
     "same_words": "the buffet tables arrived this morning",
     "other": "the website copy is still being written"},
    {"a": "the budget is approved at eight hundred euros",
     "b": "then we can sign the caterer",
     "negated": "the budget is not approved yet",
     "cancelled": "the budget has been cancelled",
     "same_words": "the budget spreadsheet is on the shared drive",
     "other": "the venue has a car park"},
    {"a": "we are pushing the launch to december",
     "b": "so the invitations need to move too",
     "negated": "the launch is still planned for october",
     "cancelled": "the launch has been cancelled",
     "same_words": "the launch party was fun last year",
     "other": "the invoices are due in march"},
]


class Scorer:
    def __init__(self, model_id: str = MODEL, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device if torch.cuda.is_available()
                                   or device == "cpu" else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype).to(self.device).eval()
        self.sep = self.tok("\n", add_special_tokens=False).input_ids or [198]
        self.model_id = model_id

    def encode(self, text: str) -> list[int]:
        return self.tok(text or "", add_special_tokens=False).input_ids

    @torch.no_grad()
    def score(self, ctx: str, tgt: str, max_ctx: int = 160,
              max_tgt: int = 80) -> float:
        c = self.encode(ctx)[-max_ctx:]
        t = self.encode(tgt)[:max_tgt]
        if not t:
            return float("nan")
        ids = torch.tensor([c + self.sep + t], device=self.device)
        logits = self.model(ids).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1)
        start = len(c) + len(self.sep)
        total = 0.0
        for i, tok in enumerate(t):
            total += float(logp[start + i - 1, tok])
        return total / len(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/v4/pretrained_link.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args(argv)

    root = repo_root()
    scorer = Scorer(args.model, args.device)
    out: dict = {"model": args.model, "finetuned": False}

    # 1. the English example
    demo = []
    for row in ENGLISH:
        b = row["b"]
        scored = {
            "after_gold": scorer.score(row["a"], b),
            "after_negated": scorer.score(row["negated"], b),
            "after_cancelled": scorer.score(row["cancelled"], b),
            "after_same_words": scorer.score(row["same_words"], b),
            "after_other": scorer.score(row["other"], b),
            "after_null": scorer.score("", b),
        }
        demo.append({"a": row["a"], "b": b,
                     **{k: round(v, 4) for k, v in scored.items()},
                     "rank_of_gold": 1 + sum(v > scored["after_gold"]
                                             for v in scored.values())})
    out["english_demo"] = demo

    # 2 + 3. the French corpus
    meetings = load_meetings(root)
    edges = load_edges(root)
    import random

    rng = random.Random(42)
    sanity_pos, sanity_neg = [], []
    rows = []
    for mid in sorted(edges):
        texts = texts_of(meetings, mid)
        gold_olds: dict[int, set[int]] = {}
        for e in edges[mid]:
            gold_olds.setdefault(e.new, set()).add(e.old)
        for i in range(len(texts) - 1):
            if not texts[i + 1].strip() or not texts[i].strip():
                continue
            j = i + 1
            k = rng.randrange(len(texts))
            if k in (i, j) or not texts[k].strip():
                continue
            sanity_pos.append(scorer.score(texts[i], texts[j]))
            sanity_neg.append(scorer.score(texts[i], texts[k]))
            if len(sanity_pos) >= 120:
                break
        for e in edges[mid]:
            pick = decoys(texts, e.old, e.new, gold_olds.get(e.new))
            if not pick:
                continue
            r = {"meeting": mid, "relation": e.relation,
                 "gold": scorer.score(texts[e.old], texts[e.new])}
            if "slot" in pick:
                r["slot"] = scorer.score(texts[pick["slot"]], texts[e.new])
            if "topic" in pick:
                r["topic"] = scorer.score(texts[pick["topic"]], texts[e.new])
            rows.append(r)
        print(mid, "done", flush=True)

    def block(key):
        rs = [r for r in rows if key in r]
        if not rs:
            return {"n": 0}
        return {"n": len(rs),
                "mean_gold": round(statistics.mean(r["gold"] for r in rs), 4),
                "mean_neg": round(statistics.mean(r[key] for r in rs), 4),
                "preferred": sum(r["gold"] > r[key] for r in rs),
                "auc": auc([r["gold"] for r in rs], [r[key] for r in rs])}

    out["sanity_next_turn"] = {
        "n": len(sanity_pos),
        "mean_next": round(statistics.mean(sanity_pos), 4),
        "mean_random": round(statistics.mean(sanity_neg), 4),
        "auc": auc(sanity_pos, sanity_neg)}
    out["gold_vs_slot"] = block("slot")
    out["gold_vs_topic"] = block("topic")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**out, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
