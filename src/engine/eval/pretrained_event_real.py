"""Pretrained embedding + synthetic-relation fine-tune, real SUMM-RE read.

N2-model's relation file gives old->new event pairs. We train a Qwen3 embedding
with LoRA only on those synthetic pairs, then measure the real pairs:
new rank by old-sentence cosine, old rank, and old/new vs negative cosine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, losses
from peft import LoraConfig

from engine.config import repo_root

SPLITS = ("train", "sel")
GOLD = {"train": "summre_temporal_train.json", "sel": "summre_temporal_sel.json"}
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
BUNDLE = "N2-model/piscine_summre_synthetic"


def load_synthetic_pairs(root: Path, split: str) -> list[tuple[str, str]]:
    units = {}
    for line in (root / BUNDLE / "units" / f"{split}.jsonl").read_text().splitlines():
        if line.strip():
            u = json.loads(line); units[u["unit_id"]] = u
    out = []
    for line in (root / BUNDLE / "relations" / f"{split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["relation"] != "supersedes":
            continue
        a, b = units.get(r["from_unit_id"]), units.get(r["to_unit_id"])
        if a and b:
            out.append((a["statement"], b["statement"]))
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


def encode(model: SentenceTransformer, texts: list[str], batch: int = 32) -> np.ndarray:
    return model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False)


def real_metrics(model, turns, pairs, mids, cache=None):
    cache = cache or {}
    vecs = {}
    for mid in mids:
        if mid not in cache:
            texts = [t["text"] for t in turns[mid]]
            cache[mid] = encode(model, texts)
        vecs[mid] = cache[mid]
    rows = []
    rng = np.random.default_rng(0)
    for mid in mids:
        for topic, old, new in pairs[mid]:
            V = vecs[mid]
            q = V[old]
            s = V @ q
            order = np.argsort(-s)
            r_old = int(np.where(order == old)[0][0]) + 1
            r_new = int(np.where(order == new)[0][0]) + 1
            neg = int(rng.integers(0, len(V)))
            rows.append({"meeting": mid, "topic": topic, "old": old, "new": new,
                         "old_rank": r_old, "new_rank": r_new,
                         "old_at_1": r_old == 1, "new_at_1": r_new == 1,
                         "new_at_5": r_new <= 5,
                         "cos_old_new": float(np.dot(V[old], V[new])),
                         "cos_old_neg": float(np.dot(V[old], V[neg]))})
    n = len(rows)
    return {
        "n": n,
        "new_at_1": sum(r["new_at_1"] for r in rows),
        "old_at_1": sum(r["old_at_1"] for r in rows),
        "new_at_5": sum(r["new_at_5"] for r in rows),
        "cos_old_new": round(float(np.mean([r["cos_old_new"] for r in rows])), 4),
        "cos_old_neg": round(float(np.mean([r["cos_old_neg"] for r in rows])), 4),
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(repo_root() / "runs/pretrained_event/result.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    turns, pairs = load_real(root)
    mids = sorted(pairs)
    train_pairs = load_synthetic_pairs(root, "train")
    print("synthetic supersedes pairs", len(train_pairs), flush=True)
    model = SentenceTransformer(args.model, device="cuda")
    print("zero-shot", flush=True)
    zs = real_metrics(model, turns, pairs, mids)
    print({k: zs[k] for k in ("n", "new_at_1", "old_at_1", "new_at_5",
                              "cos_old_new", "cos_old_neg")}, flush=True)
    examples = [InputExample(texts=[a, b]) for a, b in train_pairs]
    loader = DataLoader(examples, batch_size=args.batch, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.add_adapter(lora)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=max(1, len(loader) // 10),
              optimizer_params={"lr": 2e-4}, use_amp=False,
              show_progress_bar=True)
    print("fine-tuned", flush=True)
    ft = real_metrics(model, turns, pairs, mids)
    print({k: ft[k] for k in ("n", "new_at_1", "old_at_1", "new_at_5",
                              "cos_old_new", "cos_old_neg")}, flush=True)
    out = {"model": args.model, "train_pairs": len(train_pairs),
           "epochs": args.epochs, "zero_shot": zs, "fine_tuned": ft}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
