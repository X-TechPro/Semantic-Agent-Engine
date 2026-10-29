"""Train a small causal state model with CE, then ask: does it prefer the real
continuation over a same-topic decoy?

This is the test the earlier proxy did not run. `mlm_link.py` scored a masked
model one token at a time; that conditions on B's own tokens and measures local
fluency. Here the model is trained exactly like a language model — next-token
cross-entropy over the meeting streams, left to right — so its score for B is
`log p(b_i | state, b_<i)` with the state built from A only.

Mixer: a 2-layer GRU. It is a gated recurrent state, the same shape as a gated
delta network at this scale, and it needs no custom scan. If the effect shows
up, swapping in GDN2 changes capacity, not the experiment.

Sanity check included: can the model prefer the *next turn* over a random turn
from the same meeting? If that is at chance, the model learned no order at all
and the decoy result is void, not negative.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from engine.config import repo_root
from engine.eval.mlm_link import auc, decoys
from engine.eval.real_pairs import load_meetings
from engine.eval.ssm_model import SSMLM
from engine.eval.v4_data import load_edges, texts_of
from engine.store.fts import tokenize

PAD, UNK, BOUND = 0, 1, 2


class Vocab:
    def __init__(self, streams: list[list[str]], min_freq: int = 2):
        c = Counter(t for s in streams for t in s)
        self.itos = ["<pad>", "<unk>", "<turn>"] + [w for w, n in c.items()
                                                    if n >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk = UNK

    def encode(self, words: list[str]) -> list[int]:
        return [self.stoi.get(w, self.unk) for w in words]

    def __len__(self) -> int:
        return len(self.itos)


class CausalGRU(nn.Module):
    def __init__(self, vocab: int, emb: int = 128, hidden: int = 256,
                 layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.rnn = nn.GRU(emb, hidden, layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.drop(self.emb(ids)))
        return self.out(self.drop(h))


def stream_of(texts: list[str]) -> list[str]:
    out: list[str] = []
    for t in texts:
        out += tokenize(t)
        out.append("<turn>")
    return out


def windows(ids: list[int], size: int, stride: int):
    for s in range(0, max(1, len(ids) - size), stride):
        w = ids[s:s + size + 1]
        if len(w) > 3:
            yield w


def train_model(streams: list[list[int]], vocab_size: int, *, epochs: int,
                device: str, seed: int, hidden: int = 256, arch: str = "ssm",
                size: int = 192, stride: int = 96, batch: int = 32
                ) -> tuple[nn.Module, list]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    if arch == "ssm":
        model = SSMLM(vocab_size, d_model=128 if hidden <= 256 else hidden,
                      layers=2, d_state=16).to(device)
    elif arch == "gru":
        model = CausalGRU(vocab_size, hidden=hidden).to(device)
    else:
        raise ValueError(f"unknown arch {arch!r}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    chunks = [w for s in streams for w in windows(s, size, stride)]
    losses = []
    for _ep in range(epochs):
        rng.shuffle(chunks)
        model.train()
        for s in range(0, len(chunks), batch):
            b = chunks[s:s + batch]
            width = max(len(x) for x in b) - 1
            ids = torch.tensor([x[:-1] + [PAD] * (width - len(x) + 1)
                                for x in b], device=device)
            tgt = torch.tensor([x[1:] + [PAD] * (width - len(x) + 1)
                                for x in b], device=device)
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
    return model, losses


@torch.no_grad()
def score(model: nn.Module, vocab: Vocab, ctx: str, tgt: str,
          device: str, max_ctx: int = 64, max_tgt: int = 64) -> float:
    """Sum log p(target | context), one causal pass."""
    c = vocab.encode(tokenize(ctx))[-max_ctx:]
    t = vocab.encode(tokenize(tgt))[:max_tgt]
    if not t:
        return 0.0
    ids = torch.tensor([c + [BOUND] + t], device=device)
    logits = model(ids)[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    start = len(c) + 1
    total = 0.0
    for i, tok in enumerate(t):
        pos = start + i - 1          # logits[pos] predicts ids[pos+1]
        total += float(logp[pos, tok])
    return total / len(t)            # per-token: targets differ in length


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/v4/ssm_link.json"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--arch", choices=("ssm", "gru"), default="ssm")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    device = args.device if torch.cuda.is_available() else "cpu"

    root = repo_root()
    meetings = load_meetings(root)
    edges = load_edges(root)
    mids = sorted(meetings)

    folds = []
    for held in mids:
        if held not in edges:
            continue
        train_streams = []
        for mid in mids:
            if mid == held:
                continue
            train_streams.append(stream_of(texts_of(meetings, mid)))
        vocab = Vocab(train_streams)
        streams = [vocab.encode(s) for s in train_streams]
        model, losses = train_model(streams, len(vocab), epochs=args.epochs,
                                    device=device, seed=args.seed,
                                    hidden=args.hidden, arch=args.arch)
        model.eval()

        texts = texts_of(meetings, held)
        gold_olds: dict[int, set[int]] = {}
        for e in edges[held]:
            gold_olds.setdefault(e.new, set()).add(e.old)

        # sanity: next turn vs a random turn from the same meeting
        rng = random.Random(args.seed)
        pairs = [(i, i + 1) for i in range(len(texts) - 1) if texts[i + 1].strip()]
        rng.shuffle(pairs)
        pairs = pairs[:80]
        pos, neg = [], []
        for i, j in pairs:
            k = rng.randrange(len(texts))
            if k in (i, j) or not texts[k].strip():
                continue
            pos.append(score(model, vocab, texts[i], texts[j], device))
            neg.append(score(model, vocab, texts[i], texts[k], device))
        sanity = {"n": len(pos), "auc": auc(pos, neg)}

        rows = []
        for e in edges[held]:
            pick = decoys(texts, e.old, e.new, gold_olds.get(e.new))
            if not pick:
                continue
            r = {"relation": e.relation, "old": e.old, "new": e.new,
                 "gold": score(model, vocab, texts[e.old], texts[e.new], device)}
            if "slot" in pick:
                r["slot"] = score(model, vocab, texts[pick["slot"]],
                                  texts[e.new], device)
            if "topic" in pick:
                r["topic"] = score(model, vocab, texts[pick["topic"]],
                                   texts[e.new], device)
            rows.append(r)

        def block(key):
            rs = [r for r in rows if key in r]
            if not rs:
                return {"n": 0}
            return {"n": len(rs),
                    "mean_gold": round(sum(r["gold"] for r in rs) / len(rs), 3),
                    "mean_neg": round(sum(r[key] for r in rs) / len(rs), 3),
                    "preferred": sum(r["gold"] > r[key] for r in rs),
                    "auc": auc([r["gold"] for r in rs],
                               [r[key] for r in rs])}

        fold = {"held": held, "train_loss": round(sum(losses) / len(losses), 4),
                "vocab": len(vocab), "sanity": sanity,
                "gold_vs_slot": block("slot"), "gold_vs_topic": block("topic"),
                "by_relation": {rel: {"slot": block_rel(rows, rel, "slot"),
                                      "topic": block_rel(rows, rel, "topic")}
                                for rel in ("supports", "supersedes",
                                            "contradicts")}}
        folds.append(fold)
        print(json.dumps(fold), flush=True)
        del model
        torch.cuda.empty_cache()

    def mean(key, sub=None):
        vals = []
        for f in folds:
            v = f[key] if sub is None else f[key].get(sub)
            if isinstance(v, (int, float)):
                vals.append(v)
        return round(sum(vals) / len(vals), 4) if vals else None

    result = {
        "model": ("selective SSM (S6-style, pure torch)" if args.arch == "ssm"
                  else "causal GRU") + ", next-token CE",
        "arch": args.arch, "epochs": args.epochs, "hidden": args.hidden,
        "folds": folds,
        "mean": {
            "train_loss": mean("train_loss"),
            "sanity_auc": mean("sanity", "auc"),
            "slot_auc": mean("gold_vs_slot", "auc"),
            "topic_auc": mean("gold_vs_topic", "auc"),
        },
        "reference": {"cosine_slot_auc": 0.4422, "cosine_topic_auc": 0.6658,
                      "masked_lm_slot_auc": 0.4618,
                      "masked_lm_topic_auc": 0.541},
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}, indent=1))
    print("wrote", p)
    return 0


def block_rel(rows, relation, key):
    from engine.eval.mlm_link import auc as _auc
    rs = [r for r in rows if r["relation"] == relation and key in r]
    if not rs:
        return {"n": 0}
    return {"n": len(rs), "auc": _auc([r["gold"] for r in rs],
                                      [r[key] for r in rs])}


if __name__ == "__main__":
    raise SystemExit(main())
