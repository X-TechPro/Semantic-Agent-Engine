"""The linker's text relation head, trained on XNLI-fr (6.8b / 7.2's NLI start).

Why this module exists — measured on 2026-10-21 (`runs/frbench/xnli_pairclassifier_test_full`,
`runs/frbench/xnli-rerank_crossencoder_*`):

* the synthetic-trained pair classifier scores real French pairs at chance
  (AUC 0.44-0.54 vs 0.5; mean score 0.005 against its 0.845 threshold), so the
  shipped linker proposes **no** text-only relations on external data;
* the cross-encoder the repo already ships (bge-reranker-v2-m3) detects
  restatement well (entail-vs-neutral AUC 0.83) but ranks a contradiction
  **below an unrelated pair** (mean 0.061 vs 0.272) — relevance is not
  compatibility, exactly the "proximity is not logic" warning in 7.2. So simply
  wiring it in would mis-rank real supersede candidates; a trained head is required.

Model: full fine-tune of `almanach/moderncamembert-base` (cached locally, French,
same family as the reader's encoder -> one tokenizer story) over the three
relations XNLI's own README mapping supplies (0=entailment->supports,
1=neutral->unrelated, 2=contradiction->contradicts). Tuning and epoch selection use
**validation**; test is read once at the end (9.5). The checkpoint carries the
corpus revision sha from PROVENANCE.json so the exact training data is identifiable
months later.

Usage:
    PYTHONPATH=src python -m engine.link.nli_finetune            # train (GPU ~5 min)
    PYTHONPATH=src python -m engine.link.nli_finetune --eval-only
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

LABELS = ("supports", "unrelated", "contradicts")      # index = our class
XNLI2OUR = {0: 0, 1: 1, 2: 2}                          # entail / neutral / contradict
BASE = "almanach/moderncamembert-base"

from engine.config import MODELS_DIR, repo_root        # noqa: E402

OUT_DEFAULT = MODELS_DIR / "linker_nli"
XNLI_DIR = repo_root() / "data/external/facebook__xnli"


def load_fr(split: str) -> list[tuple[str, str, int]]:
    import pyarrow.parquet as pq
    prov = json.loads((XNLI_DIR / "PROVENANCE.json").read_text())
    fname = next(f for f in prov["files"] if f"fr/{split}-" in f)
    t = pq.read_table(XNLI_DIR / fname)
    rows = [(t.column("premise")[i].as_py(), t.column("hypothesis")[i].as_py(),
             int(t.column("label")[i].as_py())) for i in range(t.num_rows)]
    return [(a, b, XNLI2OUR[c]) for a, b, c in rows]


def macro_f1_and_confusion(gold: list[int], pred: list[int]) -> dict:
    """Per-class P/R/F1 + macro F1 + the matrix 9.1 asks for, computed here so the
    artifact is self-describing (and a 0.0 F1 stays a 0.0, never a missing key)."""
    out: dict = {}
    f1s = []
    for c, name in enumerate(LABELS):
        tp = sum(1 for g, p in zip(gold, pred) if g == c and p == c)
        fp = sum(1 for g, p in zip(gold, pred) if g != c and p == c)
        fn = sum(1 for g, p in zip(gold, pred) if g == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[name] = {"precision": round(prec, 4), "recall": round(rec, 4),
                     "f1": round(f1, 4), "n_gold": tp + fn, "n_pred": tp + fp}
        f1s.append(f1)
    conf = {LABELS[g]: {} for g in set(gold)}
    for g, p in zip(gold, pred):
        row = conf[LABELS[g]]
        row[LABELS[p]] = row.get(LABELS[p], 0) + 1
    return {"per_class": out, "macro_f1": round(sum(f1s) / len(f1s), 4),
            "accuracy": round(sum(1 for g, p in zip(gold, pred) if g == p)
                              / max(1, len(gold)), 4),
            "confusion": {k: dict(sorted(v.items())) for k, v in sorted(conf.items())}}


def _enc(tok, pairs, maxlen, device):
    return tok([p[0] for p in pairs], [p[1] for p in pairs], truncation=True,
               padding="max_length", max_length=maxlen, return_tensors="pt").to(device)


@torch.no_grad()
def evaluate(model, tok, rows: list[tuple[str, str, int]], device: str,
             maxlen: int = 128, batch: int = 64) -> dict:
    model.eval()
    gold, pred = [], []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        enc = _enc(tok, chunk, maxlen, device)
        out = model(**enc)
        pred += out.logits.argmax(-1).tolist()
        gold += [r[2] for r in chunk]
    return macro_f1_and_confusion(gold, pred)


def train(*, n_train: int = 60000, epochs: int = 2, lr: float = 2e-5,
          batch: int = 32, maxlen: int = 128, seed: int = 42,
          out_dir: Path | None = None, device: str | None = None) -> dict:
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(seed)
    random.seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir or OUT_DEFAULT)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE, num_labels=3,
        id2label={i: n for i, n in enumerate(LABELS)},
        label2id={n: i for i, n in enumerate(LABELS)})
    model.to(dev)

    train_rows = load_fr("train")
    random.Random(seed).shuffle(train_rows)
    train_rows = train_rows[:n_train]
    val_rows = load_fr("validation")          # tuning split -- never test here
    steps_per_epoch = max(1, len(train_rows) // batch)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(
        opt, int(0.06 * steps_per_epoch * epochs), steps_per_epoch * epochs)

    curve, best = [], {"macro_f1": -1.0}
    for ep in range(epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.Random(seed + ep).shuffle(order)
        run_loss, n_seen = 0.0, 0
        for si, start in enumerate(range(0, steps_per_epoch * batch, batch)):
            chunk = [train_rows[order[j]] for j in
                     range(start, min(start + batch, len(train_rows)))]
            enc = _enc(tok, chunk, maxlen, dev)
            y = torch.tensor([c[2] for c in chunk], device=dev)
            out = model(**enc, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += float(out.loss.detach()); n_seen += 1
            if si % 200 == 0:
                print(f"epoch {ep} step {si}/{steps_per_epoch} "
                      f"loss {run_loss/max(1,n_seen):.4f}", flush=True)
        ev = evaluate(model, tok, val_rows, str(dev), maxlen)
        curve.append({"epoch": ep, "train_loss": round(run_loss / max(1, n_seen), 4),
                      "val": ev})
        print(f"epoch {ep}: val acc={ev['accuracy']} macro_f1={ev['macro_f1']}",
              flush=True)
        if ev["macro_f1"] > best["macro_f1"]:
            best = ev
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "weights.pt")
            (out_dir / "best.json").write_text(json.dumps(
                {"epoch": ep, "val": ev}, indent=1), encoding="utf-8")

    prov = json.loads((XNLI_DIR / "PROVENANCE.json").read_text())
    meta = {"base_model": BASE, "task": "xnli-fr -> supports/unrelated/contradicts",
            "labels": list(LABELS), "data": {"repo": prov["repo"],
                                             "revision": prov["revision"],
                                             "n_train_rows": len(train_rows),
                                             "val_rows": len(val_rows)},
            "hparams": {"epochs": epochs, "lr": lr, "batch": batch,
                        "maxlen": maxlen, "seed": seed},
            "curve": curve, "best_val": best["macro_f1"],
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "tuning/selection on validation only (9.5); test read once by "
                    "the frbench eval, never here"}
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    return meta


def load_head(path: Path | None = None, device: str | None = None):
    """Return (model, tokenizer, meta), or None if absent -- callers must not be
    able to pretend the head ran when it was never trained."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = Path(path or OUT_DEFAULT)
    if not (path / "weights.pt").exists():
        return None
    meta = json.loads((path / "meta.json").read_text())
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(meta["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        meta["base_model"], num_labels=3,
        id2label={i: n for i, n in enumerate(LABELS)},
        label2id={n: i for i, n in enumerate(LABELS)})
    model.load_state_dict(torch.load(path / "weights.pt", map_location="cpu",
                                     weights_only=True))
    model.to(dev).eval()
    return model, tok, meta


class NLITextHead:
    """The trained head behind the linker's text-only branch (`link/supervised.py`).

    `relate(new_text, old_text)` returns one of LABELS with the premise = the OLD
    unit and hypothesis = the NEW one — the same orientation frbench scored when it
    read the test split once, so the 0.7986 number describes this exact call path.
    One forward per pair: candidate counts are small and a batching layer here
    would be a cache to invalidate, not a saving worth making tonight.
    """

    def __init__(self, model, tok, meta):
        self.model, self.tok, self.meta = model, tok, meta
        self.maxlen = int(meta["hparams"]["maxlen"])
        self.device = next(model.parameters()).device

    @classmethod
    def from_dir(cls, path=None, device=None):
        got = load_head(path, device)
        return None if got is None else cls(*got)

    @torch.no_grad()
    def relate(self, new_text: str, old_text: str) -> str:
        enc = self.tok([old_text], [new_text], truncation=True,
                       padding="longest", max_length=self.maxlen,
                       return_tensors="pt").to(self.device)
        logits = self.model(**enc).logits
        return LABELS[int(logits.argmax(-1)[0])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--maxlen", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--eval-only", action="store_true",
                    help="score the saved head on validation (no training)")
    args = ap.parse_args(argv)
    if args.eval_only:
        head = load_head(device=args.device)
        if head is None:
            print("no checkpoint at", OUT_DEFAULT, "-- run training first")
            return 1
        model, tok, meta = head
        ev = evaluate(model, tok, load_fr("validation"),
                      args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(json.dumps(ev, indent=1))
        return 0
    m = train(n_train=args.n_train, epochs=args.epochs, lr=args.lr,
              batch=args.batch, maxlen=args.maxlen, seed=args.seed,
              device=args.device)
    print("best val macro_f1:", m["best_val"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
