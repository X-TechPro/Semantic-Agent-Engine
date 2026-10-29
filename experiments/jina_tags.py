"""Step 3, apples to apples: same rows, same splits, same head, different encoder.

The earlier comparison was not fair — a fine-tuned Laya head against a zero-shot
jina embedding. This trains both encoders the same way:

    frozen   encoder frozen, only a 2-layer head trains  (representation probe)
    tuned    encoder's last 4 layers + head train        (real fine-tune)

Labels, task views, class balancing, epochs and label smoothing are identical to
`engine.read.laya_tags`, so any difference is the encoder.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from engine.read.laya_tags import TASKS, _balance, _prf, _ece, predict  # noqa: E402

V5 = "jinaai/jina-embeddings-v5-text-small"
STATE_BEARING = ("proposed", "confirmed", "cancelled", "completed")


class V5Encoder:
    name = "jina-v5-small"
    task = "classification"

    def __init__(self, device: str = "cuda", pooling: str = "last"):
        self.pooling = pooling
        from transformers import AutoModel

        self.device = device
        self.m = AutoModel.from_pretrained(
            V5, trust_remote_code=True, dtype=torch.bfloat16).to(device)
        self.tok = self.m.tokenizer
        self.tok.padding_side = "right"
        self.m.set_adapter(self.task)
        self.base = self.m.base_model.model
        self.dim = self.base.config.hidden_size

    def tokens(self, texts: list[str], max_len: int = 96):
        return self.tok(texts, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")

    def forward(self, batch) -> torch.Tensor:
        out = self.base(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
        h = out.last_hidden_state
        if self.pooling == "mean":
            mask = batch["attention_mask"].unsqueeze(-1).float()
            return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        idx = batch["attention_mask"].sum(1) - 1
        return h[torch.arange(h.size(0), device=h.device), idx]

    def patch_tensors(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}


class LayaEncoder:
    name = "laya-multilingual"
    task = 0

    def __init__(self, device: str = "cuda", subfolder: str = "multilingual"):
        from engine.read import laya_tags

        self.device = device
        self.model, self.tok, _, _ = laya_tags.load_laya(subfolder)
        self.model.to(device)
        self.base = self.model.encoder
        self.dim = self.base.config.hidden_size

    def tokens(self, texts: list[str], max_len: int = 96):
        return self.tok(texts, padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")

    def forward(self, batch) -> torch.Tensor:
        out = self.base(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
        h = out.last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def patch_tensors(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}


class Head(nn.Module):
    def __init__(self, dim: int, n: int, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                                 nn.Dropout(p), nn.Linear(256, n))

    def forward(self, x):
        return self.net(x)


def _views(data, cap):
    return {t: _balance(data["train"], t, cap, 13) for t in TASKS}


def _train_step(enc, heads, rows, task, opt, device, max_len):
    batch = enc.patch_tensors(enc.tokens([r["text"] for r in rows], max_len))
    labels = torch.tensor([TASKS[task]["labels"].index(r[task]) for r in rows],
                          device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        feat = enc.forward(batch)
    logits = heads[task](feat.float())
    loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups
                                    for p in g["params"]], 1.0)
    opt.step()
    return float(loss)


@torch.no_grad()
def _probs(enc, head, texts, device, batch=32, max_len=96):
    enc.base.eval()
    out = []
    for s in range(0, len(texts), batch):
        b = enc.patch_tensors(enc.tokens(texts[s:s + batch], max_len))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            feat = enc.forward(b)
        out.append(torch.softmax(head(feat.float()), -1).cpu())
    return torch.cat(out).numpy()


def evaluate(enc, heads, rows, task, device):
    labels = TASKS[task]["labels"]
    rows = [r for r in rows if r.get(task) in labels
            and task in r.get("tasks", ())]
    probs = _probs(enc, heads[task], [r["text"] for r in rows], device)
    pred = [labels[int(i)] for i in probs.argmax(1)]
    gold = [r[task] for r in rows]
    res = _prf(gold, pred, list(labels))
    res["ece"] = round(_ece(probs.tolist(), [labels.index(g) for g in gold]), 3)
    res["n"] = len(rows)
    return res


def evaluate_real(enc, heads, root: Path, device) -> dict:
    from engine.read.laya_tags import _auc  # same rank definition
    from engine.eval.real_pairs import load_meetings
    from engine.eval.v4_data import texts_of

    gold = json.loads((root / "data/gold/summre_change_labels.json").read_text())
    meetings = load_meetings(root)
    labels = TASKS["status"]["labels"]
    scores, ys = [], []
    for mid, doc in gold["meetings"].items():
        if mid not in meetings:
            continue
        texts = texts_of(meetings, mid)
        ys_ = doc["labels"][:len(texts)]
        texts = texts[:len(ys_)]
        probs = _probs(enc, heads["status"], texts, device)
        idx = [labels.index(s) for s in STATE_BEARING]
        scores += list(probs[:, idx].sum(1))
        ys += list(ys_)
    return {"n": len(ys), "n_changes": int(sum(ys)),
            "state_auc": round(_auc(scores, ys), 3)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=("v5", "laya"), default="v5")
    ap.add_argument("--pooling", choices=("last", "mean"), default="last")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cap", type=int, default=1200)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-enc", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/v4/tag_probe.json")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    data_dir = root / "data/synth/tags"
    data = {s: [json.loads(l) for l in
                (data_dir / f"{s}.jsonl").read_text().splitlines() if l.strip()]
            for s in ("train", "val", "test")}

    enc = (V5Encoder(args.device, pooling=args.pooling)
           if args.encoder == "v5" else LayaEncoder(args.device))
    torch.manual_seed(13)

    if args.frozen:
        for p in enc.base.parameters():
            p.requires_grad_(False)
    else:
        for p in enc.base.parameters():
            p.requires_grad_(False)
        layers = getattr(enc.base, "layers", None)
        if layers is None:                     # Laya: ModernBertModel
            layers = enc.base.layers
        for layer in list(layers)[-4:]:
            for p in layer.parameters():
                p.requires_grad_(True)
        for name in ("norm", "final_norm"):
            mod = getattr(enc.base, name, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad_(True)

    heads = {t: Head(enc.dim, len(TASKS[t]["labels"])).to(args.device)
             for t in TASKS}
    enc_params = [p for p in enc.base.parameters() if p.requires_grad]
    groups = [{"params": [p for h in heads.values() for p in h.parameters()],
               "lr": args.lr_head, "weight_decay": 0.01}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_enc,
                       "weight_decay": 0.01})
    opt = torch.optim.AdamW(groups)
    views = _views(data, args.cap)
    history = []
    for ep in range(args.epochs):
        enc.base.train()
        order, pos, run, n = [], {t: 0 for t in TASKS}, 0.0, 0
        for t in TASKS:
            order += [t] * ((len(views[t]) + args.batch - 1) // args.batch)
        import random
        random.Random(13 + ep).shuffle(order)
        for task in order:
            chunk = views[task][pos[task]:pos[task] + args.batch]
            pos[task] += args.batch
            if not chunk:
                continue
            run += _train_step(enc, heads, chunk, task, opt, args.device,
                               args.max_len)
            n += 1
        history.append({"epoch": ep + 1, "loss": round(run / max(1, n), 4)})
        print(json.dumps(history[-1]), flush=True)

    enc.base.eval()
    result = {
        "encoder": enc.name, "frozen": bool(args.frozen),
        "pooling": getattr(enc, "pooling", "mean"),
        "epochs": args.epochs, "history": history,
        "trainable_encoder_params": sum(p.numel() for p in enc_params),
        "test": {t: evaluate(enc, heads, data["test"], t, args.device)
                 for t in TASKS},
        "real": evaluate_real(enc, heads, root, args.device),
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
