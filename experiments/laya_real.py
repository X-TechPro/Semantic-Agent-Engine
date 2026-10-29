"""Fine-tune Laya on REAL meeting turns, leave one meeting out.

The shipped tagger learned from 46 generated sentence patterns and has never
seen a transcript. This trains the same checkpoint and the same option-marker
head on the 1,475 real turns and the 187 annotated changes, six times: each
meeting is scored by a model that never saw it.

Task: "does this turn change something" as a two-option question, using the
intent slot of `type_emb` so the kind/status heads are left untouched.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings
from engine.eval.v4_data import texts_of
from engine.read import laya_tags

OPTIONS = {"changed": "un changement : une valeur, une date ou une décision change",
           "same": "pas de changement : rien de nouveau n'est décidé"}
INS = "Est-ce que cette phrase apporte un changement par rapport à ce qui était dit avant ?"
QTYPE = 2


def _prf(y, pred):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & ~(y == 1)).sum())
    fn = int(((~pred) & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3),
            "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0}


def _auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    pos = ranks[y == 1]
    npos, nneg = len(pos), len(y) - len(pos)
    if not npos or not nneg:
        return 0.5
    return float((pos.sum() - npos * (npos + 1) / 2) / (npos * nneg))


def _item(text, label, tok, max_len, head_max_len):
    from laya.train import encode_item

    target = [0.0, 0.0]
    target[label] = 1.0
    return encode_item(tok, {
        "q": {"t": "choice", "ins": INS, "crit": OPTIONS},
        "state_ids": tok(text, add_special_tokens=False).input_ids,
        "qtype": QTYPE, "target": target, "label": label,
    }, max_len, head_max_len)


def _batch(rows, tok, max_len, head_max_len, device):
    from laya.common import collate_items

    enc = [_item(t, y, tok, max_len, head_max_len) for t, y in rows]
    b = collate_items([enc], tok.pad_token_id)
    inputs = {k: v.to(device) for k, v in b.items()
              if k in ("input_ids", "attention_mask", "marker_pos",
                       "marker_mask", "qtype")}
    labels = torch.tensor([y for _, y in rows], device=device)
    return inputs, labels


def _fit(train_rows, tok, base_state, cfg_root, device, epochs, batch,
         max_len, head_max_len, seed=13):
    from laya.common import build_model

    torch.manual_seed(seed)
    model = build_model({"head_layers": 2, "act_costs": {"escalate": 0.5}},
                        encoder_dir=str(Path(cfg_root) / "encoder"))
    model.load_state_dict(base_state, strict=True)
    model.to(device)
    for p in model.encoder.embeddings.tok_embeddings.parameters():
        p.requires_grad_(False)
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not n.startswith("encoder.")],
         "lr": 1e-3},
        {"params": [p for p in model.encoder.parameters() if p.requires_grad],
         "lr": 2e-5},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    rng = np.random.default_rng(seed)
    model.train()
    for _ep in range(epochs):
        order = rng.permutation(len(train_rows))
        for s in range(0, len(order), batch):
            chunk = [train_rows[i] for i in order[s:s + batch]]
            inputs, labels = _batch(chunk, tok, max_len, head_max_len, device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits, _ = model(**inputs)
            loss = F.cross_entropy(logits[:, :2].float(), labels,
                                   label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _probs(model, tok, texts, device, max_len, head_max_len, batch=32):
    out = []
    for s in range(0, len(texts), batch):
        enc = [_item(t, 0, tok, max_len, head_max_len)
               for t in texts[s:s + batch]]
        from laya.common import collate_items
        b = collate_items([enc], tok.pad_token_id)
        inputs = {k: v.to(device) for k, v in b.items()
                  if k in ("input_ids", "attention_mask", "marker_pos",
                           "marker_mask", "qtype")}
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            logits, _ = model(**inputs)
        out.extend(torch.softmax(logits[:, :2].float(), -1)[:, 1].cpu().tolist())
    return np.array(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v4/laya_real.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--head-max-len", type=int, default=128)
    args = ap.parse_args(argv)

    root = repo_root()
    meetings = load_meetings(root)
    gold = json.loads((root / "data/gold/summre_change_labels.json").read_text())

    model, tok, cfg, ckpt = laya_tags.load_laya("multilingual")
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    del model
    gc.collect()

    data: dict[str, tuple[list[str], np.ndarray]] = {}
    for mid, doc in gold["meetings"].items():
        if mid not in meetings:
            continue
        texts = texts_of(meetings, mid)
        ys = np.array(doc["labels"][:len(texts)])
        keep = [i for i in range(len(ys)) if texts[i].strip()]
        data[mid] = ([texts[i] for i in keep], ys[keep])

    folds = {}
    pooled_y, pooled_s, pooled_pred = [], [], []
    for mid in sorted(data):
        rows = [(t, int(y)) for m, (ts, ys) in data.items() if m != mid
                for t, y in zip(ts, ys)]
        # balance the training side, changes are ~1 in 8
        pos = [r for r in rows if r[1] == 1]
        neg = [r for r in rows if r[1] == 0]
        n = min(len(pos), len(neg))
        rng = np.random.default_rng(13)
        idx = rng.choice(len(neg), size=n, replace=False)
        rows = pos[:n] + [neg[i] for i in idx]
        model = _fit(rows, tok, base_state, ckpt, args.device, args.epochs,
                     args.batch, args.max_len, args.head_max_len)
        s_tr = _probs(model, tok, [r[0] for r in rows], args.device,
                      args.max_len, args.head_max_len)
        y_tr = np.array([r[1] for r in rows])
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            f1 = _prf(y_tr, s_tr >= t)["f1"]
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        texts, ys = data[mid]
        s_te = _probs(model, tok, texts, args.device, args.max_len,
                      args.head_max_len)
        pred = s_te >= best_t
        folds[mid] = {"threshold": round(best_t, 2), **_prf(ys, pred),
                      "auc": round(_auc(ys, s_te), 3), "n": len(ys),
                      "changes": int(ys.sum())}
        pooled_y.append(ys)
        pooled_s.append(s_te)
        pooled_pred.append(pred)
        print(mid, folds[mid], flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    y = np.concatenate(pooled_y)
    s = np.concatenate(pooled_s)
    pred = np.concatenate(pooled_pred)
    out = {"n": int(len(y)), "changes": int(y.sum()),
           "pooled": _prf(y, pred), "auc": round(_auc(y, s), 3),
           "folds": folds,
           "note": "each meeting predicted by a model trained on the other five"}
    p = root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
