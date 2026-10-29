"""Fine-tune Laya on the piscine tags, and score the result.

Laya is not a classifier with a fixed label set: the sequence is
`[CLS] <question> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`, each
option is scored at its own marker, and a softmax runs over the options for
that question. The answer space is therefore defined at request time.

We keep that machinery and fine-tune it with ordinary cross-entropy over the
options of one task (this is a supervised tagging problem, not the calibrated
RL objective the shipped checkpoint was trained with). Three tasks share the
encoder and the 2-layer head, told apart by the `type_emb` slot:

    kind   8 options   piscine §4 unit types
    status 5 options   proposed / confirmed / cancelled / completed / unknown
    intent 8 options   piscine §5 request intents + none

The checkpoint fine-tuned here is `laya-multilingual` (mmBERT-base, 256k
vocab): the root checkpoint is English-only, and the corpus is French ASR.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

# TensorFlow's abseil runtime can deadlock model construction when TF is
# installed; Laya only needs torch.
os.environ.setdefault("USE_TF", "0")

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from engine.schemas_v4 import INTENTS, KINDS, STATUSES  # noqa: E402

TASKS: dict[str, dict] = {
    "kind": {
        "qtype": 0,
        "ins": "Quel type d'unité sémantique est cette phrase ?",
        "options": {
            "decision": "une décision prise ou à prendre",
            "action": "une action que quelqu'un fait ou va faire",
            "commitment": "un engagement avec une promesse et une échéance",
            "question": "une question posée",
            "answer": "une réponse à une question",
            "fact": "un fait constaté, sans décision",
            "risk": "un risque ou une conséquence négative possible",
            "idea": "une idée ou une piste à explorer",
        },
        "labels": KINDS,
    },
    "status": {
        "qtype": 1,
        "ins": "Quel est le statut de ce qui est dit ?",
        "options": {
            "proposed": "proposé, pas encore validé",
            "confirmed": "confirmé, validé, acté",
            "cancelled": "annulé, abandonné, annulé",
            "completed": "déjà fait, terminé",
            "unknown": "aucun statut, non précisé",
        },
        "labels": ("proposed", "confirmed", "cancelled", "completed", "unknown"),
    },
    "intent": {
        "qtype": 2,
        "ins": "Quelle opération le locuteur demande-t-il à l'assistant ?",
        "options": {
            "answer": "répondre à une question sur ce qui est décidé",
            "compare": "comparer deux versions ou deux réunions",
            "dossier": "construire un dossier ou une synthèse",
            "email": "rédiger un email ou un message",
            "task": "créer une tâche",
            "schedule": "planifier un événement ou une réunion",
            "clarify": "demander une clarification, car c'est ambigu",
            "none": "aucune demande, ce n'est pas une instruction",
        },
        "labels": INTENTS,
    },
}

STATE_BEARING = {"proposed", "confirmed", "cancelled", "completed"}


def checkpoint_dir(subfolder: str = "multilingual") -> Path:
    env = os.environ.get("LAYA_DIR")
    if env:
        base = Path(env)
    else:
        hits = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--convaiinnovations--laya/snapshots/*/"))
        if not hits:
            raise FileNotFoundError(
                "Laya checkpoint not found; set LAYA_DIR or download "
                "convaiinnovations/laya")
        base = Path(sorted(hits)[-1])
    return base / subfolder if subfolder else base


def load_laya(subfolder: str = "multilingual"):
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from laya.common import build_model

    root = checkpoint_dir(subfolder)
    cfg = json.loads((root / "rl_agent_config.json").read_text())
    model = build_model(cfg, encoder_dir=str(root / "encoder"))
    state = load_file(str(root / "model.safetensors"))
    model.load_state_dict(state, strict=True)
    tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    return model, tok, cfg, root


def _item(row: dict, task: str, tok, max_len: int, head_max_len: int) -> dict:
    spec = TASKS[task]
    label = row[task]
    if label not in spec["labels"]:
        raise ValueError(f"{task}={label!r} not in {spec['labels']}")
    idx = spec["labels"].index(label)
    if "state_ids" not in row:
        row["state_ids"] = tok(row["text"], add_special_tokens=False).input_ids
    target = [0.0] * len(spec["labels"])
    target[idx] = 1.0
    q = {"t": "choice", "ins": spec["ins"], "crit": spec["options"]}
    from laya.train import encode_item
    return encode_item(tok, {"q": q, "state_ids": row["state_ids"],
                             "qtype": spec["qtype"], "target": target,
                             "label": idx}, max_len, head_max_len)


def _batch(rows: list[dict], task: str, tok, max_len: int, head_max_len: int,
           device):
    # encode_item drops "label" (only ids/markers/qtype/target survive), so the
    # class index is rebuilt here from the row rather than read from the batch.
    from laya.common import collate_items
    enc = [_item(r, task, tok, max_len, head_max_len) for r in rows]
    b = collate_items([enc], tok.pad_token_id)
    inputs = {k: v.to(device) for k, v in b.items()
              if k in ("input_ids", "attention_mask", "marker_pos",
                       "marker_mask", "qtype")}
    labels = torch.tensor(
        [TASKS[task]["labels"].index(r[task]) for r in rows],
        device=device)
    return inputs, labels


def _balance(rows: list[dict], task: str, cap: int, seed: int) -> list[dict]:
    import random
    rng = random.Random(seed)
    by: dict[str, list[dict]] = {}
    for r in rows:
        if r.get(task) and task in r.get("tasks", ()):
            by.setdefault(r[task], []).append(r)
    if not by:
        return []
    n = min(cap, max(len(v) for v in by.values()))
    out = []
    for label, items in by.items():
        rng.shuffle(items)
        out.extend(items[:n])
    rng.shuffle(out)
    return out


def train(model, tok, data: dict[str, list[dict]], *, device="cuda",
          epochs: int = 3, batch: int = 32, lr_enc: float = 3e-5,
          lr_head: float = 1e-3, max_len: int = 192, head_max_len: int = 128,
          cap: int = 1200, seed: int = 13, log=print):
    torch.manual_seed(seed)
    model.to(device)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.1
    model.encoder.embeddings.tok_embeddings.weight.requires_grad_(False)

    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")]
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_enc, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
    ])
    views = {t: _balance(data["train"], t, cap, seed) for t in TASKS}
    steps = max(1, sum(math.ceil(len(v) / batch) for v in views.values()))
    total = steps * epochs
    warm = max(1, int(0.06 * total))

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    history = []
    step = 0
    for ep in range(epochs):
        model.train()
        order = [t for t in TASKS for _ in range(math.ceil(len(views[t]) / batch))]
        import random
        random.Random(seed + ep).shuffle(order)
        pos = {t: 0 for t in TASKS}
        run = 0.0
        for task in order:
            view = views[task]
            start = pos[task]
            chunk = view[start:start + batch]
            pos[task] = start + batch
            if not chunk:
                continue
            inputs, labels = _batch(chunk, task, tok, max_len, head_max_len,
                                    device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=device == "cuda"):
                logits, _ = model(**inputs)
            k = len(TASKS[task]["labels"])
            loss = F.cross_entropy(logits[:, :k].float(), labels,
                                   label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            step += 1
            run += float(loss)
        history.append({"epoch": ep + 1, "loss": run / max(1, len(order))})
        log(f"epoch {ep + 1}/{epochs} loss {history[-1]['loss']:.4f}")
    return history


@torch.no_grad()
def predict(model, tok, texts: list[str], task: str, *, device="cuda",
            batch: int = 64, max_len: int = 192, head_max_len: int = 128):
    model.eval()
    out = []
    k = len(TASKS[task]["labels"])
    for s in range(0, len(texts), batch):
        rows = [{"text": t, task: TASKS[task]["labels"][0]} for t in texts[s:s + batch]]
        enc = [_item(r, task, tok, max_len, head_max_len) for r in rows]
        from laya.common import collate_items
        b = collate_items([enc], tok.pad_token_id)
        inputs = {kk: v.to(device) for kk, v in b.items()
                  if kk in ("input_ids", "attention_mask", "marker_pos",
                            "marker_mask", "qtype")}
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=device == "cuda"):
            logits, _ = model(**inputs)
        out.extend(torch.softmax(logits[:, :k].float(), -1).cpu().tolist())
    return out


def _prf(gold: list[str], pred: list[str], labels: tuple[str, ...]) -> dict:
    per = {}
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[lab] = {"p": round(prec, 3), "r": round(rec, 3),
                    "f1": round(f1, 3), "n": tp + fn}
    macro = sum(v["f1"] for v in per.values()) / len(per)
    acc = sum(g == p for g, p in zip(gold, pred)) / max(1, len(gold))
    return {"macro_f1": round(macro, 3), "accuracy": round(acc, 3),
            "per_class": per}


def _ece(probs: list[list[float]], gold_idx: list[int], bins: int = 10) -> float:
    conf = [max(p) for p in probs]
    hit = [int(max(range(len(p)), key=p.__getitem__) == g)
           for p, g in zip(probs, gold_idx)]
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        mc = sum(conf[i] for i in idx) / len(idx)
        ma = sum(hit[i] for i in idx) / len(idx)
        total += len(idx) / len(conf) * abs(mc - ma)
    return total


def evaluate(model, tok, rows: list[dict], task: str, *, device="cuda") -> dict:
    labels = TASKS[task]["labels"]
    rows = [r for r in rows if r.get(task) in labels
            and task in r.get("tasks", ())]
    texts = [r["text"] for r in rows]
    gold = [r[task] for r in rows]
    probs = predict(model, tok, texts, task, device=device)
    pred = [labels[max(range(len(p)), key=p.__getitem__)] for p in probs]
    gold_idx = [labels.index(g) for g in gold]
    res = _prf(gold, pred, labels)
    res["ece"] = round(_ece(probs, gold_idx), 3)
    res["n"] = len(rows)
    by_family: dict[str, list[bool]] = {}
    for r, p in zip(rows, pred):
        by_family.setdefault(r.get("family", "?"), []).append(p == r[task])
    res["family_accuracy"] = {
        f: round(sum(v) / len(v), 3) for f, v in sorted(by_family.items())}
    return res


def evaluate_pairs(model, tok, pairs: list[dict], *, device="cuda") -> dict:
    out = {}
    for field, task in (("kind", "kind"), ("status", "status")):
        subset = [p for p in pairs if p["field"] == field]
        if not subset:
            continue
        texts, gold = [], []
        for p in subset:
            texts += [p["a"], p["b"]]
            gold += [p["la"], p["lb"]]
        labels = TASKS[task]["labels"]
        probs = predict(model, tok, texts, task, device=device)
        pred = [labels[max(range(len(x)), key=x.__getitem__)] for x in probs]
        both = sum(pred[i] == gold[i] and pred[i + 1] == gold[i + 1]
                   for i in range(0, len(gold), 2))
        by_group: dict[str, list[bool]] = {}
        for gi, p in enumerate(subset):
            by_group.setdefault(p["group"], []).append(
                pred[2 * gi] == gold[2 * gi] and pred[2 * gi + 1] == gold[2 * gi + 1])
        out[field] = {
            "n_pairs": len(subset),
            "both_correct": round(both / len(subset), 3),
            "per_group": {g: round(sum(v) / len(v), 3)
                          for g, v in sorted(by_group.items())},
        }
    return out


def _auc(scores: list[float], labels: list[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [r for r, y in zip(ranks, labels) if y == 1]
    neg = [r for r, y in zip(ranks, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    return (sum(pos) - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def evaluate_real(model, tok, root: Path, *, device="cuda") -> dict:
    """External check: the 1,475 real turns labelled change / no-change.

    The synthetic status is not the annotation's definition, so this is a
    transfer probe, not an in-domain score: does "carries a state" line up with
    "the annotator marked a change"?
    """
    from engine.eval.real_pairs import load_meetings
    from engine.eval.v4_data import texts_of

    gold = json.loads((root / "data/gold/summre_change_labels.json").read_text())
    meetings = load_meetings(root)
    scores, labels, kind_scores = [], [], []
    for mid, doc in gold["meetings"].items():
        if mid not in meetings:
            continue
        texts = texts_of(meetings, mid)
        ys = doc["labels"][:len(texts)]
        if len(ys) < len(texts):
            texts = texts[:len(ys)]
        probs = predict(model, tok, texts, "status", device=device,
                        batch=64)
        labels_idx = TASKS["status"]["labels"]
        state = [sum(p[labels_idx.index(s)] for s in STATE_BEARING) for p in probs]
        kprobs = predict(model, tok, texts, "kind", device=device, batch=64)
        k_idx = TASKS["kind"]["labels"]
        kstate = [sum(p[k_idx.index(s)] for s in ("decision", "action", "commitment"))
                  for p in kprobs]
        scores += state
        kind_scores += kstate
        labels += list(ys)
    return {"n_turns": len(labels), "n_changes": int(sum(labels)),
            "status_auc": round(_auc(scores, labels), 3),
            "kind_auc": round(_auc(kind_scores, labels), 3)}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synth/tags")
    ap.add_argument("--out", default="runs/laya_tags/multilingual")
    ap.add_argument("--subfolder", default="multilingual")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr-enc", type=float, default=3e-5)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--head-max-len", type=int, default=128)
    ap.add_argument("--cap", type=int, default=1200)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args(argv)

    root = Path(__file__).resolve().parents[3]
    data_dir = root / args.data
    split = {s: _read_jsonl(data_dir / f"{s}.jsonl")
             for s in ("train", "val", "test")}
    pairs = _read_jsonl(data_dir / "pairs.jsonl")

    model, tok, cfg, ckpt = load_laya(args.subfolder)
    model.to(args.device)
    history = []
    if not args.skip_train:
        history = train(model, tok, split, device=args.device,
                        epochs=args.epochs, batch=args.batch,
                        lr_enc=args.lr_enc, lr_head=args.lr_head,
                        max_len=args.max_len, head_max_len=args.head_max_len,
                        cap=args.cap)

    metrics = {
        "checkpoint": str(ckpt), "subfolder": args.subfolder,
        "epochs": args.epochs, "history": history,
        "val": {t: evaluate(model, tok, split["val"], t, device=args.device)
                for t in TASKS},
        "test": {t: evaluate(model, tok, split["test"], t, device=args.device)
                 for t in TASKS},
        "pairs": evaluate_pairs(model, tok, pairs, device=args.device),
        "real": evaluate_real(model, tok, root, device=args.device),
    }
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              str(out / "model.safetensors"))
    (out / "meta.json").write_text(json.dumps({
        "source_checkpoint": str(ckpt), "subfolder": args.subfolder,
        "tasks": {t: list(TASKS[t]["labels"]) for t in TASKS},
        "max_len": args.max_len, "head_max_len": args.head_max_len,
    }, indent=1), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1),
                                      encoding="utf-8")
    print(json.dumps(metrics, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
