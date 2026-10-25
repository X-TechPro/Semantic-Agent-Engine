"""Fine-tune the cross-encoder on REAL topic pairs — the smoothness experiment.

The user's theory (2026-10-23): feed a model a PAIR of sentences and let it
predict "smoothness" — do these belong together. That model already exists
and is called a cross-encoder; the shipped reranker (bge-reranker-v2-m3,
XLM-RoBERTa-large + 1-logit head) is one, and it is the only lever that ever
moved the headline (D-015/D-041). This tool asks the untested question: can a
LoRA tune on OUR drawer gold teach it topic identity, where the dot product
could not (paircos AUC 0.6329, D-040)?

Target: same-topic relatedness, not the 92 relation pairs. The drawer gold
defines "belong together" directly — positives are same-drawer pairs (capped
per drawer), negatives are cross-drawer pairs WITHIN the same meeting (the
hard case: same domain, same speakers, different topic). That is thousands of
contrasts from the five TRAIN meetings, which is why this variant is not the
25-positive wall of D-039. Both orders of every pair are trained: the score
should be symmetric — direction is the guards' job, not the model's.

Recipe follows the official encoder-only reranker fine-tune (LoRA r16-32,
alpha 2x, attention projections, gradient checkpointing, ~1e-4, few epochs,
merged save) — with THIS model's module names: XLM-RoBERTa attention is
query/key/value/out_proj (measured against the live module list; the
q_proj/k_proj set in FlagEmbedding's LoRA example targets the GEMMA-based
DECODER rerankers and matched nothing here). fp32 master weights with bf16
autocast — LoRA masters in fp16 underflow at this LR. Implemented on the
already-installed peft — NOT pip-installing FlagEmbedding (venv
torch-shadowing trap) — and on the mechanics proven by engine/embed/
finetune.py (own peft reference for the merge; weights_moved proof; split
discipline). The classifier head is unfrozen alongside the LoRA (a frozen
head cannot learn a new task; peft freezes it by default).

Bars, declared before any fit (this docstring is the declaration):
  CONTROL   the BASE model's zero-shot AUC on the SEL pairs is computed first
            and recorded — every claim below is against it.
  G1 (sel)  LoRA AUC >= control + 0.10 AND >= 0.80. Fails -> no model is
            saved, the holdout is never read, the artefact says refused.
  G2 (holdout, single read, only if G1 clears)  same margin against the base
            on the three sealed meetings' pairs. Clears -> the merged model
            is saved to models/reranker_ft/ UNWIRED; wiring it into the
            linker (thread building -> subject keys -> real edges, G3) is a
            separate declared round.
Nothing here changes any shipped default: rerank.backend keeps pointing at
the base checkpoint, and the base model files are opened read-only.

Run:  PYTHONPATH=src python -u -m engine.tools.finetune_reranker [--epochs 3]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

import numpy as np

from engine.config import repo_root
from engine.tools.summre_paircos import auc
from engine.tools.summre_threads import SPLITS, _drawer_keys, _load

BASE = "BAAI/bge-reranker-v2-m3"
LORA_R = 16
LORA_ALPHA = 32
# XLM-RoBERTa names, verified against the live module list (measured
# 2026-10-23): attention projections are query/key/value/out_proj — the
# q_proj/k_proj set in FlagEmbedding's LoRA example is for the GEMMA-based
# DECODER rerankers, and peft matched nothing with it here (first run died
# with NoMatchingPeftModuleError). The regex is anchored inside
# attention.self so the classifier head's own out_proj is NOT LoRA-wrapped
# (a bare "out_proj" suffix would match it too); the head trains unfrozen
# instead, which is the encoder-only recipe.
TARGETS = r".*attention\.self\.(query|key|value|out_proj)$"
MAX_LEN = 256
CAP_PER_DRAWER = 60          # positive PAIRS sampled per drawer (before ×2 orders)
NEG_PER_POS = 1.0            # cross-drawer negative pairs per positive pair
BATCH = 8
ACCUM = 8
LR = 1e-4


def _text(seg: dict) -> str:
    return f"{seg.get('speaker', '?')}: {seg['text'].strip()}"


def build_rows(root: Path, meeting: str, rng: random.Random,
               cap: int = CAP_PER_DRAWER) -> tuple[list, list]:
    """(positives, negatives) as (textA, textB) pairs, both orders included.

    Positives: same-drawer pairs (the drawer IS the belonging-together gold),
    capped per drawer so one huge thread cannot dominate. Negatives:
    cross-drawer pairs inside the SAME meeting — same domain, same room,
    different topic. Easy cross-meeting negatives are deliberately not used:
    they would inflate the AUC with a skill the task never needs.
    """
    segs = _load(root)[meeting]
    keys = _drawer_keys(root, meeting)
    groups: dict[str, list[int]] = {}
    for i, k in sorted(keys.items()):
        if i < len(segs):
            groups.setdefault(k, []).append(i)
    pos_idx: list[tuple[int, int]] = []
    for k, idxs in sorted(groups.items()):
        if len(idxs) < 2:
            continue
        allp = [(i, j) for a, i in enumerate(idxs) for j in idxs[a + 1:]]
        pos_idx.extend(allp if len(allp) <= cap else rng.sample(allp, cap))
    n_neg = int(round(len(pos_idx) * NEG_PER_POS))
    cross = [(i, j) for i in sorted(keys) for j in sorted(keys)
             if i < j and keys[i] != keys[j]
             and i < len(segs) and j < len(segs)]
    neg_idx = rng.sample(cross, min(n_neg, len(cross))) if cross else []

    def expand(pairs):
        out = []
        for i, j in pairs:
            out.append((_text(segs[i]), _text(segs[j])))
            out.append((_text(segs[j]), _text(segs[i])))
        return out
    return expand(pos_idx), expand(neg_idx)


def split_rows(root: Path, split: str, seed: int) -> tuple[list, list]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r} — declared splits only")
    # a str-seeded Random is stable across processes; hash() is salted per run
    rng = random.Random(f"{seed}-{split}")
    pos, neg = [], []
    for mid in sorted(SPLITS[split]):
        p, n = build_rows(root, mid, rng)
        pos += p
        neg += n
    return pos, neg


def _load_fp16(device: str):
    import torch
    from transformers import AutoModelForSequenceClassification

    m = AutoModelForSequenceClassification.from_pretrained(BASE)
    return m.to(device=device, dtype=torch.float16)


def _load_fp32(device: str):
    import torch
    from transformers import AutoModelForSequenceClassification

    # the TRAINED copy stays fp32: LoRA master weights in fp16 underflow at
    # lr 1e-4 (the bf16-autocast + fp32-master combo is the standard safe
    # recipe; the eval-only base copy can be fp16)
    return AutoModelForSequenceClassification.from_pretrained(BASE).to(device)


def score_pairs(model, tok, pairs: list[tuple[str, str]], device: str,
                batch: int = 32) -> np.ndarray:
    import torch

    out = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(pairs), batch):
            chunk = pairs[s:s + batch]
            enc = tok([a for a, _ in chunk], [b for _, b in chunk],
                      padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits[:, 0].float().cpu().numpy()
            out.append(logits)
    return np.concatenate(out)


def weights_moved(base_model, merged_model, tok, device: str) -> dict:
    """The tune must actually change the merged model's outputs — a LoRA that
    never reached the forward path still prints a falling loss curve (guard
    proven necessary by the embedder tune)."""
    probe = [("Le buffet est confirmé.", "Donc on paie le traiteur ?"),
             ("Le budget est validé à 1,5 M€.", "La scène sera démontée lundi."),
             ("On part sur mi-juillet.", "Le lieu reste le même.")] * 8
    a = score_pairs(base_model, tok, probe, device)
    b = score_pairs(merged_model, tok, probe, device)
    mad = float(np.abs(a - b).mean())
    return {"changed": mad > 0.01, "mean_abs_logit_diff": round(mad, 4),
            "n_probed": len(probe)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    root = repo_root()
    run_dir = root / "runs" / "reranker_ft"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = root / "models" / "reranker_ft"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists — this round must not "
                         "overwrite a previous checkpoint; move it first")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer

    # ---- data (TRAIN only; the sealed holdout is not even built yet) -------
    tr_pos, tr_neg = split_rows(root, "train", args.seed)
    se_pos, se_neg = split_rows(root, "sel", args.seed)
    assert not (set(SPLITS["train"]) & set(SPLITS["sel"])
                | set(SPLITS["train"]) & set(SPLITS["holdout"])
                | set(SPLITS["sel"]) & set(SPLITS["holdout"]))
    print(f"train rows: {len(tr_pos)} pos / {len(tr_neg)} neg "
          f"({len(SPLITS['train'])} meetings) | sel: {len(se_pos)}/{len(se_neg)}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    base_model = _load_fp16(args.device)

    # ---- CONTROL first: the base model's zero-shot skill on sel ------------
    control_auc = auc(score_pairs(base_model, tok, se_pos, args.device),
                      score_pairs(base_model, tok, se_neg, args.device))
    print(f"CONTROL base zero-shot sel AUC: {control_auc}", flush=True)

    # ---- fit: frozen base + LoRA(qkvo) + unfrozen classifier head ----------
    model = _load_fp32(args.device)
    model.gradient_checkpointing_enable()
    cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
                     bias="none", task_type=TaskType.SEQ_CLS,
                     target_modules=TARGETS)
    peft_model = get_peft_model(model, cfg)
    peft_model.enable_input_require_grads()   # required with checkpointing
    for n, p in peft_model.named_parameters():
        if "score" in n or "classifier" in n:
            p.requires_grad_(True)            # peft froze the head; it must learn
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    if not trainable:
        raise SystemExit("LoRA attached but nothing is trainable")
    print(f"trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    peft_model.to(args.device)
    peft_model.train()

    all_rows = tr_pos + tr_neg
    labels = torch.tensor([1.0] * len(tr_pos) + [0.0] * len(tr_neg))
    enc = tok([a for a, _ in all_rows], [b for _, b in all_rows],
              padding=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt")
    dl = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], labels),
                    batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(trainable, lr=LR)
    bce = torch.nn.BCEWithLogitsLoss()
    history: list[float] = []
    for ep in range(1, args.epochs + 1):
        tot, seen = 0.0, 0
        opt.zero_grad()
        for k, (ids, am, y) in enumerate(dl):
            ids = ids.to(args.device)
            am = am.to(args.device)
            y = y.to(args.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = peft_model(input_ids=ids, attention_mask=am).logits[:, 0]
                loss = bce(logits.float(), y) / ACCUM
            loss.backward()
            tot += float(loss) * ACCUM
            seen += 1
            if (k + 1) % ACCUM == 0 or k + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
        history.append(round(tot / max(1, seen), 4))
        print(f"epoch {ep}/{args.epochs} loss {history[-1]:.4f}", flush=True)

    merged = peft_model.merge_and_unload()
    merged.to(args.device, dtype=torch.float16)   # eval/save in fp16 (base size)
    moved = weights_moved(base_model, merged, tok, args.device)
    if not moved["changed"]:
        raise SystemExit(f"merged model identical to base ({moved}) — trained nothing")

    # ---- G1 on sel ---------------------------------------------------------
    tp = score_pairs(merged, tok, se_pos, args.device)
    tn = score_pairs(merged, tok, se_neg, args.device)
    tuned_auc = auc(tp, tn)
    g1 = tuned_auc >= control_auc + 0.10 and tuned_auc >= 0.80
    out: dict = {
        "date": date.today().isoformat(), "seed": args.seed, "base": BASE,
        "recipe": {"lora_r": LORA_R, "alpha": LORA_ALPHA, "targets": TARGETS,
                   "epochs": args.epochs, "lr": LR, "batch": BATCH,
                   "accum": ACCUM, "max_len": MAX_LEN, "head_trained": True,
                   "master_weights": "fp32", "autocast": "bf16",
                   "grad_checkpointing": True},
        "data": {"train_meetings": sorted(SPLITS["train"]),
                 "sel_meeting": sorted(SPLITS["sel"]),
                 "train_pos_pairs": len(tr_pos), "train_neg_pairs": len(tr_neg),
                 "sel_pos_pairs": len(se_pos), "sel_neg_pairs": len(se_neg),
                 "target": "same-drawer relatedness (drawer gold), both orders; "
                           "negatives cross-drawer WITHIN the meeting"},
        "loss_history": history, "weights_moved": moved,
        "sel": {"control_base_auc": round(control_auc, 4),
                "tuned_auc": round(tuned_auc, 4),
                "pos_mean_logit": round(float(tp.mean()), 3),
                "neg_mean_logit": round(float(tn.mean()), 3)},
        "bars": {"G1_control_plus_010_and_080": g1},
    }
    stamp = out["date"]
    if not g1:
        out["verdict"] = ("REFUSED at G1 — the tune did not clear control+0.10 "
                          "on sel; no model saved, the sealed holdout was "
                          "never read")
        p = run_dir / f"result-{stamp}-refused.json"
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
        print("wrote", p)
        return 0

    # ---- G2: the single sealed read ----------------------------------------
    ho_pos, ho_neg = split_rows(root, "holdout", args.seed)
    ctrl_ho = auc(score_pairs(base_model, tok, ho_pos, args.device),
                  score_pairs(base_model, tok, ho_neg, args.device))
    tuned_ho = auc(score_pairs(merged, tok, ho_pos, args.device),
                   score_pairs(merged, tok, ho_neg, args.device))
    g2 = tuned_ho >= ctrl_ho + 0.10 and tuned_ho >= 0.80
    out["holdout"] = {"read": "once", "n_pos_pairs": len(ho_pos),
                      "control_base_auc": round(ctrl_ho, 4),
                      "tuned_auc": round(tuned_ho, 4)}
    out["bars"]["G2_holdout_same_margin"] = g2
    if g2:
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        (out_dir / "sae_provenance.json").write_text(json.dumps({
            "base": BASE, "trained_on": "drawer-relatedness pairs, train split",
            "seed": args.seed, "date": stamp,
            "sel_auc": out["sel"]["tuned_auc"],
            "holdout_auc": round(tuned_ho, 4), "wired": False,
            "note": "saved UNWIRED: rerank.backend still loads the base; "
                    "G3 (thread building -> subject keys -> linker edges) is a "
                    "separate declared round"}, indent=1), encoding="utf-8")
        out["verdict"] = ("CLEARED G1+G2 — merged model saved to "
                          "models/reranker_ft/ UNWIRED; G3 (does it replace "
                          "gold subject keys end-to-end) is the next decision")
    else:
        out["verdict"] = ("REFUSED at G2 — sel gain did not transfer to the "
                          "sealed holdout; no model saved")
    p = run_dir / f"result-{stamp}-{'pass' if g2 else 'refused'}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("sel", "holdout", "bars", "verdict")},
                     indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
