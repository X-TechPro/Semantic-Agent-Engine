"""Graph-conditioned thread assignment — the user's "established tags" idea.

D-042 scored sentence-vs-sentence relatedness (0.6547, refused). The one
configuration never tried is what the user actually described: condition on
the MEMORY GRAPH — score an incoming sentence against each ESTABLISHED THREAD
(its whole drawer history so far), and assign it to the best. The thread is
many sentences, so the weak single-pair signal gets multiple chances per
comparison; the output space is the meeting's live drawers, not 72 global
labels.

Data: the drawer gold (fine keys — the same keys the gold resolver ships to
the demos). Walking each meeting in segment order, a sentence whose drawer is
already established is an ASSIGNMENT case: positive = (sentence, its own
thread), negatives = (sentence, a wrong established thread in the same
meeting). Threads are the drawer's PRECEDING sentences only (cap 6, joined) —
built by construction so no future sentence can leak; tested explicitly.
Seed sentences (drawer not yet established) are out of scope this round:
"open a new topic" is the G3 decision, not this one. Single order — the roles
(query, thread) are asymmetric here, unlike D-042's symmetric pairs.

Bars, declared BEFORE any fit (this docstring is the declaration):
  CONTROL   the BASE model's top-1 assignment accuracy on the sel meeting,
            computed before training — the number every claim is against.
  G1 (sel)  tuned top-1 >= control + 0.10 AND >= 0.55. (Random on ~15-25
            live drawers is ~0.05; the window-attention classifier of D-030
            got 0.19-0.26 agreement — 0.55 means this mechanism actually
            beats its own predecessor, not just chance.) Fails -> nothing is
            saved, the sealed holdout is never read.
  G2 (holdout, single read, only if G1 clears)  same margin on the mean of
            the three sealed meetings' top-1 accuracy. Clears -> merged model
            saved to models/threader_ft/ UNWIRED; G3 (online assignment
            replacing gold keys end-to-end on the reach tape) is a separate
            declared round.
Nothing changes a shipped default; the base model is opened read-only; the
sealed holdout rows are not even built before G1 passes.

Run:  PYTHONPATH=src python -u -m engine.tools.finetune_threader [--epochs 3]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

from engine.config import repo_root
from engine.tools.finetune_reranker import (BASE, LORA_ALPHA, LORA_R, MAX_LEN,
                                            TARGETS, _load_fp32, _text,
                                            score_pairs, weights_moved)
from engine.tools.summre_paircos import auc
from engine.tools.summre_threads import SPLITS, _drawer_keys, _load

THREAD_MAX_SENTS = 6      # thread = the drawer's last N preceding sentences
MAX_NEG_PER_SENT = 4      # wrong threads sampled per training sentence
BATCH = 8
ACCUM = 8
LR = 1e-4


def build_rows(root: Path, meeting: str, rng: random.Random):
    """-> (pos, neg, eval_items, traces) for one meeting, walked in order.

    eval_items: (sent_idx, sent_text, gold_drawer, {drawer: thread_text})
    for every assignment case (>=2 established threads, gold established).
    traces: same rows with the segment indices behind each thread — the
    no-future-leakage proof the unit tests check.
    """
    segs = _load(root)[meeting]
    keys = _drawer_keys(root, meeting)
    by_drawer: dict[str, list[int]] = {}
    pos: list[tuple[str, str]] = []
    neg: list[tuple[str, str]] = []
    items: list[tuple[int, str, str, dict[str, str]]] = []
    traces: list[tuple[int, str, dict[str, list[int]]]] = []

    def thread_text(idxs: list[int]) -> str:
        return " | ".join(_text(segs[j]) for j in idxs[-THREAD_MAX_SENTS:])

    for i in range(len(segs)):
        k = keys.get(i)
        established = {d: idxs for d, idxs in by_drawer.items() if idxs}
        if k is not None and len(established) >= 2 and k in established:
            sent = _text(segs[i])
            own = thread_text(established[k])
            pos.append((sent, own))
            others = [d for d in established if d != k]
            for d in rng.sample(others, min(MAX_NEG_PER_SENT, len(others))):
                neg.append((sent, thread_text(established[d])))
            cands = {d: thread_text(idxs) for d, idxs in established.items()}
            items.append((i, sent, k, cands))
            traces.append((i, k, {d: list(idxs[-THREAD_MAX_SENTS:])
                                  for d, idxs in established.items()}))
        if k is not None:
            by_drawer.setdefault(k, []).append(i)   # registered AFTER scoring
    return pos, neg, items, traces


def split_rows(root: Path, split: str, seed: int):
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r} — declared splits only")
    rng = random.Random(f"{seed}-{split}")
    pos, neg, items = [], [], []
    for mid in sorted(SPLITS[split]):
        p, n, it, _ = build_rows(root, mid, rng)
        pos += p
        neg += n
        items += [(mid, *row) for row in it]
    return pos, neg, items


def assignment(model, tok, items, device: str) -> dict:
    """Top-1 over the established threads; ties count as wrong.

    items: (meeting, sent_idx, sent_text, gold_drawer, {drawer: thread_text})
    """
    flat, owner = [], []
    for gi, (_mid, _i, sent, _gold, cands) in enumerate(items):
        for d, t in sorted(cands.items()):
            flat.append((sent, t))
            owner.append((gi, d))
    scores = score_pairs(model, tok, flat, device)
    best: dict[int, tuple[float, str]] = {}
    for (gi, d), s in zip(owner, scores):
        cur = best.get(gi)
        if cur is None or s > cur[0] + 1e-9:
            best[gi] = (s, d)
        elif abs(s - cur[0]) <= 1e-9:
            best[gi] = (cur[0], "TIE")
    hit = sum(1 for gi, (_mid, _i, _s, g, _c) in enumerate(items)
              if best.get(gi, (0.0, ""))[1] == g)
    return {"n": len(items), "top1": round(hit / len(items), 4) if items else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    root = repo_root()
    run_dir = root / "runs" / "threader_ft"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_dir = root / "models" / "threader_ft"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists — must not overwrite a previous "
                         "checkpoint; move it first")

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoTokenizer

    tr_pos, tr_neg, tr_items = split_rows(root, "train", args.seed)
    se_pos, se_neg, se_items = split_rows(root, "sel", args.seed)
    print(f"train: {len(tr_pos)} pos / {len(tr_neg)} neg rows, "
          f"{len(tr_items)} assignment cases | sel: {len(se_items)} cases",
          flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    base_model = _load_fp32(args.device)

    # CONTROL before training
    control = assignment(base_model, tok, se_items, args.device)
    print(f"CONTROL base sel top-1: {control}", flush=True)

    # ---- fit (same proven mechanics as D-042) ------------------------------
    model = _load_fp32(args.device)
    model.gradient_checkpointing_enable()
    peft_model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS, target_modules=TARGETS))
    peft_model.enable_input_require_grads()
    for n, p in peft_model.named_parameters():
        if "score" in n or "classifier" in n:
            p.requires_grad_(True)
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in trainable):,}", flush=True)
    peft_model.to(args.device)
    peft_model.train()

    rows = tr_pos + tr_neg
    labels = torch.tensor([1.0] * len(tr_pos) + [0.0] * len(tr_neg))
    enc = tok([a for a, _ in rows], [b for _, b in rows], padding=True,
              truncation=True, max_length=MAX_LEN, return_tensors="pt")
    dl = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], labels),
                    batch_size=BATCH, shuffle=True)
    opt = torch.optim.AdamW(trainable, lr=LR)
    bce = torch.nn.BCEWithLogitsLoss()
    history = []
    for ep in range(1, args.epochs + 1):
        tot, seen = 0.0, 0
        opt.zero_grad()
        for kk, (ids, am, y) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = peft_model(input_ids=ids.to(args.device),
                                    attention_mask=am.to(args.device)).logits[:, 0]
                loss = bce(logits.float(), y.to(args.device)) / ACCUM
            loss.backward()
            tot += float(loss) * ACCUM
            seen += 1
            if (kk + 1) % ACCUM == 0 or kk + 1 == len(dl):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
        history.append(round(tot / max(1, seen), 4))
        print(f"epoch {ep}/{args.epochs} loss {history[-1]:.4f}", flush=True)

    merged = peft_model.merge_and_unload().to(args.device)
    moved = weights_moved(base_model, merged, tok, args.device)
    if not moved["changed"]:
        raise SystemExit(f"merged identical to base ({moved}) — trained nothing")

    # ---- G1 on sel ----------------------------------------------------------
    tuned = assignment(merged, tok, se_items, args.device)
    auc_pos = score_pairs(merged, tok, se_pos, args.device)
    auc_neg = score_pairs(merged, tok, se_neg, args.device)
    pair_auc = auc(auc_pos, auc_neg)
    g1 = (tuned["top1"] is not None and control["top1"] is not None
          and tuned["top1"] >= control["top1"] + 0.10 and tuned["top1"] >= 0.55)
    out: dict = {
        "date": date.today().isoformat(), "seed": args.seed, "base": BASE,
        "mechanism": "sentence vs ESTABLISHED THREADS (drawer history, last "
                     f"{THREAD_MAX_SENTS} sentences), top-1 over live drawers; "
                     "ties count wrong",
        "data": {"train_meetings": sorted(SPLITS["train"]),
                 "sel_meeting": sorted(SPLITS["sel"]),
                 "train_pos_rows": len(tr_pos), "train_neg_rows": len(tr_neg),
                 "sel_cases": len(se_items)},
        "recipe": {"lora_r": LORA_R, "alpha": LORA_ALPHA, "epochs": args.epochs,
                   "lr": LR, "targets": TARGETS, "master": "fp32",
                   "autocast": "bf16"},
        "loss_history": history, "weights_moved": moved,
        "sel": {"control_top1": control["top1"], "tuned_top1": tuned["top1"],
                "pair_auc_wrong_thread": round(pair_auc, 4)},
        "bars": {"G1_control_plus_010_and_055": g1},
    }
    stamp = out["date"]
    # corpus size is part of the artefact's identity (D-044): the n5 refusal
    # and the n9 re-run must not collide
    tag = f"result-{stamp}-n{len(SPLITS['train'])}"
    if not g1:
        out["verdict"] = ("REFUSED at G1 — thread conditioning did not clear "
                          "control+0.10 / 0.55 on sel; nothing saved, the "
                          "sealed holdout was never read")
        p = run_dir / f"{tag}-refused.json"
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out, indent=1))
        print("wrote", p)
        return 0

    # ---- G2: single sealed read ---------------------------------------------
    ho_pos, ho_neg, ho_items = split_rows(root, "holdout", args.seed)
    ctrl_ho = assignment(base_model, tok, ho_items, args.device)
    tuned_ho = assignment(merged, tok, ho_items, args.device)
    per_meeting = {}
    for mid in sorted(SPLITS["holdout"]):
        it = [x for x in ho_items if x[0] == mid]
        a = assignment(base_model, tok, it, args.device)
        b = assignment(merged, tok, it, args.device)
        per_meeting[mid] = {"n": len(it), "control_top1": a["top1"],
                            "tuned_top1": b["top1"]}
    g2 = (tuned_ho["top1"] is not None and ctrl_ho["top1"] is not None
          and tuned_ho["top1"] >= ctrl_ho["top1"] + 0.10
          and tuned_ho["top1"] >= 0.55)
    out["holdout"] = {"read": "once", "control_top1": ctrl_ho["top1"],
                      "tuned_top1": tuned_ho["top1"], "cases": len(ho_items),
                      "per_meeting": per_meeting}
    out["bars"]["G2_holdout_same_margin"] = g2
    if g2:
        merged.save_pretrained(str(out_dir))
        tok.save_pretrained(str(out_dir))
        (out_dir / "sae_provenance.json").write_text(json.dumps({
            "base": BASE, "mechanism": "sentence-vs-thread assignment",
            "seed": args.seed, "date": stamp,
            "sel_top1": tuned["top1"], "holdout_top1": tuned_ho["top1"],
            "wired": False,
            "note": "G3 (online assignment replacing gold keys end-to-end) is "
                    "a separate declared round"}, indent=1), encoding="utf-8")
        out["verdict"] = ("CLEARED G1+G2 — thread-conditioned assignment works "
                          "where sentence-vs-sentence did not; model saved "
                          "UNWIRED; G3 is the next decision")
    else:
        out["verdict"] = ("REFUSED at G2 — sel gain did not transfer to the "
                          "sealed holdout; nothing saved")
    p = run_dir / f"{tag}-{'pass' if g2 else 'refused'}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("sel", "holdout", "bars", "verdict")},
                     indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
