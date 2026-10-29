"""Judgement probe for the linker round (D-046): can the XNLI head tell a
supersede from a same-thread non-supersede, bare and with the thread as context?

Why this exists as a tool: the D-046 conclusion ("the judge is at chance, thread
identity is the constraint") is only citable if the number regenerates from an
artefact. The rule under test is exactly the one the round would use:

  bare          premise = old text, hypothesis = new text
  thread_prefix premise = the last K same-drawer turns before old, hypothesis =
                new text

Scored on TRAIN+SEL temporal gold (the sealed holdout was spent by D-039 and is
never read here; the spent external test is refused). Metrics: AUC per class
and the argmax-`contradicts` precision/recall, which is the only way this head
could authorise a supersede. No bars — this is a diagnostic, not a promotion.

Run: PYTHONPATH=src python -m engine.eval.linker_probe [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np

from engine.config import repo_root

SPLITS = ("train", "sel")
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
GOLD_FILES = {"train": "summre_temporal_train.json",
              "sel": "summre_temporal_sel.json"}
K_PREFIX = 3
NEG_PER_DRAWER = 300


def _load(root: Path):
    data = json.loads((root / "data/external/summre_test/meetings.json")
                      .read_text(encoding="utf-8"))
    return {m: (v["segments"] if isinstance(v, dict) else v)
            for m, v in data.items()}


def rows(root: Path, meetings: dict) -> list[tuple[str, str, int, str]]:
    """(premise, hypothesis, label, variant) rows; label 1 = gold supersede."""
    out = []
    for split in SPLITS:
        gold = json.loads((root / "data/gold" / GOLD_FILES[split])
                          .read_text(encoding="utf-8"))
        for mid, mm in gold["meetings"].items():
            if mid in SPENT:
                raise ValueError(f"{mid} is the spent external test")
            segs = meetings[mid]
            kp = root / "data/gold" / f"summre_subjects_{mid}.json"
            keys = {int(i): v for i, v in
                    ((json.loads(kp.read_text(encoding="utf-8")).get("fine_keys")
                      or json.loads(kp.read_text(encoding="utf-8")).get("keys")
                      or {})).items() if v}
            pos = {(int(r["old"]), int(r["new"]))
                   for th in mm["threads"] for r in th["relations"]
                   if r["relation"] == "supersedes"}
            groups: dict[str, list[int]] = {}
            for i, k in keys.items():
                if i < len(segs):
                    groups.setdefault(k, []).append(i)
            for idxs in groups.values():
                idxs = sorted(idxs)
                dpos = [p for p in pos if p[0] in idxs and p[1] in idxs]
                pairs = [(i, j) for ai, i in enumerate(idxs) for j in idxs[ai + 1:]]
                neg = [p for p in pairs if p not in pos]
                random.Random(7).shuffle(neg)
                for (i, j) in dpos + neg[:NEG_PER_DRAWER]:
                    label = 1 if (i, j) in dpos else 0
                    old_text = segs[i]["text"].strip()
                    new_text = segs[j]["text"].strip()
                    out.append((old_text, new_text, label, "bare"))
                    prev = [x for x in idxs if x < i][-K_PREFIX:]
                    premise = (" ".join(segs[x]["text"].strip() for x in prev)
                               if prev else old_text)
                    out.append((premise, new_text, label, f"prefix{K_PREFIX}"))
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return round(float(u / (len(pos) * len(neg))), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--maxlen", type=int, default=192)
    args = ap.parse_args(argv)
    import torch

    from engine.link.nli_finetune import LABELS, load_head

    root = repo_root()
    got = load_head(device=args.device)
    if got is None:
        print("no NLI head at models/linker_nli — nothing to probe")
        return 1
    model, tok, meta = got
    dev = next(model.parameters()).device
    all_rows = rows(root, _load(root))

    scores: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(all_rows), 64):
            chunk = all_rows[i:i + 64]
            enc = tok([r[0] for r in chunk], [r[1] for r in chunk],
                      truncation=True, padding=True, max_length=args.maxlen,
                      return_tensors="pt").to(dev)
            scores.append(torch.softmax(model(**enc).logits, -1).cpu().numpy())
    P = np.concatenate(scores)
    lab = np.array([r[2] for r in all_rows])
    var = np.array([r[3] for r in all_rows])

    out: dict = {"date": date.today().isoformat(),
                 "model": "models/linker_nli (XNLI-fr, moderncamembert-base)",
                 "splits": list(SPLITS), "holdout": "not read (D-039 spent its "
                 "single read)", "k_prefix": K_PREFIX, "maxlen": args.maxlen,
                 "per_variant": {}}
    for v in sorted(set(var)):
        sel = var == v
        p, y = P[sel], lab[sel]
        mi = p.argmax(1)
        tp = int(((mi == 2) & (y == 1)).sum())
        fp = int(((mi == 2) & (y == 0)).sum())
        fn = int(((mi != 2) & (y == 1)).sum())
        out["per_variant"][v] = {
            "n_rows": int(sel.sum()), "n_positives": int(y.sum()),
            "auc": {n: auc(p[y == 1, c], p[y == 0, c])
                    for c, n in enumerate(LABELS)},
            "contradicts_argmax": {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
                "recall": round(tp / (tp + fn), 4) if tp + fn else None},
        }
    dest = root / "runs" / "linker_local"
    dest.mkdir(parents=True, exist_ok=True)
    pth = dest / f"probe-{out['date']}.json"
    pth.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
