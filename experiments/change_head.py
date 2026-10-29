"""Step 3/5 gate trained on the real labels, leave-one-meeting-out.

The status head measures "this is phrased as settled", not "the annotator
marked a change": across a 0.4-0.9 threshold sweep its precision stays at 0.21.
This trains the gate the pipeline actually needs, on the 1,475 real binary
labels, with the late-chunked vector of each turn as the input.

Six folds, each meeting predicted by a model that never saw it. The operating
threshold is chosen on the five training meetings only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from engine.config import repo_root
from late_chunk import LateV5
from engine.eval.real_pairs import load_meetings
from engine.eval.v4_data import texts_of


def _prf(y, pred):
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
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
    return float((pos.sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * (len(y) - len(pos)))) if 0 < pos.size < len(y) else 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/v4/change_head.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    from sklearn.linear_model import LogisticRegression

    root = repo_root()
    meetings = load_meetings(root)
    gold = json.loads((root / "data/gold/summre_change_labels.json").read_text())

    late = LateV5(args.device)
    X, y, groups = [], [], []
    for mid, doc in gold["meetings"].items():
        if mid not in meetings:
            continue
        texts = texts_of(meetings, mid)
        V = late.meeting(texts)
        ys = doc["labels"][:len(texts)]
        keep = [i for i in range(len(ys)) if texts[i].strip()]
        X.append(V[keep])
        y.append(np.array(ys)[keep])
        groups.append(np.array([mid] * len(keep)))
    X = np.concatenate(X)
    y = np.concatenate(y)
    g = np.concatenate(groups)

    preds = np.zeros(len(y), int)
    scores = np.zeros(len(y), float)
    folds = {}
    for mid in sorted(set(g)):
        tr, te = g != mid, g == mid
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], y[tr])
        s_tr = clf.predict_proba(X[tr])[:, 1]
        s_te = clf.predict_proba(X[te])[:, 1]
        best_t, best_f1 = 0.5, -1.0
        for t in np.arange(0.05, 0.96, 0.05):
            f1 = _prf(y[tr], s_tr >= t)["f1"]
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        preds[te] = (s_te >= best_t).astype(int)
        scores[te] = s_te
        folds[mid] = {"threshold": round(best_t, 2),
                      **_prf(y[te], preds[te]),
                      "auc": round(_auc(y[te], s_te), 3)}
    out = {"n": int(len(y)), "changes": int(y.sum()),
           "pooled": _prf(y, preds.astype(bool)),
           "auc": round(_auc(y, scores), 3), "folds": folds}
    p = root / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
