"""Pairwise sentence-embedding dot product: can it tell real threads apart?

The user's proposal (2026-10-23): "for every chunk pass it through a sentence
embedding and do a dot product to compare if they go together." The adjacent
proposals were measured and died (D-026 cosine-to-centroid lookup, D-034
graph projection), but PAIRWISE cosine scored directly against the hand-
labelled temporal gold has never been computed. This is that number, on the
TRAIN meetings only — the sealed holdout (D-039 protocol) is not touched, and
the spent external test is refused.

Bars are not needed (this is a diagnostic, not a promotion attempt): the
artefact reports AUC for positives-vs-hard-negatives (same drawer, not
labelled related) and positives-vs-easy-negatives (cross drawer), plus the
cosine floor distribution that killed the centroid methods.

Caveat recorded in the artefact: my labelling was sparse, so an unlabelled
same-drawer pair may genuinely be related — the hard-negative AUC is a
lower bound on the model's skill and an upper bound on the labelling noise.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from engine.config import load_config, repo_root
from engine.tools.summre_threads import SPENT_TEST, SPLITS, _drawer_keys, _load


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random positive beats a random negative), Mann-Whitney."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2
    return round(float(u / (len(pos) * len(neg))), 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = _load(root)
    mids = sorted(SPLITS["train"])
    if set(mids) & set(SPENT_TEST):
        raise ValueError("spent test meeting in the train split")

    gold = json.loads((root / "data/gold/summre_temporal_train.json")
                      .read_text(encoding="utf-8"))
    pos_all, pos_sup = [], []
    for mid, mm in gold["meetings"].items():
        for th in mm["threads"]:
            for r in th["relations"]:
                (pos_sup if r["relation"] == "supersedes" else pos_all).append(
                    (mid, int(r["old"]), int(r["new"])))
    pos_all = pos_all + pos_sup
    pos_keys = {(m, min(a, b), max(a, b)) for m, a, b in pos_all}

    from engine.embed.base import build_embedder
    cfg = load_config(overrides={"embed": {
        "backend": "sentence_transformers",
        "model_id": "intfloat/multilingual-e5-base", "device": args.device}})
    emb = build_embedder(cfg)

    vecs: dict[str, np.ndarray] = {}
    hard, easy, rel_kind = [], [], []
    rng = np.random.default_rng(42)
    for mid in mids:
        segs = meetings[mid]
        texts = [f"{s.get('speaker', '')}: {s['text'].strip()}" for s in segs]
        v = emb.embed(texts, role="doc")
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        vecs[mid] = v
        keys = _drawer_keys(root, mid)
        groups: dict[str, list[int]] = {}
        for i, k in sorted(keys.items()):
            if i < len(segs):
                groups.setdefault(k, []).append(i)
        for k, idxs in groups.items():
            for ai, i in enumerate(idxs):
                for j in idxs[ai + 1:]:
                    if (mid, i, j) not in pos_keys:
                        hard.append(float(vecs[mid][i] @ vecs[mid][j]))
        # easy negatives: cross-drawer pairs, sampled
        allidx = np.arange(len(segs))
        for _ in range(4000):
            i, j = rng.choice(allidx, 2, replace=False)
            i, j = int(min(i, j)), int(max(i, j))
            if keys.get(i) and keys.get(i) == keys.get(j):
                continue
            if (mid, i, j) in pos_keys:
                continue
            easy.append(float(vecs[mid][i] @ vecs[mid][j]))
        for m, i, j in pos_all:
            if m == mid:
                rel_kind.append(float(vecs[mid][i] @ vecs[mid][j]))

    pos = np.array(rel_kind)
    hard_a = np.array(hard)
    easy_a = np.array(easy)
    sup = np.array([float(vecs[m][i] @ vecs[m][j]) for m, i, j in pos_sup
                    if m in vecs])
    out = {
        "date": date.today().isoformat(),
        "question": "does the pairwise e5 dot product separate related real-"
                    "speech pairs from unrelated ones?",
        "meetings": mids,
        "n_positive_pairs": int(len(pos)), "n_supersedes_pairs": int(len(sup)),
        "n_hard_negatives": int(len(hard_a)), "n_easy_negatives": int(len(easy_a)),
        "auc_related_vs_same_drawer_unlabelled": auc(pos, hard_a),
        "auc_related_vs_cross_drawer": auc(pos, easy_a),
        "auc_supersedes_vs_same_drawer_unlabelled": auc(sup, hard_a),
        "cos_mean": {"positives": round(float(pos.mean()), 4),
                     "hard_negatives": round(float(hard_a.mean()), 4),
                     "easy_negatives": round(float(easy_a.mean()), 4)},
        "cos_p10_p50_p90": {
            "positives": [round(float(x), 4) for x in np.percentile(pos, [10, 50, 90])],
            "hard_negatives": [round(float(x), 4)
                               for x in np.percentile(hard_a, [10, 50, 90])],
            "easy_negatives": [round(float(x), 4)
                               for x in np.percentile(easy_a, [10, 50, 90])]},
        "caveats": [
            "labelling was sparse: an unlabelled same-drawer pair may genuinely "
            "be related, so the hard-negative AUC is a lower bound on the "
            "model and an upper bound on the noise",
            "cosine is symmetric — it cannot supply the DIRECTION a supersede "
            "needs (which statement replaces which); direction must come from "
            "elsewhere (order/date), which is what the guards check",
            "e5 is the same encoder whose centroid geometry killed D-026/D-034; "
            "pairwise is a different question, answered here for the first time"],
    }
    dest = root / "runs" / "summre_paircos"
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"result-{out['date']}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
