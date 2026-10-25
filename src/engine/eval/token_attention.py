"""Concept attention on TOKENS instead of unit vectors (D-048).

The idea under test (owner, 2026-10-24): "pass the discussion through attention
and see which tokens strongly correlate to which — that gives a score for
chunk A vs chunk B." The repo's `concept/model.py` pools whole-unit embeddings;
this tool asks the same question one level down, on the tokens themselves.

Two signals, both read from the frozen `moderncamembert-base` encoder:
  align  — cosine between contextual token vectors of the two statements,
           aggregated max/mean (a soft word alignment);
  attn   — the model's own cross-attention mass from the newer statement's words
           to the older statement's words (last layers, heads averaged).
Subwords are merged into words first: measured on the owner's example, the raw
top target for "les" is "##s" and only second "pizza" — a subword artifact that
would dominate a naive score.

Scored on TRAIN+SEL temporal gold (39 same-drawer supersede pairs) against
same-drawer non-gold pairs. The sealed holdout was spent by D-039 and is not
read. Baselines recorded on the same pairs: e5 sentence cosine AUC 0.6329
(D-040), XNLI head 0.5727 (D-046).

Bars, declared BEFORE the run (this docstring is the declaration):
  T1  the signal SELECTED on the train meetings reaches AUC >= 0.70 on the
      held-out sel meeting (selecting the pooled best of 376 signals on the same
      data would be a selection leak)
  T2  only if T1 clears, a trained token-level attention head must reach
      sel AUC >= 0.75 at exact-pair precision >= 0.70 / recall >= 0.30
If T1 does not clear, no head is fitted. The artefact reports the sel AUC with a
95% Hanley–McNeil interval: the sel split carries 8 positives, so a miss is a
null at this data size, not proof that token attention carries no signal. This
is a representation test, not a promotion: the sealed holdout is untouched and
nothing is wired.

Run: PYTHONPATH=src python -m engine.eval.token_attention [--neg-per-drawer N]
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

BASE = "almanach/moderncamembert-base"
SPLITS = ("train", "sel")
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
GOLD_FILES = {"train": "summre_temporal_train.json",
              "sel": "summre_temporal_sel.json"}
BARS = {"T1_auc>=0.70": 0.70}
PIZZA = ("Quelqu'un veut manger des pizzas ?", "C'est bon je les ai acheté")

#: function words that carry no target; attention/mass on them is noise here
STOP = frozenset({
    "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux", "et", "ou",
    "a", "à", "en", "on", "que", "qui", "quoi", "est", "sont", "c", "ce", "ca",
    "ça", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "me", "te",
    "se", "sa", "son", "ses", "leur", "leurs", "mon", "ma", "mes", "ton", "ta",
    "tes", "notre", "votre", "ce", "cette", "ces", "pour", "par", "dans", "sur",
    "avec", "sans", "pas", "plus", "ne", "y", "d", "l", "s", "n", "j", "m", "t",
    "qu", "dans", "donc", "alors", "ben", "bah", "euh", "hein", "voila", "voilà",
})


def _meetings(root: Path):
    data = json.loads((root / "data/external/summre_test/meetings.json")
                      .read_text(encoding="utf-8"))
    return {m: (v["segments"] if isinstance(v, dict) else v)
            for m, v in data.items()}


def _pairs(root: Path, meetings, neg_per_drawer: int) -> list[dict]:
    rows = []
    for split in SPLITS:
        gold = json.loads((root / "data/gold" / GOLD_FILES[split])
                          .read_text(encoding="utf-8"))
        for mid, mm in gold["meetings"].items():
            if mid in SPENT:
                raise ValueError(f"{mid} is the spent external test")
            segs = meetings[mid]
            kp = root / "data/gold" / f"summre_subjects_{mid}.json"
            kg = json.loads(kp.read_text(encoding="utf-8"))
            keys = {int(i): v for i, v in
                    (kg.get("fine_keys") or kg.get("keys") or {}).items() if v}
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
                cand = [(i, j) for ai, i in enumerate(idxs) for j in idxs[ai + 1:]]
                neg = [p for p in cand if p not in pos]
                random.Random(7).shuffle(neg)
                for (i, j), label in ([(p, 1) for p in dpos]
                                      + [(p, 0) for p in neg[:neg_per_drawer]]):
                    prev = [x for x in idxs if x < i][-3:]
                    rows.append({"mid": mid, "split": split, "old": i, "new": j,
                                 "y": label,
                                 "old_text": segs[i]["text"].strip(),
                                 "new_text": segs[j]["text"].strip(),
                                 "ctx_text": " ".join(segs[x]["text"].strip()
                                                      for x in prev)})
    return rows


def _words(tok, ids: list[int]) -> list[tuple[int, int, str]]:
    """Merge WordPiece continuations into (start, end, word) spans."""
    out: list[tuple[int, int, str]] = []
    for k, tid in enumerate(ids):
        piece = tok.convert_ids_to_tokens(tid)
        if piece in tok.all_special_tokens:
            continue
        if piece.startswith("##"):
            if out:
                s, _e, w = out[-1]
                out[-1] = (s, k + 1, w + piece[2:])
            continue
        out.append((k, k + 1, piece))
    return out


@np.errstate(invalid="ignore", divide="ignore")
def _scores(model, tok, old: str, new: str, context: str | None = None,
            maxlen: int = 256) -> dict:
    import torch

    o = tok(old, add_special_tokens=False)["input_ids"]
    n = tok(new, add_special_tokens=False)["input_ids"]
    c = (tok(context, add_special_tokens=False)["input_ids"]
         if context else [])
    ids = ([tok.cls_token_id] + c + [tok.sep_token_id] + o + [tok.sep_token_id]
           + n + [tok.sep_token_id])
    ids = ids[:maxlen]
    old_start = 2 + len(c)
    new_start = old_start + len(o) + 1
    oi = [i for i in range(old_start, old_start + len(o)) if i < len(ids)]
    ni = [i for i in range(new_start, new_start + len(n)) if i < len(ids)]
    if not oi or not ni:
        return {}
    o_spans = _words(tok, [ids[i] for i in oi])
    n_spans = _words(tok, [ids[i] for i in ni])
    if not o_spans or not n_spans:
        return {}
    it = torch.tensor([ids])
    with torch.no_grad():
        out = model(input_ids=it, attention_mask=torch.ones_like(it),
                    output_attentions=True, output_hidden_states=True)
    # attention: per layer (heads averaged) and the last-4 mean
    per_layer = [a[0].mean(0).double().numpy() for a in out.attentions]
    A = np.mean(per_layer[-4:], axis=0)
    H = out.hidden_states[-1][0].double().numpy()

    def word_vecs(spans, base):
        return np.stack([H[base + s: base + e].mean(0) for s, e, _ in spans])

    ov = word_vecs(o_spans, oi[0])
    nv = word_vecs(n_spans, ni[0])
    ov = ov / np.linalg.norm(ov, axis=1, keepdims=True).clip(min=1e-9)
    nv = nv / np.linalg.norm(nv, axis=1, keepdims=True).clip(min=1e-9)
    cos = nv @ ov.T                                   # [n_new_words, n_old_words]

    n_full = [list(range(ni[0] + s, ni[0] + e)) for s, e, _ in n_spans]
    o_full = [list(range(oi[0] + s, oi[0] + e)) for s, e, _ in o_spans]

    def align_feats(keep_n, keep_o):
        if not keep_n or not keep_o:
            return {}
        C = cos[np.ix_(keep_n, keep_o)]
        return {"align_sym": float((C.max(axis=1).mean() + C.max(axis=0).mean()) / 2),
                "align_max": float(C.max())}

    def attn_feats(A_, keep_n, keep_o):
        """Word-to-word attention mass, summed over subwords (vectorised)."""
        if not keep_n or not keep_o:
            return {}
        ne = np.array([n_full[a][-1] + 1 for a in keep_n])
        ns = np.array([n_full[a][0] for a in keep_n])
        oe = np.array([o_full[b][-1] + 1 for b in keep_o])
        os_ = np.array([o_full[b][0] for b in keep_o])
        C = A_.cumsum(0).cumsum(1)
        M = C[np.ix_(ne - 1, oe - 1)].copy()
        nz_s = ns > 0
        nz_o = os_ > 0
        M[nz_s, :] -= C[np.ix_(ns[nz_s] - 1, oe - 1)]
        M[:, nz_o] -= C[np.ix_(ne - 1, os_[nz_o] - 1)]
        M[np.ix_(nz_s, nz_o)] += C[np.ix_(ns[nz_s] - 1, os_[nz_o] - 1)]
        lens = np.array([len(n_full[a]) for a in keep_n], dtype=float)
        M = M / lens[:, None]
        to_all = sum(A_[n_full[a], :].sum() for a in keep_n)
        return {"attn_max": float(M.max()),
                "attn_mean": float(M.max(axis=1).mean()),
                "attn_frac": float(M.sum() / max(1e-9, to_all)),
                "attn_sum": float(M.sum())}

    all_n = list(range(len(n_spans)))
    all_o = list(range(len(o_spans)))
    cont_n = [a for a, (_s, _e, w) in enumerate(n_spans)
              if w.lower() not in STOP and len(w) > 2]
    cont_o = [b for b, (_s, _e, w) in enumerate(o_spans)
              if w.lower() not in STOP and len(w) > 2]
    out: dict[str, float] = {}
    for tag, kn, ko in (("all", all_n, all_o), ("content", cont_n, cont_o)):
        out.update({f"{tag}_{k}": v for k, v in align_feats(kn, ko).items()})
        out.update({f"{tag}_{k}": v
                    for k, v in attn_feats(A, kn, ko).items()})
        for li, layer in enumerate(per_layer):
            for k, v in attn_feats(layer, kn, ko).items():
                out[f"{tag}_L{li:02d}_{k}"] = v
    return out


def auc_ci(a: float, n_pos: int, n_neg: int) -> tuple[float, float, float]:
    """Hanley–McNeil standard error for an AUC, and its 95% interval.

    Required disclosure on this split: 8 positives give SE ~0.10, so a point
    estimate below the 0.70 bar is a *null at this data size*, not evidence the
    signal is absent.
    """
    import math

    if not n_pos or not n_neg or a != a:
        return float("nan"), float("nan"), float("nan")
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    se = math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                    + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))
    return se, max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se)


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
    ap.add_argument("--neg-per-drawer", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    import torch
    from transformers import AutoModel, AutoTokenizer

    root = repo_root()
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE, attn_implementation="eager").eval()
    if args.device != "cpu":
        model.to(args.device)
    rows = _pairs(root, _meetings(root), args.neg_per_drawer)
    print(f"pairs: {len(rows)} (pos {sum(r['y'] for r in rows)})", flush=True)

    feats: list[dict] = []
    for k, r in enumerate(rows, 1):
        f: dict[str, float] = {}
        for tag, ctx in (("pair", None), ("ctx", r["ctx_text"] or None)):
            for name, v in _scores(model, tok, r["old_text"], r["new_text"],
                                   ctx).items():
                f[f"{tag}_{name}"] = v
        feats.append(f)
        if k % 100 == 0:
            print(f"  scored {k}/{len(rows)}", flush=True)
    names = sorted({k for f in feats for k in f})
    y = np.array([r["y"] for r in rows], dtype=bool)
    is_tr = np.array([r["split"] == "train" for r in rows])
    is_se = ~is_tr
    out = {"date": date.today().isoformat(), "splits": list(SPLITS),
           "holdout": "not read (D-039 spent its single read)",
           "n_pairs": len(rows), "n_positives": int(y.sum()),
           "n_train": int(is_tr.sum()), "n_sel": int(is_se.sum()),
           "n_signals_scanned": len(names),
           "neg_per_drawer": args.neg_per_drawer,
           "baselines": {"e5_pair_cosine_auc": 0.6329,
                         "xnli_pair_auc": 0.5727},
           "signals": {}, "bars": dict(BARS),
           "protocol": ("the signal is SELECTED on train AUC and scored on sel; "
                        "scanning many signals and keeping the pooled best would "
                        "be a selection leak")}
    for name in names:
        v = np.array([f.get(name, np.nan) for f in feats], dtype=float)
        ok = ~np.isnan(v)
        row = {"train": auc(v[ok & is_tr & y], v[ok & is_tr & ~y]),
               "sel": auc(v[ok & is_se & y], v[ok & is_se & ~y]),
               "pooled": auc(v[ok & y], v[ok & ~y])}
        out["signals"][name] = row
    ranked = sorted(out["signals"].items(),
                    key=lambda kv: (kv[1]["train"] if kv[1]["train"] ==
                                    kv[1]["train"] else -1), reverse=True)
    out["top_by_train"] = [{"name": k, **v} for k, v in ranked[:10]]
    sel_auc = ranked[0][1]["sel"]
    n_pos_se = int((is_se & y).sum())
    n_neg_se = int((is_se & ~y).sum())
    se, lo, hi = auc_ci(sel_auc, n_pos_se, n_neg_se)
    out["selected_on_train"] = {"name": ranked[0][0], **ranked[0][1],
                                "sel_n_pos": n_pos_se, "sel_n_neg": n_neg_se,
                                "sel_se": round(se, 4) if se == se else None,
                                "sel_ci95": [round(lo, 4), round(hi, 4)]
                                if lo == lo else None}
    t1 = sel_auc == sel_auc and sel_auc >= BARS["T1_auc>=0.70"]
    out["cleared"] = {"T1": bool(t1)}
    out["verdict"] = (
        "T1 cleared — a trained token head is justified (T2 next)" if t1 else
        f"INCONCLUSIVE at T1 — point estimate {sel_auc:.3f} is below the 0.70 bar, "
        f"but the sel split has only {n_pos_se} positives (95% CI "
        f"[{lo:.2f}, {hi:.2f}]); this is a null at this data size, not evidence "
        "that token attention carries no signal. No head was trained.")
    # keep the top signals' per-row scores so a future leave-one-meeting-out
    # analysis does not need another 3k-forward run
    def _num(v):
        return None if v is None or v != v else round(float(v), 6)

    top_names = [k for k, _ in ranked[:10]]
    out["rows_top10"] = [
        {"mid": r["mid"], "split": r["split"], "y": r["y"],
         "scores": {n: _num(f.get(n)) for n in top_names}}
        for r, f in zip(rows, feats)]

    # the owner's example, reported for reference (not part of the score)
    out["pizza_example"] = {"old": PIZZA[0], "new": PIZZA[1],
                            **_scores(model, tok, *PIZZA)}
    dest = root / "runs" / "token_attention"
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"result-{out['date']}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
