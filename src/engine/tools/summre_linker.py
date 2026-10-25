"""Train the linker pair scorer on REAL French threads (the (a) round, D-039).

The synthetic linker learned value-swaps its own generator produced; the
external read (D-025) showed the rules linker marks 0/28 on real threads.
This is the first attempt with a scorer FIT ON REAL PAIRS: the hand-labelled
temporal gold (summre_temporal_train / _sel / _holdout) over meetings that
also carry drawer gold, so every candidate pair is one the guards could
actually authorise.

Candidate space, declared: ordered pairs (new, old) whose two units share a
subject key (the gold-supplied drawer) inside one meeting — plus the reversed
direction, which must never be a supersede. Cross-drawer pairs are excluded
because `supersede.evaluate` refuses them structurally; scoring them would
flatter precision with unreachable negatives. Date features are constant
within a meeting (one source_date), so the model must learn the direction
from content, exactly as the reversed pairs test.

Split discipline (refused in code, not by trust):
  train    — the only meetings the fit may see;
  sel      — operating-point selection only (threshold grid);
  holdout  — scored EXACTLY ONCE after fit+selection are frozen; the trainer
             refuses a holdout meeting appearing in any other role's file;
  spent    — the external test meetings (007a/013c/014c) are never loaded.

BARS DECLARED BEFORE TRAINING (this docstring is the declaration; the
artefact records pass/fail against them):
  B1  holdout supersedes RECALL >= 0.30
  B2  holdout supersedes PRECISION >= 0.70  (lower bound: my gold is sparse,
      so an unlabeled true supersede counts against the model — a pass is
      conservative, a fail may be pessimistic)
  B3  holdout supersedes F1 > the rules linker's F1 on the SAME pairs
If any bar fails: no checkpoint is written, no backend is wired, the shipped
default stays `rules`, and the artefact says so.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import load_config, repo_root

SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}


def _load_meetings(root: Path) -> dict[str, list[dict]]:
    data = json.loads((root / "data/external/summre_test/meetings.json")
                      .read_text(encoding="utf-8"))
    return {m: (v["segments"] if isinstance(v, dict) else v)
            for m, v in data.items()}


def _gold_relations(path: Path):
    """(supersedes set, other set) of (mid, old_idx, new_idx) + split check."""
    g = json.loads(path.read_text(encoding="utf-8"))
    declared = path.stem.rsplit("_", 1)[-1]
    if g.get("split") != declared:
        raise ValueError(f"{path.name} declares split={g.get('split')!r}, "
                         f"file name says {declared!r}")
    sup: set[tuple[str, int, int]] = set()
    other: set[tuple[str, int, int]] = set()
    for mid, mm in g["meetings"].items():
        if mid in SPENT:
            raise ValueError(f"{mid} is the spent external test — never load it")
        for th in mm["threads"]:
            for r in th["relations"]:
                key = (mid, int(r["old"]), int(r["new"]))
                (sup if r["relation"] == "supersedes" else other).add(key)
    return sup, other


def harvest(meetings: dict, mids: list[str], tmp_dir: Path, tag: str):
    """Ingest each meeting with the gold drawer keys (rules reader) and emit
    same-subject ordered pair rows: (mid, new_idx, old_idx, new, old)."""
    from engine.engine import Engine
    from engine.link.model import build_features
    from eval.frbench import build_meeting_source

    root = repo_root()
    db = tmp_dir / f"linker-{tag}.db"
    for stale in tmp_dir.glob(f"linker-{tag}.db*"):
        stale.unlink()
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "slots": {"subject_resolver": "gold",
                                           "subject_model_dir": str(root / "data/gold")},
                                 "reader": {"backend": "rules"},
                                 "embed": {"backend": "hash"}})
    eng = Engine(cfg, path=str(db), linker_enabled=False)
    rows = []
    for mid in mids:
        eng.ingest(build_meeting_source(mid, meetings[mid],
                                        on_date=date(2021, 10, 5)))
        units: dict[int, Any] = {}
        for u in eng.store.units_of_source(f"summre:{mid}"):
            if u.turn_index is not None:
                units.setdefault(int(u.turn_index), u)
        groups: dict[str, list[int]] = {}
        for idx, u in units.items():
            if u.slots.subject:
                groups.setdefault(u.slots.subject, []).append(idx)
        for idxs in groups.values():
            idxs.sort()
            for a_i, i in enumerate(idxs):
                for j in idxs[a_i + 1:]:
                    rows.append((mid, j, i, units[j], units[i]))
                    rows.append((mid, i, j, units[i], units[j]))
    eng.close()
    for stale in tmp_dir.glob(f"linker-{tag}.db*"):
        stale.unlink()
    return rows


def _matrix(rows, sup, other):
    from engine.link.model import FEATURES, build_features

    X, y, meta = [], [], []
    for mid, ni, oi, new, old in rows:
        X.append([float(build_features(new, old).get(f, 0.0)) for f in FEATURES])
        y.append((mid, oi, ni) in sup)
        meta.append((mid, ni, oi, (mid, oi, ni) in other))
    return np.array(X, dtype="float64"), np.array(y, dtype=bool), meta


def _prf(pos: np.ndarray, pred: np.ndarray):
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn


def select_operating_point(grid, df, y, *, min_precision: float = 0.70) -> dict:
    """Pick the bias whose sel recall is highest AT OR ABOVE `min_precision`.

    D-046 fixed this. The round that wrote `result-2026-10-23-s42-refused.json`
    selected by `argmax F1` on a sel split carrying **6 positives**: it chose a
    bias with precision 0.0014 / recall 0.1667 *on its own selection split*, then
    reported the holdout at 0.0009. B2 (precision >= 0.70) was therefore
    unreachable no matter how good the model was — the selector, not the model,
    produced that number. Selection is now the standard precision-constrained
    one; when no point clears the floor the max-F1 point is returned with
    `precision_floor_met=False` so the artefact says the bar was unreachable
    instead of presenting a hopeless threshold as an operating point.
    """
    best_f1 = {"bias": float(grid[0]), "precision": 0.0, "recall": 0.0, "f1": -1.0}
    best = None
    for t in grid:
        p, r, f1, *_ = _prf(y, df >= t)
        if f1 > best_f1["f1"]:
            best_f1 = {"bias": float(t), "precision": p, "recall": r, "f1": f1}
        if p >= min_precision and (best is None or r > best["recall"]):
            best = {"bias": float(t), "precision": p, "recall": r, "f1": f1}
    if best is None:
        return {**best_f1, "mode": "max_f1_fallback",
                "precision_floor_met": False}
    return {**best, "mode": "precision_constrained",
            "precision_floor_met": True}


def _rules_on(rows, sup):
    """The shipped rules proposer + guards over the SAME pairs."""
    from engine.link.rules import propose as rules_propose
    from engine.link.supersede import evaluate

    pred = []
    for mid, ni, oi, new, old in rows:
        rel = rules_propose(new, old)
        if rel == "supersedes":
            rel = evaluate(new, old, "supersedes")[0]
        pred.append(rel == "supersedes")
    return np.array(pred, dtype=bool)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    np.random.seed(args.seed)

    root = repo_root()
    meetings = _load_meetings(root)
    gold = {}
    mids = {}
    for split in ("train", "sel", "holdout"):
        path = root / "data/gold" / f"summre_temporal_{split}.json"
        gold[split] = _gold_relations(path)
        g = json.loads(path.read_text(encoding="utf-8"))
        mids[split] = sorted(g["meetings"])
    for s in ("sel", "holdout"):  # a meeting may not play two roles
        overlap = set(mids[s]) & set(mids["train"])
        if overlap:
            raise ValueError(f"{s} meetings also in train: {sorted(overlap)}")
    overlap = set(mids["sel"]) & set(mids["holdout"])
    if overlap:
        raise ValueError(f"sel meetings also in holdout: {sorted(overlap)}")

    tmp_dir = root / "runs" / "summre_linker"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows_tr = harvest(meetings, mids["train"], tmp_dir, "train")
    rows_se = harvest(meetings, mids["sel"], tmp_dir, "sel")
    rows_ho = harvest(meetings, mids["holdout"], tmp_dir, "holdout")

    Xtr, ytr, _ = _matrix(rows_tr, *gold["train"])
    Xse, yse, _ = _matrix(rows_se, *gold["sel"])
    Xho, yho, meta_ho = _matrix(rows_ho, *gold["holdout"])

    from sklearn.linear_model import LogisticRegression
    from engine.link.model import FEATURES, PairClassifier

    model = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)
    model.fit(Xtr, ytr)

    se_df = model.decision_function(Xse)
    grid = np.linspace(-3.0, 3.0, 301)
    op = select_operating_point(grid, se_df, yse)
    best = op["bias"]
    sel_p, sel_r, sel_f1, *_ = _prf(yse, se_df >= best)

    ho_df = model.decision_function(Xho)
    thr = 1.0 / (1.0 + math.exp(-best))
    pred = ho_df >= best
    p, r, f1, tp, fp, fn = _prf(yho, pred)
    rules_pred = _rules_on(rows_ho, gold["holdout"][0])
    rp, rr, rf1, rtp, rfp, rfn = _prf(yho, rules_pred)

    # what the false positives actually are: read them, don't count them away
    fps = [(meta_ho[i][0], meta_ho[i][2], meta_ho[i][1],
            rows_ho[i][3].text[:70], rows_ho[i][4].text[:70],
            "labeled-other" if meta_ho[i][3] else "unlabeled")
           for i in np.where(pred & ~yho)[0]]
    missed = [(meta_ho[i][1], meta_ho[i][2], rows_ho[i][3].text[:70],
               rows_ho[i][4].text[:70])
              for i in np.where(~pred & yho)[0]]

    bars = {"B1_recall>=0.30": r >= 0.30,
            "B2_precision>=0.70": p >= 0.70,
            "B3_f1>rules_f1": f1 > rf1}
    cleared = all(bars.values())
    out: dict[str, Any] = {
        "date": date.today().isoformat(), "seed": args.seed,
        "corpus": "SUMM-RE real French, gold drawer keys + hand temporal pairs",
        "splits": {s: {"meetings": mids[s],
                       "pairs": int(len(x)), "positives": int(y.sum())}
                   for s, x, y in (("train", rows_tr, ytr), ("sel", rows_se, yse),
                                   ("holdout", rows_ho, yho))},
        "excluded": {"019c_EAPD/025b_EBDH (old temporal dev)":
                     "no drawer gold -> no same-subject candidate pairs",
                     "spent test meetings": "never loaded"},
        "operating_point": {"sel_bias": best, "threshold_sigmoid": round(thr, 6),
                            "selection": op,
                            "sel": {"precision": round(sel_p, 4),
                                    "recall": round(sel_r, 4),
                                    "f1": round(sel_f1, 4)}},
        "holdout": {"trained": {"precision": round(p, 4), "recall": round(r, 4),
                                "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn},
                    "rules_reference": {"precision": round(rp, 4),
                                        "recall": round(rr, 4), "f1": round(rf1, 4),
                                        "tp": rtp, "fp": rfp, "fn": rfn}},
        "bars_declared_before_training": bars, "cleared": cleared,
        "false_positives": fps[:25], "missed_supersedes": missed[:25],
        "weights_top": sorted(((f, round(w, 4))
                               for f, w in zip(FEATURES, model.coef_[0])),
                              key=lambda t: -abs(t[1]))[:8],
    }
    stamp = date.today().isoformat()
    if cleared:
        clf = PairClassifier([float(x) for x in model.coef_[0]],
                             float(model.intercept_[0]), threshold=thr,
                             meta={"trained_on": "SUMM-RE real threads (D-039)",
                                   "seed": args.seed, "sel_bias": best,
                                   "holdout": out["holdout"],
                                   "splits": out["splits"]})
        out["model"] = str(clf.save(root / "models" / "linker_summre.json"))
    dest = tmp_dir / f"result-{stamp}-s{args.seed}-{'pass' if cleared else 'refused'}.json"
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("splits", "operating_point", "holdout",
                                          "bars_declared_before_training", "cleared")},
                     indent=1))
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
