"""Train the linker pair classifier (BUILD_GUIDE 6.8 baseline (b), 9.5 tuning rule).

Tuning discipline: fit on **train** projects, pick the operating point on
**dev**, and only then report test. The first draft of this file wrote the
dev-selected threshold into a JSON key that inference never read, so the trained
model silently ran at decision_function >= 0 instead of its tuned point — a
number that looked trained but was not. Because sigmoid is strictly monotonic,
`decision_function >= t` is *exactly* `score >= sigmoid(t)`, so the operating
point lives in the one field the classifier actually consults.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.link.model import FEATURES, PairClassifier, build_features
from gen.ledger import Ledger, build_ledger
from gen.render import render_meeting
from gen.splice import write_manifest


def harvest(eng: Engine, ledger: Ledger, projects: set[str]
            ) -> list[dict[str, Any]]:
    """Feature rows for every ordered within-project unit pair, each pair once.

    Both directions are emitted deliberately: (later, earlier) can be a
    supersede, (earlier, later) never should be, so the reversed pairs are
    the cheapest signal that the model learned date order, not just topic.

    The `seen` guard is the fix for a double-count: iterating `a` over all units and
    `b` over all units, emitting *both* orders at each step, visits every ordered
    pair twice — once from (a, b) and once from (b, a). `runs/linker_lr/result.json`
    said 5096 pairs / 240 positives, and every count there halves to an integer
    (2548 / 120 / 2030 / 226 / 150 / 22), which is exactly that fingerprint.

    This is NOT a bookkeeping fix. I first claimed the fit was unaffected because
    duplicated rows scale tp/fp/fn uniformly — true of the *metric*, false of the
    *model*: fitting the same rows with and without duplication changed the
    coefficients (max abs diff 1.541) and held-out F1 (0.3487 -> 0.3299). Duplicated
    rows double the loss while the L2 penalty stays fixed, so each row counts twice
    against regularisation and the effective `C` doubles. Deduping therefore trains a
    different classifier; `runs/linker_lr/result_prefix_dup.json` keeps the old one.

    Do not read the old-vs-new artefacts as a controlled comparison of that change —
    they differ in more than duplication (the older file has no `answers` rows and
    `contradicts: 300` against 301 now), so the only clean evidence is the isolated
    fit above.
    """
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for proj in projects:
        units = [u for u in eng.store.all_units()
                 if u.project == proj and u.gold_event_ids]
        for a in units:
            for b in units:
                if a.unit_id == b.unit_id:
                    continue
                for new, old in ((a, b), (b, a)):
                    key = (new.unit_id, old.unit_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        gold = ledger.gold_relation(new.gold_event_ids[0],
                                                    old.gold_event_ids[0])
                    except KeyError:
                        continue
                    rows.append({"features": build_features(new, old), "gold": gold,
                                 "pair": (new.unit_id, old.unit_id),
                                 "units": (new, old)})
    return rows


def _rules_on(y: np.ndarray, rows: list[dict[str, Any]]
              ) -> tuple[float, float, float]:
    """Run the actual rules linker + guards over the same pairs."""
    from engine.link.rules import propose as rules_propose
    from engine.link.supersede import evaluate

    pred = []
    for r in rows:
        new, old = r["units"]
        rel = rules_propose(new, old)
        if rel == "supersedes":
            rel = evaluate(new, old, "supersedes")[0]
        pred.append(rel == "supersedes")
    return precision_recall(y, np.array(pred, dtype=bool))


def precision_recall(pos: np.ndarray, pred: np.ndarray
                     ) -> tuple[float, float, float]:
    tp = int(np.sum(pred & pos))
    fp = int(np.sum(pred & ~pos))
    fn = int(np.sum(~pred & pos))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def _matrix(rows: list[dict[str, Any]]):
    X = np.array([[r["features"].get(f, 0.0) for f in FEATURES] for r in rows],
                 dtype="float64")
    y = np.array([r["gold"] == "supersedes" for r in rows])
    return X, y


def _harvest_split(ledger: Ledger, projects: list[str], seed: int, run_dir: Path,
                   embed_backend: str, tag: str) -> list[dict[str, Any]]:
    """Feature rows for one held-out split, in its own throwaway store."""
    db = run_dir / f"harvest_{tag}.db"
    for stale in run_dir.glob(f"harvest_{tag}.db*"):
        stale.unlink()
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "embed": {"backend": embed_backend}})
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    eng = Engine(cfg, path=db, roster=roster, linker_enabled=False)
    for m in ledger.meetings:
        if m.project in set(projects):
            eng.ingest(render_meeting(ledger, m.meeting_id, seed=seed))
    rows = harvest(eng, ledger, set(projects))
    eng.close()
    for stale in run_dir.glob(f"harvest_{tag}.db*"):
        stale.unlink()
    return rows


def train(projects: int = 20, seed: int = 42, out: Path | None = None,
          embed_backend: str = "hash") -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression

    root = repo_root()
    run_dir = root / "runs" / "linker_lr"
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = build_ledger(n_projects=projects, seed=seed)
    sp = write_manifest(root / "data/splits/ledger.json", ledger.projects,
                        seed=seed)["split"]

    train_rows = _harvest_split(ledger, sp["train"], seed, run_dir, embed_backend,
                                "train")
    labels = np.array([r["gold"] for r in train_rows])
    n_pos = int((labels == "supersedes").sum())
    if n_pos < 5:
        raise SystemExit(f"too few positive pairs to train ({n_pos}); raise --projects")

    Xtr, ytr = _matrix(train_rows)
    model = LogisticRegression(class_weight="balanced", max_iter=3000, C=1.0)
    model.fit(Xtr, ytr)

    dev_rows = _harvest_split(ledger, sp["dev"], seed, run_dir, embed_backend, "dev")
    Xde, yde = _matrix(dev_rows)
    dev_df = model.decision_function(Xde)
    # operating point chosen on DEV F1, never on test
    grid = np.linspace(-3.0, 3.0, 121)
    scores = [precision_recall(yde, dev_df >= t)[2] for t in grid]
    best = float(grid[int(np.argmax(scores))])
    dev_p, dev_r, dev_f1 = precision_recall(yde, dev_df >= best)
    dev_rules = _rules_on(yde, dev_rows)

    te_rows = _harvest_split(ledger, sp["test"], seed, run_dir, embed_backend, "test")
    if te_rows:
        Xte, yte = _matrix(te_rows)
        te_df = model.decision_function(Xte)
        test_p, test_r, test_f1 = precision_recall(yte, te_df >= best)
        # same pairs, same guards, real rules linker: the only fair reference
        rules_p, rules_r, rules_f1 = _rules_on(yte, te_rows)
    else:
        test_p = test_r = test_f1 = 0.0
        rules_p = rules_r = rules_f1 = 0.0

    # sigmoid is monotonic, so this threshold reproduces decision_function >= best
    threshold = 1.0 / (1.0 + math.exp(-best))
    clf = PairClassifier([float(x) for x in model.coef_[0]],
                         float(model.intercept_[0]), threshold=threshold,
                         meta={"seed": seed, "n_train_pairs": len(train_rows),
                               "n_positive": n_pos, "dev_bias": best,
                               "threshold_is_sigmoid_of_bias": True,
                               "selected_on": "dev", "embed_backend": embed_backend,
                               "train_projects": sorted(sp["train"]),
                               "dev": {"precision": round(dev_p, 4),
                                       "recall": round(dev_r, 4),
                                       "f1": round(dev_f1, 4),
                                       "rules_reference": {"precision": round(dev_rules[0], 4),
                                                           "recall": round(dev_rules[1], 4),
                                                           "f1": round(dev_rules[2], 4)}},
                               "test": {"precision": round(test_p, 4),
                                        "recall": round(test_r, 4),
                                        "f1": round(test_f1, 4),
                                        "rules_reference": {"precision": round(rules_p, 4),
                                                            "recall": round(rules_r, 4),
                                                            "f1": round(rules_f1, 4)}}})
    saved = clf.save(out or (root / "models" / "linker_lr.json"))
    result = {"model": str(saved), "train_pairs": len(train_rows),
              "positives": n_pos,
              "gold_mix_train": {k: int(v) for k, v in Counter(labels).items()},
              "dev_bias": best, "threshold": round(threshold, 6),
              "dev": clf.meta["dev"], "test": clf.meta["test"],
              "train_rules_reference": list(
                  map(lambda x: round(x, 4), _rules_on(ytr.astype(bool), train_rows))),
              "splits": {k: len(v) for k, v in sp.items()},
              "weights_top": sorted(((f, round(w, 4))
                                     for f, w in zip(FEATURES, model.coef_[0])),
                                    key=lambda t: -abs(t[1]))[:8]}
    (run_dir / "result.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--embed", default="hash", choices=["hash", "st"])
    args = ap.parse_args(argv)
    res = train(args.projects, args.seed, Path(args.out) if args.out else None,
                args.embed)
    print(json.dumps(res, indent=1))
    print(f"\nthreshold {res['threshold']} (dev bias {res['dev_bias']})\n"
          f"  dev  P={res['dev']['precision']} R={res['dev']['recall']} "
          f"F1={res['dev']['f1']}\n"
          f"  test P={res['test']['precision']} R={res['test']['recall']} "
          f"F1={res['test']['f1']} (rules ref "
          f"P={res['test']['rules_reference']['precision']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
