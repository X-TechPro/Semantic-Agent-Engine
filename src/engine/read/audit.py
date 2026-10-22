"""Audit: why does the trained reader change the relation layer's result?

The reader is upstream of the guards, so its labels are not a quality metric
they can be traded against each other — `type` and `status` are the *inputs*
G7/G3 read. This script measures the mechanism directly on one split:

  * how many gold supersedes each reader lets through, and which guard refused
    the rest (the `guard` column on the edge is the record of that);
  * whether the two readers disagree on the specific fields the guards test;
  * how many units each reader put into the index at all, because `is_chatter`
    runs on the reading and so a different reader can change the corpus — which
    would make any end-to-end comparison a comparison of two different corpora,
    not two readers.

Run: PYTHONPATH=src python -m engine.read.audit --split dev
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.link.supersede import SUPERSEDING_TYPES, VALUE_BEARING_TYPES
from gen.ledger import build_ledger
from gen.render import render_meeting
from gen.splice import write_manifest


def _unit_index(eng: Engine) -> tuple[dict[str, str], dict[str, object]]:
    """event_id -> unit_id, and unit_id -> unit."""
    by_unit = {}
    event_of: dict[str, str] = {}
    for u in eng.store.all_units():
        by_unit[u.unit_id] = u
        for eid in u.gold_event_ids:
            event_of.setdefault(eid, u.unit_id)
    return event_of, by_unit


def _edges(eng: Engine) -> list:
    out, seen = [], set()
    for u in eng.store.all_units():
        for e in eng.store.edges_for(u.unit_id, direction="both"):
            key = (e.from_id, e.to_id, e.relation)
            if key not in seen:
                seen.add(key)
                out.append(e)
    return out


def audit(split: str = "dev", projects: int = 20, seed: int = 42,
          backends: tuple[str, ...] = ("rules", "model")) -> dict:
    root = repo_root()
    ledger = build_ledger(n_projects=projects, seed=seed)
    sp = write_manifest(root / "data/splits/ledger.json", ledger.projects,
                        seed=seed)["split"]
    holdout = set(sp[split])
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    gold = [g for g in ledger.gold_supersede_edges()
            if ledger.event(g["new"]).project in holdout
            and ledger.event(g["old"]).project in holdout]

    reports: dict[str, dict] = {}
    for backend in backends:
        tmp = root / "runs" / "reader_audit"
        tmp.mkdir(parents=True, exist_ok=True)
        db = tmp / f"{split}_{backend}.db"
        for stale in tmp.glob(f"{split}_{backend}.db*"):
            stale.unlink()
        cfg = load_config(overrides={"reader": {"backend": backend},
                                     "store": {"path": str(db)},
                                     "embed": {"backend": "hash"}})
        eng = Engine(cfg, path=db, roster=roster, linker_enabled=True)
        assert eng.reader_backend.startswith(backend if backend == "rules"
                                             else "model"), eng.reader_backend
        tally = Counter()
        for m in ledger.meetings:
            if m.project in holdout:
                rep = eng.ingest(render_meeting(ledger, m.meeting_id, seed=seed))
                for k in ("n_units", "n_stored", "n_skipped", "n_queries"):
                    tally[k] += getattr(rep, k)
                for rel, n in rep.relations.items():
                    tally[f"rel:{rel}"] += n
        event_of, by_unit = _unit_index(eng)
        edges = _edges(eng)
        # exact (new, old) pairs. "Did this unit supersede *anything*" would score
        # a supersede against the wrong predecessor as a hit and inflate recall
        # — the same instrument class that already cost me two retracted numbers.
        pred_pairs = {(e.from_id, e.to_id) for e in edges
                      if e.relation == "supersedes"}
        gold_units = {(event_of.get(g["new"]), event_of.get(g["old"]))
                      for g in gold if g["new"] in event_of and g["old"] in event_of}
        hit = gold_units & pred_pairs
        guard_of = {(e.from_id, e.to_id): e for e in edges}
        # which guard refused each MISSED gold supersede
        refused = Counter()
        for pair in gold_units - hit:
            e = guard_of.get(pair)
            refused[e.guard if e else "never_proposed"] += 1
        false = Counter()
        for pair in pred_pairs - gold_units:
            nu = by_unit.get(pair[0])
            false[f"{nu.type}/{nu.status}" if nu else "?"] += 1
        labels = Counter()
        for uid, _old in pred_pairs:
            new = by_unit.get(uid)
            if new is None:
                continue
            labels[f"type={new.type}"] += 1
            labels[f"status={new.status}"] += 1
        # the two fields the guards test, measured on the gold-pair sources
        mismatch = Counter()
        for (u, o) in gold_units:
            nu, ou = by_unit.get(u), by_unit.get(o)
            if not nu or not ou:
                continue
            if nu.type not in SUPERSEDING_TYPES:
                mismatch[f"new_type_not_superseding:{nu.type}"] += 1
            if ou.type not in VALUE_BEARING_TYPES:
                mismatch[f"old_type_not_value_bearing:{ou.type}"] += 1
            if nu.status not in {"confirmed", "cancelled"}:
                mismatch[f"new_status_not_assertive:{nu.status}"] += 1
        # the `answers` relation, which was unreachable before 7.3 was applied
        gold_ans = {(event_of.get(e.event_id), event_of.get(e.answer_of))
                    for e in ledger.events
                    if e.kind == "answer" and e.answer_of
                    and e.project in holdout
                    and event_of.get(e.event_id) and event_of.get(e.answer_of)}
        ans_pairs = {(e.from_id, e.to_id) for e in edges if e.relation == "answers"}
        ans_pairs |= {(e.to_id, e.from_id) for e in edges if e.relation == "answers"}
        ans_hit = gold_ans & ans_pairs
        reports[backend] = {
            "reader_backend": eng.reader_backend,
            "corpus": dict(tally),
            "n_gold_pairs": len(gold_units),
            "n_supersedes_written": len(pred_pairs),
            "gold_found": len(hit),
            "recall": round(len(hit) / len(gold_units), 4) if gold_units else None,
            "precision": round(len(hit) / len(pred_pairs), 4) if pred_pairs else None,
            "answers": {"n_gold": len(gold_ans), "n_written": len(ans_pairs),
                        "gold_found": len(ans_hit),
                        "recall": round(len(ans_hit) / len(gold_ans), 4)
                        if gold_ans else None},
            "question_units_stored": sum(
                1 for u in by_unit.values() if u.type == "question"),
            "refused_by_guard": dict(refused),
            "false_by_source_label": dict(false),
            "gold_pair_label_blockers": dict(mismatch),
            "superseding_source_labels": {k: v for k, v in labels.items()
                                          if k.startswith(("type=", "status="))},
        }
        eng.close()
        for stale in tmp.glob(f"{split}_{backend}.db*"):
            stale.unlink()
    return {"split": split, "seed": seed, "backends": reports}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    res = audit(args.split, args.projects, args.seed)
    # written as well as printed: the stdout stream carries HF load noise, so a
    # parseable artefact is the only way to re-read these numbers later.
    out = repo_root() / "runs" / "reader_audit" / f"audit_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("\nwrote", out)
    for name, r in res["backends"].items():
        print(f"\n{name}: corpus={r['corpus'].get('n_stored')} stored, "
              f"skipped {r['corpus'].get('n_skipped')} | "
              f"supersedes {r['n_supersedes_written']} vs "
              f"{r['n_gold_pairs']} gold | recall={r['recall']} "
              f"precision={r['precision']}")
        print("   blockers:", r["gold_pair_label_blockers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
