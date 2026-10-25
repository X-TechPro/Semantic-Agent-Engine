"""Ranking failure census — why does temporal@1 miss when the gold is in pool?

D-039 proved `gold_in_pool` = 1.0 on every 20-project artefact: the in-force
unit is always retrieved, so every headline miss is a RANKING failure. This
counts the causes instead of guessing them. The store is rebuilt
deterministically (seed 42, 20 projects, test split, shipped defaults) into a
NEW run directory — the headline artefacts are never touched — and every
scored current-type question is replayed through the same
`temporal_correctness_at_1` ruler the headline uses.

Cause taxonomy for a miss (rank-1 is not the in-force gold):
  stale_rank1_stamped    the gold's superseded twin sits at rank 1 and the
                         store KNOWS it (validity demotion too weak)
  stale_rank1_unstamped  a stale unit at rank 1 the linker never stamped
                         (relation coverage, not ranking)
  wrong_version          rank-1 is a current unit of the same event family
  same_subject_other     rank-1 is current, same drawer/subject, different event
  off_topic              rank-1 is unrelated to the gold's subject
plus the distance measure: at what rank does the best gold unit actually sit.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Query
from eval import metrics as M
from eval.temporal import CANONICAL_PROJECTS, gold_unit_map
from gen.ledger import build_ledger
from gen.qa_from_ledger import build_questions
from gen.render import render_meeting
from gen.splice import write_manifest

SCORED = ("current_decision", "proposal_vs_decision", "owner", "date")


def main(seed: int = 42, split: str = "test") -> int:
    root = repo_root()
    ledger = build_ledger(n_projects=CANONICAL_PROJECTS, meetings_per_project=3,
                          seed=seed)
    # regenerate the manifest exactly as run_bench does — reading the file on
    # disk would take whatever split the LAST bench (eval-fast: 8 projects)
    # wrote and census a different question set than the headline measured
    manifest = write_manifest(root / "data/splits/ledger.json",
                              ledger.projects, seed=seed)
    holdout = set(manifest["split"][split])
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    out_dir = root / "runs" / f"ranking_census_{split}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    db = out_dir / "store_B.db"
    for stale in out_dir.glob("store_B.db*"):
        stale.unlink()
    t0 = time.perf_counter()
    # the ruler of the recorded D-040 verdict was rerank-none; pin it so a
    # re-run diagnoses that ruler (the promoted reranker gets its own census
    # only after a re-baselined headline exists)
    eng = Engine(load_config(overrides={"retrieve": {"rerank":
                                                      {"backend": "none"}}}),
                 path=str(db), linker_enabled=True, roster=roster)
    for meeting in ledger.meetings:
        if meeting.project not in holdout:
            continue
        eng.ingest(render_meeting(ledger, meeting.meeting_id, seed=seed))
    gmap = gold_unit_map(eng.store)
    by_id = {u.unit_id: u for u in eng.store.all_units()}
    gold_stale = {u for e in {ev.supersedes for ev in ledger.events if ev.supersedes}
                  for u in gmap.get(e, [])}
    qa = build_questions(ledger, seed=seed)
    items = [q for q in qa.items if q["project"] in holdout and q["kind"] in SCORED]

    causes: Counter[str] = Counter()
    by_kind: dict[str, Counter] = {}
    gold_rank: list[int] = []
    examples: list[dict] = []
    # pre-flight for the mention-demotion fix: how often is the WINNING gold
    # unit itself status=mention? if often, demoting mentions hurts.
    win_status: Counter[str] = Counter()
    miss_status: Counter[str] = Counter()
    n_miss = n_q = 0
    for q in items:
        gold_units = {u for e in q["gold_event_ids"] for u in gmap.get(e, [])}
        superseded = {u for e in q.get("gold_superseded_event_ids", [])
                      for u in gmap.get(e, [])}
        hits = eng.retrieve(Query(text=q["question"], project=q["project"],
                                  mode="current"))
        ranked = [h.unit.unit_id for h in hits]
        t1 = M.temporal_correctness_at_1(gold_units, ranked, superseded)
        n_q += 1
        if t1 == 1.0:
            win_status[by_id[ranked[0]].status if ranked else "?"] += 1
            continue
        n_miss += 1
        top = ranked[0] if ranked else None
        if top in by_id:
            miss_status[by_id[top].status] += 1
        gold_ranks = [i + 1 for i, uid in enumerate(ranked) if uid in gold_units]
        gold_rank.append(min(gold_ranks) if gold_ranks else 99)
        if top is None:
            cause = "empty"
        elif top in superseded or top in gold_stale:
            cause = ("stale_rank1_stamped" if top in by_id
                     and not by_id[top].is_current else "stale_rank1_unstamped")
        else:
            tu = by_id.get(top)
            gu = next((by_id[g] for g in sorted(gold_units) if g in by_id), None)
            if tu and gu and set(tu.gold_event_ids) & set(gu.gold_event_ids):
                cause = "wrong_version"
            elif tu and gu and tu.slots.subject and tu.slots.subject == gu.slots.subject:
                cause = "same_subject_other"
            else:
                cause = "off_topic"
        causes[cause] += 1
        by_kind.setdefault(q["kind"], Counter())[cause] += 1
        if len(examples) < 20 and top in by_id:
            examples.append({
                "kind": q["kind"], "question": q["question"],
                "rank1_text": by_id[top].text[:110],
                "rank1_status": by_id[top].status,
                "rank1_current": by_id[top].is_current,
                "gold_text": (by_id[next(iter(sorted(gold_units)))].text[:110]
                              if gold_units else None),
                "gold_best_rank": min(gold_ranks) if gold_ranks else None})
    eng.close()
    out = {
        "date": time.strftime("%Y-%m-%d"), "seed": seed, "split": split,
        "questions_scored": n_q, "misses": n_miss,
        "miss_rate": round(n_miss / n_q, 4) if n_q else None,
        "causes": dict(causes), "causes_pct": {k: round(v / n_miss, 4)
                                               for k, v in causes.items()},
        "win_status": dict(win_status), "miss_status": dict(miss_status),
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "gold_best_rank_hist": dict(Counter(gold_rank)),
        "median_gold_rank": sorted(gold_rank)[len(gold_rank) // 2] if gold_rank else None,
        "examples": examples,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "census.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("questions_scored", "misses", "miss_rate",
                                          "causes", "causes_pct", "by_kind",
                                          "median_gold_rank")}, indent=1))
    print("wrote", out_dir / "census.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
