"""beta_mention A/B — the ranking fix the census asked for (D-040).

The census (runs/ranking_census_test_42/census.json) found mention units —
statements that name a subject without asserting a state — on rank 1 in 8 of
17 headline misses, while only 3 of 18 wins had a mention at rank 1, and the
median gold rank among misses is 2. This experiment decides whether
demoting mentions on current queries (the `beta_mention` validity knob) is
worth shipping, on the SAME ruler the headline uses (eval.temporal.
_score_system), with stores rebuilt deterministically into a new directory.

Bars, declared before any number was measured:
  REPRODUCE  the beta=1.0 control pass must equal the recorded headline
             artefacts question-for-question on both splits. A mismatch
             means the ruler or the corpus moved — the experiment is void.
  SEL        dev x seeds 42/43/44 over betas {1.0 (control), 0.3, 0.5, 0.7}:
             the winner is the beta with the best dev temporal@1 mean that
             also beats the control. No winner -> knob stays off, test is
             never read with a treatment.
  CONFIRM    test x 3 seeds, single read with the dev winner:
             T1 mean temporal@1 >= recorded mean + 0.03;
             T2 paired bootstrap (new vs recorded per question, grouped by
                project, n=1000) CI excludes zero on 3/3 seeds;
             T3 no guard regression on any seed: gold_in_pool,
                abstain_correct, stale@10, hyp_leak at least as good.
  SHIP       all bars clear -> DEFAULTS beta_mention becomes the winner,
             the fast gate is re-recorded, make demo must still pass 19/19.
             Any miss -> the knob stays 1.0 (off) and this artefact records
             the refusal; the shipped default is unchanged either way.
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Source  # noqa: F401  (kept for parity with bench)
from eval import metrics as M
from eval.temporal import (CANONICAL_PROJECTS, _score_system, gold_unit_map)
from gen.ledger import build_ledger
from gen.qa_from_ledger import build_questions
from gen.render import render_meeting
from gen.splice import write_manifest

SEEDS = (42, 43, 44)
DEV_BETAS = (1.0, 0.3, 0.5, 0.7)
SCORED_TYPES = ("current_decision", "history", "proposal_vs_decision",
                "owner", "date", "unanswerable")
GUARDS = ("gold_in_pool", "abstain_correct", "stale@10", "hyp_leak")


def _recorded(split: str, seed: int) -> dict:
    root = repo_root()
    p = (root / "runs" / f"m1_m2_temporal_{split}_{seed}_p20_r-rules_k-none"
         / "metrics.json")
    return json.loads(p.read_text(encoding="utf-8"))["system_B"]


def _score_passes(seed: int, split: str, betas: tuple[float, ...],
                  out_dir: Path) -> dict[float, dict]:
    """One deterministic store, scored once per beta (validity is query-time,
    so ingest is beta-independent — the same reuse run_bench relies on)."""
    root = repo_root()
    ledger = build_ledger(n_projects=CANONICAL_PROJECTS, meetings_per_project=3,
                          seed=seed)
    manifest = write_manifest(root / "data/splits/ledger.json",
                              ledger.projects, seed=seed)
    holdout = set(manifest["split"][split])
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    db = out_dir / f"store_{split}_s{seed}.db"
    for stale in out_dir.glob(db.name + "*"):
        stale.unlink()
    # the control compares against the RECORDED k-none headline artefacts;
    # the whole experiment runs on that ruler (D-041 promoted the reranker
    # afterwards — an unpinned re-run would fail its own reproduction gate)
    cfg = load_config(overrides={"store": {"path": str(db)},
                                 "retrieve": {"rerank": {"backend": "none"}}})
    eng = Engine(cfg, path=str(db), linker_enabled=True, roster=roster)
    for meeting in ledger.meetings:
        if meeting.project not in holdout:
            continue
        eng.ingest(render_meeting(ledger, meeting.meeting_id, seed=seed))
    gmap = gold_unit_map(eng.store)
    qa = build_questions(ledger, seed=seed)
    items = [q for q in qa.items if q["project"] in holdout
             and q["kind"] in SCORED_TYPES]
    res: dict[float, dict] = {}
    for beta in betas:
        eng.cfg = load_config(overrides={"store": {"path": str(db)},
                                         "retrieve": {"rerank":
                                                      {"backend": "none"},
                                                      "validity":
                                                      {"beta_mention": beta}}})
        eng._pipeline = None  # rebuild from the swapped cfg (same code path)
        res[beta] = _score_system(eng, items, gmap, ledger, eng.cfg,
                                  arm="system_B")
    eng.close()
    for stale in out_dir.glob(db.name + "*"):
        stale.unlink()  # store binaries are disposable; the JSON is the proof
    return res


def _paired(old: dict, new: dict) -> dict:
    """Paired bootstrap of new vs recorded, per question, grouped by project."""
    va, vb, groups = old["_values"]["temporal@1"], new["_values"]["temporal@1"],\
        old["_groups"]
    assert len(va) == len(vb) == len(groups)
    d = M.paired_bootstrap_diff(va, vb, groups, n=1000)
    return {"delta": round(d["delta"], 4), "ci95": [round(d["lo"], 4),
                                                    round(d["hi"], 4)],
            "excludes_zero_up": bool(d["significantly_better"] > 0)}


def main() -> int:
    root = repo_root()
    out_dir = root / "runs" / "beta_mention_ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # ---- dev selection ----------------------------------------------------
    dev: dict[int, dict[float, float]] = {}
    dev_full: dict[int, dict[float, dict]] = {}
    for seed in SEEDS:
        rec = _recorded("dev", seed)
        passes = _score_passes(seed, "dev", DEV_BETAS, out_dir)
        repro = [d["qid"] for d in rec["per_question"]] == \
                [d["qid"] for d in passes[1.0]["per_question"]] and \
                all(abs((a["temporal@1"] or 0.0) - (b["temporal@1"] or 0.0))
                    < 1e-9 for a, b in zip(rec["per_question"],
                                           passes[1.0]["per_question"]))
        if not repro:
            raise RuntimeError(f"dev control does not reproduce seed {seed} — "
                               "the ruler moved; experiment void, nothing ships")
        dev[seed] = {b: p["temporal@1"] for b, p in passes.items()}
        dev_full[seed] = {str(b): {k: p[k] for k in ("temporal@1",) + GUARDS}
                          for b, p in passes.items()}
    means = {b: round(sum(dev[s][b] for s in SEEDS) / len(SEEDS), 4)
             for b in DEV_BETAS}
    ctrl = means[1.0]
    candidates = {b: m for b, m in means.items()
                  if b != 1.0 and m > ctrl
                  and all(dev[s][b] >= dev[s][1.0] for s in SEEDS)}
    winner = max(candidates, key=candidates.get) if candidates else None

    result: dict = {"date": date.today().isoformat(),
                    "question": "does demoting mention units on current "
                                "queries raise the headline?",
                    "census_basis": "runs/ranking_census_test_42/census.json",
                    "dev": {"per_seed": {str(s): {str(b): dev[s][b]
                                                  for b in DEV_BETAS}
                                          for s in SEEDS},
                            "means": {str(b): m for b, m in means.items()},
                            "winner": winner,
                            "rule": "best mean over control, no seed worse"},
                    "reproduce_control": "exact on all dev seeds (hard gate)"}

    if winner is None:
        result["verdict"] = ("REFUSED at dev — no beta beat the control; the "
                             "test split was never read with a treatment; "
                             "knob stays 1.0 (off)")
        result["elapsed_s"] = round(time.perf_counter() - t0, 1)
        p = out_dir / f"result-{result['date']}.json"
        p.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
        return 0

    # ---- single confirmation read on test ---------------------------------
    test = {}
    for seed in SEEDS:
        rec = _recorded("test", seed)
        passes = _score_passes(seed, "test", (1.0, winner), out_dir)
        assert all(abs((a["temporal@1"] or 0.0) - (b["temporal@1"] or 0.0))
                   < 1e-9 for a, b in zip(rec["per_question"],
                                          passes[1.0]["per_question"])), \
            f"test control does not reproduce seed {seed} — ruler moved"
        new = passes[winner]
        guards = {g: {"recorded": rec[g], "new": new[g],
                      "ok": (new[g] >= rec[g]) if g in ("gold_in_pool",
                                                        "abstain_correct")
                      else (new[g] <= rec[g])}
                  for g in GUARDS}
        test[seed] = {"recorded": rec["temporal@1"], "new": new["temporal@1"],
                      "paired": _paired(rec, new), "guards": guards}
    mean_new = round(sum(t["new"] for t in test.values()) / 3, 4)
    mean_old = round(sum(t["recorded"] for t in test.values()) / 3, 4)
    t1 = mean_new >= mean_old + 0.03
    t2 = all(t["paired"]["excludes_zero_up"] for t in test.values())
    t3 = all(g["ok"] for t in test.values() for g in t["guards"].values())
    cleared = t1 and t2 and t3
    result["test"] = {"winner": winner, "per_seed": {str(s): t for s, t in
                                                     test.items()},
                      "mean_recorded": mean_old, "mean_new": mean_new,
                      "bars": {"T1_mean_gain>=0.03": t1,
                               "T2_ci_excludes_zero_3of3": t2,
                               "T3_no_guard_regression": t3},
                      "cleared": cleared}
    result["verdict"] = (
        f"CLEARED — DEFAULTS beta_mention becomes {winner}; gate re-record, "
        f"make demo and report regeneration follow" if cleared else
        f"REFUSED at test — beta {winner} improved dev but missed "
        f"{[k for k, v in result['test']['bars'].items() if not v]}; "
        "knob stays 1.0 (off)")
    result["elapsed_s"] = round(time.perf_counter() - t0, 1)
    p = out_dir / f"result-{result['date']}.json"
    p.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("dev", "test", "verdict")},
                     indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
