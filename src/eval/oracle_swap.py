"""Oracle swaps (BUILD_GUIDE 9.3) — give exactly one stage its gold input, keep
everything else at the shipped configuration, and read the end-to-end change.

Why this sits next to the per-stage evaluation in `run_stage.py`: a stage can score
well alone and still cost nothing end to end, or score badly and be harmless because
a later stage repairs it. 9.3 asks the opposite question, and the answer is where
the remaining error actually lives — which is what M5 has to be measured against.

Two disciplines this file exists to keep:

* **No second copy of anything.** Labels come from `gen.ledger.reader_gold` and the
  event's own fields — the same single sources `run_stage.py` scores against — and
  the end-to-end numbers come from `eval.temporal._score_system`, the headline's own
  scorer. A hand-rolled scoring loop here would measure a different ruler than the
  one `runs/temporal_mean.json` reports, and the two would drift silently. A
  duplicated gold table has already inverted one verdict in this project (D-008).
* **Each row states what its oracle does and does not reach.** `reader` and `slots`
  rewrite stored rows, so they perfect the labels *retrieval* sees while the edges
  were still built from the rules reader's output at ingest; `linker` writes gold
  edges onto a linker-disabled ingest, so it is a true one-stage swap of the
  relation layer. Reading a reader gain as "the reader is fine" would over-claim.

Run: PYTHONPATH=src python -m eval.oracle_swap --split dev
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Edge, utcnow
from gen.ledger import build_ledger, reader_gold
from gen.qa_from_ledger import build_questions
from gen.render import render_meeting
from gen.splice import write_manifest
from eval.temporal import _score_system, gold_unit_map

#: order matters: the summary reads `baseline` as the zero of every gain
SWAPS = ("baseline", "reader", "slots", "linker", "reader+slots+linker")

#: what each row can honestly claim
LIMITS = {
    "baseline": "shipped configuration (rules reader, rules slots, rules linker)",
    "reader": "kind/type/status corrected in the store; edges were still built at "
              "ingest from the rules reader, so this is the retrieval-side reader "
              "ceiling, not a full reader replacement",
    "slots": "slots corrected in the store; same ingest-time caveat as `reader`",
    "linker": "true one-stage swap: ingested with the linker OFF, then the gold "
              "supersede edges written through the engine's own mark_superseded",
    "reader+slots+linker": "all three; the ceiling on the shipped ranking path",
}


def _questions(ledger, holdout, seed: int) -> list[dict]:
    qa = build_questions(ledger, seed=seed)
    kinds = ("current_decision", "history", "proposal_vs_decision", "owner", "date",
             "unanswerable")
    return [q for q in qa.items if q["project"] in holdout and q["kind"] in kinds]


def _ingest(cfg, ledger, holdout, roster, path: Path, *, linker: bool) -> Engine:
    """Index the holdout projects, exactly as `run_bench` does.

    The polarity is easy to invert and the failure is silent: indexing the training
    projects instead leaves every evaluated question with no gold unit in the store,
    so recall reads 0.0 and looks like a broken oracle rather than a broken harness.
    """
    for stale in path.parent.glob(path.name + "*"):
        stale.unlink()
    eng = Engine(cfg, path=path, roster=roster, linker_enabled=linker)
    for m in ledger.meetings:
        if m.project not in holdout:
            continue
        eng.ingest(render_meeting(ledger, m.meeting_id, seed=cfg.seed))
    return eng


def _gold_units(eng: Engine, ledger) -> list[tuple]:
    """(unit, event) pairs. Units with no gold event are skipped: the opening line
    of a rendered meeting carries no event, so a gold reader must abstain there
    rather than invent a label (measured on a 2-project ledger)."""
    out = []
    for u in eng.store.all_units():
        if not u.gold_event_ids:
            continue
        try:
            ev = ledger.event(u.gold_event_ids[0])
        except KeyError:
            continue
        out.append((u, ev))
    return out


def _apply_reader_gold(eng: Engine, ledger) -> int:
    n = 0
    for u, ev in _gold_units(eng, ledger):
        # `reader_gold` returns the whole (kind, type, status) triple — the one
        # table this project derives all three from. Unpacking it any other way
        # binds a tuple into a text column (it did: sqlite3.ProgrammingError on
        # parameter 9).
        kind, type_, status = reader_gold(ev)
        if (u.kind, u.type, u.status) == (kind, type_, status):
            continue
        eng.store.update_unit(u.model_copy(update={"kind": kind, "type": type_,
                                                   "status": status}))
        n += 1
    return n


def _apply_slots_gold(eng: Engine, ledger) -> int:
    from engine.slots.regex_dates import parse_amount

    n = 0
    for u, ev in _gold_units(eng, ledger):
        slots = u.slots
        updates: dict[str, Any] = {}
        if ev.subject_key and slots.subject != ev.subject_key:
            updates["subject"] = ev.subject_key
        if ev.person and slots.owner != ev.person:
            updates["owner"] = ev.person
        if ev.due and slots.due != ev.due.isoformat():
            updates["due"] = ev.due.isoformat()
        if (ev.value or "").endswith("€"):
            amount = parse_amount(ev.value or "")
            if amount and slots.amount != amount:
                updates["amount"] = amount
        if not updates:
            continue
        eng.store.update_unit(u.model_copy(update={"slots": slots.model_copy(
            update=updates)}))
        n += 1
    return n


def _apply_linker_gold(eng: Engine, ledger) -> dict[str, int]:
    """Gold edges through the engine's own supersede path, so validity weighting,
    `is_current` and `supersede_confidence` all see exactly what a perfect linker
    would have written."""
    unit_of: dict[str, str] = {}
    for u in eng.store.all_units():
        for eid in u.gold_event_ids:
            unit_of.setdefault(eid, u.unit_id)
    written = skipped = refused = 0
    now = utcnow()
    for g in ledger.gold_supersede_edges():
        new_id, old_id = unit_of.get(g["new"]), unit_of.get(g["old"])
        if not new_id or not old_id:
            skipped += 1                       # one side never entered the corpus
            continue
        old = eng.store.get(old_id)
        if old is None:
            skipped += 1
            continue
        at = date.fromisoformat(g["on"])
        try:
            eng.store.mark_superseded(old_id, new_id, at)
        except ValueError:
            # 6.8.4 refuses a supersede dated before the old unit's valid_from.
            # Refusing silently here would quietly shrink the oracle.
            refused += 1
            continue
        eng.store.add_edges([Edge(from_id=new_id, to_id=old_id, relation="supersedes",
                                  score=1.0, evidence_source_id=old.source_id,
                                  created_at=now, guard="ORACLE")])
        written += 1
    return {"written": written, "no_unit": skipped, "refused": refused}


def run(split: str = "dev", projects: int = 20, seed: int = 42) -> dict[str, Any]:
    root = repo_root()
    ledger = build_ledger(n_projects=projects, seed=seed)
    manifest = write_manifest(root / "data/splits/ledger.json", ledger.projects,
                              seed=seed)
    holdout = set(manifest["split"][split])
    roster = sorted({p for proj in ledger.projects for p in proj["people"]})
    items = _questions(ledger, holdout, seed)

    out_dir = root / "runs" / f"oracle_swap_{split}_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="oracle-swap-"))
    # ruler pin: the recorded headroom numbers (+0.057 / 0.000) were measured
    # pre-D-041 with rerank none; a swap is only meaningful against that ruler
    cfg = load_config(overrides={"retrieve": {"rerank": {"backend": "none"}}})

    results: dict[str, Any] = {"split": split, "seed": seed, "projects": projects,
                               "n_questions": len(items),
                               "config_hash": cfg.hash(), "limits": LIMITS,
                               "swap": {}}
    for name in SWAPS:
        slug = name.replace("+", "_")
        # the linker swap must start from NO rules edges, or "gold relations" would
        # mean "gold plus whatever the rules guessed"
        linker_on = name not in ("linker", "reader+slots+linker")
        scfg = load_config(overrides={"store": {"path": str(tmp / f"{slug}.db")},
                                      "embed": {"backend": "hash"}, "seed": seed,
                                      "retrieve": {"rerank":
                                                   {"backend": "none"}}})
        eng = _ingest(scfg, ledger, holdout, roster, tmp / f"{slug}.db",
                      linker=linker_on)
        gmap = gold_unit_map(eng.store)
        rewrote = {"reader": 0, "slots": 0, "linker": {"written": 0}}
        if "reader" in name:
            rewrote["reader"] = _apply_reader_gold(eng, ledger)
        if "slots" in name:
            rewrote["slots"] = _apply_slots_gold(eng, ledger)
        if "linker" in name:
            rewrote["linker"] = _apply_linker_gold(eng, ledger)
        row = _score_system(eng, items, gmap, ledger, cfg, arm="system_B")
        results["swap"][name] = {
            "temporal@1": row.get("temporal@1"), "recall@5": row.get("recall@5"),
            "gold_in_pool": row.get("gold_in_pool"),
            "stale@10": row.get("stale@10"), "hyp_leak": row.get("hyp_leak"),
            "headroom_pct": row.get("headroom_pct"),
            "units_rewritten": rewrote,
            "backends": {"reader": eng.reader_backend,
                         "linker": getattr(eng.ingest_pipeline, "linker_backend", "?"),
                         "rerank": eng.rerank_backend},
            "limitation": LIMITS[name],
        }
        eng.close()

    base = (results["swap"]["baseline"].get("temporal@1") or 0)
    for name, row in results["swap"].items():
        v = row.get("temporal@1")
        row["gain_over_baseline"] = None if v is None else round(v - base, 4)
    results["baseline_headroom"] = round(1.0 - base, 4)

    (out_dir / "result.json").write_text(json.dumps(results, indent=1, default=str),
                                        encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(results), encoding="utf-8")
    results["_out"] = str(out_dir)
    return results


def _summary(r: dict[str, Any]) -> str:
    rows = r["swap"]
    width = 36
    L = [f"# oracle swaps — {r['split']} seed {r['seed']} | {r['projects']} projects",
         "",
         f"- {r['n_questions']} generated questions on the holdout split; config "
         f"`{r['config_hash']}`; baseline headroom {r['baseline_headroom']}",
         "",
         "| configuration | temporal@1 | gain | recall@5 | gold_in_pool | stale@10 "
         "| hyp_leak | rewritten |", "|---|---|---|---|---|---|---|---|"]
    for name in SWAPS:
        d = rows[name]
        t = d["units_rewritten"]
        link = t["linker"] if isinstance(t["linker"], dict) else {}
        rew = (f"reader {t['reader']}, slots {t['slots']}, "
               f"edges {link.get('written', 0)}")
        L.append(f"| {name} | {d['temporal@1']} | {d['gain_over_baseline']:+} | "
                 f"{d['recall@5']} | {d['gold_in_pool']} | {d['stale@10']} | "
                 f"{d['hyp_leak']} | {rew} |")
    L += ["", "temporal@1", ""]
    for name in SWAPS:
        v = rows[name].get("temporal@1")
        if v is None:
            L.append(f"  {name:<24} not measured")
            continue
        filled = int(round(v * width))
        L.append(f"  {name:<24} {'#' * filled}{'.' * (width - filled)} {v:.4f}")
    L += ["", "What each row means, stated so it cannot be over-read:"]
    for name in SWAPS:
        L.append(f"- **{name}** — {rows[name]['limitation']}")
    link = rows["linker"]["units_rewritten"]["linker"]
    if isinstance(link, dict) and (link.get("refused") or link.get("no_unit")):
        L += ["", f"Gold edges not written: {link.get('no_unit')} with no stored unit "
                  f"on a side, {link.get('refused')} refused by 6.8.4's date rule."]
    L += ["",
          "The gain column is the stage's headroom on THIS split. A near-zero gain "
          "with a non-zero baseline headroom means the stage is not the bottleneck; "
          "a large gain names where the remaining error lives, which is the bar M5 "
          "has to clear (9.7)."]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--projects", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    res = run(args.split, args.projects, args.seed)
    print(_summary(res))
    print("wrote", res["_out"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
