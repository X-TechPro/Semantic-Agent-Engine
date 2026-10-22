"""B-13 instrument: is the supersedes chain REACHABLE on real French, per reader?

Two numbers, in the order the guards need them (the null was a reach-null, so
this measures reachability, never ranking quality):

1. **status firing** — the chain requires assertive units
   (`SUPERSEDING_STATUSES`); the rules reader called 994/1019 real units
   `mention`. Count statuses per shipped reader (rules / reader / reader_v2)
   over the dev meetings — raw French, no gold consulted;
2. **marked rate** — end to end: ingest the dev meetings, count how many
   gold-old units carry `superseded_by` afterwards.

The five extractor proposals this tool used to gate — (b) cosine lookup,
(c) LLM, (d/e) span head, (f) drawer classifier, (g) graph projection — were
each measured against one pre-registered gate (D-026…D-036), none crossed it,
and their code was removed from the repo (D-037). Their verdict artefacts
stay in `runs/` and `docs/REPORT.md` §6e renders them; the gates are not
re-runnable because re-running refusals is not what closes B-13 — annotation
scale is. `--run --arms rules,v2` remains the check the shipped system gets.

Split discipline: the gold file declares its own `split` and it must match
`--split`; the spent TEST set can never be re-run (--split test refuses once
a test artefact exists). VRAM discipline (memory: arms-share-models): all
Engine arms inject ONE shared e5 embedder, released between runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from datetime import date
from typing import Any

from engine.config import repo_root

SUPERSEDING = ("confirmed", "cancelled")
ARM_BUILDERS = ("rules", "v2")


def _load_pair_lists(gold_path, meetings: dict):
    """Per-meeting thread membership -> ({mid: (same-pairs, cross-pairs)},
    relations as (mid, old, new, relation)). Indices are within-meeting
    positions — a global pool would pair meeting A's segments against meeting
    B's vectors, which is nonsense that looks like data."""
    g = json.loads(gold_path.read_text(encoding="utf-8"))
    per: dict[str, tuple[list, list]] = {}
    sup = []
    for mid, mm in g["meetings"].items():
        thread_of: dict[int, int] = {}
        for ti, th in enumerate(mm["threads"]):
            for r in th["relations"]:
                sup.append((mid, r["old"], r["new"], r["relation"]))
                thread_of[r["old"]] = ti
                thread_of[r["new"]] = ti
        idx = sorted(thread_of)
        same, cross = [], []
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1:]:
                (same if thread_of[i] == thread_of[j] else cross).append((i, j))
        per[mid] = (same, cross)
    return g, per, sup


def _eng_cfg(store_path, reader: dict, slots: dict, device: str):
    from engine.config import load_config
    return load_config(overrides={"store": {"path": str(store_path)},
                                  "embed": {"backend": "sentence_transformers",
                                            "model_id": "intfloat/multilingual-e5-base",
                                            "device": device},
                                  "reader": reader, "slots": slots,
                                  "link": {"backend": "rules"}})


def _shared_embedder(device: str):
    from engine.config import load_config
    from engine.embed.base import build_embedder
    cfg = load_config(overrides={"embed": {"backend": "sentence_transformers",
                                           "model_id": "intfloat/multilingual-e5-base",
                                           "device": device}})
    return build_embedder(cfg)


def _release(eng=None, emb=None) -> None:
    """Free a just-finished arm before the next is built — one full embedder
    runtime per 8 GB card (the shared-model memory, in the other direction)."""
    if eng is not None:
        eng.close()
    time.sleep(0.2)
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _ingest_dev(device: str, reader: dict, slots: dict, meetings: dict,
                mids: list[str], tmp_dir, name: str, emb):
    """One arm's store over the dev meetings; returns (engine, source ids).
    Stable db name per arm; the embedder is injected, never rebuilt per arm."""
    from engine.engine import Engine
    from eval.frbench import build_meeting_source

    path = tmp_dir / f"reach-{name}.db"
    path.unlink(missing_ok=True)
    eng = Engine(_eng_cfg(path, reader, slots, device), embedder=emb, path=path)
    sids = []
    for mid in mids:
        src = build_meeting_source(mid, meetings[mid], on_date=date(2021, 10, 5))
        eng.ingest(src)
        sids.append(src.source_id)
    return eng, sids


def status_firing(device: str, meetings: dict, mids: list[str], tmp_dir,
                  emb) -> dict:
    """Reader comparison on raw French — no gold touched, this column alone
    cannot bias anything."""
    out = {}
    for name, reader in (("rules", {"backend": "rules"}),
                         ("model(reader)", {"backend": "model"}),
                         ("model(reader_v2)",
                          {"backend": "model",
                           "model_dir": str(repo_root() / "models" / "reader_v2")})):
        eng, sids = _ingest_dev(device, reader, {"subject_resolver": "rules"},
                                meetings, mids, tmp_dir, f"status-{name}", emb)
        units = [u for sid in sids for u in eng.store.units_of_source(sid)]
        out[name] = {
            "n_units": len(units),
            "status_mix": dict(Counter(str(u.status) for u in units)),
            "assertive_units": sum(1 for u in units if str(u.status) in SUPERSEDING),
            "decision_units": sum(1 for u in units if u.type == "decision"),
            "subject_resolved": sum(1 for u in units if u.slots.subject),
        }
        _release(eng)
    return out


def run_reach(device: str, meetings: dict, mids: list[str], gold_path,
              tmp_dir, emb, arms: tuple[str, ...] = ("rules", "v2")) -> dict:
    """End-to-end marked rate per shipped arm on the dev pairs."""
    from eval.frbench import summre_pair_scores

    for a in arms:
        if a not in ARM_BUILDERS:
            raise ValueError(
                f"unknown arm {a!r}; only {ARM_BUILDERS} remain — the extractor "
                "arms (lookup/llm/supervised/drawer/graph) were measured and "
                "removed (D-037)")
    _g, _per, sup = _load_pair_lists(gold_path, meetings)
    sup = [(m, o, n) for m, o, n, r in sup if r == "supersedes"]
    arm_defs = {
        "rules": ({"backend": "rules"}, {"subject_resolver": "rules"}),
        "v2": ({"backend": "model",
                "model_dir": str(repo_root() / "models" / "reader_v2")},
               {"subject_resolver": "rules"}),
    }
    out = {}
    for name in arms:
        reader, slots = arm_defs[name]
        eng, sids = _ingest_dev(device, reader, slots, meetings, mids,
                                tmp_dir, name, emb)
        uid_of = {}
        for sid in sids:
            for u in eng.store.units_of_source(sid):
                if u.turn_index is not None:
                    uid_of.setdefault((sid.split(":", 1)[1], int(u.turn_index)),
                                      u.unit_id)
        marked = 0
        rows = []
        for mid, old, new in sup:
            ou = uid_of.get((mid, old))
            nu = uid_of.get((mid, new))
            if ou and nu:
                unit = eng.store.get(ou)
                marked += int(bool(unit is not None and unit.superseded_by))
            q = meetings[mid][old]["text"]
            hits = [h.unit.unit_id for h in eng.retrieve(_query(q, mid))]
            r_old = hits.index(ou) if ou and ou in hits else None
            r_new = hits.index(nu) if nu and nu in hits else None
            rows.append({"rank_old": r_old, "rank_new": r_new})
        n_subject = sum(1 for sid in sids
                        for u in eng.store.units_of_source(sid) if u.slots.subject)
        n_sup_total = sum(1 for sid in sids
                          for u in eng.store.units_of_source(sid)
                          if u.superseded_by)
        out[name] = {"n_pairs": len(sup), "marked_superseded": marked,
                     "marked_rate": round(marked / len(sup), 4) if sup else None,
                     "subject_resolved": n_subject,
                     "store_superseded_total": n_sup_total,
                     "ranking": summre_pair_scores(rows)}
        _release(eng)
    return out


def _query(text: str, project: str):
    from engine.schemas import Query
    return Query(text=text, project=project, mode="current")


def _refuse_spent_test(root, split: str) -> None:
    """The B-13 read against the test set is spent (§9.5). Once a test-split
    reach artefact exists, a second one would be tuning on the holdout."""
    if split != "test":
        return
    done = list((root / "runs" / "summre_reach").glob("*test*"))
    done += list((root / "runs" / "frbench").glob("summre_*_test*"))
    if done:
        raise RuntimeError(
            f"the reach TEST split is spent and stays spent; refusing because "
            f"these artefacts exist: {[str(p) for p in done]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--gate", action="store_true",
                    help="status firing only (no gold pair scoring)")
    ap.add_argument("--run", action="store_true", help="reach run (scores gold pairs)")
    ap.add_argument("--arms", default="rules,v2",
                    help="comma list for --run: " + ",".join(ARM_BUILDERS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    if not (args.gate or args.run):
        ap.error("choose --gate and/or --run")

    root = repo_root()
    _refuse_spent_test(root, args.split)
    meetings = json.loads((root / "data/external/summre_test/meetings.json")
                          .read_text(encoding="utf-8"))
    gold_path = root / "data/gold" / f"summre_temporal_{args.split}.json"
    g = json.loads(gold_path.read_text(encoding="utf-8"))
    if g.get("split") != args.split:
        raise ValueError(f"{gold_path.name} declares split={g.get('split')!r}, "
                         f"asked for {args.split!r} — wrong file or wrong flag")
    mids = sorted(g["meetings"])
    stamp = date.today().isoformat()
    tmp_dir = root / "runs" / "summre_reach" / f"work-{args.split}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"split": args.split, "meetings": mids, "date": stamp}

    emb = None
    try:
        emb = _shared_embedder(args.device)
        if args.gate:
            out["status_firing"] = status_firing(args.device, meetings, mids,
                                                 tmp_dir, emb)
        if args.run:
            out["reach"] = run_reach(args.device, meetings, mids, gold_path,
                                     tmp_dir, emb,
                                     arms=tuple(a for a in args.arms.split(",") if a))
    finally:
        if emb is not None:
            _release(emb=emb)

    mode = "+".join(m for m in (args.gate and "gate", args.run and "run") if m)
    arms_tag = ("" if not args.run or args.gate
                else "-" + "-".join(a for a in args.arms.split(",") if a))
    dest = (root / "runs" / "summre_reach"
            / f"reach-{args.split}-{stamp}-{mode}{arms_tag}.json")
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False)[:4000], flush=True)
    print("wrote", dest, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
