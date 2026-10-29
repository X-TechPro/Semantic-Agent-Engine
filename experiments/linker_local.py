"""Pair-local linker round (D-046): reach and precision on real French threads.

The measured problem this instrument exists to answer: the shipped linker needs
`slots.subject` on both sides of a pair, and real ASR resolves one for ~2% of
units, so recall is capped at zero before any model is asked anything. The
round adds (a) a deterministic coarse `ThreadIndex` supplying pair-level slot
identity and (b) value witnesses that read spoken French numbers/negations
(`link/values.py`). This tool scores the three paths on the SAME gold pairs:

  shipped     — rules proposer + guards, no thread (the current default);
  pair_local  — the same guards with `same_thread=True` on ThreadIndex mates;
  drawer      — pair_local restricted to the hand-labelled drawers (upper bound
                of what a perfect threader would give the mechanism).

Split discipline: TRAIN + SEL only. The sealed holdout was read exactly once by
D-039 and is never read again here; the spent external test is refused in code.

Bars, declared BEFORE the run (this docstring is the declaration):
  L1 reach    >= 0.50  (gold old units that reach the guards at all)
  L2 precision >= 0.70 on predicted `supersedes`
  L3 recall   >= 0.30 on gold `supersedes`
All three must clear for `link.thread_context` to be promoted. Anything else:
the config stays False, the artefact says why.

Run: PYTHONPATH=src python -m engine.eval.linker_local [--reader rules|v2]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from engine.config import load_config, repo_root
from engine.link.rules import propose
from engine.link.supersede import evaluate, same_slot
from engine.link.thread import ThreadIndex
from engine.schemas import Slots, Span, Unit, utcnow
from engine.slots import rules as slot_rules

STAMP = date(2021, 10, 5)
SPLITS = ("train", "sel")
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
BARS = {"L1_reach>=0.50": 0.50, "L2_precision>=0.70": 0.70, "L3_recall>=0.30": 0.30}

GOLD_FILES = {"train": "summre_temporal_train.json",
              "sel": "summre_temporal_sel.json",
              "holdout": "summre_temporal_holdout.json"}


def _meetings(root: Path) -> dict[str, list[dict]]:
    data = json.loads((root / "data/external/summre_test/meetings.json")
                      .read_text(encoding="utf-8"))
    return {m: (v["segments"] if isinstance(v, dict) else v)
            for m, v in data.items()}


def _gold(root: Path, split: str) -> dict:
    if split == "holdout":
        raise ValueError("D-039 read the sealed holdout once; this tool refuses a "
                         "second read — use train+sel")
    doc = json.loads((root / "data/gold" / GOLD_FILES[split])
                     .read_text(encoding="utf-8"))
    out = {}
    for mid, mm in doc["meetings"].items():
        if mid in SPENT:
            raise ValueError(f"{mid} is the spent external test — never load it")
        sup = {(int(r["old"]), int(r["new"]))
               for th in mm["threads"] for r in th["relations"]
               if r["relation"] == "supersedes"}
        other = {(int(r["old"]), int(r["new"]))
                 for th in mm["threads"] for r in th["relations"]
                 if r["relation"] != "supersedes"}
        out[mid] = {"sup": sup, "other": other}
    return out


def _unit(mid: str, idx: int, seg: dict, reader, roster) -> Unit:
    text = seg["text"].strip()
    u = Unit(unit_id=f"{mid}#{idx}", source_id=f"summre:{mid}",
             source_type="meeting", source_date=STAMP, ingested_at=utcnow(),
             text=text, sentences=[Span.whole(text)],
             speaker=seg.get("speaker"), project=mid.split("_")[0],
             turn_index=idx, kind="info", slots=Slots())
    reading = reader.read(u)
    u = u.model_copy(update={"kind": reading.kind, "type": reading.type,
                             "status": reading.status,
                             "valid_from": STAMP})
    u = u.model_copy(update={"slots": slot_rules.extract(u, roster)})
    return u


def _prf(pred: set, gold: set) -> dict:
    tp = len(pred & gold)
    return {"n_pred": len(pred), "n_gold": len(gold), "tp": tp,
            "precision": round(tp / len(pred), 4) if pred else 0.0,
            "recall": round(tp / len(gold), 4) if gold else None}


def run(reader_backend: str = "rules") -> dict:
    root = repo_root()
    meetings = _meetings(root)
    cfg = load_config(overrides={
        "reader": ({"backend": "model", "model_dir": str(root / "models/reader_v2")}
                   if reader_backend == "v2" else {"backend": "rules"}),
        "slots": {"subject_resolver": "rules"}})
    from engine.read.infer import build_reader
    reader, backend = build_reader(cfg)

    gold = {s: _gold(root, s) for s in SPLITS}
    gold_sup = {(mid, o, n) for s in SPLITS for mid, d in gold[s].items()
                for (o, n) in d["sup"]}
    gold_other = {(mid, o, n) for s in SPLITS for mid, d in gold[s].items()
                  for (o, n) in d["other"]}
    mids = sorted({mid for s in SPLITS for mid in gold[s]})
    if set(mids) & SPENT:
        raise ValueError("a scored meeting is the spent external test")

    pred = {"shipped": set(), "pair_local": set(), "drawer": set()}
    guard = Counter()
    gold_reach = {"shipped": 0, "pair_local": 0, "drawer": 0}

    for mid in mids:
        segs = meetings[mid]
        units = [_unit(mid, i, s, reader, set()) for i, s in enumerate(segs)]
        keys = {}
        kp = root / "data/gold" / f"summre_subjects_{mid}.json"
        if kp.exists():
            kg = json.loads(kp.read_text(encoding="utf-8"))
            keys = {int(i): v for i, v in (kg.get("fine_keys") or kg.get("keys")
                                           or {}).items() if v}
        thread = ThreadIndex(window=8)
        for i, new in enumerate(units):
            mates = thread.context(new)
            mate_ids = {u.unit_id for u in mates}
            # --- pair_local: every ThreadIndex mate is judged as a candidate
            for old in mates:
                rel, g = evaluate(new, old, propose(new, old, thread=True),
                                  same_thread=True)
                guard[g] += 1
                if rel == "supersedes":
                    pred["pair_local"].add((mid, int(old.turn_index), i))
            # --- drawer: same, but only mates sharing the gold drawer (upper bound)
            for old in mates:
                ki, ko = keys.get(i), keys.get(int(old.turn_index))
                if ki and ki == ko:
                    rel, g = evaluate(new, old, propose(new, old, thread=True),
                                      same_thread=True)
                    if rel == "supersedes":
                        pred["drawer"].add((mid, int(old.turn_index), i))
            # --- shipped: the gold drawer space, guards without a thread
            prev = [units[j] for j in range(i) if keys.get(j) and keys.get(j) == keys.get(i)]
            for old in prev:
                rel, g = evaluate(new, old, propose(new, old))
                if rel == "supersedes":
                    pred["shipped"].add((mid, int(old.turn_index), i))
            thread.add(new)

        sup_here = set().union(*(gold[s].get(mid, {}).get("sup", set())
                                for s in SPLITS))
        for (o, n) in sup_here:
            if o >= len(units) or n >= len(units):
                continue
            if same_slot(units[n], units[o]):
                gold_reach["shipped"] += 1
            # pair_local reach: old was a thread mate when new arrived
            replay = ThreadIndex(window=8)
            for j in range(n):
                replay.add(units[j])
            if any(int(u.turn_index) == o for u in replay.context(units[n])):
                gold_reach["pair_local"] += 1
            if keys.get(o) and keys.get(o) == keys.get(n):
                gold_reach["drawer"] += 1

    out: dict = {
        "date": date.today().isoformat(),
        "mechanism": ("deterministic coarse thread (resolved subject else "
                      "attribute/date/amount/number) + spoken-number and negation "
                      "value witnesses; guards unchanged, G1 substituted only when "
                      "the subject key is MISSING"),
        "reader_backend": backend,
        "splits": list(SPLITS),
        "holdout": "not read (D-039 spent its single read)",
        "n_gold_supersedes": len(gold_sup),
        "n_gold_other": len(gold_other),
        "reach": {k: {"gold_pairs_reaching_guards": v,
                      "rate": round(v / len(gold_sup), 4) if gold_sup else 0.0}
                  for k, v in gold_reach.items()},
        "guards": dict(sorted(guard.items())),
        "bars_declared_before_run": BARS,
    }
    for name in ("shipped", "pair_local", "drawer"):
        m = _prf(pred[name], gold_sup)
        fps = pred[name] - gold_sup
        m["fp_labeled_other"] = sum(1 for p in fps if p in gold_other)
        m["fp_unlabeled"] = sum(1 for p in fps if p not in gold_other)
        out[name] = m
    out["cleared"] = {
        "L1": all(out["reach"][k]["rate"] >= BARS["L1_reach>=0.50"]
                  for k in ("pair_local",)),
        "L2": out["pair_local"]["precision"] >= BARS["L2_precision>=0.70"],
        "L3": (out["pair_local"]["recall"] or 0) >= BARS["L3_recall>=0.30"],
    }
    out["verdict"] = ("PROMOTE" if all(out["cleared"].values()) else
                      "REFUSED — link.thread_context stays False")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", choices=("rules", "v2"), default="rules")
    args = ap.parse_args(argv)
    out = run(args.reader)
    dest = repo_root() / "runs" / "linker_local"
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"result-{out['date']}-{out['reader_backend'].split('(')[0]}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
