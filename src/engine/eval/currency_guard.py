"""Does the unresolved-currency guard refuse instead of guessing? (D-047)

Measured on the hand-labelled SUMM-RE train+sel meetings (40 gold supersede
pairs; the sealed holdout was spent by D-039 and is never read here). For every
gold pair the query is the OLD statement's own text with mode=current — the
case where the system must NOT present that version as the one in force once a
newer one exists and no relation links them.

Two answers are built over the same hits:
  unguarded — `competing_versions` patched to return [] (the old behaviour);
  guarded   — the shipped path.
The artefact records whether the stale unit was cited at all, whether the newer
one was even retrieved, and whether the guard refused.

Control sample: non-gold segments asked the same way. On real speech some of
those genuinely contain a disagreement, so their refusal rate is reported as
information, not as an over-refusal bar; the hard no-regression check is the
demo/e2e suite (19/19 + tiers green), where the links are correct.

Two earlier mechanisms were measured and refused on this same corpus before the
shipped one:
  A. full refusal whenever any disagreeing group exists among the cited hits —
     refused: caveat/refusal on only 11/22 stale cases AND on 34.8% of control
     segments (too loose where it should be quiet, too narrow where it should
     speak);
  B. A plus "the lead must belong to the group" — refused: 0/22, because the
     two statements do not share a topic or an attribute, so the group cannot
     form (the D-046 wall again).
The shipped mechanism needs no topic identity: if the answer cannot ground the
unit's currency (no resolved subject AND no assertive status), it says so.

A third variant — caveat only when the LEAD is ungrounded — was measured and
refused too: 25/40 (0.625), because the answer's lead is often not the stale
unit being asked about. The shipped rule is claim-level: if ANY claim the answer
shows cannot be grounded, it says so.

Bars declared BEFORE this run (docstring = declaration):
  C1  an HONEST answer (caveat or refusal — both name the uncertainty) on
      >= 0.95 of the pairs whose unguarded answer cited the old unit
  C2  the newer unit is retrieved on >= 0.50 of the pairs — reported as context,
      NOT a gate: the caveat does not need to see the newer statement
  C3  the demo/e2e no-regression suites stay green AND no expected demo answer
      is caveated (enforced in `eval/e2e.py`, not trusted here)
If C1 fails the caveat is too narrow and `agent.currency_guard` goes back to
False. If C1 and C3 hold, it ships ON — naming the uncertainty is the safer
error, and "signal missing/ambiguous information" is a stated requirement.

Run: PYTHONPATH=src python -m engine.eval.currency_guard [--limit-meetings N]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import patch

from engine.config import load_config, repo_root

SPLITS = ("train", "sel")
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}
GOLD_FILES = {"train": "summre_temporal_train.json",
              "sel": "summre_temporal_sel.json"}
BARS = {"C1_caveat>=0.95": 0.95, "C2_new_in_hits>=0.50": 0.50}


def _meetings(root: Path) -> dict[str, list[dict]]:
    data = json.loads((root / "data/external/summre_test/meetings.json")
                      .read_text(encoding="utf-8"))
    return {m: (v["segments"] if isinstance(v, dict) else v)
            for m, v in data.items()}


def _gold(root: Path) -> dict[str, dict]:
    out = {}
    for split in SPLITS:
        doc = json.loads((root / "data/gold" / GOLD_FILES[split])
                         .read_text(encoding="utf-8"))
        for mid, mm in doc["meetings"].items():
            if mid in SPENT:
                raise ValueError(f"{mid} is the spent external test")
            sup = {(int(r["old"]), int(r["new"]))
                   for th in mm["threads"] for r in th["relations"]
                   if r["relation"] == "supersedes"}
            out.setdefault(mid, set()).update(sup)
    return out


def run(limit_meetings: int | None = None) -> dict:
    from engine.agent.answer import build_answer
    from engine.agent.currency import CURRENCY_CAVEAT, UNRESOLVED_REASON
    from engine.engine import Engine
    from engine.schemas import Query
    from eval.frbench import build_meeting_source

    root = repo_root()
    meetings = _meetings(root)
    gold = _gold(root)
    mids = sorted(gold)
    if limit_meetings:
        mids = mids[:limit_meetings]

    cfg = load_config(overrides={
        "reader": {"backend": "rules"},
        "slots": {"subject_resolver": "rules"},
        "embed": {"backend": "hash"}})
    thr = float(cfg["agent"]["no_answer_threshold"])
    ov = float(cfg["agent"]["no_answer_overlap"])

    pairs = {"n": 0, "old_in_hits": 0, "new_in_hits": 0,
             "unguarded_cited_old": 0, "caveated": 0, "refused": 0,
             "caveated_when_unguarded_cited_old": 0,
             "honest_when_unguarded_cited_old": 0,
             "still_confident_stale": 0}
    control = {"n": 0, "caveated": 0}
    per_meeting = {}

    with tempfile.TemporaryDirectory() as td:
        for mid in mids:
            db = Path(td) / f"{mid}.db"
            eng = Engine(cfg, path=str(db), linker_enabled=True)
            eng.ingest(build_meeting_source(mid, meetings[mid],
                                            on_date=date(2021, 10, 5)))
            uid = {(u.turn_index): u.unit_id
                   for u in eng.store.units_of_source(f"summre:{mid}")
                   if u.turn_index is not None}
            local = {"n": 0, "unguarded_cited_old": 0, "caveated": 0}
            for old, new in sorted(gold.get(mid, set())):
                if old not in uid or new not in uid:
                    continue
                q = Query(text=meetings[mid][old]["text"], project=mid,
                          mode="current")
                hits = eng.retrieve(q)
                ids = [h.unit.unit_id for h in hits]
                with patch("engine.agent.currency.competing_versions",
                           return_value=[]):
                    before = build_answer(hits, q, min_score=thr, overlap_floor=ov)
                after = build_answer(hits, q, min_score=thr, overlap_floor=ov)
                pairs["n"] += 1
                local["n"] += 1
                pairs["old_in_hits"] += int(uid[old] in ids)
                pairs["new_in_hits"] += int(uid[new] in ids)
                cited_old = any(uid[old] in c.unit_ids for c in before.claims)
                caveated = CURRENCY_CAVEAT in (after.answer or "")
                refused = after.no_answer_reason == UNRESOLVED_REASON
                pairs["unguarded_cited_old"] += int(cited_old)
                pairs["caveated"] += int(caveated)
                pairs["refused"] += int(refused)
                local["unguarded_cited_old"] += int(cited_old)
                local["caveated"] += int(caveated)
                if cited_old:
                    honest = caveated or refused
                    pairs["caveated_when_unguarded_cited_old"] += int(caveated)
                    pairs["honest_when_unguarded_cited_old"] += int(honest)
                    pairs["still_confident_stale"] += int(not honest)
            # control: non-gold segments asked the same way
            gold_here = {i for p in gold.get(mid, set()) for i in p}
            for i in range(0, len(meetings[mid]), 11):
                if i in gold_here or i not in uid:
                    continue
                if control["n"] >= 200:
                    break
                q = Query(text=meetings[mid][i]["text"], project=mid,
                          mode="current")
                ans = build_answer(eng.retrieve(q), q, min_score=thr,
                                   overlap_floor=ov)
                control["n"] += 1
                control["caveated"] += int(CURRENCY_CAVEAT in (ans.answer or ""))
            per_meeting[mid] = local
            eng.close()

    cited = pairs["unguarded_cited_old"]
    out = {
        "date": date.today().isoformat(),
        "splits": list(SPLITS),
        "holdout": "not read (D-039 spent its single read)",
        "meetings": mids,
        "mechanism": ("claim-level currency caveat (status `mention`/None = "
                      "currency unknown, whatever the subject) + explicit refusal "
                      "when two live value statements disagree"),
        "reader_backend": "rules",
        "embed_backend": "hash",
        "pairs": pairs,
        "control_segments": {
            **control,
            "caveat_rate": round(control["caveated"] / control["n"], 4)
                           if control["n"] else None},
        "per_meeting": per_meeting,
        "bars": {
            "C1_honest_on_stale>=0.95": (
                round(pairs["honest_when_unguarded_cited_old"] / cited, 4)
                if cited else None),
            "C2_new_in_hits>=0.50": (
                round(pairs["new_in_hits"] / pairs["n"], 4) if pairs["n"] else None),
        },
    }
    c1 = cited > 0 and (pairs["honest_when_unguarded_cited_old"] / cited) >= 0.95
    out["cleared"] = {"C1": c1}
    out["verdict"] = ("PROMOTE caveat ON" if c1 else
                      "REFUSED — currency_guard goes back to False on raw speech")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-meetings", type=int, default=None)
    args = ap.parse_args(argv)
    out = run(args.limit_meetings)
    dest = repo_root() / "runs" / "currency_guard"
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / f"result-{out['date']}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
