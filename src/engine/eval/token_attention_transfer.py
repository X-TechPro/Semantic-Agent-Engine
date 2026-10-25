"""Does the token signal transfer between meetings? (D-048 follow-up)

Reads `runs/token_attention/result-*.json` (which stores the top signals'
per-row scores) and answers the owner's question — "what if we train more?" —
with two views that do not need another encoder pass:

  per_meeting  the selected signal's AUC inside each meeting, one by one;
  lomo         leave one meeting out: choose the signal on the OTHER meetings,
               score it on the held-out one.

If the spread is between meetings rather than within them, more epochs on the
same five meetings cannot fix it — the lever is more labelled meetings.

Run: PYTHONPATH=src python -m engine.eval.token_attention_transfer
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

import numpy as np

from engine.config import repo_root


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    a = np.concatenate([np.array(pos), np.array(neg)])
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    ranks[order] = np.arange(1, len(a) + 1)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return round(float(u / (len(pos) * len(neg))), 4)


def _rows_for(rows: list[dict], name: str) -> tuple[list[float], list[float]]:
    pos = [r["scores"][name] for r in rows
           if r["y"] == 1 and r["scores"].get(name) is not None]
    neg = [r["scores"][name] for r in rows
           if r["y"] == 0 and r["scores"].get(name) is not None]
    return pos, neg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default=None)
    args = ap.parse_args(argv)
    root = repo_root()
    src = args.result
    if src is None:
        hits = sorted((root / "runs/token_attention").glob("result-*.json"),
                      key=lambda p: p.stat().st_mtime)
        if not hits:
            print("no token_attention result artefact — run the scorer first")
            return 1
        src = str(hits[-1])
    d = json.loads(open(src, encoding="utf-8").read())
    rows = d.get("rows_top10") or []
    if not rows:
        print("artefact has no rows_top10 — re-run engine.eval.token_attention")
        return 1
    selected = (d.get("selected_on_train") or {}).get("name")
    names = list(rows[0]["scores"])
    mids = sorted({r["mid"] for r in rows})

    per_meeting = {}
    for mid in mids:
        sub = [r for r in rows if r["mid"] == mid]
        pos, neg = _rows_for(sub, selected)
        per_meeting[mid] = {"auc": auc(pos, neg), "positives": len(pos),
                            "negatives": len(neg)}
    vals = [v["auc"] for v in per_meeting.values() if v["auc"] is not None]

    lomo = {}
    for mid in mids:
        held = [r for r in rows if r["mid"] == mid]
        other = [r for r in rows if r["mid"] != mid]
        if len({r["y"] for r in held}) < 2:
            continue
        scores = {}
        for name in names:
            pos, neg = _rows_for(other, name)
            scores[name] = auc(pos, neg)
        best = max(scores, key=lambda n: scores[n] if scores[n] is not None else -1)
        pos, neg = _rows_for(held, best)
        lomo[mid] = {"chosen": best, "auc": auc(pos, neg)}
    lval = [v["auc"] for v in lomo.values() if v["auc"] is not None]

    out = {
        "date": date.today().isoformat(), "source": src,
        "selected_signal": selected,
        "per_meeting": per_meeting,
        "per_meeting_mean": round(float(np.mean(vals)), 4) if vals else None,
        "per_meeting_sd": round(float(np.std(vals)), 4) if vals else None,
        "per_meeting_range": [min(vals), max(vals)] if vals else None,
        "leave_one_meeting_out": lomo,
        "lomo_mean": round(float(np.mean(lval)), 4) if lval else None,
        "reading": ("the spread is BETWEEN meetings, not within them: training "
                    "longer on the same meetings fits the ones that already work "
                    "and leaves the rest at chance"),
    }
    dest = root / "runs" / "token_attention"
    p = dest / f"transfer-{out['date']}.json"
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
