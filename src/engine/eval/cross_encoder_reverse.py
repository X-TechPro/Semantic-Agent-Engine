"""Cross-encoder rerank on the corrected new->old candidate pool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings
from engine.link.cross_encoder_linker import Candidate, CrossEncoderLinker


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--out", default=str(repo_root() / "runs/cross_encoder/reverse.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(root)
    stage = json.loads((root / "runs/continuity_stages_v2/result.json").read_text())
    ce = CrossEncoderLinker(args.model, device="cuda")
    rows = []
    for r in stage["rows"]:
        mid, old, new = r["meeting"], r["old"], r["new"]
        turns = meetings[mid]
        cand = [Candidate(turn=j, text=turns[j]["text"],
                          speaker=str(turns[j].get("speaker") or ""),
                          timestamp=str(turns[j].get("timestamp") or ""))
                for j in r["candidate_turns"]]
        rec = {"meeting": mid, "old": old, "new": new,
               "candidate_rank": r["candidate_rank"],
               "scorer_rank": r["rerank_rank"]}
        qctx = " || ".join(str(turns[k].get("text") or "")
                           for k in range(max(0, new - 2), new + 1))
        for style in ("plain", "metadata", "context", "metadata_context"):
            order = ce.rank(turns[new]["text"], cand, turns, style=style)
            rec[f"ce_{style}"] = order.index(old) + 1 if old in order else None
            order_ctx = ce.rank(qctx, cand, turns, style=style)
            rec[f"ce_qctx_{style}"] = order_ctx.index(old) + 1 if old in order_ctx else None
        rows.append(rec)
    def m(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return {"at_1": sum(v == 1 for v in vals),
                "at_5": sum(v <= 5 for v in vals),
                "covered": len(vals),
                "mrr": round(sum(1.0 / v for v in vals) / len(rows), 4)}
    out = {"model": args.model, "direction": "new query -> old antecedent",
           "n": len(rows), "candidate_recall": sum(r["candidate_rank"] is not None
                                                    for r in rows),
           "metrics": {k: m(k) for k in ("scorer_rank", "ce_plain", "ce_metadata",
                                         "ce_context", "ce_metadata_context",
                                         "ce_qctx_plain", "ce_qctx_metadata",
                                         "ce_qctx_context", "ce_qctx_metadata_context")},
           "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
