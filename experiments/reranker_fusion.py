"""Small fusion reranker: candidate-pool rank + query-context cross-encoder.

No 16-feature model. Tune one mixing weight on training meetings only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings
from engine.link.cross_encoder_linker import Candidate, CrossEncoderLinker


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/reranker_fusion/result.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    meetings = load_meetings(root)
    stage = json.loads((root / "runs/continuity_stages_v2/result.json").read_text())
    ce = CrossEncoderLinker("BAAI/bge-reranker-v2-m3", device="cuda")
    pairs = []
    for r in stage["rows"]:
        mid, old, new = r["meeting"], r["old"], r["new"]
        cand = r["candidate_turns"]
        if not cand:
            continue
        turns = meetings[mid]
        qctx = " || ".join(str(turns[k].get("text") or "")
                           for k in range(max(0, new - 2), new + 1))
        cs = [Candidate(turn=j, text=str(turns[j].get("text") or ""),
                        speaker=str(turns[j].get("speaker") or ""),
                        timestamp=str(turns[j].get("timestamp") or ""))
              for j in cand]
        ce_s = ce.scores(qctx, cs, turns, style="plain")
        ce_m = ce.scores(qctx, cs, turns, style="metadata_context")
        pairs.append({"mid": mid, "old": old, "new": new, "cand": cand,
                      "ce": ce_s, "ce_m": ce_m})
    alphas = (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3)
    def rank_for(p, alpha, key="ce"):
        c = np.asarray(p[key], dtype=float)
        c = (c - c.mean()) / max(c.std(), 1e-9)
        # lower candidate rank is better
        s = -np.asarray([1.0 / (1 + i) for i in range(len(p["cand"]))]) + alpha * c
        order = np.argsort(-s)
        return p["cand"].index(p["old"]) + 1 if p["old"] in p["cand"] else None, order
    folds = []
    for held in sorted({p["mid"] for p in pairs}):
        tr = [p for p in pairs if p["mid"] != held]
        te = [p for p in pairs if p["mid"] == held]
        best = None
        for alpha in alphas:
            got = []
            for p in tr:
                r, _ = rank_for(p, alpha)
                if r is not None:
                    got.append(r)
            at1 = sum(r == 1 for r in got)
            at5 = sum(r <= 5 for r in got)
            key = (at1, at5)
            if best is None or key > best[0]:
                best = (key, alpha)
        alpha = best[1]
        local = []
        for p in te:
            r, _ = rank_for(p, alpha)
            if r is not None:
                local.append(r)
        folds.append({"held": held, "alpha": alpha, "n": len(local),
                      "at_1": sum(r == 1 for r in local),
                      "at_5": sum(r <= 5 for r in local),
                      "mrr": round(sum(1.0 / r for r in local) / len(local), 4)
                      if local else 0})
    out = {"n": sum(f["n"] for f in folds),
           "at_1": sum(f["at_1"] for f in folds),
           "at_5": sum(f["at_5"] for f in folds),
           "folds": folds}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "folds"}, indent=1))
    for f in folds:
        print(f)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
