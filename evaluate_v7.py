#!/usr/bin/env python3
"""Unified v7 evaluation: every method on the same unseen-project, held-out-phrase test split.

Also prints per-project metrics for the unseen test discussions. Writes
artifacts/v7_benchmark.json.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

import memory_engine as me

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
DATA = ROOT / "data"
METHODS = ["bm25", "bge_only", "colbert", "rexu_v2_1", "rexu_v2_2",
           "graphrag_ft_router"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    engine = me.MeetingMemoryEngine(device=args.device, batch_size=args.batch_size)
    result = engine.benchmark("synthetic", split=args.split, methods=METHODS)
    mid_project = {row["meeting_id"]: row["project"]
                   for rows in json.loads((DATA / "french_synthetic_meetings.json").read_text()).values()
                   for row in rows if row.get("split") == args.split}
    per_project = {}
    for method in METHODS:
        data = result["methods"].get(method, {})
        by_project = collections.defaultdict(list)
        for row in data.get("rankings", []):
            by_project[mid_project.get(row["meeting_id"], "?")].append(row["rank"])
        per_project[method] = {
            project: {"Recall@1": float(np.mean([r <= 1 for r in ranks])),
                      "MRR@10": float(np.mean([1 / r if r <= 10 else 0 for r in ranks])),
                      "n": len(ranks)}
            for project, ranks in by_project.items()
        }
    payload = {"split": args.split, "methods": result["methods"], "per_project": per_project}
    ART.mkdir(exist_ok=True)
    (ART / "v7_benchmark.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"v7 {args.split}: {result['query_count']} queries")
    print(f"{'method':20} {'R@1':>6} {'AnyR@1':>7} {'R@5':>6} {'AnyR@5':>7} {'MRR':>7} {'nDCG':>7}")
    for method in METHODS:
        m = result["methods"].get(method, {}).get("metrics")
        if not m:
            print(f"{method:20} {result['methods'].get(method, {}).get('status', 'UNAVAILABLE')}")
            continue
        print(f"{method:20} {m['Recall@1']:6.3f} {m.get('Any Recall@1', 0):7.3f} "
              f"{m['Recall@5']:6.3f} {m.get('Any Recall@5', 0):7.3f} {m['MRR@10']:7.3f} {m['nDCG@5']:7.3f}")
    print("\nPer unseen project (Recall@1 / MRR@10):")
    for method in ("bge_only", "colbert", "rexu_v2_1", "rexu_v2_2",
                   "graphrag_ft_router"):
        print(" ", method, {p: (round(v["Recall@1"], 3), round(v["MRR@10"], 3))
                            for p, v in per_project.get(method, {}).items()})
    print("\nSaved", ART / "v7_benchmark.json")


if __name__ == "__main__":
    main()
