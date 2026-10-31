#!/usr/bin/env python3
"""Run BM25 / custom dense / feature-aware / French ME5 / multilingual E5 on one dataset."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
from memory_engine import MeetingMemoryEngine

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="synthetic", help="Dataset slug already imported into this app process (synthetic by default).")
    ap.add_argument("--meetings-json", help="Import a JSON/JSONL corpus before benchmarking.")
    ap.add_argument("--queries-json", help="Optional query/evidence JSON for labelled evaluation.")
    ap.add_argument("--name", default="Benchmark custom", help="Display name for --meetings-json.")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "custom"])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--batch-size", type=int, default=64, help="Embedding batch size; 64 is a safe RTX 5060 8GB starting point.")
    ap.add_argument("--methods", nargs="+", default=["bm25", "dense", "feature_aware", "french_me5", "multilingual_e5"],
                    choices=["bm25", "dense", "feature_aware", "french_me5", "multilingual_e5"])
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    engine = MeetingMemoryEngine(device=args.device, batch_size=args.batch_size)
    slug = args.dataset
    if args.meetings_json:
        path = Path(args.meetings_json)
        query_bytes = Path(args.queries_json).read_bytes() if args.queries_json else None
        imported = engine.import_dataset(path.name, path.read_bytes(), args.name, query_bytes,
                                         Path(args.queries_json).name if args.queries_json else None)
        slug = imported["slug"]
        print(f"Imported {slug}: {imported['meeting_count']} meetings / {imported['message_count']} messages")
    start = time.time()
    result = engine.benchmark(slug, args.split, args.methods)
    result["runtime_seconds"] = round(time.time() - start, 3)
    result["runtime_device"] = engine.device
    result["model_status"] = engine.model_status()
    output = Path(args.output) if args.output else ART / f"comparison_{slug}_{args.split}_{int(time.time())}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result.get("status"), "dataset": slug, "split": args.split,
                      "output": str(output), "runtime_seconds": result["runtime_seconds"],
                      "methods": {k: v.get("metrics", v) for k, v in result.get("methods", {}).items()}},
                     ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "OK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
