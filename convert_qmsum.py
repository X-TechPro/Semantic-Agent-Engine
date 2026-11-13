#!/usr/bin/env python3
"""Convert the QMSum test split into the v10 memory format (meetings + queries).

Usage: python convert_qmsum.py /path/to/QMSum/repo/data
Source: https://github.com/Yale-LILY/QMSum (MIT). Raw data is not copied into the repo;
only the converted files data/qmsum_meetings.json and data/qmsum_queries.json are written.

Mapping
- one QMSum meeting -> one memory meeting; each speaker turn -> one message
  (message_id = turn index + 1, chunk_id = "<meeting>:<message_id:03d>").
- QMSum specific queries -> queries; relevant_text_span [start, end] (turn indices,
  treated as inclusive) -> target_chunk_ids. General (summary-style) queries are skipped.
- QMSum has no timestamps or dates: time is synthesized from the turn index and
  meeting_date is left empty (the loader defaults it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOMAINS = ("Academic", "Committee", "Product")


def domain_of(raw_root: Path, stem: str) -> str:
    for dom in DOMAINS:
        if (raw_root / dom / "all" / f"{stem}.json").exists():
            return dom
    return "unknown"


def main(raw_data: Path) -> None:
    test_dir = raw_data / "ALL" / "test"
    files = sorted(test_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no test meetings under {test_dir}")

    meetings, queries = [], []
    skipped = 0
    for path in files:
        stem = path.stem
        src = json.loads(path.read_text(encoding="utf-8"))
        mid = f"qmsum-{stem}"
        dom = domain_of(raw_data, stem)
        turns = src["meeting_transcripts"]
        messages = []
        for i, turn in enumerate(turns):
            messages.append({
                "id": f"{mid}:{i + 1:03d}",
                "message_id": i + 1,
                "time": f"{(i // 60) % 24:02d}:{i % 60:02d}",
                "speaker": str(turn.get("speaker", "Inconnu")).strip() or "Inconnu",
                "text": str(turn.get("content", "")).strip(),
            })
        meetings.append({
            "meeting_id": mid,
            "project_id": f"qmsum-{dom.lower()}",
            "project": f"QMSum · {dom}",
            "title": f"QMSum {stem} ({dom})",
            "style": "qmsum",
            "split": "test",
            "messages": messages,
        })
        for j, q in enumerate(src["specific_query_list"]):
            spans = q.get("relevant_text_span") or []
            gold: list[str] = []
            for start, end in spans:
                lo, hi = int(start), int(end)
                if lo < 0 or hi >= len(turns) or lo > hi:
                    skipped += 1
                    continue
                gold.extend(f"{mid}:{k + 1:03d}" for k in range(lo, hi + 1))
            gold = list(dict.fromkeys(gold))
            if not gold:
                skipped += 1
                continue
            queries.append({
                "query_id": f"{mid}-q{j + 1}",
                "meeting_id": mid,
                "split": "test",
                "type": "qmsum_specific",
                "query": str(q["query"]).strip(),
                "answer": str(q.get("answer", "")).strip(),
                "evidence_chunk_ids": gold,
                "target_chunk_ids": gold,
            })

    out_m = HERE / "data" / "qmsum_meetings.json"
    out_q = HERE / "data" / "qmsum_queries.json"
    out_m.write_text(json.dumps({
        "dataset": "qmsum-test",
        "description": "QMSum test split (Yale-LILY, MIT), converted to the v10 memory format.",
        "meetings": meetings,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    out_q.write_text(json.dumps({"queries": queries}, ensure_ascii=False, indent=1), encoding="utf-8")
    turns_total = sum(len(m["messages"]) for m in meetings)
    print(f"meetings={len(meetings)} turns={turns_total} queries={len(queries)} skipped_spans={skipped}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]))
