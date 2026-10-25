"""Build and print decision nodes for one finished session (read-only)."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from engine.config import load_config
from engine.read.baseline_rules import RuleReader
from engine.schemas import Source, Turn
from engine.segment.chunklet_adapter import Segmenter
from engine.threads import build_session_nodes


def annotate(unit, reader):
    r = reader.read(unit)
    return unit.model_copy(update={"kind": r.kind, "type": r.type,
                                   "status": r.status})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", default="data/external/summre_test/meetings.json")
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--as-of", default=None, help="YYYY-MM-DD")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    raw = json.loads(Path(args.meetings).read_text(encoding="utf-8"))[args.meeting]
    cfg = load_config()
    src = Source(source_id=args.meeting, source_type="meeting",
                 source_date=date(2026, 10, 23), project=args.meeting,
                 turns=[Turn(speaker=u["speaker"], text=u["text"], turn_index=i)
                        for i, u in enumerate(raw)])
    units = [annotate(u, RuleReader()) for u in Segmenter(cfg).segment(src)]
    nodes = build_session_nodes(units)
    day = date.fromisoformat(args.as_of) if args.as_of else None
    out = Path(args.out or f"runs/session_nodes/{args.meeting}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "meeting": args.meeting,
        "source_date": str(units[0].source_date) if units else None,
        "n_units": len(units),
        "nodes": [{
            "node_id": n.node_id, "project": n.project, "topic": n.topic,
            "children": [c.unit_id for c in n.children],
            "turns": [c.turn_index for c in n.children],
            "dates": [str(c.source_date) for c in n.children],
            "latest": ({"unit_id": n.latest.unit_id,
                        "turn": n.latest.turn_index,
                        "date": str(n.latest.source_date),
                        "text": n.latest.text} if n.latest else None),
        } for n in nodes],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.meeting}: {len(units)} units -> {len(nodes)} decision nodes -> {out}")
    for n in nodes:
        latest = n.latest
        print(f"\n[{n.node_id}] children={len(n.children)}")
        for c in n.children:
            mark = "*" if c is latest else " "
            print(f"  {mark} ({c.turn_index}) {c.text[:110]}")
        if day is not None:
            at = n.as_of(day)
            print(f"  as-of {day}: "
                  f"{('(' + str(at.turn_index) + ') ' + at.text[:100]) if at else '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
