"""State-layer read on the graph-first corpus: state vs latest-sentence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.config import repo_root
from engine.memory.event_state import EventStateTable, StateMention


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/latent_v1")
    ap.add_argument("--out", default=str(repo_root() / "runs/event_state/synthetic.json"))
    args = ap.parse_args(argv)
    root = repo_root()
    rows = []
    for split in ("train", "validation", "test"):
        d = root / args.data / split
        turns = {t["turn_id"]: t for t in read_jsonl(d / "turns.jsonl")}
        events = read_jsonl(d / "events.jsonl")
        mentions = read_jsonl(d / "mentions.jsonl")
        table = EventStateTable()
        ordered = sorted(mentions, key=lambda m: (turns[m["turn_id"]]["turn_index"],
                                                  m["mention_id"]))
        for m in ordered:
            table.apply(StateMention(m["mention_id"], m["event_id"], m["value"],
                                     m["status"], m["turn_id"], m.get("evidence", "")))
        for ev in events:
            if ev["type"] == "hard_negative":
                continue
            st = table.current(ev["event_id"])
            latest = ordered[-1] if ordered else None
            ev_m = [m for m in ordered if m["event_id"] == ev["event_id"]]
            latest = ev_m[-1] if ev_m else None
            state_ok = bool(st and st.current_value == ev["current_value"]
                            and st.current_status == ev["current_status"])
            latest_ok = bool(latest and latest["value"] == ev["current_value"]
                             and latest["status"] == ev["current_status"])
            unresolved_ok = bool(st and (st.current_status == "open") == bool(ev["unresolved"]))
            rows.append({"split": split, "event_id": ev["event_id"],
                         "gold_value": ev["current_value"],
                         "gold_status": ev["current_status"],
                         "unresolved": bool(ev["unresolved"]),
                         "state_value": st.current_value if st else None,
                         "state_status": st.current_status if st else None,
                         "state_ok": state_ok, "latest_ok": latest_ok,
                         "unresolved_ok": unresolved_ok,
                         "history": [{"value": h.value, "status": h.status}
                                     for h in (st.history if st else [])]})
    n = len(rows)
    out = {"data": args.data, "n_events": n,
           "state_ok": sum(r["state_ok"] for r in rows),
           "latest_ok": sum(r["latest_ok"] for r in rows),
           "unresolved_ok": sum(r["unresolved_ok"] for r in rows),
           "state_rate": round(sum(r["state_ok"] for r in rows) / n, 4) if n else 0,
           "latest_rate": round(sum(r["latest_ok"] for r in rows) / n, 4) if n else 0,
           "rows": rows}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
