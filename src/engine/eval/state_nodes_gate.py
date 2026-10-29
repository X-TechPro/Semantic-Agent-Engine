"""State-aware v3 gate: topic nodes + explicit state-point nodes.

Measures the same six-meeting gold pairs as `node_view_gate`, plus the state
nodes' own current/at_turn. This is an additive view: the topic nodes are still
there, so a pair can be reached by either key.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from engine.config import repo_root
from engine.eval.decision_nodes import MEETINGS, gold_pairs, units_of
from engine.threads import build_session_nodes
from engine.threads.state_reader import StateReader


def metrics(*, state: bool) -> dict:
    pairs = gold_pairs()
    reader = StateReader() if state else None
    rows = []
    tot_pairs = tot_same = tot_at = tot_current = 0
    tot_units = tot_filed = 0
    mutated = 0
    for mid in MEETINGS:
        units = units_of(mid)
        nodes = build_session_nodes(units, state_reader=reader)
        by_turn: dict[int, list] = defaultdict(list)
        for n in nodes:
            for c in n.children:
                by_turn[c.turn_index].append(n)
                if c.valid_to is not None or c.superseded_by is not None:
                    mutated += 1
        gold = pairs.get(mid, set())
        same = at = current = 0
        for old, new in gold:
            onodes, nnodes = by_turn.get(old, []), by_turn.get(new, [])
            shared = [n for n in onodes if any(n is m for m in nnodes)]
            if shared:
                same += 1
            # at_turn/current are measured on the new turn's own nodes, not
            # only on the pairs that already share a node.
            at += int(any((n.at_turn(new) is not None
                           and n.at_turn(new).turn_index == new)
                          for n in nnodes))
            current += int(any(
                (n.current_turn == new)
                or (n.current_turn is None and n.latest is not None
                    and n.latest.turn_index == new)
                for n in nnodes))
        filed = sum(1 for u in units if u.turn_index in by_turn)
        tot_units += len(units); tot_filed += filed
        tot_pairs += len(gold); tot_same += same
        tot_at += at; tot_current += current
        rows.append({"meeting": mid, "units": len(units), "nodes": len(nodes),
                     "pairs": len(gold), "same_node": same,
                     "state_at_new": at, "current_is_new": current})
        print(f"{mid}: units={len(units)} nodes={len(nodes)} pairs={len(gold)} "
              f"same={same} at={at} current={current}")
    out = {"arch": "v3+state" if state else "v3", "meetings": rows,
           "coverage": tot_filed / tot_units if tot_units else 0.0,
           "pair_reach": tot_same / tot_pairs if tot_pairs else 0.0,
           "state_at_new": tot_at / tot_pairs if tot_pairs else 0.0,
           "current_is_new": tot_current / tot_pairs if tot_pairs else 0.0,
           "mutated_units": mutated}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(repo_root() / "runs/decision_nodes/state-all6.json"))
    args = ap.parse_args(argv)
    base = metrics(state=False)
    state = metrics(state=True)
    out = {"topic_only": base, "with_state": state,
           "delta_pair_reach": round(state["pair_reach"] - base["pair_reach"], 4),
           "delta_current": round(state["current_is_new"] - base["current_is_new"], 4)}
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "meetings"}, indent=1))
    print("artefact", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
