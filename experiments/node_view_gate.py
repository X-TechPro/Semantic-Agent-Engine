"""v3 gate: node reach and historical state on all six gold meetings.

Measures, per meeting:
  * pair reach      — gold old and new in one node (merged topic);
  * state at new    — `node.at_turn(new)` returns the new message (the node can
                      retrace the truth at that point of the meeting);
  * current         — `node.latest` is the gold new message (final state);
  * mutation        — every child still has valid_to/superseded_by = None.

This is the all-six ruler. The detector (which child changed a value) is only
labelled on 024c and 047a; its numbers are attached from the hybrid artefact.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from engine.config import repo_root
from decision_nodes import MEETINGS, gold_pairs, units_of
from engine.threads import build_session_nodes


def main() -> int:
    pairs = gold_pairs()
    rows = []
    tot_units = tot_filed = tot_pairs = tot_same = tot_at = tot_latest = 0
    mutated = 0
    for mid in MEETINGS:
        units = units_of(mid)
        nodes = build_session_nodes(units)
        node_of_turn = {}
        for n in nodes:
            for c in n.children:
                node_of_turn[c.turn_index] = n
                if c.valid_to is not None or c.superseded_by is not None:
                    mutated += 1
        filed = sum(1 for u in units if u.turn_index in node_of_turn)
        gold = pairs.get(mid, set())
        same = at_new = latest_new = 0
        for old, new in gold:
            no, nn = node_of_turn.get(old), node_of_turn.get(new)
            same += int(no is not None and no is nn)
            if nn is not None:
                at = nn.at_turn(new)
                at_new += int(at is not None and at.turn_index == new)
                latest_new += int(nn.latest is not None and nn.latest.turn_index == new)
        rows.append({"meeting": mid, "units": len(units), "filed": filed,
                     "nodes": len(nodes), "pairs": len(gold),
                     "same_node": same, "state_at_new": at_new,
                     "current_is_new": latest_new})
        tot_units += len(units); tot_filed += filed
        tot_pairs += len(gold); tot_same += same
        tot_at += at_new; tot_latest += latest_new
        print(f"{mid}: units={len(units)} nodes={len(nodes)} pairs={len(gold)} "
              f"same={same} state_at_new={at_new} current_is_new={latest_new}")
    det = json.loads((repo_root() / "runs/decision_nodes/hybrid-2026-10-25.json").read_text())
    out = {
        "arch": "v3",
        "meetings": rows,
        "coverage": tot_filed / tot_units,
        "pair_reach": tot_same / tot_pairs,
        "state_at_new": tot_at / tot_pairs,
        "current_is_new": tot_latest / tot_pairs,
        "mutated_units": mutated,
        "detector": det,
        "note": ("pair reach/state/current are measured on all six gold meetings; "
                 "detector precision/recall is labelled on 024c and 047a only"),
    }
    path = repo_root() / "runs/decision_nodes/v3-all6.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\ncoverage={out['coverage']:.3f} pair_reach={out['pair_reach']:.3f} "
          f"state_at_new={out['state_at_new']:.3f} "
          f"current_is_new={out['current_is_new']:.3f} mutated={mutated}")
    print(f"artefact {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
