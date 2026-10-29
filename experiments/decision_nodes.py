"""Benchmark the decision-node view (D-058).

Ground truth available:
  * gold drawers per segment (`data/gold/summre_subjects_*.json`);
  * hand-labelled supersede pairs (`data/gold/summre_temporal_*.json`).

Measures:
  * coverage      — messages filed into a node;
  * purity        — majority gold drawer per node, weighted;
  * V-measure     — node assignment vs gold drawer (homogeneity/completeness);
  * pair reach    — gold old+new in the same node (the node-level analogue of
                    D-046's 22/40 thread reach);
  * no mutation   — every child keeps valid_to/superseded_by = None.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from sklearn.metrics import (adjusted_rand_score, homogeneity_completeness_v_measure)

from engine.config import load_config, repo_root
from engine.read.baseline_rules import RuleReader
from engine.schemas import Source, Turn
from engine.segment.chunklet_adapter import Segmenter
from engine.threads import build_session_nodes

MEETINGS = ["024c_EBPH", "047a_EDRH", "060a_EDRH", "072c_EEPH", "016c_EBPZ", "047c_EDPH"]
RAW = json.loads(Path("data/external/summre_test/meetings.json").read_text())
CFG = load_config()
READER = RuleReader()


def units_of(mid: str):
    src = Source(source_id=mid, source_type="meeting", source_date=date(2026, 10, 23),
                 project=mid,
                 turns=[Turn(speaker=u["speaker"], text=u["text"], turn_index=i)
                        for i, u in enumerate(RAW[mid])])
    out = []
    for u in Segmenter(CFG).segment(src):
        r = READER.read(u)
        out.append(u.model_copy(update={"kind": r.kind, "type": r.type,
                                        "status": r.status}))
    return out


def gold_pairs():
    out = defaultdict(set)
    for p in ("data/gold/summre_temporal_train.json",
              "data/gold/summre_temporal_sel.json"):
        d = json.loads(Path(p).read_text())
        for mid, m in d["meetings"].items():
            for th in m["threads"]:
                for r in th["relations"]:
                    if r["relation"] == "supersedes":
                        out[mid].add((r["old"], r["new"]))
    return out


def main() -> int:
    pairs = gold_pairs()
    rows = []
    tot_units = tot_filed = tot_gold = tot_same = tot_reach = 0
    all_node, all_gold = [], []
    mutated = 0
    for mid in MEETINGS:
        units = units_of(mid)
        nodes = build_session_nodes(units)
        node_of = {}
        for n in nodes:
            for c in n.children:
                node_of[c.unit_id] = n.node_id
                if c.valid_to is not None or c.superseded_by is not None:
                    mutated += 1
        gold = json.loads(Path(f"data/gold/summre_subjects_{mid}.json").read_text())["keys"]
        by_turn = defaultdict(list)
        for u in units:
            by_turn[u.turn_index].append(u)
        # coverage + purity + V-measure
        filed = sum(1 for u in units if u.unit_id in node_of)
        labelled = [(node_of[u.unit_id], gold.get(str(u.turn_index)))
                    for u in units if u.unit_id in node_of
                    and gold.get(str(u.turn_index))]
        node_drawers = defaultdict(Counter)
        for nd, gd in labelled:
            node_drawers[nd][gd] += 1
        purity = (sum(c.most_common(1)[0][1] for c in node_drawers.values())
                  / len(labelled)) if labelled else 0.0
        if labelled:
            h, c, v = homogeneity_completeness_v_measure(
                [x[0] for x in labelled], [x[1] for x in labelled])
            ari = adjusted_rand_score([x[0] for x in labelled],
                                      [x[1] for x in labelled])
        else:
            h = c = v = ari = 0.0
        all_node += [x[0] for x in labelled]
        all_gold += [x[1] for x in labelled]
        # pair reach
        gold_mid = pairs.get(mid, set())
        same = 0
        for o, n in gold_mid:
            if o not in by_turn or n not in by_turn:
                continue
            if all(u.unit_id in node_of for u in by_turn[o] + by_turn[n]):
                same += int(len({node_of[u.unit_id] for u in by_turn[o] + by_turn[n]}) == 1)
        tot_units += len(units)
        tot_filed += filed
        tot_gold += len(labelled)
        tot_same += same
        tot_reach += len(gold_mid)
        print(f"{mid}: units={len(units)} filed={filed} nodes={len(nodes)} "
              f"gold_labelled={len(labelled)} purity={purity:.3f} "
              f"V={v:.3f} ARI={ari:.3f} pair_reach={same}/{len(gold_mid)}")
        rows.append(dict(meeting=mid, units=len(units), filed=filed,
                         nodes=len(nodes), labelled=len(labelled),
                         purity=purity, homogeneity=h, completeness=c,
                         v_measure=v, ari=ari, pairs=len(gold_mid), same=same))

    h, c, v = homogeneity_completeness_v_measure(all_node, all_gold)
    ari = adjusted_rand_score(all_node, all_gold)
    print("\nTOTAL")
    print(f"  coverage        {tot_filed}/{tot_units} = {tot_filed/tot_units:.3f}")
    print(f"  purity          {sum(r['purity']*r['labelled'] for r in rows)/tot_gold:.3f}")
    print(f"  homogeneity     {h:.3f}  completeness {c:.3f}  V-measure {v:.3f}  ARI {ari:.3f}")
    print(f"  pair reach      {tot_same}/{tot_reach} = {tot_same/tot_reach:.3f}")
    print(f"  mutated units   {mutated} (must be 0)")
    out = repo_root() / "runs" / "decision_nodes" / "benchmark-2026-10-25.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meetings": rows,
                               "coverage": tot_filed / tot_units,
                               "purity": sum(r["purity"] * r["labelled"] for r in rows) / tot_gold,
                               "homogeneity": h, "completeness": c,
                               "v_measure": v, "ari": ari,
                               "pair_reach": tot_same / tot_reach,
                               "mutated_units": mutated}, indent=2))
    print(f"  artefact        {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
