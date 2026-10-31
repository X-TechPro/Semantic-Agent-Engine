#!/usr/bin/env python3
"""Build a hand-labelled relation hard-case set from the transcribed corpus.

Relations are inferred from the corpus structure, not from the relation model:
decision -> commitment = creates_action; objection -> decision = contradicts;
date reference -> referenced decision = supports; cross-project decisions =
unrelated. The set is intentionally OOD relative to the synthetic training pairs.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

import memory_engine as me

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")


def main():
    engine = me.MeetingMemoryEngine(device="cpu", batch_size=64)
    ds = engine._get_dataset("transcribed")
    records = ds["records"]
    by_meeting = ds["by_meeting"]
    by_date = {m["meeting_date"]: mid for mid, m in ds["meetings"].items()}
    groups = defaultdict(list)
    for mid, m in ds["meetings"].items():
        groups[m.get("project_id") or m.get("project")].append(mid)
    for mids in groups.values():
        mids.sort(key=lambda mid: ds["meetings"][mid]["meeting_date"])

    decisions, commitments, objections, references = {}, {}, {}, {}
    for mid, cids in by_meeting.items():
        d, c, o = [], [], []
        for cid in cids:
            text = records[cid]["message_text"].casefold()
            if "décision:" in text or "decision:" in text:
                d.append(cid)
            if re.search(r"prends\s+(?:prends\s+)?en charge", text):
                c.append(cid)
            if "réserve" in text or "reserve" in text:
                o.append(cid)
            if "avait acté" in text or "avait acte" in text:
                dates = DATE.findall(text)
                if dates and dates[0] in by_date and by_date[dates[0]] != mid:
                    references.setdefault(mid, []).append((cid, by_date[dates[0]]))
        decisions[mid], commitments[mid], objections[mid] = d, c, o

    cases = []
    seen = set()

    def add(label, source, target, note):
        key = (source, target, label)
        if source == target or key in seen:
            return
        seen.add(key)
        cases.append({"pair_id": f"rel-{len(cases)+1:03d}", "source": source,
                      "target": target, "label": label, "note": note})

    for mid in by_meeting:
        if decisions[mid] and commitments[mid]:
            add("creates_action", decisions[mid][0], commitments[mid][0], "décision -> engagement")
        if objections[mid] and decisions[mid]:
            add("contradicts", objections[mid][0], decisions[mid][0], "objection -> décision")

    for mid, refs in references.items():
        for source, target_mid in refs:
            if decisions.get(target_mid):
                add("supports", source, decisions[target_mid][0], "référence datée -> décision citée")

    projects = sorted(groups)
    for i, a in enumerate(projects):
        for b in projects[i + 1:]:
            ma = groups[a][0]
            mb = groups[b][0]
            if decisions.get(ma) and decisions.get(mb):
                add("unrelated", decisions[ma][0], decisions[mb][0], "projets différents")

    # Balance to at most 10 per label, keep all supports available.
    per = defaultdict(list)
    for case in cases:
        per[case["label"]].append(case)
    balanced = []
    for label in ("supports", "contradicts", "creates_action", "unrelated"):
        balanced.extend(per[label][:10])
    for i, case in enumerate(balanced, 1):
        case["pair_id"] = f"rel-{i:03d}"
    (DATA / "relation_hard_cases.json").write_text(json.dumps(balanced, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": len(balanced), "by_label": {k: min(10, len(v)) for k, v in per.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
