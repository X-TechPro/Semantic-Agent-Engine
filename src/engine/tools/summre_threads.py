"""Temporal relation labelling on real meetings — the linker's training supply.

The synthetic linker (link/train.py) learns supersede pairs from ledger gold
that is by construction value-swaps; the external read (D-025) showed the
rules linker marks 0/28 on real threads. This tool is the supply side for a
linker trained on REAL pairs, mirroring summre_label.py's discipline:

- `dump MEETING` prints candidate threads: the meeting's segments grouped by
  the hand-labelled drawer keys (fine keys when present — they read tighter),
  in segment order, only drawers with >= 2 segments. A thread is a decision
  that evolved; chatter and singletons are not shown.
- `compile` turns a spec of pair lines into a temporal gold file with the
  SAME schema as summre_temporal_dev/test.json (meetings -> threads ->
  relations {old, new, relation, note}), so every existing reader keeps
  working. Texts are read back from meetings.json into the note, so the gold
  cannot drift from the extract.

Split discipline, declared BEFORE any labelling (the protocol field records
it; the trainer refuses by split):
  train    — may be trained on. D-043 fitted on five (024c, 047a, 060a, 072c,
             016c); D-044 expanded to nine by adding every remaining meeting
             with complete drawer gold (012c, 019a, 059b, 070b) — the scale
             hypothesis tested directly, bars unchanged. The two meetings of
             the OLD dev file (019c, 025b) carry no drawer gold and stay
             excluded — measured, recorded.
  sel      — threshold/operating-point selection only (047c)
  holdout  — labelled, sealed, scored EXACTLY ONCE (008a, 049b, 053b)
  measure  — the spent external test (007a, 013c, 014c): never dumped,
             never compiled, never trained — the spent-test refusal in
             summre_reach covers the scoring side; this list covers the
             labelling side. 039b (the drawer campaign's measure tape) stays
             out of train too.

A relation is a PAIR (old, new) of segment indices with old < new (segments
are chronological). Where the thread leaves the state change genuinely
undecidable, no line is written — silence in the file means "not claimed",
never "not a relation".
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from engine.config import repo_root

RELATIONS = ("supersedes", "contradicts", "supports")

# declared BEFORE labelling; compile refuses a meeting outside its split
SPLITS = {
    # D-044 (2026-10-23): train expanded 5 -> 9 with every remaining meeting
    # carrying complete drawer gold. sel and holdout UNTOUCHED — the sealed
    # holdout has still never been read; the spent test stays unreachable.
    "train": ["024c_EBPH", "047a_EDRH", "060a_EDRH", "072c_EEPH", "016c_EBPZ",
              "012c_EBPZ", "019a_EARD", "059b_EBDH", "070b_EBDL"],
    "sel": ["047c_EDPH"],
    "holdout": ["008a_EARH", "049b_EBDH", "053b_EDDH"],
}
SPENT_TEST = ["007a_ECRH", "013c_EAPD", "014c_EAPD"]

SPEC_LINE = re.compile(r"^(\d+)\s*->\s*(\d+)\s*:\s*(\w+)(?:\s+(.*))?$")

PROTOCOL = ("hand-labelled by the assistant on {d}: each thread read in full "
            "(segments grouped by the meeting's drawer gold, in order); "
            "relations are PAIRS (old, new) of segment indices in "
            "data/external/summre_test/meetings.json, labelled "
            "{{supersedes, contradicts, supports}}; where the thread left the "
            "state change undecidable no line was written; split declared "
            "before labelling (train / sel / holdout)")


def _load(root: Path):
    meetings = json.loads((root / "data/external/summre_test/meetings.json")
                          .read_text(encoding="utf-8"))
    return {m: (v["segments"] if isinstance(v, dict) else v)
            for m, v in meetings.items()}


def _drawer_keys(root: Path, meeting: str) -> dict[int, str]:
    """fine keys when present (tighter reading), else merged keys."""
    p = root / "data/gold" / f"summre_subjects_{meeting}.json"
    if not p.exists():
        raise FileNotFoundError(f"no drawer gold for {meeting} — label subjects first")
    g = json.loads(p.read_text(encoding="utf-8"))
    src = g.get("fine_keys") or g.get("keys") or {}
    return {int(i): k for i, k in src.items() if k}


def dump(meeting: str, min_len: int = 12) -> None:
    root = repo_root()
    if meeting in SPENT_TEST:
        raise ValueError(f"{meeting} is the spent external test — labelling it "
                         "now would tune on the burned holdout")
    segs = _load(root)[meeting]
    keys = _drawer_keys(root, meeting)
    threads: dict[str, list[int]] = {}
    for i, k in sorted(keys.items()):
        threads.setdefault(k, []).append(i)
    for k, idxs in sorted(threads.items()):
        if len(idxs) < 2:
            continue
        print(f"=== {k} ({len(idxs)} segs) ===")
        for i in idxs:
            t = segs[i]["text"].strip()
            if len(t) < min_len:
                continue
            print(f"[{i}] ({segs[i].get('speaker', '?')}) {t}")
        print()


def parse_spec(text: str, n_segments: int) -> list[dict]:
    out = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SPEC_LINE.match(line)
        if not m:
            raise ValueError(f"bad spec line: {line!r}")
        old, new, rel, note = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
        if rel not in RELATIONS:
            raise ValueError(f"unknown relation {rel!r} in {line!r}")
        if not (0 <= old < new < n_segments):
            raise ValueError(f"pair {old}->{new} not ordered/in range "
                             f"(0..{n_segments - 1})")
        out.append({"old": old, "new": new, "relation": rel, "note": note or ""})
    return out


def compile_gold(meeting: str, spec_path: Path, split: str, out_dir: Path) -> Path:
    root = repo_root()
    if meeting not in SPLITS.get(split, []):
        raise ValueError(f"{meeting} is not declared in split {split!r} — the "
                         "split was fixed before labelling; refusing to move it")
    segs = _load(root)[meeting]
    rels = parse_spec(spec_path.read_text(encoding="utf-8"), len(segs))
    for r in rels:  # texts ride in the note so drift is visible
        r["note"] = (f"OLD: {segs[r['old']]['text'].strip()[:90]} | "
                     f"NEW: {segs[r['new']]['text'].strip()[:90]}"
                     + (f" | {r['note']}" if r["note"] else ""))
    by_topic: dict[str, list[dict]] = {}
    keys = _drawer_keys(root, meeting)
    for r in rels:
        by_topic.setdefault(keys.get(r["new"], "unthreaded"), []).append(r)
    dest = out_dir / f"summre_temporal_{split}.json"
    doc = {"split": split,
           "corpus": "linagora/SUMM-RE, data/external/summre_test/meetings.json extract",
           "protocol": PROTOCOL.format(d=date.today().isoformat()),
           "declared_splits": {k: v for k, v in SPLITS.items()},
           "spent_test_never_labelled": SPENT_TEST,
           "meetings": {}}
    if dest.exists():  # accumulate meetings within one split file
        doc = json.loads(dest.read_text(encoding="utf-8"))
    doc["meetings"][meeting] = {
        "sentences_reviewed": len(segs),
        "threads": [{"topic": t, "relations": rs} for t, rs in sorted(by_topic.items())],
    }
    rels_all = [r for mm in doc["meetings"].values() for th in mm["threads"]
                for r in th["relations"]]
    doc["n_relations"] = len(rels_all)
    doc["n_supersedes_pairs"] = sum(1 for r in rels_all
                                    if r["relation"] == "supersedes")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("meeting")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--split", choices=list(SPLITS), default=None)
    args = ap.parse_args(argv)
    root = repo_root()
    if args.dump:
        dump(args.meeting)
        return 0
    if not (args.spec and args.split):
        ap.error("--dump or (--spec and --split) required")
    p = compile_gold(args.meeting, Path(args.spec), args.split, root / "data/gold")
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
