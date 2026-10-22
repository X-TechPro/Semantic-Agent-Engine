"""Hand-labelling tool for real-French subject keys (the supervised corpus).

The four §6.3 proposers failed for a measured, corpus-shaped reason (D-029):
nothing real ever asserted *which drawer a sentence files into*. This is the
supply side: dump a meeting's segments numbered, decide the drawer key per
segment (short folded French noun, or null for chatter), and compile a compact
run-length spec into the gold file — the same assistant-labelled workflow that
produced the temporal dev gold, one layer further down.

Split discipline is on disk, in the file itself: `"split": "train"` meetings
may train; `"split": "measure"` meetings are the gate's tape and must never be
trained on (the trainer refuses train-labelled files by the declared split).
Keys are validated through the SAME `normalize_attr` the model's output goes
through, so a gold key and a predicted key can never be two different formats.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from engine.config import repo_root
from engine.slots.subjects import normalize_attr

PROTOCOL = ("hand-labelled by the assistant on {d}: subject drawer key per "
            "segment read in windows of 8 with neighbour context; short folded "
            "French noun per the shared normalize_attr rules, null where the "
            "utterance files into no drawer (chatter, greetings, pure "
            "reactions); one primary drawer per segment")


def parse_spec(spec_text: str, n_segments: int) -> dict[int, str | None]:
    """Run-length spec -> {segment index: key}. Lines like `12-18:null` or
    `30:salle` assign inclusive ranges; a bare `47:buffet` assigns one index.
    Anything not covered must be listed `0-{n-1}` explicitly across the file —
    silently-unlabelled segments would become fake null rows."""
    out: dict[int, str | None] = {}
    for line in spec_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?\s*:\s*(\S.*)?", line)
        if not m:
            raise ValueError(f"bad spec line: {line!r}")
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        raw = (m.group(3) or "null").strip()
        key = None if raw.lower() in ("null", "none", "-") else normalize_attr(raw)
        if raw.lower() not in ("null", "none", "-") and key is None:
            raise ValueError(f"spec key {raw!r} fails the shared validator — "
                             "gold and prediction must be the same format")
        for i in range(a, b + 1):
            if not 0 <= i < n_segments:
                raise ValueError(f"index {i} outside 0..{n_segments - 1}")
            out[i] = key
    return out


def compile_gold(meeting: str, spec_path: Path, meetings_data: dict,
                 split: str, out_dir: Path,
                 merge: dict[str, str] | None = None) -> Path:
    """Spec -> gold file. With `merge`, the fine labels become family keys via
    the ONE map (data/gold/merge.json) applied identically to train, dev and
    the measure tape — a coarser partition, never a relabelling: the per-meeting
    spec files stay the source of truth and the gold keeps `fine_keys` beside
    `keys` so either view is auditable. A fine key absent from the map is a
    loud failure (silence is not a default — same rule as unlabelled segments).
    """
    segs = meetings_data[meeting]
    n = len(segs)
    labels = parse_spec(spec_path.read_text(encoding="utf-8"), n)
    missing = [i for i in range(n) if i not in labels]
    if missing:
        raise ValueError(
            f"{len(missing)} segments unlabelled (e.g. {missing[:8]}) — silence "
            "is not null; every index must be assigned or explicitly nulled")
    merged = labels
    if merge:
        unknown = {v for v in labels.values() if v and v not in merge}
        if unknown:
            raise ValueError(
                f"{meeting}: fine keys {sorted(unknown)} absent from the merge "
                "map — extend data/gold/merge.json or fix the spec; refusing a "
                "silent identity")
        merged = {i: (merge[v] if v else None) for i, v in labels.items()}
    gold = {"split": split, "meeting": meeting, "n_segments": n,
            "protocol": PROTOCOL.format(d=date.today().isoformat()),
            "keys": {str(i): merged[i] for i in range(n)},
            "counts": _mix(merged)}
    if merge:
        gold["merge"] = {"file": "data/gold/merge.json",
                         "families": sorted({merge[k] for k in merge})}
        gold["fine_keys"] = {str(i): labels[i] for i in range(n)}
    p = out_dir / f"summre_subjects_{meeting}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gold, indent=1, ensure_ascii=False), encoding="utf-8")
    return p


def _mix(labels: dict[int, str | None]) -> dict[str, int]:
    from collections import Counter
    c = Counter(v if v is not None else "null" for v in labels.values())
    return dict(sorted(c.items(), key=lambda x: -x[1]))


def dump(meeting: str, meetings_data: dict, window: int = 8) -> None:
    segs = meetings_data[meeting]
    n = len(segs)
    for start in range(0, n, window):
        print(f"--- window {start}..{min(start + window, n) - 1} ---")
        for i in range(start, min(start + window, n)):
            s = segs[i]
            print(f"[{i}] ({s.get('speaker', '?')}) {s['text']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("meeting")
    ap.add_argument("--dump", action="store_true", help="print numbered segments")
    ap.add_argument("--spec", default=None, help="run-length label spec file")
    ap.add_argument("--split", choices=["train", "dev", "measure"], default="train",
                    help="train = may be trained on; dev = selection only; "
                         "measure = the gate's tape, never trained or dev-touched")
    ap.add_argument("--merge", default=None,
                    help="family map JSON (data/gold/merge.json): compiles the "
                         "spec's fine labels into coarse families; fine_keys "
                         "are preserved, an unmapped fine key is a loud error")
    args = ap.parse_args(argv)
    root = repo_root()
    data = json.loads((root / "data/external/summre_test/meetings.json")
                      .read_text(encoding="utf-8"))
    meetings_data = {m: (v["segments"] if isinstance(v, dict) else v)
                     for m, v in data.items()}
    if args.meeting not in meetings_data:
        raise ValueError(f"unknown meeting {args.meeting!r}")
    if args.dump:
        dump(args.meeting, meetings_data)
        return 0
    if not args.spec:
        ap.error("--dump or --spec required")
    merge = (json.loads(Path(args.merge).read_text(encoding="utf-8"))
             if args.merge else None)
    p = compile_gold(args.meeting, Path(args.spec), meetings_data, args.split,
                     root / "data/gold", merge=merge)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
