"""SUMM-RE test-split extraction for the external temporal gold (phase 8).

Why this is a range-read tool and not another `fetch_french` file pin: every
parquet shard in `linagora/SUMM-RE` embeds full meeting AUDIO (~0.4-0.6 GB per
shard; 28 test shards). The temporal label set needs only three text columns,
and parquet's columnar layout lets pyarrow fetch exactly those bytes over HTTP
-- measured 2026-10-21: ~1.3 s per shard instead of a ~12 GB download. The
pinned revision is the one `make frdata` already proved out (PROVENANCE.json of
the example file); reusing it keeps the identity story one-revision-deep.

Filter, stated because hand-labeling starts from it (test-only, never tuned on):
`sil`/`w_N`/`*` ASR placeholders drop, segments under 15 chars drop (nothing
labellable fits), ordering is by segment start across the whole meeting.

Usage:
    PYTHONPATH=src python -m engine.tools.summre_extract [--shards 4]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from engine.config import repo_root

REV = "47cb4ec3437ef5497b6092b8c6d1e0b5b1d86dc0"
REPO = "linagora/SUMM-RE"
N_TEST_SHARDS = 28
MIN_CHARS = 15

#: ASR garbage that must never reach a label pass: silence markers, word-index
#: placeholders, unintelligible stars. Anchored: "sil" mid-sentence is real text.
GARBAGE_RE = re.compile(r"^\s*(sil\b|w_\d|\*)")


def is_clean(text: str) -> bool:
    t = (text or "").strip()
    return len(t) >= MIN_CHARS and not GARBAGE_RE.match(t) and t != "*"


def meeting_segments(rows: list[dict]) -> dict[str, list[dict]]:
    """rows (meeting_id, speaker_id, segments) -> meeting -> time-ordered
    clean segments. Deterministic: stable sort on (start, speaker)."""
    out: dict[str, list[dict]] = {}
    for r in rows:
        for s in (r.get("segments") or []):
            text = (s.get("transcript") or "").strip()
            if not is_clean(text):
                continue
            out.setdefault(r["meeting_id"], []).append(
                {"start_min": round(float(s.get("start", 0.0)) / 60.0, 2),
                 "speaker": r["speaker_id"], "text": text})
    return {m: sorted(v, key=lambda x: (x["start_min"], x["speaker"]))
            for m, v in sorted(out.items())}


def shard_urls(n: int) -> list[str]:
    return [f"data/test/test-{i:05d}-of-{N_TEST_SHARDS:05d}.parquet"
            for i in range(n)]


def read_shards(n: int) -> list[dict]:
    import fsspec
    import pyarrow.parquet as pq
    rows: list[dict] = []
    for rel in shard_urls(n):
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{rel}"
        pf = pq.ParquetFile(fsspec.open(url, "rb").open())
        rows += pf.read(columns=["meeting_id", "speaker_id",
                                 "segments"]).to_pylist()
    return rows


def extract(shards: int = 4) -> dict:
    meetings = meeting_segments(read_shards(shards))
    n_seg = sum(len(v) for v in meetings.values())
    out_dir = repo_root() / "data/external/summre_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meetings.json").write_text(
        json.dumps(meetings, ensure_ascii=False, indent=1), encoding="utf-8")
    prov = {"repo": REPO, "revision": REV, "split": "test",
            # files = repo paths actually read. The DATASETS generator counts
            # this key; leaving it out made the extract render as "0 file(s)"
            # -- a generated doc understating what happened on the network.
            "files": shard_urls(shards),
            "shards_read": shards, "columns": ["meeting_id", "speaker_id",
                                               "segments"],
            "access": "remote parquet range-read (text columns only; audio "
                      "never downloaded)",
            "filter": {"drop": "segments matching ^\\s*(sil\\b|w_\\d|\\*) "
                               f"or shorter than {MIN_CHARS} chars",
                       "order": "start_min then speaker, per meeting"},
            "n_meetings": len(meetings), "n_segments": n_seg,
            "licence": "cc-by-sa-4.0 (canonical card; see docs/DATASETS.md)",
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "note": "test-only source: the hand-labelled temporal gold built "
                    "from this extract is never used for tuning (9.5)"}
    (out_dir / "PROVENANCE.json").write_text(
        json.dumps(prov, indent=1), encoding="utf-8")
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=4,
                    help="test shards to read (5 rows ~= 2 meetings each)")
    args = ap.parse_args(argv)
    prov = extract(args.shards)
    print(json.dumps({k: prov[k] for k in
                      ("shards_read", "n_meetings", "n_segments", "revision")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
