"""Annotation-supplied subject keys — the demo path, not an extractor.

B-13 measured six proposals and enabled none: real casual French does not yet
yield trustworthy subject keys from any of them. The engine's temporal
behaviour, however, is demonstrable the moment keys EXIST — which is what this
supplier provides for the real-data demo: it reads the hand-labelled drawer
gold (`data/gold/summre_subjects_*.json`, every key already validated by the
labelling tool through `normalize_attr`) and answers text -> subject key.

This is not a seventh extractor and must never be reported as one: the
resolver reads `gold` in `stats()["subject_resolver"]`, the gold file's own
split field is loaded with it, and the demo artefact records
`subject_source: "gold-annotation"`. It supplies annotation; it extracts
nothing. A missing gold directory is a startup error (the D-011 class), and
lookup is exact-text — an unlabelled sentence gets NO key, never a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.slots.subjects import subject_key

GOLD_GLOB = "summre_subjects_*.json"


class GoldSubjectModel:
    """text -> drawer key, per meeting, from the labelled gold files."""

    def __init__(self, gold_dir: Path | str, meetings_path: Path | str):
        gold_dir = Path(gold_dir)
        files = sorted(gold_dir.glob(GOLD_GLOB))
        if not files:
            raise FileNotFoundError(
                f"subject_resolver=gold: no {GOLD_GLOB} under {gold_dir} — "
                "label the meetings first (engine.tools.summre_label); "
                "refusing to report 'gold' while running rules")
        meetings = json.loads(Path(meetings_path).read_text(encoding="utf-8"))
        self.map: dict[str, dict[str, str]] = {}
        self.splits: dict[str, str] = {}
        for p in files:
            g = json.loads(p.read_text(encoding="utf-8"))
            mm = meetings.get(g["meeting"])
            if mm is None:
                continue
            segs = mm["segments"] if isinstance(mm, dict) else mm
            keys = g.get("keys") or {}
            per: dict[str, str] = {}
            for i, s in enumerate(segs):
                drawer = keys.get(str(i))
                if drawer is not None:
                    per.setdefault(s["text"], drawer)   # first wins; gold is 1:1
            self.map[g["meeting"]] = per
            self.splits[g["meeting"]] = g.get("split")
        self.stats = {"asks": 0, "hit": 0, "miss": 0}

    def subject_for(self, text: str, project: str | None, *,
                    owner_slot: bool = False, context=None) -> str | None:
        self.stats["asks"] += 1
        drawer = self.map.get(project or "", {}).get(text)
        if drawer is None:
            self.stats["miss"] += 1
            return None
        self.stats["hit"] += 1
        return subject_key(project, drawer, owner_slot=owner_slot)

    def coverage_note(self) -> dict[str, Any]:
        return {"meetings": sorted(self.map),
                "splits": dict(self.splits),
                "labelled_texts": {m: len(v) for m, v in self.map.items()}}
