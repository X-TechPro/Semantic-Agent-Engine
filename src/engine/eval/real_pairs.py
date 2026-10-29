"""Shared real SUMM-RE loaders for continuity/stage experiments."""
from __future__ import annotations

import json
from pathlib import Path

from engine.config import repo_root

SPLITS = ("train", "sel")
GOLD = {"train": "summre_temporal_train.json", "sel": "summre_temporal_sel.json"}
SPENT = {"007a_ECRH", "013c_EAPD", "014c_EAPD"}


def load_meetings(root: Path | None = None) -> dict[str, list[dict]]:
    root = root or repo_root()
    raw = json.loads((root / "data/external/summre_test/meetings.json").read_text())
    return {m: (v["segments"] if isinstance(v, dict) else v)
            for m, v in raw.items()}


def load_pairs(root: Path | None = None) -> dict[str, list[tuple[str, int, int]]]:
    root = root or repo_root()
    pairs: dict[str, list[tuple[str, int, int]]] = {}
    for split in SPLITS:
        doc = json.loads((root / "data/gold" / GOLD[split]).read_text())
        for mid, mm in doc["meetings"].items():
            if mid in SPENT:
                raise ValueError(f"{mid} is the spent external test")
            for th in mm["threads"]:
                for r in th["relations"]:
                    if r["relation"] == "supersedes":
                        pairs.setdefault(mid, []).append(
                            (str(th.get("topic") or "sujet"),
                             int(r["old"]), int(r["new"])))
    return pairs


def pair_rows(pairs: dict[str, list[tuple[str, int, int]]]):
    for mid in sorted(pairs):
        for topic, old, new in sorted(pairs[mid]):
            yield mid, topic, old, new


__all__ = ["load_meetings", "load_pairs", "pair_rows", "SPENT", "SPLITS", "GOLD"]
