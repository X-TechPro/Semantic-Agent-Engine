"""v4 evaluation data: gold relations, change labels, and citation sets.

Two independent gold annotations exist for the same six real meetings:

* `data/gold/summre_temporal_{train,sel}.json` — typed edges between two turns
  (supports 60, supersedes 40, contradicts 14). All 40 supersede targets are
  also labelled as changes in the second file, so the two agree.
* `data/gold/summre_change_labels.json` — one binary label per turn
  (1475 turns, 187 changes, 12.7%). This is the primary target
  (`docs/CHANGE_DEFINITION.md`): "did this message change something?".

Neither file was produced here. They are the annotations already in the repo;
this module only loads them, splits them by meeting, and derives negatives that
the annotation does not spell out.

Citation sets
-------------
A change can be evidenced by more than one earlier turn. Scoring "the one turn
the annotator picked" is a guess-the-annotator game: 37/40 pairs have another
earlier turn that is an equally good citation by word overlap. So the fair
target is a *set*: the annotated old turn plus every earlier turn that states
the same value for the same point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from engine.config import repo_root
from engine.eval.real_pairs import load_meetings


@dataclass(frozen=True)
class Edge:
    meeting: str
    topic: str
    old: int
    new: int
    relation: str
    note: str = ""


@dataclass(frozen=True)
class ChangeRow:
    meeting: str
    turn: int
    text: str
    label: int


def load_edges(root: Path | None = None) -> dict[str, list[Edge]]:
    root = root or repo_root()
    out: dict[str, list[Edge]] = {}
    for name in ("summre_temporal_train.json", "summre_temporal_sel.json"):
        doc = json.loads((root / "data/gold" / name).read_text(encoding="utf-8"))
        for mid, m in doc["meetings"].items():
            for th in m["threads"]:
                for r in th["relations"]:
                    out.setdefault(mid, []).append(Edge(
                        mid, str(th.get("topic") or "sujet"), int(r["old"]),
                        int(r["new"]), str(r["relation"]),
                        str(r.get("note") or "")))
    return out


def load_change_labels(root: Path | None = None) -> dict[str, list[int]]:
    root = root or repo_root()
    doc = json.loads((root / "data/gold/summre_change_labels.json")
                     .read_text(encoding="utf-8"))
    return {mid: [int(x) for x in m["labels"]]
            for mid, m in doc["meetings"].items()}


def change_rows(meetings: dict[str, list[dict]],
                labels: dict[str, list[int]]) -> list[ChangeRow]:
    out: list[ChangeRow] = []
    for mid in sorted(labels):
        texts = [str(t.get("text") or "") for t in meetings.get(mid, [])]
        for i, label in enumerate(labels[mid]):
            if i >= len(texts) or not texts[i].strip():
                continue
            out.append(ChangeRow(mid, i, texts[i], label))
    return out


def edge_rows(meetings: dict[str, list[dict]],
              edges: dict[str, list[Edge]]) -> list[dict]:
    """Gold pair rows: (query text = the later turn, candidate = the earlier)."""
    rows = []
    for mid in sorted(edges):
        texts = [str(t.get("text") or "") for t in meetings.get(mid, [])]
        for e in edges[mid]:
            if e.old >= len(texts) or e.new >= len(texts):
                continue
            rows.append({"meeting": mid, "topic": e.topic, "old": e.old,
                         "new": e.new, "relation": e.relation,
                         "query": texts[e.new], "candidate": texts[e.old]})
    return rows


def unrelated_rows(meetings: dict[str, list[dict]],
                   edges: dict[str, list[Edge]], per_meeting: int = 20,
                   seed: int = 42) -> list[dict]:
    """Same meeting, different topic: the honest `unrelated` class.

    The gold annotation only marks pairs that are related; a four-way
    classifier needs the fourth class spelled out.
    """
    import random

    rng = random.Random(seed)
    out = []
    for mid in sorted(edges):
        texts = [str(t.get("text") or "") for t in meetings.get(mid, [])]
        topics = {e.topic for e in edges[mid]}
        used = {(e.old, e.new) for e in edges[mid]}
        pool = list(range(len(texts)))
        made = 0
        tries = 0
        while made < per_meeting and tries < per_meeting * 40 and pool:
            tries += 1
            a, b = rng.sample(pool, 2)
            if a == b or (min(a, b), max(a, b)) in used:
                continue
            if abs(a - b) < 3:
                continue
            # different topic: neither turn is a gold endpoint of the other
            if any(e.topic in topics and (e.old in (a, b) or e.new in (a, b))
                   for e in edges[mid]):
                continue
            used.add((min(a, b), max(a, b)))
            out.append({"meeting": mid, "topic": "other", "old": b, "new": a,
                        "relation": "unrelated", "query": texts[a],
                        "candidate": texts[b]})
            made += 1
    return out


def texts_of(meetings: dict[str, list[dict]], mid: str) -> list[str]:
    return [str(t.get("text") or "") for t in meetings.get(mid, [])]


def citation_set(meetings: dict[str, list[dict]], mid: str, old: int,
                 new: int) -> set[int]:
    """Every earlier turn that states the value the annotated old turn states.

    The annotated `old` is always a member. Any other earlier turn with the
    same point and the same value is an equally valid citation, so a system is
    not punished for returning a paraphrase of the evidence.
    """
    from engine.read.v4_tags import read_tags

    texts = texts_of(meetings, mid)
    if old >= len(texts) or new >= len(texts):
        return {old}
    old_tag = read_tags(texts[old])
    valid = {old}
    if not old_tag.value:
        return valid
    target = old_tag.value.strip().lower()
    for j in range(min(new, len(texts))):
        if j == old:
            continue
        tag = read_tags(texts[j])
        if tag.slot == old_tag.slot and tag.value.strip().lower() == target:
            valid.add(j)
    return valid


def card(meetings: dict[str, list[dict]], edges: dict[str, list[Edge]],
         labels: dict[str, list[int]]) -> dict:
    """The numbers that belong in the dataset documentation."""
    from collections import Counter

    rel = Counter(e.relation for mid in edges for e in edges[mid])
    per = {mid: {"turns": len(meetings.get(mid, [])),
                 "changes": sum(labels.get(mid, [])),
                 "edges": len(edges.get(mid, []))} for mid in sorted(labels)}
    return {
        "corpus": "SUMM-RE public meetings, French ASR, six meetings",
        "splits": "leave-one-meeting-out; no meeting appears in train and test",
        "turns": sum(len(v) for v in labels.values()),
        "changes": sum(sum(v) for v in labels.values()),
        "edges": sum(rel.values()),
        "edge_types": dict(rel),
        "per_meeting": per,
        "annotation": "docs/CHANGE_DEFINITION.md",
    }


__all__ = ["Edge", "ChangeRow", "load_edges", "load_change_labels",
           "change_rows", "edge_rows", "unrelated_rows", "citation_set",
           "texts_of", "card", "load_meetings"]
