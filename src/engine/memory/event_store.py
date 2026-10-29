"""Mention/event JSON store — the event-sourced version of N3/N4.

A mention is one sentence/segment with a timestamp, tags and an embedding. An
event groups mentions by a stable key (project + point) and exposes its current
row. N3 becomes the event key; N4 becomes `current()` (latest or latest
confirmed). No separate clustering service.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class MentionRecord:
    mention_id: str
    project: str
    turn: int
    timestamp: str
    text: str
    point: str
    value: str
    status: str
    evidence: str
    embedding: list[float] | None = None

    @property
    def event_key(self) -> str:
        return f"{self.project}:{self.point}"

    def to_json(self) -> dict:
        return {
            "mention_id": self.mention_id, "project": self.project,
            "turn": self.turn, "timestamp": self.timestamp,
            "text": self.text, "point": self.point, "value": self.value,
            "status": self.status, "evidence": self.evidence,
            "event_key": self.event_key, "embedding": self.embedding,
        }

    @classmethod
    def from_json(cls, row: dict) -> "MentionRecord":
        return cls(**{k: row[k] for k in (
            "mention_id", "project", "turn", "timestamp", "text", "point",
            "value", "status", "evidence", "embedding")})


@dataclass
class EventRecord:
    event_key: str
    project: str
    point: str
    mentions: list[MentionRecord] = field(default_factory=list)

    def add(self, mention: MentionRecord) -> None:
        self.mentions.append(mention)
        self.mentions.sort(key=lambda m: (m.timestamp, m.turn))

    def current(self, mode: str = "latest_confirmed") -> MentionRecord | None:
        rows = self.mentions
        if mode == "latest_confirmed":
            rows = [m for m in rows if m.status in ("confirmed", "cancelled")]
        if not rows:
            return None
        return max(rows, key=lambda m: (m.timestamp, m.turn))

    def open(self) -> list[MentionRecord]:
        return [m for m in self.mentions if m.status in ("proposed", "open")]

    def history(self) -> list[MentionRecord]:
        return list(self.mentions)


@dataclass
class QueryResult:
    event: EventRecord
    current: MentionRecord | None
    score: float


class EventStore:
    """In-memory JSON-shaped store with optional cosine search."""

    def __init__(self) -> None:
        self.mentions: list[MentionRecord] = []
        self.events: dict[str, EventRecord] = {}
        self._vectors: np.ndarray | None = None

    def add(self, mention: MentionRecord, embedding: Iterable[float] | None = None,
            *, normalize: bool = True) -> None:
        if embedding is not None:
            v = np.asarray(list(embedding), dtype=np.float32)
            if normalize and np.linalg.norm(v) > 0:
                v = v / np.linalg.norm(v)
            mention.embedding = v.tolist()
        self.mentions.append(mention)
        event = self.events.get(mention.event_key)
        if event is None:
            event = self.events[mention.event_key] = EventRecord(
                event_key=mention.event_key, project=mention.project,
                point=mention.point)
        event.add(mention)
        self._vectors = None

    def _matrix(self) -> np.ndarray | None:
        if self._vectors is not None:
            return self._vectors
        rows = [m for m in self.mentions if m.embedding is not None]
        if not rows:
            return None
        self._vectors = np.asarray([m.embedding for m in rows], dtype=np.float32)
        self._vector_ids = [m.mention_id for m in rows]
        return self._vectors

    def point_events(self, project: str | None = None,
                     point: str | None = None) -> list[EventRecord]:
        out = []
        for e in self.events.values():
            if project and e.project != project:
                continue
            if point and e.point != point:
                continue
            out.append(e)
        return sorted(out, key=lambda e: e.event_key)

    def query(self, text: str, *, embedder=None, project: str | None = None,
              point: str | None = None, top_k: int = 8,
              mode: str = "latest_confirmed") -> list[QueryResult]:
        """Closest event by any mention, then that event's current row.

        `point` is the exact filter (N3). Without it, vector similarity proposes
        candidate mentions (N2/N3); the event key groups them; `current()` is
        the N4 policy.
        """
        if point:
            out = []
            for e in self.point_events(project, point):
                cur = e.current(mode)
                if cur is not None:
                    out.append(QueryResult(e, cur, 1.0))
            return out
        if embedder is None:
            return []
        q = np.asarray(embedder(text), dtype=np.float32)
        n = np.linalg.norm(q)
        if n:
            q = q / n
        mat = self._matrix()
        if mat is None:
            return []
        scores = mat @ q
        order = np.argsort(-scores)[: max(1, top_k)]
        ids = {self._vector_ids[int(i)] for i in order}
        by_event: dict[str, tuple[EventRecord, float]] = {}
        for m in self.mentions:
            if m.mention_id not in ids or m.embedding is None:
                continue
            s = float(np.dot(np.asarray(m.embedding, dtype=np.float32), q))
            prev = by_event.get(m.event_key)
            if prev is None or s > prev[1]:
                by_event[m.event_key] = (self.events[m.event_key], s)
        out = []
        for event, s in by_event.values():
            cur = event.current(mode)
            if cur is not None:
                out.append(QueryResult(event, cur, s))
        return sorted(out, key=lambda r: -r.score)

    def save(self, path: str | Path) -> Path:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for m in self.mentions:
                f.write(json.dumps(m.to_json(), ensure_ascii=False) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "EventStore":
        store = cls()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                store.add(MentionRecord.from_json(json.loads(line)))
        return store


__all__ = ["EventStore", "EventRecord", "MentionRecord", "QueryResult"]
