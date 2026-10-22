"""Shared test factories (importable as `from helpers import ...`).

Kept out of conftest so test modules can import them at module level, and
deliberately free of `hash()`: Python randomises `hash(str)` per process
(PYTHONHASHSEED), so an id built from it silently breaks determinism assertions.
"""

from __future__ import annotations

import hashlib
from datetime import date

from engine.schemas import Edge, Slots, Span, Unit, utcnow


def make_unit(text: str, *, source_id: str = "meeting-01",
              unit_id: str | None = None, source_date: str = "2026-10-01",
              speaker: str | None = "Marie", project: str | None = "atlas",
              kind: str | None = "info", type: str | None = "fact",
              status: str | None = "confirmed", slots: Slots | None = None,
              source_type: str = "meeting", turn_index: int | None = 0,
              valid_to=None, superseded_by: str | None = None,
              gold_event_ids: list[str] | None = None) -> Unit:
    d = date.fromisoformat(source_date) if isinstance(source_date, str) else source_date
    uid = unit_id or "u-" + hashlib.blake2b(
        f"{source_id}|{text}".encode(), digest_size=6).hexdigest()
    return Unit(
        unit_id=uid,
        source_id=source_id,
        source_type=source_type,
        source_date=d,
        ingested_at=utcnow(),
        text=text,
        sentences=[Span.whole(text)],
        speaker=speaker,
        project=project,
        turn_index=turn_index,
        kind=kind,
        type=type,
        status=status,
        slots=slots if slots is not None else Slots(),
        valid_from=None if kind == "query" else d,
        valid_to=valid_to,
        superseded_by=superseded_by,
        gold_event_ids=list(gold_event_ids or []),
    )


def make_edge(from_id: str, to_id: str, relation: str = "supports",
              score: float = 0.9, guard: str | None = None) -> Edge:
    return Edge(from_id=from_id, to_id=to_id, relation=relation, score=score,
                evidence_source_id="meeting-01", created_at=utcnow(), guard=guard)
