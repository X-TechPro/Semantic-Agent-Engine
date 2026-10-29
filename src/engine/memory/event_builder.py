"""Build mention/event JSON from a meeting using N2's state reader."""
from __future__ import annotations

from datetime import date
from typing import Callable, Iterable

from engine.link.thread import detect_topic
from engine.memory.event_store import EventStore, MentionRecord
from engine.threads.state import DecisionState
from engine.threads.state_reader import StateReader, fold


def mentions_from_meeting(meeting_id: str, turns: list[dict],
                          reader: StateReader | None = None,
                          on_date: date | None = None,
                          embed_fn: Callable[[str], Iterable[float]] | None = None
                          ) -> list[MentionRecord]:
    """One mention per detected point (or topic fallback) per turn.

    The state is read sequentially so N2 can close an open point. Every mention
    carries its evidence and timestamp; the event key is project + point.
    """
    reader = reader or StateReader()
    state = DecisionState()
    stamp = (on_date or date(2021, 10, 5)).isoformat()
    out: list[MentionRecord] = []
    for i, turn in enumerate(turns):
        text = str(turn.get("text") or "")
        low = fold(text)
        vals = reader.values(low)
        status = reader.status(low, state, vals)
        points = dict(vals)
        if not points:
            topic = detect_topic(text)
            if topic:
                points[f"topic:{topic}"] = text.strip()
        for point, value in points.items():
            emb = list(embed_fn(text)) if embed_fn is not None else None
            out.append(MentionRecord(
                mention_id=f"{meeting_id}:{i}:{point}", project=meeting_id,
                turn=i, timestamp=stamp, text=text, point=point,
                value=str(value), status=status, evidence=text,
                embedding=emb))
        state.apply_all(reader.update(text, i, state))
    return out


def build_store(meeting_id: str, turns: list[dict],
                reader: StateReader | None = None,
                on_date: date | None = None,
                embed_fn: Callable[[str], Iterable[float]] | None = None
                ) -> EventStore:
    store = EventStore()
    for m in mentions_from_meeting(meeting_id, turns, reader, on_date, embed_fn):
        store.add(m)
    return store


__all__ = ["mentions_from_meeting", "build_store"]
