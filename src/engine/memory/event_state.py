"""Explicit event state: confirmed/cancelled values, open slots, history.

This is deliberately separate from the vector store. Recency is metadata, not
the state rule: a later proposal cannot displace an earlier confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StateMention:
    mention_id: str
    event_id: str
    value: str
    status: str
    turn_id: str
    evidence: str = ""


@dataclass
class EventState:
    event_id: str
    current_value: str | None = None
    current_status: str = "unknown"
    current_mention_id: str | None = None
    open_questions: list[str] = field(default_factory=list)
    history: list[StateMention] = field(default_factory=list)

    def apply(self, m: StateMention) -> None:
        self.history.append(m)
        if m.status in ("proposed", "open"):
            self.open_questions.append(m.value)
            # An open proposal is the current value only while nothing has
            # been confirmed/cancelled yet. A later proposal never displaces
            # a confirmed state.
            if self.current_mention_id is None:
                self.current_value = m.value
                self.current_status = m.status
                self.current_mention_id = m.mention_id
            return
        if m.status in ("confirmed", "cancelled", "completed"):
            self.current_value = m.value
            self.current_status = m.status
            self.current_mention_id = m.mention_id
            self.open_questions = []

    def answer(self) -> str:
        if self.current_value is None:
            return "unknown"
        return self.current_value


@dataclass
class EventStateTable:
    events: dict[str, EventState] = field(default_factory=dict)

    def apply(self, m: StateMention) -> None:
        ev = self.events.setdefault(m.event_id, EventState(m.event_id))
        ev.apply(m)

    def current(self, event_id: str) -> EventState | None:
        return self.events.get(event_id)


__all__ = ["StateMention", "EventState", "EventStateTable"]
