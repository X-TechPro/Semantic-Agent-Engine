"""Explicit decision state for a meeting.

The state is a small table of decision points, not a transcript summary. A row
is only written when a turn names a point and a value that is grounded in that
turn. Confirmed/cancelled updates replace the current row; proposed updates stay
in `open` until a later turn closes them. Every row keeps its evidence turn, so
the node view can always point back to the immutable message.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

StateStatus = Literal["proposed", "confirmed", "cancelled", "open"]


@dataclass(frozen=True)
class StateRow:
    point: str
    value: str
    status: StateStatus
    turn_index: int
    evidence: str
    old_value: str | None = None

    @property
    def is_current(self) -> bool:
        return self.status in ("confirmed", "cancelled")


@dataclass(frozen=True)
class StateUpdate:
    point: str
    value: str
    status: StateStatus
    turn_index: int
    evidence: str
    old_value: str | None = None


@dataclass
class DecisionState:
    """Current confirmed rows + unresolved proposals, with full history."""

    rows: dict[str, StateRow] = field(default_factory=dict)
    open: dict[str, StateRow] = field(default_factory=dict)
    history: list[StateRow] = field(default_factory=list)
    active: str | None = None

    def apply(self, update: StateUpdate) -> bool:
        """Apply one grounded update. Returns False when it changes nothing."""
        value = (update.value or "").strip()
        if not value or not update.point:
            return False
        old = self.rows.get(update.point) or self.open.get(update.point)
        row = StateRow(
            point=update.point,
            value=value,
            status=update.status,
            turn_index=int(update.turn_index),
            evidence=update.evidence,
            old_value=(old.value if old else update.old_value),
        )
        self.history.append(row)
        self.active = update.point
        if update.status in ("confirmed", "cancelled"):
            self.rows[update.point] = row
            self.open.pop(update.point, None)
        elif update.status in ("proposed", "open"):
            self.open[update.point] = row
        return True

    def apply_all(self, updates: list[StateUpdate]) -> int:
        return sum(1 for u in updates if self.apply(u))

    def current(self, point: str) -> StateRow | None:
        return self.rows.get(point)

    def current_turn(self, point: str) -> int | None:
        row = self.rows.get(point)
        return row.turn_index if row else None

    def open_points(self) -> dict[str, StateRow]:
        return dict(self.open)

    def latest_update(self, turn_index: int | None = None) -> StateRow | None:
        rows = [r for r in self.history
                if turn_index is None or r.turn_index <= turn_index]
        return rows[-1] if rows else None

    def snapshot(self) -> dict[str, dict[str, object]]:
        """Serializable current state, for review queues and artefacts."""
        return {p: {"value": r.value, "status": r.status,
                    "turn": r.turn_index, "evidence": r.evidence}
                for p, r in sorted(self.rows.items())}


__all__ = ["DecisionState", "StateRow", "StateUpdate", "StateStatus"]
