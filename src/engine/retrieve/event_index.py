"""Event index: event/slot membership, status and unresolved state."""
from __future__ import annotations

from collections import defaultdict

from engine.link.thread import detect_topic
from engine.threads.state_reader import StateReader, fold


class EventIndex:
    def __init__(self):
        self.members: dict[str, list[dict]] = defaultdict(list)
        self.unresolved: set[str] = set()

    @staticmethod
    def key(text: str, unit_id: str) -> str:
        topic = detect_topic(text)
        return f"{topic or 'other'}:{unit_id}"

    def add(self, unit: dict) -> str:
        topic = detect_topic(unit.get("text", "")) or "other"
        slot = unit.get("slot") or unit.get("type") or "mention"
        key = f"{topic}:{slot}"
        self.members[key].append(unit)
        if unit.get("status") in ("open", "unknown") or not unit.get("status"):
            self.unresolved.add(key)
        return key

    def current(self, key: str) -> dict | None:
        rows = self.members.get(key, [])
        good = [r for r in rows if r.get("status") in ("confirmed", "cancelled", "completed")]
        return max(good, key=lambda r: int(r.get("turn", 0))) if good else (
            max(rows, key=lambda r: int(r.get("turn", 0))) if rows else None)

    def search(self, text: str) -> list[dict]:
        topic = detect_topic(text)
        if not topic:
            return []
        out = []
        for key, rows in self.members.items():
            if key.startswith(topic + ":"):
                out.extend(rows)
        return out


__all__ = ["EventIndex"]
