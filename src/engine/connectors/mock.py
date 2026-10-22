"""Mock connectors (6.15) — they record proposals, they never execute.

Test guarantees (12.5): no network, no writes outside `runs/`, and every stored
record has `executed=false`. The journal is a JSONL audit trail, not a side
effect on any external system.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from engine.schemas import DraftEmail, EventProposal, TaskProposal, assert_no_execution


class MockConnector:
    """In-memory + JSONL journal under runs/. Nothing leaves the process."""

    tool = "mock"

    def __init__(self, journal: str | Path | None = None):
        self.records: list[dict[str, Any]] = []
        self.journal = Path(journal) if journal else None

    def submit(self, proposal: DraftEmail | TaskProposal | EventProposal,
               *, at: datetime | None = None) -> dict[str, Any]:
        assert_no_execution(proposal)  # refuses anything that claims to have run
        rec = {"tool": self.tool, "recorded_at": (at or datetime.now()).isoformat(),
               "status": "pending_confirmation", "proposal":
               proposal.model_dump(mode="json")}
        rec["proposal"]["executed"] = False
        rec["proposal"]["needs_confirmation"] = True
        self.records.append(rec)
        self._append(rec)
        return rec

    def _append(self, rec: dict[str, Any]) -> None:
        if self.journal is None:
            return
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def pending(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r["status"] == "pending_confirmation"]

    def executed(self) -> list[dict[str, Any]]:
        """Always empty — the whole point of the mock (6.15)."""
        return [r for r in self.records if r["proposal"].get("executed")]


class MockMail(MockConnector):
    tool = "email.create_draft"


class MockTasks(MockConnector):
    tool = "tasks.create"


class MockCalendar(MockConnector):
    tool = "calendar.create_event"


class MockDocs(MockConnector):
    tool = "docs.search"

    def search(self, query: str, hits: list[Any]) -> list[dict[str, Any]]:
        return [{"unit_id": h.unit.unit_id, "text": h.unit.text, "score": h.score}
                for h in hits]


class ConnectorSuite:
    def __init__(self, runs_dir: str | Path):
        base = Path(runs_dir)
        self.mail = MockMail(base / "connector_mail.jsonl")
        self.tasks = MockTasks(base / "connector_tasks.jsonl")
        self.calendar = MockCalendar(base / "connector_calendar.jsonl")
        self.docs = MockDocs(base / "connector_docs.jsonl")

    def submit(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, DraftEmail):
            return self.mail.submit(payload)
        if isinstance(payload, TaskProposal):
            return self.tasks.submit(payload)
        if isinstance(payload, EventProposal):
            return self.calendar.submit(payload)
        raise TypeError(f"not an actionable proposal: {type(payload).__name__}")

    @property
    def _all(self) -> list[MockConnector]:
        return [self.mail, self.tasks, self.calendar, self.docs]

    def pending(self) -> list[dict[str, Any]]:
        return [r for c in self._all for r in c.pending()]

    def all_records(self) -> list[dict[str, Any]]:
        return [r for c in self._all for r in c.records]

    def executed_total(self) -> int:
        return sum(len(c.executed()) for c in self._all)
