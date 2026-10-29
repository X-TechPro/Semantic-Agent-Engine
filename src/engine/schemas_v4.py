"""v4 contracts: immutable messages/chunks; append-only links/state.

Field sets follow docs/ARCH_V4.md sections 1-5. The label tuples are the single
source of truth for every v4 module; a value outside them is a bug, not a
variant. Nothing here imports a model: the contracts must stay importable
without torch.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

# --- label sets (ARCH_V4 §3, §4, §5) ----------------------------------------
KINDS: tuple[str, ...] = (
    "decision", "action", "commitment", "question", "answer", "fact", "risk",
    "idea",
)
STATUSES: tuple[str, ...] = (
    "proposed", "confirmed", "cancelled", "completed", "replaced", "unknown",
)
INTENTS: tuple[str, ...] = ("answer", "compare", "dossier", "email", "task",
                            "schedule", "clarify")
ENTITY_TYPES: tuple[str, ...] = (
    "person", "project", "object", "date", "amount", "action", "event",
)
RELATIONS: tuple[str, ...] = (
    "supersedes", "answers", "supports", "creates_action", "contradicts",
    "unrelated",
)
ORIGINS: tuple[str, ...] = ("real", "synthetic")

#: §5: only these statuses update the current view.
LIVE_STATUSES: frozenset[str] = frozenset({"confirmed", "cancelled", "completed"})
#: §5: a proposal stays open and can never displace a live value.
OPEN_STATUSES: frozenset[str] = frozenset({"proposed"})
#: §5: a soft unit never supersedes (same hard rule as v3, 6.2).
NON_ACTIONABLE_STATUSES: frozenset[str] = frozenset({"proposed", "unknown"})


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Evidence:
    """Character span inside a message — the proof a tag or link points at."""

    start: int
    end: int
    text: str

    @classmethod
    def whole(cls, text: str) -> "Evidence":
        return cls(0, len(text or ""), text or "")

    @classmethod
    def slice_of(cls, text: str, start: int, end: int) -> "Evidence":
        text = text or ""
        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        return cls(start, end, text[start:end])

    def as_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True)
class MessageRecord:
    """§1: one immutable row per received message."""

    message_id: str
    source_id: str
    project_id: str
    speaker: str
    timestamp: str
    text: str
    source_type: str = "meeting"
    origin: str = "real"
    ingested_at: str = field(default_factory=utcnow)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.content_hash:
            object.__setattr__(self, "content_hash", content_hash(self.text))
        if self.origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, got {self.origin!r}")
        if self.source_type not in ("meeting", "document", "code", "note"):
            raise ValueError(f"unknown source_type {self.source_type!r}")


@dataclass(frozen=True)
class ChunkRecord:
    """§2: exact sentence span. `message.text[start:end] == text` is invariant."""

    chunk_id: str
    message_id: str
    start: int
    end: int
    text: str
    sentence_index: int


@dataclass(frozen=True)
class SemanticUnit:
    """§3: a group of adjacent chunks read as one statement.

    `slot`/`value` are what §5 turns into a StateEvent; `evidence` is the span
    the tags were read from, not the whole message.
    """

    unit_id: str
    message_id: str
    chunk_ids: tuple[str, ...]
    text: str
    kind: str = "fact"
    status: str = "unknown"
    intent: str = "answer"
    entities: tuple[dict, ...] = ()
    owner: str = ""
    deadline: str = ""
    slot: str = ""
    point: str = ""
    value: str = ""
    evidence: Evidence | None = None
    confidence: float = 0.0
    model_version: str = "v4-rules"
    origin: str = "real"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {self.status!r}")
        if self.intent not in INTENTS:
            raise ValueError(f"intent must be one of {INTENTS}, got {self.intent!r}")
        for e in self.entities:
            if e.get("type") not in ENTITY_TYPES:
                raise ValueError(f"unknown entity type {e.get('type')!r}")

    def entity_values(self, etype: str | None = None) -> list[str]:
        return [str(e["value"]) for e in self.entities
                if etype is None or e.get("type") == etype]


@dataclass(frozen=True)
class LinkEvent:
    """§4/§5: appended typed edge. Never edited; a correction is a new edge."""

    link_id: str
    from_unit_id: str
    to_unit_id: str
    relation: str
    confidence: float
    evidence: str
    model_version: str = "v4"
    created_at: str = field(default_factory=utcnow)
    origin: str = "real"

    def __post_init__(self) -> None:
        if self.relation not in RELATIONS:
            raise ValueError(f"relation must be one of {RELATIONS}, "
                             f"got {self.relation!r}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class StateEvent:
    """§5: appended state transition. Current is a view over these rows."""

    state_id: str
    event_id: str
    slot: str
    value: str
    status: str
    evidence_unit: str
    created_at: str = field(default_factory=utcnow)
    origin: str = "real"

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, "
                             f"got {self.status!r}")


__all__ = [
    "ChunkRecord", "Evidence", "LinkEvent", "MessageRecord", "SemanticUnit",
    "StateEvent", "content_hash", "utcnow",
    "KINDS", "STATUSES", "INTENTS", "ENTITY_TYPES", "RELATIONS", "ORIGINS",
    "LIVE_STATUSES", "OPEN_STATUSES", "NON_ACTIONABLE_STATUSES",
]
