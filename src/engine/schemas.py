"""Data contracts — BUILD_GUIDE 2.3. These types are the only things that cross
stage boundaries (the "lego rule", 2.2). Changing a field here requires updating
`schemas/*.json`, the tests, and the changelog in IDEA.md 17.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- label sets (single source of truth; exported to schemas/*.json) ---------
SourceType = Literal["meeting", "document", "code", "note"]
Kind = Literal["info", "query"]
TypeLabel = Literal[
    "decision", "action", "commitment", "question", "answer",
    "fact", "risk", "idea", "other",
]
StatusLabel = Literal[
    "proposed", "hypothesis", "conditional", "confirmed", "cancelled", "mention",
]
IntentLabel = Literal[
    "answer", "compare", "build_dossier", "draft_email", "create_task",
    "schedule_event", "clarify", "none",
]
Relation = Literal[
    "supports", "contradicts", "supersedes", "creates_action", "answers",
    "completes", "unrelated",
]
QueryMode = Literal["current", "history", "compare"]
ResolutionStatus = Literal["resolved", "email_missing", "ambiguous", "not_found"]

SOURCE_TYPES: tuple[str, ...] = ("meeting", "document", "code", "note")
KINDS: tuple[str, ...] = ("info", "query")
TYPES: tuple[str, ...] = (
    "decision", "action", "commitment", "question", "answer", "fact", "risk",
    "idea", "other",
)
STATUSES: tuple[str, ...] = (
    "proposed", "hypothesis", "conditional", "confirmed", "cancelled", "mention",
)
INTENTS: tuple[str, ...] = (
    "answer", "compare", "build_dossier", "draft_email", "create_task",
    "schedule_event", "clarify", "none",
)
RELATIONS: tuple[str, ...] = (
    "supports", "contradicts", "supersedes", "creates_action", "answers",
    "completes", "unrelated",
)

# 6.2 / 6.8 hard rule: these statuses may never supersede, and may never create
# an action object (task / event / draft).
NON_ACTIONABLE_STATUSES: frozenset[str] = frozenset({"hypothesis", "proposed", "conditional"})
SUPERSEDING_STATUSES: frozenset[str] = frozenset({"confirmed", "cancelled"})

PositiveInt = Annotated[int, Field(ge=0)]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def make_unit_id(source_id: str, seq: int, start: int, end: int,
                 speaker: str | None = None) -> str:
    """Deterministic unit id: a function of (source_id, offset) only (2.3).

    Never derived from the text, so a re-read of the same source yields the
    same ids; never derived from wall-clock, so runs are reproducible.
    """
    key = f"{source_id}|{seq}|{start}|{end}|{speaker or ''}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return f"u-{digest}{seq:04d}"


class _Model(BaseModel):
    """Frozen-by-default base: stages must not mutate each other's payloads."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True,
                              frozen=True)


class Span(_Model):
    """A character span inside a unit's `text` — the evidence granularity.

    Invariant (6.1 acceptance: 100% span reconstruction): `text[start:end]`
    equals `surface` after the segmenter's normalisation.
    """

    start: PositiveInt
    end: PositiveInt
    surface: str

    @field_validator("end")
    @classmethod
    def _ordered(cls, end: int, info) -> int:
        start = info.data.get("start", 0)
        if end < start:
            raise ValueError(f"span end {end} < start {start}")
        return end

    def slice(self, text: str) -> str:
        return text[self.start:self.end]

    @classmethod
    def whole(cls, text: str) -> "Span":
        """A span covering an entire string — the common single-sentence case."""
        return cls(start=0, end=len(text), surface=text)


class Slots(_Model):
    persons: list[str] = Field(default_factory=list)
    owner: str | None = None
    subject: str | None = None
    date_value: str | None = None
    amount: str | None = None
    due: str | None = None
    action: str | None = None


class Unit(_Model):
    unit_id: str
    source_id: str
    source_type: SourceType
    source_date: date
    ingested_at: datetime
    text: str
    sentences: list[Span] = Field(default_factory=list)
    speaker: str | None = None
    project: str | None = None
    turn_index: int | None = None
    # filled by reader
    kind: Kind | None = None
    type: TypeLabel | None = None
    status: StatusLabel | None = None
    slots: Slots = Field(default_factory=Slots)
    # filled by store / linker
    valid_from: date | None = None
    valid_to: date | None = None
    superseded_by: str | None = None
    embed_model_version: str | None = None
    # ledger provenance (synthetic gold; None for real corpora)
    gold_event_ids: list[str] = Field(default_factory=list)

    def with_reader(self, kind: Kind | None, type_: TypeLabel | None,
                   status: StatusLabel | None, slots: Slots,
                   intent: IntentLabel | None = None) -> "Unit":
        """Reader output merged into the unit (stages return new values)."""
        return self.model_copy(
            update={"kind": kind, "type": type_, "status": status, "slots": slots}
        )

    @property
    def is_current(self) -> bool:
        return self.valid_to is None and self.superseded_by is None

    @property
    def can_supersede(self) -> bool:
        return self.kind == "info" and self.status in SUPERSEDING_STATUSES

    @property
    def can_create_action(self) -> bool:
        """6.2 hard rule: hypothesis / proposed / conditional never act."""
        return self.status not in NON_ACTIONABLE_STATUSES


class Source(_Model):
    """An ingested artefact: one meeting, one document, one code file."""

    source_id: str
    source_type: SourceType
    source_date: date
    title: str | None = None
    project: str | None = None
    default_speaker: str | None = None
    turns: list["Turn"] = Field(default_factory=list)
    text: str | None = None
    code: str | None = None
    language: str | None = None
    gold_event_ids: list[str] = Field(default_factory=list)

    def body(self) -> str:
        if self.source_type in ("meeting", "note"):
            return "\n".join(t.text for t in self.turns)
        if self.source_type == "code":
            return self.code or ""
        return self.text or ""


class Turn(_Model):
    speaker: str | None = None
    text: str = ""
    timestamp: datetime | None = None
    turn_index: int | None = None
    #: the project this turn is *about*. It may differ from Source.project: a
    #: distractor turn discusses another project inside this meeting, and units
    #: must inherit the discussed project or cross-project supersede guards
    #: (6.8.3 G0) can never fire.
    project: str | None = None
    gold_event_ids: list[str] = Field(default_factory=list)


Source.model_rebuild()


class ReadContext(_Model):
    """Optional window of previously read units (6.2.3 context ablation)."""

    previous: list[Unit] = Field(default_factory=list)
    project: str | None = None
    default_speaker: str | None = None


class Reading(_Model):
    """The reader's verdict for one unit."""

    unit_id: str
    kind: Kind
    type: TypeLabel
    status: StatusLabel
    intent: IntentLabel = "none"
    slots: Slots = Field(default_factory=Slots)
    probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    # 6.2.4: a query unit that also asserts something is stored as info too
    also_info: bool = False
    abstained: bool = False


class Edge(_Model):
    from_id: str
    to_id: str
    relation: Relation
    score: float = Field(ge=0.0, le=1.0)
    evidence_source_id: str
    created_at: datetime
    guard: str | None = None  # which deterministic rule fired, for debugging


class Query(_Model):
    text: str
    role: Literal["query"] = "query"
    mode: QueryMode = "current"
    project: str | None = None
    intent: IntentLabel = "answer"
    slots: Slots = Field(default_factory=Slots)
    source_unit_id: str | None = None
    as_of: date | None = None
    #: the reader's verdict on the request itself (6.2: the request side of
    #: "hypotheses never act" — a hypothetical QUERY must not act even when
    #: the store's lead unit for the subject is confirmed)
    status: StatusLabel | None = None


class Filters(_Model):
    projects: list[str] = Field(default_factory=list)
    source_types: list[SourceType] = Field(default_factory=list)
    statuses: list[StatusLabel] = Field(default_factory=list)
    types: list[TypeLabel] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    current_only: bool | None = None
    subject_prefix: str | None = None

    def is_empty(self) -> bool:
        return not (self.projects or self.source_types or self.statuses or
                    self.types or self.speakers or self.date_from or
                    self.date_to or self.current_only or self.subject_prefix)


class Hit(_Model):
    unit: Unit
    score: float
    score_parts: dict[str, float] = Field(default_factory=dict)
    evidence: str = ""
    path: list[str] = Field(default_factory=list)
    rank: int | None = None


# --- agent action objects ---------------------------------------------------
class Recipient(_Model):
    name: str
    email: str | None = None
    evidence_unit_ids: list[str] = Field(default_factory=list)
    resolution_status: ResolutionStatus = "not_found"
    justification: str = ""
    alternatives: list[str] = Field(default_factory=list)


class _Action(_Model):
    """Every action is a proposal. Nothing executes (BUILD_GUIDE 0.9, 6.15)."""

    needs_confirmation: Literal[True] = True
    executed: Literal[False] = False


class DraftEmail(_Action):
    intent: Literal["draft_email"] = "draft_email"
    tool_candidate: Literal["email.create_draft"] = "email.create_draft"
    recipient: Recipient
    subject: str
    body: str
    source_unit_ids: list[str] = Field(default_factory=list)


class TaskProposal(_Action):
    intent: Literal["create_task"] = "create_task"
    tool_candidate: Literal["tasks.create"] = "tasks.create"
    title: str
    owner: str | None = None
    due: str | None = None
    project: str | None = None
    description: str = ""
    source_unit_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class EventProposal(_Action):
    intent: Literal["schedule_event"] = "schedule_event"
    tool_candidate: Literal["calendar.create_event"] = "calendar.create_event"
    title: str
    start: str | None = None
    end: str | None = None
    attendees: list[str] = Field(default_factory=list)
    project: str | None = None
    source_unit_ids: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class Clarification(_Model):
    question: str
    options: list[str] = Field(default_factory=list)
    reason: str = ""


class Claim(_Model):
    text: str
    unit_ids: list[str] = Field(min_length=1)


class DossierSection(_Model):
    title: str
    claims: list[Claim] = Field(default_factory=list)


class Dossier(_Model):
    project: str | None = None
    sections: list[DossierSection] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    markdown: str = ""


class AnswerWithEvidence(_Model):
    answer: str | None = None
    claims: list[Claim] = Field(default_factory=list)
    no_answer_reason: str | None = None
    status_breakdown: dict[str, list[str]] = Field(default_factory=dict)
    superseded_history: list[dict[str, Any]] = Field(default_factory=list)
    query_mode: QueryMode = "current"


class ComparisonItem(_Model):
    subject: str
    relation: Relation
    explanation: str = ""
    unit_ids: list[str] = Field(default_factory=list)


class Comparison(_Model):
    left_source: str
    right_source: str
    unchanged: list[ComparisonItem] = Field(default_factory=list)
    changed: list[ComparisonItem] = Field(default_factory=list)
    superseded: list[ComparisonItem] = Field(default_factory=list)
    contradicted: list[ComparisonItem] = Field(default_factory=list)
    summary: str = ""


class ActionResult(_Model):
    """Uniform envelope returned by `act()` (2.2)."""

    intent: IntentLabel
    operation: str
    payload: dict[str, Any]
    hit_unit_ids: list[str] = Field(default_factory=list)
    clarification: Clarification | None = None
    used_llm: bool = False
    degraded_to_template: bool = False
    notes: list[str] = Field(default_factory=list)

    @property
    def model(self) -> BaseModel:
        return _PAYLOAD_MODELS.get(self.operation, None)  # type: ignore[arg-type]


_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {}


def _register_payload_models() -> None:
    _PAYLOAD_MODELS.update({
        "answer": AnswerWithEvidence,
        "compare": Comparison,
        "build_dossier": Dossier,
        "draft_email": DraftEmail,
        "create_task": TaskProposal,
        "schedule_event": EventProposal,
        "clarify": Clarification,
    })


_register_payload_models()


class IngestReport(_Model):
    source_id: str
    n_units: int = 0
    n_stored: int = 0
    n_skipped: int = 0
    n_queries: int = 0
    n_edges: int = 0
    n_superseded: int = 0
    relations: dict[str, int] = Field(default_factory=dict)
    elapsed_s: float = 0.0


# --- safety helpers (used by tests and by the agent post-check) -------------
ACTION_OBJECT_TYPES: tuple[type[_Model], ...] = (DraftEmail, TaskProposal,
                                                 EventProposal)


def assert_no_execution(obj: Any, path: str = "root") -> None:
    """BUILD_GUIDE 0.9 / 12.5: no action object may ever claim it executed."""
    if isinstance(obj, _Action):
        if obj.executed is not False:
            raise AssertionError(f"{path}: executed must be False")
        if obj.needs_confirmation is not True:
            raise AssertionError(f"{path}: needs_confirmation must be True")
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("executed",) and v is not False:
                raise AssertionError(f"{path}.{k}: executed must be False")
            if k in ("needs_confirmation",) and v is not True:
                raise AssertionError(f"{path}.{k}: needs_confirmation must be True")
            assert_no_execution(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_no_execution(v, f"{path}[{i}]")


def model_to_json(obj: BaseModel) -> dict[str, Any]:
    return obj.model_dump(mode="json")


__all__ = [
    "ACTION_OBJECT_TYPES", "AnswerWithEvidence", "ActionResult", "AssertionError",
    "Claim", "Clarification", "Comparison", "ComparisonItem", "Dossier",
    "DossierSection", "DraftEmail", "Edge", "EventProposal", "Filters", "Hit",
    "IngestReport", "Query", "ReadContext", "Reading", "Recipient", "Relation",
    "Source", "Span", "Slots", "TaskProposal", "Turn", "Unit", "utcnow",
    "make_unit_id", "model_to_json", "assert_no_execution",
    "KINDS", "TYPES", "STATUSES", "INTENTS", "RELATIONS", "SOURCE_TYPES",
    "NON_ACTIONABLE_STATUSES", "SUPERSEDING_STATUSES",
]
