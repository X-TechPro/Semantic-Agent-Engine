"""Contract tests for BUILD_GUIDE 2.3 — schemas validate/reject (12.1)."""

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from engine.schemas import (
    ACTION_OBJECT_TYPES, DraftEmail, Edge, EventProposal, Hit, Recipient, Slots,
    Span, TaskProposal, Unit, Clarification, Dossier, AnswerWithEvidence,
    assert_no_execution, make_unit_id, utcnow, Reading, Query, Filters,
)


def _unit(**kw):
    base = dict(
        unit_id="u-000123", source_id="meeting-03", source_type="meeting",
        source_date=date(2026, 10, 1), ingested_at=utcnow(),
        text="Le budget est validé.", sentences=[Span(start=0, end=20, surface="Le budget est validé.")],
    )
    base.update(kw)
    return Unit(**base)


class TestUnit:
    def test_minimal_unit_valid(self):
        u = _unit()
        assert u.kind is None and u.status is None
        assert u.slots == Slots()
        assert u.is_current is True

    def test_unknown_source_type_rejected(self):
        with pytest.raises(ValidationError):
            _unit(source_type="slack")

    def test_unknown_status_rejected(self):
        with pytest.raises(ValidationError):
            _unit(status="definitely_sure")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            _unit(embdding="x")

    def test_unit_is_frozen(self):
        u = _unit()
        with pytest.raises(ValidationError):
            u.text = "changed"

    def test_with_reader_returns_new_unit(self):
        u = _unit()
        v = u.with_reader("info", "decision", "confirmed", Slots(subject="atlas.budget"))
        assert v is not u
        assert u.kind is None
        assert v.status == "confirmed"
        assert v.slots.subject == "atlas.budget"

    def test_negative_span_rejected(self):
        with pytest.raises(ValidationError):
            Span(start=5, end=4, surface="x")

    def test_unit_id_is_deterministic_and_offset_derived(self):
        a = make_unit_id("meeting-03", 4, 0, 20, "Marie")
        b = make_unit_id("meeting-03", 4, 0, 20, "Marie")
        assert a == b
        assert a.startswith("u-")
        # different offset -> different id
        assert make_unit_id("meeting-03", 4, 21, 40, "Marie") != a
        # different source -> different id
        assert make_unit_id("meeting-04", 4, 0, 20, "Marie") != a
        # speaker is part of the key (units never cross speakers)
        assert make_unit_id("meeting-03", 4, 0, 20, "Karim") != a

    def test_can_supersede_rules(self):
        assert _unit(kind="info", status="confirmed").can_supersede is True
        assert _unit(kind="info", status="cancelled").can_supersede is True
        for s in ("hypothesis", "proposed", "conditional"):
            assert _unit(kind="info", status=s).can_supersede is False
        assert _unit(kind="query", status="confirmed").can_supersede is False

    def test_can_create_action_rules(self):
        for s in ("hypothesis", "proposed", "conditional"):
            assert _unit(status=s).can_create_action is False
        for s in ("confirmed", "cancelled", "mention"):
            assert _unit(status=s).can_create_action is True

    def test_roundtrip_json(self):
        u = _unit(kind="info", type="decision", status="confirmed",
                  slots=Slots(persons=["Marie"], subject="atlas.budget"))
        again = Unit.model_validate_json(u.model_dump_json())
        assert again == u


class TestActionSafety:
    def test_draft_email_never_executed(self):
        d = DraftEmail(
            recipient=Recipient(name="Marie", email="marie@example.org",
                                evidence_unit_ids=["u-1"], resolution_status="resolved"),
            subject="Budget Atlas", body="Bonjour Marie,", source_unit_ids=["u-1"],
        )
        assert d.needs_confirmation is True
        assert d.executed is False
        assert_no_execution(d)
        assert_no_execution(d.model_dump(mode="json"))

    def test_draft_email_cannot_claim_executed(self):
        with pytest.raises(ValidationError):
            DraftEmail(
                recipient=Recipient(name="Marie"), subject="s", body="b",
                executed=True,
            )

    def test_draft_email_cannot_drop_confirmation(self):
        with pytest.raises(ValidationError):
            DraftEmail(
                recipient=Recipient(name="Marie"), subject="s", body="b",
                needs_confirmation=False,
            )

    def test_recipient_email_may_be_null_but_status_is_closed_set(self):
        r = Recipient(name="Karim", email=None, resolution_status="email_missing")
        assert r.email is None
        with pytest.raises(ValidationError):
            Recipient(name="Karim", resolution_status="probably")

    def test_task_and_event_are_actions(self):
        t = TaskProposal(title="Envoyer le dossier", owner="Marie")
        e = EventProposal(title="Revue budget", start="2026-11-02")
        for obj in (t, e):
            assert obj.executed is False and obj.needs_confirmation is True
        assert len(ACTION_OBJECT_TYPES) == 3

    def test_assert_no_execution_catches_tampered_dict(self):
        with pytest.raises(AssertionError):
            assert_no_execution({"draft": {"executed": True}})
        with pytest.raises(AssertionError):
            assert_no_execution([{"needs_confirmation": False}])


class TestOtherContracts:
    def test_edge_score_bounds(self):
        with pytest.raises(ValidationError):
            Edge(from_id="a", to_id="b", relation="supersedes", score=1.4,
                 evidence_source_id="s", created_at=utcnow())

    def test_edge_relation_closed_set(self):
        with pytest.raises(ValidationError):
            Edge(from_id="a", to_id="b", relation="derives", score=0.5,
                 evidence_source_id="s", created_at=utcnow())

    def test_hit_carries_score_parts_and_path(self):
        h = Hit(unit=_unit(), score=0.5,
                score_parts={"dense": 0.4, "bm25": 0.1, "validity": 1.0},
                evidence="Le budget est validé.", path=["u-000123"])
        assert h.score_parts["validity"] == 1.0

    def test_claim_requires_at_least_one_unit(self):
        from engine.schemas import Claim

        with pytest.raises(ValidationError):
            Claim(text="sans preuve", unit_ids=[])

    def test_dossier_and_answer_constructible(self):
        d = Dossier(project="atlas", sections=[], references=[], markdown="")
        a = AnswerWithEvidence(answer=None, no_answer_reason="aucune preuve")
        c = Clarification(question="Qui est responsable ?", options=["Marie", "Karim"])
        assert d.project == "atlas" and a.answer is None and len(c.options) == 2

    def test_reading_defaults(self):
        r = Reading(unit_id="u-1", kind="query", type="question", status="mention")
        assert r.intent == "none" and r.also_info is False

    def test_query_and_filters(self):
        q = Query(text="Quand est le lancement ?", mode="history")
        f = Filters(projects=["atlas"], current_only=True)
        assert q.role == "query" and f.current_only is True

    def test_utcnow_is_timezone_aware(self):
        assert utcnow().tzinfo is not None
