"""Linker guard tests — one per rule (BUILD_GUIDE 6.8.3, 12.1).

Each guard exists because of an observed failure; the test names record which.
"""

from __future__ import annotations

from datetime import date

import pytest

from engine.link.supersede import (SUPERSEDING_TYPES, VALUE_BEARING_TYPES,
                                   apply_supersede, comparable_values, evaluate,
                                   same_slot)
from engine.schemas import Slots
from helpers import make_unit

pytestmark = pytest.mark.unit


def U(text, *, date_, status="confirmed", type_="decision", subject="atlas.budget",
      amount=None, date_value=None, owner=None, kind="info", project="atlas",
      uid=None):
    return make_unit(text, source_date=date_, kind=kind, type=type_, status=status,
                     project=project,
                     slots=Slots(subject=subject, amount=amount,
                                 date_value=date_value, owner=owner),
                     unit_id=uid)


class TestG0CrossProject:
    def test_different_project_is_unrelated(self):
        a = U("Le budget d'Atlas est de 1,5 M€.", date_="2026-10-05", amount="1500000")
        b = U("Le budget de Boréal est de 2 M€.", date_="2026-10-01", amount="2000000",
              subject="boreal.budget", project="boreal")
        assert evaluate(a, b)[0] == "unrelated"

    def test_guard_name_is_recorded(self):
        a = U("x", date_="2026-10-05", subject="atlas.budget", amount="1")
        b = U("y", date_="2026-10-01", subject="boreal.budget", amount="2",
              project="boreal")
        assert evaluate(a, b)[1] == "G0_cross_project"


class TestG1SameSlot:
    def test_same_topic_other_attribute_is_unrelated(self):
        """12.3: budget vs launch date of the same project -> unrelated."""
        new = U("Finalement le budget du projet Atlas est de 2,4 M€.",
                date_="2026-10-12", subject="atlas.budget", amount="2400000")
        old = U("Le lancement du projet Atlas est prévu en décembre.",
                date_="2026-10-01", subject="atlas.launch_date", type_="fact",
                status="mention", date_value="2026-12-01")
        rel, guard = evaluate(new, old)
        assert rel == "unrelated"
        assert guard == "G1_different_attribute"

    def test_missing_slot_is_not_supersedeable(self):
        new = U("Le budget est révisé.", date_="2026-10-12", subject=None)
        old = U("Le budget est acté.", date_="2026-10-01", subject=None)
        assert evaluate(new, old)[0] == "unrelated"
        assert same_slot(new, old) is False


class TestG7ValueBearingSpeechActs:
    """Measured defect: `risk` and `commitment` units produced false supersedes
    (19 and 18 edges) because they share a subject key with value statements."""

    def test_risk_never_supersedes_a_decision(self):
        new = U("Le risque de conformité du projet Atlas est élevé.",
                date_="2026-10-05", type_="risk", subject="atlas.risk_compliance")
        old = U("La conformité du projet Atlas est validée.", date_="2026-10-01",
                subject="atlas.risk_compliance")
        rel, guard = evaluate(new, old)
        assert rel != "supersedes"
        assert guard == "G7_source_not_a_value_assertion"

    def test_nothing_supersedes_a_risk(self):
        new = U("Le budget du projet Atlas est révisé à 2,4 M€.",
                date_="2026-10-12", subject="atlas.risk_compliance", amount="2400000")
        old = U("Le risque du projet Atlas est moyen.", date_="2026-10-01",
                type_="risk", subject="atlas.risk_compliance")
        assert evaluate(new, old)[1] == "G7_target_not_value_bearing"

    def test_question_and_answer_are_not_supersedable_sources(self):
        assert "question" not in SUPERSEDING_TYPES
        assert "answer" not in SUPERSEDING_TYPES
        assert "risk" not in SUPERSEDING_TYPES
        assert "commitment" not in SUPERSEDING_TYPES
        assert "commitment" not in VALUE_BEARING_TYPES


class TestG8OnlyTheCurrentHead:
    def test_superseding_an_already_superseded_unit_is_supports(self):
        """6.8.4 writes old.valid_to, which only means something for the head."""
        new = U("Le budget du projet Atlas est de 3 M€.", date_="2026-10-20",
                amount="3000000")
        old = U("Le budget du projet Atlas est de 2 M€.", date_="2026-10-05",
                amount="2000000").model_copy(update={"valid_to": date(2026, 10, 20),
                                                     "superseded_by": "u-other"})
        assert old.is_current is False
        rel, guard = evaluate(new, old)
        assert rel == "supports"
        assert guard == "G8_target_not_current"

    def test_a_terse_value_statement_is_replaceable_even_without_status_words(self):
        """The rule this test used to assert was wrong, and it broke 12.3.

        It vetoed every `status == "mention"` target. But "Le lancement est prévu
        en décembre" carries no status keyword, so the reader labels it mention —
        meaning the veto silently blocked the flagship December->January update
        and left System B scoring identically to Baseline A on the kill test. The
        legitimate veto is by *speech act* (see TestG7), not by absence of a
        status word.
        """
        new = make_unit("Finalement le lancement du projet Atlas passe en janvier.",
                        unit_id="u-jan", source_date="2026-10-05", type="decision",
                        status="confirmed",
                        slots=Slots(subject="atlas.launch_date",
                                    date_value="2027-01-01"))
        old = make_unit("Le lancement du projet Atlas est prévu en décembre.",
                        unit_id="u-dec", source_date="2026-10-01", type="fact",
                        status="mention",
                        slots=Slots(subject="atlas.launch_date",
                                    date_value="2026-12-01"))
        rel, guard = evaluate(new, old)
        assert rel == "supersedes", (rel, guard)
        assert old.is_current and new.source_date > old.source_date

    def test_two_units_with_no_comparable_value_are_not_ordered(self):
        """With the mention veto gone, undecidability is still refused (G9)."""
        a = make_unit("On en reparle demain.", unit_id="u-v1",
                      source_date="2026-10-12", type="decision", status="confirmed",
                      slots=Slots(subject="atlas.budget"))
        b = make_unit("Encore un mot sur ce point.", unit_id="u-v2",
                      source_date="2026-10-01", type="fact", status="mention",
                      slots=Slots(subject="atlas.budget"))
        assert evaluate(a, b)[1] == "G9_values_not_comparable"


class TestG9ComparableValues:
    def test_no_comparable_value_is_undecidable(self):
        new = U("On en reparle.", date_="2026-10-12", subject="atlas.budget").model_copy(
            update={"slots": Slots(subject="atlas.budget")})
        old = U("Sujet sensible.", date_="2026-10-01", subject="atlas.budget").model_copy(
            update={"slots": Slots(subject="atlas.budget")})
        comparable, _ = comparable_values(new, old)
        assert comparable is False
        assert evaluate(new, old)[1] == "G9_values_not_comparable"

    def test_amounts_are_comparable_and_distinguish_minimal_pairs(self):
        """12.3: 1,5 M€ vs 15 M€ must not be read as the same value."""
        a = U("Le budget est de 1,5 M€.", date_="2026-10-05", amount="1500000")
        b = U("Le budget est de 15 M€.", date_="2026-10-01", amount="15000000")
        comparable, equal = comparable_values(a, b)
        assert comparable is True and equal is False

    def test_ratification_of_the_same_value_still_supersedes(self):
        """proposed -> confirmed with an unchanged value is a state change."""
        old = U("Je propose 1,5 M€ pour le budget d'Atlas.", date_="2026-10-01",
                status="proposed", type_="idea", amount="1500000")
        new = U("C'est acté, le budget d'Atlas est de 1,5 M€.", date_="2026-10-05",
                amount="1500000")
        comparable, equal = comparable_values(new, old)
        assert equal is True and old.status not in SUPERSEDING_TYPES
        assert evaluate(new, old)[0] == "supersedes"

    def test_restatement_of_a_confirmed_value_supports(self):
        old = U("Le budget d'Atlas est validé à 1,5 M€.", date_="2026-10-05",
                amount="1500000")
        new = U("Je le redis, le budget d'Atlas est de 1,5 M€.", date_="2026-10-12",
                 amount="1500000")
        rel, guard = evaluate(new, old)
        assert rel == "supports" and guard == "G4_restatement"


class TestG2DateOrder:
    def test_older_document_ingested_later_does_not_supersede(self):
        """12.3 out-of-order: source_date, not ingestion time, decides."""
        new = U("Le budget d'Atlas était de 1 M€.", date_="2026-09-01",
                amount="1000000")
        old = U("Le budget d'Atlas est de 2 M€.", date_="2026-10-01", amount="2000000")
        rel, guard = evaluate(new, old)
        assert rel == "contradicts" and guard == "G2_stale_source_date"


class TestG3Assertiveness:
    @pytest.mark.parametrize("status", ["hypothesis", "proposed", "conditional"])
    def test_non_assertive_never_supersedes_confirmed(self, status):
        old = U("Le budget d'Atlas est validé à 2 M€.", date_="2026-10-01",
                amount="2000000")
        new = U("On pourrait peut-être mettre 3 M€ au budget d'Atlas.",
                date_="2026-10-12", status=status, type_="idea", amount="3000000")
        rel = evaluate(new, old)[0]
        assert rel != "supersedes"

    def test_query_kind_cannot_supersede(self):
        old = U("Le budget d'Atlas est de 2 M€.", date_="2026-10-01", amount="2000000")
        new = U("Le budget d'Atlas est-il de 3 M€ ?", date_="2026-10-12", kind="query",
                type_="question", status="mention", amount="3000000")
        assert evaluate(new, old)[0] != "supersedes"


class TestG5SimultaneousConflict:
    def test_same_date_same_status_is_a_conflict_not_an_update(self):
        a = U("Le budget d'Atlas est de 2 M€.", date_="2026-10-05", amount="2000000")
        b = U("Le budget d'Atlas est de 3 M€.", date_="2026-10-05", amount="3000000")
        rel, guard = evaluate(a, b)
        assert rel == "contradicts" and guard == "G5_simultaneous_conflict"


class TestApplySupersede:
    def test_writes_valid_to_and_superseded_by_and_keeps_text(self):
        old = U("Le lancement d'Atlas est prévu en décembre.", date_="2026-10-01",
                date_value="2026-12-01")
        new = U("Finalement le lancement d'Atlas passe en janvier.",
                date_="2026-10-05", date_value="2027-01-01")
        upd = apply_supersede(old, new)
        assert upd.valid_to == date(2026, 10, 5)
        assert upd.superseded_by == new.unit_id
        assert upd.text == old.text            # history stays honest
        assert upd.status == old.status         # status is not rewritten
        assert upd.is_current is False

    def test_valid_to_is_never_before_valid_from(self):
        old = U("Le budget d'Atlas est de 2 M€.", date_="2026-10-01", amount="2000000")
        new = old.model_copy(update={"source_date": date(2026, 9, 1)})
        upd = apply_supersede(old, new)         # clamps, does not raise
        assert upd.valid_to >= upd.valid_from

    def test_chains_are_composable(self):
        a = U("budget 1 M€", date_="2026-10-01", amount="1000000", uid="u-a")
        b = U("budget 2 M€", date_="2026-10-05", amount="2000000", uid="u-b")
        c = U("budget 3 M€", date_="2026-10-12", amount="3000000", uid="u-c")
        a2 = apply_supersede(a, b)
        b2 = apply_supersede(b, c)
        assert a2.superseded_by == "u-b" and b2.superseded_by == "u-c"
        assert a2.is_current is False and b2.is_current is False
