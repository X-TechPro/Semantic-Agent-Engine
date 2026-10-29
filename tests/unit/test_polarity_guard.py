"""G13 polarity + strict turn order on the supersede path.

The two B-13 holes this pins:
* same typed value + opposite polarity is a contradiction, never a support or a
  replacement ("budget 2 M€" vs "budget n'est pas 2 M€");
* a later turn cannot supersede an earlier one when both are in the same source
  and the meeting carries one `source_date`.

The layer stays deterministic; no model or extra dependency is involved.
"""
from __future__ import annotations

import pytest

from engine.link.rules import propose
from engine.link.supersede import evaluate, turn_order_ok
from engine.link.values import polarity
from engine.schemas import Slots
from helpers import make_unit

pytestmark = pytest.mark.unit


def U(text, *, subject="atlas.budget", status="mention", amount=None,
      turn=0, source_id="meeting-01", uid=None):
    return make_unit(text, source_id=source_id, unit_id=uid,
                     source_date="2026-10-05", kind="info", type="decision",
                     status=status, project="atlas",
                     slots=Slots(subject=subject, amount=amount),
                     turn_index=turn)


class TestPolarity:
    def test_affirmative_and_negative_values(self):
        assert polarity("Le budget est de 2 M€") == "affirmative"
        assert polarity("Le budget n'est pas de 2 M€") == "negative"
        assert polarity("on n'a plus de budget") == "negative"
        assert polarity("Ce n'est pas encore validé") == "negative"

    def test_ambiguous_plus_and_affirmative_sans(self):
        assert polarity("il faut plus de budget") is None
        assert polarity("sans budget") == "negative"
        assert polarity("sans doute") == "affirmative"
        assert polarity("pas mal") == "affirmative"

    def test_scope_follows_the_subject_clause(self):
        assert polarity("Le budget est de 2 M€, mais la salle n'est pas prête",
                        "atlas.budget") == "affirmative"
        assert polarity("La salle n'est pas prête, le budget est de 2 M€",
                        "atlas.budget") == "affirmative"

    def test_non_pas_is_negative(self):
        assert polarity("non pas maître gims") == "negative"


class TestG13:
    def test_same_value_opposite_polarity_is_a_contradiction(self):
        old = U("Le budget est de 2 M€", amount="2000000", turn=0)
        new = U("Le budget n'est pas de 2 M€", amount="2000000",
                status="confirmed", turn=1)
        assert propose(new, old) == "supersedes"   # routed to the guards
        rel, guard = evaluate(new, old, "supersedes")
        assert rel == "contradicts" and guard == "G13_polarity"

    def test_a_new_typed_value_still_supersedes(self):
        old = U("Le budget est de 2 M€", amount="2000000", turn=0)
        new = U("Le budget n'est pas de 2 M€, il est de 3 millions",
                amount="3000000", status="confirmed", turn=1)
        rel, guard = evaluate(new, old, "supersedes")
        assert rel == "supersedes", guard

    def test_an_explicit_positive_correction_is_allowed(self):
        old = U("Le budget n'est pas de 2 M€", amount="2000000", turn=0)
        new = U("Finalement le budget est de 2 M€", amount="2000000",
                status="confirmed", turn=1)
        rel, guard = evaluate(new, old, "supersedes")
        assert rel == "supersedes", guard

    def test_a_bare_negation_never_becomes_a_support(self):
        old = U("Le budget est de 2 M€", amount="2000000", turn=0)
        new = U("Ce n'est pas encore validé", amount=None,
                status="confirmed", turn=1)
        rel, guard = evaluate(new, old, "supersedes")
        assert rel == "contradicts" and guard == "G13_polarity"


class TestTurnOrder:
    def test_same_source_later_turn_is_required(self):
        old = U("Le budget est de 2 M€", amount="2000000", turn=5)
        new = U("Le budget est de 3 M€", amount="3000000",
                status="confirmed", turn=2)
        assert not turn_order_ok(new, old)
        rel, guard = evaluate(new, old, "supersedes")
        assert rel == "contradicts" and guard == "G2_turn_order"

    def test_same_source_later_turn_passes(self):
        old = U("Le budget est de 2 M€", amount="2000000", turn=5)
        new = U("Le budget est de 3 M€", amount="3000000",
                status="confirmed", turn=6)
        assert turn_order_ok(new, old)
        rel, guard = evaluate(new, old, "supersedes")
        assert rel == "supersedes", guard

    def test_unknown_or_equal_turn_is_not_ordered(self):
        old = U("Le budget est de 2 M€", amount="2000000", turn=0)
        new = U("Le budget est de 3 M€", amount="3000000", turn=0)
        assert turn_order_ok(new, old)

    def test_different_sources_use_dates_not_turn_numbers(self):
        old = U("Le budget est de 2 M€", amount="2000000", turn=5,
                source_id="meeting-a")
        new = U("Le budget est de 3 M€", amount="3000000", turn=2,
                source_id="meeting-b")
        assert turn_order_ok(new, old)
