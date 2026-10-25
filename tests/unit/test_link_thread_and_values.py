"""Pair-local linker round (D-046): value witnesses, threads, guard scoping.

Pins the three things the round depends on:
1. spoken French values are readable at all (otherwise the guards see no change);
2. a coarse thread is deterministic and never merges two resolved subjects;
3. `same_thread=True` substitutes for a MISSING subject key only — the shipped
   default still refuses at G1, so every recorded number keeps its meaning.
"""
from __future__ import annotations

from datetime import date

import pytest

from engine.config import load_config
from engine.link.rules import propose
from engine.link.supersede import evaluate
from engine.link.thread import ThreadIndex
from engine.link.values import (has_change_cue, normalise_numbers,
                                pair_slot_witness, slot_signature, spoken_numbers,
                                value_witness)
from engine.schemas import Slots
from helpers import make_unit

pytestmark = pytest.mark.unit


def U(text, *, subject=None, type_="decision", status="mention", kind="info",
      date_="2021-10-05", project="053b", uid=None):
    return make_unit(text, source_date=date_, kind=kind, type=type_, status=status,
                     project=project, slots=Slots(subject=subject), unit_id=uid)


class TestSpokenValues:
    def test_spoken_numbers_are_read(self):
        assert normalise_numbers("le onze") == "le 11"
        assert normalise_numbers("le vingt et un juin") == "le 21 juin"
        assert "21" in spoken_numbers("on part sur vingt et un")
        assert "21" in spoken_numbers("vingt-et-un")

    def test_articles_are_not_numbers(self):
        assert spoken_numbers("un projet et une salle") == frozenset()

    def test_date_change_is_a_strong_witness(self):
        assert value_witness("sinon le vingt et un juin", "le onze") == "date"

    def test_negation_flip_is_a_witness(self):
        assert value_witness("non c'est bon mets maitre gims",
                             "non pas maitre gims") in ("negation", "number", None)
        assert value_witness("c'est bon mets maitre gims",
                             "non pas maitre gims") == "negation"

    def test_weak_witness_needs_shared_content(self):
        """Two stray numbers in one meeting must not become a replacement."""
        assert pair_slot_witness("on a neuf films", "il y a deux salles") is None
        assert pair_slot_witness("on a neuf films ce soir",
                                 "on a deux films ce soir") == "number"

    def test_change_cue(self):
        assert has_change_cue("finalement on repousse à janvier")
        assert not has_change_cue("le budget est de 2 millions")

    def test_slot_signature_prefers_attribute_then_typed_value(self):
        assert slot_signature("le budget du projet") == "a:budget"
        assert slot_signature("le vingt et un juin") == "date"
        assert slot_signature("ben euh ouais") is None


class TestThreadIndex:
    def test_units_sharing_a_coarse_slot_are_threaded(self):
        t = ThreadIndex(window=8)
        a = U("le onze", subject=None)
        b = U("le vingt et un juin", subject=None)
        t.add(a)
        assert t.context(b) == [a]
        assert t.same_thread(a, b)

    def test_different_resolved_subjects_never_merge(self):
        t = ThreadIndex(window=8)
        a = U("le budget est de 2 M€", subject="atlas.budget")
        b = U("le lancement est en juin", subject="atlas.launch_date")
        t.add(a)
        assert t.context(b) == []

    def test_window_bounds_the_thread(self):
        t = ThreadIndex(window=2)
        for i in range(5):
            t.add(U(f"le {i + 2} juin", subject=None))
        assert len(t.context(U("le 20 juin", subject=None))) == 2

    def test_named_topic_persists_across_turns(self):
        """216 says 'billetterie'; 218-226 never repeat it and still join."""
        t = ThreadIndex(window=8, topic_window=12)
        a = U("on met en place une billetterie en ligne", uid="a")
        b = U("en ligne ça peut être sympa", uid="b")
        c = U("une réservation en ligne et un stand à l'entrée", uid="c")
        for u in (a, b, c):
            t.add(u)
        assert t.topic_of(b) == "billetterie"
        assert t.same_thread(a, b) and t.same_thread(a, c)

    def test_a_new_topic_starts_a_new_thread(self):
        t = ThreadIndex(window=8, topic_window=12)
        a = U("on parle de la billetterie", uid="a")
        b = U("maintenant la sécurité et les bouchons", uid="b")
        c = U("il faut deux vigiles", uid="c")
        for u in (a, b, c):
            t.add(u)
        assert not t.same_thread(a, c)
        assert t.same_thread(b, c)
        assert t.topic_of(c) == "securite"

    def test_a_topic_ages_out(self):
        t = ThreadIndex(window=8, topic_window=1)
        a = U("la billetterie en ligne", uid="a")
        b = U("ben ouais", uid="b")
        c = U("autre chose", uid="c")
        for u in (a, b, c):
            t.add(u)
        assert t.topic_of(b) == "billetterie"
        assert t.topic_of(c) is None


class TestPairLocalGuard:
    def test_shipped_default_still_refuses_a_missing_subject(self):
        new = U("sinon le vingt et un juin")
        old = U("le onze")
        rel, guard = evaluate(new, old, "supersedes")
        assert rel == "unrelated" and guard == "G1_no_slot"

    def test_different_resolved_subjects_still_refuse_even_in_thread(self):
        new = U("sinon le vingt et un juin", subject="053b.launch_date")
        old = U("le onze", subject="053b.budget")
        rel, guard = evaluate(new, old, "supersedes", same_thread=True)
        assert rel == "unrelated" and guard == "G1_different_attribute"

    def test_pair_local_supersede_with_a_date_witness_and_cue(self):
        new = U("sinon le vingt et un juin")
        old = U("le onze")
        rel, guard = evaluate(new, old, "supersedes", same_thread=True)
        assert rel == "supersedes", guard
        assert guard == "OK_pair_date"

    def test_pair_local_needs_a_change_cue_to_stand_in_for_status(self):
        new = U("le vingt et un juin")          # no cue, status still mention
        old = U("le onze")
        rel, guard = evaluate(new, old, "supersedes", same_thread=True)
        assert rel == "contradicts" and guard == "G3_not_assertive"

    def test_proposer_only_offers_supersedes_when_threaded(self):
        new = U("sinon le vingt et un juin")
        old = U("le onze")
        assert propose(new, old) == "unrelated"
        assert propose(new, old, thread=True) == "supersedes"


def test_thread_context_ships_off():
    """The measured round is refused; the shipped behaviour must not move."""
    assert load_config()["link"]["thread_context"] is False
    assert load_config()["link"]["thread_window"] == 8
