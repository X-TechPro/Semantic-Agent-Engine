"""`completes` — a later turn carries out an earlier plan or request.

This is the "pizza" case from D-049: the link is invisible while the object is
a pronoun, and becomes visible once coreference resolves it. The cue alone is
never enough; the pair must share a content noun.
"""
from __future__ import annotations

from datetime import date

import pytest

from engine.link.rules import RuleLinker, propose
from engine.schemas import Slots, Unit, utcnow

pytestmark = pytest.mark.unit


def _unit(uid, text, *, type_, turn, subject=None):
    return Unit(unit_id=uid, source_id="pizza", source_type="meeting",
                source_date=date(2026, 10, 24), ingested_at=utcnow(),
                text=text, sentences=[], project="pizza", turn_index=turn,
                kind="info", type=type_, status="mention",
                slots=Slots(subject=subject))


OLD = _unit("u0", "Est-ce que quelqu'un veut manger des pizzas ce midi ?",
            type_="question", turn=0)
RESOLVED = _unit("u1", "C'est bon, j'ai acheté les pizzas hier soir.",
                 type_="fact", turn=1)
PRONOUN = _unit("u2", "C'est bon, je les ai achetées hier soir.",
                type_="fact", turn=1)


def test_resolved_object_completes_the_plan():
    assert propose(RESOLVED, OLD) == "completes"


def test_unresolved_pronoun_cannot_complete():
    assert propose(PRONOUN, OLD) == "unrelated"


def test_cue_without_a_shared_object_is_not_a_link():
    other = _unit("u3", "Le budget est confirmé.", type_="decision", turn=0)
    done = _unit("u4", "C'est bon, j'ai acheté le pain.", type_="fact", turn=1)
    assert propose(done, other) == "unrelated"


def test_link_writes_the_edge():
    edges = RuleLinker().link(RESOLVED, [OLD])
    assert [(e.relation, e.guard) for e in edges] == [("completes", "G0_model")]


def test_question_answer_still_wins_over_completes():
    answer = _unit("u5", "Oui, les pizzas sont commandées.", type_="answer", turn=1)
    assert propose(answer, OLD) == "answers"
