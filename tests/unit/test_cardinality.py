"""G12 — a set-valued slot accumulates; it never retires its older facts.

Graphiti's invalidation bug (issue #1728) retired 41% of a production graph,
mostly additive facts, because the judge had no notion of predicate cardinality.
The two shapes are otherwise identical: `plugin PART_OF plugin` is functional,
`benchmark TESTED_WITH model` is not.
"""
from __future__ import annotations

from datetime import date

import pytest

from engine.link.rules import propose
from engine.link.supersede import evaluate
from engine.schemas import Slots, Unit, utcnow
from engine.slots import rules as slot_rules

pytestmark = pytest.mark.unit


def _unit(uid, text, subject, *, day):
    u = Unit(unit_id=uid, source_id="s1", source_type="meeting",
             source_date=date(2026, 10, day), ingested_at=utcnow(),
             text=text, sentences=[], project="projet", turn_index=day,
             kind="info", type="decision", status="confirmed", slots=Slots())
    slots = slot_rules.extract(u, set()).model_copy(update={"subject": subject})
    return u.model_copy(update={"slots": slots})


def test_set_valued_slot_never_supersedes():
    old = _unit("u0", "Le film Dune est validé.", "projet.films", day=20)
    new = _unit("u1", "Le film Alien est validé.", "projet.films", day=24)
    assert propose(new, old) == "supersedes"
    assert evaluate(new, old, "supersedes") == ("supports", "G12_set_valued_additive")


def test_functional_slot_still_supersedes():
    old = _unit("u2", "Le lancement est validé le 12 décembre 2026.",
                "projet.launch_date", day=20)
    new = _unit("u3", "Le lancement est validé le 15 janvier 2027.",
                "projet.launch_date", day=24)
    assert evaluate(new, old, "supersedes") == ("supersedes", "OK")


def test_unknown_cardinality_is_allowed_by_default():
    old = _unit("u4", "La billetterie est validée en ligne.", "projet.billetterie",
                day=20)
    new = _unit("u5", "La billetterie est validée sur place.", "projet.billetterie",
                day=24)
    assert evaluate(new, old, "supersedes")[0] == "supersedes"


def test_unknown_cardinality_can_contest_instead_of_retire():
    """Opt-in: unknown is not silently functional; keep both and surface it."""
    old = _unit("u6", "La billetterie est validée en ligne.", "projet.billetterie",
                day=20)
    new = _unit("u7", "La billetterie est validée sur place.", "projet.billetterie",
                day=24)
    assert evaluate(new, old, "supersedes",
                    unknown_cardinality="contested") == (
                        "supports", "G12_unknown_cardinality")
