"""Review-queue selection: one latest old per (new turn, canonical point)."""
from __future__ import annotations

import pytest

from engine.tools.session_changes import value_category, select_latest

pytestmark = pytest.mark.unit


def test_value_category_folds_accents_and_synonyms():
    assert value_category("début juillet") == "date_period"
    assert value_category("du lundi au vendredi") == "weekday"
    assert value_category("cinq jours") == "duration"
    assert value_category("mode de billetterie") == "mode"
    assert value_category("Sécurité et vigiles") == "securite"
    assert value_category("autre chose") == "autre"


def test_select_latest_keeps_one_old_per_new_and_point():
    rows = [
        {"old": 10, "new": 20, "point": "date", "old_value": "juin", "new_value": "juillet"},
        {"old": 14, "new": 20, "point": "date de départ", "old_value": "juin", "new_value": "juillet"},
        {"old": 12, "new": 20, "point": "lieu", "old_value": "salle Lyon",
         "new_value": "arene Paris"},
    ]
    got = select_latest(rows)
    assert [(r["old"], r["new"], r["point"]) for r in got] == [
        (14, 20, "date_period"), (12, 20, "lieu")]


def test_reversed_and_same_turn_pairs_are_dropped():
    rows = [
        {"old": 20, "new": 10, "point": "date", "old_value": "juin", "new_value": "juillet"},
        {"old": 10, "new": 10, "point": "date", "old_value": "juin", "new_value": "juillet"},
        {"old": 9, "new": 10, "point": "date", "old_value": "juin", "new_value": "juillet"},
    ]
    assert [r["old"] for r in select_latest(rows)] == [9]
