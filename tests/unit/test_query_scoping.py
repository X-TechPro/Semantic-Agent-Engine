"""Query-side project scoping — the accent fold (D-038).

Measured 2026-10-22 by the hand-tester: "projet Boreal" (one missing accent)
became a hard project filter for a spelling the store does not hold, so the UI
answered "no results" while "projet Boréal" found the row. The resolver folds
accents against the store's own facets (a freshly opened UI/API process has
never ingested, so the session project set is empty), and an unknown project
stays unknown rather than borrowing another project's rows.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.agent.router import resolve_project_for_text

pytestmark = pytest.mark.unit

FACETS = {"projects": {"Boréal", "Atlas", "Iris"}, "speakers": set()}


def _fresh_engine():
    # like the UI/API process: nothing ingested here, the store knows the corpus
    return SimpleNamespace(projects=set(),
                           store=SimpleNamespace(facets=lambda: dict(FACETS)))


def test_unaccented_capture_resolves_to_stored_spelling():
    eng = _fresh_engine()
    q = ("Quelle décision a finalement été prise concernant le "
         "budget du projet Boreal ?")
    assert resolve_project_for_text(eng, q) == "Boréal"


def test_accented_capture_resolves_to_itself():
    eng = _fresh_engine()
    assert resolve_project_for_text(eng, "le budget du projet Boréal") == "Boréal"


def test_bare_mention_without_the_word_projet_resolves():
    eng = _fresh_engine()
    assert resolve_project_for_text(eng, "le budget de boreal a bougé") == "Boréal"


def test_unknown_project_stays_unknown():
    # honesty over helpfulness: a project the store has never seen must not
    # silently borrow another project's rows — the raw capture makes retrieval
    # abstain, which is the answerable answer
    eng = _fresh_engine()
    assert resolve_project_for_text(eng, "le budget du projet Chimère") == "Chimère"
