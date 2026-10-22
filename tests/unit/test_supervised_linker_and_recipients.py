"""The trained linker's proposal step (6.8b) and recipient resolution (6.12).

`link/supervised.py` sat at 63% and `agent/recipients.py` at 79% — the two
lowest offline modules left, and both hold an invariant the project's claims
depend on:

* **A model may propose, never authorise.** `SupervisedLinker` replaces only
  `propose`; `link()` still runs `supersede.evaluate` afterwards. The test below
  builds a classifier deliberately trained to say "supersedes" about a pair whose
  slots differ, and asserts the guard still refuses it. Without that pairing the
  6.11 sentence "a model never overrules a guard" is a comment, not a behaviour.
* **`build_linker` must name the backend it actually gave you** (D-011). The
  fallback branch used to be able to claim "trained" while running rules.
* **No address is ever invented** (6.12.4): the email comes only from the mock
  directory, and a known person without one becomes `email_missing` with
  `email=None`. The untested lines were precisely that branch and the
  ambiguity/not-found branches around it.
"""

from __future__ import annotations

import json

import pytest

from engine.agent.recipients import ContactBook, candidates, resolve_recipient
from engine.config import load_config
from engine.link.model import FEATURES, PairClassifier
from engine.link.rules import RuleLinker
from engine.link.supervised import SupervisedLinker, build_linker
from engine.schemas import Slots
from helpers import make_unit

pytestmark = pytest.mark.unit


def _clf(*, weight_on: str = "same_project", w: float = 12.0, bias: float = -2.0,
         threshold: float = 0.5) -> PairClassifier:
    weights = [0.0] * len(FEATURES)
    weights[FEATURES.index(weight_on)] = w
    return PairClassifier(weights, bias, threshold=threshold)


@pytest.fixture
def same_slot_pair():
    """The pair the rules linker *does* want to supersede."""
    new = make_unit("Le budget est finalement à 2,4 M€.", unit_id="u-new",
                    source_date="2026-06-01", type="decision", status="confirmed",
                    slots=Slots(subject="atlas.budget.value", amount="2,4 M€"))
    old = make_unit("Le budget est à 1,8 M€.", unit_id="u-old",
                    source_date="2026-01-05", type="fact", status="confirmed",
                    slots=Slots(subject="atlas.budget.value", amount="1,8 M€"))
    return new, old


@pytest.fixture
def different_slot_pair():
    """Same project, a different attribute — the rules refuse it as `unrelated`."""
    new = make_unit("Le budget est finalement à 2,4 M€.", unit_id="u-new",
                    source_date="2026-06-01", type="decision", status="confirmed",
                    slots=Slots(subject="atlas.budget.value", amount="2,4 M€"))
    other = make_unit("Le prestataire retenu est Nova Cloud.", unit_id="u-vendor",
                      source_date="2026-06-01", type="decision", status="confirmed",
                      slots=Slots(subject="atlas.vendor.value"))
    return new, other


# ------------------------------------------------------- proposal vs authorisation

def test_rules_proposal_short_circuits_the_model_and_records_the_route(same_slot_pair):
    new, old = same_slot_pair
    sl = SupervisedLinker(_clf())
    assert sl.name == "logreg"
    assert sl.propose(new, old) == "supersedes"
    # a pair the rules already accept is not sent to the model: the recorded score
    # is the rules' 1.0 and the route says "rules", so an audit can tell them apart
    assert sl.last_scores == [(new.unit_id, old.unit_id, 1.0, "rules")]


def test_the_model_proposes_but_the_guard_still_refuses(different_slot_pair):
    """6.11 measured, not asserted in a docstring.

    The classifier below is hand-built to fire on `same_project` alone, so it
    scores the cross-slot pair 1.0 and *does* propose a supersede that the rules
    refused. `link()` must still consult `supersede.evaluate`, and the edge it
    writes must carry the guard's verdict rather than the model's.
    """
    new, other = different_slot_pair
    sl = SupervisedLinker(_clf())
    assert sl.propose(new, other) == "supersedes"      # the model's proposal
    assert sl.last_scores[-1][3] == "model"

    edges = sl.link(new, [other])
    assert all(e.relation != "supersedes" for e in edges), \
        f"the model overruled the guard: {[ (e.relation, e.guard) for e in edges ]}"
    # every recorded edge keeps its guard name, so the refusal is auditable
    assert all(e.guard for e in edges), [(e.relation, e.guard) for e in edges]


def test_a_low_model_score_leaves_the_rules_verdict_alone(same_slot_pair):
    new, _old = same_slot_pair
    far = make_unit("Le risque de conformité reste élevé.", unit_id="u-risk",
                    source_date="2026-02-02", type="risk", status="confirmed",
                    slots=Slots(subject="atlas.risk.value"))
    weak = _clf(w=0.0, bias=-4.0)                     # everything scores ~0.018
    sl = SupervisedLinker(weak)
    rel = sl.propose(new, far)
    assert sl.last_scores[-1] == (new.unit_id, far.unit_id,
                                  pytest.approx(weak.score(new, far)), "model")
    assert rel != "supersedes", "a sub-threshold score must not invent a supersede"


# ------------------------------------------------------------------ build_linker

def test_build_linker_falls_back_to_rules_and_says_so(tmp_path):
    """D-011: a run must not be able to claim 'trained' while using rules."""
    cfg = load_config(overrides={"link": {"backend": "logreg",
                                          "model_path": str(tmp_path / "absent.json")}})
    linker, label = build_linker(cfg)
    assert isinstance(linker, RuleLinker) and not isinstance(linker, SupervisedLinker)
    assert label.startswith("rules[fallback: no model at")
    assert linker.name == "rules"


def test_build_linker_uses_a_real_model_file_and_names_it(tmp_path):
    path = tmp_path / "clf.json"
    _clf().save(path)
    cfg = load_config(overrides={"link": {"backend": "logreg", "model_path": str(path)}})
    linker, label = build_linker(cfg)
    assert isinstance(linker, SupervisedLinker)
    assert label == f"logreg({path.name})"
    assert linker.min_score == cfg["link"]["supersede_min_score"]
    # an alias of the same backend must land on the same branch, not on rules
    alias = load_config(overrides={"link": {"backend": "supervised",
                                            "model_path": str(path)}})
    assert isinstance(build_linker(alias)[0], SupervisedLinker)
    rules_cfg = load_config(overrides={"link": {"backend": "rules"}})
    assert build_linker(rules_cfg)[1] == "rules"


def test_a_model_file_with_a_stale_feature_order_is_refused(tmp_path):
    """Refusing beats predicting with weights bound to the wrong columns."""
    path = tmp_path / "stale.json"
    path.write_text(json.dumps({"features": ["nope"], "weights": [1.0], "bias": 0.0}),
                    encoding="utf-8")
    cfg = load_config(overrides={"link": {"backend": "model", "model_path": str(path)}})
    linker, label = build_linker(cfg)
    assert isinstance(linker, RuleLinker)
    assert label.startswith("rules[fallback")


# --------------------------------------------------------------- recipient (6.12)

@pytest.fixture
def book(tmp_path):
    """Same shape as the shipped `data/contacts.json`: keyed by the name the units
    carry, dict entries with `email`, and one plain-string entry to prove the loader
    accepts both."""
    p = tmp_path / "contacts.json"
    p.write_text(json.dumps({"Marie": {"email": "marie@example.org"},
                             "Paul": "paul@example.org",
                             "Anaïs": {"email": None}}), encoding="utf-8")
    return ContactBook.load(p)


def test_contact_book_reads_both_shapes_and_finds_nothing_it_does_not_have(book, tmp_path):
    assert book.email_for("MARIE") == "marie@example.org"     # case-insensitive key
    assert book.email_for("PAUL") == "paul@example.org"       # plain string entry
    assert book.email_for("Nobody") is None
    # no fuzzy match: a book keyed "Marie" must not answer for "Marie Dupont",
    # because guessing that a directory entry is the same person is inventing one
    assert book.email_for("Marie Dupont") is None
    assert book.known("paul") is True and book.known("Nobody") is False
    # `email: null` is a known person *without* an address, not an unknown one
    assert book.known("Anaïs") is True and book.email_for("Anaïs") is None
    assert ContactBook.load(None).emails == {}
    assert ContactBook.load(tmp_path / "missing.json").emails == {}


@pytest.fixture
def owner_store(tmp_store):
    marie = make_unit("Marie prend le pilotage du budget d'Atlas.", unit_id="o-marie",
                      source_date="2026-03-01", type="commitment", status="confirmed",
                      slots=Slots(owner="Marie", subject="atlas.owner.budget"))
    paul = make_unit("Paul assure la reprise du budget.", unit_id="o-paul",
                     source_date="2026-01-05", type="decision", status="confirmed",
                     slots=Slots(owner="Paul", subject="atlas.owner.budget"))
    no_owner = make_unit("Le budget est à 2,4 M€.", unit_id="o-value",
                         source_date="2026-03-01", slots=Slots(subject="atlas.budget.value"))
    stray = make_unit("Chez Boréal, Zoé pilote le budget.", unit_id="o-zoe",
                      source_date="2026-04-01", project="boreal", type="commitment",
                      status="confirmed",
                      slots=Slots(owner="Zoé", subject="boreal.owner.budget"))
    assert tmp_store.add_units([marie, paul, no_owner, stray]) == 4
    return tmp_store


def test_candidates_keep_only_owner_units_of_the_requested_project(owner_store):
    got = candidates(owner_store, "atlas", "budget")
    assert {u.unit_id for u in got} == {"o-marie", "o-paul"}, \
        "a value-only unit and another project's owner must not qualify"
    assert candidates(owner_store, "atlas", "budget", roster={"Marie"}) == \
        [u for u in got if u.slots.owner == "Marie"]
    # a person mentioned but never assigned is not a responsible party
    assert all(u.slots.owner for u in got)


def test_one_current_owner_resolves_and_uses_the_directory(owner_store, book):
    r = resolve_recipient(owner_store, project="atlas", attribute="budget", book=book)
    assert r.resolution_status == "resolved"
    assert r.name == "Marie"                      # most recent explicit assignment
    assert r.email == "marie@example.org"
    assert r.evidence_unit_ids == ["o-marie"]
    assert r.justification


def test_two_owners_on_the_same_day_are_ambiguous_not_guessed(owner_store, book):
    """6.12.3: a tie must be clarified, never broken by dict order."""
    twin = make_unit("Paul reprend aussi le pilotage du budget.", unit_id="o-paul2",
                     source_date="2026-03-01", type="commitment", status="confirmed",
                     slots=Slots(owner="Paul", subject="atlas.owner.budget"))
    owner_store.add_units([twin])
    r = resolve_recipient(owner_store, project="atlas", attribute="budget", book=book)
    assert r.resolution_status == "ambiguous"
    assert r.alternatives == ["Marie", "Paul"]
    assert " / " in r.name
    assert r.email is None, "an ambiguous recipient must never get an address"


def test_a_known_person_without_an_address_is_email_missing_never_invented(owner_store,
                                                                          book):
    """The never-fabricate branch (6.12.4) — the one that had no test."""
    ana = make_unit("Anaïs prend le pilotage du budget.", unit_id="o-anais",
                    source_date="2026-05-01", type="commitment", status="confirmed",
                    slots=Slots(owner="Anaïs", subject="atlas.owner.budget"))
    owner_store.add_units([ana])
    r = resolve_recipient(owner_store, project="atlas", attribute="budget", book=book)
    assert r.resolution_status == "email_missing"
    assert r.email is None
    assert r.name == "Anaïs"
    assert "aucune n'est inventée" in r.justification


def test_an_unknown_person_named_in_the_request_is_resolved_without_an_email(owner_store,
                                                                            book):
    r = resolve_recipient(owner_store, project="atlas", attribute="budget", book=book,
                          person="Camille")
    assert r.resolution_status == "email_missing"
    assert r.name == "Camille" and r.email is None
    assert r.evidence_unit_ids == []


def test_no_owner_unit_anywhere_is_not_found(owner_store, book):
    r = resolve_recipient(owner_store, project="atlas", attribute="recrutement",
                          book=book)
    assert r.resolution_status == "not_found"
    assert r.email is None and r.alternatives == []
    assert "aucune unité" in r.justification
