"""Determinism seeds, score normalisation, and chain expansion (6.10).

Three small pieces of the engine that every headline number depends on and that
had no test of their own:

* `seeding.seed_all` — §16 requires seeds 42/43/44 to be reproducible. It imports
  torch *inside* a try because the venv may not have it (D-002), so the branch that
  actually seeds torch was never executed: without it a run can differ purely
  because of RNG order, which is exactly the flake this project already hit once
  (the toy-encoder test that reseeded the global RNG inside `fit()`).
* `fusion.normalised_scores` — reporting only, but its degenerate cases (no rows,
  all-equal scores) are where a min-max divide-by-zero would appear.
* `graph_expand.expand`/`predecessors` — 6.10's promise that "which decision is in
  force" returns the chain head and exposes what it replaced. This is the
  user-visible half of the relation layer and it was only ever run through the
  full pipeline, so its budget (`max_added`) and its provenance path were untested.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from engine.retrieve.fusion import normalised_scores, rrf
from engine.retrieve.graph_expand import expand, follow_chain, predecessors
from engine.seeding import seed_all, subseed, utcnow_stamp
from engine.schemas import Hit, Slots
from engine.slots.rules import merge
from helpers import make_edge, make_unit

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ determinism

def test_seed_all_makes_the_global_rdrs_reproducible():
    """Same seed -> same draws, for both stdlib random and numpy, and twice."""
    seed_all(42)
    first = (random.random(), float(np.random.random()), random.randint(0, 10**6))
    seed_all(42)
    again = (random.random(), float(np.random.random()), random.randint(0, 10**6))
    assert first == again, "seed_all did not reset both RNGs"

    seed_all(43)
    other = (random.random(), float(np.random.random()))
    assert other != first[:2], "different seeds produced the same stream"


def test_seed_all_sets_pythonhashseed_and_torch_when_available():
    """The `try: import torch` body is the point: a run that seeds numpy but not
    torch can still diverge on any torch-backed stage. The two cudnn flags are
    process-global, so this test puts them back the way it found them."""
    import os

    before = (torch_flags("deterministic"), torch_flags("benchmark"))
    seed_all(44, deterministic_torch=True)
    assert os.environ["PYTHONHASHSEED"] == "44"
    try:
        import torch
    except ImportError:
        pytest.skip("torch absent: the optional branch (D-002) is not applicable")
    assert torch.initial_seed() == 44
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = before


def torch_flags(name: str):
    try:
        import torch
    except ImportError:
        return None
    return getattr(torch.backends.cudnn, name)


def test_subseed_is_stable_ordered_and_distinct():
    assert subseed(42, "dev", "atlas") == subseed(42, "dev", "atlas")
    assert subseed(42, "dev", "atlas") != subseed(42, "atlas", "dev"), \
        "subseed must depend on the order of its labels"
    assert subseed(42, "dev") != subseed(43, "dev")
    assert isinstance(subseed(42, "x"), int) and 0 <= subseed(42, "x") < 2**32


def test_utcnow_stamp_is_second_precision_utc():
    """Generated docs embed this stamp; microseconds would make every regenerated
    MODELCARDS.md diff show clock noise instead of content changes."""
    import re

    s = utcnow_stamp()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", s), s


# ------------------------------------------------------------- score normalisation

def test_normalised_scores_is_min_max_and_never_divides_by_zero():
    got = normalised_scores([("a", 10.0), ("b", 5.0), ("c", 0.0)])
    assert got == {"a": 1.0, "b": 0.5, "c": 0.0}
    # all-equal scores: hi == lo would divide by zero
    assert normalised_scores([("a", 3.0), ("b", 3.0)]) == {"a": 1.0, "b": 1.0}
    assert normalised_scores([]) == {}
    assert normalised_scores([("solo", -2.0)]) == {"solo": 1.0}


def test_rrf_ranks_by_position_and_breaks_ties_deterministically():
    """Hand-computed (6.4 / 12.1). `a` leads both lists; `b` and `c` land on the
    *same* score by symmetry (1/62 + 1/63 either way), so the only thing separating
    them is the tie-break — which must be the unit id, not dict or hash order."""
    fused = rrf({"lexical": ["a", "b", "c"], "vector": ["a", "c", "b"]}, k=60)
    assert [h.unit_id for h in fused] == ["a", "b", "c"]
    assert fused[0].score == pytest.approx(2 / 61)
    assert fused[1].score == pytest.approx(1 / 62 + 1 / 63)
    assert fused[1].score == pytest.approx(fused[2].score), "b/c are symmetric by design"
    assert fused[0].parts["rrf_lexical"] == pytest.approx(1 / 61)

    # a weight scales one ranking's contribution without touching the ranks:
    # `b` is lexical rank 2 and vector rank 1, so it is 1/62 + 2*(1/61)
    weighted = rrf({"lexical": ["a", "b"], "vector": ["b", "a"]}, k=60,
                   weights={"vector": 2.0})
    assert weighted[0].unit_id == "b"
    assert weighted[0].score == pytest.approx(1 / 62 + 2 / 61)


# ------------------------------------------------------------------ 6.10 expand

@pytest.fixture
def chain_store(tmp_store):
    """u-old -> u-mid -> u-head, the three-link chain 6.10 describes."""
    a = make_unit("Le budget est à 1,8 M€.", unit_id="u-old", source_date="2026-01-05")
    b = make_unit("Le budget est à 2,1 M€.", unit_id="u-mid", source_date="2026-03-02",
                  superseded_by="u-head")
    c = make_unit("Le budget est à 2,4 M€.", unit_id="u-head", source_date="2026-06-01")
    a = a.model_copy(update={"superseded_by": "u-mid"})
    assert tmp_store.add_units([a, b, c]) == 3
    tmp_store.add_edges([make_edge("u-mid", "u-old", "supersedes", 0.9),
                         make_edge("u-head", "u-mid", "supersedes", 0.95)])
    return tmp_store


def test_follow_chain_walks_superseded_by_to_the_head(chain_store):
    assert follow_chain(chain_store, "u-old") == ["u-old", "u-mid", "u-head"]
    assert follow_chain(chain_store, "u-head") == ["u-head"], \
        "a chain head must not invent a successor"


def test_expand_adds_the_chain_head_with_a_provenance_path(chain_store):
    """A hit on a superseded unit must surface the unit in force, tagged with the
    path it came from and a `graph` score part, and must not duplicate it."""
    old = chain_store.get("u-old")
    hits = [Hit(unit=old, score=0.4, score_parts={"lexical": 0.4}, evidence=old.text)]
    out = expand(chain_store, hits)
    ids = [h.unit.unit_id for h in out]
    assert ids == ["u-old", "u-head"], ids
    added = out[-1]
    assert added.score_parts["graph"] == 1.0
    assert added.path == ["u-old", "u-mid", "u-head"]
    assert added.evidence == "Le budget est à 2,4 M€."
    # the original hit now carries the path too
    assert out[0].path == ["u-old", "u-mid", "u-head"]


def test_expand_does_not_re_add_a_head_already_present(chain_store):
    old, head = chain_store.get("u-old"), chain_store.get("u-head")
    hits = [Hit(unit=old, score=0.4), Hit(unit=head, score=0.3)]
    out = expand(chain_store, hits)
    assert [h.unit.unit_id for h in out] == ["u-old", "u-head"]


def test_expand_respects_max_added_and_leaves_lonely_hits_untouched(chain_store):
    old = chain_store.get("u-old")
    hits = [Hit(unit=old, score=0.4)]
    assert [h.unit.unit_id for h in expand(chain_store, hits, max_added=0)] == ["u-old"]

    lonely = make_unit("Rien d'autre ici.", unit_id="u-lonely")
    tmp = chain_store  # same store; u-lonely is not in it, so get() finds no chain
    out = expand(tmp, [Hit(unit=lonely, score=0.2)])
    assert [h.unit.unit_id for h in out] == ["u-lonely"]


def test_predecessors_returns_what_a_decision_replaced(chain_store):
    """6.10's history half: asked about the unit in force, show what it superseded."""
    hist = predecessors(chain_store, "u-head")
    assert [u.unit_id for u in hist] == ["u-mid"]
    assert predecessors(chain_store, "u-old") == []


# ------------------------------------------------------------------ slots merge

def test_merge_prefers_non_empty_and_unions_persons_without_duplicates():
    base = Slots(persons=["Marie"], owner=None, subject="atlas.budget.value",
                 amount="1,8 M€", date_value=None, due=None, action=None)
    extra = Slots(persons=["Marie", "Paul"], owner="Paul", subject=None,
                  amount="2,4 M€", date_value="2026-06-01", due=None, action="valider")
    m = merge(base, extra)
    assert m.persons == ["Marie", "Paul"]
    assert m.owner == "Paul"
    assert m.amount == "2,4 M€"
    assert m.date_value == "2026-06-01"
    assert m.action == "valider"
    # `subject` is empty in `extra`, so base's value must survive: an empty field
    # overwriting a filled one is how a slot silently disappears.
    assert m.subject == "atlas.budget.value"


def test_merge_with_an_empty_extra_is_a_no_op():
    base = Slots(persons=["Marie"], subject="atlas.budget.value")
    m = merge(base, Slots())
    assert m.persons == ["Marie"] and m.subject == "atlas.budget.value"
