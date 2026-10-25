"""The beta_mention validity knob (D-040) — mechanism tests.

The ranking census found mention units (statements that name a subject
without asserting a state) on rank 1 in 8 of 17 headline misses. The knob
down-weights them on CURRENT queries only; these tests pin the boundary —
history/compare queries untouched, soft statuses untouched, the superseded
path untouched, and the shipped default (1.0) a true no-op. Whether 0.5
SHIPS is decided by runs/beta_mention_ab/, not here.
"""
from __future__ import annotations

from datetime import date

import pytest

from engine.config import load_config
from engine.retrieve.validity import apply_validity, validity_factor
from engine.schemas import Query, Slots, Unit, utcnow

pytestmark = pytest.mark.unit


def _unit(uid, status, *, current=True):
    return Unit(unit_id=uid, source_id="s1", source_type="meeting",
                source_date=date(2026, 10, 1), ingested_at=utcnow(),
                text="le budget", sentences=[], project="Atlas", kind="info",
                type="fact", status=status, slots=Slots(),
                **({} if current else
                   {"superseded_by": "other", "valid_to": date(2026, 10, 2)}))


def test_mention_demoted_on_current_only():
    u = _unit("u1", "mention")
    assert validity_factor(u, "current", beta_mention=0.5) == 0.5
    assert validity_factor(u, "history", beta_mention=0.5) == 1.0
    assert validity_factor(u, "compare", beta_mention=0.5) == 1.0


def test_default_is_a_true_noop():
    """Shipped behaviour: with the config default the knob changes nothing."""
    assert load_config().get_path("retrieve.validity.beta_mention") == 1.0
    for status in ("mention", "confirmed", "proposed"):
        u = _unit(f"u-{status}", status)
        assert validity_factor(u, "current",
                               beta_soft=0.5) == validity_factor(
                                   u, "current", beta_soft=0.5,
                                   beta_mention=1.0)


def test_other_statuses_untouched_by_the_knob():
    assert validity_factor(_unit("u2", "confirmed"), "current",
                           beta_mention=0.5) == 1.0
    # soft statuses keep their own penalty, not the mention one
    assert validity_factor(_unit("u3", "proposed"), "current",
                           beta_soft=0.5, beta_mention=0.2) == 0.5


def test_superseded_path_wins_over_the_mention_path():
    u = _unit("u4", "mention", current=False)
    # alpha_superseded=0.0 with full confidence drops it regardless of beta
    assert validity_factor(u, "current", beta_mention=0.5,
                           alpha_superseded=0.0, confidence=1.0) == 0.0


def test_apply_validity_threads_the_knob():
    q = Query(text="quelle décision sur le budget ?", mode="current",
              project="Atlas")
    scored = [(_unit("m", "mention"), 0.02), (_unit("c", "confirmed"), 0.018)]
    off = apply_validity(scored, q, beta_mention=1.0)
    on = apply_validity(scored, q, beta_mention=0.5)
    # apply_validity preserves input order (the pipeline sorts on the final
    # score) — so the test asserts the factors and finals, then the ORDER the
    # pipeline would produce from them
    assert [u.unit_id for u, *_ in on] == ["m", "c"]
    assert on[0][2] == 0.5 and on[0][3] == pytest.approx(0.01)
    assert on[1][2] == 1.0 and on[1][3] == pytest.approx(0.018)
    assert off[0][2] == 1.0
    ranked = sorted(on, key=lambda t: (-t[3], t[0].unit_id))
    assert [u.unit_id for u, *_ in ranked] == ["c", "m"]   # demoted past it
