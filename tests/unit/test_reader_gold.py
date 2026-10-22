"""The reader's gold labels have exactly ONE source (BUILD_GUIDE 6.2, 9.2).

`gen.ledger.READER_GOLD` is that source. `gen.render.reader_status` /
`reader_type`, `ledger._STATUS_OF` and the training set in `engine.read.dataset`
all read through it. Before this test existed they were three literal tables that
had ALREADY drifted: the `conditional` refinement lived only in
`render.reader_status`, so conditional events were trained as `proposed`, and the
rules baseline's status macro-F1 moved 0.6883 -> 0.8745 on identical rows once the
labels were fixed. A measurement whose value depends on which copy of a table you
happened to read is not a measurement.
"""

from __future__ import annotations

from datetime import date
from typing import get_args

import pytest

from engine.schemas import KINDS, STATUSES, TYPES
from gen.ledger import (Event, EventKind, READER_GOLD, _STATUS_OF, build_ledger,
                        reader_gold)
from gen.render import _STATUS_MAP, _TYPE_MAP, reader_status, reader_type

pytestmark = pytest.mark.unit


def ev(kind: EventKind, notes: str = "") -> Event:
    return Event("e-1", "atlas", "meeting-atlas-1", date(2026, 1, 1), kind,
                 notes=notes)


class TestOneTable:
    def test_every_event_kind_has_gold(self):
        """A new EventKind must be labelled deliberately, never defaulted to."""
        assert set(READER_GOLD) == set(get_args(EventKind)), (
            "READER_GOLD and EventKind disagree; add the missing kind to "
            "gen/ledger.py instead of letting training fall back to a guess")

    def test_gold_labels_are_valid_schema_values(self):
        """A typo here would poison every training row that uses it."""
        for kind, (k, t, s) in READER_GOLD.items():
            assert k in KINDS, kind
            assert t in TYPES, kind
            assert s in STATUSES, kind

    def test_derived_views_match_the_table(self):
        assert _STATUS_OF == {k: v[2] for k, v in READER_GOLD.items()}
        assert _STATUS_MAP == {k: v[2] for k, v in READER_GOLD.items()}
        assert _TYPE_MAP == {k: v[1] for k, v in READER_GOLD.items()}

    @pytest.mark.parametrize("kind", sorted(READER_GOLD))
    def test_render_delegates_to_ledger(self, kind):
        """The single-valued claim: render and ledger cannot disagree."""
        e = ev(kind)
        _k, t, s = reader_gold(e)
        assert reader_type(e) == t == reader_gold(kind)[1]
        assert reader_status(e) == s

    def test_meeting_question_is_info_not_query(self):
        """7.3: a question asked inside a meeting is not a user query.

        This is what made `answers` unreachable: `kind=query` units are never
        stored (2.1), so there was never a question neighbour to link to.
        """
        assert reader_gold("question")[0] == "info"
        assert reader_gold("question")[1] == "question"
        assert reader_gold(ev("question"))[0] == "info"
        assert reader_gold("answer")[0] == "info"


class TestConditionalRefinement:
    def test_conditional_note_beats_the_kind(self):
        """`proposed` + notes="conditional" is `conditional`, not `proposed`.

        This is the case the duplicated tables lost. It is load-bearing:
        `conditional` is in NON_ACTIONABLE_STATUSES, so mislabelling it as
        `proposed` happens to stay safe for the action gate but erases the class
        the status head is supposed to learn.
        """
        assert reader_gold(ev("proposed", notes="conditional"))[2] == "conditional"
        assert reader_status(ev("proposed", notes="conditional")) == "conditional"

    def test_plain_proposal_is_unaffected(self):
        assert reader_gold(ev("proposed"))[2] == "proposed"

    def test_kind_string_cannot_see_the_note(self):
        """Documented limitation, asserted so it is never silently relied on.

        A caller holding only the kind gets the base mapping — which is why
        `engine.read.dataset.unit_rows` passes the Event, not the kind string.
        """
        assert reader_gold("proposed")[2] == "proposed"
        assert reader_gold(ev("proposed", notes="conditional"))[2] == "conditional"


class TestGoldReachesTraining:
    def test_ledger_events_all_resolvable(self):
        """Every generated event kind is labelled, or training KeyErrors."""
        ledger = build_ledger(n_projects=2, seed=42)
        assert {e.kind for e in ledger.events} <= set(READER_GOLD)

    def test_dataset_gold_uses_the_event_not_the_kind(self):
        """The regression that labelled conditional rows as proposed."""
        from engine.read.dataset import event_of, unit_rows
        from engine.read.model import HEADS, MASK
        from engine.schemas import Slots, Unit, utcnow

        ledger = build_ledger(n_projects=6, seed=42)
        cond = [e for e in ledger.events
                if e.kind == "proposed" and e.notes == "conditional"]
        if not cond:
            pytest.skip("this seed generated no conditional event")
        target = cond[0]
        units = [Unit(unit_id="u-cond", source_id=target.meeting_id,
                      source_type="meeting", source_date=target.on,
                      ingested_at=utcnow(),
                      text="Sous réserve d'audit, 1,5 M€.", speaker="Marie",
                      project=target.project, turn_index=0, slots=Slots(),
                      gold_event_ids=[target.event_id])]
        rows = unit_rows(ledger, units)
        assert rows, "a unit with a gold pointer must yield a row"
        assert rows[0]["labels"]["status"] != MASK
        assert HEADS["status"][rows[0]["labels"]["status"]] == "conditional"
        assert event_of(ledger)[target.event_id] is target
