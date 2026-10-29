"""Explicit state table + state-conditioned node reader."""
from __future__ import annotations

from datetime import date, datetime, timezone

from engine.schemas import Slots, Unit
from engine.threads import build_session_nodes
from engine.threads.state import DecisionState, StateUpdate
from engine.threads.state_reader import StateReader


def _u(i: int, text: str) -> Unit:
    return Unit(unit_id=f"u-{i}", source_id="meeting-state",
                source_type="meeting", source_date=date(2026, 10, 27),
                ingested_at=datetime(2026, 10, 27, tzinfo=timezone.utc),
                text=text, sentences=[], speaker=None, project="atlas",
                slots=Slots(), turn_index=i)


def test_proposed_then_confirmed_replaces_and_keeps_open_until_closed():
    s = DecisionState()
    assert s.apply(StateUpdate("food", "buvette", "proposed", 1, "buvette ?"))
    assert s.open["food"].value == "buvette"
    assert s.current("food") is None
    assert s.apply(StateUpdate("food", "deux buvettes", "confirmed", 2,
                               "deux buvettes d'une part et d'autre"))
    assert s.current("food").value == "deux buvettes"
    assert s.current("food").old_value == "buvette"
    assert "food" not in s.open


def test_reader_writes_grounded_food_and_closes_an_open_budget_point():
    s = DecisionState()
    r = StateReader()
    r.apply("le buffet est confirmé", 0, s)
    assert s.current("food").value == "buffet"
    s.apply(StateUpdate("budget", "payer", "proposed", 1, "on le paie ?"))
    updates = r.update("donc on le paie ?", 2, s)
    assert updates and updates[0].point == "budget"
    assert updates[0].status == "confirmed"
    r.apply("donc on le paie ?", 2, s)
    assert s.current("budget").turn_index == 2


def test_state_reader_adds_a_second_node_set_but_default_stays_topic_only():
    units = [_u(0, "on part sur une billetterie en ligne"),
             _u(1, "finalement on ajoute une billetterie sur place")]
    plain = build_session_nodes(units)
    assert len(plain) == 1
    with_state = build_session_nodes(units, state_reader=StateReader())
    assert any(n.node_id.endswith(":state:mode") for n in with_state)
    mode = next(n for n in with_state if n.node_id.endswith(":state:mode"))
    assert mode.current_turn == 1
    assert [c.turn_index for c in mode.children] == [0, 1]


def test_state_node_current_is_the_confirmed_row_not_a_later_proposal():
    from engine.threads.view import NodeView

    units = [_u(0, "on part sur une billetterie en ligne"),
             _u(1, "on pourrait ajouter une billetterie sur place")]
    view = NodeView(units, state=True)
    mode = next(n for n in view.nodes if n.node_id.endswith(":state:mode"))
    assert mode.latest.turn_index == 1
    assert view.at(mode).turn_index == 0
