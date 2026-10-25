"""Decision nodes are a view over immutable messages (mentor's rule)."""
from __future__ import annotations

from datetime import date

import pytest

from engine.schemas import Slots
from engine.threads import build_session_nodes
from helpers import make_unit

pytestmark = pytest.mark.unit


def U(text, *, day, turn, project="pizza", uid=None):
    u = make_unit(text, source_date=day, project=project, unit_id=uid)
    return u.model_copy(update={"turn_index": turn})


def test_messages_group_into_one_node_and_latest_is_current():
    a = U("quelqu'un veut manger des pizzas ce midi", day=date(2026, 10, 1), turn=0)
    b = U("c'est bon, j'ai acheté les pizzas", day=date(2026, 10, 1), turn=1)
    nodes = build_session_nodes([a, b])
    assert len(nodes) == 1
    node = nodes[0]
    assert node.topic == "restauration"
    assert [c.unit_id for c in node.children] == [a.unit_id, b.unit_id]
    assert node.latest.unit_id == b.unit_id


def test_as_of_returns_the_state_then_without_mutating_children():
    a = U("le budget est validé à 1,5 M€", day=date(2026, 10, 1), turn=0)
    b = U("finalement le budget est révisé à 2,4 M€", day=date(2026, 10, 12), turn=1)
    node = build_session_nodes([a, b])[0]
    assert node.as_of(date(2026, 10, 5)).unit_id == a.unit_id
    assert node.as_of(date(2026, 10, 12)).unit_id == b.unit_id
    # the earlier message is untouched: no supersede stamp, same text
    assert a.text.startswith("le budget est validé")
    assert a.valid_to is None and a.superseded_by is None


def test_a_later_message_appends_and_never_rewrites():
    a = U("le budget est validé à 1,5 M€", day=date(2026, 10, 1), turn=0)
    node = build_session_nodes([a])[0]
    b = U("finalement le budget est révisé à 2,4 M€", day=date(2026, 10, 12), turn=1)
    node.add(b)
    assert [c.unit_id for c in node.children] == [a.unit_id, b.unit_id]
    assert node.as_of(date(2026, 10, 1)).text == a.text
    assert node.latest.text == b.text
