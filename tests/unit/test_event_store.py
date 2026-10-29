"""Event-sourced JSON: N3 key + N4 latest-confirmed rule."""
from __future__ import annotations

import json

from engine.memory.event_builder import build_store
from engine.memory.event_store import EventStore, MentionRecord


def M(uid: str, turn: int, value: str, status: str, point="date"):
    return MentionRecord(uid, "m1", turn, "2026-10-01", f"text {value}",
                         point, value, status, "evidence")


def test_event_current_is_latest_confirmed_and_proposals_stay_open():
    s = EventStore()
    s.add(M("a", 0, "11", "confirmed"))
    s.add(M("b", 1, "12", "proposed"))
    s.add(M("c", 2, "12", "confirmed"))
    e = s.events["m1:date"]
    assert e.current("latest_confirmed").mention_id == "c"
    assert [m.mention_id for m in e.open()] == ["b"]
    assert e.current("latest").mention_id == "c"


def test_latest_mode_would_pick_a_later_proposal_but_confirmed_does_not():
    s = EventStore()
    s.add(M("a", 0, "11", "confirmed"))
    s.add(M("b", 1, "13", "proposed"))
    e = s.events["m1:date"]
    assert e.current("latest").mention_id == "b"
    assert e.current("latest_confirmed").mention_id == "a"


def test_point_query_returns_only_the_event_current():
    s = EventStore()
    s.add(M("a", 0, "11", "confirmed", point="date"))
    s.add(M("b", 0, "2,4 M€", "confirmed", point="budget"))
    got = s.query("où en est la date ?", point="date", project="m1")
    assert len(got) == 1
    assert got[0].current.mention_id == "a"


def test_jsonl_roundtrip_keeps_event_key_and_current():
    s = EventStore()
    s.add(M("a", 0, "11", "confirmed"))
    s.add(M("b", 1, "12", "confirmed"))
    p = "/tmp/event_store_test.jsonl"
    s.save(p)
    t = EventStore.load(p)
    assert t.events["m1:date"].current().mention_id == "b"
    assert json.loads(open(p).readline())["event_key"] == "m1:date"


def test_builder_turns_a_meeting_into_mentions_and_events():
    turns = [
        {"text": "on part sur une billetterie en ligne"},
        {"text": "finalement on ajoute une billetterie sur place"},
    ]
    s = build_store("m1", turns)
    assert any(e.point in ("mode", "topic:billetterie") for e in s.events.values())
    assert all(m.evidence == m.text for m in s.mentions)
