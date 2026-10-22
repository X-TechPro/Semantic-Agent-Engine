"""Hypothetical QUERIES never act — the real-data demo found this gap.

6.2's "hypothesis never triggers an action" guarded the STORE unit only; the
request itself — "si on annulait la buvette, il faudrait créer une tâche" —
reached create_task while the lead row for buvette was confirmed
(`runs/demo_real/...-pre-fix.json`, check 7 FAIL, 2026-10-22). Both sides are
now non-assertive-proof. Offline: stub engines, no store, no GPU — the
assertion direction matters as much as the refusal: an assertive imperative
must still act, or the guard became a blanket veto.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from engine.agent.router import Agent
from engine.schemas import Hit, Slots, Unit

pytestmark = pytest.mark.unit


def _stub(read_status):
    unit = Unit(unit_id="u1", source_id="s1", source_type="note",
                source_date=date(2026, 1, 1),
                ingested_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                text="Le budget est voté à 180 k€", sentences=[],
                project="Atlas", slots=Slots(subject="atlas.budget"))
    read = SimpleNamespace(unit=unit, status=read_status, intent=None)
    eng = SimpleNamespace(
        cfg={"agent": {"contacts_path": "x", "no_answer_threshold": 0.0,
                       "no_answer_overlap": 0.0}},
        store=SimpleNamespace(all_units=lambda: [unit],
                              db=SimpleNamespace(execute=lambda q: [])),
        projects={"Atlas"}, roster=set(),
        read_text=lambda t: read,
        retrieve=lambda q: [],
        intent_of=lambda t: "answer")
    return Agent(eng, book=object()), unit


def test_hypothetical_query_is_refused_even_when_the_lead_is_confirmed():
    agent, _ = _stub("conditional")
    res = agent.handle("Si on annulait le budget, il faudrait créer une tâche ?",
                       intent="create_task", project="Atlas")
    assert res.operation == "clarify"
    assert res.payload["guard"] == "hypothesis_never_acts"
    assert res.payload["refused"] == "conditional"


def test_assertive_imperative_still_reaches_the_operation():
    # the other direction: the guard must not become a blanket veto
    agent, unit = _stub("confirmed")
    res = agent.handle("Crée une tâche pour le budget voté à 180 k€.",
                       intent="create_task", project="Atlas")
    assert res.operation == "create_task", res.payload


def test_query_carries_the_reader_status():
    agent, _ = _stub("hypothesis")
    q = agent._query("Si on faisait ça ?", intent="create_task",
                     project="Atlas", mode="current")
    assert q.status == "hypothesis"


def test_module_level_act_entry_keeps_its_old_shape():
    # module-level act() builds a Query without a status: None must never
    # block (the store-side guard still applies)
    agent, unit = _stub("confirmed")
    from engine.schemas import Query
    res = agent.act("create_task", [Hit(unit=unit, score=0.9)],
                    Query(text="Crée une tâche", project="Atlas",
                          intent="create_task"))
    assert res.operation == "create_task"
