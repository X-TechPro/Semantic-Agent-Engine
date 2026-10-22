"""agent.llm_answer — the diagram's last box, verify-gated, default OFF.

Three outcomes and no fourth: composed-and-verified (rewrite kept), rejected
by the verifier (invented name/number — template stands), unavailable/empty
(template stands). The note travels with the result so an artefact can never
claim an LLM wrote an answer the verifier bounced. Offline: stub backend
patched over build_llm; ContactBook never loaded (Agent.__new__).
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest

from engine.agent.answer import build_answer, compose_answer
from engine.agent.llm import LLMUnavailable
from engine.agent.router import Agent
from engine.config import load_config
from engine.schemas import Hit, Query, Slots, Unit, utcnow

pytestmark = pytest.mark.unit

FACT = "Le budget est voté à 180 k€"


class Stub:
    def __init__(self, out=None, fail=False):
        self._out, self._fail = out, fail

    def generate(self, prompt, schema=None, system=None):
        if self._fail:
            raise LLMUnavailable("ollama down")
        return self._out


def _payload_hits_query():
    unit = Unit(unit_id="u-x1", source_id="s1", source_type="meeting",
                source_date=date(2026, 1, 5), ingested_at=utcnow(),
                text=FACT, sentences=[], project="Atlas", kind="info",
                type="decision", status="confirmed",
                slots=Slots(subject="atlas.budget"))
    hits = [Hit(unit=unit, score=0.9)]
    q = Query(text="Quel est le budget voté ?", project="Atlas", intent="answer")
    payload = build_answer(hits, q, min_score=0.0, overlap_floor=0.0)
    return payload, hits, q


def _agent(monkeypatch, stub):
    monkeypatch.setattr("engine.agent.llm.build_llm", lambda cfg: stub)
    a = Agent.__new__(Agent)                     # no ContactBook load
    a.engine = SimpleNamespace(cfg=load_config(overrides={
        "agent": {"llm_answer": True, "llm": {"backend": "ollama",
                                              "model_id": "x"}}}))
    return a


def test_default_off_shipped_surface_untouched():
    assert load_config()["agent"]["llm_answer"] is False


def test_verified_compose_rejects_invention_and_tolerates_absence(monkeypatch):
    payload, hits, q = _payload_hits_query()
    good = _agent(monkeypatch, Stub(out={"answer": f"{FACT}, retenu à 180 k€ [u-x1]."}))
    p2, note = good._compose(payload, hits, q)
    assert note == "llm:composed+verified"
    assert p2.answer != payload.answer            # rewrite applied
    assert [c.unit_ids for c in p2.claims] == [c.unit_ids for c in payload.claims]

    bad = _agent(monkeypatch, Stub(out={"answer": "Zorglub a voté 999 k€ [u-x1]."}))
    p3, note3 = bad._compose(payload, hits, q)
    assert p3.answer == payload.answer            # template stands
    assert "rejected" in note3 and "Zorglub" in note3

    down = _agent(monkeypatch, Stub(fail=True))
    p4, note4 = down._compose(payload, hits, q)
    assert p4.answer == payload.answer and "unavailable" in note4

    empty = _agent(monkeypatch, Stub(out={}))
    p5, note5 = empty._compose(payload, hits, q)
    assert "nothing composed" in note5


def test_compose_answer_refuses_without_claims_or_answer():
    payload, _hits, q = _payload_hits_query()
    abstain = payload.model_copy(update={"answer": None, "claims": []})
    assert compose_answer(Stub(out={"answer": "x"}), q, abstain) is None
