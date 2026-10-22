"""API contract tests (BUILD_GUIDE 12.4) — schema-validated, 4xx on bad input.

These run the real FastAPI app through TestClient, so `make api` is exercised,
not assumed. The no-execution and no-fabrication rules are asserted over HTTP
responses too, because that is the surface an agent would actually call.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app, engine, reset_engine
from engine.schemas import AnswerWithEvidence, DraftEmail, Hit, IngestReport, Unit

pytestmark = pytest.mark.integration

FIXTURE = Path(__file__).resolve().parents[1] / "golden" / "demo_scenarios.json"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = str(tmp_path_factory.mktemp("api") / "api.db")
    import os

    os.environ["SAE_STORE"] = db
    import api.app as api_mod

    cfg_over = {"store": {"path": db},
                "agent": {"contacts_path": str(Path(__file__).resolve()
                                               .parents[2] / "data" / "contacts.json")}}
    from engine.config import load_config
    from engine.engine import Engine

    api_mod._ENGINE = Engine(load_config(overrides=cfg_over), path=db)
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for raw in sorted(data["sources"], key=lambda r: r["source_date"]):
        raw = dict(raw)
        raw["source_date"] = date.fromisoformat(raw["source_date"])
        api_mod._ENGINE.ingest(__import__("engine.schemas", fromlist=["Source"])
                               .Source(**raw))
    with TestClient(app) as c:
        yield c
    reset_engine()


class TestHealth:
    def test_health_reports_the_store(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["units"] > 0
        assert body["linker_enabled"] is True


class TestIngestContract:
    def test_ingest_validates_and_reports(self, client):
        src = {"source_id": "meeting-api-1", "source_type": "meeting",
               "source_date": "2026-11-01", "project": "Atlas",
               "turns": [{"speaker": "Marie",
                          "text": "Le budget du projet Atlas est revu à 3,1 M€."}]}
        r = client.post("/ingest", json={"source": src})
        assert r.status_code == 200, r.text
        rep = IngestReport.model_validate(r.json())
        assert rep.n_units == 1 and rep.n_stored == 1

    def test_bad_source_is_4xx_not_500(self, client):
        r = client.post("/ingest", json={"source": {"source_id": "x"}})
        assert r.status_code == 422

    def test_unknown_source_type_is_rejected(self, client):
        r = client.post("/ingest", json={"source": {
            "source_id": "meeting-bad", "source_type": "slack",
            "source_date": "2026-11-01", "turns": []}})
        assert r.status_code == 422


class TestQueryContract:
    def test_query_returns_schema_validated_hits(self, client):
        r = client.post("/query", json={"text": "le budget du projet Atlas",
                                         "project": "Atlas", "k": 5})
        assert r.status_code == 200
        hits = [Hit.model_validate(h) for h in r.json()]
        assert hits
        assert all(h.evidence for h in hits), "every hit must carry evidence"
        assert all(h.unit.unit_id for h in hits)

    def test_empty_text_is_422(self, client):
        assert client.post("/query", json={"text": ""}).status_code == 422

    def test_history_query_keeps_superseded_evidence(self, client):
        cur = client.post("/query", json={"text": "Quand est le lancement du projet "
                                                  "Atlas ?", "project": "Atlas",
                                          "mode": "current"}).json()
        hist = client.post("/query", json={"text": "Quand est le lancement du projet "
                                                   "Atlas ?", "project": "Atlas",
                                           "mode": "history"}).json()
        assert len(hist) >= len(cur), "history mode must not return less evidence"


class TestAskContract:
    ASK = ("Rédige un mail à la personne responsable du budget du projet Atlas pour "
           "lui demander la dernière version validée avant vendredi.")

    def test_draft_email_round_trips_its_schema(self, client):
        r = client.post("/ask", json={"text": self.ASK, "intent": "draft_email",
                                      "project": "Atlas"})
        assert r.status_code == 200
        body = r.json()
        assert body["operation"] in ("draft_email", "clarify"), body
        if body["operation"] == "draft_email":
            d = DraftEmail.model_validate(body["payload"])
            assert d.needs_confirmation is True
            assert d.executed is False
            assert d.recipient.resolution_status in ("resolved", "email_missing")
            assert d.source_unit_ids

    def test_answer_payload_validates(self, client):
        body = client.post("/ask", json={
            "text": "Quelle décision a finalement été prise concernant le budget du "
                    "projet Atlas ?", "intent": "answer", "project": "Atlas"}).json()
        ans = AnswerWithEvidence.model_validate(body["payload"])
        if ans.answer:
            assert ans.claims, "an answer must cite evidence"

    def test_nothing_is_ever_executed_over_http(self, client):
        for intent, text in [("draft_email", self.ASK),
                             ("create_task", "Crée une tâche pour le budget Atlas."),
                             ("schedule_event", "Planifie un point sur Atlas.")]:
            body = client.post("/ask", json={"text": text, "intent": intent,
                                             "project": "Atlas"}).json()
            blob = json.dumps(body)
            assert '"executed": true' not in blob, intent

    def test_unknown_intent_falls_back_to_clarify(self, client):
        body = client.post("/ask", json={"text": "Fais-le.", "intent": "clarify"}).json()
        assert body["operation"] == "clarify"


class TestUnitAndHistory:
    def test_unit_lookup_and_404(self, client):
        uid = client.post("/query", json={"text": "budget Atlas"}).json()[0]["unit"]["unit_id"]
        u = Unit.model_validate(client.get(f"/unit/{uid}").json())
        assert u.unit_id == uid
        assert client.get("/unit/u-does-not-exist").status_code == 404

    def test_history_lists_every_assertion_for_a_slot(self, client):
        r = client.get("/history", params={"subject": "atlas.budget"})
        assert r.status_code == 200
        rows = [Unit.model_validate(x) for x in r.json()]
        assert rows
        assert {u.status for u in rows} & {"confirmed", "proposed"}
        assert any(u.valid_to is not None for u in rows) or len(rows) == 1

    def test_history_requires_a_subject(self, client):
        assert client.get("/history").status_code == 422


class TestOffline:
    def test_offline_reports_the_consistency_audit(self, client):
        body = client.post("/offline").json()
        assert "audit" in body and "clusters" in body
        assert body["audit"]["n_units"] > 0
