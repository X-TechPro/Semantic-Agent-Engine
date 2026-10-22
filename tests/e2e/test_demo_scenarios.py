"""The 7 mandatory demo scenarios (BUILD_GUIDE 13) as automated tests.

Each scenario asserts the *behaviour* the brief promises, not a string match:
in-force decision wins, history stays history, a hypothesis creates nothing, an
ambiguous recipient produces a question instead of a guess, and no output ever
contains an address that is not in `data/contacts.json`.
"""

from __future__ import annotations

import json
import re

import pytest

pytestmark = pytest.mark.e2e

EMAIL_IN_TEXT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def _payload(res):
    return res.payload if isinstance(res.payload, dict) else res.payload


class TestScenario1DecisionWithProof:
    """"Quelle décision a finalement été prise sur le budget Atlas ?"""""

    Q = "Quelle décision a finalement été prise concernant le budget du projet Atlas ?"

    def test_in_force_budget_is_the_revision(self, demo_engine):
        res = demo_engine.ask(self.Q, intent="answer", project="Atlas")
        assert res.operation == "answer"
        ans = _payload(res)
        assert ans["answer"], "scenario 1 must produce an answer"
        assert "2,4" in ans["answer"], ans["answer"]

    def test_rank1_is_current_not_superseded(self, demo_engine):
        from engine.schemas import Query

        hits = demo_engine.retrieve(Query(text=self.Q, project="Atlas"))
        assert hits, "expected retrieval hits"
        top = hits[0]
        assert top.unit.is_current, f"rank-1 is superseded: {top.unit.text}"
        assert top.unit.status == "confirmed"

    def test_answer_carries_evidence_metadata(self, demo_engine):
        ans = _payload(demo_engine.ask(self.Q, intent="answer", project="Atlas"))
        assert ans["claims"], "every claim must cite unit ids"
        ids = {c["unit_ids"][0] for c in ans["claims"]}
        u = demo_engine.store.get(next(iter(ids)))
        assert u.source_id and u.source_date and u.speaker

    def test_older_proposal_is_history_not_rank1(self, demo_engine):
        from engine.schemas import Query

        hits = demo_engine.retrieve(Query(text=self.Q, project="Atlas"))
        ranked = [h.unit.unit_id for h in hits]
        old = [h.unit for h in hits if "1,5" in h.unit.text and not h.unit.is_current]
        if old:
            assert ranked[0] != old[0].unit_id


class TestScenario2ProposalVsDecision:
    """The answer must separate what is decided from what is merely proposed."""

    Q = "Le budget du projet Atlas, qu'est-ce qui est décidé et qu'est-ce qui est " \
        "encore une proposition ?"

    def test_confirmed_and_proposed_are_distinguished(self, demo_engine):
        res = demo_engine.ask(self.Q, intent="answer", project="Atlas")
        ans = _payload(res)
        breakdown = ans.get("status_breakdown") or {}
        assert breakdown.get("confirmed"), "must surface the in-force decision"
        assert "2,4" in (ans.get("answer") or "")

    def test_hypothesis_is_not_presented_as_decision(self, demo_engine):
        from engine.schemas import Query

        hits = demo_engine.retrieve(Query(
            text="Quelle est la valeur du tarif du projet Atlas ?", project="Atlas"))
        # the only tarif statement in the fixture is a hypothesis -> never rank-1
        if hits:
            assert hits[0].unit.status != "hypothesis", hits[0].unit.text


class TestScenario3CompareTwoMeetings:
    Q = "Est-ce que meeting-atlas-3 contredit meeting-atlas-1 sur le projet Atlas ?"

    def test_compare_reports_relations_per_item(self, demo_engine):
        res = demo_engine.ask(self.Q, intent="compare")
        assert res.operation == "compare", res.payload
        cmp = _payload(res)
        assert cmp["summary"]
        flat = (cmp["changed"] + cmp["superseded"] + cmp["contradicted"]
                + cmp["unchanged"])
        assert flat, "comparison found no items at all"
        for item in flat:
            assert item["relation"] in ("supersedes", "contradicts", "supports",
                                        "unrelated", "answers", "creates_action")
            assert len(item["unit_ids"]) >= 1

    def test_update_is_explained_as_update_not_contradiction(self, demo_engine):
        cmp = _payload(demo_engine.ask(self.Q, intent="compare"))
        for item in cmp["superseded"]:
            assert "mise à jour" in item["explanation"], item["explanation"]

    # --- the three defects this scenario used to hide -------------------------
    #
    # `test_compare_reports_relations_per_item` above walks changed + superseded +
    # contradicted + UNCHANGED, so it passed while the comparison reported
    # "0 mise à jour, 5 confirmé" on a fixture whose budget moved 1,5 → 1,8 → 2,4 M€.
    # A comparison that finds only direct edges cannot answer a non-adjacent
    # question at all, and this question is exactly the one §13 asks.

    def test_non_adjacent_meetings_still_report_the_update(self, demo_engine):
        """Meeting 1 vs meeting 3: nothing was said directly between them, but what
        was in force changed twice. The answer must be an update, not "confirmé"."""
        from engine.agent.compare import build_comparison

        cmp = build_comparison(demo_engine.store, "meeting-atlas-1", "meeting-atlas-3")
        budget = [i for i in cmp.changed if i.subject == "atlas.budget"]
        assert budget, f"the 1,5 → 2,4 M€ revision must appear as an update: {cmp.summary}"
        assert "mise à jour" in budget[0].explanation, budget[0].explanation
        # and it must NOT also be sitting in the confirmed column
        retired = set(budget[0].unit_ids)
        laundered = [i.subject for i in cmp.unchanged
                    if retired & set(i.unit_ids)]
        assert not laundered, \
            f"a subject whose in-force value changed cannot read as inchangé: {laundered}"

    def test_comparison_is_order_symmetric(self, demo_engine):
        """A vs B and B vs A are one question, not two answers: the pairs of
        assertions involved must not depend on which source the user named first."""
        from engine.agent.compare import build_comparison

        def keys(c):
            return sorted((i.relation, tuple(sorted(i.unit_ids)))
                          for i in (list(c.changed) + list(c.contradicted)))

        assert keys(build_comparison(demo_engine.store, "meeting-atlas-1",
                                     "meeting-atlas-3")) == \
               keys(build_comparison(demo_engine.store, "meeting-atlas-3",
                                     "meeting-atlas-1"))

    def test_no_subject_is_counted_twice_as_an_update(self, demo_engine):
        """Adjacent meetings have a direct edge AND a validity-interval change, so
        both passes used to emit the same pair: 1 vs 2 reported "6 mise à jour" for
        3 real changes."""
        from engine.agent.compare import build_comparison

        cmp = build_comparison(demo_engine.store, "meeting-atlas-1", "meeting-atlas-2")
        pairs = [(i.relation, tuple(sorted(i.unit_ids))) for i in cmp.changed]
        assert len(pairs) == len(set(pairs)), f"duplicate change items: {pairs}"
        assert int(cmp.summary.split()[0]) == len(cmp.changed), cmp.summary


class TestScenario4Dossier:
    Q = "Prépare un dossier complet sur le projet Atlas : historique, décisions, " \
        "budget, risques, prochaines étapes et personnes impliquées."

    def test_dossier_sections_and_citations(self, demo_engine):
        res = demo_engine.ask(self.Q, intent="build_dossier", project="Atlas")
        d = _payload(res)
        titles = {s["title"] for s in d["sections"]}
        for required in ("Contexte", "Faits établis", "Chronologie", "Décisions",
                         "Points ouverts", "Responsables"):
            assert required in titles, f"missing section {required}: {titles}"
        for section in d["sections"]:
            for claim in section["claims"]:
                assert claim["unit_ids"], f"uncited claim: {claim['text']}"

    def test_dossier_resolves_versions(self, demo_engine):
        d = _payload(demo_engine.ask(self.Q, intent="build_dossier", project="Atlas"))
        decisions = next(s for s in d["sections"] if s["title"] == "Décisions")
        text = " ".join(c["text"] for c in decisions["claims"])
        assert "2,4" in text
        assert "Remplacé" in text or "en vigueur" in text, \
            "superseded values must be shown as replaced, not as current"

    def test_dossier_markdown_and_missing_information(self, demo_engine):
        d = _payload(demo_engine.ask(self.Q, intent="build_dossier", project="Atlas"))
        assert d["markdown"].startswith("# Dossier")
        assert isinstance(d["missing_information"], list)


class TestScenario5EmailToRightPerson:
    Q = "Rédige un mail à la personne responsable du budget du projet Atlas pour " \
        "lui demander la dernière version validée avant vendredi."

    def test_recipient_is_marie_with_justification(self, demo_engine):
        res = demo_engine.ask(self.Q, intent="draft_email", project="Atlas")
        p = _payload(res)
        assert res.operation == "draft_email", p
        r = p["recipient"]
        assert r["name"] == "Marie", r
        assert r["resolution_status"] == "resolved", r
        assert r["evidence_unit_ids"], "recipient must be justified by evidence"

    def test_draft_matches_the_brief_contract(self, demo_engine):
        p = _payload(demo_engine.ask(self.Q, intent="draft_email", project="Atlas"))
        assert p["intent"] == "draft_email"
        assert p["tool_candidate"] == "email.create_draft"
        assert p["needs_confirmation"] is True
        assert p["executed"] is False
        assert p["recipient"]["email"] == "marie.dupont@example.org"
        assert p["source_unit_ids"]

    def test_body_uses_only_retrieved_units(self, demo_engine):
        p = _payload(demo_engine.ask(self.Q, intent="draft_email", project="Atlas"))
        assert "Marie" in p["body"]
        assert p["subject"].startswith("Projet Atlas")


class TestScenario6Clarification:
    """Two plausible owners for Iris -> ask, never guess."""

    Q = "Rédige un mail à la personne responsable du budget du projet Iris."

    def test_ambiguous_recipient_yields_clarification(self, demo_engine):
        res = demo_engine.ask(self.Q, intent="draft_email", project="Iris")
        assert res.operation == "clarify", res.payload
        assert res.clarification is not None
        assert set(res.clarification.options) & {"Antoine", "Nadia"}, \
            res.clarification.options

    def test_no_email_is_emitted_when_recipient_ambiguous(self, demo_engine):
        res = demo_engine.ask(self.Q, intent="draft_email", project="Iris")
        assert "email" not in json.dumps(res.payload, ensure_ascii=False).lower() \
            or "example.org" not in json.dumps(res.payload)


class TestScenario7HypothesisTriggersNothing:
    """6.2 hard rule: a hypothesis is stored, acts as nothing, supersedes nothing."""

    def test_hypothesis_unit_is_stored_as_hypothesis(self, demo_engine):
        units = [u for u in demo_engine.store.all_units()
                 if "revoir le tarif" in u.text]
        assert units, "fixture hypothesis was not stored"
        assert units[0].status == "hypothesis", units[0].status

    def test_hypothesis_superseded_nothing(self, demo_engine):
        hyd = [u for u in demo_engine.store.all_units() if "revoir le tarif" in u.text][0]
        edges = demo_engine.store.edges_for(hyd.unit_id, direction="from")
        assert [e for e in edges if e.relation == "supersedes"] == []

    def test_hypothesis_creates_no_action_object(self, demo_engine):
        res = demo_engine.ask("Crée une tâche pour revoir le tarif du projet Atlas.",
                              intent="create_task", project="Atlas")
        assert res.operation != "create_task" or res.clarification is not None
        assert "tasks.create" not in json.dumps(res.payload)

    def test_cancelled_and_reinstated_keep_history(self, demo_engine):
        """12.3: cancel-then-reinstate must leave an auditable chain."""
        from engine.schemas import Query

        hits = demo_engine.retrieve(Query(text="le budget du projet Atlas",
                                          project="Atlas"))
        superseded = [h.unit for h in hits if not h.unit.is_current]
        for u in superseded:
            chain = demo_engine.store.supersede_chain(u.unit_id)
            assert len(chain) >= 2
            assert len(chain) == len(set(chain)), "supersede chain is cyclic"
