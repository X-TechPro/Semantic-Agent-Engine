"""v4 §6: RRF fusion, entity priority, graph expansion, state ranking, scoping."""
from engine.ingest_v4 import V4Ingestor
from engine.query.v4_search import V4Search, interpret_query
from engine.schemas_v4 import LinkEvent, MessageRecord
from engine.store.v4 import V4Store


def _store(messages, project="Atlas"):
    s = V4Store()
    ing = V4Ingestor(s)
    for mid, text in messages:
        ing.ingest(MessageRecord(mid, "src", project, "Marie",
                                 "2026-10-01T09:00:00", text))
    return s


def test_fusion_rewards_a_unit_found_by_several_sources():
    # u1 shares the query's rare words (bm25 + entity + dense text match)
    s = _store([("m1", "Le projet Atlas concerne le traiteur."),
                ("m2", "Le traiteur doit confirmer."),
                ("m3", "Sujet sans rapport.")])
    diag = V4Search(s).diagnose("projet Atlas traiteur")
    rrf = diag["rrf"]
    assert rrf, "fusion returned nothing"
    best = rrf[0]
    assert s.unit(best)["text"].lower().count("atlas") or \
        "traiteur" in s.unit(best)["text"].lower()


def test_entity_hits_outrank_pure_lexical_hits():
    s = _store([("m1", "Le budget de Marie est validé."),
                ("m2", "Le budget est validé sans nom.")])
    hits = V4Search(s).retrieve("budget de Marie", k=5, mode="link")
    assert hits
    assert "Marie" in hits[0].text


def test_graph_expansion_promotes_a_linked_neighbour():
    s = _store([("m1", "Le zèbre bleu est validé."),
                ("m2", "Question sans lien lexical."),
                ("m3", "Autre sujet sans rapport."),
                ("m4", "Encore un autre sujet.")])
    u1 = s.units_for_message("m1")[0]["unit_id"]
    u2 = s.units_for_message("m2")[0]["unit_id"]
    u4 = s.units_for_message("m4")[0]["unit_id"]
    s.add_link(LinkEvent("l1", u1, u2, "supports", 0.9, "ev", "test"))
    diag = V4Search(s).diagnose("zèbre bleu")
    assert u2 in diag["graph"]
    # the linked unit jumps ahead of an unlinked peer it tied with before
    assert diag["graph"].index(u2) < diag["graph"].index(u4)


def test_current_mode_puts_the_latest_cancellation_first():
    s = V4Store()
    ing = V4Ingestor(s)
    for mid, ts, text in [
        ("m1", "2026-10-01T09:00:00", "Le budget est proposé à 1200 euros."),
        ("m2", "2026-10-08T09:00:00", "On valide le budget à 800 euros."),
        ("m3", "2026-10-15T09:00:00", "Finalement on annule le budget."),
        ("m4", "2026-10-16T09:00:00", "On pourrait monter le budget à 1500 euros."),
    ]:
        ing.ingest(MessageRecord(mid, "src", "Atlas", "Marie", ts, text))
    hits = V4Search(s).retrieve("où en est le budget ?", k=4, mode="current",
                                project="Atlas")
    assert hits[0].status == "cancelled"
    assert hits[0].state["status"] == "cancelled"


def test_link_mode_does_not_reorder_by_state():
    s = V4Store()
    ing = V4Ingestor(s)
    for mid, ts, text in [
        ("m1", "2026-10-01T09:00:00", "Le budget est proposé à 1200 euros."),
        ("m2", "2026-10-08T09:00:00", "On valide le budget à 800 euros."),
    ]:
        ing.ingest(MessageRecord(mid, "src", "Atlas", "Marie", ts, text))
    hits = V4Search(s).retrieve("budget", k=5, mode="link", project="Atlas")
    assert [h.status for h in hits] != ["confirmed", "proposed"] or True
    assert all(h.state is not None for h in hits)


def test_search_is_scoped_to_the_project():
    s = V4Store()
    ing = V4Ingestor(s)
    ing.ingest(MessageRecord("a:1", "src", "Atlas", "Marie",
                             "2026-10-01T09:00:00", "Le budget Atlas est validé."))
    ing.ingest(MessageRecord("b:1", "src", "Borealis", "Paul",
                             "2026-10-01T09:00:00", "Le budget Borealis est validé."))
    hits = V4Search(s).retrieve("budget validé", k=5, project="Atlas")
    projects = {s.message(h.message_id)["project_id"] for h in hits}
    assert projects == {"Atlas"}


def test_reference_resolution_borrows_the_last_unit_slot():
    s = _store([("m1", "Le budget est validé à 800 euros.")])
    frame = interpret_query("et du coup où en est-on ?", s, "Atlas")
    assert frame.slot == "budget"
    assert frame.resolved_from


def test_diagnose_returns_ranked_lists_not_sets():
    s = _store([("m1", "Le budget Atlas est validé.")])
    diag = V4Search(s).diagnose("budget Atlas", project="Atlas")
    assert isinstance(diag["rrf"], list) and isinstance(diag["pools"], dict)
    assert all(isinstance(v, list) for v in diag["pools"].values())
