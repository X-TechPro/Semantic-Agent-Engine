"""v4 §7: citation of the evidence that carries the state, transition, abstention."""
from engine.answer.v4 import answer, verify
from engine.ingest_v4 import V4Ingestor
from engine.query.v4_search import V4Search
from engine.schemas_v4 import MessageRecord
from engine.store.v4 import V4Store, event_id_for


def _scenario():
    s = V4Store()
    ing = V4Ingestor(s)
    for mid, ts, text in [
        ("m1", "2026-10-01T09:00:00", "Le budget est proposé à 1200 euros."),
        ("m2", "2026-10-08T09:00:00", "On valide le budget à 800 euros."),
        ("m3", "2026-10-15T09:00:00", "Finalement on annule le budget."),
    ]:
        ing.ingest(MessageRecord(mid, "src", "Atlas", "Marie", ts, text))
    return s


def test_abstains_without_hits_or_state():
    a = answer("budget", [], None)
    assert a.unknown and "assez" in a.text and a.citations == []
    a = answer("budget", [{"unit_id": "u1", "message_id": "m1", "text": "x",
                           "evidence": None, "state": None}], store=None)
    assert a.unknown


def test_cites_the_state_evidence_not_the_rank_one_hit():
    s = _scenario()
    hits = V4Search(s).retrieve("où en est le budget ?", k=5, mode="current",
                                project="Atlas")
    a = answer("le budget", hits, store=s)
    assert not a.unknown and a.verified
    cited_units = {c.unit_id for c in a.citations}
    # the cancellation owns the current state, the earlier confirmation is its
    # transition source; both are cited even though rank 1 may neither.
    key = event_id_for(s.units_for_message("m3")[0], s)
    state = s.current_state(key, "budget")
    assert state["evidence_unit"] in cited_units
    assert state["transition"]["from_evidence_unit"] in cited_units
    assert any(c.chunk_id for c in a.citations)


def test_transition_and_value_are_reported():
    s = _scenario()
    hits = V4Search(s).retrieve("le budget", k=5, mode="current", project="Atlas")
    a = answer("le budget", hits, store=s)
    assert a.state_transition["from"] == "800 euros"
    assert "800 euros" in a.text


def test_verifier_rejects_an_invented_number():
    assert verify("le budget est 800 euros", [{"text": "budget 800 euros"}])
    assert not verify("le budget est 900 euros", [{"text": "budget 800 euros"}])


def test_invented_state_never_reaches_the_answer():
    s = _scenario()
    hits = V4Search(s).retrieve("le budget", k=5, mode="current", project="Atlas")
    bogus = dict(hits[0].state)
    bogus["value"] = "9999 euros"
    a = answer("le budget", hits, state=bogus, store=s)
    assert a.unknown and not a.verified


def test_open_proposal_is_answered_as_a_proposal():
    s = V4Store()
    V4Ingestor(s).ingest(MessageRecord(
        "m1", "src", "Atlas", "Marie", "2026-10-01T09:00:00",
        "Le budget est proposé à 1200 euros."))
    hits = V4Search(s).retrieve("le budget", k=3, mode="current", project="Atlas")
    a = answer("le budget", hits, store=s)
    assert "proposé" in a.text and a.status == "proposed"
