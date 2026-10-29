"""v4 §5: proposal stays open, live state wins, history is kept, unknown never acts."""
from engine.schemas_v4 import SemanticUnit
from engine.store.v4 import V4Store
from engine.threads.state_v4 import derive, event_key, replaced_for, state_for


def _unit(uid, text, status, slot="budget", value="800 euros", project="Atlas",
          kind="decision"):
    return SemanticUnit(unit_id=uid, message_id=uid.split(":u")[0],
                        chunk_ids=(f"{uid}:s0",), text=text, kind=kind,
                        status=status, intent="answer", slot=slot, value=value)


def _apply(store, unit, project="Atlas", prior=None):
    for row in derive(unit, project, prior_value=prior):
        store.add_state(row)


def test_a_proposal_is_open_and_a_later_proposal_never_displaces_a_confirmation():
    s = V4Store()
    _apply(s, _unit("m1:u0", "budget proposé 1200", "proposed", value="1200 euros"))
    assert s.current_state("Atlas:budget", "budget")["open"] is True

    _apply(s, _unit("m2:u0", "budget validé 800", "confirmed", value="800 euros"),
           prior="1200 euros")
    _apply(s, _unit("m3:u0", "on pourrait monter à 1500", "proposed",
                    value="1500 euros"), prior="800 euros")

    cur = s.current_state("Atlas:budget", "budget")
    assert cur["value"] == "800 euros" and cur["status"] == "confirmed"
    assert cur["open"] is False
    assert [r["status"] for r in s.history("Atlas:budget", "budget")] == [
        "proposed", "confirmed", "proposed"]


def test_cancellation_updates_current_and_keeps_both_values():
    s = V4Store()
    _apply(s, _unit("m1:u0", "validé 800", "confirmed", value="800 euros"))
    _apply(s, _unit("m2:u0", "finalement on annule", "cancelled",
                    value="finalement on annule"), prior="800 euros")
    cur = s.current_state("Atlas:budget", "budget")
    assert cur["status"] == "cancelled" and cur["value"] == "800 euros"
    assert cur["transition"]["from"] == "800 euros"
    assert cur["transition"]["from_status"] == "confirmed"


def test_superseding_appends_a_replaced_row_and_keeps_the_old_value():
    s = V4Store()
    old = _unit("m1:u0", "budget 1200", "confirmed", value="1200 euros")
    new = _unit("m2:u0", "budget 800", "confirmed", value="800 euros")
    _apply(s, old)
    for row in derive(new, "Atlas", superseded=[old]):
        s.add_state(row)
    statuses = [(r["value"], r["status"]) for r in
                s.history("Atlas:budget", "budget")]
    assert ("1200 euros", "replaced") in statuses
    assert s.current_state("Atlas:budget", "budget")["value"] == "800 euros"


def test_unknown_status_is_history_only():
    s = V4Store()
    _apply(s, _unit("m1:u0", "peut-être le budget", "unknown", value="800 euros"))
    assert s.current_state("Atlas:budget", "budget") is None
    assert len(s.history("Atlas:budget", "budget")) == 1


def test_state_needs_a_slot_and_a_value():
    s = V4Store()
    assert state_for(_unit("m1:u0", "x", "confirmed", slot="", value=""),
                     "Atlas") is None
    assert state_for(_unit("m1:u0", "x", "confirmed", value=""), "Atlas") is None
    assert s.state_events() == []


def test_event_key_separates_projects():
    a = state_for(_unit("m1:u0", "x", "confirmed", value="800"), "Atlas")
    b = state_for(_unit("m1:u0", "x", "confirmed", value="900"), "Borealis")
    assert a.event_id != b.event_id
    assert a.event_id == event_key("Atlas", "budget")


def test_replaced_row_id_carries_the_replacing_unit():
    old = _unit("m1:u0", "1200", "confirmed", value="1200")
    new = _unit("m2:u0", "800", "confirmed", value="800")
    row = replaced_for(old, "Atlas", new)
    assert new.unit_id in row.state_id
    assert row.status == "replaced"
