"""Explicit state rule: confirmation survives a later proposal."""
from engine.memory.event_state import EventState, StateMention


def m(uid, value, status, turn="t"):
    return StateMention(uid, "e1", value, status, turn, "evidence")


def test_late_proposal_does_not_displace_confirmed_value():
    st = EventState("e1")
    st.apply(m("a", "800", "proposed"))
    st.apply(m("b", "800", "confirmed"))
    st.apply(m("c", "1000", "proposed"))
    assert st.current_value == "800"
    assert st.current_status == "confirmed"
    assert st.open_questions == ["1000"]
    assert [h.value for h in st.history] == ["800", "800", "1000"]


def test_open_payment_stays_unknown_until_confirmed():
    st = EventState("e1")
    st.apply(m("a", "pending", "open"))
    assert st.current_status == "open"
    assert st.answer() == "pending"
    st.apply(m("b", "300", "confirmed"))
    assert st.current_status == "confirmed"
    assert st.answer() == "300"


def test_cancelled_is_a_current_state_and_clears_open_questions():
    st = EventState("e1")
    st.apply(m("a", "200", "proposed"))
    st.apply(m("b", "0", "cancelled"))
    assert st.current_status == "cancelled"
    assert st.open_questions == []
