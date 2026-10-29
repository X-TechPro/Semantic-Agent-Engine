"""§5 append-only state derivation from a read unit.

The rules are the architecture's, not recency:

* a `proposed` unit stays open; it is current only while nothing live exists;
* `confirmed` / `cancelled` / `completed` update the current view;
* a unit whose status cannot be read is appended as `unknown` — it is history,
  never current, so a missing evidence never silently becomes the answer;
* a superseded value gets a `replaced` row; the old value is not deleted.

Rows are keyed by (project, slot) and by the evidence unit, so re-ingesting the
same message is idempotent and cannot duplicate state.
"""
from __future__ import annotations

from engine.schemas_v4 import SemanticUnit, StateEvent

STATUS_TO_STATE = {
    "proposed": "proposed",
    "confirmed": "confirmed",
    "cancelled": "cancelled",
    "completed": "completed",
    "unknown": "unknown",
}


def event_key(project: str, slot: str, point: str = "") -> str:
    key = f"{project or 'default'}:{slot or 'subject'}"
    return f"{key}:{point}" if point else key


def state_for(unit: SemanticUnit, project: str,
              inherit_value: bool = False,
              prior_value: str | None = None) -> StateEvent | None:
    """One row per unit that names a slot and carries a value.

    A cancellation/completion that names no new value inherits the value it
    closes (`inherit_value`), because §5 keeps "both values visible": the row
    says "this value is now cancelled", not "the whole sentence is the value".
    """
    if not unit.slot:
        return None
    value = (unit.value or "").strip()
    if inherit_value and prior_value:
        value = prior_value
    if not value:
        return None
    if not inherit_value and value.strip().lower() == (unit.text or "").strip().lower():
        # the reader fell back to the clause: not a value, and letting it in
        # lets "ok je mets à jour le budget" overwrite a real amount
        return None
    status = STATUS_TO_STATE.get(unit.status, "unknown")
    if status == "replaced":
        return None
    return StateEvent(
        state_id=f"{event_key(project, unit.slot, unit.point)}:{unit.unit_id}",
        event_id=event_key(project, unit.slot, unit.point), slot=unit.slot,
        value=value, status=status,
        evidence_unit=unit.unit_id, origin=unit.origin)


def replaced_for(old: SemanticUnit, project: str, by: SemanticUnit
                 ) -> StateEvent | None:
    """The `replaced` row a supersede appends for the old value.

    The id includes the *replacing* unit: an old value can be superseded twice
    (two later turns link to it), and append-only means both rows are kept
    rather than one silently overwriting the other.
    """
    if not old.slot or not (old.value or "").strip():
        return None
    return StateEvent(
        state_id=(f"{event_key(project, old.slot, old.point)}:{old.unit_id}"
                  f":replaced:{by.unit_id}"),
        event_id=event_key(project, old.slot, old.point), slot=old.slot,
        value=old.value.strip(), status="replaced",
        evidence_unit=by.unit_id, origin=old.origin)


def derive(unit: SemanticUnit, project: str,
           superseded: list[SemanticUnit] | None = None,
           prior_value: str | None = None) -> list[StateEvent]:
    out: list[StateEvent] = []
    inherit = (unit.status in ("cancelled", "completed")
               and (unit.value or "").strip() == (unit.text or "").strip())
    own = state_for(unit, project, inherit_value=inherit,
                    prior_value=prior_value)
    if own is not None:
        out.append(own)
    for old in superseded or []:
        row = replaced_for(old, project, unit)
        if row is not None:
            out.append(row)
    return out


__all__ = ["derive", "state_for", "replaced_for", "event_key", "STATUS_TO_STATE"]
