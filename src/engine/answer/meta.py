"""Deterministic lookups for meta questions: first/last decision or message.

The router (`engine.query.route`) decides *what* is asked; this module answers
from the store, so "when" and "who" are read from the evidence turn instead of
being generated.
"""
from __future__ import annotations

from engine.answer.v4 import Answer, _citations_for


def decision_rows(store) -> list[tuple[tuple, list[dict]]]:
    """Live states grouped by the turn that confirmed them, oldest first."""
    groups: dict[tuple, list[dict]] = {}
    for row in store.state_events():
        if row.get("status") not in ("confirmed", "cancelled", "completed"):
            continue
        unit = store.unit(str(row.get("evidence_unit") or "")) or {}
        msg = store.message(unit.get("message_id", "")) or {}
        key = (str(msg.get("timestamp") or ""), str(msg.get("speaker") or ""),
               unit.get("message_id", ""))
        groups.setdefault(key, []).append(row)
    return sorted(groups.items(), key=lambda kv: kv[0][0])


def _decision(store, ordinal: str) -> Answer | None:
    rows = decision_rows(store)
    if not rows:
        return None
    (when, who, _mid), facts = rows[0] if ordinal == "first" else rows[-1]
    ev = str(facts[0].get("evidence_unit") or "")
    body = "; ".join(f"{r.get('slot')} = {r.get('value')}" for r in facts)
    when_txt = when[:19].replace("T", " ") or "date inconnue"
    text = f"{body} — décidé le {when_txt} par {who or '?'} [{ev}]"
    return Answer(text, _citations_for([ev], store, []), None, unknown=False,
                  verified=True, status=str(facts[0].get("status") or "unknown"))


def _message(store, ordinal: str) -> Answer | None:
    rows = [m for m in store.all_messages() if m.get("timestamp")]
    if not rows:
        return None
    rows.sort(key=lambda m: str(m.get("timestamp")))
    msg = rows[0] if ordinal == "first" else rows[-1]
    units = store.units_for_message(msg["message_id"])
    cites = _citations_for([units[0]["unit_id"]], store, []) if units else []
    when = str(msg.get("timestamp"))[:19].replace("T", " ")
    text = f"{msg.get('text', '')} ({when} / {msg.get('speaker') or '?'})"
    return Answer(text, cites, None, unknown=False, verified=True,
                  status="answered")


def lookup(store, obj: str, ordinal: str) -> Answer | None:
    if store is None or ordinal not in ("first", "last"):
        return None
    if obj == "decision":
        return _decision(store, ordinal)
    if obj == "message":
        return _message(store, ordinal)
    return None
