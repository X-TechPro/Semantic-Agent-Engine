"""`draft_email` / `create_task` / `schedule_event` / `clarify` (6.11 / 6.12).

Two invariants hold here and are tested in tests/e2e:
  * nothing is executed — every object is a proposal with
    `needs_confirmation=True`, `executed=False`;
  * nothing is invented — the recipient comes from the resolver, the address
    only from the contacts file, and the body only cites retrieved units.
Hypothesis / proposed / conditional units can never reach these operations.
"""

from __future__ import annotations

import re

from engine.agent.intents import REFUSAL
from engine.agent.recipients import ContactBook, resolve_recipient
from engine.schemas import (Clarification, DraftEmail, EventProposal, Hit, Query,
                            TaskProposal, Unit)


def guard_action(unit: Unit | None) -> str | None:
    """6.2 hard rule: a non-assertive unit never creates an action."""
    if unit is None:
        return None
    if unit.status in REFUSAL:
        return REFUSAL[unit.status]
    return None


def build_email(query: Query, hits: list[Hit], *, store, book: ContactBook,
                roster: set[str] | None = None, attribute: str | None = None,
                person: str | None = None) -> tuple[DraftEmail | Clarification, str]:
    """Returns (payload, status). `ambiguous`/`not_found` → Clarification, never
    a guessed recipient."""
    recip = resolve_recipient(store, project=query.project, attribute=attribute,
                              book=book, roster=roster, person=person)
    if recip.resolution_status in ("ambiguous", "not_found"):
        opts = recip.alternatives or ["préciser le projet", "préciser le sujet"]
        return (Clarification(
            question=("Plusieurs personnes peuvent correspondre à ce rôle. "
                      "Laquelle voulez-vous adresser ce mail ?"),
            options=opts, reason=recip.justification), recip.resolution_status)

    src = [h.unit.unit_id for h in hits[:6]]
    body_units = [h.unit for h in hits[:6]]
    subject = _subject(query, recip, attribute)
    body = _body(recip, query, body_units)
    return (DraftEmail(recipient=recip, subject=subject, body=body,
                       source_unit_ids=src), recip.resolution_status)


def _subject(query: Query, recip, attribute: str | None) -> str:
    topic = (attribute or query.slots.subject or "le sujet en cours")
    return f"Projet {query.project or '—'} : {topic} — version validée demandée"


def _body(recip, query: Query, units: list[Unit]) -> str:
    lines = [f"Bonjour {recip.name},", ""]
    facts = [u.text.strip() for u in units if u.is_current][:5]
    if facts:
        lines.append("Voici l'état actuel d'après nos sources :")
        lines += [f"- {f}" for f in facts]
        lines.append("")
    ask = query.text.strip()
    lines.append(f"Ma demande : {ask}".rstrip("?") + "?")
    lines.append("")
    lines.append("Merci de confirmer la version en vigueur avant l'échéance indiquée.")
    lines.append("")
    lines.append("Cordialement,")
    return "\n".join(lines)


def build_task(query: Query, hits: list[Hit], *, blocking_status: str | None
               ) -> TaskProposal | Clarification:
    if blocking_status:
        return Clarification(question=("Ce point n'est pas encore acté. Dois-je quand "
                                        "même créer une tâche ?"),
                             options=["Oui, créer quand même", "Non"],
                             reason=blocking_status)
    owner = _first(hits, "owner")
    due = _first(hits, "due") or _first(hits, "date_value")
    title = _first(hits, "action") or query.text.strip()[:80]
    missing = [f for f, v in (("owner", owner), ("due", due)) if not v]
    return TaskProposal(title=title, owner=owner, due=due, project=query.project,
                        description=query.text.strip(),
                        source_unit_ids=[h.unit.unit_id for h in hits[:5]],
                        missing_fields=missing)


def build_event(query: Query, hits: list[Hit], *, blocking_status: str | None
                ) -> EventProposal | Clarification:
    if blocking_status:
        return Clarification(question=("Rien n'est acté pour ce point : je ne "
                                        "programme pas d'événement. Voulez-vous "
                                        "malgré tout le créer ?"),
                             options=["Oui, planifier", "Non"], reason=blocking_status)
    start = _first(hits, "date_value")
    attendees = sorted({p for h in hits[:5] for p in h.unit.slots.persons})
    missing = [] if start else ["start"]
    return EventProposal(title=query.text.strip()[:80], start=start,
                         attendees=attendees, project=query.project,
                         source_unit_ids=[h.unit.unit_id for h in hits[:5]],
                         missing_fields=missing)


def _first(hits: list[Hit], field: str) -> str | None:
    for h in hits:
        v = getattr(h.unit.slots, field, None)
        if v:
            return v
    return None
