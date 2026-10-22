"""Recipient resolution (6.12) — the never-fabricate path.

Order of evidence, exactly as specified:
  1. candidates = units carrying `slots.owner` for the requested role/topic, with
     a commitment/assignment pattern;
  2. explicitness + recency: a *mention* is not responsibility, a superseded
     assignment is not current, the most recent explicit assignment wins;
  3. one candidate → `resolved`; several at the same recency → `ambiguous`
     (caller must clarify); none → `not_found`;
  4. the address comes **only** from the mock directory; a known person with no
     address → `email_missing` with `email=None`. An address is never invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from engine.schemas import Recipient, Unit

OWNER_TYPES = ("decision", "commitment")
MENTION_ONLY_TYPES = ("fact", "other", "risk")


@dataclass
class ContactBook:
    """name -> email | None. Anything absent here simply has no address."""

    emails: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "ContactBook":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text(encoding="utf-8"))
        emails: dict[str, str] = {}
        for k, v in raw.items():
            if isinstance(v, dict):
                emails[k] = v.get("email")
            elif isinstance(v, str):
                emails[k] = v
        return cls(emails=emails)

    def email_for(self, name: str) -> str | None:
        for k, v in self.emails.items():
            if k.lower() == name.lower():
                return v
        return None

    def known(self, name: str) -> bool:
        return any(k.lower() == name.lower() for k in self.emails)


def candidates(store, project: str | None, attribute: str | None,
               roster: set[str] | None = None) -> list[Unit]:
    """Units that assert ownership for `project.attribute` (or any owner unit)."""
    subject = f"{(project or '').lower()}.owner.{attribute}" if attribute and project \
        else (f"{project.lower()}.owner" if project else None)
    units: list[Unit] = []
    pool = store.all_units()
    for u in pool:
        if not u.slots.owner:
            continue
        if u.type in MENTION_ONLY_TYPES and u.type not in OWNER_TYPES:
            continue
        if subject and u.slots.subject and u.slots.subject.lower() != subject.lower():
            # The subject key *is* what the sentence is about, so a different
            # attribute is a different responsibility. The old exemption kept any
            # unit whose subject ended in ".budget" whatever was asked, and since
            # `atlas.owner.budget` already matches the budget query exactly it never
            # fired for its own case: measured on the demo corpus, `budget`,
            # `recrutement`, `planning` and `conformite` all returned Marie with a real
            # address, i.e. the module fabricated a responsible party for 3 of 4
            # questions (and `data/contacts.json` says those roles belong to Nadia and
            # Julien). Comparing case-insensitively covers the only thing the
            # exemption was actually needed for.
            if attribute:
                continue
            # No attribute was asked: then any owner unit of this project is the
            # responsible party, which is what "who owns this project" means.
            if not u.slots.subject.lower().startswith(subject.lower() + "."):
                continue
        if project and u.project and u.project.lower() != project.lower():
            continue
        if roster and u.slots.owner not in roster:
            continue
        units.append(u)
    if not units and attribute:  # relax: owner slot may be the bare attribute
        for u in pool:
            if u.slots.owner and (u.slots.subject or "").endswith(attribute):
                units.append(u)
    return units


def resolve_recipient(store, *, project: str | None, attribute: str | None,
                      book: ContactBook, roster: set[str] | None = None,
                      person: str | None = None) -> Recipient:
    if person:
        return _finalize(person, [], book, "resolved",
                         "person named explicitly in the request")
    cands = candidates(store, project, attribute, roster)
    if not cands:
        return Recipient(name=(attribute or "unknown"), resolution_status="not_found",
                         justification=("aucune unité ne désigne explicitement un "
                                        "responsable pour ce sujet"),
                         alternatives=[])
    current = [u for u in cands if u.is_current]
    pool = current or cands
    # most recent explicit assignment wins (6.12.2)
    pool.sort(key=lambda u: (u.source_date, u.unit_id))
    latest = pool[-1]
    tied = [u for u in pool if u.source_date == latest.source_date]
    names = sorted({u.slots.owner for u in tied if u.slots.owner})
    evidence = [u.unit_id for u in tied]
    if len(names) > 1:
        return Recipient(name=" / ".join(names), resolution_status="ambiguous",
                         evidence_unit_ids=evidence, alternatives=names,
                         justification=("plusieurs responsables plausibles à la même "
                                        "date: " + ", ".join(names)))
    return _finalize(names[0], evidence, book, "resolved",
                     f"assignation explicite et à jour ({latest.source_date})")


def _finalize(name: str, evidence: list[str], book: ContactBook,
              status: str, why: str) -> Recipient:
    email = book.email_for(name)
    if status == "resolved" and not email:
        status = "email_missing" if book.known(name) or email is None else "not_found"
        return Recipient(name=name, email=None, evidence_unit_ids=evidence,
                         resolution_status="email_missing",
                         justification=why + " — adresse inconnue: aucune n'est inventée")
    return Recipient(name=name, email=email, evidence_unit_ids=evidence,
                     resolution_status=status, justification=why)
