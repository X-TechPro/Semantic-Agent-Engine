"""Rule-based slot extractor — the 2.2 `read`/`slots` boundary implementation."""

from __future__ import annotations

import re

from engine.schemas import Slots, Unit
from engine.slots import persons as P
from engine.slots import subjects as S
from engine.slots.regex_dates import parse_amount, parse_date, parse_due

_ACTION_RE = re.compile(
    r"\b(?:mettre à jour|envoyer|préparer|rédiger|relancer|valider|arbitrer|livrer|"
    r"planifier|checker|vérifier|contacter|demander|organiser|ajuster)\b[^.;]{2,60}",
    re.IGNORECASE)


def extract(unit: Unit, roster: set[str] | None = None, *,
            subject_model=None) -> Slots:
    """Fill persons/owner/subject/date/amount/due/action from surface rules.

    Subject resolution is (a) the attribute lexicon only. `subject_model` is
    consulted when rules miss, and the only supplier left is the annotation
    gold (`gold_subject.py`, demos) — 6.3's trained/prompted extractors were
    measured and removed (D-037), so nothing automatic proposes a key here;
    an absent annotation leaves `subject` None exactly as before, and the
    guards treat a missing subject as "undecidable", never as a guess.
    """
    text = unit.text
    found = P.find_persons(text, roster)
    owner = P.find_owner(unit, roster)
    if owner and owner not in found:
        found = found + [owner]

    owner_slot = bool(re.search(r"\bpilotage\b|\bresponsab|\bqui gère\b|\bcharge\b",
                                text, re.IGNORECASE))
    subject = S.resolve(text, unit.project, owner_slot=owner_slot)
    if subject is None and subject_model is not None:
        subject = subject_model.subject_for(text, unit.project,
                                            owner_slot=owner_slot)
    # An *ownership* assertion is about the owner slot, not the value slot.
    # Without this promotion "Marie s'occupe du budget" is filed under
    # atlas.budget, where the budget-value chain supersedes it and a genuine
    # two-owner ambiguity becomes invisible (measured on the e2e run).
    if owner and subject and ".owner." not in subject:
        subject = S.as_owner_key(subject)
    if subject is None and unit.project:
        # attribute not in the lexicon: fall back to the project-level key so the
        # linker can still decide "same slot" on explicit restatements
        subject = None

    date_iso, _rule = parse_date(text, unit.source_date)
    amount = parse_amount(text)
    due_iso, _ = parse_due(text, unit.source_date)
    m = _ACTION_RE.search(text)
    action = m.group(0).strip().rstrip(".,;:") if m else None

    return Slots(
        persons=found, owner=owner, subject=subject,
        date_value=date_iso, amount=amount, due=due_iso, action=action,
    )


def merge(base: Slots, extra: Slots) -> Slots:
    """Prefer non-empty values from `extra`."""
    data = base.model_dump()
    for k, v in extra.model_dump().items():
        if v in (None, [], ""):
            continue
        if k == "persons":
            data[k] = list(dict.fromkeys((data.get(k) or []) + v))
        else:
            data[k] = v
    return Slots(**data)
