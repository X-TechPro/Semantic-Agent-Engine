"""Unresolved-currency guard (D-047) — refuse when two live versions compete.

The measured failure this fixes: on real meeting speech the linker cannot tell
which statement replaced which (D-046: a perfect threader is needed), yet the
answer template still led with rank 1 and called it current. That is a guess
presented as a fact.

This module answers one question: *does the evidence shown contain two current
statements that assert different values about the same thing, with no relation
deciding between them?* If yes, the honest answer is "I cannot tell which one is
in force", not the top hit. It is deliberately narrow:

  * only `is_current` units whose speech act can hold a value
    (`VALUE_BEARING_TYPES`), never questions/risks/commitments;
  * soft statuses (proposed/hypothesis/conditional) never create a conflict —
    they are not in force to begin with;
  * when one assertive statement exists it is the head and no conflict is
    raised; abstention is for the case with **no** head or **several**;
  * two different resolved subjects never compete (line 403);
  * when neither side has a resolved subject, a conflict needs shared content
    words *and* a value change — two stray numbers in one meeting are not a
    disagreement.

No new state, no model: this reads the units the retriever already returned.
"""
from __future__ import annotations

from engine.link.supersede import VALUE_BEARING_TYPES
from engine.link.values import content_words, value_witness
from engine.retrieve.validity import SOFT_STATUSES
from engine.schemas import Unit

#: statuses that put a statement in force
ASSERTIVE = frozenset({"confirmed", "cancelled"})

UNRESOLVED_REASON = (
    "Plusieurs versions se contredisent et aucun lien ne dit laquelle est en "
    "vigueur ; je ne tranche pas.")
UNRESOLVED_LEAD = (
    "Je ne peux pas dire quelle version est en vigueur : plusieurs éléments se "
    "contredisent et aucun lien ne tranche.")
#: the honest label when the answer cannot be tied to a topic or a status
CURRENCY_CAVEAT = (
    "Attention : je ne peux pas confirmer que cet élément est encore la version "
    "valable (aucun sujet ni statut n'y est rattaché).")


def _compete(a: Unit, b: Unit) -> str | None:
    """The kind of value change between two statements, or None."""
    sa = (a.slots.subject or "").strip()
    sb = (b.slots.subject or "").strip()
    if sa and sb and sa != sb:
        return None                       # two known, different slots
    if sa and sb:
        return value_witness(a.text, b.text)
    from engine.slots.subjects import match_attribute

    attr_a, attr_b = match_attribute(a.text), match_attribute(b.text)
    if attr_a and attr_a == attr_b:
        return value_witness(a.text, b.text)
    # neither an explicit slot nor a shared attribute: demand shared content, so
    # two unrelated numbers/dates in one meeting never look like a conflict
    if len(content_words(a.text) & content_words(b.text)) < 2:
        return None
    return value_witness(a.text, b.text)


def competing_versions(units: list[Unit], *, max_list: int = 4) -> list[Unit]:
    """Current value assertions that disagree with no relation between them.

    Returns the largest disagreeing group (ordered by date, then turn), or [].
    """
    pool = [u for u in units
            if u.is_current
            and u.type in VALUE_BEARING_TYPES
            and u.status not in SOFT_STATUSES]
    # de-duplicate restatements of the same text before looking for conflict
    seen: dict[str, Unit] = {}
    for u in pool:
        seen.setdefault((u.text or "").strip().lower(), u)
    pool = list(seen.values())
    if len(pool) < 2:
        return []
    assertive = [u for u in pool if u.status in ASSERTIVE]
    if len(assertive) == 1:
        return []                         # one in-force statement = a head exists
    if len(assertive) >= 2:
        pool = assertive

    parent = {u.unit_id: u.unit_id for u in pool}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i, a in enumerate(pool):
        for b in pool[i + 1:]:
            if _compete(a, b):
                union(a.unit_id, b.unit_id)

    groups: dict[str, list[Unit]] = {}
    for u in pool:
        groups.setdefault(find(u.unit_id), []).append(u)
    best = max(groups.values(), key=len) if groups else []
    if len(best) < 2:
        return []
    best.sort(key=lambda u: (u.source_date, u.turn_index if u.turn_index is not None
                             else 0, u.unit_id))
    return best[:max_list]


def unconfirmed_currency(unit: Unit) -> bool:
    """True when this unit's status cannot ground its currency.

    The reader's own vocabulary says it: `mention` "names a subject, asserts no
    state". A subject therefore proves *what* the sentence is about, never that
    it is still true — the measured lesson of D-046, where 11 of 40 stale units
    carried a resolved subject and were still confidently presented. Grounds are
    `confirmed`/`cancelled` (in force) and the soft statuses (already labelled
    as proposals by the answer's own note); `mention`/None is unknown.
    """
    if unit.type not in VALUE_BEARING_TYPES:
        return False
    return unit.status is None or str(unit.status) == "mention"


def unresolved_answer(units: list[Unit]) -> str:
    lines = [UNRESOLVED_LEAD, "Versions trouvées :"]
    for u in units:
        bits = [u.source_date.isoformat()]
        if u.speaker:
            bits.append(u.speaker)
        if u.status:
            bits.append(str(u.status))
        lines.append(f"- {' / '.join(bits)} — {u.text} [{u.unit_id}]")
    lines.append("Précisez la date ou la réunion pour que je tranche.")
    return "\n".join(lines)


__all__ = ["competing_versions", "unconfirmed_currency", "unresolved_answer",
           "UNRESOLVED_REASON", "UNRESOLVED_LEAD", "CURRENCY_CAVEAT"]
