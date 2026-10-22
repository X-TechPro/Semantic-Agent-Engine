"""Persons and ownership (6.3).

The rule that matters: **a person mentioned is not a person responsible.**
Ownership requires an explicit commitment/assignment pattern. Negative cases are
unit-tested ("Karim a parlé du budget" must not make Karim the owner).

Default backend is rules + the roster seen in the corpus (no model). GLiNER is
optional and behind `slots.persons_backend = "gliner"`.
"""

from __future__ import annotations

import re

from engine.schemas import Slots, Unit

# "Marie s'engage à…" / "je m'occupe de…" / "la responsabilité revient à Karim"
OWNER_PATTERNS = [
    # French elision: `s'occupera DU budget` is not `s'occupera DE budget`.
    re.compile(r"\b([A-ZÀ-Ý][\w’'-]+)\s+(?:s['’]engage|s['’]occup(?:era|e|ions|ez|er)"
               r"|prend|gère|va gérer|se charge|tiendra|tient)\b", re.UNICODE),
    re.compile(r"\b([A-ZÀ-Ý][\w’'-]+)\s+(?:c['’]est qui|qui gère)\b", re.UNICODE),
    re.compile(r"\bc['’]est\s+([A-ZÀ-Ý][\w’'-]+)\s+qui\b", re.UNICODE),
    # The gap has to cover a full attribute phrase: "La responsabilité de la date
    # de lancement du projet Atlas revient à Adam" sits at 43 characters, and the
    # old 40-char window silently failed it (measured: 352 golden owner rows).
    re.compile(r"\b(?:responsabilité|pilotage|taskforce)[^.]{0,90}\brevient à\s+"
               r"([A-ZÀ-Ý][\w’'-]+)\b", re.UNICODE),
    # Sentence-initial "On" is capitalised, so a lower-case literal never matched.
    # `[Oo]n` fixes that without `re.IGNORECASE`, which would also un-capitalise
    # the capture class below it and let "on part sur **le** budget" name an owner.
    re.compile(r"\b[Oo]n part sur\s+([A-ZÀ-Ý][\w’'-]+)\b", re.UNICODE),
    re.compile(r"\bassigné(?:e)? à\s+([A-ZÀ-Ý][\w’'-]+)\b", re.UNICODE),
    re.compile(r"\bdemande à\s+([A-ZÀ-Ý][\w’'-]+)\b", re.UNICODE),
    re.compile(r"\bcharge\s+([A-ZÀ-Ý][\w’'-]+)\s+de\b", re.UNICODE),
]
# first-person commitment: the speaker is the owner (6.3)
SELF_COMMIT = re.compile(r"\b(?:je\s+m['’]occupe|je\s+gère|c['’]est\s+moi\s+qui|"
                         r"je\s+prends|je\s+m['’]engage|je\s+livre|je\s+fais)\b")
# explicit *non*-ownership
DISPERSE = re.compile(r"\b(?:n['’]est pas|ne\s+s['’]occupe pas|plus du tout)\b")

NAME_RE = re.compile(r"\b([A-ZÀ-Ý][a-zà-ÿ'’-]{2,})\b")
STOP_CAPS = {
    "Le", "La", "Les", "Un", "Une", "Des", "Du", "De", "En", "Et", "Ou", "Pour",
    "Sur", "Dans", "Ce", "Cet", "Cette", "Il", "Elle", "Ils", "Elles", "On",
    "Nous", "Vous", "Bonjour", "Attention", "Finalement", "Du", "Nova",
    "Atlas", "Boréal", "Cigale", "Delta", "Etincelle", "Étincelle", "Fjord",
    "Galet", "Hélios", "Iris", "Jade", "Kir", "Lagune", "Mistral", "Némo",
    "Onyx", "Palme", "Quark", "Rivage", "Sirocco", "Topaze", "Ulysse", "Vega",
    "PostgreSQL", "FastAPI", "Python", "DuckDB", "Node", "Azure", "Rust",
    "ClickHouse", "RGPD", "C", "Sauf", "Si", "Quand", "Où", "Comment",
}


def find_persons(text: str, roster: set[str] | None = None) -> list[str]:
    """Names present in the unit. With a roster, only roster names count."""
    found = [m.group(1) for m in NAME_RE.finditer(text)]
    out: list[str] = []
    for n in found:
        if n in STOP_CAPS:
            continue
        if roster and not _in_roster(n, roster):
            continue
        if n not in out:
            out.append(n)
    return out


def _in_roster(name: str, roster: set[str]) -> bool:
    if name in roster:
        return True
    low = name.lower()
    return any(r.lower() == low for r in roster)


def canonical_person(name: str, roster: set[str]) -> str | None:
    for r in roster:
        if r.lower() == name.lower():
            return r
    return None


def find_owner(unit: Unit, roster: set[str] | None = None) -> str | None:
    """Explicit ownership only. Mention ≠ responsibility."""
    text = unit.text
    if unit.speaker and SELF_COMMIT.search(text) and not DISPERSE.search(text):
        return canonical_person(unit.speaker, roster or {unit.speaker}) or unit.speaker
    for pat in OWNER_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        cand = m.group(1)
        if DISPERSE.search(text[max(0, m.start() - 30):m.end() + 30]):
            return None
        # "Marie demande à Karim de…" → Karim is asked, not the owner, unless the
        # pattern itself was the "demande à" form (delegation makes him owner).
        return canonical_person(cand, roster or {cand}) or cand
    return None


def extract(unit: Unit, roster: set[str] | None = None,
            slots: Slots | None = None) -> Slots:
    base = slots or unit.slots
    persons = find_persons(unit.text, roster)
    owner = find_owner(unit, roster)
    if owner and owner not in persons:
        persons = persons + [owner]
    return base.model_copy(update={"persons": persons, "owner": owner})
