"""Deterministic coarse threads for the linker (D-046).

The measured failure this addresses: `supersede.evaluate` can only fire when
`slots.subject` is identical on both sides, and real ASR resolves a subject for
~2% of units (17/781, `runs/summre_reach`). With the production resolver, only
1 of 32 train and **0 of 11 sealed-holdout** gold supersede pairs even reach the
guards, so recall is capped at zero before any model is asked anything.

This index supplies the missing *pair-level* identity cheaply: a unit's own
resolved subject when the reader gave one, else the coarse slot its words name
(`values.slot_signature`: an attribute, a date, an amount or a bare number).
Units sharing a slot inside one source form a thread, and a pair in one thread
may be judged by `evaluate(..., same_thread=True)` instead of being refused at
G1. It is deliberately NOT a semantic topic model — every such attempt was
measured and refused (D-026…D-037, D-042); this only replaces a missing key
with the strongest deterministic evidence available.

Threads are per source and built in ingestion order, so "old precedes new" is
true by construction inside a thread; that is why `evaluate` can skip the
same-date conflict rule for pair-local pairs.
"""
from __future__ import annotations

import re

from engine.link import values
from engine.schemas import Unit
from engine.slots import subjects as S

#: Meeting topics, as words people actually say. This is a *persistent* signal,
#: not a per-sentence classifier: a topic named once ("la billetterie") stays
#: active until another topic is named or the window expires, so the next turns
#: ("en ligne ça peut être sympa", "une réservation en ligne") inherit it even
#: though they never repeat the word. That is the trace's failure: turn 216 says
#: "billetterie", turns 218–226 never do, and the linker sees no shared key.
TOPIC_FORMS: tuple[tuple[str, str], ...] = tuple(sorted([
    ("billetterie", ("billetterie", "billets", "billet", "reservation",
                     "reservations", "reserver", "guichet", "entrees", "entree")),
    ("programmation", ("programmation", "films", "film", "seance", "projection",
                       "artistes", "artiste", "concert", "concerts", "musique",
                       "chanteur", "chanteuse", "groupe")),
    ("animations", ("animations", "animation", "buvette", "snack", "stand",
                    "stands", "atelier", "ateliers", "confetti", "escape game")),
    ("calendrier", ("calendrier", "horaire", "horaires", "planning", "semaine",
                    "semaines", "ouverture", "fermeture")),
    ("organisation", ("organisation", "logistique", "equipe", "equipes",
                      "benevoles", "benevole", "staff", "repartition",
                      "coordination")),
    ("communication", ("communication", "promo", "promotion", "affiche",
                       "affiches", "reseaux", "instagram", "facebook",
                       "publication", "presse", "flyers", "flyer")),
    ("financement", ("financement", "budget", "subvention", "subventions",
                     "sponsor", "sponsors", "recettes", "depenses", "couts",
                     "tarif", "tarifs", "prix")),
    ("securite", ("securite", "secours", "bruit", "bouchons", "bracelets",
                  "alcool", "pompier", "pompiers")),
    ("lieu", ("lieu", "lieux", "salle", "arene", "site", "espace", "endroit")),
    ("materiel", ("materiel", "lumiere", "chaises", "tables", "tente", "tentes",
                  "barnum", "barrieres")),
    ("public", ("public", "spectateurs", "participants", "invites", "visiteurs")),
    ("restauration", ("restauration", "repas", "pizzas", "pizza", "nourriture",
                      "boissons", "boisson", "food truck")),
    ("transport", ("transport", "parking", "navette", "acces")),
], key=lambda x: -max(len(f) for f in x[1])))

_FOLDED_FORMS: tuple[tuple[str, str], ...] = tuple(
    (S.fold(form), topic) for topic, forms in TOPIC_FORMS for form in forms)


def detect_topic(text: str) -> str | None:
    """First named topic in the turn, longest surface form first."""
    hay = S.fold(text or "")
    for form, topic in _FOLDED_FORMS:
        if re.search(rf"(?<!\w){re.escape(form)}(?!\w)", hay):
            return topic
    return None


class ThreadIndex:
    """Recent units per coarse slot, for one source being ingested."""

    def __init__(self, window: int = 8, topic_window: int = 12):
        self.window = max(1, int(window))
        self.topic_window = max(1, int(topic_window))
        self._by_slot: dict[str, list[Unit]] = {}
        self._topic_of: dict[str, str | None] = {}
        self._active_topic: str | None = None
        self._age = 0

    def _advance_topic(self, text: str) -> str | None:
        """A named topic starts/refreshes it; silence ages it out."""
        topic = detect_topic(text)
        if topic:
            self._active_topic = topic
            self._age = 0
        elif self._active_topic:
            self._age += 1
            if self._age > self.topic_window:
                self._active_topic = None
        return self._active_topic

    def topic_of(self, unit: Unit) -> str | None:
        if unit.unit_id in self._topic_of:
            return self._topic_of[unit.unit_id]
        return detect_topic(unit.text) or self._active_topic

    def slot_of(self, unit: Unit) -> str | None:
        """Resolved subject first, then the live topic, then words."""
        subject = getattr(getattr(unit, "slots", None), "subject", None)
        if subject:
            return f"k:{subject}"
        topic = self.topic_of(unit)
        if topic:
            return f"t:{topic}"
        return values.slot_signature(unit.text)

    def add(self, unit: Unit) -> None:
        topic = self._advance_topic(unit.text)
        self._topic_of[unit.unit_id] = topic
        slot = self.slot_of(unit)
        if slot is None:
            return
        bucket = self._by_slot.setdefault(slot, [])
        bucket.append(unit)
        if len(bucket) > self.window:
            del bucket[: len(bucket) - self.window]

    def context(self, unit: Unit) -> list[Unit]:
        """Thread-mates that arrived before `unit`, most recent last."""
        slot = self.slot_of(unit)
        if slot is None:
            return []
        return list(self._by_slot.get(slot, []))

    def same_thread(self, a: Unit, b: Unit) -> bool:
        sa, sb = self.slot_of(a), self.slot_of(b)
        return bool(sa) and sa == sb


__all__ = ["ThreadIndex"]
