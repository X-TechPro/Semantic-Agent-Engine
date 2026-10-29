"""Separate entity index for people, projects, amounts, dates, actions, events."""
from __future__ import annotations

import re
from collections import defaultdict

from engine.link.thread import detect_topic

MONTHS = {"janvier", "fevrier", "février", "mars", "avril", "mai", "juin",
          "juillet", "aout", "août", "septembre", "octobre", "novembre", "decembre", "décembre"}
AMOUNT = re.compile(r"\b\d[\d\s]*(?:euros?|€)\b", re.I)
DATE = re.compile(r"\b\d{1,2}\s+(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b", re.I)
CAP = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}\b")
ACTION = re.compile(r"\b(?:payer|paiement|valider|validation|annuler|annulation|"
                    r"reserver|réservation|envoyer|envoi|recruter|recrutement|"
                    r"confirmer|confirmation|devis|facture|budget|date|lieu|"
                    r"traiteur|prestataire|billet|billetterie)\b", re.I)


def extract_entities(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for m in AMOUNT.findall(text or ""):
        out["amount"].append(m.lower())
    for m in DATE.findall(text or ""):
        out["date"].append(m.lower())
    for m in CAP.findall(text or ""):
        out["name"].append(m.lower())
    for m in ACTION.findall(text or ""):
        out["action"].append(m.lower())
    topic = detect_topic(text or "")
    if topic:
        out["event"].append(topic)
    return dict(out)


class EntityIndex:
    def __init__(self):
        self._index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def add(self, unit_id: str, entities: dict[str, list[str]]) -> None:
        for etype, values in entities.items():
            for value in values:
                self._index[etype][value].add(unit_id)

    def search(self, etype: str, value: str) -> set[str]:
        return set(self._index.get(etype, {}).get(value.lower(), set()))

    def types(self) -> list[str]:
        return sorted(self._index)


__all__ = ["extract_entities", "EntityIndex"]
