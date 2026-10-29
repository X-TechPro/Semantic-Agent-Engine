"""Query interpretation and light reference resolution for v4."""
from __future__ import annotations

import re
from dataclasses import dataclass

SLOTS = {
    "amount": ("combien", "montant", "budget", "prix", "euros", "€", "coût", "cout"),
    "date": ("quand", "date", "jour", "mois", "échéance", "echeance", "deadline"),
    "person": ("qui", "personne", "responsable", "owner", "destinataire"),
    "venue": ("où", "ou", "lieu", "salle", "endroit"),
    "action": ("action", "tâche", "tache", "faire", "envoyer", "payer"),
}
REFS = re.compile(r"\b(?:il|elle|ils|elles|le|la|les|lui|leur|ça|ca|cela|"
                  r"celui|celle|alors|ensuite|du coup|donc)\b", re.I)


@dataclass
class QueryFrame:
    text: str
    intent: str
    slot: str = ""
    references: list[str] = None
    needs_resolution: bool = False


def interpret(text: str) -> QueryFrame:
    low = text.lower()
    refs = [m.group(0).lower() for m in REFS.finditer(text)]
    slot = ""
    for name, words in SLOTS.items():
        if any(w in low for w in words):
            slot = name
            break
    if any(w in low for w in ("compare", "différence", "difference", "changé", "change")):
        intent = "compare"
    elif any(w in low for w in ("dossier", "synthèse", "synthese", "rapport")):
        intent = "dossier"
    elif any(w in low for w in ("mail", "email", "courriel")):
        intent = "email"
    elif any(w in low for w in ("tâche", "tache", "action", "faire")):
        intent = "task"
    else:
        intent = "answer"
    return QueryFrame(text, intent, slot, refs, bool(refs) and not slot)


__all__ = ["QueryFrame", "interpret"]
