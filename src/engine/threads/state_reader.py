"""Node 2-read: turn -> grounded state updates.

The reader does not append history. It detects the decision point, extracts a
value grounded in the current turn, and lets the state table decide what is
open/confirmed. The next read can then use the open rows, not the transcript.

This is deliberately a small, inspectable layer. It is not a language model and
does not guess a value that is absent from the turn.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from engine.read.baseline_rules import RuleReader
from engine.threads.state import DecisionState, StateUpdate

MONTHS = ("janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet",
          "aout", "septembre", "octobre", "novembre", "decembre")
WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
            "dimanche", "week end", "weekend")
SEASONS = ("ete", "hiver", "printemps", "automne", "vacances")
NUM = r"(?:\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze)"

#: point -> value patterns, first match wins. Ordered most-specific first.
VALUE_PATTERNS: dict[str, tuple[str, ...]] = {
    "duration": (
        rf"\b{NUM}\s+(?:jours?|semaines?|mois|ans?)\b",
        r"\b(?:un|deux|trois|quatre|cinq|six|sept)\s+mois\s+a\s+l avance\b",
    ),
    "time": (r"\b\d{1,2}\s*h(?:\d{2})?\b", r"\b\d{1,2}\s+heures?\b"),
    "date": (
        r"\b(?:debut|mi|fin)\s+(?:" + "|".join(MONTHS) + r")\b",
        r"\b(?:" + "|".join(MONTHS) + r")\b",
        r"\b(?:" + "|".join(WEEKDAYS) + r")\b",
        r"\b(?:ete|hiver|printemps|automne|vacances)\b",
    ),
    "budget": (
        r"\b(?:\d[\d\s.,]*)\s*(?:euros?|€)\b",
        r"\b(?:budget|prix|tarif|paiement|payer|paie|subventions?)\b",
    ),
    "count": (rf"\b{NUM}\s+(?:films?|personnes?|places?|benevoles?|volontaires?)\b",),
    "food": (r"\b(?:buffet|buvettes?|snacks?|food\s?truck|repas|pizzas?|nourriture|boissons?)\b",),
    "mode": (r"\b(?:billetterie|reservations?|pre\s?vente|prevente|en ligne|sur place)\b",),
    "venue": (r"\b(?:salle|arene|arena|place|terrain|lieu|espace|scene|gradins)\b",),
    "transport": (r"\b(?:velibs?|velos?|parking|navette|transport|bus|voitures?)\b",),
    "public": (r"\b(?:familles?|public|spectateurs?|habitants?|etudiants?)\b",),
    "programme": (
        r"\b(?:film|films|artiste|artistes|concert|musique|chanson|chansons|"
        r"confetti|dj|playlist|feu d artifice)\b",
        r"\b(?:pop|rock|jazz|rap|classique|electro)\b",
    ),
    "role": (
        r"\b(?:responsables?|charge(?:e)?|directeur|coordinateur|community manager|"
        r"je m occupe|qui s occupe|volontaires?|recrutement|recruter)\b",
    ),
    "security": (r"\b(?:securite|vigiles?|vigipirate|pompiers?|secours|bracelets?|bouchons?)\b",),
    "communication": (
        r"\b(?:publicite|pub|affiches?|reseaux|instagram|facebook|tiktok|"
        r"influenceurs?|t shirts?|logo)\b",
    ),
    "choice": (
        r"\b(?:ouvrir|cloturer|choisir|instaurer|prendre|plutot|au lieu|"
        r"finalement|anglais|italien|langue)\b",
    ),
}
ORDER = tuple(VALUE_PATTERNS)

CONFIRM = re.compile(
    r"\b(?:on part|on reste|on prend|on choisit|on valide|on fait|on met|"
    r"on prevoit|on va|on peut le faire|on est d accord|je suis d accord|"
    r"d accord|oui|ok|ca marche|c est bon|finalement|plutot|au lieu|"
    r"c est acte|on annule|on repousse|on abandonne|confirme|valide|acte|"
    r"decide|retenu)\b")
CANCEL = re.compile(r"\b(?:annul|abandonn|repouss|suspend|on oublie)\w*\b")
QUESTION = re.compile(r"[?]")
DECISION_Q = re.compile(
    r"^(?:donc|alors|bon|du coup)?\s*(?:on|nous)\s+"
    r"(?:le|la|les|en|y)?\s*(?:paie|paye|part|reste|prend|fait|met|choisit|"
    r"valide|garde|change|annule|repousse|commence|contacte)\b")
SHORT_ACK = re.compile(r"^(?:oui|ouais|ok|d accord|parfait|ca marche|tres bien)\b")


def fold(text: object) -> str:
    s = unicodedata.normalize("NFD", str(text or "").lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.replace("_", " ").replace("'", " "))


def _first(pattern: str, low: str) -> str | None:
    m = re.search(pattern, low)
    return m.group(0).strip() if m else None


@dataclass
class StateReader:
    """Deterministic point/value/status reader. `state` is read, not mutated."""

    rule: RuleReader = RuleReader()

    def values(self, low: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for point in ORDER:
            for pattern in VALUE_PATTERNS[point]:
                v = _first(pattern, low)
                if v:
                    out[point] = v
                    break
        return out

    def status(self, low: str, state: DecisionState,
               values: dict[str, str]) -> str:
        base = self.rule.status_of(low)
        if CANCEL.search(low):
            return "cancelled"
        if base == "confirmed":
            return "confirmed"
        if CONFIRM.search(low):
            return "confirmed"
        if base in ("proposed", "hypothesis", "conditional"):
            return "proposed"
        if DECISION_Q.search(low) or SHORT_ACK.search(low):
            # a short decision-question is a close, not a new question, when
            # the state has something open for it
            if state.open:
                return "confirmed"
        return "open"

    def continuation(self, text: str) -> bool:
        low = fold(text)
        return bool(CONFIRM.search(low) or DECISION_Q.search(low)
                    or SHORT_ACK.search(low))

    def update(self, text: str, turn_index: int,
               state: DecisionState) -> list[StateUpdate]:
        low = fold(text)
        vals = self.values(low)
        status = self.status(low, state, vals)
        updates: list[StateUpdate] = []
        if vals:
            for point, value in vals.items():
                if status == "open" and point not in state.open:
                    continue
                updates.append(StateUpdate(
                    point=point, value=value, status=status,  # type: ignore[arg-type]
                    turn_index=turn_index, evidence=text,
                    old_value=(state.current(point).value
                              if state.current(point) else None)))
        elif state.open and (DECISION_Q.search(low) or SHORT_ACK.search(low)
                             or CONFIRM.search(low)):
            # “donc on le paie ?” has no explicit value: close the most recent
            # open point with the turn itself as evidence.
            point = state.active if state.active in state.open else list(state.open)[-1]
            updates.append(StateUpdate(
                point=point, value=text.strip(), status="confirmed",
                turn_index=turn_index, evidence=text,
                old_value=state.open[point].value))
        return updates

    def apply(self, text: str, turn_index: int,
              state: DecisionState) -> list[StateUpdate]:
        updates = self.update(text, turn_index, state)
        state.apply_all(updates)
        return updates


__all__ = ["StateReader", "fold", "VALUE_PATTERNS", "ORDER"]
