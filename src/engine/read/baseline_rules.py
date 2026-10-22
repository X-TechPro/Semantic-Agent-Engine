"""Rule-based reader baseline (BUILD_GUIDE 6.2.2) — the floor for the model.

One pass over a unit yields kind / type / status / intent. The lexicon is
French; ordering matters: **negation is tested before any status keyword**, so
"ce n'est pas encore validé" is not read as a confirmation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from engine.schemas import IntentLabel, Kind, Reading, Slots, StatusLabel, TypeLabel, Unit

# --- ordered rule tables ----------------------------------------------------
# (pattern, status, confidence). First match wins within a group.
NEGATION = re.compile(
    r"\b(?:pas|plus|jamais|aucun(?:e)?|ni|sans|nuncl?)\b[^.;]{0,40}?"
    r"\b(?:valid[ée]e?s?|act[ée]e?s?|confirm[ée]e?s?|décidé(?:e)?|retenu(?:e)?)\b"
    r"|\bpas encore\b|\bni l'un ni l'autre\b", re.IGNORECASE)

HYPOTHESIS = [
    r"\bon pourrait peut[-\s]être\b", r"\bça reste une idée\b", r"\bhypoth[èe]se\b",
    r"\bsans s['’]engager\b", r"\bpourquoi pas\b", r"\bpiste\(?\s?\(?non acté",
    r"\bne serait[-\s]ce que\b", r"\ben supposant que\b", r"\bà ne pas acter\b",
    r"\bsi jamais\b", r"\breste une piste\b",
]
CONDITIONAL = [
    r"\bsous réserve\b", r"\bà condition\b", r"\bsauf\b", r"\bd['’]accord si\b",
    r"\bsi .{0,40} alors\b", r"\buniquement si\b", r"\bà charge pour\b",
    r"\bsi ça passe\b", r"\ben cas de\b",
]
CANCEL = [
    r"\bannul[ée]e?s?\b", r"\bon abandonne\b", r"\bc['’]est tomb[ée]?\b",
    r"\bon oublie\b", r"\bsuspendu\b", r"\babrogé\b", r"\bno go\b",
    # 12.3's own examples include "On suspend X". `suspendu` alone missed the
    # live verb form, so a cancellation was read as a bare mention.
    r"\bsuspend\w*\b",
]
CONFIRM = [
    r"\bc['’]est act[ée]e?s?\b", r"\bon valide\b", r"\bc['’]est valid[ée]e?s?\b",
    r"\bc['’]est décidé\b", r"\bvalidé\b", r"\bconfirmé\b", r"\barrêté\b",
    r"\bc['’]est tranché\b", r"\bon retient\b", r"\bc['’]est parti\b",
    r"\brétabli\b", r"\brevient\b", r"\bon remet\b", r"\bressuscite\b",
    r"\bça revient\b",
    # "la date est fixée" is an assertion of a settled value; without it the
    # reader said `mention` and a legitimate supersede was refused.
    r"\best[-\s]fix[ée](?:e|és|es)?\b", r"\bfixée?s?\b",
    # The bare passive "le budget est acté" is the commonest French way to settle
    # something, and `\bc'est acté\b` needs the "c'est" in front of it, so it was
    # invisible to the reader (measured: 260/260 status rows fell to `mention`).
    r"\best[-\s]act[ée]e?s?\b",
]
PROPOSE = [
    r"\bje propose\b", r"\bma proposition\b", r"\bproposition\b", r"\bà confirmer\b",
    r"\bon partirait bien\b", r"\bmoi je dirais\b", r"\bon pourrait\b",
    r"\bonvisager\b", r"\ben discussion\b", r"\bpour l['’]instant\b",
    r"\bfaut voir\b",
    # 12.3's first status example is literally "Le budget est proposé." The list
    # had `proposition` and `je propose` but not the passive, so the most obvious
    # proposal phrasing in French read as `mention`.
    r"\best[-\s]propos[ée](?:e|és|es)?\b", r"\ben attente de validation\b",
]
CHANGE = [
    r"\bfinalement\b", r"\bdésormais\b", r"\bça bouge\b", r"\bon a revu\b",
    r"\bnouvelle valeur\b", r"\bpasse à\b", r"\bon a changé\b", r"\bdu coup on a changé\b",
    r"\brévision\b", r"\bmaintenant c['’]est\b",
]
COMMITMENT = [
    r"\bje m['’]occupe\b", r"\bc['’]est moi qui\b", r"\bje gère\b", r"\bje prends\b",
    r"\béchéance\b", r"\bje livre\b", r"\bje vous ça\b", r"\bje m['’]engage\b",
    r"\bje fais au plus tard\b", r"\bje gère\b",
]
# NB: French elision matters. `s'occupera du budget` is *not* matched by
# `s'occupera de`, and the gap made an assignment read as a plain fact, which
# hid a genuine two-owner ambiguity (measured on the e2e run, scenario 6).
ASSIGN = [
    r"\bs['’]occup(?:era|e|ions|ez|er)\b",          # s'occupera / s'occupe / occupent
    r"\bse charge\b", r"\bqui gère\b", r"\bgère\b",
    r"\brevient à\b", r"\bon part sur \w+", r"\bc['’]est \w+ qui\b",
    r"\bresponsabilité\b", r"\ble pilotage\b", r"\bassigné(?:e)? à\b",
    r"\bdemande à\b", r"\bcharge \w+ de\b",
]
RISK = [r"\brisque\b", r"\bvigilance\b", r"\bfaites attention\b", r"\binquiétude\b",
        r"\bblocage\b", r"\bdépendance\b", r"\balerte\b"]
REPEAT = [r"\bje le redis\b", r"\bje confirme\b", r"\binchangé\b", r"\bje remets une couche\b",
          r"\bencore une fois\b", r"\bcomme dit plus haut\b", r"\bpour éviter toute confusion\b"]
QUESTION_FORM = [r"\best[-\s]ce que\b", r"\bqu[ei] (?:est|sera|coûte)\b", r"\boù en est\b",
                 r"\bstatut\s*\??", r"\bcombien\b", r"\bquand\b", r"\bpour quand\b",
                 r"\bqui s['’]occupe\b", r"\best[-\s]ce qu['’]on\b", r"\bdu coup\b"]
IMPERATIVE = [re.compile(
    r"\b(?:rédige|prépare|écris|envoie|planifie|ajoute|crée|génère|fais|liste|"
    r"compare|résume|programme|cale)\b", re.IGNORECASE)]

#: Imperative payloads aimed at the assistant itself — the brief's injection
#: hard case, and the demo fixture ships one verbatim. The synthetic corpus
#: contains no '@' address and no such phrasing (measured: 0 matches in
#: data/ledger/ledger-42.json), so this fires on directives only, never on
#: meeting speech. Such a turn stays stored as data (nothing is deleted) but
#: is typed "other" and — through the same predicate — never quoted as
#: evidence by the answer or the dossier (DECISIONS D-038: measured 2026-10-22,
#: the hand-tester found it served as answer `Contexte`).
INJECTION_RE = re.compile(
    r"\b(?:ignore|disregard|forget|passer|oublier)\b[^.!?\n]{0,40}"
    r"\b(?:previous|prior|above|earlier|précédent\w*|ci[-\s]dessus)\b[^.!?\n]{0,40}"
    r"\binstructions?\b"
    # exfiltration arm: a bare imperative ("send ... to <address>") opening the
    # turn or chained after a conjunction — "Karim envoie le fichier à
    # client@exemple.fr" is a declarative about Karim and must NOT match, so
    # the verb needs a sentence/conjunction boundary in front of it.
    r"|(?:^|[,;]\s*|\b(?:and|et|puis|alors)\s+)"
    r"(?:send|forward|envoie[z]?|transf\w+|transmets?)\b[^.!?\n]{0,60}"
    r"\b[\w.+-]+@[\w.-]+\.\w+",
    re.IGNORECASE)


def is_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text or ""))

INTENT_RULES: list[tuple[str, re.Pattern]] = [
    ("draft_email", re.compile(r"\b(?:rédige|écris|prépare)\b.{0,40}\b(?:mail|e-?mail|courriel|message)\b",
                               re.IGNORECASE | re.DOTALL)),
    ("create_task", re.compile(r"\b(?:crée|ajoute|ouvre)\b.{0,30}\b(?:une tâche|un ticket|un todo|l'action)\b",
                               re.IGNORECASE)),
    ("schedule_event", re.compile(r"\b(?:planifie|programme|fixe|cale|met)\b.{0,30}\b(?:réunion|point|call|rdv|meeting)\b",
                                  re.IGNORECASE)),
    ("build_dossier", re.compile(r"\b(?:dossier|synthèse|récapitulatif|note de service|compte rendu complet)\b",
                                 re.IGNORECASE)),
    ("compare", re.compile(r"\b(?:compare|comparer|contredit|contradictoire|différence|changé par rapport)\b",
                           re.IGNORECASE)),
    ("answer", re.compile(r"\b(?:quelle décision|qu['’]est[-\s]ce qui|où en|quand|combien|qui s['’]occupe|statut|est[-\s]ce que)\b",
                          re.IGNORECASE)),
]

_PUNCT_Q = re.compile(r"[?¿]")


def _any(patterns: list[str], text: str) -> re.Match | None:
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m
    return None


@dataclass(slots=True)
class RuleHit:
    label: str
    rule: str


class RuleReader:
    """6.2.2 baseline. Deterministic, no model, no vocabulary download."""

    name = "rules"

    def read(self, unit: Unit, ctx=None) -> Reading:
        text = unit.text
        low = text.lower()
        status = self.status_of(low)
        kind, intent = self.kind_and_intent(text, low, status,
                                            from_meeting=unit.source_type == "meeting")
        type_ = self.type_of(low, status, kind, unit)
        slots = Slots()  # the slots stage owns these; reader stays label-only
        conf = 0.75 if kind == "query" else 0.6
        probs = {
            "kind": {kind: conf, ("query" if kind == "info" else "info"): 1 - conf},
            "status": {status: conf},
            "type": {type_: conf},
            "intent": {intent: conf},
        }
        return Reading(unit_id=unit.unit_id, kind=kind, type=type_, status=status,
                       intent=intent, slots=slots, probabilities=probs)

    # --- heads -------------------------------------------------------------
    def status_of(self, low: str) -> StatusLabel:
        if NEGATION.search(low):
            # "pas encore validé" is an explicit NOT-yet: never a confirmation
            return "proposed" if _any(PROPOSE, low) else "mention"
        if _any(HYPOTHESIS, low):
            return "hypothesis"
        if _any(CONDITIONAL, low):
            return "conditional"
        if _any(CANCEL, low):
            return "cancelled"
        if _any(CHANGE, low):
            return "confirmed"          # a change statement asserts the new state
        if _any(REPEAT, low):
            return "confirmed"
        if _any(COMMITMENT, low) or _any(ASSIGN, low):
            return "confirmed"
        if _any(CONFIRM, low):
            return "confirmed"
        if _any(PROPOSE, low):
            return "proposed"
        if _any(RISK, low):
            return "mention"
        if _PUNCT_Q.search(low):
            return "mention"
        return "mention"

    def kind_and_intent(self, text: str, low: str, status: StatusLabel,
                        *, from_meeting: bool = False
                        ) -> tuple[Kind, IntentLabel]:
        intent: IntentLabel = "none"
        for name, pat in INTENT_RULES:
            if pat.search(text):
                intent = name  # type: ignore[assignment]
                break
        is_question = bool(_PUNCT_Q.search(text)) or bool(_any(QUESTION_FORM, low))
        is_imperative = bool(IMPERATIVE[0].search(low))
        if is_question and from_meeting:
            # 7.3: a question asked *inside* a meeting is not a user query. It was
            # being labelled query, and query units are never stored (2.1), so
            # every meeting question silently left the index and no `answers` edge
            # could ever be written — measured: 19 gold answer/answer_of pairs in
            # the dev ledger, 0 answers edges produced by System B.
            return "info", "none"
        if intent != "none" and (is_imperative or is_question or
                                 intent in ("build_dossier", "compare")):
            # 6.2.4: an imperative request is a query even if it asserts nothing
            return "query", intent
        if is_question:
            return "query", (intent if intent != "none" else "answer")
        return "info", intent

    def type_of(self, low: str, status: StatusLabel, kind: Kind,
                unit: Unit) -> TypeLabel:
        if kind == "query":
            return "question"
        if INJECTION_RE.search(low):
            # a directive aimed at the assistant asserts nothing about the
            # project: stored as data, typed "other", never quoted as
            # evidence (answer.py, dossier.py) -- D-038
            return "other"
        if _PUNCT_Q.search(low):
            # a meeting question is `kind=info` (7.3), so the question speech act
            # has to be recovered from punctuation here or it reads as a `fact`
            return "question"
        if _any(COMMITMENT, low):
            return "commitment"
        if _any(ASSIGN, low):
            return "decision"
        if _any(RISK, low):
            return "risk"
        if status == "cancelled" or status == "confirmed":
            if _any(CHANGE, low) or _any(REPEAT, low):
                return "decision"
            return "decision"
        if status == "proposed":
            return "idea"
        if status == "hypothesis":
            return "idea"
        if status == "conditional":
            return "idea"
        if _any(REPEAT, low):
            return "fact"
        return "fact"


READER = RuleReader()

#: Meeting boilerplate. These turns are real speech but assert no fact, and when
#: they are indexed they outrank the decisions they surround — measured on the
#: first end-to-end run, where "Sujet suivant." took rank 1 over the budget
#: decision. Filtering them is a *reader* decision applied before the store, so
#: Baseline A and System B index the same corpus and the §10 comparison stays
#: fair (docs/DECISIONS.md D-007).
CHATTER_RE = re.compile(
    r"\b(?:bonjour|bonsoir|salut|bienvenue|merci d['’]être (?:là|venu|venus)|"
    r"ordre du jour|sujet suivant|on (?:accélère|attaque|enchaîne|enchaine)|"
    r"je (?:récapitule|note ça)|ça marche|c['’]est noté|on (?:clôt|clôture|termine)|"
    r"à (?:jeudi|mardi|mercredi|lundi|vendredi)|bonne (?:journée|fin de journée)|"
    r"quelqu(?:'|’)a\s+(?:a|des)\s+points?|points?\s+d['’]?\s*blockage)\b",
    re.IGNORECASE)
MEETING_LABEL_RE = re.compile(
    r"\b(?:cette réunion|la réunion|cette (?:revue|session)|réunion|session)\b",
    re.IGNORECASE)
_FACT_WORDS = CONFIRM + PROPOSE + CHANGE + CANCEL + HYPOTHESIS + CONDITIONAL + \
    COMMITMENT + ASSIGN + RISK + REPEAT


def is_chatter(unit: Unit) -> bool:
    """True for turns that assert nothing: no slot, no value, conversational glue."""
    s = unit.slots
    if s.subject or s.owner or s.date_value or s.amount or s.due or s.persons \
            or s.action:
        return False
    text = (unit.text or "").strip()
    if not text:
        return True
    if len(text.split()) <= 1:
        return True
    if CHATTER_RE.search(text):
        return True
    # a pure framing sentence ("… porte sur le projet Atlas") states no fact
    lowered = text.lower()
    if MEETING_LABEL_RE.search(text) and not _any(_FACT_WORDS, lowered):
        return True
    return False


def read(unit: Unit, ctx=None) -> Reading:
    """Module entry for the 2.2 contract."""
    return READER.read(unit, ctx)
