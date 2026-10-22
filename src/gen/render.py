"""Ledger -> French meeting dialogue (BUILD_GUIDE 7.1 step 2).

Two renderers ship here:

* ``template`` (default) — deterministic phrase tables. It is the *rule-based
  "ugly" renderer* the guide asks for (7.1.3), and it is what the tests and the
  baseline runs use, because it needs no LLM and cannot hallucinate a fact.
* ``llm`` — behind ``--renderer llm``, only if an LLM backend is configured
  (never used by default; see docs/DECISIONS.md D-006).

Every rendered sentence carries the `event_id` it expresses, so after
segmentation a unit inherits exact gold `type/status/slots/relations`.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Any

from engine.schemas import Source, Turn, utcnow
from engine.seeding import subseed
from gen.ledger import ATTRIBUTES, Event, Ledger, READER_GOLD, reader_gold

# --- phrase tables ----------------------------------------------------------
# style A = neutral/professional, style B = oral/disconnected, C = terse
PROPOSED = {
    "A": ["Je propose qu'on retienne {value} pour {np}.",
          "Ma proposition sur {np}, c'est {value}."],
    "B": ["en fait pour {np} je partirais bien sur {value}",
          "nous sur {np} moi je dirais {value}, enfin à confirmer"],
    "C": ["{np} : proposition {value}."],
}
CONFIRMED = {
    "A": ["C'est acté : pour {np} on retient {value}.",
          "On valide {value} pour {np}."],
    "B": ["bon {np} c'est validé on part sur {value}",
          "ok donc pour {np} ce sera {value}, c'est décidé"],
    "C": ["{np} : validé, {value}."],
}
CHANGED = {
    "A": ["Finalement, {np} passe à {value}.",
          "Nous avons revu {np} : la nouvelle valeur est {value}."],
    "B": ["du coup on a changé {np}, maintenant c'est {value}",
          "attention {np} ça bouge, on est sur {value} désormais"],
    "C": ["{np} : révision, {value}."],
}
CANCELLED = {
    "A": ["{np_cap} est annulé, on ne le fait pas.",
          "On abandonne {np}."],
    "B": ["non en fait {np} c'est tombé, c'est annulé",
          "{np} ? on oublie, c'est annulé"],
    "C": ["{np} : annulé."],
}
REINSTATED = {
    "A": ["{np_cap} revient : on remet {value}.",
          "Finalement on rétablit {np} à {value}."],
    "B": ["du coup {np} ça revient, on remet {value}",
          "on ressuscite {np}, toujours {value}"],
    "C": ["{np} : rétabli, {value}."],
}
HYPOTHESIS = {
    "A": ["On pourrait peut-être revoir {np} vers {value}, sans s'engager.",
          "Hypothèse de travail : {np} à {value}, à ne pas acter pour l'instant."],
    "B": ["en gros on pourrait peut-être faire {value} pour {np}, mais c'est pas gagné",
          "ça reste une idée mais {np} à {value}, pourquoi pas"],
    "C": ["{np} : piste {value} (non acté)."],
}
CONDITIONAL = {
    "A": ["D'accord pour {value} sur {np}, sous réserve que la ligne budgétaire passe.",
          "{np} à {value} seulement si la direction confirme l'enveloppe."],
    "B": ["ok pour {value} sur {np} mais seulement si ça passe en comit",
          "{np} {value} pourquoi pas à condition d'avoir les moyens"],
    "C": ["{np} : {value} sous réserve."],
}
COMMITMENT = {
    "A": ["Je m'occupe de {action} et je livre pour le {due}.",
          "C'est moi qui prends {action}, échéance {due}."],
    "B": ["je gère {action}, je vous ça pour {due}",
          "{action} c'est pour moi, je fais au plus tard {due}"],
    "C": ["{action} : moi, pour {due}."],
}
ASSIGNMENT = {
    "A": ["{person} s'occupera de {np}.",
          "La responsabilité de {np} revient à {person}."],
    "B": ["{np} c'est {person} qui gère",
          "pour {np} on part sur {person}"],
    "C": ["{np} : {person}."],
}
RISK = {
    "A": ["Je souligne un risque : {value}.",
          "Point de vigilance, le risque est {value}."],
    "B": ["faites attention quand même, {value}",
          "mon inquiétude c'est {value}"],
    "C": ["risque : {value}."],
}
QUESTION = {
    "A": ["Où en est {np} exactement ?",
          "On a tranché pour {np} ?"],
    "B": ["du coup {np} on en est où",
          "{np} ça dit quoi ?"],
    "C": ["{np} : statut ?"],
}
ANSWER = {
    "A": ["Pour {np}, on est à {value}.",
          "{np_cap} : la réponse est {value}."],
    "B": ["{np} alors on a {value}",
          "pour {np} c'est {value}"],
    "C": ["{np} : {value}."],
}
MENTION = {
    "A": ["Pour rappel, sur le dossier {proj}, {np} est à {value}.",
          "Je note au passage que {np} de {proj} vaut {value}."],
    "B": ["ah et pour {proj}, {np} c'est {value}",
          "petit point sur {proj} : {np} à {value}"],
    "C": ["{proj} — {np} : {value}."],
}
REPEATED = {
    "A": ["Je le redit : {np}, c'est {value}.",
          "Pour éviter toute confusion, {np} reste {value}."],
    "B": ["je remets une couche, {np} = {value}",
          "{np} toujours {value}, je confirme"],
    "C": ["{np} : inchangé, {value}."],
}
# question about a *date* is phrased like the real user queries
QUESTION_DATE = {
    "A": ["Du coup, {np}, c'est prévu pour quand ?",
          "On est calés sur quelle date pour {np} ?"],
    "B": ["{np} c'est quand déjà", "niveau date {np} on dit quoi"],
    "C": ["{np} : date ?"],
}
TABLES = {
    "proposed": PROPOSED, "confirmed": CONFIRMED, "changed": CHANGED,
    "cancelled": CANCELLED, "reinstated": REINSTATED, "hypothesis": HYPOTHESIS,
    "commitment": COMMITMENT, "assignment": ASSIGNMENT, "risk": RISK,
    "question": QUESTION, "answer": ANSWER, "mention": MENTION,
    "repeated": REPEATED, "conditional": CONDITIONAL,
}

FILLERS = [
    "Bonjour à toutes et à tous, merci d'être là.",
    "On attaque l'ordre du jour.",
    "Sujet suivant.",
    "Ça marche, on note.",
    "Je vous renvoie le compte rendu après.",
    "On est à dix minutes du créneau, on accélère.",
    "Quelqu'un a des points d'blockage ?",
    "Très bien, je récapitule en fin de réunion.",
]
OPENERS = [
    "Bienvenue sur la réunion {title}.",
    "Bonjour, cette réunion porte sur le projet {proj}.",
    "On se retrouve pour la revue du dossier {proj}.",
]


def _np(event: Event, attribute_of: dict[str, str]) -> str:
    """French noun phrase for the event's subject (``le budget`` etc.)."""
    attr = _attribute_of(event, attribute_of)
    if attr is None:
        return "le sujet"
    if event.subject_key and event.subject_key.endswith(".owner"):
        return f"le pilotage de {ATTRIBUTES[attr]['np']}"
    return ATTRIBUTES[attr]["np"]


def _attribute_of(event: Event, attribute_of: dict[str, str]) -> str | None:
    if event.role:
        return event.role
    key = event.subject_key or ""
    for attr in attribute_of:
        if key.endswith("." + attr) or key.endswith("." + attr + ".owner"):
            return attr
    return None


def render_sentence(event: Event, attribute_of: dict[str, str], style: str,
                    rng: random.Random, speaker: str | None = None) -> str:
    kind = event.kind
    if kind == "proposed" and event.notes == "conditional":
        kind = "conditional"
    if kind == "question" and attribute_of.get(_attribute_of(event, attribute_of) or "",
                                               "date") == "date":
        table = QUESTION_DATE
    else:
        table = TABLES[kind]
    tmpl = rng.choice(table[style])
    attr = _attribute_of(event, attribute_of)
    np_ = _np(event, attribute_of)
    value = event.value or ""
    if kind == "assignment":
        value = event.person or ""
    fmt = {
        "np": np_.strip(), "np_cap": np_.strip().capitalize(),
        "value": value, "person": event.person or speaker or "",
        "action": event.action or "le point",
        "due": event.due.isoformat() if event.due else "bientôt",
        "proj": event.project,
        "title": f"{event.project}",
    }
    line = tmpl.format(**fmt)
    if kind == "commitment" and speaker and "je" in line.lower()[:6]:
        line = line  # first-person commitment: the speaker IS the owner
    return line


def render_meeting(ledger: Ledger, meeting_id: str, styles: str = "ABC",
                   filler_rate: float = 0.25, seed: int = 42,
                   relative_dates: bool = False) -> Source:
    """One ledger meeting -> one `Source` of speaker turns with gold pointers."""
    meeting = ledger.meeting(meeting_id)
    attribute_of: dict[str, str] = {}
    for p in ledger.projects:
        for a in p["attributes"]:
            attribute_of[a] = p["attribute_meta"][a]
    rng = random.Random(subseed(seed, "render", meeting_id, styles))
    style = rng.choice(styles)
    turns: list[Turn] = []
    events = sorted((ledger.event(eid) for eid in meeting.event_ids),
                    key=lambda e: (e.on, e.event_id))
    idx = 0
    if rng.random() < 0.8:
        turns.append(Turn(speaker=rng.choice(meeting.participants),
                          text=rng.choice(OPENERS).format(proj=meeting.project,
                                                           title=meeting.title),
                          project=meeting.project, turn_index=idx))
        idx += 1
    for ev in events:
        speaker = ev.person or rng.choice(meeting.participants)
        if ev.kind == "commitment":
            speaker = ev.owner or speaker
        text = render_sentence(ev, attribute_of, style, rng, speaker)
        # the unit is about the EVENT's project, which is not the meeting's
        # project for a distractor turn; stamping the meeting's project onto it
        # would make cross-project collisions look same-scope (6.8.3 G0)
        turns.append(Turn(speaker=speaker, text=text, turn_index=idx,
                          project=ev.project, gold_event_ids=[ev.event_id]))
        idx += 1
        if rng.random() < filler_rate:
            turns.append(Turn(speaker=rng.choice(meeting.participants),
                              text=rng.choice(FILLERS), project=meeting.project,
                              turn_index=idx))
            idx += 1
    if rng.random() < 0.5:
        turns.append(Turn(speaker=rng.choice(meeting.participants),
                          text=rng.choice(FILLERS), project=meeting.project,
                          turn_index=idx))
    return Source(
        source_id=meeting.meeting_id, source_type="meeting", source_date=meeting.on,
        title=meeting.title, project=meeting.project,
        default_speaker=meeting.participants[0], turns=turns,
        gold_event_ids=list(meeting.event_ids),
    )


def render_ledger(ledger: Ledger, styles: str = "ABC", seed: int = 42,
                  relative_dates: bool = False) -> list[Source]:
    return [render_meeting(ledger, m.meeting_id, styles=styles, seed=seed,
                           relative_dates=relative_dates) for m in ledger.meetings]


def gold_index(ledger: Ledger) -> dict[str, dict[str, Any]]:
    """event_id -> everything a downstream stage is scored against."""
    out: dict[str, dict[str, Any]] = {}
    attribute_of = {}
    for p in ledger.projects:
        for a in p["attributes"]:
            attribute_of[a] = p["attribute_meta"][a]
    for e in ledger.events:
        out[e.event_id] = {
            "project": e.project, "kind": e.kind, "on": e.on.isoformat(),
            "subject_key": e.subject_key, "value": e.value, "person": e.person,
            "owner": e.owner, "due": e.due.isoformat() if e.due else None,
            "status": _READER_STATUS(e), "type": _READER_TYPE(e),
            "supersedes": e.supersedes, "answer_of": e.answer_of,
            "commit_ref": e.commit_ref, "distractor": e.distractor,
        }
    return out


#: Views onto the ledger's single gold table, kept as names because the QA
#: generator imports `_STATUS_MAP`. These were literal dicts, and literals drift:
#: `reader_status` carried a `conditional` refinement the literal did not, so the
#: two "sources of truth" disagreed on 9 dev events. Deriving makes that drift
#: structurally impossible.
_TYPE_MAP = {k: v[1] for k, v in READER_GOLD.items()}
_STATUS_MAP = {k: v[2] for k, v in READER_GOLD.items()}


def reader_status(e: Event) -> str:
    """GOLD status for an event — delegates to `ledger.reader_gold`.

    Public because the evaluator must read labels from here. The `conditional`
    refinement (`proposed` with notes="conditional") now lives in the one table:
    reading `_STATUS_MAP[e.kind]` directly used to lose it, which made a correct
    reader score `conditional` F1 = 0.00 (measured on the first stage run).
    """
    return reader_gold(e)[2]


def reader_type(e: Event) -> str:
    """GOLD type for an event — delegates to `ledger.reader_gold`."""
    return reader_gold(e)[1]


# kept as private aliases: existing call sites (gold_index, qa generator)
_READER_STATUS = reader_status
_READER_TYPE = reader_type
