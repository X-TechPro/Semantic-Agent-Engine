"""v4 reading: §3 tags with an evidence span, a slot/value and a confidence.

The reader is the v3 rule reader (French lexicon, deterministic) remapped to the
v4 label sets, plus the three things §3 asks for and the previous ingest dropped:

* `kind` is the **speech act** (decision/action/commitment/question/answer/
  fact/risk/idea), taken from the reader's `type`, not its `kind`. In a meeting
  the reader calls a question `kind=info`; v4 must call it `question`.
* `status` uses the v4 set. `mention` becomes `unknown`; hypothesis and
  conditional become `proposed`; `completed` is detected separately because the
  v3 reader has no such label.
* every tag carries the **span it was read from** and the probability of the
  chosen label. A tag without a span is not evidence.

`slot`/`value` are what §5 needs to append a StateEvent. They are read from the
same clause as the status cue, so a state row is always grounded in one span.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from engine.link.thread import detect_topic
from engine.read.baseline_rules import (
    ASSIGN, CANCEL, CHANGE, COMMITMENT, CONFIRM, NEGATION, PROPOSE,
    REPEAT, RISK, RuleReader,
)
from engine.retrieve.entity_index import ACTION
from engine.schemas import Unit

MODEL_VERSION = "v4-rules-1"

# --- status ----------------------------------------------------------------
COMPLETED = [
    r"\bc['’]est fait\b", r"\bc['’]est termin[ée]\b", r"\btermin[ée]e?s?\b",
    r"\blivr[ée]e?s?\b", r"\br[ée]alis[ée]e?s?\b", r"\bcl[oô]tur[ée]e?s?\b",
    r"\bboucl[ée]e?s?\b", r"\bc['’]est r[ée]gl[ée]\b", r"\bd[ée]j[àa] fait\b",
]
#: Ordered: a completed statement is a live state, so it is tested before the
#: generic confirmation cues that would otherwise swallow "c'est fait, validé".
COMPLETED_RE = re.compile("|".join(COMPLETED), re.I)
#: A statement that sets a value now. Checked before the generic reader, which
#: only knows written-register cues ("est validé") and returned `mention` for
#: "j'ai négocié à 16 500" or "il peut faire 4 500".
SETTLE_RE = re.compile(
    r"\b(?:c['’]est act[ée]|c['’]est valid[ée]|on valide|on confirme|"
    r"on prend|on part (?:bien )?sur|on garde|on baisse|on d[ée]cale|"
    r"on coupe|on rouvre|on passe [àa]|on fixe|on retient|"
    r"j['’]ai n[ée]goci[ée]|n[ée]goci[ée]|il peut faire|il propose|"
    r"au plus tard|[àa] verser|on va dire|d['’]accord pour|"
    r"il faut (?:passer|valider|baisser|prendre|d[ée]caler)|normalement)\b"
    r"|^\s*(?:ok|oui|d['’]accord)\b", re.I)
PROPOSE_RE = re.compile(
    r"\b(?:on pourrait|je propose|une option|on partirait|il faudrait|"
    r"on peut envisager|soit\b|et si on)\b", re.I)

STATUS_MAP = {
    "cancelled": "cancelled",
    "confirmed": "confirmed",
    "proposed": "proposed",
    "hypothesis": "proposed",
    "conditional": "proposed",
    "mention": "unknown",
}
KIND_MAP = {
    "decision": "decision", "action": "action", "commitment": "commitment",
    "question": "question", "answer": "answer", "fact": "fact", "risk": "risk",
    "idea": "idea", "other": "fact",
}
INTENT_MAP = {
    "answer": "answer", "compare": "compare", "build_dossier": "dossier",
    "draft_email": "email", "create_task": "task", "schedule_event": "schedule",
    "clarify": "clarify", "none": "answer",
}

# --- entities --------------------------------------------------------------
NUM_WORD = (r"(?:un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|"
            r"douze|treize|quatorze|quinze|seize|vingt|trente|quarante|"
            r"cinquante|soixante|cent|mille)")
CURRENCY = r"(?:m€|k€|millions?|euros?|€|dollars?|livres?)"
AMOUNT = re.compile(
    rf"\b(?:\d[\d\s.,]*)\s*(?:{CURRENCY})"
    rf"|\b{NUM_WORD}(?:[\s-]+(?:{NUM_WORD}))*\s*{CURRENCY}\b", re.I)
CLOCK = re.compile(r"\b\d{1,2}\s*h\s*\d{2}\b", re.I)
#: What a value is about, when the sentence names it: "le 15 août" alone does
#: not, but "la fibre ... le 20 août" does. Without this, every date in a
#: meeting shares one state slot and the lease end overwrites the move date.
POINT_WORDS = (
    "clés", "fibre", "bail", "loyer", "salaire", "acompte", "budget",
    "lancement", "kit presse", "post mortem", "seuil", "panne", "partage",
    "recrutement", "traiteur", "déménagement", "salle", "menu", "capacité",
    "devis", "contrat", "offre", "date", "prix", "tarif", "effectif",
)
OBJECT_AFTER = re.compile(
    r"\b(?:coupe|garde|prend|retient|choisit|d[ée]cale|valide|propose|"
    r"option|p[ée]rim[èe]tre)\s+(?:le |la |les |l['’]|un |une |des )?"
    r"([a-zà-ÿ][\wà-ÿ-]{2,})", re.I)
BARE_AMOUNT = re.compile(
    r"\b\d{1,3}(?:[ \u00a0.]\d{3})+(?:[.,]\d+)?\b"
    r"|\b\d+(?:[.,]\d+)?\s*(?:k|m€|millions?|minutes?|personnes?|m²|%)\b"
    r"|\b\d+\s*(?:minutes?|personnes?|m²)\b", re.I)
MONTHS = ("janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|"
          "septembre|octobre|novembre|decembre|décembre")
DATE = re.compile(
    rf"\b\d{{1,2}}(?:er)?\s+(?:{MONTHS})\b"
    rf"|\b(?:{MONTHS})\s+\d{{4}}\b"
    rf"|\b(?:début|fin|mi)[-\s](?:{MONTHS})\b"
    rf"|\b(?:{MONTHS})\b"
    rf"|\b(?:aujourd['’]hui|demain|hier|apr[eè]s-demain|"
    rf"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche|"
    rf"la semaine prochaine|le mois prochain|fin du mois|"
    rf"ce soir|cette semaine)\b", re.I)
PROJECT = re.compile(r"\bprojet\s+([A-ZÀ-Ý][\wÀ-ÿ-]{1,})", re.UNICODE)
CAP = re.compile(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}\b")
PERSON_HINT = re.compile(r"\b(?:m|mme|mr|monsieur|madame)\.?\s+[A-ZÀ-Ý]", re.I)
OWNER_CUE = re.compile(
    r"\b(?:responsable|en charge|s['’]occupe|se charge|pilote|refer(?:ent)?|"
    r"référent|contact)\b", re.I)
DEADLINE_CUE = re.compile(
    r"\b(?:échéance|echeance|au plus tard|avant le|pour le|d['’]ici|deadline)\b",
    re.I)

# --- slots -----------------------------------------------------------------
#: Ordered: the first cue that matches names the slot. `subject` is the
#: fallback so a statement without a known attribute still has an identity.
SLOT_RULES: list[tuple[str, re.Pattern]] = [
    # owner first: "Marie s'occupe du budget" is an assignment, not a budget
    # value, and the more specific cue has to win.
    ("owner", re.compile(r"\b(?:responsable|owner|pilote|qui s['’]occupe|"
                         r"en charge|r[ée]f[ée]rent|s['’]occup\w*|"
                         r"se charge)\b", re.I)),
    ("budget", re.compile(r"\b(?:budget|montant|co[uû]t|prix|tarif|somme|"
                          r"euros?|€|facture|devis|acompte|verser|pour cent|"
                          r"loyer|salaire|pourcentage)\b|%", re.I)),
    ("date", re.compile(rf"\b(?:date|livraison|[ée]ch[ée]ance|deadline|"
                        rf"rendez-vous|r[ée]union|{MONTHS})\b", re.I)),
    ("venue", re.compile(r"\b(?:lieu|salle|endroit|adresse|local|"
                         r"domicile|si[èe]ge)\b", re.I)),
    ("mode", re.compile(r"\b(?:option|plut[oô]t|soit|mode|fa[çc]on|"
                        r"ouvert|ferm[ée]|cl[oô]tur)\b", re.I)),
    ("quantity", re.compile(r"\b(?:nombre|quantit[ée]|places?|personnes?|"
                            r"invit[ée]s?|participants?)\b", re.I)),
    ("scope", re.compile(r"\b(?:p[ée]rim[èe]tre|contenu|scope|p[ée]rim[ée]tre|"
                         r"inclut|exclut)\b", re.I)),
]
SUBJECT = re.compile(r"\b(?:on|nous)\s+(?:va|allons|part|partons|retient|"
                     r"retient|d[ée]cide|d[ée]cidons)\b", re.I)

CLAUSE_BREAK = re.compile(r"[.;!?\n]|,\s")


@dataclass(frozen=True)
class Tags:
    point: str
    kind: str
    status: str
    intent: str
    entities: tuple[dict, ...]
    owner: str
    deadline: str
    slot: str
    value: str
    evidence: tuple[int, int, str]
    confidence: float
    model_version: str = MODEL_VERSION


def _clause(text: str, pos: int) -> tuple[int, int]:
    start = 0
    for m in CLAUSE_BREAK.finditer(text):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            end = m.start()
            return start, end
    return start, len(text)


def _clause_span(text: str, m: re.Match | None) -> tuple[int, int, str]:
    if m is None:
        return 0, len(text), text
    s, e = _clause(text, m.start())
    clip = text[s:e].strip()
    if not clip:
        s, e, clip = 0, len(text), text
    else:
        s = s + (len(text[s:e]) - len(text[s:e].lstrip()))
        e = s + len(clip)
    return s, e, clip


def extract_entities(text: str) -> tuple[dict, ...]:
    """§3 entity kinds, with spans. Word numbers count (ASR writes "cent")."""
    text = text or ""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add(etype: str, value: str, m: re.Match | None = None) -> None:
        value = value.strip()
        if not value:
            return
        key = (etype, value.lower())
        if key in seen:
            return
        seen.add(key)
        s, e = (m.start(), m.end()) if m else (0, 0)
        out.append({"type": etype, "value": value, "start": s, "end": e})

    for m in AMOUNT.finditer(text):
        add("amount", m.group(0), m)
    for m in BARE_AMOUNT.finditer(text):
        add("amount", m.group(0), m)
    for m in DATE.finditer(text):
        add("date", m.group(0), m)
    for m in CLOCK.finditer(text):
        add("date", m.group(0), m)
    for m in PROJECT.finditer(text):
        add("project", m.group(1), m)
    for m in ACTION.finditer(text):
        add("action", m.group(0), m)
    projects = {str(e["value"]).lower() for e in out if e["type"] == "project"}
    months = {"janvier", "fevrier", "février", "mars", "avril", "mai", "juin",
              "juillet", "aout", "août", "septembre", "octobre", "novembre",
              "decembre", "décembre"}
    for m in CAP.finditer(text):
        val = m.group(0)
        low = val.lower()
        if low in COMMON_FR or low in months or low in projects:
            continue
        add("person" if _looks_like_person(val) else "object", val, m)
    topic = detect_topic(text)
    if topic:
        add("event", topic)
    return tuple(out)


_PERSON_STOP = {"projet", "budget", "date", "salle", "lieu", "option", "mode",
                "semaine", "mois", "jours", "euros", "réunion", "reunion",
                "point", "ordre", "sujet", "retour", "également"}

#: French words that are capitalised at a sentence start without being names.
COMMON_FR = {"le", "la", "les", "un", "une", "des", "du", "de", "ce", "cette",
             "ces", "on", "il", "elle", "ils", "elles", "nous", "vous", "je",
             "tu", "donc", "alors", "ensuite", "finalement", "apres", "après",
             "avant", "pour", "dans", "sur", "avec", "sans", "quand", "si",
             "oui", "non", "ok", "voila", "voilà", "ben", "euh", "mais", "et",
             "ou", "car", "comme", "tout", "tous", "toute", "toutes", "c'est",
             "aujourd", "demain", "hier", "lundi", "mardi", "mercredi",
             "jeudi", "vendredi", "samedi", "dimanche", "monsieur", "madame"}


def _looks_like_person(token: str) -> bool:
    return token.lower() not in _PERSON_STOP and len(token) >= 3


def _entity(ents: tuple[dict, ...], etype: str) -> str:
    for e in ents:
        if e["type"] == etype:
            return str(e["value"])
    return ""


#: What the *question* asks for. Without this, "Quand le poste est-il
#: ouvert ?" matched the `mode` cue on "ouvert" and could never match a date.
QUERY_SLOTS: list[tuple[str, re.Pattern]] = [
    # "combien de temps" is a duration, not an amount
    ("date", re.compile(r"\bquand\b|\bquelle date\b|\b[àa] quelle heure\b|"
                        r"\bcombien de temps\b|\bquelle dur[ée]e\b", re.I)),
    ("budget", re.compile(r"\bcombien\b|\bquel(?:le)?\s+(?:prix|budget|"
                          r"montant|co[uû]t|loyer|salaire|tarif|acompte)\b",
                          re.I)),
    ("owner", re.compile(r"\bqui\b|\bquelle personne\b", re.I)),
    # bare "où" is idiomatic ("où en est-on"), so it needs a place noun
    ("venue", re.compile(r"\bquel(?:le)?\s+(?:lieu|salle|adresse|bureau)\b",
                          re.I)),
]


def _entity_of(ents: tuple[dict, ...], etype: str, last: bool = False):
    found = [e for e in ents if e.get("type") == etype]
    if not found:
        return None
    return found[-1] if last else found[0]


def slot_of(text: str, ents: tuple[dict, ...]) -> str:
    low = (text or "").strip()
    if low.endswith("?") or low.lower().startswith(
            ("quand", "combien", "qui", "où", "quel", "quelle")):
        for slot, pattern in QUERY_SLOTS:
            if pattern.search(low):
                return slot
    for slot, pattern in SLOT_RULES:
        if pattern.search(text or ""):
            return slot
    # no cue: the value's own type names the slot, so two amounts on different
    # points do not collide under `subject`
    if _entity_of(ents, "amount", last=True):
        return "budget"
    if _entity_of(ents, "date", last=True):
        return "date"
    if _entity_of(ents, "person", last=True):
        return "owner"
    return "subject"


def point_of(text: str, ents: tuple[dict, ...] = ()) -> str:
    """The object a value is about, or "" when the sentence does not name it."""
    low = (text or "").lower()
    m = OBJECT_AFTER.search(low)
    if m:
        return m.group(1)
    for word in POINT_WORDS:
        if word in low:
            return word
    return ""


def value_of(text: str, slot: str, ents: tuple[dict, ...]) -> str:
    """The value for a slot: the LAST entity of its type, else the clause.

    Last, not first: "55 000 … mais 52 000" is a change to 52 000, and storing
    the whole sentence made every downstream comparison a comparison of prose.
    """
    if slot == "budget":
        e = _entity_of(ents, "amount", last=True)
        return str(e["value"]) if e else text.strip()
    if slot == "date":
        e = _entity_of(ents, "date", last=True)
        return str(e["value"]) if e else text.strip()
    if slot == "owner":
        e = _entity_of(ents, "person", last=True)
        return str(e["value"]) if e else text.strip()
    if slot == "venue":
        # "la salle peut accueillir 100 personnes": capacity is the amount
        amt = _entity_of(ents, "amount", last=True)
        if amt:
            return str(amt["value"])
        for e in ents:
            if e["type"] == "object":
                return str(e["value"])
        return text.strip()
    if slot == "quantity":
        e = _entity_of(ents, "amount", last=True)
        return str(e["value"]) if e else text.strip()
    # a decision on an object has no amount or date: "on coupe le partage"
    m = OBJECT_AFTER.search(text or "")
    if m and slot in ("subject", "mode", "scope"):
        return m.group(1)
    return text.strip()


def _status(text: str, reading) -> tuple[str, float]:
    low = (text or "").lower()
    probs = (reading.probabilities or {}).get("status", {})
    if PROPOSE_RE.search(low):
        return "proposed", 0.6
    if SETTLE_RE.search(low):
        return "confirmed", 0.8
    if COMPLETED_RE.search(low) and not NEGATION.search(low):
        return "completed", float(probs.get("confirmed", 0.6))
    mapped = STATUS_MAP.get(str(reading.status), "unknown")
    conf = float(probs.get(str(reading.status), 0.0)) or 0.0
    if mapped == "unknown":
        conf = 0.0
    return mapped, conf


def read_tags(text: str, speaker: str = "", source_date: date | None = None,
              source_type: str = "meeting") -> Tags:
    """Read one unit into §3 tags. Never raises on odd input."""
    text = text or ""
    unit = Unit(
        unit_id="v4-read", source_id="v4", source_type=source_type,
        source_date=source_date or date.today(),
        ingested_at=datetime.utcnow(), text=text, speaker=speaker or None)
    try:
        reading = RuleReader().read(unit)
    except Exception:                                  # pragma: no cover
        return Tags("fact", "unknown", "answer", extract_entities(text), "",
                    "", "subject", text.strip(), (0, len(text), text), 0.0)

    status, sconf = _status(text, reading)
    kind = KIND_MAP.get(str(reading.type), "fact")
    intent = INTENT_MAP.get(str(reading.intent), "answer")
    ents = extract_entities(text)

    # evidence = the clause of the status cue that fired, in rule order
    cue = None
    for table in (COMPLETED, CANCEL, CONFIRM, ASSIGN, COMMITMENT, CHANGE,
                  REPEAT, PROPOSE, RISK):
        pats = table if isinstance(table, list) else [table]
        for pat in pats:
            m = re.search(pat if isinstance(pat, str) else pat, text, re.I)
            if m:
                cue = m
                break
        if cue:
            break
    ev_start, ev_end, ev_text = _clause_span(text, cue)

    owner = ""
    if OWNER_CUE.search(text) or ASSIGN or COMMITMENT:
        owner = _entity(ents, "person")
    deadline = _entity(ents, "date") if DEADLINE_CUE.search(text) else ""

    slot = slot_of(text, ents)
    value = value_of(text, slot, ents)
    conf = round(min(1.0, max(0.0, sconf)), 4)
    if kind == "fact" and str(reading.type) == "other":
        conf = min(conf, 0.2)
    return Tags(point_of(text, ents), kind, status, intent, ents, owner,
                deadline, slot, value, (ev_start, ev_end, ev_text), conf)


__all__ = ["Tags", "read_tags", "extract_entities", "slot_of", "value_of",
           "point_of",
           "MODEL_VERSION", "KIND_MAP", "STATUS_MAP", "INTENT_MAP",
           "COMPLETED_RE"]
