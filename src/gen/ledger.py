"""Ledger generator v0 — BUILD_GUIDE 7.1 step 1 (pure Python, seeded).

The ledger is the reason every label in this project is exact: we author
*events* (a subject had value V, in state S, on date D, said by P) and derive
gold units, relations, slots, QA and hard negatives from them instead of
guessing them from text.

Gold relations are produced here as a pure function of the ledger
(``gold_relation``), which is what the linker is trained/evaluated against.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal

from engine.seeding import subseed

# --- vocabulary -------------------------------------------------------------
FIRST_NAMES = [
    "Marie", "Karim", "Sophie", "Julien", "Nadia", "Thomas", "Inès", "Lucas",
    "Fatima", "Antoine", "Camille", "Yanis", "Élodie", "Rachid", "Léa",
    "Hugo", "Amélie", "Nicolas", "Sarah", "Maxime", "Chloé", "Idriss",
    "Julie", "Paul", "Manon", "Tarek", "Alice", "Bruno", "Nina", "Olivier",
]

PROJECT_CODES = [
    "Atlas", "Boréal", "Cigale", "Delta", "Étincelle", "Fjord", "Galet",
    "Hélios", "Iris", "Jade", "Kir", "Lagune", "Mistral", "Némo", "Onyx",
    "Palme", "Quark", "Rivage", "Sirocco", "Topaze", "Ulysse", "Vega",
    "Waipi", "Xylo", "Yucca", "Zéphyr",
]

ROLES = ["budget", "technique", "produit", "client", "juridique", "recrutement",
         "communication", "qualité"]

# attribute key -> (value generator name, French noun phrase used in speech)
ATTRIBUTES: dict[str, dict[str, str]] = {
    "launch_date": {"kind": "date", "np": "la date de lancement",
                    "verb": "Le lancement est prévu"},
    "budget": {"kind": "amount", "np": "le budget",
               "verb": "Le budget est de"},
    "vendor": {"kind": "vendor", "np": "le prestataire",
               "verb": "Le prestataire retenu est"},
    "scope": {"kind": "scope", "np": "le périmètre",
              "verb": "Le périmètre porte sur"},
    "deadline_report": {"kind": "date", "np": "la deadline du rapport",
                        "verb": "Le rapport doit partir"},
    "stack": {"kind": "stack", "np": "la stack technique",
              "verb": "La stack retenue est"},
    "pricing": {"kind": "amount", "np": "le tarif",
                "verb": "Le tarif est de"},
    "hiring": {"kind": "hiring", "np": "le recrutement",
               "verb": "On recrute"},
    "risk_compliance": {"kind": "risk", "np": "le risque de conformité",
                        "verb": "Le risque de conformité"},
    "env_strategy": {"kind": "stack", "np": "la stratégie d'environnements",
                     "verb": "La stratégie d'environnements"},
}

VENDORS = ["Nova Cloud", "Studio Meridian", "Atelier Kappa", "DataOps Plus",
           "Agence Véga", "Solutions Lumen"]
STACKS = ["PostgreSQL et FastAPI", "Python et DuckDB", "Node et Postgres",
          ".NET et Azure", "Rust et ClickHouse"]
SCOPES = ["les trois premiers clients pilotes", "le périmètre complet",
          "la phase pilote uniquement", "les marchés francophones",
          "le portail interne"]
HIRING = ["un ingénieur données", "un chef de produit", "deux développeurs",
          "un alternant", "un expert conformité"]

EventKind = Literal[
    "proposed", "confirmed", "changed", "cancelled", "reinstated", "hypothesis",
    "commitment", "assignment", "risk", "question", "answer", "mention",
    "repeated",
]

# events whose value, if later re-asserted, supersedes the earlier holder.
# `assignment` belongs: 6.12 says the most recent explicit assignment wins, so a
# re-assignment replaces the previous one rather than contradicting it.
SUPERSEDING_KINDS: frozenset[str] = frozenset(
    {"confirmed", "changed", "cancelled", "reinstated", "assignment"})


@dataclass(slots=True)
class Event:
    event_id: str
    project: str
    meeting_id: str
    on: date
    kind: EventKind
    subject_key: str | None = None
    value: str | None = None
    person: str | None = None
    owner: str | None = None
    role: str | None = None
    due: date | None = None
    action: str | None = None
    answer_of: str | None = None      # question event_id for kind == "answer"
    commit_ref: str | None = None     # decision event_id a commitment stems from
    supersedes: str | None = None     # event_id this event replaces (gold)
    distractor: bool = False          # cross-project vocabulary overlap
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["on"] = self.on.isoformat()
        d["due"] = self.due.isoformat() if self.due else None
        return d


@dataclass(slots=True)
class Meeting:
    meeting_id: str
    project: str
    on: date
    title: str
    participants: list[str]
    event_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["on"] = self.on.isoformat()
        return d


class Ledger:
    """All events + meetings for one project set, with gold derivation."""

    def __init__(self, projects: list[dict[str, Any]],
                 meetings: list[Meeting], events: list[Event],
                 seed: int, meta: dict[str, Any] | None = None):
        self.projects = projects
        self.meetings = meetings
        self.events = events
        self.seed = seed
        self.meta = meta or {}
        self._by_id = {e.event_id: e for e in events}
        self._mby = {m.meeting_id: m for m in meetings}

    # --- accessors ---------------------------------------------------------
    def event(self, event_id: str) -> Event:
        return self._by_id[event_id]

    def meeting(self, meeting_id: str) -> Meeting:
        return self._mby[meeting_id]

    def events_by_project(self, project: str) -> list[Event]:
        return [e for e in self.events if e.project == project]

    # --- gold relation (6.8 label space, derived not guessed) --------------
    def gold_relation(self, new_id: str, old_id: str) -> str:
        """Label for the ordered pair (new, old) — what the linker must learn.

        The `supersedes` pointer authored by the generator is the ground truth:
        it encodes "same slot + later + assertive state change", which is
        exactly the 6.8 rule. Everything else is derived from the pair's
        attributes.
        """
        new, old = self._by_id[new_id], self._by_id[old_id]
        if new.project != old.project:
            return "unrelated"
        if new.kind == "answer" and new.answer_of == old.event_id:
            return "answers"
        if old.kind == "answer" and old.answer_of == new.event_id:
            return "answers"
        if new.kind == "commitment" and new.commit_ref == old.event_id:
            return "creates_action"
        if old.kind == "commitment" and old.commit_ref == new.event_id:
            return "creates_action"
        # A question asserts nothing, so it can neither support nor contradict the
        # statement it happens to be about: it is a retrieval target, not a claim.
        # Placed after the `answers` branches above (which are exactly the
        # question/answer pairs that DO carry a relation) and before the value
        # comparison below, because that comparison reads a question's `value` --
        # which is just the subject label -- against a real value and reports
        # `contradicts` for every mismatch. Measured on the dev split: 75 of 154
        # labelled edges came out gold=contradicts / predicted=supports, and in the
        # sampled cases the linker was the more correct of the two. A reference
        # label that is wrong in this direction turns a working linker into a
        # 0.15 macro-F1, so the guard belongs in the gold, not in the metric.
        if new.kind == "question" or old.kind == "question":
            return "unrelated"
        if old.supersedes == new.event_id or new.supersedes == old.event_id:
            # pointer wins: a state change always replaces its predecessor,
            # even when the surface value is unchanged (cancel / reinstate)
            return "supersedes"
        if new.kind == "repeated" or (new.subject_key and new.subject_key == old.subject_key
                                      and new.value == old.value and new.kind == old.kind):
            return "supports"
        if new.subject_key and new.subject_key == old.subject_key:
            if new.kind in ("proposed", "hypothesis"):
                # a proposal/hypothesis cannot replace a confirmed statement,
                # but it does conflict with it (6.8.3 downgrade path)
                return "contradicts" if old.kind == "confirmed" and new.value != old.value \
                    else "supports"
            if new.kind in SUPERSEDING_KINDS:
                # assertive + same slot + not pointed at: same value => restatement
                return "supports" if new.value == old.value else "contradicts"
            return "supports" if new.value == old.value else "contradicts"
        if new.subject_key or old.subject_key:
            # same topic (project), different attribute -> unrelated (6.8.3)
            return "unrelated"
        if new.value and new.value == old.value and new.kind == old.kind:
            return "supports"
        return "unrelated"

    def gold_supersede_edges(self) -> list[dict[str, str]]:
        return [{"new": e.event_id, "old": e.supersedes, "on": e.on.isoformat()}
                for e in self.events if e.supersedes]

    def current_state(self, as_of: date | None = None) -> dict[str, dict[str, Any]]:
        """subject_key -> the in-force value/status at `as_of` (bi-temporal)."""
        out: dict[str, dict[str, Any]] = {}
        for e in sorted(self.events, key=lambda x: (x.on, x.event_id)):
            if not e.subject_key:
                continue
            if as_of and e.on > as_of:
                break
            if e.kind in ("question", "answer", "risk", "assignment",
                          "commitment", "mention"):
                continue
            out[e.subject_key] = {"project": e.project, "value": e.value,
                                  "status": _STATUS_OF[e.kind], "event_id": e.event_id,
                                  "on": e.on.isoformat()}
        return out

    # --- persistence -------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"meta": {**self.meta, "seed": self.seed},
                "projects": self.projects,
                "meetings": [m.to_dict() for m in self.meetings],
                "events": [e.to_dict() for e in self.events]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Ledger":
        events = []
        for e in d["events"]:
            e = dict(e)
            e["on"] = date.fromisoformat(e["on"])
            e["due"] = date.fromisoformat(e["due"]) if e.get("due") else None
            events.append(Event(**e))
        meetings = []
        for m in d["meetings"]:
            m = dict(m)
            m["on"] = date.fromisoformat(m["on"])
            meetings.append(Meeting(**m))
        return cls(d["projects"], meetings, events, d["meta"].get("seed", 42),
                   d["meta"])

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
                        encoding="utf-8")
        return path

    def audit_lines(self) -> list[str]:
        """Markdown sample for data/ledger/audit.md (7.1 step 5)."""
        lines = [f"# Ledger audit — seed {self.seed}", "",
                 f"- projects: {len(self.projects)}",
                 f"- meetings: {len(self.meetings)}",
                 f"- events: {len(self.events)}",
                 f"- gold supersede edges: {len(self.gold_supersede_edges())}", ""]
        for e in self.events[:20]:
            lines.append(f"- `{e.event_id}` {e.project} {e.on} **{e.kind}** "
                         f"`{e.subject_key}` = {e.value} "
                         f"{('supersedes ' + e.supersedes) if e.supersedes else ''}")
        return lines


#: EventKind -> reader labels, the ONE authoritative table. `current_state`,
#: `gen.render.reader_status/reader_type` and `engine.read.dataset` all read it.
#: Only the generator can authorise these labels: it wrote the event, so they are
#: gold by construction (9.2), not a second guess of the same text. The rules
#: reader derives `type` from surface patterns, which is exactly the thing under
#: test — scoring a trained reader against the rules' own output would only ever
#: measure imitation.
#:
#: `kind` is "info" for every meeting turn, including meeting *questions*: 7.3
#: distinguishes a question asked inside a meeting from a user query, and only a
#: user query may come from the client. `answer` is the same — it answers a
#: meeting question, not the retrieval caller.
#:
#: `repeated` is deliberately **not** a decision. 6.8's gold relation for a
#: repeated event is `supports` (see `Ledger.gold_relation`), and a reader that
#: called it a decision would hand the linker a superseding speech act for
#: something the generator says merely restates the in-force value.
READER_GOLD: dict[str, tuple[str, str, str]] = {
    "proposed": ("info", "idea", "proposed"),
    "hypothesis": ("info", "idea", "hypothesis"),
    "confirmed": ("info", "decision", "confirmed"),
    "changed": ("info", "decision", "confirmed"),
    "reinstated": ("info", "decision", "confirmed"),
    "cancelled": ("info", "decision", "cancelled"),
    "assignment": ("info", "decision", "confirmed"),
    "commitment": ("info", "commitment", "confirmed"),
    "risk": ("info", "risk", "confirmed"),
    "question": ("info", "question", "mention"),
    "answer": ("info", "answer", "confirmed"),
    "mention": ("info", "fact", "mention"),
    "repeated": ("info", "fact", "confirmed"),
}

#: EventKind -> status, for `Ledger.current_state`. DERIVED from READER_GOLD.
#: This used to be its own literal table, and literal tables drift: the
#: `conditional` refinement lived only in `render.reader_status`, so two "sources
#: of truth" disagreed on 9 dev events. Verified behaviour-identical for all 13
#: kinds before deleting the literal.
_STATUS_OF: dict[str, str] = {k: v[2] for k, v in READER_GOLD.items()}


def reader_gold(event: "Event | str") -> tuple[str, str, str]:
    """(kind, type, status) gold for one ledger event — **the only such table**.

    `gen.render.reader_status` / `reader_type` and `engine.read.dataset` all go
    through here. Two copies of this mapping existed before (one keyed on kind,
    one handling the conditional note) and they already disagreed: the second
    one's conditional refinement was missing here, so 9 conditional events were
    trained as `proposed`. A test now asserts the delegation stays single-valued.

    Accepting an EventKind string as well as an Event is deliberate: a caller
    that only has the kind cannot get the conditional refinement, so it gets the
    base mapping — which is why callers with an Event should pass the Event.
    """
    kind = event if isinstance(event, str) else event.kind
    try:
        gold = READER_GOLD[kind]
    except KeyError:
        raise KeyError(
            f"EventKind {kind!r} has no reader gold label; add it to "
            f"READER_GOLD in gen/ledger.py") from None
    if not isinstance(event, str) and event.kind == "proposed" \
            and event.notes == "conditional":
        # a conditional is its own status, not a plain proposal: 6.2 puts it in
        # NON_ACTIONABLE_STATUSES, and scoring it as `proposed` hides the class
        # the reader is supposed to learn (measured: conditional F1 0.00)
        return gold[0], gold[1], "conditional"
    return gold


# --- value sampling ---------------------------------------------------------
def _sample_value(kind: str, rng: random.Random, anchor: date,
                  project: str) -> str:
    if kind == "date":
        d = anchor + timedelta(days=rng.choice([14, 21, 30, 45, 60, 75]))
        months = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                  "août", "septembre", "octobre", "novembre", "décembre"]
        if rng.random() < 0.5:
            return f"{d.day} {months[d.month - 1]} {d.year}"
        return months[d.month - 1]
    if kind == "amount":
        return rng.choice(["180 k€", "240 k€", "1,5 M€", "2,4 M€", "3,2 M€",
                           "450 k€", "15 M€", "950 k€"])
    if kind == "vendor":
        return rng.choice(VENDORS)
    if kind == "scope":
        return rng.choice(SCOPES)
    if kind == "stack":
        return rng.choice(STACKS)
    if kind == "hiring":
        return rng.choice(HIRING)
    if kind == "risk":
        return rng.choice(["élevé", "moyen", "faible", "sous contrôle"])
    return rng.choice(["décidé", "reporté", "en attente"])


def _subject_key(project: str, attribute: str, role: str | None = None) -> str:
    key = f"{project.lower()}.{attribute}"
    return f"{key}.owner" if role else key


# --- the generator ----------------------------------------------------------
# The demo project (§13) must deterministically carry the subjects the seven
# scenarios ask about; a random attribute draw could silently omit `budget` and
# then "Quelle décision sur le budget Atlas?" has no evidence to retrieve at all
# (measured on the first bench run — a fixture bug, not an engine bug).
DEMO_PROJECT = {
    "code": "Atlas",
    "people": ["Marie", "Karim", "Sophie", "Julien"],
    "attributes": ["budget", "launch_date", "vendor", "risk_compliance"],
}


def build_ledger(n_projects: int = 20, meetings_per_project: int = 3,
                 seed: int = 42, distractor_rate: float = 0.35,
                 start: date | None = None, first_names: list[str] | None = None,
                 project_codes: list[str] | None = None,
                 demo_project: bool = True) -> Ledger:
    """Deterministic project/event timeline (7.1)."""
    rng = random.Random(subseed(seed, "ledger", n_projects, meetings_per_project))
    start = start or date(2026, 9, 1)
    names = first_names or FIRST_NAMES
    codes = list(project_codes or PROJECT_CODES)
    # NB: never grow the caller's list here — that used to swallow the
    # "too few codes" error below. `Atlas` is already first in PROJECT_CODES.
    if len(codes) < n_projects:
        raise ValueError(f"need >= {n_projects} project codes, got {len(codes)}")

    projects: list[dict[str, Any]] = []
    meetings: list[Meeting] = []
    meetings_by_id: dict[str, Meeting] = {}
    events: list[Event] = []
    counter = 0

    def new_event(**kw: Any) -> Event:
        """Author an event and attach it to *its own* meeting.

        Attaching to `meetings[-1]` instead silently duplicated every answer
        event across two meetings (an answer legitimately belongs to the next
        meeting), which made the event->unit gold map non-injective and corrupted
        every downstream label. An unknown meeting id now raises.
        """
        nonlocal counter
        counter += 1
        ev = Event(event_id=f"e-{counter:05d}", **kw)
        events.append(ev)
        host = meetings_by_id.get(ev.meeting_id)
        if host is None:
            raise KeyError(f"event {ev.event_id} targets unknown meeting "
                           f"{ev.meeting_id!r}")
        host.event_ids.append(ev.event_id)
        return ev

    for p_index in range(n_projects):
        code = codes[p_index]
        if demo_project and code == DEMO_PROJECT["code"]:
            people = list(DEMO_PROJECT["people"])
            attrs = list(DEMO_PROJECT["attributes"])
        else:
            n_people = rng.choice([3, 4, 4, 5])
            offset = rng.randrange(0, max(1, len(names) - n_people))
            people = names[offset:offset + n_people]
            attrs = rng.sample(list(ATTRIBUTES), k=rng.choice([3, 4]))
        n_people = len(people)
        roles = rng.sample(ROLES, k=min(n_people, len(ROLES)))
        assignments = dict(zip(people, roles))
        projects.append({"project": code, "people": people, "roles": assignments,
                         "attributes": attrs,
                         "attribute_meta": {a: ATTRIBUTES[a]["kind"] for a in attrs}})

        m_ids = [f"meeting-{p_index + 1:02d}-{k + 1}" for k in range(meetings_per_project)]
        state: dict[str, dict[str, Any]] = {}

        # Create every meeting for this project BEFORE authoring events: an
        # `answer` event is legitimately spoken in the *next* meeting, and
        # `new_event` now refuses to attach to a meeting that does not exist.
        for m_i, m_id in enumerate(m_ids):
            mdate = start + timedelta(days=7 * (3 * p_index + m_i) + rng.choice([0, 1, 2]))
            participants = people if rng.random() < 0.6 else rng.sample(
                people, k=max(2, n_people - 1))
            m = Meeting(meeting_id=m_id, project=code, on=mdate,
                        title=f"Revue {code} #{m_i + 1}", participants=list(participants))
            meetings.append(m)
            meetings_by_id[m_id] = m

        for m_i, m_id in enumerate(m_ids):
            meeting = meetings_by_id[m_id]
            mdate = meeting.on
            participants = meeting.participants
            speaker = rng.choice(participants)

            # 1. the scripted lifecycle: propose -> confirm -> change (-> cancel/reinstate)
            for attr in attrs:
                meta = ATTRIBUTES[attr]
                key = _subject_key(code, attr)
                if m_i == 0:
                    val = _sample_value(meta["kind"], rng, mdate, code)
                    ev = new_event(project=code, meeting_id=m_id, on=mdate,
                                   kind="proposed", subject_key=key, value=val,
                                   person=speaker)
                    state[key] = {"value": val, "event_id": ev.event_id, "status": "proposed"}
                elif m_i == 1:
                    cur = state.get(key)
                    if cur is None:
                        continue
                    if rng.random() < 0.75:  # confirm, possibly the same value
                        val = cur["value"] if rng.random() < 0.5 else _sample_value(
                            meta["kind"], rng, mdate, code)
                        # confirming a proposal replaces it; confirming an already
                        # confirmed value does not (that would be `repeated`)
                        sup = cur["event_id"] if cur["status"] != "confirmed" else None
                        ev = new_event(project=code, meeting_id=m_id, on=mdate,
                                       kind="confirmed", subject_key=key, value=val,
                                       person=speaker, supersedes=sup)
                        state[key] = {"value": val, "event_id": ev.event_id, "status": "confirmed"}
                    else:  # hypothesis after a proposal: must never supersede
                        val = _sample_value(meta["kind"], rng, mdate, code)
                        new_event(project=code, meeting_id=m_id, on=mdate,
                                  kind="hypothesis", subject_key=key, value=val,
                                  person=speaker)
                else:  # final meeting: change / cancel / reinstate / repeat
                    cur = state.get(key)
                    if cur is None:
                        continue
                    roll = rng.random()
                    if roll < 0.40:  # changed -> supersede
                        val = _sample_value(meta["kind"], rng, mdate, code)
                        if val == cur["value"]:
                            val = _sample_value(meta["kind"], rng, mdate + timedelta(days=7), code)
                        sup = _valid_predecessor(cur, events)
                        ev = new_event(project=code, meeting_id=m_id, on=mdate,
                                       kind="changed", subject_key=key, value=val,
                                       person=speaker, supersedes=sup)
                        state[key] = {"value": val, "event_id": ev.event_id, "status": "confirmed"}
                    elif roll < 0.55:  # cancelled
                        sup = _valid_predecessor(cur, events)
                        ev = new_event(project=code, meeting_id=m_id, on=mdate,
                                       kind="cancelled", subject_key=key,
                                       value=cur["value"], person=speaker,
                                       supersedes=sup)
                        state[key] = {"value": None, "event_id": ev.event_id, "status": "cancelled"}
                    elif roll < 0.68:  # reinstated
                        prev = _earliest_holder(key, events)
                        val = prev.value if prev else _sample_value(meta["kind"], rng, mdate, code)
                        sup = _valid_predecessor(cur, events)
                        ev = new_event(project=code, meeting_id=m_id, on=mdate,
                                       kind="reinstated", subject_key=key, value=val,
                                       person=speaker, supersedes=sup)
                        state[key] = {"value": val, "event_id": ev.event_id, "status": "confirmed"}
                    elif roll < 0.85:  # repeated statement -> supports
                        new_event(project=code, meeting_id=m_id, on=mdate,
                                  kind="repeated", subject_key=key,
                                  value=cur["value"], person=speaker)
                    else:  # conditional agreement, never supersedes
                        new_event(project=code, meeting_id=m_id, on=mdate,
                                  kind="proposed", subject_key=key,
                                  value=cur["value"], person=speaker,
                                  notes="conditional")

            # 2. ownership: commitments and role assignments
            if rng.random() < 0.7:
                owner = rng.choice(participants)
                attr = rng.choice(attrs)
                key = f"{code.lower()}.owner.{attr}"
                due = mdate + timedelta(days=rng.choice([7, 14, 21]))
                action = f"mettre à jour {ATTRIBUTES[attr]['np']}"
                prev = [e for e in events if e.subject_key == key and e.kind == "assignment"]
                sup = prev[-1].event_id if prev and prev[-1].person != owner else None
                new_event(project=code, meeting_id=m_id, on=mdate, kind="assignment",
                          subject_key=key, person=owner, role=attr,
                          value=owner, supersedes=sup)
                if rng.random() < 0.5:
                    ref = state.get(f"{code.lower()}.{attr}", {}).get("event_id")
                    new_event(project=code, meeting_id=m_id, on=mdate, kind="commitment",
                              subject_key=f"{code.lower()}.{attr}", owner=owner,
                              person=owner, due=due, action=action, value=owner,
                              commit_ref=ref, notes="commitment")

            # 3. question / answer pair (open point, then resolved next meeting)
            if rng.random() < 0.6:
                qa_attr = rng.choice(attrs)
                q = new_event(project=code, meeting_id=m_id, on=mdate, kind="question",
                              subject_key=f"{code.lower()}.{qa_attr}",
                              person=rng.choice(participants),
                              value=ATTRIBUTES[qa_attr]["np"])
                if m_i + 1 < meetings_per_project:
                    a_date = mdate + timedelta(days=7)
                    new_event(project=code, meeting_id=m_ids[m_i + 1], on=a_date,
                              kind="answer", subject_key=q.subject_key,
                              person=rng.choice(participants), answer_of=q.event_id,
                              value=_sample_value(ATTRIBUTES[qa_attr]["kind"],
                                                  rng, a_date, code))

            # 4. risk
            if rng.random() < 0.4:
                new_event(project=code, meeting_id=m_id, on=mdate, kind="risk",
                          subject_key=f"{code.lower()}.{rng.choice(attrs)}",
                          person=rng.choice(participants),
                          value=rng.choice(["délai trop court", "dépendance prestataire",
                                            "budget non arbitré", "données clients sensibles"]))

            # 5. distractors — another project's subject discussed in this meeting,
            #    overlapping vocabulary (7.1.1). Gold: never supersede across projects.
            if rng.random() < distractor_rate and len(projects) > 1:
                other = rng.choice(projects[:-1])
                dcode, dattr = other["project"], rng.choice(other["attributes"])
                new_event(project=dcode, meeting_id=m_id, on=mdate, kind="mention",
                          subject_key=_subject_key(dcode, dattr),
                          value=_sample_value(ATTRIBUTES[dattr]["kind"], rng, mdate, dcode),
                          person=rng.choice(participants), distractor=True)

    meta = {"generator": "ledger v0", "n_projects": n_projects,
            "meetings_per_project": meetings_per_project,
            "distractor_rate": distractor_rate, "start": start.isoformat(),
            "attribute_vocab_version": "v0"}
    return Ledger(projects, meetings, events, seed, meta)


def event(event_id: str, pool: list[Event]) -> Event:
    for e in pool:
        if e.event_id == event_id:
            return e
    raise KeyError(event_id)


def _valid_predecessor(cur: dict[str, Any], pool: list[Event]) -> str | None:
    eid = cur.get("event_id")
    if not eid:
        return None
    prev = event(eid, pool)
    # a cancelled/hypothesis holder cannot be superseded again
    if prev.kind in ("hypothesis",):
        return None
    return eid


def _earliest_holder(key: str, pool: list[Event]) -> Event | None:
    holders = [e for e in pool if e.subject_key == key and e.kind in
               ("confirmed", "changed", "reinstated", "proposed")]
    return holders[0] if holders else None


def summarise(ledger: Ledger) -> dict[str, Any]:
    kinds: dict[str, int] = {}
    for e in ledger.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    return {"projects": len(ledger.projects), "meetings": len(ledger.meetings),
            "events": len(ledger.events), "kinds": kinds,
            "supersede_edges": len(ledger.gold_supersede_edges())}
