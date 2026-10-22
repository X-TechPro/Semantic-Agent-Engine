"""Intent router + agent operations (6.11) — `act(intent, hits) -> ActionResult`.

Routing is rule/slot based and evaluated separately from the LLM, per the brief.
The router never invents: a missing required slot produces a `Clarification`, and
a non-assertive status produces a refusal instead of an action object.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from engine.agent.actions import (build_email, build_event, build_task,
                                 guard_action)
from engine.agent.answer import build_answer
from engine.agent.compare import build_comparison
from engine.agent.dossier import build_dossier
from engine.agent.intents import ACTION_INTENTS, REFUSAL
from engine.agent.recipients import ContactBook
from engine.schemas import (ActionResult, Clarification, DraftEmail, Hit, Query,
                            TaskProposal, EventProposal, AnswerWithEvidence,
                            Comparison, Dossier, NON_ACTIONABLE_STATUSES)

if TYPE_CHECKING:  # composition root, for typing only (no import cycle)
    from engine.engine import Engine

_PROJECT_RE = re.compile(r"\bprojet\s+([A-ZÀ-Ý][\w'’-]+)", re.IGNORECASE)
_TO_PERSON_RE = re.compile(r"\b(?:à|au|aux|pour)\s+([A-ZÀ-Ý][\w'’-]+)\b")
_SOURCE_RE = re.compile(r"meeting-[\w.-]+")
_PRONOUNS = {"lui", "elle", "moi", "soi", "yon", "celui"}

#: request vocabulary -> subject attribute, for recipient resolution (6.12)
_ATTR_WORDS = {
    "budget": "budget", "lancement": "launch_date", "date": "launch_date",
    "prestataire": "vendor", "fournisseur": "vendor", "périmètre": "scope",
    "recrutement": "hiring", "tarif": "pricing", "prix": "pricing",
    "stack": "stack", "conformité": "risk_compliance", "rapport": "deadline_report",
    "deadline": "deadline_report", "risque": "risk_compliance",
}

DEFAULT_CLARIFY_OPTIONS = ["répondre", "comparer deux réunions", "faire un dossier",
                           "rédiger un mail", "créer une tâche", "planifier un point"]


class Agent:
    def __init__(self, engine: "Engine", book: ContactBook | None = None):
        self.engine = engine
        self.book = book or ContactBook.load(engine.cfg["agent"]["contacts_path"])

    # --- entry point -------------------------------------------------------
    def handle(self, text: str, *, intent: str | None = None,
               project: str | None = None, mode: str = "current") -> ActionResult:
        q = self._query(text, intent=intent, project=project, mode=mode)
        hits = self.engine.retrieve(q)
        chosen = intent or self.engine.intent_of(text)
        return self.act(chosen, hits, q)

    def act(self, intent: str, hits: list[Hit], query: Query) -> ActionResult:
        store = self.engine.store
        blocking = self._blocking_status(hits, intent, query)
        if intent in ACTION_INTENTS and not blocking:
            # 6.2 was guarded on the STORE unit only; the request itself can BE
            # the hypothesis ("si on annulait la buvette, il faudrait créer une
            # tâche") — the brief's hard case. The synthetic fixture never
            # phrased an imperative hypothetically; the real-data demo did and
            # the task sailed through (demo-real 2026-10-22, check 7 FAIL).
            if getattr(query, "status", None) in NON_ACTIONABLE_STATUSES:
                blocking = query.status

        if intent in ACTION_INTENTS and blocking:
            # 6.2 hard rule: hypothesis/proposed/conditional never act
            return ActionResult(
                intent=intent, operation="clarify",
                payload={"refused": blocking, "guard": "hypothesis_never_acts",
                         "question": "Ce point n'est pas acté : rien n'est créé."},
                hit_unit_ids=[h.unit.unit_id for h in hits],
                clarification=Clarification(
                    question="Ce point n'est pas acté. Confirmez-vous la décision "
                             "pour que je crée l'objet correspondant ?",
                    options=["Oui, c'est acté", "Non, rester en proposition"],
                    reason=REFUSAL.get(blocking, blocking)),
                notes=["guard:hypothesis_never_acts"])

        if intent == "answer":
            agent_cfg = self.engine.cfg["agent"]
            payload = build_answer(
                hits, query,
                min_score=float(agent_cfg["no_answer_threshold"]),
                overlap_floor=float(agent_cfg["no_answer_overlap"]))
            if agent_cfg.get("llm_answer"):
                payload, note = self._compose(payload, hits, query)
                return self._wrap("answer", payload, hits, notes=[note])
            return self._wrap("answer", payload, hits)
        if intent == "compare":
            pair = self._compare_sources(query, hits)
            if len(pair) < 2:
                return self._clarify("compare", hits,
                                     "Quelles deux réunions voulez-vous comparer ?",
                                     self._source_options())
            return self._wrap("compare", build_comparison(store, pair[0], pair[1], hits),
                              hits)
        if intent == "build_dossier":
            if not query.project:
                return self._clarify("build_dossier", hits,
                                     "De quel projet voulez-vous le dossier ?",
                                     sorted(self.engine.projects) or
                                     [u.project for u in store.all_units() if u.project])
            return self._wrap("build_dossier", build_dossier(store, query.project), hits)
        if intent == "draft_email":
            payload, status = self._email(query, hits)
            if isinstance(payload, Clarification):
                return ActionResult(intent="draft_email", operation="clarify",
                                    payload=payload.model_dump(mode="json"),
                                    hit_unit_ids=[h.unit.unit_id for h in hits],
                                    clarification=payload,
                                    notes=[f"recipient:{status}"])
            return self._wrap("draft_email", payload, hits,
                              notes=[f"recipient:{status}"])
        if intent == "create_task":
            return self._wrap("create_task",
                              build_task(query, hits, blocking_status=blocking), hits)
        if intent == "schedule_event":
            return self._wrap("schedule_event",
                              build_event(query, hits, blocking_status=blocking), hits)
        return self._clarify(intent or "none", hits,
                             "Je ne suis pas certain de l'opération demandée. "
                             "Que voulez-vous que je fasse ?", DEFAULT_CLARIFY_OPTIONS,
                             reason=f"intent low-confidence ({intent})")

    # --- operations needing extra resolution -------------------------------
    def _email(self, query: Query, hits: list[Hit]):
        attribute = self._attribute(query)
        named = None
        m = _TO_PERSON_RE.search(query.text)
        if m and m.group(1).lower() not in _PRONOUNS:
            named = m.group(1)
        return build_email(query, hits, store=self.engine.store, book=self.book,
                          roster=self.engine.roster or None, attribute=attribute,
                          person=named)

    def _compose(self, payload, hits: list[Hit], query: Query):
        """The diagram's last box, verify-gated (agent.llm_answer, default off).

        The LLM may only RE-FORMAT an answer whose claims are already cited;
        `agent/verify.check_output` compares the rewrite against the cited
        units and any invented name/number bounces it to the template — the
        6.11 render-only contract, enforced rather than assumed. The note says
        which of the three outcomes happened, so no artefact can claim an LLM
        wrote an answer that the verifier actually rejected.
        """
        from engine.agent.answer import compose_answer
        from engine.agent.llm import LLMUnavailable, build_llm
        from engine.agent.verify import check_output

        try:
            text = compose_answer(build_llm(self.engine.cfg), query, payload)
        except LLMUnavailable:
            return payload, "llm:unavailable — template stands"
        if text is None:
            return payload, "llm:nothing composed — template stands"
        cited = {i for c in payload.claims for i in c.unit_ids}
        units = [h.unit for h in hits if h.unit.unit_id in cited]
        verdict = check_output(text, units)
        if not verdict.ok:
            return payload, (f"llm:rejected by verifier "
                             f"(names={verdict.unsupported_names[:3]} "
                             f"numbers={verdict.unsupported_numbers[:3]}) — "
                             "template stands")
        return payload.model_copy(update={"answer": text}), "llm:composed+verified"

    # --- helpers -----------------------------------------------------------
    def _query(self, text: str, *, intent: str | None, project: str | None,
               mode: str) -> Query:
        read = self.engine.read_text(text)
        proj = project or self._project_in(text) or self._project_from_slots(read.unit)
        return Query(text=text, mode=mode, project=proj, slots=read.unit.slots,
                     intent=intent or read.intent or "answer",
                     source_unit_id=read.unit.unit_id, status=read.status)

    def _known_projects(self) -> set[str]:
        # `engine.projects` fills during ingest *in this process*: an engine
        # opened over an existing corpus (UI, API) has never ingested, so the
        # store's own facets are the other half of "what projects exist".
        return set(self.engine.projects) | set(self.engine.store.facets()["projects"])

    def _project_in(self, text: str) -> str | None:
        return resolve_project_for_text(self.engine, text)

    def _project_from_slots(self, unit) -> str | None:
        if unit.slots.subject and "." in unit.slots.subject:
            guess = unit.slots.subject.split(".", 1)[0]
            hit = {_fold(p): p for p in self._known_projects()}.get(_fold(guess))
            if hit:
                return hit
        return unit.project

    def _attribute(self, query: Query) -> str | None:
        if query.slots.subject and ".owner." in query.slots.subject:
            return query.slots.subject.rsplit(".", 1)[-1]
        low = _fold(query.text)
        for word, attr in _ATTR_WORDS.items():
            if _fold(word) in low:
                return attr
        return None

    def _blocking_status(self, hits: list[Hit], intent: str,
                         query: Query | None = None) -> str | None:
        """6.2 hard rule: hypothesis / proposed / conditional never create an
        action object.

        The rule is about **the unit the request is about**, not about whatever
        happened to rank first. Ranking the wrong referent in reverses the rule
        in both directions at once (measured on the e2e run): a *confirmed*
        ownership unit vetoed an e-mail because a conditional row ranked higher,
        while a *hypothesis*-only subject was allowed to create a task because a
        confirmed row ranked higher.
        """
        if intent not in ACTION_INTENTS:
            return None
        lead = self._lead_for_subject(hits, query)
        return guard_action(lead.unit) if lead else None

    @staticmethod
    def _lead_for_subject(hits: list[Hit], query: Query | None) -> Hit | None:
        """The hit that answers the request, chosen by *slot specificity*.

        Falling back to "subject endswith <attribute>" is not enough: an
        ownership request (`atlas.owner.budget`) then matches the budget-value
        row (`atlas.budget`) too, and a conditional budget row vetoed a confirmed
        assignment (measured on the e2e run). Preference order: exact slot →
        same owner slot → same attribute → any current.
        """
        subject = getattr(query, "slots", None) and query.slots.subject
        current = [h for h in hits if h.unit.is_current]
        if subject:
            exact = [h for h in current if h.unit.slots.subject == subject]
            if exact:
                return exact[0]
            attr = subject.rsplit(".", 1)[-1]
            if ".owner." in subject:
                # an ownership request must be judged on an ownership row
                same_owner = [h for h in current
                              if (h.unit.slots.subject or "").endswith(".owner." + attr)]
                if same_owner:
                    return same_owner[0]
            near = [h for h in current
                    if (h.unit.slots.subject or "").endswith(attr)]
            if near:
                return near[0]
        return current[0] if current else None

    def _compare_sources(self, query: Query, hits: list[Hit]) -> list[str]:
        explicit = _SOURCE_RE.findall(query.text)
        if len(explicit) >= 2:
            return explicit[:2]
        seen: list[str] = []
        for h in hits:
            if h.unit.source_id not in seen:
                seen.append(h.unit.source_id)
            if len(seen) >= 2:
                break
        return seen

    def _source_options(self) -> list[str]:
        rows = self.engine.store.db.execute(
            "SELECT source_id FROM units GROUP BY source_id "
            "ORDER BY MIN(source_date), source_id").fetchall()
        return [r["source_id"] for r in rows][:10]

    def _wrap(self, operation: str, payload, hits: list[Hit],
              notes: list[str] | None = None) -> ActionResult:
        if isinstance(payload, Clarification):
            return ActionResult(intent=operation, operation="clarify",
                                payload=payload.model_dump(mode="json"),
                                hit_unit_ids=[h.unit.unit_id for h in hits],
                                clarification=payload, notes=notes or [])
        return ActionResult(intent=operation, operation=operation,
                            payload=payload.model_dump(mode="json"),
                            hit_unit_ids=[h.unit.unit_id for h in hits],
                            notes=notes or [])

    def _clarify(self, intent: str, hits: list[Hit], question: str,
                 options: list[str], reason: str = "missing slot") -> ActionResult:
        clar = Clarification(question=question, options=list(dict.fromkeys(
            o for o in options if o)), reason=reason)
        return ActionResult(intent=intent, operation="clarify",
                            payload=clar.model_dump(mode="json"),
                            hit_unit_ids=[h.unit.unit_id for h in hits],
                            clarification=clar)


def _fold(s: str) -> str:
    import unicodedata

    # NFD alone is not an accent fold: "Boréal" decomposes to "bore" + a
    # combining acute that still blocks equality with "Boreal". Drop the
    # combining marks -- the same fold the lexical analyser applies
    # (store/fts.py, strip_accents), so query and index agree on spellings.
    return "".join(c for c in unicodedata.normalize("NFD", (s or "").lower())
                   if not unicodedata.combining(c))


def resolve_project_for_text(engine: "Engine", text: str) -> str | None:
    """The project a query is about, resolved to the spelling the store uses.

    `Query.project` is a HARD filter (retrieve/validity.py), so the raw
    "projet X" capture used to go straight into it: one missing accent
    ("Boreal" for "Boréal") filtered every real row out and the user saw "no
    results" for a one-keystroke spelling difference (measured 2026-10-22).
    The capture is therefore folded against the known project set first; an
    unknown name still passes through raw, because abstaining on a project the
    store has never seen is the honest answer.

    Known projects = this session's ingests UNION the store's own facets: a
    freshly opened engine (UI, API) has never ingested, and its router used to
    see an empty project set while the database held three projects.
    """
    known = set(engine.projects) | set(engine.store.facets()["projects"])
    m = _PROJECT_RE.search(text)
    if m:
        cand = m.group(1)
        return {_fold(p): p for p in known}.get(_fold(cand), cand)
    folded = _fold(text)
    for p in sorted(known, key=len, reverse=True):
        if re.search(rf"\b{re.escape(_fold(p))}\b", folded):
            return p
    return None


def act(intent: str, hits: list[Hit], engine: "Engine", query: Query | None = None
        ) -> ActionResult:
    """Module-level entry for the 2.2 contract."""
    agent = Agent(engine)
    q = query or Query(text=" ".join(h.unit.text for h in hits[:3]))
    return agent.act(intent, hits, q)
