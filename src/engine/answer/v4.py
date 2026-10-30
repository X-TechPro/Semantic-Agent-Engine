"""§7 answer: cited evidence, the state transition, or an explicit unknown.

The template is the product. Each claim carries a `message_id`, a `chunk_id` and
the exact span it was read from; a claim whose numbers are absent from those
spans is rejected by the verifier and the engine abstains instead of speaking.
An open proposal is answered as a proposal — never promoted to a decision.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

NUM = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class Citation:
    unit_id: str
    message_id: str
    chunk_id: str | None = None
    start: int | None = None
    end: int | None = None
    text: str = ""

    def as_dict(self) -> dict:
        return {"unit_id": self.unit_id, "message_id": self.message_id,
                "chunk_id": self.chunk_id, "start": self.start, "end": self.end,
                "text": self.text}


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    state_transition: dict | None = None
    unknown: bool = False
    verified: bool = True
    status: str = "unknown"

    def as_dict(self) -> dict:
        return {"text": self.text,
                "citations": [c.as_dict() for c in self.citations],
                "state_transition": self.state_transition,
                "unknown": self.unknown, "verified": self.verified,
                "status": self.status}


def _field(hit, name: str, default=None):
    if isinstance(hit, dict):
        return hit.get(name, default)
    return getattr(hit, name, default)


def _evidence_text(hit) -> str:
    ev = _field(hit, "evidence") or {}
    text = ev.get("text") if isinstance(ev, dict) else ""
    return text or (_field(hit, "text") or "")


def verify(text: str, evidence: list[dict]) -> bool:
    """Reject a number in the answer that is absent from every evidence span."""
    hay = " ".join(str(e.get("text") or "") for e in evidence)
    for n in NUM.findall(text):
        if n not in hay:
            return False
    return True


def _citations_for(unit_ids: list[str], store=None,
                   hits: list | None = None) -> list[Citation]:
    """Cite the units that carry the state, not the top-ranked hit.

    A state can be owned by a unit other than the rank-1 hit (the cancellation
    beats the proposal). Citing the hit would make the verifier reject the
    transition's number, which is the correct behaviour — so cite the evidence
    the state names.
    """
    out: list[Citation] = []
    seen: set[str] = set()
    for uid in unit_ids:
        if not uid or uid in seen:
            continue
        seen.add(uid)
        u = store.unit(uid) if store is not None else None
        if u is None:
            for h in hits or []:
                if _field(h, "unit_id") == uid:
                    out.append(Citation(unit_id=uid,
                                        message_id=_field(h, "message_id", ""),
                                        text=_evidence_text(h)))
                    break
            continue
        ev = u.get("evidence") or {}
        chunks = store.chunks_for_unit(uid)
        if not chunks:
            out.append(Citation(unit_id=uid, message_id=u["message_id"],
                                start=ev.get("start"), end=ev.get("end"),
                                text=ev.get("text") or u.get("text", "")))
        for c in chunks:
            out.append(Citation(unit_id=uid, message_id=c["message_id"],
                                chunk_id=c["chunk_id"], start=c["start"],
                                end=c["end"], text=c["text"]))
    return out


def _query_slot(query: str) -> str:
    try:
        from engine.read.v4_tags import extract_entities, slot_of

        slot = slot_of(query or "", tuple(extract_entities(query or "")))
        return "" if slot == "subject" else slot
    except Exception:                                  # pragma: no cover
        return ""


_STOP = set("le la les un une des de du au aux pour par avec sans sur dans est "
            "sont ai as ont que qui quoi dont ou et mais donc or ni car ce cette "
            "ces son sa ses leur leurs il elle on nous vous ils elles je tu se "
            "ete être avoir fait ca cela plus bien tout".split())


def _words(text: str) -> set[str]:
    import re as _re

    return {w for w in _re.findall(r"[\wà-ÿ]+", (text or "").lower())
            if len(w) > 2 and w not in _STOP}


def _supported(query: str, hit) -> bool:
    text = (_evidence_text(hit) or "") + " " + (_field(hit, "text") or "")
    return bool(_words(query) & _words(text))


_QUESTION_SHAPE = __import__("re").compile(
    r"\?\s*$|\b(?:quand|combien|qui|où|quel(?:le|s|les)?)\b",
    __import__("re").I)

_YESNO_Q = __import__("re").compile(
    r"^\s*(?:a-t-on|a-t-il|a-t-elle|est-ce que|est-il|est-elle|ont-ils)\b",
    __import__("re").I)

_COMPLETION_Q = __import__("re").compile(
    r"\b(?:a-t-il|a-t-elle|est-il|est-elle|ont-ils)\b.*"
    r"\b(?:[ée]t[ée]|fait|sign[ée]|publi[ée]|vers[ée]|pay[ée]|livr[ée])\b",
    __import__("re").I)


def _answers(query: str, query_slot: str, hit, state: dict,
             allow_fallback: bool = False) -> bool:
    if not state or not str(state.get("value") or "").strip():
        return False
    if state.get("open") and _COMPLETION_Q.search(query):
        # "l'acompte a-t-il été versé ?" is not answered by the proposal of a
        # deposit: an open state proves nothing was done
        return False
    slot = str(state.get("slot") or "")
    if query_slot and slot and slot != query_slot:
        return False
    value = str(state["value"]).lower()
    text = ((_evidence_text(hit) or "") + " "
            + (_field(hit, "text") or "")).lower()
    if value not in text:
        return False
    # the point is what the value is about: a salary question must not be
    # answered with the budget, even though both are amounts
    from engine.read.v4_tags import point_of

    q_point = point_of(query)
    eid = str(state.get("event_id") or "")
    s_point = eid.split(":", 2)[2] if eid.count(":") >= 2 else ""
    if q_point:
        if s_point and s_point != q_point:
            return False
        if not s_point and q_point not in text:
            # the statement does not name the point ("on va dire 52 000" for a
            # salary question): fall back to the slot, never for a question
            # that asks whether something was done
            if not allow_fallback or _COMPLETION_Q.search(query):
                return False
            if not (query_slot and slot and slot == query_slot):
                return False
    else:
        # no point to match: require lexical support, or the first state in
        # the meeting answers any question
        if _YESNO_Q.search(query) and not query_slot:
            return _supported(query, hit)
        return _supported(query, hit)
    return True


def _direct_answer(query: str, query_slot: str, hits, store,
                   exact_point: bool = True) -> Answer | None:
    """Answer from a turn that names the point but carries no state.

    "Qui rédige le post mortem ?" is answered by "c'est moi je le fais": the
    turn has no slot value, so the state path cannot use it, but it is the
    answer and its span can be cited.
    """
    if store is None:
        return None
    from engine.read.v4_tags import point_of

    q_point = point_of(query)
    allowed = {"answer", "decision", "commitment", "action", "fact"}
    for h in hits:
        uid = _field(h, "unit_id", "")
        u = store.unit(uid) if uid else None
        if not u or (u.get("kind") not in allowed):
            continue
        if u.get("kind") in ("fact", "decision", "commitment", "action"):
            val = str(u.get("value") or "").strip()
            if not val or val == str(u.get("text") or "").strip():
                # "c'est moi je le fais" answers a "who" question without a
                # value; any other slot needs a real value to quote
                if not (query_slot == "owner" and q_point):
                    continue
        if query_slot and u.get("slot") and u.get("slot") != query_slot:
            continue
        if _COMPLETION_Q.search(query) and (u.get("status") != "completed"):
            # "has the deposit been paid?" is not answered by "a deposit is
            # due": only a completed turn proves the act
            continue
        if q_point:
            if (u.get("point") or "") != q_point:
                continue
        elif exact_point or not _supported(query, h):
            continue
        body = (u.get("evidence") or {}).get("text") or u.get("text", "")
        if _QUESTION_SHAPE.search(body or ""):
            continue      # a question is not its own answer
        ev = u.get("evidence") or {}
        quote = ev.get("text") or u.get("text", "")
        if not quote.strip():
            continue
        msg = store.message(u["message_id"]) or {}
        who = msg.get("speaker") or "?"
        when = str(msg.get("timestamp") or "")[:10]
        text = f"D'après {who} ({when}) : « {quote.strip()} »"
        cites = _citations_for([uid], store, hits)
        # verify the quote, not the metadata: the date in the prefix is a
        # citation field and is not part of the evidence span
        if not verify(quote, [{"text": c.text} for c in cites if c.text]):
            continue
        return Answer(text, cites, None, unknown=False, verified=True,
                      status=str(u.get("status") or "unknown"))
    return None


def answer(query: str, hits, state: dict | None = None,
           store=None) -> Answer:
    hits = [h for h in (hits or [])]
    if not hits:
        return Answer("Je n'ai pas assez d'information pour répondre.",
                      unknown=True, status="unknown")
    query_slot = _query_slot(query)
    provided = state is not None
    chosen = None
    # 1. a state whose point matches exactly
    for h in hits:
        cand = state if provided else _field(h, "state")
        if cand and _answers(query, query_slot, h, cand, allow_fallback=False):
            chosen = (h, cand)
            break
    # 2. a turn that answers without a state but names the same point: a fact
    #    with status unknown still answers "when does the lease end"
    if chosen is None and not provided:
        direct = _direct_answer(query, query_slot, hits, store, exact_point=True)
        if direct is not None:
            return direct
    # 3. a state that names only the slot (never for "has it been done?")
    if chosen is None and not provided:
        for h in hits:
            cand = _field(h, "state")
            if cand and _answers(query, query_slot, h, cand, allow_fallback=True):
                chosen = (h, cand)
                break
    if chosen is None:
        # an injected state is authoritative: never substitute a hit for it
        direct = (None if provided else
                  _direct_answer(query, query_slot, hits, store,
                                 exact_point=False))
        if direct is not None:
            return direct
        return Answer("Je n'ai pas assez d'information pour répondre.",
                      _citations_for([_field(hits[0], "unit_id", "")], store, hits),
                      unknown=True, verified=not provided, status="unknown")
    top, st = chosen

    status = str(st.get("status") or "unknown")
    value = str(st.get("value"))
    open_ = bool(st.get("open"))
    transition = st.get("transition") or None

    # name the point, not the question: echoing the question back
    # ("État courant pour Quelle est la date... : 5 avril") read as a bug
    eid = str(st.get("event_id") or "")
    parts = eid.split(":")
    point = st.get("point") or (parts[2] if len(parts) >= 3 else None)
    label = str(point or st.get("slot") or _query_slot(query) or "Réponse").strip()
    fr = {"confirmed": "confirmé", "proposed": "proposé",
          "cancelled": "annulé", "completed": "réalisé",
          "unknown": "inconnu"}
    if open_:
        text = f"{label} : {value} (proposé, non confirmé)."
    elif status == "cancelled":
        text = f"{label} : annulé (valeur précédente : {value})."
    elif status == "completed":
        text = f"{label} : réalisé ({value})."
    else:
        text = f"{label} : {value} ({fr.get(status, status)})."
    if transition and str(transition.get("from")) != str(value):
        # a "transition" from 16 500 to 16 500 is the tagger restating the same
        # value on a later turn; it must not be announced as a replacement
        text += (f" Remplace « {transition.get('from')} » "
                 f"({transition.get('from_status')}).")

    ids = [str(st.get("evidence_unit") or _field(top, "unit_id", ""))]
    if transition and transition.get("from_evidence_unit"):
        ids.append(str(transition["from_evidence_unit"]))
    cites = _citations_for(ids, store, hits)
    ok = verify(text, [{"text": c.text} for c in cites if c.text])
    if not ok:
        return Answer("Je n'ai pas assez d'information pour répondre.",
                      citations=cites, unknown=True, verified=False,
                      status=status)
    return Answer(text, cites, transition, unknown=False, verified=True,
                  status=status)


__all__ = ["Answer", "Citation", "answer", "verify"]
