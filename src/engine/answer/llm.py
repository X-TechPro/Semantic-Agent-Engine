"""Step 7 as the architecture defines it: retrieved messages + LLM + proof.

The previous implementation was a state-template machine: it had to decide which
slot and point a question asks for, and it failed on anything anaphoric
("c'est moi je le fais"). This passes the retrieved turns to a local model and
keeps only answers that cite a provided message and whose numbers appear in it.

Refusal is a first-class outcome: "INSUFFISANT" means the transcripts do not
contain the answer, which is exactly what the tricky questions test.
"""
from __future__ import annotations

import json
import re

from engine.answer.v4 import Answer, Citation, verify

INSUFFICIENT = "INSUFFISANT"
MAX_HITS = 10

SYSTEM = (
    "Tu réponds uniquement à partir des extraits fournis. Règles:\n"
    "1. Utilise seulement les extraits; n'ajoute aucune connaissance externe.\n"
    "2. Cite l'identifiant de chaque extrait utilisé, entre crochets, par "
    "exemple [nova:12].\n"
    f"3. Si l'information n'est pas dans les extraits, réponds exactement "
    f"{INSUFFICIENT}.\n"
    "4. N'écris un chiffre que s'il apparaît dans un extrait ou dans l'état "
    "mémoire que tu cites.\n"
    "5. Si plusieurs extraits donnent des valeurs différentes pour la même "
    "chose, la valeur actuelle est celle de l'extrait le plus récent (la date "
    "la plus tardive).\n"
    "6. Ne déduis pas. Une intention (« je vais envoyer », « à verser cette "
    "semaine ») ne prouve pas que l'action a eu lieu. Si aucun extrait ne dit "
    "que c'est fait, réponds " + INSUFFICIENT + ".\n"
    "7. Pour « première » ou « dernière » décision, compare les heures des "
    "tours cités : la première est la plus petite, la dernière la plus grande.\n"
    "8. Si tu ne peux pas confirmer ce que la question demande (une action "
    "qui n'a pas eu lieu, une personne qui n'est pas nommée), réponds "
    f"exactement {INSUFFICIENT} — sans phrase explicative.\n"
    "9. Réponds en français, en une ou deux phrases."
)

ID_IN_TEXT = re.compile(r"\[([^\[\]\s:]{1,80}(?::[^\[\]\s]+)?)\]")


def _messages(hits, store) -> list[dict]:
    rows = []
    for h in hits[:MAX_HITS]:
        uid = h.unit_id
        u = store.unit(uid) if store is not None else None
        msg = (store.message(u["message_id"]) if (u and store) else None) or {}
        who = msg.get("speaker") or "?"
        when = str(msg.get("timestamp") or "")[:10]
        text = (u or {}).get("text") or h.text
        rows.append({"unit_id": uid, "message_id": msg.get("message_id", ""),
                     "timestamp": msg.get("timestamp", ""),
                     "label": f"[{uid}] {when} {who}: {text}"})
    return rows


def _state_lines(store, hits) -> list[tuple[str, str]]:
    """The memory's current view: what steps 4-5 already decided.

    Step 7 is retrieval + memory, not retrieval alone: "what is the current
    budget" is answered by the state the pipeline derived, with the message
    that proves it.
    """
    if store is None:
        return []
    # every live state of the store, not only the ones whose evidence was
    # retrieved: "when was the salary confirmed" must not depend on whether
    # turn 12 made the top 10 for this question.
    live = {"confirmed", "cancelled", "completed"}
    latest: dict[str, dict] = {}
    for row in store.state_events():
        if row.get("status") not in live:
            continue
        latest[row["event_id"]] = row          # append-only: last wins
    if not latest:
        return []
    lines = []
    rows = []
    for event_id, row in latest.items():
        slot = str(row.get("slot") or "")
        # the event key is project:slot:point; without the point the model saw
        # "date = 15 août" and "date = 1er septembre" with nothing to tell them
        # apart (the keys vs the move).
        parts = str(event_id).split(":")
        unit = store.unit(str(row.get("evidence_unit") or "")) or {}
        # the event key drops the point when the tagger had none; the unit still
        # carries it, and without it "date = 1er septembre" reads like a date
        point = (parts[2] if len(parts) >= 3 else None) or unit.get("point")
        label = f"{slot} · {point}" if point and point != slot else slot
        # who confirmed it, resolved from the evidence turn: "who decided" is
        # the speaker of the confirming turn, not everyone who mentioned it
        msg = (store.message(unit.get("message_id", "")) if unit else None) or {}
        by = f" par {msg.get('speaker')}" if msg.get("speaker") else ""
        when = str(msg.get("timestamp") or row.get("created_at") or "")[:16].replace("T", " ")
        fr = {"confirmed": "confirmé", "cancelled": "annulé",
              "completed": "réalisé", "proposed": "proposé"}.get(
                  str(row.get("status")), str(row.get("status")))
        rows.append((when, label, str(row.get("value")),
                     f"- {label} : {row.get('value')} — {fr} le {when}{by}"
                     f" [{row.get('evidence_unit')}]",
                     str(row.get("evidence_unit") or "")))
    # the same slot is often written under several keys (project:slot,
    # project:slot:point, project:slot:slot). Keep the newest line per
    # (slot, value) or the model sees five "budget = 52 000" rows.
    best: dict[tuple, tuple] = {}
    for when, label, value, line, uid in rows:
        key = (label, value)
        if key not in best or when > best[key][0]:
            best[key] = (when, line, uid)
    ordered = sorted(best.values(), key=lambda t: t[0], reverse=True)
    return [(line, uid) for _, line, uid in ordered[:12]]


def build_prompt(query: str, hits, store) -> tuple[str, list[dict]]:
    rows = _messages(hits, store)
    body = "\n".join(r["label"] for r in rows)
    state_lines = _state_lines(store, hits)
    state = "\n".join(line for line, _ in state_lines)
    mem = f"État connu en mémoire:\n{state}\n\n" if state else ""
    # the state names its evidence unit; that turn must be citable too, even
    # when retrieval did not return it, or the model can only refuse
    have = {r["unit_id"] for r in rows}
    for _, uid in state_lines:
        if uid in have or not uid:
            continue
        u = (store.unit(uid) if store is not None else None) or {}
        msg = (store.message(u.get("message_id", "")) if store is not None else None) or {}
        when = str(msg.get("timestamp") or "")[:16].replace("T", " ")
        rows.append({"unit_id": uid, "message_id": msg.get("message_id", ""),
                     "timestamp": msg.get("timestamp", ""),
                     "label": f"[{uid}] {when} {msg.get('speaker', '?')}: "
                              f"{u.get('text', '')}"})
        have.add(uid)
    body = "\n".join(r["label"] for r in rows)
    user = (f"Question: {query}\n\n{mem}Extraits:\n{body}\n\nRéponds.")
    return user, rows


def cite_and_verify(reply: str, rows: list[dict], store) -> Answer:
    text = (reply or "").strip()
    if not text or INSUFFICIENT.lower() in text.lower():
        return Answer("Je n'ai pas assez d'information pour répondre.",
                      unknown=True, status="unknown")
    by_id = {r["unit_id"]: r for r in rows}
    cited = [i for i in dict.fromkeys(ID_IN_TEXT.findall(text)) if i in by_id]
    if not cited:
        # an answer without a citation is not proof: refuse rather than trust it
        return Answer("Je n'ai pas assez d'information pour répondre.",
                      unknown=True, verified=False, status="unknown")
    citations: list[Citation] = []
    evidence_texts: list[str] = []
    for uid in cited:
        u = store.unit(uid) if store is not None else None
        if u is None:
            continue
        ev = u.get("evidence") or {}
        citations.append(Citation(unit_id=uid, message_id=u["message_id"],
                                  start=ev.get("start"), end=ev.get("end"),
                                  text=ev.get("text") or u.get("text", "")))
        evidence_texts.append(u.get("text", ""))
    # ids contain digits ("hiring:12:u0"): strip them before checking that every
    # number in the claim appears in the cited messages
    claim = re.sub(r"\[[^\]]*\]", " ", text)
    evidence_texts += [r.get("timestamp", "") for r in rows if r["unit_id"] in cited]
    if not verify(claim, [{"text": t} for t in evidence_texts]):
        return Answer("Je n'ai pas assez d'information pour répondre.",
                      citations=citations, unknown=True, verified=False,
                      status="unknown")
    return Answer(text, citations, None, unknown=False, verified=True,
                  status="answered")


def call_ollama(model: str, system: str, user: str, *,
                base_url: str = "http://127.0.0.1:11434",
                timeout: float = 300.0, max_tokens: int = 2000) -> str:
    """Native /api/chat, not the OpenAI shim.

    `think: false` is the point: reasoning models (lfm3, lfm2.5-thinking) put
    the whole budget into their reasoning channel and return an empty answer
    with `finish_reason: length` when it is left on.
    """
    import os
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "keep_alive": os.environ.get("SAE_OLLAMA_KEEP_ALIVE", "5m"),
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/api/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return (body.get("message") or {}).get("content") or ""


def answer_llm(query: str, hits, store=None, *, model="qwen3.5:4b",
               **kw) -> Answer:
    if not hits:
        return Answer("Je n'ai pas assez d'information pour répondre.",
                      unknown=True, status="unknown")
    user, rows = build_prompt(query, hits, store)
    reply = call_ollama(model, SYSTEM, user, **kw)
    a = cite_and_verify(reply, rows, store)
    a.model = f"llm:{model}"  # type: ignore[attr-defined]
    return a
