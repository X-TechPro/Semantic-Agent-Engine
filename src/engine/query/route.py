"""Classify what a question asks for, with the local model.

No substring lists: the model returns a two-field JSON object and the store
answers the lookup. If the model is unavailable, `classify` returns None and the
caller falls back to ordinary answering, with no meta branch.
"""
from __future__ import annotations

import json
import re

SYSTEM = (
    "Tu classes une question posée sur une réunion. Réponds uniquement en JSON, "
    'sans texte autour : {"object": "decision|message|value", '
    '"ordinal": "first|last|none"}.\n'
    "object=decision si la question porte sur une ou des décision(s) ; "
    "object=message si elle porte sur un ou des message(s)/tour(s) ; "
    "object=value sinon.\n"
    "ordinal=first si elle demande le premier / la première ; last si elle "
    "demande le dernier / la dernière ; none sinon.")

ALLOWED_OBJECTS = ("decision", "message", "value")
ALLOWED_ORDINALS = ("first", "last", "none")


def classify(question: str, model: str = "qwen3.5:4b",
             timeout: float = 30.0) -> dict | None:
    from engine.answer.llm import call_ollama

    try:
        reply = call_ollama(model, SYSTEM, question, timeout=timeout,
                            max_tokens=120)
    except Exception:
        return None
    match = re.search(r"\{.*?\}", reply or "", re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    obj = str(data.get("object", "")).strip().lower()
    ordinal = str(data.get("ordinal", "none")).strip().lower()
    if obj not in ALLOWED_OBJECTS:
        return None
    if ordinal not in ALLOWED_ORDINALS:
        ordinal = "none"
    return {"object": obj, "ordinal": ordinal}
