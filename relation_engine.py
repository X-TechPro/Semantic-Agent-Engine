"""Structural relation engine for meeting units.

Typed memory gives signals a generic classifier cannot see: project identity,
chronology, act cues and explicit date references. A small learned classifier is
useful in-distribution; this engine is the auditable fallback/hybrid for OOD
pairs, and follows the piscine guidance to use dates and provenance rather than
similarity alone.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import numpy as np

from hybrid_retrieval import ACTS, TOPICS

REFERENCE_RE = re.compile(r"(comme la derni[eè]re fois|r[ée]union pr[ée]c[ée]dente|s[ée]ance pr[ée]c[ée]dente|avait act[ée])", re.I)
DECISION_RE = re.compile(r"(d[ée]cision\s*:|c'est act[ée]|finalement retenue|on a act[ée]|valid[ée])", re.I)
COMMITMENT_RE = re.compile(r"(prends\s+(?:prends\s+)?en charge|s'engage|je m'engage)", re.I)
OBJECTION_RE = re.compile(r"(r[ée]serve|objection|pas d'accord|conteste)", re.I)
RISK_RE = re.compile(r"(risque|danger|menace)", re.I)
STOP = set("le la les de des du un une et ou mais que qui pour dans avec sur sans en a au aux ce cette ces il elle ils elles je tu nous vous on est sont".split())


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zà-ÿ0-9]+", text.casefold()) if t not in STOP and len(t) > 2}


def record_act(rec: dict[str, Any]) -> str:
    if rec.get("source_act"):
        return str(rec["source_act"])
    text = rec.get("message_text", "")
    if DECISION_RE.search(text):
        return "decision"
    if COMMITMENT_RE.search(text):
        return "engagement"
    if OBJECTION_RE.search(text):
        return "objection"
    if RISK_RE.search(text):
        return "risque"
    if "?" in text:
        return "question"
    if rec.get("act_probs") is not None:
        return ACTS[int(np.argmax(rec["act_probs"]))]
    return "information"


def record_topic(rec: dict[str, Any]) -> str:
    if rec.get("topic"):
        return str(rec["topic"])
    if rec.get("topic_probs") is not None:
        return TOPICS[int(np.argmax(rec["topic_probs"]))]
    return ""


def record_time(rec: dict[str, Any]) -> str:
    return f"{rec.get('meeting_date', '1970-01-01')}T{rec.get('timestamp', rec.get('raw', {}).get('timestamp', '00:00:00'))}"


def record_project(rec: dict[str, Any]) -> str:
    return str(rec.get("project_id") or rec.get("project") or "")


def predict_relation(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    a_act, b_act = record_act(source), record_act(target)
    a_text, b_text = source.get("message_text", ""), target.get("message_text", "")
    a_project, b_project = record_project(source), record_project(target)
    a_topic, b_topic = record_topic(source), record_topic(target)

    if a_project and b_project and a_project != b_project:
        return {"relation": "unrelated", "confidence": 0.98, "reason": "projets différents"}
    if REFERENCE_RE.search(a_text):
        return {"relation": "supports", "confidence": 0.96, "reason": "référence explicite à une session antérieure"}
    if a_act == "question" and b_act in ("reponse", "answer"):
        return {"relation": "answers", "confidence": 0.9, "reason": "question suivie d'une réponse"}
    if a_act in ("reponse", "answer") and b_act == "question":
        return {"relation": "answers", "confidence": 0.85, "reason": "réponse associée à une question"}
    if a_act == "decision" and b_act == "engagement":
        return {"relation": "creates_action", "confidence": 0.95, "reason": "décision suivie d'un engagement"}
    if a_act == "engagement" and b_act == "decision":
        return {"relation": "creates_action", "confidence": 0.9, "reason": "engagement issu d'une décision"}
    if a_act == "objection" and b_act in ("proposition", "decision"):
        return {"relation": "contradicts", "confidence": 0.9, "reason": "objection contre une proposition/décision"}
    if a_act == "proposition" and b_act == "decision":
        return {"relation": "supports", "confidence": 0.85, "reason": "proposition reprise par une décision"}
    if a_act == "decision" and b_act == "decision":
        try:
            later = datetime.fromisoformat(record_time(target)) > datetime.fromisoformat(record_time(source))
        except ValueError:
            later = True
        if later and (not a_topic or not b_topic or a_topic == b_topic):
            return {"relation": "supersedes", "confidence": 0.9, "reason": "décision plus récente"}
        return {"relation": "supports", "confidence": 0.7, "reason": "décisions du même périmètre"}
    overlap = len(tokens(a_text) & tokens(b_text)) / max(1, len(tokens(a_text) | tokens(b_text)))
    if (a_topic and a_topic == b_topic) or overlap >= 0.18:
        return {"relation": "supports", "confidence": 0.65 + min(0.2, overlap), "reason": "contenu partagé"}
    return {"relation": "unrelated", "confidence": 0.8, "reason": "aucun lien explicite"}
