"""Runtime view over decision nodes (v3).

The meeting is immutable. A node groups the messages that belong to the same
decision; `current` is the latest child, and `at(day)` is what the node said on
that day. Nothing here writes to a unit: the view only selects an existing
message and returns it with its provenance.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from engine.link.thread import detect_topic
from engine.schemas import Unit
from engine.threads.session import DecisionNode, build_session_nodes

_STOP = {"le", "la", "les", "des", "du", "de", "un", "une", "et", "ou", "a",
         "au", "aux", "en", "dans", "sur", "pour", "par", "ce", "cette", "il",
         "elle", "on", "nous", "vous", "est", "sont", "quel", "quelle", "quand",
         "comment", "pourquoi", "que", "qui", "quoi"}


def _fold(text: object) -> str:
    s = unicodedata.normalize("NFD", str(text or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _terms(text: object) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", _fold(text))
            if len(w) > 2 and w not in _STOP}


@dataclass(frozen=True)
class NodeMatch:
    node: DecisionNode
    score: float
    current: Unit


class NodeView:
    """All decision nodes in a store, searchable by named topic or words."""

    def __init__(self, units: list[Unit], *, merge_topic: bool = True):
        self.nodes = build_session_nodes(units, merge_topic=merge_topic)

    @staticmethod
    def at(node: DecisionNode, day: date | None = None,
           turn: int | None = None) -> Unit | None:
        if turn is not None:
            return node.at_turn(turn)
        return node.as_of(day) if day is not None else node.latest

    def find(self, text: str, *, project: str | None = None,
             as_of: date | None = None, turn: int | None = None, k: int = 2,
             min_score: float = 1.0) -> list[NodeMatch]:
        qterms = _terms(text)
        qtopic = detect_topic(text)
        out: list[NodeMatch] = []
        for node in self.nodes:
            if project and node.project and node.project != project:
                continue
            child_terms: set[str] = set()
            for c in node.children:
                child_terms |= _terms(c.text)
            score = len(qterms & child_terms)
            if qtopic and qtopic == node.topic:
                score += 2.0
            if score < min_score:
                continue
            current = self.at(node, as_of, turn)
            if current is None:
                continue
            out.append(NodeMatch(node=node, score=score, current=current))
        out.sort(key=lambda m: (-m.score, -len(m.node.children), m.node.node_id))
        return out[:k]


__all__ = ["NodeMatch", "NodeView"]
