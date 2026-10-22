"""Rule-based linker — the 2.2 `link(new, neighbours) -> list[Edge]` baseline.

6.8.3's guards do the work; the "model" here is a cheap similarity test whose
only job is to decide *which* relation to propose, after which `supersede.evaluate`
decides whether a `supersedes` is legal. The trained cross-encoder replaces
`propose`, never the guards.
"""

from __future__ import annotations

import re
from datetime import datetime

from engine.schemas import (NON_ACTIONABLE_STATUSES, SUPERSEDING_STATUSES, Edge,
                            Unit, utcnow)
from engine.link.supersede import (comparable_values, confidence_for,
                                   difference_kind, evaluate, same_slot)

_QUESTION_RE = re.compile(r"\?|\best[-\s]ce que|\boù en\b|\bstatut\b", re.IGNORECASE)
_ACTIONISH_RE = re.compile(r"\b(?:mettre à jour|envoyer|préparer|rédiger|relancer|"
                          r"livrer|arbitrer|valider)\b", re.IGNORECASE)


def _lex_overlap(a: str, b: str) -> float:
    from engine.store.fts import tokenize

    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def propose(new: Unit, old: Unit, *, topical_min: float = 0.5,
            equal_max: float = 0.8) -> str:
    """Cheap classifier stand-in: what relation does this pair look like?"""
    if new.project != old.project:
        return "unrelated"
    # 7.3 / 6.8: an answer links to the question it answers by *speech act*. Keying
    # this on `kind == "query"` made `answers` unreachable in practice: a meeting
    # question is kind=info (7.3), and query units are never stored (2.1), so there
    # was never a `question` neighbour to find. Measured on the dev ledger: 19 gold
    # answer/answer_of pairs, 0 `answers` edges produced by either reader.
    #
    # NB: a slot-matching fallback ("any value-bearing unit on a question's subject
    # is its answer") was tried and REJECTED by measurement — it wrote 72 edges for
    # 5 gold pairs (precision 0.055), and after the one-answer-per-question
    # invariant it kept the wrong unit and hit 0/5. All 19 ledger pairs span
    # meetings, so the slot signal is not the answer signal. `answers` therefore
    # requires the reader to label the reply `answer`, which the rules baseline
    # cannot yet do (docs/BLOCKERS.md).
    if old.type == "question" and new.type == "answer":
        return "answers"
    if new.type == "question" and old.type == "answer":
        return "answers"
    if old.kind == "query" and new.type == "answer":
        return "answers"
    if new.kind == "query" and old.type == "answer":
        return "answers"
    if same_slot(new, old):
        comparable, equal = comparable_values(new, old, topical_min=topical_min,
                                              equal_max=equal_max)
        if not comparable:
            return "unrelated"
        if equal:
            # Equal VALUES are not an equal STATE. Confirming a proposal keeps
            # the value but changes what it is worth, which 6.8.4 models as a
            # supersede; only restating an already-assertive unit is `supports`.
            # Returning "supports" for every equal pair made the ratification
            # branch of supersede.evaluate dead code (measured: 17 gold
            # supersedes lost with guard=G0_model, i.e. never consulted).
            if old.status in {"confirmed", "cancelled"}:
                return "supports"
            return ("supersedes" if new.status in {"confirmed", "cancelled"}
                    else "supports")
        if new.status in {"confirmed", "cancelled"}:
            return "supersedes"
        if old.status == "confirmed" and new.status in {"proposed", "hypothesis",
                                                         "conditional"}:
            return "contradicts"
        return "supports"
    if new.type == "action" or (_ACTIONISH_RE.search(new.text) and old.type == "decision"):
        return "creates_action"
    if (new.slots.subject and old.slots.subject
            and new.slots.subject != old.slots.subject):
        # 6.8.3 / line 403: same project, DIFFERENT attribute = unrelated. The
        # subject key is the field that says what a sentence is about, so it has to
        # outrank word overlap. It did not: two rendered sentences share boilerplate
        # ("du projet Atlas est acté") above the similarity floor, so the linker
        # claimed `supports` for facts that merely mention the same project
        # (measured: 130/130 same-topic pairs). Both keys must be present, so a
        # genuine paraphrase with an unresolved subject still falls through.
        return "unrelated"
    if _QUESTION_RE.search(new.text) and _QUESTION_RE.search(old.text):
        return "unrelated"
    return "supports" if _lex_overlap(new.text, old.text) > 0.45 else "unrelated"


class RuleLinker:
    name = "rules"

    #: the proposal step only; `link()` always runs the guards afterwards
    propose = staticmethod(propose)

    def __init__(self, guards: bool = True, min_score: float = 0.55, *,
                 topical_min: float = 0.5, equal_max: float = 0.8,
                 word_only_supersedes: bool = True,
                 prefer_assertive_head: bool = True):
        self.guards = guards
        self.min_score = min_score
        self.topical_min = topical_min
        self.equal_max = equal_max
        self.word_only = word_only_supersedes
        self.prefer_assertive_head = prefer_assertive_head

    def link(self, new: Unit, neighbours: list[Unit], *,
             created_at: datetime | None = None) -> list[Edge]:
        edges: list[Edge] = []
        now = created_at or utcnow()
        # G12: "target must be current" (G8) is necessary but not sufficient.
        # A hypothesis is never superseded, so it stays current forever and a
        # slot ends up with several current units; superseding the soft one
        # instead of the in-force decision produced 3 of dev's false edges and
        # the double heads the consistency audit kept flagging.
        assertive_slots = {o.slots.subject for o in neighbours
                           if o.is_current and o.slots.subject
                           and o.status in SUPERSEDING_STATUSES}
        for old in neighbours:
            if old.unit_id == new.unit_id:
                continue
            rel = self.propose(new, old, topical_min=self.topical_min,
                             equal_max=self.equal_max)
            guard = "G0_model"
            score = 0.6
            if rel == "supersedes":
                rel, guard = (evaluate(new, old, "supersedes",
                                topical_min=self.topical_min,
                                equal_max=self.equal_max,
                                word_only_supersedes=self.word_only,
                                ) if self.guards
                              else ("supersedes", "OFF"))
                # `score` is evidence strength, not decoration: the validity
                # filter scales how far it demotes by exactly this number, so a
                # word-only guess cannot delete the correct answer the way a
                # typed change does (measured: 12 false edges cost 13% of gold)
                if (rel == "supersedes" and self.prefer_assertive_head
                        and old.status in NON_ACTIONABLE_STATUSES
                        and old.slots.subject in assertive_slots):
                    rel, guard = "supports", "G12_soft_is_not_the_head"
                score = (confidence_for(difference_kind(new, old))
                         if rel == "supersedes" else 0.7)
            elif rel == "supports":
                _, equal = comparable_values(new, old, topical_min=self.topical_min,
                                            equal_max=self.equal_max)
                score = 0.75 if (same_slot(new, old) and equal) else 0.6
            elif rel == "contradicts":
                score = 0.7
            if rel == "unrelated":
                # unrelated edges are still recorded when the guard fired: the
                # audit needs to see why a supersede was refused
                if not guard.startswith("G0"):
                    score = 0.5
                else:
                    continue
            if score < self.min_score and rel != "supersedes":
                continue
            edges.append(Edge(from_id=new.unit_id, to_id=old.unit_id, relation=rel,
                              score=round(score, 3), evidence_source_id=new.source_id,
                              created_at=now, guard=guard))
        return edges


def link(new: Unit, neighbours: list[Unit], cfg=None) -> list[Edge]:
    cfg = cfg or {}
    link_cfg = (cfg.get("link", {}) if isinstance(cfg, dict) else {})
    return RuleLinker(guards=link_cfg.get("guards", True),
                      min_score=link_cfg.get("supersede_min_score", 0.55),
                      # fallback must equal engine.config.DEFAULTS["link"]["value_topical_min"];
                      # it used to say 0.5 here and 0.0 in supervised.build_linker, so a
                      # config missing the key gave the two linkers different G9 thresholds
                      # and any rules-vs-model ablation stopped being like-for-like.
                      topical_min=link_cfg.get("value_topical_min", 0.0),
                      equal_max=link_cfg.get("value_equal_max", 0.8),
                      word_only_supersedes=link_cfg.get(
                          "word_only_supersedes", True),
                      prefer_assertive_head=link_cfg.get(
                          "prefer_assertive_head", True),
                      ).link(new, neighbours)
