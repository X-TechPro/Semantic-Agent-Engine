"""Validity resolver (6.10) — the second half of the thesis.

    final = fused_or_rerank_score × validity_factor

A superseded unit is not deleted and not hidden by default: it is *down-weighted*
by `alpha` (0.0 for "current" queries → effectively filtered, 1.0/0.3 for
"history" queries), and a hypothesis/proposed/conditional unit is down-weighted
by `beta` unless the query explicitly asks for proposals. `hard_filter=True`
turns the soft penalty into a drop (ablation 11.2).
"""

from __future__ import annotations

from engine.schemas import Filters, Query, QueryMode, Unit

SOFT_STATUSES = frozenset({"hypothesis", "proposed", "conditional"})
HISTORY_WORDS = ("historique", "avant", "initialement", "au départ", "d'abord",
                 "history", "previous", "à l'époque", "hier", "rappele")
PROPOSAL_WORDS = ("proposition", "propositions", "hypothèse", "hypothèses", "piste",
                  "pistes", "idée", "idées", "proposal", "proposals", "et si on")


def wants_proposals(query: Query) -> bool:
    """Does the question explicitly ask for proposals/hypotheses? (6.10)"""
    text = (query.text or "").lower()
    return any(w in text for w in PROPOSAL_WORDS)


def is_history_query(query: Query) -> bool:
    text = (query.text or "").lower()
    return query.mode == "history" or any(w in text for w in HISTORY_WORDS)


def validity_factor(unit: Unit, mode: QueryMode = "current", *,
                    alpha_superseded: float = 0.0, alpha_history: float = 0.3,
                    beta_soft: float = 0.5, proposals_requested: bool = False,
                    hard_filter: bool = False, confidence: float = 1.0) -> float:
    if not unit.is_current:
        if mode == "history":
            # A history question *asks for* the superseded statement, so that
            # unit is the target, not stale noise. Measured on the e5 run: giving
            # it alpha_history=0.3 pushed the gold unit out of top-5 on 6 of 14
            # history questions (recall@5 0.929 -> 0.571), which accounted for
            # the entire apparent recall cost of the relation layer. 6.10's 0.3 is
            # written for "current" queries where old evidence may still surface;
            # applying it to the query type that seeks that evidence inverts it.
            return 1.0
        if mode == "compare":
            # both generations should stay visible, current first
            return max(0.0, min(1.0, alpha_history)) if alpha_history > 0 else 1.0
        if hard_filter:
            return 0.0
        # 0.4 confidence must not erase evidence as hard as 0.9 does: a wrong
        # supersede is unrecoverable once the gold unit leaves the pool.
        conf = min(1.0, max(0.0, confidence))
        eff = alpha_superseded + (1.0 - alpha_superseded) * (1.0 - conf)
        return max(0.0, min(1.0, eff))
    if unit.status in SOFT_STATUSES and not proposals_requested:
        if hard_filter:
            return 0.0
        return max(0.0, min(1.0, beta_soft))
    return 1.0


def apply_validity(scored: list[tuple[Unit, float]], query: Query, *,
                   alpha_superseded: float = 0.0, alpha_history: float = 0.3,
                   beta_soft: float = 0.5, hard_filter: bool = False,
                   confidence_of=None
                   ) -> list[tuple[Unit, float, float, float]]:
    """-> [(unit, fused_score, validity_factor, final_score)], order preserved."""
    proposals = wants_proposals(query)
    mode: QueryMode = "history" if is_history_query(query) else query.mode
    out: list[tuple[Unit, float, float, float]] = []
    for unit, score in scored:
        v = validity_factor(unit, mode, alpha_superseded=alpha_superseded,
                            alpha_history=alpha_history, beta_soft=beta_soft,
                            proposals_requested=proposals, hard_filter=hard_filter,
                            confidence=(confidence_of(unit) if confidence_of
                                        else 1.0))
        if v <= 0.0:
            continue
        out.append((unit, score, v, score * v))
    return out


def filters_for_query(query: Query) -> Filters:
    """6.7: metadata filters go in *before* scoring when the slot is explicit.

    Only an explicit `query.project` becomes a hard filter. The project is NOT
    inferred from `slots.subject` here: when no project was given, the slot
    extractor guesses `global.<attr>`, and filtering on that placeholder made
    `/query {"text": "budget Atlas"}` return zero hits (caught by
    tests/integration/test_api.py). Inferring a target and then filtering on it
    starves retrieval; the subject only ever acts as a soft boost downstream.
    """
    return Filters(projects=[query.project] if query.project else [])
