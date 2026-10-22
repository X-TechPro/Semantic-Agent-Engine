"""Intent vocabulary for the agent (6.11)."""

from __future__ import annotations

from engine.schemas import IntentLabel

#: intents that may only fire on an assertive unit (6.2 hard rule)
ACTION_INTENTS: frozenset[str] = frozenset({"create_task", "schedule_event",
                                            "draft_email"})
RENDER_INTENTS: frozenset[str] = frozenset({"answer", "compare", "build_dossier"})
ALL_INTENTS: tuple[str, ...] = (
    "answer", "compare", "build_dossier", "draft_email", "create_task",
    "schedule_event", "clarify", "none",
)

#: what the agent must say when a guard refuses an action
REFUSAL = {
    "hypothesis": ("Ce point est une hypothèse, pas une décision : je ne crée rien "
                   "tant que ce n'est pas acté."),
    "proposed": ("Ce point est une proposition, pas une décision : je ne crée rien "
                 "tant que ce n'est pas acté."),
    "conditional": ("Ce point est conditionnel : rien n'est acté tant que la "
                    "condition n'est pas remplie."),
}


def is_action(intent: IntentLabel | str) -> bool:
    return intent in ACTION_INTENTS
