"""§4 runtime linker: candidates in, typed LinkEvents out.

The heads never decide alone. §4's trained heads sit behind the same hard
guards v3 needed, because a relation classifier on 40 real pairs will happily
call two unrelated turns `supersedes`:

* a unit whose status is `proposed`/`unknown` never supersedes (soft units do
  not act);
* `supersedes` requires the same slot and a *different* value;
* `answers` requires the candidate to be a question;
* `creates_action` requires the incoming unit to be a commitment or action;
* `supports` requires the same slot and the same value.

If no trained checkpoint is present the linker returns no edges and says so in
`model_version`; it never substitutes encoder cosine for a relation label.
"""
from __future__ import annotations

from pathlib import Path

from engine.schemas_v4 import (
    NON_ACTIONABLE_STATUSES, LinkEvent, SemanticUnit,
)

RELATION_WEIGHTS = {
    "supersedes": 1.0, "answers": 0.8, "supports": 0.7,
    "creates_action": 0.6, "contradicts": 0.5, "unrelated": 0.0,
}


def _norm(v: str) -> str:
    return " ".join((v or "").lower().split())


class V4Linker:
    def __init__(self, model=None, checkpoint: str | Path | None = None,
                 device: str = "cuda", min_confidence: float = 0.5,
                 status_override: float = 0.75):
        self.model = model
        self.min_confidence = min_confidence
        self.status_override = status_override
        self.model_version = "rules-only"
        if self.model is None and checkpoint is not None:
            p = Path(checkpoint)
            if (p / "config.json").exists():
                from engine.link.flaubert_relation import FlaubertRelationModel
                self.model = FlaubertRelationModel.from_checkpoint(p, device=device)
                self.model_version = f"flaubert-relation@{p.name}"

    @property
    def available(self) -> bool:
        return bool(self.model is not None and getattr(self.model, "trained", False))

    # --- status ------------------------------------------------------------
    def status(self, unit: SemanticUnit) -> tuple[str, float, str]:
        """Status head over the unit, keeping the rule status if the head is shy."""
        if not self.available:
            return unit.status, unit.confidence, "v4-rules-1"
        label, conf = self.model.predict_status(unit.text)
        if conf >= self.status_override:
            return label, conf, self.model_version
        return unit.status, unit.confidence, "v4-rules-1"

    # --- relations ---------------------------------------------------------
    def allowed(self, unit: SemanticUnit, cand: SemanticUnit) -> list[str]:
        out = ["unrelated"]
        same_slot = bool(unit.slot) and unit.slot == cand.slot
        different_value = _norm(unit.value) != _norm(cand.value)
        if (unit.status not in NON_ACTIONABLE_STATUSES and same_slot
                and different_value):
            out += ["supersedes", "contradicts"]
        if cand.kind == "question":
            out.append("answers")
        if unit.kind in ("commitment", "action") and same_slot:
            out.append("creates_action")
        if same_slot and not different_value:
            out.append("supports")
        return out

    def link(self, unit: SemanticUnit, candidates: list[SemanticUnit],
             *, evidence: str | None = None) -> list[LinkEvent]:
        if not self.available or not candidates:
            return []
        rows: list[LinkEvent] = []
        for cand in candidates:
            if cand.unit_id == unit.unit_id:
                continue
            allowed = self.allowed(unit, cand)
            if allowed == ["unrelated"]:
                continue
            probs = self.model.relation_probs([unit.text], [cand.text])[0]
            relation = max(allowed, key=lambda r: probs.get(r, 0.0))
            conf = float(probs.get(relation, 0.0))
            if relation == "unrelated" or conf < self.min_confidence:
                continue
            rows.append(LinkEvent(
                link_id=f"{unit.unit_id}->{cand.unit_id}:{relation}",
                from_unit_id=unit.unit_id, to_unit_id=cand.unit_id,
                relation=relation, confidence=round(conf, 4),
                evidence=(evidence or (unit.evidence.text if unit.evidence
                                       else unit.text))[:400],
                model_version=self.model_version, origin=unit.origin))
        return rows


__all__ = ["V4Linker", "RELATION_WEIGHTS"]
