"""Deterministic v4 linker: same slot, different value, settled new turn.

The architecture's own rule, not a model. A relation head trained on other
meetings scored 0 of 7 on purpose-built data; the rule needs a value, and step 3
now provides one.
"""
from __future__ import annotations

from engine.schemas_v4 import LinkEvent, SemanticUnit

SETTLED = {"confirmed", "cancelled", "completed"}


class V4RuleLinker:
    available = True
    model_version = "v4-rules-1"

    def link(self, new: SemanticUnit, neighbours: list[SemanticUnit], *,
             evidence: str = "") -> list[LinkEvent]:
        if new.status not in SETTLED or not new.slot or not new.value:
            return []
        value = str(new.value).strip().lower()
        cands = [c for c in neighbours
                 if c.unit_id != new.unit_id
                 and c.slot == new.slot
                 and c.value
                 and str(c.value).strip().lower() != value]
        if not cands:
            return []
        # candidates arrive in store order (chronological): the latest earlier
        # statement of a different value is the one being replaced
        old = cands[-1]
        return [LinkEvent(
            link_id=f"{new.unit_id}->{old.unit_id}:supersedes",
            from_unit_id=new.unit_id, to_unit_id=old.unit_id,
            relation="supersedes", confidence=1.0,
            evidence=evidence or (new.evidence.text if new.evidence else new.text),
            model_version=self.model_version, origin=new.origin)]
