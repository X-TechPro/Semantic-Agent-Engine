"""Graph expansion (6.10): a superseded hit also returns the chain head.

"Which decision is in force?" must answer with the chain head and expose the
older units as history — that is the user-visible half of the relation layer.
"""

from __future__ import annotations

from engine.schemas import Hit, Unit


def follow_chain(store, unit_id: str) -> list[str]:
    return store.supersede_chain(unit_id)


def expand(store, hits: list[Hit], *, max_added: int = 10) -> list[Hit]:
    """Append chain heads that are not already present; record the path.

    Every returned hit carries `path` = the unit ids followed through
    `supersedes` edges, so provenance is auditable.
    """
    present = {h.unit.unit_id for h in hits}
    out: list[Hit] = []
    added = 0
    for h in hits:
        chain = follow_chain(store, h.unit.unit_id)
        path = chain if len(chain) > 1 else h.path
        if len(chain) > 1:
            head_id = chain[-1]
            out.append(h.model_copy(update={"path": path}))
            if head_id not in present and added < max_added:
                head = store.get(head_id)
                if head is not None:
                    present.add(head_id)
                    added += 1
                    out.append(Hit(unit=head, score=h.score,
                                   score_parts={**h.score_parts, "graph": 1.0},
                                   evidence=head.text, path=chain))
        else:
            out.append(h)
    return out


def predecessors(store, unit_id: str) -> list[Unit]:
    """Units superseded *by* this one (history behind a current decision).

    Edge orientation is `from_id = newer, to_id = older` (see `ingest.py` and
    `SQLiteStore.supersede_confidence`, which reads `to_id` to ask "who replaced
    me"). Reading the *incoming* side here therefore returned the unit's
    successors: for a chain head — which has none — it returned nothing, so the
    history 6.10 promises could never be shown. Outgoing edges are the ones this
    unit replaced.
    """
    out: list[Unit] = []
    for e in store.edges_for(unit_id, direction="from", relation="supersedes"):
        u = store.get(e.to_id)
        if u is not None:
            out.append(u)
    return out
