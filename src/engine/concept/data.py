"""Context sets for concept attention (6.13) — built from the store, not invented.

A sample is one centre unit plus its k nearest neighbours in the *base* space,
which is exactly the "attention over concepts instead of tokens" premise: the
model sees the frozen embeddings and the relations already in the ledger, and
has to produce vectors that separate what the base space cannot.

The labels come from data the engine already stores, so no new annotation is
needed and none is smuggled in:

* **positives** — a real edge between centre and neighbour (`supersedes`,
  `supports`). The relation is the label; the model is not asked to rediscover it.
* **hard negatives** — same subject slot, *different status*. This is the pair
  6.13 names specifically and the pair the frozen space demonstrably loses on:
  "budget proposed at 1,8 M€" and "budget confirmed at 1,8 M€" are near-identical
  lexically and must end up far apart.
* **validity targets** — 1.0 for an in-force assertive unit, 0.0 for a superseded
  one. Anything else is left *unlabelled* (mask 0) rather than given a soft
  default: `proposed` is not "half-valid", and inventing a target for it would
  teach the head a fiction of mine instead of a fact of the ledger.

`train` split only, by project (§9.5 / 0.5). Nothing here reads test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from engine.concept.model import KINDS, SOURCE_TYPES, STATUSES, TYPES, ConceptAttention
from engine.schemas import Unit

#: statuses that assert a state of the world (mirrors `supersede.SUPERSEDING_STATUSES`
#: semantics for the validity target; duplicated deliberately so a guard change
#: cannot silently re-label the training set underneath a published checkpoint).
ASSERTIVE_STATUSES = ("confirmed", "cancelled")


@dataclass
class ContextSet:
    centre_id: str
    ids: list[str]                      # index 0 is the centre
    base: np.ndarray                    # [N, dim] frozen base vectors
    src: list[int]
    kind: list[int]
    type: list[int]
    status: list[int]
    gap_days: list[float | None]
    turn: list[int | None]
    pos_of: dict[str, int]              # unit_id -> index, for label lookups
    positives: list[int] = field(default_factory=list)     # indices into ids
    hard_negatives: list[int] = field(default_factory=list)
    validity: dict[int, float] = field(default_factory=dict)  # index -> 0.0/1.0

    @property
    def n(self) -> int:
        return len(self.ids)


def _index(table: tuple[str, ...], value: str | None) -> int:
    return ConceptAttention.index_of(table, value)


def gap_days(newest: date, d: date | None) -> float | None:
    """Days between the corpus's newest date and this unit's, negative for none.

    Returned as `None` rather than 0 when the unit has no date, so the model's
    "unknown date" embedding row stays distinguishable from "same day".
    """
    if d is None:
        return None
    return float((newest - d).days)


def edge_pairs(store) -> dict[tuple[str, str], str]:
    """(from_id, to_id) -> relation, for the whole store."""
    out: dict[tuple[str, str], str] = {}
    for r in store.db.execute("SELECT from_id, to_id, relation FROM edges").fetchall():
        out[(r["from_id"], r["to_id"])] = r["relation"]
    return out


def subject_of(unit: Unit) -> str | None:
    s = unit.slots.subject
    if not s:
        return None
    return s.lower()


def build_sets(store, *, k: int = 16, ids: list[str] | None = None,
               restrict: set[str] | None = None) -> list[ContextSet]:
    """One context set per unit: itself + k nearest base-space neighbours.

    `restrict` is the split gate — a set of unit ids that may be *used* (as the
    centre). Neighbours are still drawn from the whole corpus, because that is
    what retrieval would see at query time; filtering neighbours to the train
    projects would make the refined space cheaper than the live one and the A/B
    would stop being the same experiment. The split discipline applies to what is
    trained on and what is scored, not to what a unit can see.
    """
    all_ids = [u.unit_id for u in store.all_units()]
    if not all_ids:
        return []
    units = {u.unit_id: u for u in store.all_units()}
    vec_ids, mat = store.vector_matrix("base")
    if len(vec_ids) == 0:
        return []
    vec_of = {uid: mat[i] for i, uid in enumerate(vec_ids)}
    if ids is None:
        ids = [uid for uid in all_ids if uid in vec_of]
    newest = max((u.source_date for u in units.values() if u.source_date),
                 default=date.today())
    pairs = edge_pairs(store)
    sets: list[ContextSet] = []
    for centre in ids:
        if restrict is not None and centre not in restrict:
            continue
        cu = units.get(centre)
        v = vec_of.get(centre)
        if cu is None or v is None:
            continue
        nbrs = [uid for uid, _s in store.neighbors(v, k + 1, use="base")
                if uid != centre and uid in vec_of][:k]
        member_ids = [centre] + nbrs
        if len(member_ids) < 2:
            continue          # attention over one concept is the identity map
        idx = {uid: i for i, uid in enumerate(member_ids)}
        positives, negatives = [], []
        for uid in nbrs:
            i = idx[uid]
            nu = units[uid]
            rel = pairs.get((centre, uid)) or pairs.get((uid, centre))
            if rel in ("supersedes", "supports", "answers", "creates_action"):
                positives.append(i)
                continue
            same_slot = (subject_of(cu) is not None
                         and subject_of(cu) == subject_of(nu))
            if same_slot and cu.status != nu.status:
                negatives.append(i)          # 6.13's hard negative, by construction
            elif rel == "contradicts":
                negatives.append(i)
        validity: dict[int, float] = {}
        for uid in member_ids:
            u = units[uid]
            if u.superseded_by:
                validity[idx[uid]] = 0.0
            elif u.is_current and u.status in ASSERTIVE_STATUSES:
                validity[idx[uid]] = 1.0
        sets.append(ContextSet(
            centre_id=centre,
            ids=member_ids,
            base=np.stack([vec_of[uid] for uid in member_ids]).astype("float32"),
            src=[_index(SOURCE_TYPES, units[uid].source_type) for uid in member_ids],
            kind=[_index(KINDS, units[uid].kind) for uid in member_ids],
            type=[_index(TYPES, units[uid].type) for uid in member_ids],
            status=[_index(STATUSES, units[uid].status) for uid in member_ids],
            gap_days=[gap_days(newest, units[uid].source_date) for uid in member_ids],
            turn=[units[uid].turn_index if units[uid].turn_index is not None else 0
                  for uid in member_ids],
            pos_of=idx, positives=positives, hard_negatives=negatives,
            validity=validity))
    return sets


def label_counts(sets: list[ContextSet]) -> dict[str, float]:
    """Reported before any training run, so a degenerate label set is visible
    rather than discovered after the checkpoint says the model converged."""
    if not sets:
        return {"sets": 0}
    pos = sum(len(s.positives) for s in sets)
    neg = sum(len(s.hard_negatives) for s in sets)
    lab = sum(len(s.validity) for s in sets)
    cells = sum(s.n for s in sets)
    return {"sets": len(sets), "mean_context": round(cells / len(sets), 2),
            "positives": pos, "hard_negatives": neg,
            "validity_labelled": lab, "validity_supervised_rate": round(lab / cells, 4),
            "sets_with_positive": round(sum(1 for s in sets if s.positives) / len(sets), 4),
            "sets_with_hard_negative": round(
                sum(1 for s in sets if s.hard_negatives) / len(sets), 4)}


__all__ = ["ContextSet", "build_sets", "label_counts", "edge_pairs", "gap_days",
           "ASSERTIVE_STATUSES"]
