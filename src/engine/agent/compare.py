"""`compare` operation (6.11) — two sources, real relation per changed item.

The distinction the brief asks for is *mise à jour de décision* vs
*contradiction réelle*, and it comes straight from the edge table: a
`supersedes` edge is an update, a `contradicts` edge with no supersede is a
conflict the user has to know about.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from engine.schemas import Comparison, ComparisonItem, Hit, Unit

EXPLAIN = {
    "supersedes": "mise à jour de décision",
    "contradicts": "contradiction réelle",
    "supports": "confirmé / inchangé",
    "unrelated": "sujets distincts",
    "answers": "réponse apportée",
    "creates_action": "génère une action",
}


def _in_force(store, subject: str, project: str | None, on_date: date) -> list[Unit]:
    """Every assertion in force for a subject on a date (6.8 bi-temporal).

    `valid_to` is written only by `mark_superseded`, so a unit is in force from
    `valid_from` up to but NOT including `valid_to` (the supersede takes effect on
    that date). A SET is returned, not one unit: the same subject can carry a
    `proposed` and a `hypothesis` on the same day (atlas.budget does in the §13
    fixture), and picking one by row order made the chain test miss its link and
    report a real 1.5 M€ → 2.4 M€ update as a "contradiction réelle".

    This is what makes "what did we believe at meeting 1 vs meeting 3" answerable
    at all — `current()` can only answer it for *now*.
    """
    out = []
    for u in store.history(subject, project):
        start = u.valid_from or u.source_date
        if start > on_date:
            continue
        if u.valid_to is not None and on_date >= u.valid_to:
            continue
        out.append(u)
    return out


def _source_date(units: list[Unit], fallback: Unit | None = None) -> date | None:
    dates = [u.source_date for u in units]
    if not dates:
        return fallback.source_date if fallback else None
    return max(dates)


def build_comparison(store, left_id: str, right_id: str,
                     hits: list[Hit] | None = None) -> Comparison:
    left = store.units_of_source(left_id)
    right = store.units_of_source(right_id)
    edges = store.db.execute(
        """SELECT e.relation, e.from_id, e.to_id, e.score FROM edges e
           JOIN units uf ON uf.unit_id = e.from_id
           JOIN units ut ON ut.unit_id = e.to_id
           WHERE (uf.source_id=? AND ut.source_id=?)
              OR (uf.source_id=? AND ut.source_id=?)
           ORDER BY e.edge_id""",
        (left_id, right_id, right_id, left_id)).fetchall()

    items_by_rel: dict[str, list[ComparisonItem]] = defaultdict(list)
    # deduped on the (relation, old, new) triple ACROSS BOTH passes. Measured
    # before this existed: comparing meeting 1 with meeting 2 reported "6 mise à
    # jour" for 3 real changes, because the direct edge and the validity-interval
    # comparison each produced the same pair.
    seen: set[tuple[str, str, str]] = set()
    for e in edges:
        new_u = store.get(e["from_id"])
        old_u = store.get(e["to_id"])
        if new_u is None or old_u is None:
            continue
        # orient: the later statement is the "new" side
        if old_u.source_date > new_u.source_date:
            new_u, old_u = old_u, new_u
        key = (e["relation"], old_u.unit_id, new_u.unit_id)
        if key in seen:
            continue
        seen.add(key)
        items_by_rel[e["relation"]].append(ComparisonItem(
            subject=(new_u.slots.subject or new_u.text[:40]),
            relation=e["relation"],
            explanation=_explain(e["relation"], old_u, new_u),
            unit_ids=[old_u.unit_id, new_u.unit_id]))

    # The direct edges above are not enough, and this was a real defect: in the
    # §13 demo fixture the budget goes 1.8 -> 2.1 -> 2.4 across meetings 1, 2 and
    # 3, so comparing meeting 1 with meeting 3 finds NO direct edge between them
    # — only a `supports` edge from 3 to 1. The old code treated any edge as
    # coverage and reported "0 mise à jour, 5 confirmé": the exact opposite of the
    # truth, on the one comparison the demo asks for.
    #
    # The question is a state question, so it is answered with the validity
    # intervals: whatever was in force at the earlier meeting vs whatever was in
    # force at the later one.
    dl, dr = _source_date(left), _source_date(right)
    subjects = {u.slots.subject for u in left + right if u.slots.subject}
    changed_units: set[str] = set()
    if dl and dr and dl != dr:
        early_date, late_date = (dl, dr) if dl < dr else (dr, dl)
        project = next((u.project for u in left + right if u.project), None)
        # `seen` is the SAME set the direct-edge pass filled, and the key is the
        # same (relation, old, new) triple: a second local set here made this pass
        # blind to the first one and the adjacent-meeting comparison counted every
        # change twice.
        for subject in sorted(subjects):
            early = {u.unit_id: u for u in _in_force(store, subject, project, early_date)}
            late = {u.unit_id: u for u in _in_force(store, subject, project, late_date)}
            # An assertion that was in force at the earlier meeting and is gone at
            # the later one was retired. If the supersede chain from it lands on an
            # assertion in force at the later meeting, that is an update; if it
            # lands nowhere in the later state, the earlier claim simply does not
            # hold any more — a conflict the user has to be told about.
            for uid, old in early.items():
                if uid in late:
                    continue
                succ = next((sid for sid in store.supersede_chain(uid)
                             if sid in late), None)
                new = late[succ] if succ else None
                rel = "supersedes" if new else "contradicts"
                if new is None:
                    # nothing to show as the "after" side: report the retirement
                    # against the latest assertion the chain did produce, if any
                    tail = store.get(store.supersede_chain(uid)[-1])
                    if tail is None or tail.unit_id == uid:
                        continue
                    new = tail
                key = (rel, old.unit_id, new.unit_id)
                if key in seen:
                    continue
                seen.add(key)
                items_by_rel[rel].append(ComparisonItem(
                    subject=subject, relation=rel,
                    explanation=_explain(rel, old, new),
                    unit_ids=[old.unit_id, new.unit_id]))
                changed_units.update({old.unit_id, new.unit_id})

    # A subject whose in-force assertion actually changed between the two dates is
    # not "confirmed", however many `supports` edges touch it.
    unchanged = [i for i in items_by_rel.get("supports", [])
                 if not set(i.unit_ids or []) & changed_units]
    changed = [i for i in items_by_rel.get("supersedes", [])]
    contradicted = items_by_rel.get("contradicts", [])
    summary = (f"{len(changed)} mise(s) à jour, {len(contradicted)} contradiction(s), "
               f"{len(unchanged)} point(s) confirmé(s).")
    return Comparison(left_source=left_id, right_source=right_id,
                      unchanged=unchanged, changed=changed,
                      superseded=changed, contradicted=contradicted, summary=summary)


def _explain(rel: str, old: Unit, new: Unit) -> str:
    base = EXPLAIN.get(rel, rel)
    return (f"{base} : « {old.text.strip()} » ({old.source_date}, {old.status}) → "
            f"« {new.text.strip()} » ({new.source_date}, {new.status})")


def sources_in_order(store) -> list[str]:
    rows = store.db.execute(
        "SELECT DISTINCT source_id FROM units ORDER BY MIN(source_date), source_id"
    ).fetchall() if False else store.db.execute(
        "SELECT source_id FROM units GROUP BY source_id "
        "ORDER BY MIN(source_date), source_id").fetchall()
    return [r["source_id"] for r in rows]
