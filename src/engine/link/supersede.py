"""Deterministic supersede rules (BUILD_GUIDE 6.8.3 / 6.8.4).

Every guard is one named rule, unit-tested on its own, and applied **after**
whatever the (optional) pair classifier proposed. The model may propose; only
these guards may let a `supersedes` edge through — that is what keeps a false
supersede (the worst error in this project) rare and explainable.
"""

from __future__ import annotations

import re
from datetime import date

from engine.schemas import (SUPERSEDING_STATUSES, StatusLabel, Unit)


def same_slot(new: Unit, old: Unit) -> bool:
    return bool(new.slots.subject) and new.slots.subject == old.slots.subject


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")

#: 6.8 line 403: "repeated statement -> `supports`". Deliberately NARROW — only
#: phrases that say *nothing has changed*. It must not swallow a real change: "je
#: confirme, on passe à 2 M€" is a ratification, and `je confirme` alone is too
#: weak to call that a repeat. The link stage keeps its own copy rather than
#: importing the reader's `REPEAT` list (16.4: no stage reads another's
#: internals), even though the two overlap.
_REPEAT_CUE = __import__("re").compile(
    r"\bje (?:le )?redis\b|\bje (?:la |le )?répète\b|\bcomme (?:dit|disait) plus haut\b|"
    r"\bencore une fois\b|\bune fois de plus\b|\binchang[ée]e?s?\b|"
    r"\breste (?:le même|la même|identique)\b|\brien ne change\b",
    re.IGNORECASE)

#: speech acts that can hold a slot value. A `risk` note, a `commitment`, a
#: `question` or an `answer` are not value assertions about a subject, so they
#: must never sit on either side of a supersede edge (measured: 19 of 171 false
#: supersedes came from `risk`, 18 from `commitment`).
VALUE_BEARING_TYPES: frozenset[str] = frozenset({"decision", "idea", "fact"})
SUPERSEDING_TYPES: frozenset[str] = frozenset({"decision"})


#: scaffolding words that appear in every rendered statement and carry no value
_SCAFFOLD_WORDS = {
    "projet", "budget", "lancement", "date", "validation", "valid", "validement",
    "prestataire", "retenu", "perimetre", "stack", "technique", "rapport",
    "deadline", "tarif", "recrutement", "conformite", "risque", "environnements",
    "strategie", "prevu", "finalement", "du", "des", "de", "la", "le", "les",
    "est", "sont", "pour", "sur", "dans", "et", "a", "au", "avec", "on", "nous",
    "nouvelle", "valeur", "retient", "proposition", "act", "porte", "part",
    "occupe", "s", "occupera", "charge", "revient", "attention", "point",
    "vigilance", "m", "k",
}


def value_signature(unit: Unit) -> set[str]:
    """What this unit *asserts* as the slot's value, normalised.

    Slot fields first (they are already resolved): amount, date, owner. Then the
    content words of the sentence, which is what makes a word-valued attribute
    (vendor = "Nova Cloud", scope, stack) decidable at all — without this,
    vendor/stack/scope pairs have no comparable value and every legitimate
    supersede on them is lost (measured: recall 0.80 -> 0.36).
    """
    from engine.retrieve.relevance import content_words
    from engine.slots.subjects import fold

    sig: set[str] = set()
    if unit.slots.amount:
        sig.add(f"amount:{unit.slots.amount}")
    if unit.slots.date_value:
        sig.add(f"date:{unit.slots.date_value}")
    if unit.slots.owner:
        sig.add(f"person:{fold(unit.slots.owner)}")
    if unit.slots.persons:
        sig |= {f"person:{fold(p)}" for p in unit.slots.persons}
    words = {fold(w) for w in content_words(unit.text)}
    sig |= {w for w in words if w not in _SCAFFOLD_WORDS and len(w) > 3}
    return sig


def comparable_values(new: Unit, old: Unit, *, topical_min: float = 0.5,
                      equal_max: float = 0.8) -> tuple[bool, bool]:
    """(comparable, equal).

    `comparable` requires that the two units can actually be *read as two claims
    about one value*: either both carry a typed slot value (amount / date /
    person), or they phrase the same thing (topical overlap >= 0.5). Without the
    second condition, word-valued attributes (vendor, scope, stack) never
    compare and every legitimate supersede on them is lost (measured: recall
    0.80 -> 0.36). With only signatures and no floor, any two sentences sharing
    a subject key became "comparable", which is how the rules linker reached 171
    supersedes against 25 gold pairs.

    `equal` means the same value is asserted. That is *not* the same as "no
    supersede": ratifying a proposal (same value, proposed -> confirmed) is a
    state change and must supersede, while restating an already confirmed value
    is a `supports` (6.8.3). `evaluate` combines `equal` with both statuses.
    """
    sn, so = value_signature(new), value_signature(old)
    if not sn or not so:
        return False, False
    typed_n = {x for x in sn if ":" in x}
    typed_o = {x for x in so if ":" in x}
    if typed_n and typed_o:
        return True, typed_n == typed_o
    inter, union = len(sn & so), len(sn | so)
    if not union:
        return False, False
    topical = inter / min(len(sn), len(so))
    if topical < topical_min:
        return False, False
    return True, (inter / union) >= equal_max


def difference_kind(new: Unit, old: Unit) -> str:
    """How the two values differ: 'typed', 'word' or 'none'.

    'typed' means a resolved slot value changed (amount, date, owner, number) —
    strong evidence of a real replacement. 'word' means only free-text values
    differ ("portail interne" vs "phase pilote"), which is much weaker and must
    not be punished as hard, because a wrong supersede deletes the correct
    answer from the pool (measured: 12 false edges on dev cost 13% of gold).
    """
    for f in ("date_value", "amount"):
        nv, ov = getattr(new.slots, f), getattr(old.slots, f)
        if nv and ov:
            return "typed" if nv != ov else "none"
    from engine.slots.subjects import fold
    if new.slots.owner and old.slots.owner:
        return "typed" if fold(new.slots.owner) != fold(old.slots.owner) else "none"
    nnum = _NUMBER_RE.findall(new.text or "")
    onum = _NUMBER_RE.findall(old.text or "")
    if nnum and onum:
        return "typed" if set(nnum) != set(onum) else "none"
    sn, so = value_signature(new), value_signature(old)
    if sn and so and sn != so:
        return "word"
    return "none"


def confidence_for(kind: str) -> float:
    """0.9 for a typed replacement, 0.6 for a word-only one (6.8 `score`)."""
    return {"typed": 0.9, "word": 0.6, "none": 0.45}.get(kind, 0.6)


def date_order_ok(new: Unit, old: Unit) -> bool:
    return new.source_date >= old.source_date


def new_is_assertive(new: Unit) -> bool:
    return new.kind == "info" and new.status in {"confirmed", "cancelled"}


def evaluate(new: Unit, old: Unit, proposed: str = "supersedes", *,
             topical_min: float = 0.5, equal_max: float = 0.8,
             word_only_supersedes: bool = True,
             same_thread: bool = False,
             unknown_cardinality: str = "allow") -> tuple[str, str]:
    """Return (relation, guard_name). `proposed` is the classifier's guess.

    6.8.3 rules, in order:
      G1 same slot required            → else unrelated
      G7 value-bearing speech acts     → risk/commitment/question/answer cannot
                                         sit on either side of a replacement
      G9 values must be comparable     → else undecidable, never a replacement
      G4 restatement                   → supports, not supersedes
      G2 date order                    → older arriving later cannot supersede
      G3 assertiveness of the new unit → hypothesis/proposed/conditional never supersede
      G5 same date + same status       → keep as a real conflict (contradicts)

    `same_thread=True` is the D-046 pair-local regime, used only when a
    deterministic `thread.ThreadIndex` says the two units share a coarse slot
    and `old` arrived first. It substitutes for a *missing* subject key — never
    for a present one: two different resolved subjects still refuse (line 403).
    """
    if new.project != old.project:
        return "unrelated", "G0_cross_project"
    witness: str | None = None
    if not same_slot(new, old):
        if new.slots.subject and old.slots.subject:
            return "unrelated", "G1_different_attribute"
        if not same_thread:
            return "unrelated", "G1_no_slot"
        from engine.link.values import pair_slot_witness

        witness = pair_slot_witness(new.text, old.text)
        if witness is None:
            return "unrelated", "G1_no_slot"
    if proposed != "supersedes":
        return proposed, "G0_model"
    if new.type not in SUPERSEDING_TYPES:
        return "unrelated", "G7_source_not_a_value_assertion"
    if old.type not in VALUE_BEARING_TYPES:
        return "unrelated", "G7_target_not_value_bearing"
    if not old.is_current:
        # 6.8.4 writes old.valid_to, which only means something for the unit
        # that is *in force*. Superseding an already-superseded unit turned a
        # chain of N statements into N-1 edges (measured: 102 false supersedes).
        return "supports", "G8_target_not_current"
    # G12 cardinality: a set-valued slot accumulates, so a new value is an
    # addition, never a replacement. This is the guard Graphiti's invalidation
    # path is missing (issue #1728: 41% of facts retired, most of them additive;
    # their own conclusion is that predicate cardinality is the only
    # discriminator). Unknown cardinality is *not* silently functional when
    # `unknown_cardinality="contested"`: the failure direction must be a visible
    # extra fact, not a silently deleted true one. Default `allow` preserves the
    # measured gold behaviour, where most real-ASR drawers are coined keys.
    from engine.slots.subjects import cardinality

    kind = cardinality(new.slots.subject)
    if kind == "set":
        return "supports", "G12_set_valued_additive"
    if kind == "unknown" and unknown_cardinality == "contested":
        return "supports", "G12_unknown_cardinality"
    # NB: no veto on `old.status == "mention"`. A terse value statement ("Le
    # lancement est prévu en décembre") carries no status keyword, so the reader
    # labels it mention; vetoing on that made the 12.3 flagship December->January
    # update impossible and left System B scoring exactly like Baseline A on the
    # kill test. The legitimate veto is by speech act, which VALUE_BEARING_TYPES
    # already encodes (answer/question/risk/commitment are excluded there).
    comparable, equal = comparable_values(new, old, topical_min=topical_min,
                                         equal_max=equal_max)
    if not comparable:
        if witness:                 # pair-local value evidence already parsed
            comparable, equal = True, False
        else:
            # Undecidable is not "unrelated" when both sides assert a state for the
            # same slot: two in-force heads on one slot is exactly what 6.5 forbids
            # "unless explicitly marked parallel", so record the conflict instead of
            # staying silent. Measured: 13 such slots survived as unexplained
            # double-heads while the audit emitted zero `contradicts` edges for them.
            # 15's fallback is the same preference — contradict and ask, never guess.
            both_assertive = (new.status in SUPERSEDING_STATUSES
                              and old.status in SUPERSEDING_STATUSES)
            return ("contradicts" if both_assertive else "unrelated"), \
                "G9_values_not_comparable"
    if equal and old.status in SUPERSEDING_STATUSES:
        # restating a value that is already asserted -> supports, not supersedes
        return "supports", "G4_restatement"
    if _REPEAT_CUE.search(new.text or ""):
        # 6.8 line 403: "repeated statement" is `supports`. G4 only catches a
        # repeat whose typed value is identical, so "je le redis, ça reste
        # inchangé" — which names no amount, date or person at all — slipped past
        # it and became a replacement (measured: 260/260 golden `repeated` rows).
        # The cue lives here, not imported from the reader: no stage may read
        # another stage's internals (16.4).
        return "supports", "G4b_repeated_statement"
    # equal-but-old-is-soft is a *ratification* ("je propose 1,5 M€" -> "c'est
    # acté, 1,5 M€"): the value is unchanged but the state is, so it supersedes.
    if (not word_only_supersedes
            and difference_kind(new, old) == "word"):
        # free-text-only disagreement: too weak to demote, worth showing
        return "contradicts", "G11_word_only_value"
    if not date_order_ok(new, old):
        # 12.3 "out-of-order": an older document ingested later must NOT supersede
        return "contradicts", "G2_stale_source_date"
    if not new_is_assertive(new):
        # hypothesis / proposed / conditional after a decision. In the pair-local
        # regime a terse assertion carries no status word (rules reader: 768/781
        # `mention` on real ASR), so an explicit change cue stands in for it —
        # never for a question or a non-superseding speech act (G7 above).
        from engine.link.values import has_change_cue

        pair_assertive = (same_thread and witness is not None and new.kind == "info"
                          and new.type in SUPERSEDING_TYPES
                          and has_change_cue(new.text))
        if not pair_assertive:
            return "contradicts", "G3_not_assertive"
    # G6 (cancel-of-proposal -> contradicts) is GONE. 6.8.3 names `confirmed` and
    # `cancelled` as the superseding statuses and exempts nothing about the OLD
    # unit's state, so "je propose X" followed by "X est annulé" retires the
    # proposal. Inventing a veto there turned a state change into a disagreement,
    # left the proposal in force, and cost the guide's own cancel case its edge
    # (measured: 156/312 golden rows). Anything stricter than the guide has to be
    # justified by a metric, not by caution.
    if (not (same_thread and witness)
            and new.source_date == old.source_date and new.status == old.status):
        # simultaneous statements at the same strength: a real conflict, not an update.
        # Inside a thread the order is ingestion order, so equal source dates are
        # expected (one meeting = one source_date) and carry no simultaneity.
        return "contradicts", "G5_simultaneous_conflict"
    return "supersedes", (f"OK_pair_{witness}" if witness else "OK")


def apply_supersede(old: Unit, new: Unit, at: date | None = None) -> Unit:
    """6.8.4: valid_to = new.source_date; text/status of `old` are preserved."""
    when = at or new.source_date
    if old.valid_from and when < old.valid_from:
        when = old.valid_from
    return old.model_copy(update={"valid_to": when, "superseded_by": new.unit_id})


def chain_is_acyclic(store, unit_id: str) -> bool:
    return len(store.supersede_chain(unit_id)) == len(set(store.supersede_chain(unit_id)))
