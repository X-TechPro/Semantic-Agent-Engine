# Change definition (v1, 2026-10-26)

This is the target the labels, the review queue, and any detector must share.
It replaces the old implicit definition "a supersede pair with a canonical old
turn", which treated a refinement as no change and made the 047a labels and the
7B judge disagree.

## A change

A **new turn** counts as a change when it does at least one of these:

1. **decides** between options that were still open;
2. **closes or cancels** an open option;
3. **asserts a different value** on the same point (date, amount, place,
   number, owner, mode);
4. **commits** a choice that was still a proposal.

A refinement that selects one of the discussed options counts. The canonical
example is:

> ANCIEN: « on pourrait soit ouvrir soit clôturer »
> NOUVEAU: « plutôt clôturer »
> → change.

The old turn is a **citation**, not part of the label. Any earlier turn that
the new turn resolves is a valid citation; the gold pair is one canonical
citation, not the definition of the change.

## Not a change

- a restatement or agreement with the same value;
- an example, a detail, or a new clause that does not select or close an option;
- a new unrelated topic;
- a question that keeps the option open.

## Label policy

- Label the **new turn** binary: `change` / `no_change`.
- Record zero or more supporting old turns as citations.
- Score **new-turn recall/precision** first; score citation accuracy separately.
- Exact-pair scoring is a citation metric, never the primary target.

## Measured constraint

The definition was tested on the six labelled meetings on 2026-10-26
(D-062). The 7B judge accepted almost every contentful turn under it; the
turn-only classifier could not rank the positives. This document defines the
target; it does not claim the current model can enforce it.
