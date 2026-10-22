"""Hard negatives (BUILD_GUIDE 6.4 / 7.3): *constructed* minimal pairs.

These are not mined — they are authored from the ledger, so the gold label is
exact by construction. Five flip types, each a pair (positive, negative) whose
texts differ in exactly one respect the engine is supposed to be able to tell
apart:

  status     "Le budget est proposé."      vs "Le budget est validé."
  person     owner Marie                    vs owner Karim
  date       "en décembre"                  vs "en janvier"
  number     "1,5 M€"                       vs "15 M€"
  negation   "Le budget est validé."        vs "Ce n'est pas encore validé."
  attribute  budget of Atlas                vs launch date of Atlas

6.4 rule: constructed negatives are **never** filtered by the NV-Retriever
similarity rule — they are near-identical on purpose.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

from engine.seeding import subseed
from gen.ledger import ATTRIBUTES, Ledger


@dataclass(slots=True)
class HardNegative:
    pair_id: str
    flip: str
    anchor: str            # a query that carries the *discriminating cue*
    cue: str               # the gold element the positive has and the negative lacks
    positive: str          # correct-status text
    negative: str          # the trap
    gold_positive_unit_hint: str = ""
    project: str = ""
    subject: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"pair_id": self.pair_id, "flip": self.flip, "anchor": self.anchor,
                "cue": self.cue, "positive": self.positive, "negative": self.negative,
                "project": self.project, "subject": self.subject, "meta": self.meta}


#: The cue is DECLARED with its templates, and `assert_discriminating` then makes
#: it impossible for the declaration to drift from the sentences.
#:
#: This file used to keep cues in a separate `STATUS_CUE = {kind: word}` table.
#: That table had silently drifted from the templates it described: measured on 78
#: scoreable pairs, **27 status pairs had a cue that was in the negative (so the
#: anchor named the wrong member), in neither, or in both**. An anchor that cannot
#: tell the two texts apart pins pairwise accuracy at chance, which makes the whole
#: 6.4 hard-negative number unauditable — and it is a positive result produced by a
#: broken instrument, which is worse than a negative one.
#:
#: Cue placement is the whole point: the cue must live in the POSITIVE. So the
#: negation pair below has the negated sentence as its positive, because the
#: anchor asks about the thing that is *not yet* validated.
STATUS_FLIPS = [
    # (kind, cue, positive, negative)
    ("proposed", "proposé", "{np_cap} est proposé.", "{np_cap} est validé."),
    ("proposed", "proposé", "{np_cap} est proposé.", "{np_cap} est annulé."),
    ("hypothesis", "peut-être", "On pourrait peut-être revoir {np}.",
     "{np_cap} est acté."),
    ("conditional", "sous réserve", "D'accord pour {np} sous réserve du budget.",
     "{np_cap} est acté, sans condition."),
    ("negation", "pas encore", "Ce n'est pas encore validé pour {np}.",
     "{np_cap} est validé."),
]
PERSON_TPL = ("C'est Marie qui gère {np}.", "C'est Karim qui gère {np}.")
PERSON_CUE = "Marie"
DATE_TPL = ("{np_cap} est prévu en décembre.", "Finalement {np_cap} passe en janvier.")
DATE_CUE = "décembre"
NUMBER_TPL = ("{np_cap} est de 1,5 M€.", "{np_cap} est de 15 M€.")
NUMBER_CUE = "1,5 M€"
ATTRIBUTE_TPL = ("Le budget du projet {proj} est de 240 k€.",
                 "La date de lancement du projet {proj} est en mars.")
ATTRIBUTE_CUE = "budget"
REPEAT_TPL = ("Je confirme, {np} : {value}.", "{np_cap} : {value}.")


def assert_discriminating(cue: str, positive: str, negative: str,
                          pair_id: str = "") -> None:
    """Raise unless `cue` is in `positive` and absent from `negative`.

    Each failure branch names one of the three drift shapes actually measured in
    the old table, so a future edit cannot quietly reintroduce a chance-level pair.
    """
    if not cue:
        return                      # no cue = neutral anchor, marked unscoreable
    in_pos, in_neg = cue.lower() in positive.lower(), cue.lower() in negative.lower()
    if in_neg:
        raise ValueError(
            f"cue {cue!r} appears in BOTH members of {pair_id} ({positive!r} / "
            f"{negative!r}), so the anchor cannot rank them — pairwise accuracy "
            f"on this pair is chance, not a measurement")
    if not in_pos:
        raise ValueError(
            f"cue {cue!r} is absent from the positive {positive!r} of {pair_id}; "
            f"the anchor names the wrong member (the old STATUS_CUE bug)")


def anchor_phrase(np_: str, cue: str) -> str:
    """The subject phrase for the anchor, without saying the cue twice.

    The `attribute` flip's cue ("budget") is already inside its noun phrase, and
    inserting it produced "sur le budget budget du projet".
    """
    if not cue:
        return np_
    if cue.lower() in np_.lower():
        return np_
    return f"{np_} {cue}"


def build_hard_negatives(ledger: Ledger, seed: int = 42,
                         per_project: int = 6) -> list[HardNegative]:
    """Minimal pairs whose anchor carries the *discriminating cue*.

    A pair is only a test if the anchor can tell the two members apart. With a
    neutral anchor the two texts are equidistant by construction, so pairwise
    accuracy is pinned at 0.50 and a better embedder could not show it (measured
    on the first bench run: `anchor_pairwise_accuracy=0.5`, person=0.0).
    """
    rng = random.Random(subseed(seed, "hardneg", per_project))
    out: list[HardNegative] = []
    counter = 0
    for proj in ledger.projects:
        code = proj["project"]
        attrs = proj["attributes"]
        events = [e for e in ledger.events_by_project(code) if e.subject_key]
        for _ in range(per_project):
            counter += 1
            attr = attrs[counter % len(attrs)] if attrs else "budget"
            np_ = ATTRIBUTES.get(attr, {}).get("np", "le budget")
            matching = [e for e in events if e.subject_key and
                        e.subject_key.endswith("." + attr)]
            value = (matching[-1].value if matching and matching[-1].value else "240 k€")
            flip = rng.choice(["status", "status", "person", "date", "number",
                               "attribute", "repeat"])
            cue = ""
            #: ONE format dict for every template family. `np` already carries its
            #: article ("le budget"), so templates must use `{np_cap}` at sentence
            #: start — writing "Le {np}" produced "Le le budget" and the person
            #: template produced "envoyer le dossier le budget".
            fmt = {"np": np_, "np_cap": np_[:1].upper() + np_[1:],
                   "proj": code, "value": value}
            #: the cue travels WITH its templates now. It used to be looked up in a
            #: separate `STATUS_CUE` table, which had drifted: 27/78 status pairs
            #: named the negative, nothing, or both. `assert_discriminating` below
            #: makes that class of drift raise at build time instead of quietly
            #: turning the 6.4 metric into chance.
            if flip == "status":
                kind, cue, pos, neg = rng.choice(STATUS_FLIPS)
                pair = (pos.format(**fmt), neg.format(**fmt))
                meta = {"status_kind": kind}
            elif flip == "person":
                pair = tuple(t.format(**fmt) for t in PERSON_TPL)
                cue = PERSON_CUE
                meta = {"person": PERSON_CUE}
            elif flip == "date":
                pair = tuple(t.format(**fmt) for t in DATE_TPL)
                cue = DATE_CUE
                meta = {}
            elif flip == "number":
                pair = tuple(t.format(**fmt) for t in NUMBER_TPL)
                cue = NUMBER_CUE
                meta = {}
            elif flip == "attribute":
                pair = tuple(t.format(**fmt) for t in ATTRIBUTE_TPL)
                cue = ATTRIBUTE_CUE
                np_, meta = "le budget", {}
            else:
                pair = tuple(t.format(**fmt) for t in REPEAT_TPL)
                # both members assert the same value; the gold relation between
                # them is `supports`, so ranking one over the other is not a
                # measurable claim — excluded from accuracy downstream
                cue, meta = "", {"relation": "supports", "scoreable": False}
            pid = f"hn-{counter:04d}"
            assert_discriminating(cue, pair[0], pair[1], pid)
            phrase = anchor_phrase(np_, cue)
            anchor = (f"Qu'est-ce qui est dit sur {phrase} du projet {code} ?"
                      if cue else f"Qu'en est-il de {np_} du projet {code} ?")
            out.append(HardNegative(
                pair_id=pid, flip=flip, anchor=anchor, cue=cue,
                positive=pair[0], negative=pair[1], project=code,
                subject=f"{code.lower()}.{attr}", meta=meta))
    return out


def to_jsonl(pairs: list[HardNegative]) -> str:
    import json

    return "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in pairs)


def flip_counts(pairs: list[HardNegative]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in pairs:
        out[p.flip] = out.get(p.flip, 0) + 1
    return out
