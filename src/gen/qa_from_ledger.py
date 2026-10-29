"""Questions derived from the ledger (BUILD_GUIDE 7.1.4 `qa_from_ledger.py`).

Because the ledger authored the truth, a question's gold evidence is not
annotated — it is *looked up*: the event that answers it, then the units that
event was rendered into. Question types mirror 9.4: current decision, history,
who/when, compare, dossier, unanswerable.
"""

from __future__ import annotations

import random
from dataclasses import asdict, field
from datetime import date
from typing import Any

from engine.seeding import subseed
from gen.ledger import ATTRIBUTES, Ledger

#: surface attribute for a question ("le budget", "la date de lancement").
#: Tolerant of attributes outside ATTRIBUTES (a hand-written or real-corpus
#: subject key must degrade to a readable noun phrase, not raise).
def attr_np(attribute: str | None) -> str:
    if not attribute:
        return "le sujet"
    known = ATTRIBUTES.get(attribute)
    if known:
        return known["np"]
    return attribute.replace("_", " ")


def parse_subject(subject_key: str | None) -> tuple[str | None, str | None, bool]:
    """`<project>.owner.<attr>` / `<project>.<attr>` -> (project, attribute, is_owner).

    One parser for the whole repo: the ledger writes owner keys as
    `<project>.owner.<attr>` and a naive `split(".owner")[0]` silently yields
    "owner.risk_compliance", which then KeyErrors (measured on the first bench
    run). Both shapes are accepted here so no caller has to guess.
    """
    if not subject_key or "." not in subject_key:
        return None, None, False
    project, _, rest = subject_key.partition(".")
    is_owner = rest.startswith("owner.")
    attr = rest[len("owner."):] if is_owner else rest
    return project, (attr or None), is_owner


def base_attribute(subject_key: str | None) -> str | None:
    return parse_subject(subject_key)[1]


CURRENT_TMPL = [
    "Quelle décision a finalement été prise sur {np} du projet {proj} ?",
    "{np_cap} du projet {proj}, on en est où ?",
    "Où en est {np} pour le projet {proj} aujourd'hui ?",
    "Qu'est-ce qui est acté pour {np} du projet {proj} ?",
]
HISTORY_TMPL = [
    "Qu'est-ce qui avait été dit d'abord sur {np} du projet {proj}, avant le changement ?",
    "Historiquement, quelle valeur avait {np} pour {proj} ?",
    "Avant la dernière réunion, qu'est-ce qui était prévu pour {np} du projet {proj} ?",
]
OWNER_TMPL = [
    "Qui s'occupe de {np} sur le projet {proj} ?",
    "Qui est responsable de {np} dans le projet {proj} ?",
]
DATE_TMPL = [
    "C'est prévu pour quand {np} du projet {proj} ?",
    "{np_cap} du projet {proj}, on a retenu quelle date ?",
]
UNANSWERABLE_TMPL = [
    "Qui est le client final du projet {proj} ?",
    "Quel est le numéro de téléphone du prestataire de {proj} ?",
    "Quelle est la couleur du logo {proj} ?",
    "Combien de jours de congé restent-t-il à {person} ?",
]
COMPARE_TMPL = [
    "Est-ce que {right} contredit {left} sur le projet {proj} ?",
    "Qu'est-ce qui a changé entre {left} et {right} pour {proj} ?",
]
DOSSIER_TMPL = [
    "Prépare un dossier complet sur le projet {proj} : historique, décisions, budget, "
    "risques, prochaines étapes et personnes impliquées.",
]


def _fill(tmpl: str, **kw: Any) -> str:
    """Fill a question template, deriving the capitalised surface form.

    `str.format` cannot call methods, so "{np.capitalize()}" is a syntax error at
    runtime; templates use {np_cap} and the value is computed here.
    """
    if "np" in kw and "np_cap" not in kw:
        kw["np_cap"] = str(kw["np"]).capitalize()
    return tmpl.format(**kw)


class QASet:
    def __init__(self, items: list[dict[str, Any]], meta: dict[str, Any]):
        self.items = items
        self.meta = meta

    def by_type(self, *kinds: str) -> list[dict[str, Any]]:
        return [i for i in self.items if i["kind"] in kinds]

    def to_jsonl(self) -> str:
        import json

        return "\n".join(json.dumps(i, ensure_ascii=False) for i in self.items)


def build_questions(ledger: Ledger, *, seed: int = 42, per_project: int = 8,
                    include_unanswerable: bool = True) -> QASet:
    """Gold-bearing questions for every project in the ledger."""
    rng = random.Random(subseed(seed, "qa", per_project))
    # event_id -> the units it was rendered into (filled by the caller from the
    # store); here we keep event ids, which is what makes the gold exact.
    items: list[dict[str, Any]] = []
    for proj in ledger.projects:
        code = proj["project"]
        events = ledger.events_by_project(code)
        by_subject: dict[str, list] = {}
        for e in events:
            if e.subject_key:
                by_subject.setdefault(e.subject_key, []).append(e)
        subjects = sorted(by_subject)
        for s_i, subject in enumerate(subjects[:per_project]):
            # Owner keys are a different speech act. Asking "what is the value
            # of atlas.owner.launch_date?" and scoring the answer against "Sophie
            # owns it" is a malformed temporal question; the owner loop below
            # emits the proper "who owns it?" question.
            if parse_subject(subject)[2]:
                continue
            chain = sorted(by_subject[subject], key=lambda e: (e.on, e.event_id))
            _ = _terminal(chain, ledger)
            attr = base_attribute(subject)
            np_ = attr_np(attr)
            gold_current = [e.event_id for e in chain if not _is_superseded(e, ledger)]
            gold_all = [e.event_id for e in chain]
            if not gold_current:
                continue
            kind_word = "current_decision"
            q = _fill(rng.choice(CURRENT_TMPL), np=np_, proj=code)
            items.append({
                "qid": f"qa-{code}-{s_i}-cur", "kind": kind_word, "project": code,
                "question": q, "gold_event_ids": gold_current,
                "gold_superseded_event_ids": [e for e in gold_all
                                              if e not in gold_current],
                "gold_value": _.value, "subject": subject,
                "expected_current_status": _STATUS_NAME(_.kind),
            })
            if len(chain) > 1 and _.kind != "proposed":
                hist = [e.event_id for e in chain if _is_superseded(e, ledger)]
                if hist:
                    items.append({
                        "qid": f"qa-{code}-{s_i}-hist", "kind": "history",
                        "project": code,
                        "question": _fill(rng.choice(HISTORY_TMPL), np=np_, proj=code),
                        "gold_event_ids": hist, "gold_superseded_event_ids": [],
                        "gold_value": chain[0].value, "subject": subject,
                    })
            if _.kind in ("confirmed", "changed", "reinstated", "cancelled"):
                items.append({
                    "qid": f"qa-{code}-{s_i}-soft", "kind": "proposal_vs_decision",
                    "project": code,
                    "question": _fill(rng.choice(CURRENT_TMPL), np=np_, proj=code),
                    "gold_event_ids": gold_current,
                    "gold_superseded_event_ids": [e.event_id for e in chain
                                                  if e.kind in ("proposed", "hypothesis")],
                    "gold_value": _.value, "subject": subject,
                })
        # Gold must be the *in-force* assignment/date. Golding every matching
        # event made a correctly-superseded assignment count as a miss, which
        # penalised System B for doing its job (measured: owner gold_in_pool
        # A=0.857 vs B=0.429 before this fix).
        owners = [e for e in events if e.kind == "assignment" and e.person
                  and not _is_superseded(e, ledger)]
        for o_i, o in enumerate(owners[:3]):
            attr = o.role or "budget"
            items.append({
                "qid": f"qa-{code}-own-{o_i}", "kind": "owner", "project": code,
                "question": _fill(rng.choice(OWNER_TMPL), np=attr_np(attr), proj=code),
                "gold_event_ids": [o.event_id], "gold_superseded_event_ids": [],
                "gold_value": o.person, "subject": o.subject_key,
            })
        dates = [e for e in chain_like_date_events(ledger, code)]
        for d_i, e in enumerate(dates[:2]):
            attr = base_attribute(e.subject_key) or "launch_date"
            items.append({
                "qid": f"qa-{code}-date-{d_i}", "kind": "date", "project": code,
                "question": _fill(rng.choice(DATE_TMPL), np=attr_np(attr), proj=code),
                "gold_event_ids": [e.event_id], "gold_superseded_event_ids": [],
                "gold_value": e.value, "subject": e.subject_key,
            })
        if include_unanswerable:
            person = (proj["people"] or ["Marie"])[0]
            items.append({
                "qid": f"qa-{code}-na", "kind": "unanswerable", "project": code,
                "question": _fill(rng.choice(UNANSWERABLE_TMPL), proj=code, person=person),
                "gold_event_ids": [], "gold_superseded_event_ids": [], "gold_value": None,
            })
        mids = [m.meeting_id for m in ledger.meetings if m.project == code]
        if len(mids) >= 2:
            items.append({
                "qid": f"qa-{code}-cmp", "kind": "compare", "project": code,
                "question": _fill(rng.choice(COMPARE_TMPL), left=mids[0], right=mids[-1],
                                                              proj=code),
                "gold_event_ids": [], "gold_superseded_event_ids": [],
                "left_source": mids[0], "right_source": mids[-1],
            })
        items.append({
            "qid": f"qa-{code}-oss", "kind": "build_dossier", "project": code,
            "question": _fill(DOSSIER_TMPL[0], proj=code),
            "gold_event_ids": [e.event_id for e in events if e.kind in
                               ("confirmed", "changed", "cancelled", "reinstated")],
            "gold_superseded_event_ids": [],
        })
    meta = {"generator": "qa_from_ledger v0", "seed": seed,
            "n_questions": len(items), "n_projects": len(ledger.projects),
            "kinds": _tally(items)}
    return QASet(items, meta)


def chain_like_date_events(ledger: Ledger, code: str):
    for e in ledger.events_by_project(code):
        if e.kind in ("confirmed", "changed", "reinstated") and e.subject_key \
                and ATTRIBUTES.get(base_attribute(e.subject_key) or "", {}).get("kind") == "date" \
                and not _is_superseded(e, ledger):
            yield e


def _terminal(chain, ledger):
    """The in-force event of a subject chain (latest, not superseded)."""
    alive = [e for e in chain if not _is_superseded(e, ledger)]
    return (alive or chain)[-1]


def _is_superseded(e, ledger) -> bool:
    return any(o.supersedes == e.event_id for o in ledger.events)


def _STATUS_NAME(kind: str) -> str:
    from gen.render import _STATUS_MAP

    return _STATUS_MAP.get(kind, "mention")


def _tally(items: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i["kind"]] = out.get(i["kind"], 0) + 1
    return out
