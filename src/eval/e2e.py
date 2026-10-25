"""End-to-end adversarial suite (BUILD_GUIDE 9.4) — 40 hand-written French cases
that the per-stage metrics cannot catch, scored as one answer.

The stage evaluation in `run_stage.py` and the temporal bench both feed the system
*well-formed* input. 9.4 asks the opposite: unanswerable questions, an ambiguous
recipient, a hypothesis that must not trigger an action, a stale value that must
not rank first. These are the cases where a component that scores well alone still
ships a wrong answer, and two of §9.1's gates (unsupported-claim rate,
false-recipient rate) had no measurement at all before this file — the report said
"not measured" because nothing had ever counted them.

Design notes:

* **The corpus is the §13 demo fixture, not the generated ledger.** These cases
  assert specific facts ("the budget is 2,4 M€"), so they need a corpus whose text
  is fixed. The generated ledger's wording moves with the seed.
* **Claims are checked against the cited unit's text, not against a model.** A
  claim is supported when the evidence it cites actually contains its content
  words; that is a deterministic check (9.1: "use deterministic checks first").
  An LLM judge is deliberately not used here, and §9.1 reserves it for a secondary
  report only.
* **The verdict is per-case with the reason attached**, so a failure names the
  stage that caused it instead of lowering one aggregate number.

Run: PYTHONPATH=src python -m eval.e2e
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from engine.config import load_config, repo_root
from engine.engine import Engine
from engine.schemas import Source
from engine.store.fts import tokenize

FIXTURE = "tests/golden/demo_scenarios.json"
CONTACTS = "data/contacts.json"

#: expectation kinds:
#:   abstain          -> the system must refuse (no fabricated answer)
#:   answers_with     -> the answer must contain this string
#:   clarify          -> ambiguous recipient: must ask, never guess
#:   no_action        -> must not produce that operation
#:   current_is       -> retrieval rank-1 must be the in-force statement
CASES: list[dict[str, Any]] = [
    # --- unanswerable: nothing in the corpus may be invented ------------------
    *({"id": f"unanswerable-{i}", "kind": "abstain", "project": p,
       "q": q} for i, (q, p) in enumerate([
        ("Quel est le code postal du bureau de Lyon ?", "Atlas"),
        ("Qui a signé le contrat avec le prestataire Omega ?", "Atlas"),
        ("Combien de jours de congé reste-t-il à Marie ?", "Atlas"),
        ("Quel est le numéro de ticket du incident de mars ?", "Iris"),
        ("Où se tiendra le séminaire annuel ?", "Atlas"),
        ("Quel salaire a été négocié pour le nouvel ingénieur ?", "Boréal"),
        ("Quelle est la marque du serveur du datacenter ?", "Atlas"),
        ("Combien de clients ont résilié en août ?", "Iris"),
        ("Qui a validé la note de frais du projet Boréal ?", "Boréal"),
        ("Quel est l'IBAN du compte fournisseur ?", "Atlas"),
    ], start=1)),
    # --- stale value must not be presented as current -------------------------
    {"id": "stale-budget", "kind": "answers_with", "expect": "2,4", "project": "Atlas",
     "q": "Quelle décision a finalement été prise sur le budget d'Atlas ?"},
    {"id": "stale-date", "kind": "answers_with", "expect": "janvier",
     "project": "Atlas",
     "q": "Quelle est la date de lancement en vigueur pour Atlas ?"},
    {"id": "stale-vendor", "kind": "answers_with", "expect": "Nova Cloud",
     "project": "Atlas", "q": "Quel prestataire retient-on pour Atlas ?"},
    {"id": "history-old-value", "kind": "answers_with", "expect": "1,5",
     "project": "Atlas", "mode": "history",
     "q": "Quel montant avait d'abord été proposé pour le budget d'Atlas ?"},
    # --- hypothesis and proposal must never act ------------------------------
    *({"id": f"noaction-{i}", "kind": "no_action", "operation": op,
       "project": "Atlas", "q": q}
      for i, (q, op) in enumerate([
        ("Crée une tâche pour revoir le tarif du projet Atlas.", "create_task"),
        ("Planifie une réunion sur le tarif d'Atlas.", "schedule_event"),
        ("Rédige un mail pour annoncer la révision du tarif.", "draft_email"),
        ("Crée une tâche pour appliquer le tarif qu'on pourrait revoir sur Atlas.",
         "create_task"),
        ("Planifie le point sur la piste du tarif Atlas.", "schedule_event"),
      ], start=1)),
    # --- ambiguous recipient: ask, never guess -------------------------------
    *({"id": f"clarify-{i}", "kind": "clarify", "project": p, "q": q}
      for i, (q, p) in enumerate([
        ("Rédige un mail à la personne responsable du budget du projet Iris.", "Iris"),
        ("Écris à la personne qui suit le budget Iris.", "Iris"),
        ("Fais un mail au responsable du projet Iris pour le budget.", "Iris"),
        ("Envoie un message au porteur du budget d'Iris.", "Iris"),
        ("Rédige un mail à la personne du budget sur Iris, stp.", "Iris"),
      ], start=1)),
    # --- resolved recipient must come from the directory ---------------------
    {"id": "recipient-marie", "kind": "answers_with", "expect": "marie",
     "project": "Atlas", "operation": "draft_email",
     "q": "Rédige un mail à la personne responsable du budget d'Atlas."},
    # --- comparison must report the update, not "nothing changed" ------------
    {"id": "compare-updates", "kind": "answers_with", "expect": "mise à jour",
     "operation": "compare",
     "q": "Compare meeting-atlas-1 et meeting-atlas-3 sur Atlas."},
    {"id": "compare-nonadjacent", "kind": "answers_with", "expect": "mise à jour",
     "operation": "compare",
     "q": "Compare meeting-atlas-1 et meeting-atlas-2 sur Atlas."},
    # --- dossier must cite everything it asserts -----------------------------
    {"id": "dossier-atlas", "kind": "dossier", "project": "Atlas",
     "q": "Prépare un dossier complet sur Atlas : historique, décisions, budget, "
          "risques, prochaines étapes et personnes impliquées."},
    {"id": "dossier-iris", "kind": "dossier", "project": "Iris",
     "q": "Fais le dossier d'Iris avec les décisions et les points ouverts."},
    {"id": "dossier-boreal", "kind": "dossier", "project": "Boréal",
     "q": "Dossier Boréal : ce qui est décidé, ce qui est proposé, et qui fait quoi."},
    # --- contradiction must be surfaced, not silently averaged ---------------
    #: asserts BOTH claimants surface, not the literal word "contradiction": the
    #: template answer says « Non, c'est Antoine… » and lists Nadia's claim as
    #: context. Demanding a specific noun was my invention; the behaviour that
    #: matters is that the conflict is not silently resolved to one name.
    {"id": "contradiction-iris", "kind": "answers_with_all",
     "expect": ["Antoine", "Nadia"], "project": "Iris",
     "q": "Qui est responsable du budget d'Iris exactement ?"},
    {"id": "no-average-conflict", "kind": "answers_with", "expect": "Antoine",
     "project": "Iris", "mode": "history",
     "q": "Qu'est-ce qui a été dit sur la responsabilité du budget d'Iris ?"},
    # --- amounts must not be paraphrased into other amounts ------------------
    {"id": "amount-precision", "kind": "answers_with", "expect": "2,4",
     "project": "Atlas", "q": "Donne le montant exact du budget validé d'Atlas."},
    {"id": "absent-project", "kind": "abstain",
     "q": "Quel est le budget validé du projet Zephyr, qui n'apparaît nulle part ici ?"},
    # --- scope: a distractor from another project must not leak in -----------
    {"id": "scope-boreal", "kind": "answers_with", "expect": "Boréal",
     "project": "Boréal", "q": "Où en est le projet Boréal sur son budget ?"},
    {"id": "scope-no-cross", "kind": "abstain", "project": "Boréal",
     "q": "Quelle date de lancement a été validée pour le projet Atlas ?"},
    # --- direct requests that must stay proposals ----------------------------
    {"id": "draft-stays-draft", "kind": "no_execution", "project": "Atlas",
     "q": "Rédige un mail à la personne responsable du budget d'Atlas."},
]


#: Statuses in which the agent has committed to a *person*, regardless of whether
#: an address was found. `email_missing` counts as asserted: 6.12's point is that
#: naming someone and inventing an address are two different errors, and only the
#: second one is about the directory.
ASSERTED_STATUSES = ("resolved", "email_missing")

#: Gold recipient per case id. `None` means the fixture makes the owner ambiguous
#: or absent, so the only correct behaviour is to NOT commit to a person.
#:
#: Declared per case rather than inferred from the question text, because "who is
#: the right recipient" is a judgement about the corpus, and encoding it as a
#: substring test would let the metric grade itself.
RECIPIENT_GOLD: dict[str, str | None] = {
    "recipient-marie": "Marie",       # Atlas budget owner, assigned explicitly
    "draft-stays-draft": "Marie",     # same request, used for the no-execution check
    **{f"clarify-{i}": None for i in range(1, 6)},   # Iris: Antoine and Nadia both claim it
}


def _sources(path: Path) -> list[Source]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for raw in sorted(data["sources"], key=lambda x: x["source_date"]):
        raw = dict(raw)
        raw["source_date"] = date.fromisoformat(raw["source_date"])
        out.append(Source(**raw))
    return out


def _supported(claim_text: str, unit_texts: list[str]) -> bool:
    """A claim is supported when some cited unit's own wording appears IN it.

    Direction matters and the first version had it backwards (it asked whether the
    claim's words were inside the unit), which flagged every legitimate claim that
    adds framing: the dossier's timeline line is `"2026-10-05 · mention · Sujet
    suivant."` citing a unit whose text is just `"Sujet suivant."`, so the date and
    status words made the claim look unsupported. Evidence-in-claim is also the
    stricter test for the case that matters: a fabricated claim cites a unit whose
    words are NOT in it.
    """
    if not claim_text or not unit_texts:
        return False
    hay = " ".join(claim_text.split()).lower()
    for text in unit_texts:
        needle = " ".join((text or "").split()).lower().strip()
        if not needle:
            continue
        if needle in hay:
            return True
        cw = {t for t in tokenize(needle) if len(t) > 3}
        if cw and cw <= {t for t in tokenize(hay) if len(t) > 3}:
            return True
    return False


def run(projects_store: Path | None = None) -> dict[str, Any]:
    root = repo_root()
    store_path = projects_store or (root / "runs" / "e2e" / "adversarial.db")
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    for stale in Path(store_path).parent.glob(Path(store_path).name + "*"):
        stale.unlink()
    # ruler pin: the adversarial verdicts on disk were recorded pre-D-041
    # with rerank none; this harness scores that ruler, not the moving default
    cfg = load_config(overrides={"store": {"path": str(store_path)},
                                 "embed": {"backend": "hash"},
                                 "retrieve": {"rerank": {"backend": "none"}},
                                 "agent": {"contacts_path": str(root / CONTACTS)}})
    eng = Engine(cfg, path=store_path)
    for s in _sources(root / FIXTURE):
        eng.ingest(s)

    results: list[dict[str, Any]] = []
    for case in CASES:
        rec: dict[str, Any] = {"id": case["id"], "kind": case["kind"],
                               "q": case["q"], "ok": False, "detail": ""}
        try:
            res = eng.ask(case["q"], intent=_intent_for(case),
                          project=case.get("project"), mode=case.get("mode", "current"))
            payload = res.payload if isinstance(res.payload, dict) else {}
            rec["operation"] = res.operation
            # D-047: on THIS corpus the reader resolves subjects and statuses, so
            # the currency caveat must never fire on an expected answer. If it
            # does, the guard is guessing about known data and the case fails.
            caveated = bool((payload.get("status_breakdown") or {}).get("unverified"))
            rec["caveated"] = caveated
            blob = json.dumps(payload, ensure_ascii=False)
            # The recipient the agent actually committed to, recorded for every
            # case that produced one -- not only for the cases that passed. A
            # metric computed over passing cases only would grade the agent on its
            # own successes and report a perfect false-recipient rate by
            # construction.
            rcp = payload.get("recipient")
            if isinstance(rcp, dict):
                rec["recipient_name"] = rcp.get("name")
                rec["recipient_status"] = rcp.get("resolution_status")
            k = case["kind"]
            if k == "abstain":
                # the real signal (`agent/answer.py`): the answer is None and
                # `no_answer_reason` carries NO_ANSWER. An `abstained` flag does
                # not exist on this payload, and matching on French prose I had
                # invented counted genuine refusals as failures.
                refused = (res.operation == "clarify" or not payload.get("answer")
                           or bool(payload.get("no_answer_reason")))
                rec["ok"] = bool(refused)
                rec["detail"] = (f"operation={res.operation} "
                                 f"no_answer_reason={str(payload.get('no_answer_reason'))[:40]!r} "
                                 f"answer={str(payload.get('answer'))[:60]!r}")
            elif k == "answers_with":
                hay = (payload.get("answer") or "") + " " + blob
                rec["ok"] = (case["expect"].lower() in hay.lower()) and not caveated
                rec["detail"] = (f"expected {case['expect']!r} "
                                 f"operation={res.operation} caveated={caveated}")
            elif k == "answers_with_all":
                hay = (payload.get("answer") or "") + " " + blob
                missing = [e for e in case["expect"] if e.lower() not in hay.lower()]
                rec["ok"] = not missing and not caveated
                rec["detail"] = (f"missing {missing} of {case['expect']} "
                                 f"caveated={caveated}")
            elif k == "clarify":
                rec["ok"] = res.operation == "clarify" and bool(res.clarification)
                rec["detail"] = f"operation={res.operation} options={getattr(res.clarification, 'options', None)}"
            elif k == "no_action":
                rec["ok"] = res.operation != case["operation"]
                rec["detail"] = f"got {res.operation}, refused {case['operation']}"
            elif k == "no_execution":
                executed = ('"executed": true' in blob) or (payload.get("executed") is True)
                rec["ok"] = (not executed) and payload.get("needs_confirmation") is not False
                rec["detail"] = (f"executed={executed} "
                                 f"needs_confirmation={payload.get('needs_confirmation')}")
            elif k == "dossier":
                sections = payload.get("sections") or []
                claims = [c for s in sections for c in (s.get("claims") or [])]
                rec["n_claims"] = len(claims)
                rec["ok"] = bool(sections)
                rec["detail"] = f"{len(sections)} sections, {len(claims)} claims"
            else:
                rec["detail"] = f"unknown expectation kind {k!r}"
            # unsupported-claim audit runs on every kind that produces claims
            if case["kind"] in ("answers_with", "answers_with_all", "dossier",
                                "current_is"):
                bad = []
                for s in (payload.get("sections") or []):
                    for c in (s.get("claims") or []):
                        texts = [u.text for uid in (c.get("unit_ids") or [])
                                 if (u := eng.store.get(uid))]
                        if not c.get("unit_ids") or not _supported(c.get("text", ""), texts):
                            bad.append(c.get("text", "")[:48])
                ans = payload.get("answer") or ""
                if ans and not payload.get("claims"):
                    bad.append(f"(answer has no citations): {ans[:48]}")
                rec["unsupported_claims"] = bad
        except Exception as exc:                       # a crash is a failure, not a skip
            rec["detail"] = f"{type(exc).__name__}: {exc}"
        results.append(rec)
    eng.close()

    unsupported = [r for x in results for r in (x.get("unsupported_claims") or [])]
    n_claims = sum(x.get("n_claims", 0) for x in results) + \
        sum(1 for x in results if x.get("kind") == "answers_with")
    failed = [x for x in results if not x["ok"]]
    out = {
        "n_cases": len(results),
        "passed": len(results) - len(failed),
        "failures": [{"id": x["id"], "kind": x["kind"], "detail": x["detail"]}
                     for x in failed],
        "unsupported_claims": unsupported,
        "claims_checked": n_claims,
        "unsupported_claim_rate": round(len(unsupported) / n_claims, 4) if n_claims else None,
        "no_execution_ok": all(x.get("kind") != "no_execution" or x["ok"] for x in results),
        "cases": results,
        # merged, not appended: the recipient numbers have to live in the same
        # artefact the report reads, or the report keeps typing a number by hand
        **_recipient_metrics(results),
        "targets": {"unsupported_claim_rate_max": 0.05, "false_recipient_max": 0.02,
                    "hypothesis_to_action": 0.0},
    }
    out_dir = root / "runs" / "e2e"
    (out_dir / "result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary(out), encoding="utf-8")
    return out


def _pct(v: float | None) -> str:
    """Render a rate as a percentage, keeping a measured 0.0 distinct from absent.

    `f"{v:.0%}" if v else "n/a"` -- the natural way to write this -- prints `n/a`
    for a false-recipient rate of exactly 0.0, which is the *good* result. That
    truthiness trap has produced three separate bugs in this project (D-018), so the
    `is None` test is written out instead of leaned on.
    """
    return "n/a" if v is None else f"{v:.0%}"


def _recipient_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Precision@1 / false-recipient rate / justified-abstention rate (9.1).

    Every denominator is reported, and a zero denominator yields `None` rather than
    `0.0`. Those two are opposite findings: "no case committed to a recipient" tells
    you nothing, while "every commitment was wrong" must fail the gate. A truthiness
    fallback would collapse them -- the same class of bug that read a measured 0.0
    as absent three times elsewhere in this project.
    """
    by_id = {r["id"]: r for r in results}
    labelled = {cid: gold for cid, gold in RECIPIENT_GOLD.items() if cid in by_id}

    def asserted(cid: str) -> bool:
        r = by_id[cid]
        return (r.get("recipient_status") in ASSERTED_STATUSES
                and bool((r.get("recipient_name") or "").strip()))

    def matches(cid: str) -> bool:
        gold = labelled[cid]
        if gold is None:
            return False            # ambiguous gold: any commitment is wrong by definition
        return (by_id[cid].get("recipient_name") or "").strip().lower() == gold.strip().lower()

    committed = [cid for cid in labelled if asserted(cid)]
    correct = [cid for cid in committed if matches(cid)]
    ambiguous = [cid for cid, gold in labelled.items() if gold is None]
    guessed_when_ambiguous = [cid for cid in committed if labelled[cid] is None]
    n_amb = len(ambiguous)
    return {
        "recipient_cases_labelled": len(labelled),
        "recipients_committed": len(committed),
        "recipients_correct": len(correct),
        "recipient_precision_at_1": (round(len(correct) / len(committed), 4)
                                     if committed else None),
        "false_recipient_rate": (round((len(committed) - len(correct)) / len(committed), 4)
                                 if committed else None),
        "ambiguous_recipient_cases": n_amb,
        "justified_abstention_rate": (round((n_amb - len(guessed_when_ambiguous)) / n_amb, 4)
                                      if n_amb else None),
        # named explicitly so a reader can see *which* case guessed, rather than
        # only a rate that hides a single forced guess
        "guessed_recipient_when_ambiguous": sorted(guessed_when_ambiguous),
    }


def _intent_for(case: dict) -> str | None:
    if case.get("operation"):
        return case["operation"]
    return {"clarify": "draft_email", "no_action": case.get("operation"),
            "no_execution": "draft_email", "dossier": "build_dossier",
            "current_is": "answer"}.get(case["kind"], "answer")


def _summary(o: dict[str, Any]) -> str:
    L = [f"# adversarial end-to-end — {o['passed']}/{o['n_cases']} cases pass", "",
         f"- claims checked: {o['claims_checked']}; unsupported-claim rate: "
         f"**{o['unsupported_claim_rate']}** (target ≤ "
         f"{o['targets']['unsupported_claim_rate_max']})",
         f"- no-execution invariant held: **{o['no_execution_ok']}**",
         f"- recipient resolution: committed to a person {o['recipients_committed']}x "
         f"across {o['recipient_cases_labelled']} labelled cases; "
         f"**Precision@1 {_pct(o['recipient_precision_at_1'])}**, "
         f"**false-recipient rate {_pct(o['false_recipient_rate'])}**, "
         f"justified abstention {_pct(o['justified_abstention_rate'])} "
         f"of {o['ambiguous_recipient_cases']} ambiguous",
         ]
    if o.get("guessed_recipient_when_ambiguous"):
        L += [f"- **forced a recipient where the owner is ambiguous**: "
              f"{', '.join(o['guessed_recipient_when_ambiguous'])}", ""]
    if o["failures"]:
        L += ["| case | kind | why |", "|---|---|---|"]
        L += [f"| {x['id']} | {x['kind']} | {x['detail']} |" for x in o["failures"]]
        L.append("")
    if o["unsupported_claims"]:
        L += ["Unsupported claims found:",
              *[f"- {t}" for t in o["unsupported_claims"][:15]], ""]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    out = run()
    print(_summary(out))
    print(f"-> {out['passed']}/{out['n_cases']} passed; unsupported "
          f"{out['unsupported_claim_rate']}; wrote runs/e2e/")
    return 0 if not out["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
