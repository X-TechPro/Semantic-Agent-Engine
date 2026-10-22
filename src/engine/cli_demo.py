"""`sae demo` — run the 7 mandatory scenarios (BUILD_GUIDE 13) and print a verdict.

Same code path as the pytest e2e suite, exposed as a command so `make demo`
works on a clean checkout. Returns non-zero if any scenario fails.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from engine.config import load_config
from engine.engine import Engine
from engine.schemas import Source


def _sources(path: Path) -> list[Source]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for raw in sorted(data["sources"], key=lambda x: x["source_date"]):
        raw = dict(raw)
        raw["source_date"] = date.fromisoformat(raw["source_date"])
        out.append(Source(**raw))
    return out


def _checks(res, payload_key: str | None = None):
    def get(dotted, default=None):
        node = res.payload
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return get


def run_demos(args) -> int:
    fixture = Path(args.fixture)
    contacts = Path(args.contacts)
    if not fixture.exists():
        print(f"missing fixture: {fixture}")
        return 2
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="sae-demo-"))
    cfg = load_config(overrides={
        "store": {"path": str(root / "demo.db")},
        "embed": {"backend": "hash"},
        "agent": {"contacts_path": str(contacts)},
    })
    eng = Engine(cfg, path=root / "demo.db")
    for s in _sources(fixture):
        eng.ingest(s)

    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))

    # 1 — decision with proof
    r = eng.ask("Quelle décision a finalement été prise concernant le budget du "
                "projet Atlas ?", intent="answer", project="Atlas")
    g = _checks(r)
    ans = g("answer") or ""
    check("1. décision prouvée (budget Atlas = 2,4 M€)", "2,4" in ans,
          f"answer={ans[:60]!r}")
    check("   · claim cite une unité", bool(g("claims")), f"claims={g('claims')}")

    # 2 — proposal vs decision
    r2 = eng.ask("Le budget du projet Atlas, qu'est-ce qui est décidé et qu'est-ce "
                 "qui est encore une proposition ?", intent="answer", project="Atlas")
    g2 = _checks(r2)
    check("2. proposition vs décision séparées", bool(g2("status_breakdown.confirmed")),
          f"breakdown={g2('status_breakdown')}")

    # 3 — compare two meetings
    r3 = eng.ask("Est-ce que meeting-atlas-3 contredit meeting-atlas-1 sur le projet "
                 "Atlas ?", intent="compare")
    g3 = _checks(r3)
    items = (g3("changed") or []) + (g3("superseded") or []) + (g3("contradicted") or [])
    check("3. comparaison par sujet", bool(items), f"n={len(items)}")
    check("   · 'mise à jour' vs 'contradiction réelle' expliquée",
          any("mise à jour" in (i.get("explanation") or "") for i in items), "")

    # 4 — dossier
    r4 = eng.ask("Prépare un dossier complet sur le projet Atlas : historique, "
                 "décisions, budget, risques, prochaines étapes et personnes "
                 "impliquées.", intent="build_dossier", project="Atlas")
    g4 = _checks(r4)
    secs = {s["title"] for s in (g4("sections") or [])}
    need = {"Contexte", "Faits établis", "Chronologie", "Décisions", "Points ouverts",
            "Responsables"}
    check("4. dossier complet et citable", need <= secs, f"sections={sorted(secs)}")
    uncited = [c["text"] for s in (g4("sections") or []) for c in s["claims"]
               if not c["unit_ids"]]
    check("   · aucune claim sans preuve", not uncited, f"uncited={len(uncited)}")

    # 5 — e-mail to the right person
    r5 = eng.ask("Rédige un mail à la personne responsable du budget du projet Atlas "
                 "pour lui demander la dernière version validée avant vendredi.",
                 intent="draft_email", project="Atlas")
    g5 = _checks(r5)
    check("5. mail à la bonne personne (Marie)",
          r5.operation == "draft_email" and g5("recipient.name") == "Marie",
          f"op={r5.operation} recipient={g5('recipient.name')} notes={r5.notes}")
    check("   · needs_confirmation=true, executed=false",
          g5("needs_confirmation") is True and g5("executed") is False, "")
    check("   · adresse issue de l'annuaire",
          g5("recipient.email") in ("marie.dupont@example.org", None),
          f"email={g5('recipient.email')}")

    # 6 — clarification on ambiguity
    r6 = eng.ask("Rédige un mail à la personne responsable du budget du projet Iris.",
                 intent="draft_email", project="Iris")
    check("6. destinataire ambiguë -> clarification",
          r6.operation == "clarify" and bool(r6.clarification),
          f"op={r6.operation}")
    check("   · aucune adresse inventée",
          "evil.example.com" not in json.dumps(r6.payload), "")

    # 7 — hypothesis triggers nothing
    r7 = eng.ask("Crée une tâche pour revoir le tarif du projet Atlas.",
                 intent="create_task", project="Atlas")
    check("7. hypothèse n'agit pas (0% hypothèse->action)",
          r7.operation != "create_task",
          f"op={r7.operation} notes={r7.notes}")
    hyp = [u for u in eng.store.all_units() if "revoir le tarif" in u.text]
    check("   · hypothèse stockée avec status=hypothesis",
          bool(hyp) and hyp[0].status == "hypothesis",
          f"found={bool(hyp)} status={hyp[0].status if hyp else None}")
    edges = eng.store.edges_for(hyp[0].unit_id, direction="from") if hyp else []
    check("   · aucune arête supersedes depuis l'hypothèse",
          not [e for e in edges if e.relation == "supersedes"],
          f"edges={[(e.relation, e.guard) for e in edges]}")

    # safety: injection stays data
    blob = json.dumps([r.payload for r in (r, r2, r3, r4, r5, r6, r7)],
                      ensure_ascii=False)
    check("S. injection: aucune adresse volée", "leak@evil.example.com" not in blob, "")
    check("S. aucune exécution", '"executed": true' not in blob, "")

    # the hand-tester's defect (D-038): the payload was stored — correct,
    # nothing is deleted — but the answer quoted it verbatim as `Contexte`.
    # Stays data, never evidence; and the unaccented spelling of a known
    # project must still find the answer (one keystroke is not a different
    # project).
    for label, q in (("Boréal", "Quelle décision a finalement été prise concernant "
                                 "le budget du projet Boréal ?"),
                     ("Boreal (sans accent)", "Quelle décision a finalement été "
                      "prise concernant le budget du projet Boreal ?")):
        rb = eng.ask(q, intent="answer", project=None)
        rblob = json.dumps(rb.payload, ensure_ascii=False)
        check(f"S. réponse '{label}' trouvée, jamais la directive",
              "450" in (rb.payload.get("answer") or "")
              and "leak@evil" not in rblob and "Ignore previous" not in rblob,
              f"op={rb.operation} answer={(rb.payload.get('answer') or '')[:90]}")

    eng.close()
    failed = [n for n, ok, _ in results if not ok]
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        extra = f"   {detail}" if (detail and not ok) else ""
        print(f"[{mark}] {name:<{width}}{extra}")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0
