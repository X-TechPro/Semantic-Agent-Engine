"""`sae demo-real` — the 7 mandatory scenarios run on REAL French meetings.

The synthetic `demo` proves the architecture on data our own generator wrote;
the assignment explicitly forbids measuring only that (and B-13 showed how it
fails: the rules reader called 994/1,019 real units `mention`). This harness
re-runs the same seven behaviours against real transcribed meetings (SUMM-RE,
licence-verified) with the honesty the measured gaps require:

* subject keys come from the hand-labelled drawer gold — the resolver is named
  `gold` in every read-back, `subject_source: "gold-annotation"` is printed in
  the artefact header, and the demo never pretends extraction worked (B-13
  stays open in the artefact's own words);
* the reader is `models/reader_v2` — the checkpoint that fires assertive
  statuses on real French (510/781 measured vs rules' 9);
* retrieval, guards, linking, dossier, recipient logic and the router all run
  LIVE on the real transcript;
* no check reads a value expected from the transcript — the assertions are
  structural (claims cited, breakdown present, edges exist, recipients never
  invented, hypothesis inert), so the verdict cannot be tuned to the data.

Exit non-zero if any scenario fails, exactly like the synthetic demo. The
JSON written to `runs/demo_real/` is the deliverable; REPORT.md §6f renders it.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path

from engine.config import load_config, repo_root
from engine.engine import Engine

MEETINGS_DEFAULT = "024c_EBPH,047a_EDRH,012c_EBPZ"

QUERIES = {
    "tarif": ("Quel tarif de billetterie a été retenu pour l'événement ?",
              "047a_EDRH", "answer"),
    "breakdown": ("Qu'est-ce qui est décidé et qu'est-ce qui est encore une "
                  "proposition pour la billetterie ?", "047a_EDRH", "answer"),
    "dossier": ("Prépare un dossier complet sur cette réunion : historique, "
                "décisions, points ouverts et personnes impliquées.",
                "024c_EBPH", "build_dossier"),
    "email_owner": ("Rédige un mail à la personne qui s'occupe de la "
                    "billetterie pour avoir la dernière version validée.",
                    "047a_EDRH", "draft_email"),
    "email_ambiguous": ("Envoie un mail à la personne responsable du "
                        "nettoyage du site.", "047a_EDRH", "draft_email"),
    "hypo_task": ("Si on annulait la buvette, il faudrait créer une tâche de "
                  "remboursement, non ?", "047a_EDRH", "create_task"),
}


def _get(payload, dotted, default=None):
    node = payload or {}
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def run_real_demos(args) -> int:
    from eval.frbench import build_meeting_source

    root = repo_root()
    meetings = json.loads((root / "data/external/summre_test/meetings.json")
                          .read_text(encoding="utf-8"))
    mids = [m for m in args.meetings.split(",") if m]
    tmp = Path(tempfile.mkdtemp(prefix="sae-demo-real-"))
    cfg = load_config(overrides={
        "store": {"path": str(tmp / "demo.db")},
        "embed": {"backend": "sentence_transformers",
                  "model_id": "intfloat/multilingual-e5-base",
                  "device": args.device},
        "reader": {"backend": "model", "model_dir": str(root / args.reader_model)},
        "slots": {"subject_resolver": "gold",
                  "subject_model_dir": str(root / "data/gold")},
        "link": {"backend": "rules"},
        "agent": {"contacts_path": str(args.contacts),
                  **({"llm": {"backend": "ollama",
                              "model_id": args.compose_model},
                      "llm_answer": True}
                      if getattr(args, "compose", False) else {})},
    })
    eng = Engine(cfg, path=tmp / "demo.db")
    stats = {"subject_source": "gold-annotation (B-13 open: no extractor is "
                               "trusted on real speech; keys come from the "
                               "hand-labelled drawer gold)"}

    for mid in mids:
        segs = meetings[mid]["segments"] if isinstance(meetings[mid], dict) \
            else meetings[mid]
        eng.ingest(build_meeting_source(mid, segs, on_date=date(2021, 10, 5)))

    units = eng.store.all_units()
    by_mid = {}
    for u in units:
        by_mid.setdefault(u.project or "?", []).append(u)
    stats["n_units"] = len(units)
    stats["per_meeting"] = {
        m: {"units": len(us),
            "subjects_resolved": sum(1 for u in us if u.slots.subject),
            "assertive": sum(1 for u in us
                             if str(u.status) in ("confirmed", "cancelled"))}
        for m, us in by_mid.items()}
    sup_edges = eng.store.db.execute(
        "SELECT from_id, to_id FROM edges WHERE relation='supersedes'").fetchall()
    hyp_edges = eng.store.db.execute(
        """SELECT e.from_id FROM edges e JOIN units u ON u.unit_id = e.from_id
           WHERE e.relation='supersedes' AND u.status='hypothesis'""").fetchall()

    results: list[tuple[str, bool, str]] = []

    def check(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    # 1 — décision prouvée: an answer about the real ticket pricing, every
    # claim citing real units
    q, proj, intent = QUERIES["tarif"]
    r1 = eng.ask(q, intent=intent, project=proj)
    claims = _get(r1.payload, "claims") or []
    cited = [c for c in claims if c.get("unit_ids")]
    check("1. décision prouvée sur meeting réel",
          bool(_get(r1.payload, "answer")) and bool(claims) and len(cited) == len(claims),
          f"claims={len(claims)} cited={len(cited)}")

    # 2 — proposition vs décision séparées by real statuses
    q, proj, intent = QUERIES["breakdown"]
    r2 = eng.ask(q, intent=intent, project=proj)
    brk = _get(r2.payload, "status_breakdown") or {}
    check("2. statut breakdown sur statuts réels",
          bool(brk.get("confirmed")),
          f"breakdown={ {k: len(v) for k, v in brk.items()} }")

    # 3 — the temporal core on real speech: at least one old unit actually
    # stamped superseded (guards authorised it, not the annotation). Edge
    # direction from the linker: from_id is the NEW unit, to_id the stamped
    # OLD one (mark_superseded(e.to_id, e.from_id, ...) in ingest._link).
    old_ids = [e["to_id"] for e in sup_edges]
    stamped = [u.unit_id for u in units if u.superseded_by]
    check("3. supersede réel: edge(s) gardes-autorisés sur français brut",
          len(sup_edges) >= 1 and len(stamped) >= 1,
          f"edges={len(sup_edges)} stamped_units={len(stamped)}")
    check("   · l'unité dépassée n'est plus current",
          all((eng.store.get(i) is not None and not eng.store.get(i).is_current)
              for i in old_ids), f"olds={old_ids[:3]}")

    # 4 — dossier citable sur une vraie réunion
    q, proj, intent = QUERIES["dossier"]
    r4 = eng.ask(q, intent=intent, project=proj)
    secs = {s["title"] for s in (_get(r4.payload, "sections") or [])}
    need = {"Contexte", "Faits établis", "Chronologie", "Décisions",
            "Points ouverts", "Responsables"}
    check("4. dossier complet, claims tous cités",
          need <= secs and not [c for s in (_get(r4.payload, "sections") or [])
                                for c in s["claims"] if not c["unit_ids"]],
          f"sections={sorted(secs)}")

    # 5/6 — recipients: resolved or clarified, NEVER invented (brief clause)
    q, proj, intent = QUERIES["email_owner"]
    r5 = eng.ask(q, intent=intent, project=proj)
    rec5 = _get(r5.payload, "recipient") or {}
    check("5. mail: destinataire résolu ou clarifié, jamais inventé",
          (r5.operation == "clarify" and bool(r5.clarification))
          or (rec5 and _get(r5.payload, "needs_confirmation") is True
              and _get(r5.payload, "executed") is not True),
          f"op={r5.operation} recipient={rec5.get('name')}")
    q, proj, intent = QUERIES["email_ambiguous"]
    r6 = eng.ask(q, intent=intent, project=proj)
    check("6. destinataire absent -> clarification",
          r6.operation == "clarify" or not (_get(r6.payload, "recipient") or {}).get("email"),
          f"op={r6.operation}")

    # 7 — hypothèse n'agit pas (real phrasing) + guard invariant on the store
    q, proj, intent = QUERIES["hypo_task"]
    r7 = eng.ask(q, intent=intent, project=proj)
    check("7. hypothèse déclenchée -> aucune action",
          r7.operation != "create_task", f"op={r7.operation} notes={r7.notes}")
    check("   · aucune arête supersedes depuis un status=hypothesis",
          len(hyp_edges) == 0, f"n={len(hyp_edges)}")

    # safety: nothing executed anywhere
    blob = json.dumps([r.payload.__dict__ if hasattr(r.payload, '__dict__')
                       else r.payload for r in (r1, r2, r4, r5, r6, r7)],
                      ensure_ascii=False, default=str)
    check("S. rien n'est jamais exécuté", '"executed": true' not in blob
          and '"executed":true' not in blob, "")

    backends = eng.stats()   # read back BEFORE close: stats() hits the store
    eng.close()
    failed = [n for n, ok, _ in results if not ok]
    out = {"date": date.today().isoformat(), "meetings": mids,
           "reader": args.reader_model, "embed": "e5-base (injected once)",
           "llm_answer": (f"ollama {args.compose_model} (verify-gated rewrite)"
                          if getattr(args, "compose", False)
                          else "template — the LLM answer box is off "
                               "(agent.llm_answer; --compose tests it)"),
           "engine_backends": backends,
           "subject_source": stats["subject_source"], "corpus_stats": stats,
           "supersede_edges": len(sup_edges),
           # the composed-demo evidence: which outcome each answer got
           # ("llm:composed+verified" vs rejected/unavailable — a run with the
           # LLM configured proves nothing unless the per-query outcome is on
           # disk; "report verification evidence")
           "llm_notes": {"answer": list(r1.notes), "breakdown": list(r2.notes)},
           "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in results],
           "failed": failed}
    tag = "-".join(mids)
    if getattr(args, "compose", False):
        tag += "-compose"          # a composed run is a DIFFERENT experiment:
                                   # it never overwrites the template artefact
    dest = root / "runs" / "demo_real" / f"demo-real-{out['date']}-{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    width = max(len(n) for n, _, _ in results)
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name:<{width}}"
              f"{('   ' + detail) if (detail and not ok) else ''}")
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    print("supersede edges on real speech:", len(sup_edges), "| per-meeting:",
          stats["per_meeting"])
    print("wrote", dest)
    return 1 if failed else 0
