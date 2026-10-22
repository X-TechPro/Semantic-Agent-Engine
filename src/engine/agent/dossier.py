"""`build_dossier` operation (6.11) — structured, deduped, every claim cited.

Sections are the ones the brief names: context, established facts, timeline,
decisions, open points, owners, references — plus "Informations manquantes"
for anything the sources do not answer, which is reported rather than guessed.
"""

from __future__ import annotations

from collections import defaultdict

from engine.schemas import Claim, Dossier, DossierSection, Unit
from engine.read.baseline_rules import is_injection

SOFT = {"proposed", "hypothesis", "conditional"}
SECTION_ORDER = ["Contexte", "Faits établis", "Chronologie", "Décisions",
                 "Points ouverts", "Responsables", "Références",
                 "Informations manquantes"]


def build_dossier(store, project: str, *, max_items: int = 40,
                  attribute_labels: dict[str, str] | None = None) -> Dossier:
    units = [u for u in store.all_units() if (u.project or "").lower() == project.lower()]
    if not units:
        return Dossier(project=project, missing_information=[
            f"aucune source trouvée pour le projet {project}"],
                       markdown=f"# Dossier {project}\n\nAucune information.")
    units.sort(key=lambda u: (u.source_date, u.unit_id))
    labels = attribute_labels or {}

    facts, decisions, open_points, owners, timeline = [], [], [], [], []
    for u in units:
        cite = Claim(text=u.text.strip(), unit_ids=[u.unit_id])
        if not u.is_current:
            continue
        if u.type == "other" or is_injection(u.text):
            # a directive aimed at the assistant is data, not a fact about the
            # project: it stays in the store and in raw retrieval, but the
            # dossier never quotes it (same guard as answer.py, D-038). The
            # text test stands on its own because a trained reader does not
            # apply INJECTION_RE and could hand us the payload as `fact`.
            continue
        if u.type == "decision" and u.status in ("confirmed", "cancelled"):
            decisions.append(cite)
        elif u.type in ("commitment",):
            owners.append(Claim(text=_owner_line(u), unit_ids=[u.unit_id]))
        elif u.type == "question":
            open_points.append(cite)
        elif u.status in SOFT:
            open_points.append(Claim(text=f"({u.status}) {u.text.strip()}",
                                     unit_ids=[u.unit_id]))
        else:
            facts.append(cite)
        timeline.append(Claim(text=_timeline_line(u, labels), unit_ids=[u.unit_id]))

    # history: what replaced what — resolved via supersede edges, not guessed
    resolved = []
    for u in units:
        if u.is_current:
            continue
        head = store.supersede_chain(u.unit_id)[-1]
        cur = store.get(head)
        txt = f"Remplacé : {u.text.strip()}"
        if cur is not None:
            txt += f" → en vigueur : {cur.text.strip()}"
        resolved.append(Claim(text=txt, unit_ids=[u.unit_id] + ([cur.unit_id] if cur else [])))
    decisions += resolved

    missing = _missing(store, units, project)
    refs = [{"source_id": s, "date": d, "units": int(n)} for s, d, n in
            store.db.execute(
                "SELECT source_id, MIN(source_date), COUNT(*) FROM units "
                "WHERE project=? GROUP BY source_id ORDER BY MIN(source_date)",
                (project,)).fetchall()]

    #: Contexte is the project's own opening statements, quoted verbatim.
    #: It used to hold the generated sentence "Dossier automatique du projet X
    #: établi à partir de N source(s)", and because the schema requires every
    #: claim to cite a unit, that sentence was made to cite `units[0]` — an
    #: unrelated unit, so the dossier's headline "evidence" was fabricated. A
    #: claim must be something its cited unit actually says; the provenance
    #: sentence belongs in the markdown header, and the source counts already
    #: have a home in `dossier.references`.
    context = [Claim(text=u.text.strip(), unit_ids=[u.unit_id])
               for u in sorted((x for x in units
                                if x.type != "other" and not is_injection(x.text)),
                               key=lambda x: (x.source_date, x.unit_id))[:3]]

    sections = [
        DossierSection(title="Contexte", claims=_dedupe(context)[:max_items]),
        DossierSection(title="Faits établis", claims=_dedupe(facts)[:max_items]),
        DossierSection(title="Chronologie", claims=_dedupe(timeline)[:max_items]),
        DossierSection(title="Décisions", claims=_dedupe(decisions)[:max_items]),
        DossierSection(title="Points ouverts", claims=_dedupe(open_points)[:max_items]),
        DossierSection(title="Responsables", claims=_dedupe(owners)[:max_items]),
    ]
    # sources are *references*, not claims: a Claim must cite a unit, so the
    # file-level provenance lives in `dossier.references` (it used to be built
    # as a Claim with unit_ids=[], which the schema rejects).
    dossier = Dossier(project=project, sections=sections, references=refs,
                      missing_information=missing, markdown="")
    return dossier.model_copy(update={"markdown": to_markdown(dossier)})


def _owner_line(u: Unit) -> str:
    who = u.slots.owner or "non précisé"
    due = f" (échéance {u.slots.due})" if u.slots.due else ""
    act = f" : {u.slots.action}" if u.slots.action else ""
    return f"{who} — {u.text.strip()}{due}{act}"


def _timeline_line(u: Unit, labels: dict[str, str]) -> str:
    label = labels.get(u.slots.subject or "", "")
    tag = f" [{label}]" if label else ""
    return f"{u.source_date} · {u.status}{tag} · {u.text.strip()}"


def _dedupe(claims: list[Claim]) -> list[Claim]:
    seen: set[str] = set()
    out: list[Claim] = []
    for c in claims:
        key = c.text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _missing(store, units: list[Unit], project: str) -> list[str]:
    out: list[str] = []
    subjects = {u.slots.subject for u in units if u.slots.subject}
    for u in units:
        if u.type == "question" and u.is_current:
            answered = store.db.execute(
                """SELECT 1 FROM edges e WHERE e.relation='answers' AND e.to_id=?
                   LIMIT 1""", (u.unit_id,)).fetchone()
            if not answered:
                out.append(f"Question sans réponse en vigueur : {u.text.strip()}")
    if not any(u.slots.owner for u in units):
        out.append("Aucun responsable explicitement désigné dans les sources.")
    if not subjects:
        out.append("Aucun sujet normalisé détecté : l'historique par attribut est vide.")
    return sorted(set(out))


def to_markdown(d: Dossier) -> str:
    lines = [f"# Dossier — projet {d.project}", ""]
    for s in d.sections:
        if not s.claims:
            continue
        lines.append(f"## {s.title}")
        for c in s.claims:
            cite = " ".join(f"[{u}]" for u in c.unit_ids)
            lines.append(f"- {c.text} {cite}".rstrip())
        lines.append("")
    if d.references:
        lines.append("## Références")
        for r in d.references:
            lines.append(f"- {r['source_id']} — {r['date']} ({r['units']} unités)")
        lines.append("")
    if d.missing_information:
        lines.append("## Informations manquantes")
        lines += [f"- {m}" for m in d.missing_information]
    return "\n".join(lines)
