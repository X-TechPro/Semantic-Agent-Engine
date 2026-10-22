"""Generate the 12.3 golden minimal-pair suites as reviewable JSONL.

Why a generator instead of 300 hand-typed rows: 12.3 demands **>= 30 pairs per
category** across twelve categories, and the content is combinatorial (project x
attribute x value). Determinism matters more than brevity — no RNG, fixed
vocabulary, sorted output — because a golden fixture that changes between runs
cannot be reviewed, and 12.3 exists to be the thing the architecture is judged
against. The vocabulary is taken from `gen.ledger.ATTRIBUTES` so the fixtures use
the same attribute nouns the corpus itself renders, not a parallel invented one.

THE GOLD IS AUTHORED FROM THE GUIDE, NEVER FROM THE ENGINE. Every `gold` comes
from IDEA.md 6.8.3 — "`supersedes` requires: `new.source_date >= old.source_date`
and same `(project, subject_key)` and `new.status in {confirmed}` (or `cancelled`)
and `new.kind=info`. Otherwise downgrade to `contradicts` (if incompatible) or
`supports/unrelated`" — from 12.3's twelve categories, or from line 403's four
tricky negatives. Where the guide only says "not `supersedes`", the gold is
`not_supersedes` and the test asserts that weaker claim rather than inventing a
specific relation. A test whose expectation is derived from the implementation
measures nothing.

Two row shapes, isolating one stage each:

* `label` — one utterance + the reader (or slot) verdict 12.3 names for it;
* `pair`  — two labelled units + the relation the guide requires between them.

Deliberately NOT claimed as gold:
* `type` for the status category. The rules reader tests `RISK` before the status
  branches, so "le risque de conformité est validé" legitimately types `risk`;
  asserting `decision` there would be a fixture bug, not an engine bug.
* `owner` precision for people who are not in the roster — the slot stage can
  only resolve a name it has been given, so the test passes the roster.

Run: PYTHONPATH=src python -m engine.tools.gen_golden_pairs
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from gen.ledger import ATTRIBUTES

#: >= 30 rows per category means the project vocabulary has to be wide enough
#: for the narrowest category (temporal covers only the two `date` attributes).
PROJECTS = ["Atlas", "Boréal", "Cendre", "Delta", "Eywa", "Fer", "Galaxie",
            "Hélix", "Ivoire", "Jade", "Korrigent", "Lumen", "Myral", "Nordane",
            "Onyx", "Pyrane", "Quercy", "Rodan", "Sirius", "Thulé", "Umbra",
            "Vaurien", "Wapiti", "Xylo", "Yucca", "Zéphyr"]

#: (attribute key, French noun phrase, value kind) from the ledger's own table.
ATTRS = [(k, v["np"], v["kind"]) for k, v in ATTRIBUTES.items()]
DATE_ATTRS = [(k, np) for k, np, kind in ATTRS if kind == "date"]
AMOUNT_ATTRS = [(k, np) for k, np, kind in ATTRS if kind == "amount"]
ANY_ATTRS = [(k, np) for k, np, _ in ATTRS]

#: 12.3's status bullets, verbatim in spirit: the utterance has one gold status.
#: Only `status` is asserted — see the module docstring for why `type` is not.
STATUS_LABELS: list[tuple[str, str, str]] = [
    ("Le budget est proposé.", "proposed", "12.3 status: a proposal is not confirmed"),
    ("Le budget est validé.", "confirmed", "12.3 status: validation is confirmed"),
    ("Le budget est annulé.", "cancelled", "12.3 status: cancellation"),
    ("On pourrait peut-être revoir le budget.", "hypothesis",
     "12.3 status: hypothesis -> no action"),
    ("D'accord si le budget passe.", "conditional", "12.3 status: conditional"),
    ("Ce n'est pas encore validé.", "not_confirmed",
     "12.3 status: negation must NOT read as confirmed"),
    ("Rien n'a été décidé pour le budget.", "mention",
     "6.2.2: nothing asserted -> mention"),
    ("Le budget n'est pas acté.", "not_confirmed",
     "6.2.2: 'pas acté' is a not-yet, never a confirmation"),
    ("Le budget reste en discussion.", "proposed", "6.2.2: still open"),
    ("C'est suspendu pour le budget.", "cancelled", "6.2.2: suspension is a cancel"),
]

STATUS_TEMPLATES = [
    ("{np} est proposé.", "proposed"),
    ("{np} est validé.", "confirmed"),
    ("{np} est acté.", "confirmed"),
    ("{np} est annulé.", "cancelled"),
    ("On a changé {np}.", "confirmed"),
    ("Finalement {np} passe à une autre valeur.", "confirmed"),
    ("On pourrait peut-être revoir {np}.", "hypothesis"),
    ("Ça reste une idée pour {np}.", "hypothesis"),
    ("{np} est sous réserve d'audit.", "conditional"),
    ("D'accord si {np} passe.", "conditional"),
    # 12.3 promises only "negation -> not confirmed". It does NOT promise
    # `proposed`, and asserting that here manufactured 522 fake failures: the
    # engine said `mention`, which satisfies the guide. So these rows carry a
    # negative claim, which the test checks as one.
    ("Ce n'est pas encore validé pour {np}.", "not_confirmed"),
    ("{np} n'est pas acté.", "not_confirmed"),
    ("{np} est en discussion.", "proposed"),
    ("On suspend {np}.", "cancelled"),
    ("{np}, c'est tombé.", "cancelled"),
    ("On remet {np} à l'ordre du jour.", "confirmed"),
    ("{np} est fixé.", "confirmed"),
    ("Le sujet {np} reste une piste.", "hypothesis"),
    ("{np} est en attente de validation.", "proposed"),
    ("Sauf imprévu, {np} est validé.", "conditional"),
]

PEOPLE = [("Marie", "Karim"), ("Sophie", "Nadia"), ("Léa", "Hugo"),
          ("Adam", "Chloé"), ("Nina", "Sami"), ("Iris", "Noé"),
          ("Julie", "Marc"), ("Sara", "Yanis"), ("Inès", "Théo"),
          ("Louise", "Mehdi"), ("Camille", "Ethan"), ("Alice", "Noah"),
          ("Zoé", "Liam"), ("Jade", "Malik"), ("Romy", "Adem"),
          ("Manon", "Tom")]

#: 12.3 person category. `owner` is "{p}", "{q}" or None, resolved at build time.
PERSON_TEMPLATES = [
    ("{p} s'occupera de {np} du projet {proj}.", "{p}",
     "6.3: ownership needs an explicit assignment pattern"),
    ("{p} s'engage à envoyer le dossier du projet {proj}.", "{p}",
     "12.3 person: a commitment binds the speaker"),
    ("{p} demande à {q} d'envoyer le dossier du projet {proj}.", "{q}",
     "12.3 person: the delegatee owns it, not the requester"),
    ("C'est {p} qui gère {np} sur le projet {proj}.", "{p}",
     "6.3: 'c'est X qui' is an assignment"),
    ("{q} a parlé du budget du projet {proj}.", None,
     "12.3 person: mention != responsible"),
    ("{p} a évoqué {np} du projet {proj} avec {q}.", None,
     "6.3 role vs mention: an aside names two people, owns neither"),
    ("On part sur {q} pour piloter {np} du projet {proj}.", "{q}",
     "6.3: 'on part sur X pour piloter' assigns"),
    ("La responsabilité de {np} du projet {proj} revient à {p}.", "{p}",
     "6.3: 'revient à X' assigns"),
    ("{np} du projet {proj} : c'est {q} qui s'en charge.", "{q}",
     "6.3 elision trap: 's'en charge' still assigns"),
    ("Karim a dit que {np} du projet {proj} concerne {p}.", None,
     "6.3: 'concerne' is a mention, not an ownership"),
]

_ADIAC = re.compile(r"[\u0300-\u036f]")


def _slug(value: str) -> str:
    """Accent-free lower slug, so a subject key never depends on locale."""
    raw = _ADIAC.sub("", unicodedata.normalize("NFKD", value))
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


def _slot(proj: str, attr_key: str) -> str:
    return f"{_slug(proj)}.{_slug(attr_key)}"


def _pair(new_text: str, new_date: str, new_type: str, new_status: str,
          new_slots: dict[str, str], old_text: str, old_date: str, old_type: str,
          old_status: str, old_slots: dict[str, str], gold: str, why: str,
          category: str) -> dict:
    """One guide-labelled comparison between two already-labelled units."""
    return {"kind": "pair", "category": category,
            "new": {"text": new_text, "date": new_date, "kind": "info",
                    "type": new_type, "status": new_status, **new_slots},
            "old": {"text": old_text, "date": old_date, "kind": "info",
                    "type": old_type, "status": old_status, **old_slots},
            "gold": gold, "why": why}


def temporal_rows() -> list[dict]:
    """12.3 temporal: the later statement replaces the earlier one, same slot."""
    rows = []
    months = [("décembre", "janvier", "2026-12-01", "2027-01-15"),
              ("novembre", "mars", "2026-11-10", "2027-03-02"),
              ("avril", "juin", "2026-04-05", "2026-06-20"),
              ("mai", "septembre", "2026-05-01", "2026-09-15")]
    for i, proj in enumerate(PROJECTS):
        for key, _np in DATE_ATTRS:
            old_m, new_m, old_d, new_d = months[i % len(months)]
            slot = _slot(proj, key)
            rows.append(_pair(
                f"Finalement le lancement du projet {proj} passe en {new_m}.",
                "2026-10-05", "decision", "confirmed",
                {"project": proj, "subject": slot, "date_value": new_d},
                f"Le lancement du projet {proj} est prévu en {old_m}.", "2026-10-01",
                "fact", "mention", {"project": proj, "subject": slot,
                                    "date_value": old_d},
                "supersedes",
                "6.8.3: newer source_date, same slot, new.status confirmed",
                "temporal"))
    return rows


def out_of_order_rows() -> list[dict]:
    """12.3 out-of-order: an OLDER statement cannot replace a newer one."""
    rows = []
    for i, proj in enumerate(PROJECTS):
        for key, _np in DATE_ATTRS:
            slot = _slot(proj, key)
            rows.append(_pair(
                f"Le point sur {key} du projet {proj} datait d'avril.", "2026-09-30",
                "decision", "confirmed", {"project": proj, "subject": slot,
                                          "date_value": "2026-04-01"},
                f"Le point sur {key} du projet {proj} est fixé en septembre.",
                "2026-10-02", "decision", "confirmed",
                {"project": proj, "subject": slot, "date_value": "2026-09-01"},
                "not_supersedes",
                "12.3 out-of-order: 6.8.3 requires new.date >= old.date",
                "out_of_order"))
            i += 1
    return rows


def value_flip_rows() -> list[dict]:
    """12.3 number/date flips: same slot, different value => supersede."""
    rows = []
    flips = [("1,5 M€", "1500000", "2,4 M€", "2400000"),
             ("15 M€", "15000000", "1,5 M€", "1500000"),
             ("12 000 €", "12000", "2 000 €", "2000"),
             ("30 000 €", "30000", "45 000 €", "45000")]
    for i, proj in enumerate(PROJECTS):
        for key, _np in AMOUNT_ATTRS:
            f = flips[i % len(flips)]
            slot = _slot(proj, key)
            rows.append(_pair(
                f"Le {key} du projet {proj} est révisé à {f[2]}.", "2026-10-05",
                "decision", "confirmed", {"project": proj, "subject": slot,
                                          "amount": f[3]},
                f"Le {key} du projet {proj} est acté à {f[0]}.", "2026-10-01",
                "decision", "confirmed", {"project": proj, "subject": slot,
                                          "amount": f[1]},
                "supersedes", "6.8.3: same slot, newer, confirmed, other amount",
                "value_flips"))
    # the day-vs-day flip 12.3 names explicitly ("22 novembre" vs "2 novembre")
    for proj in PROJECTS[:16]:
        slot = _slot(proj, "deadline_report")
        rows.append(_pair(
            f"La deadline du rapport du projet {proj} est le 2 novembre.", "2026-10-05",
            "decision", "confirmed", {"project": proj, "subject": slot,
                                      "date_value": "2026-11-02"},
            f"La deadline du rapport du projet {proj} est le 22 novembre.", "2026-10-01",
            "decision", "confirmed", {"project": proj, "subject": slot,
                                      "date_value": "2026-11-22"},
            "supersedes", "12.3 number/date flips: 22 nov -> 2 nov is a new value",
            "value_flips"))
    return rows


def ratification_rows() -> list[dict]:
    """6.8.3: same value, soft -> assertive is a state change, so it supersedes."""
    rows = []
    for proj in PROJECTS:
        for key, np in ANY_ATTRS:
            slot = _slot(proj, key)
            rows.append(_pair(
                f"C'est acté : {np} du projet {proj}.", "2026-10-05",
                "decision", "confirmed", {"project": proj, "subject": slot},
                f"Je propose {np} pour le projet {proj}.", "2026-10-01",
                "idea", "proposed", {"project": proj, "subject": slot},
                "supersedes", "6.8.3 ratification: value kept, state changed",
                "ratification"))
    return rows


def restatement_rows() -> list[dict]:
    """Line 403 + 6.8.3: restating an in-force value is `supports`, not a new head."""
    rows = []
    for proj in PROJECTS:
        for key, np in ANY_ATTRS:
            slot = _slot(proj, key)
            rows.append(_pair(
                f"Je le redis, {np} du projet {proj} reste inchangé.", "2026-10-05",
                "decision", "confirmed", {"project": proj, "subject": slot},
                f"{np.capitalize()} du projet {proj} est acté.", "2026-10-01",
                "decision", "confirmed", {"project": proj, "subject": slot},
                "supports", "12.3 repeated statement -> supports", "repeated"))
    return rows


def hypothesis_after_decision_rows() -> list[dict]:
    """Line 403: a hypothesis or proposal arriving after a decision is not a head."""
    rows = []
    for proj in PROJECTS:
        for key, np in ANY_ATTRS[:6]:
            slot = _slot(proj, key)
            rows.append(_pair(
                f"On pourrait peut-être revoir {np} du projet {proj}.", "2026-10-05",
                "idea", "hypothesis", {"project": proj, "subject": slot},
                f"{np.capitalize()} du projet {proj} est validé.", "2026-10-01",
                "decision", "confirmed", {"project": proj, "subject": slot},
                "not_supersedes", "12.3: hypothesis after decision -> no supersede",
                "hypothesis_after_decision"))
            rows.append(_pair(
                f"Je propose une autre valeur pour {np} du projet {proj}.", "2026-10-05",
                "idea", "proposed", {"project": proj, "subject": slot},
                f"{np.capitalize()} du projet {proj} est acté.", "2026-10-01",
                "decision", "confirmed", {"project": proj, "subject": slot},
                "not_supersedes", "line 403: proposal vs decision is not `supersedes`",
                "hypothesis_after_decision"))
    return rows


def same_topic_other_attribute_rows() -> list[dict]:
    """Line 403: same project, different attribute => unrelated."""
    rows = []
    pairs = [("budget", "launch_date"), ("vendor", "scope"),
             ("pricing", "hiring"), ("stack", "env_strategy"),
             ("deadline_report", "risk_compliance")]
    for proj in PROJECTS:
        for a, b in pairs:
            rows.append(_pair(
                f"{b} du projet {proj} est acté.", "2026-10-05", "decision", "confirmed",
                {"project": proj, "subject": _slot(proj, b)},
                f"{a} du projet {proj} est acté.", "2026-10-01", "decision", "confirmed",
                {"project": proj, "subject": _slot(proj, a)},
                "unrelated", "6.8.3 same topic / other attribute -> unrelated",
                "same_topic_other_attribute"))
    return rows


def cross_project_rows() -> list[dict]:
    """12.3 two projects with shared vocabulary: no cross-project supersede."""
    rows = []
    shared = ["budget", "vendor", "stack", "scope", "pricing"]
    #: two different neighbours per project, so the category clears 12.3's floor
    #: without weakening the claim — every row is still "same attribute name,
    #: different project".
    for offset in (1, 2):
        for i, a in enumerate(PROJECTS):
            b = PROJECTS[(i + offset) % len(PROJECTS)]
            key = shared[i % len(shared)]
            rows.append(_pair(
                f"{key} du projet {b} est acté à 2 M€.", "2026-10-05", "decision",
                "confirmed", {"project": b, "subject": _slot(b, key),
                              "amount": "2000000"},
                f"{key} du projet {a} est acté à 1,5 M€.", "2026-10-01", "decision",
                "confirmed", {"project": a, "subject": _slot(a, key),
                              "amount": "1500000"},
                "unrelated", "12.3 two projects sharing vocabulary stay separate",
                "cross_project"))
    return rows


def apple_rows() -> list[dict]:
    """12.3's informative (NOT gating) apple test: same word, three senses.

    The guide asks whether the first links to the second and not the third, and
    says explicitly it is informative rather than a gate, so `gold` is
    `informative` and the harness records the produced relation instead of
    asserting one.
    """
    rows = []
    for proj in PROJECTS:
        slot = _slot(proj, "vendor")
        rows.append(_pair(
            "Apple a sorti un nouveau téléphone.", "2026-10-05", "fact", "mention",
            {"project": proj, "subject": slot},
            "J'ai un téléphone Apple.", "2026-10-01", "fact", "mention",
            {"project": proj, "subject": slot},
            "informative", "12.3 apple test: report the link, do not gate on it",
            "apple"))
        rows.append(_pair(
            "J'ai mangé une pomme.", "2026-10-05", "fact", "mention",
            {"project": proj, "subject": slot},
            "J'ai un téléphone Apple.", "2026-10-01", "fact", "mention",
            {"project": proj, "subject": slot},
            "informative", "12.3 apple test: the fruit must not join the brand",
            "apple"))
    return rows


def cancel_rows() -> list[dict]:
    """12.3 status + 6.8.3: `cancelled` is a superseding status.

    NOTE on a real guide/code conflict: 6.8.3 names confirmed and cancelled as
    the superseding statuses and exempts nothing about the OLD unit's status, so
    a cancellation after a proposal is a supersede on the guide's letter. The
    shipped G6 currently downgrades cancel-of-proposal to `contradicts`. This
    fixture keeps the guide's reading so the divergence is visible in the harness
    rather than hidden by a fixture that mirrors the code.
    """
    rows = []
    for proj in PROJECTS:
        for key, np in ANY_ATTRS[:6]:
            slot = _slot(proj, key)
            rows.append(_pair(
                f"Finalement {np} du projet {proj} est annulé.", "2026-10-05",
                "decision", "cancelled", {"project": proj, "subject": slot},
                f"Je propose {np} pour le projet {proj}.", "2026-10-01",
                "idea", "proposed", {"project": proj, "subject": slot},
                "supersedes", "6.8.3: cancelled is a superseding status",
                "cancel_reinstate"))
            rows.append(_pair(
                f"{np.capitalize()} du projet {proj} est de nouveau acté.", "2026-10-09",
                "decision", "confirmed", {"project": proj, "subject": slot},
                f"{np.capitalize()} du projet {proj} est annulé.", "2026-10-05",
                "decision", "cancelled", {"project": proj, "subject": slot},
                "supersedes", "12.3 cancel then reinstate keeps one head",
                "cancel_reinstate"))
    return rows


def _gold_for(status: str) -> dict[str, str]:
    """Turn a guide promise into gold, negative ones included.

    `"not_confirmed"` is not a label — it is the constraint 12.3 actually states
    for a negation ("not confirmed"), and pinning it to a specific label would
    test my own guess instead of the guide.
    """
    if status == "not_confirmed":
        return {"not_status": "confirmed"}
    return {"status": status}


def label_rows() -> list[dict]:
    """Reader-verdict rows: 12.3's status category, asserted on status only."""
    rows: list[dict] = []
    for text, status, why in STATUS_LABELS:
        rows.append({"kind": "label", "stage": "reader", "text": text,
                     "gold": _gold_for(status), "why": why, "category": "status"})
    for proj in PROJECTS:
        for key, np in ANY_ATTRS:
            for tmpl, status in STATUS_TEMPLATES:
                rows.append({"kind": "label", "stage": "reader",
                             "text": f"{tmpl.format(np=np)} (projet {proj}).",
                             "gold": _gold_for(status),
                             "why": f"12.3 status category: {status}",
                             "category": "status"})
    return rows


def person_label_rows() -> list[dict]:
    """Slot-verdict rows: who owns it, with the roster the engine would have."""
    rows: list[dict] = []
    for (p, q) in PEOPLE:
        for proj in PROJECTS[:6]:
            for key, np in ANY_ATTRS[:3]:
                for tmpl, owner_t, why in PERSON_TEMPLATES:
                    text = tmpl.format(p=p, q=q, np=np, proj=proj)
                    owner = None if owner_t is None else owner_t.format(p=p, q=q)
                    rows.append({"kind": "label", "stage": "slots", "text": text,
                                 "gold": {"owner": owner},
                                 "roster": sorted({p, q, "Karim"}),
                                 "why": why, "category": "person",
                                 "project": proj, "subject": _slot(proj, key)})
    return rows


GENERATORS = {
    "status": label_rows,
    "person": person_label_rows,
    "temporal": temporal_rows,
    "out_of_order": out_of_order_rows,
    "value_flips": value_flip_rows,
    "ratification": ratification_rows,
    "repeated": restatement_rows,
    "hypothesis_after_decision": hypothesis_after_decision_rows,
    "same_topic_other_attribute": same_topic_other_attribute_rows,
    "cross_project": cross_project_rows,
    "cancel_reinstate": cancel_rows,
    "apple": apple_rows,
}
MIN_PAIRS = 30


def generate_all() -> dict[str, list[dict]]:
    """Every suite, deterministically ordered, without touching the disk.

    The freshness test calls this and compares against the committed JSONL, so a
    fixture cannot drift away from the generator that produced it.
    """
    return {cat: sorted(gen(), key=lambda r: json.dumps(r, sort_keys=True))
            for cat, gen in GENERATORS.items()}


def build(out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for cat, rows in generate_all().items():
        path = out_dir / f"pairs_{cat}.jsonl"
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True)
                                  for r in rows) + "\n", encoding="utf-8")
        counts[cat] = len(rows)
    return counts


def main(argv=None) -> int:
    out = Path(__file__).resolve().parents[3] / "tests" / "golden"
    counts = build(out)
    short = {k: v for k, v in counts.items() if v < MIN_PAIRS}
    for k, v in sorted(counts.items()):
        print(f"  {k:26s} {v:5d}{'  <-- BELOW 12.3 FLOOR' if v < MIN_PAIRS else ''}")
    print(f"\nwrote {len(counts)} suites -> {out}")
    if short:
        print(f"FAIL: 12.3 requires >= {MIN_PAIRS} pairs per category: "
              f"{sorted(short)}")
        return 1
    print(f"all categories >= {MIN_PAIRS} (12.3 floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
