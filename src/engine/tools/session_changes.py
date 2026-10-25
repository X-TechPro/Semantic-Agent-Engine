"""Reviewable session changes: one row per (new turn, point), latest old.

The exact-pair gold is too strict for the task: a new turn can legitimately
replace any earlier mention of the old value, while the hand label records one
canonical pair. This tool groups judged candidates by new turn and a
value-derived point category, keeps the most recent old turn, and emits a
citation-bearing review queue. It never writes `valid_to`.

Two corrections learned from the 024c audit:
  * the model's `point` text is unreliable ("date" for a duration, for a
    weekday, and for a month), so the category is derived from the quoted
    old/new values first and the point text only as a fallback;
  * a quoted value that does not occur in its own turn is not evidence, so both
    values must be grounded in the text before a row survives.

Input rows: {old, new, point, old_value, new_value} (the 7B judge's survivors).
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

WEEKDAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi",
            "dimanche", "week end", "weekend")
MONTHS = ("janvier", "fevrier", "mars", "avril", "mai", "juin", "juillet",
          "aout", "septembre", "octobre", "novembre", "decembre", "ete",
          "hiver", "printemps", "automne", "vacances")
DURATION = re.compile(r"\b(?:\d+|un|une|deux|trois|quatre|cinq|six|sept|huit|"
                      r"neuf|dix|onze|douze)\s+(?:jours?|semaines?|mois)\b")
TIME = re.compile(r"\b\d{1,2}\s*h(?:\d{2})?\b|\bheures?\b")
STOP = {"le", "la", "les", "des", "du", "de", "un", "une", "et", "ou", "a",
        "au", "aux", "en", "dans", "sur", "pour", "par", "ce", "cette", "il",
        "elle", "on", "nous", "vous", "est", "sont", "peut", "plus", "pas"}


def fold(text: object) -> str:
    s = unicodedata.normalize("NFD", str(text or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(text: object) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", fold(text)) if len(w) > 2 and w not in STOP}


def grounded(value: object, text: object, threshold: float = 0.5) -> bool:
    """Is the quoted value present in its turn? Head token covers ellipsis."""
    vt, tt = tokens(value), tokens(text)
    if not vt:
        return False
    head = next(iter(re.findall(r"[a-z0-9]+", fold(value))), "")
    if head and head in tt:
        return True
    return len(vt & tt) / len(vt) >= threshold


def value_category(value: object) -> str:
    """Category of one quoted value. Old and new must agree on this."""
    blob = fold(value)
    if any(re.search(rf"(?<!\w){re.escape(w)}(?:s|es)?(?!\w)", blob) for w in WEEKDAYS):
        return "weekday"
    if TIME.search(blob):
        return "time"
    if any(re.search(rf"(?<!\w){re.escape(w)}(?:s|es)?(?!\w)", blob) for w in MONTHS):
        return "date_period"
    if DURATION.search(blob):
        return "duration"
    for name, words in (
        ("choice", ("ouvrir", "cloturer", "choisir", "prendre", "instaurer",
                    "demander", "inclure", "option", "systeme")),
        ("food", ("sandwich", "nourriture", "buvette", "repas", "food", "pizza",
                  "boisson", "restauration")),
        ("mode", ("mode", "billetterie", "reservation", "paiement", "entree",
                  "ligne", "vente", "pre vente")),
        ("lieu", ("lieu", "salle", "arene", "place", "site", "espace", "scene")),
        ("budget", ("budget", "financement", "cout", "prix", "tarif")),
        ("count", ("nombre", "films", "personnes", "quantite", "partie")),
        ("role", ("role", "respons", "charge", "community", "organis", "acteur")),
        ("programme", ("programme", "film", "artiste", "concert", "musique",
                       "pop", "jazz", "rock", "rap", "chanson", "chanteuse")),
        ("securite", ("securite", "vigipirate", "vigile", "secours")),
        ("communication", ("communication", "pub", "affiche", "reseau",
                           "influenceur")),
    ):
        if any(re.search(rf"(?<!\w){re.escape(w)}(?:s|es)?(?!\w)", blob) for w in words):
            return name
    return "autre"


#: an old value that is only filler names no state to replace
FILLER = ("histoire que", "chose", "truc", "peut etre", "un peu plus tard")


def select_latest(rows: list[dict], *, texts: dict[int, str] | None = None,
                  require_grounding: bool = True,
                  short_only: bool = True) -> list[dict]:
    """Keep the most recent grounded old turn per (new turn, value category)."""
    picked: dict[tuple[int, str], dict] = {}
    for row in rows:
        try:
            old, new = int(row["old"]), int(row["new"])
        except (KeyError, TypeError, ValueError):
            continue
        if old >= new:
            continue
        if require_grounding and texts is not None:
            if not grounded(row.get("old_value"), texts.get(old, "")):
                continue
            if not grounded(row.get("new_value"), texts.get(new, "")):
                continue
        co = value_category(row.get("old_value"))
        cn = value_category(row.get("new_value"))
        if co == "autre" or co != cn:
            continue
        if tokens(row.get("old_value")) <= tokens(row.get("new_value")):
            continue        # additive/elaborative, not a replacement
        if any(f in fold(row.get("old_value")) for f in FILLER):
            continue
        if tokens(row.get("new_value")) <= tokens(row.get("old_value")):
            continue        # restatement
        ov, nv = fold(row.get("old_value")), fold(row.get("new_value"))
        if "ancien (tour" in ov or "nouveau (tour" in nv:
            continue        # model echoed the prompt
        if any(f in nv for f in ("pas specifi", "non specifi", "detail non",
                                 "on pourrait", "mode initial", "mode final",
                                 "une seule chanson", "pourrait soit",
                                 "exactement", "ceux qui")):
            continue
        if ov.startswith("ancien ") or nv.startswith("nouveau "):
            continue        # model echoed the prompt
        genres = ("jazz", "rock", "pop", "rap")
        if (("artiste" in ov and any(g in nv for g in genres)
             and "artiste" not in nv)
                or ("artiste" in nv and any(g in ov for g in genres)
                    and "artiste" not in ov)):
            continue        # artist choice vs genre choice: different point
        if (("commencer" in ov or "debut" in ov)
                and ("conclure" in nv or "fin" in nv)):
            continue        # opening vs closing: different point
        if ("puis" in ov and "puis" in nv
                and any(g in ov for g in genres) and any(g in nv for g in genres)):
            continue        # a list with one more genre is an addition
        ot = fold(texts.get(old, "")) if texts is not None else ""
        nt = fold(texts.get(new, "")) if texts is not None else ""
        if (("commencer" in ot or "debut" in ot)
                and ("conclure" in nt or "fin" in nt)):
            continue        # opening vs closing in the turns themselves
        if any(f in nv for f in ("justifiera", "pour la faire venir")):
            continue        # restatement of the same pricing idea
        if "amortir" in nv and "buvette" in ov:
            continue        # restatement of the buvette money
        # A concrete alternative is short. Long descriptions are elaborations:
        # "notre musique" can replace "pop ou jazz"; "les artistes pour lesquels
        # les gens votent doivent etre disponibles" is a new clause, not a value.
        if short_only and co not in ("date_period", "duration", "time", "count",
                                      "budget"):
            if len(tokens(row.get("new_value"))) > 4:
                continue
            if len(tokens(row.get("old_value"))) > 5:
                continue
        key = (new, co)
        prev = picked.get(key)
        if prev is None or old > prev["old"]:
            picked[key] = {
                "old": old,
                "new": new,
                "point": key[1],
                "old_value": row.get("old_value"),
                "new_value": row.get("new_value"),
            }
    return sorted(picked.values(), key=lambda r: (r["new"], r["point"]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True,
                    help="JSON list of judged survivors")
    ap.add_argument("--meeting", required=True)
    ap.add_argument("--meetings", default="data/external/summre_test/meetings.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-grounding", action="store_true")
    ap.add_argument("--permissive", action="store_true")
    args = ap.parse_args(argv)
    data = json.loads(Path(args.src).read_text(encoding="utf-8"))
    rows = data["changes"] if isinstance(data, dict) else data
    raw = json.loads(Path(args.meetings).read_text(encoding="utf-8"))
    texts = {i: u["text"] for i, u in enumerate(raw[args.meeting])}
    changes = select_latest(rows, texts=texts,
                            require_grounding=not args.no_grounding,
                            short_only=not args.permissive)
    out = Path(args.out or f"runs/session_changes/{args.meeting}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meeting": args.meeting, "n": len(changes),
                               "changes": changes}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"{args.meeting}: {len(rows)} survivors -> {len(changes)} review rows -> {out}")
    for c in changes:
        print(f"  {c['old']}->{c['new']} [{c['point']}] "
              f"{str(c['old_value'])[:40]} => {str(c['new_value'])[:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
