"""Graph-first synthetic meetings: hidden event graph -> noisy dialogue.

The graph is the source of truth. Dialogue is rendered after it, with pronouns,
corrections, side-chat and ASR-like noise. Output keeps event history, typed
mentions/relations, current state and unresolved slots.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

PEOPLE = ["Nora", "Malik", "Camille", "Rachid", "Ines", "Hugo", "Lea", "Yanis",
          "Mina", "Noah", "Salma", "Adam"]
PROJECTS = ["Atlas", "Orion", "Mosaique", "Canopee", "Horizon", "Etincelle",
            "Traverse", "Latitude", "Pollen", "Osmose"]
VENUES = ["salle Atlas", "amphi B", "campus Nord", "halle des ateliers",
          "foyer etudiant", "rotonde", "arena du pays d'Aix", "mediatheque"]
SUPPLIERS = ["Maison Pollen", "Traiteur du Parc", "Cuisine des Quais",
             "Comptoir Mistral", "Table Mobile"]
ACTIONS = ["le dossier technique", "la billetterie en ligne", "les badges",
           "la sono", "le planning des benevoles"]
SIDE = ["pardon le micro a coupe", "je retrouve la bonne version du doc",
        "on peut revenir la-dessus", "je note mais je ne suis pas sur",
        "deux secondes je reponds au chat", "je reformule ce que j'ai dit",
        "ca a coupe apres vendredi", "je crois que l'export n'est pas a jour"]

TEMPLATES = {
    "A": {
        "budget_proposed": "Pour le budget, je pars sur {amount} euros, a confirmer.",
        "budget_confirmed": "On valide le budget a {amount} euros.",
        "budget_changed": "Finalement le budget passe a {amount} euros.",
        "date_proposed": "On pourrait viser le {day}.",
        "date_confirmed": "Je bloque le {day} au final.",
        "date_moved": "Finalement on decale au {day}.",
        "venue_proposed": "On peut regarder la {venue}.",
        "venue_confirmed": "Le lieu retenu est la {venue}.",
        "supplier_proposed": "Le traiteur {supplier} est confirme.",
        "payment_pending": "Le devis de {supplier} est arrive, le paiement est en attente.",
        "payment_confirmed": "On valide le paiement de {amount} euros a {supplier}.",
        "payment_cancelled": "On annule le paiement a {supplier}.",
        "action_assigned": "{owner} s'occupe de {action} avant le {day}.",
        "action_done": "{owner} a termine {action}.",
        "question": "Ou en est {slot} pour le projet {project} ?",
    },
    "B": {
        "budget_proposed": "Cote enveloppe, {amount} euros me semble une base.",
        "budget_confirmed": "L'enveloppe est arretee a {amount} euros.",
        "budget_changed": "On revoit l'enveloppe a {amount} euros.",
        "date_proposed": "Le {day} pourrait etre une option.",
        "date_confirmed": "On retient le {day}.",
        "date_moved": "La date bouge au {day}.",
        "venue_proposed": "La {venue} est une piste.",
        "venue_confirmed": "On part sur la {venue}.",
        "supplier_proposed": "Le prestataire retenu est {supplier}.",
        "payment_pending": "Facture {supplier} recue, reglement pas encore valide.",
        "payment_confirmed": "Reglement de {amount} euros valide pour {supplier}.",
        "payment_cancelled": "Le reglement {supplier} est annule.",
        "action_assigned": "Pour {action}, {owner} prend le suivi avant le {day}.",
        "action_done": "{owner} a boucle {action}.",
        "question": "Quelle est la situation de {slot} sur {project} ?",
    },
    "C": {
        "budget_proposed": "On peut prevoir {amount} euros, sans validation.",
        "budget_confirmed": "C'est bon pour {amount} euros, c'est valide.",
        "budget_changed": "Nouveau montant valide : {amount} euros.",
        "date_proposed": "Why not le {day}, a voir.",
        "date_confirmed": "Le {day}, c'est acte.",
        "date_moved": "On repart sur le {day}, c'est la nouvelle date.",
        "venue_proposed": "On pourrait essayer la {venue}.",
        "venue_confirmed": "La {venue}, c'est confirme.",
        "supplier_proposed": "On est partis avec {supplier}.",
        "payment_pending": "Le devis {supplier} est la, mais on n'a pas paye.",
        "payment_confirmed": "Le virement de {amount} euros a {supplier} est passe.",
        "payment_cancelled": "On ne paie pas {supplier}, c'est annule.",
        "action_assigned": "{owner} doit faire {action} pour le {day}.",
        "action_done": "{action} est fait par {owner}.",
        "question": "Pour {project}, on en est ou de {slot} ?",
    },
}


def day(rng: random.Random) -> str:
    return f"{rng.randint(1, 28)} {rng.choice(['janvier', 'fevrier', 'mars', 'avril', 'mai', 'juin', 'juillet', 'septembre', 'octobre', 'novembre'])}"


def amount(rng: random.Random) -> int:
    return rng.choice([300, 800, 1200, 1450, 1800, 2200, 2400, 3000])


def event_id(project: str, kind: str) -> str:
    return f"{project}:{kind}"


def add_spec(specs: list[dict], event: dict, slot: str, status: str, text: str,
             order: int, relation: str, value: str, owner: str = "",
             deadline: str = "") -> None:
    specs.append({"event_id": event["event_id"], "slot": slot, "status": status,
                  "text": text, "order": order, "relation": relation,
                  "value": value, "owner": owner, "deadline": deadline})


def make_project(project: str, idx: int, split: str, rng: random.Random) -> dict:
    variant = {"train": "A", "validation": "B", "test": "C"}[split]
    T = TEMPLATES[variant]
    events: list[dict] = []
    specs: list[dict] = []
    relations: list[dict] = []
    mentions: list[dict] = []
    queries: list[dict] = []
    owner = rng.choice(PEOPLE)
    venue = rng.choice(VENUES)
    supplier = rng.choice(SUPPLIERS)
    action = rng.choice(ACTIONS)
    b0, b1 = amount(rng), amount(rng)
    if b0 == b1:
        b1 = b0 + 300
    d0, d1, d2 = day(rng), day(rng), day(rng)
    order = 2

    def event(kind: str, slot: str, value: str, status: str) -> dict:
        ev = {"event_id": event_id(project, kind), "project_id": project,
              "type": kind, "slot": slot, "current_value": value,
              "current_status": status, "history": [], "mentions": [],
              "open": [], "unresolved": []}
        events.append(ev); return ev

    budget = event("budget", "budget", str(b0), "proposed")
    add_spec(specs, budget, "budget", "proposed",
             T["budget_proposed"].format(amount=b0), order, "starts", str(b0)); order += 2
    add_spec(specs, budget, "budget", "confirmed",
             T["budget_confirmed"].format(amount=b1), order, "supersedes", str(b1)); order += 2
    if rng.random() < 0.35:
        b2 = b1 + 200
        add_spec(specs, budget, "budget", "confirmed",
                 T["budget_changed"].format(amount=b2), order, "supersedes", str(b2)); order += 2
        b1 = b2
    budget["current_value"], budget["current_status"] = str(b1), "confirmed"
    budget["history"] = [{"value": str(b0), "status": "proposed"},
                         {"value": str(b1), "status": "confirmed"}]
    # A late proposal after confirmation: latest sentence is not current state.
    if rng.random() < 0.45:
        late = b1 + rng.choice([100, 200, 300])
        add_spec(specs, budget, "budget", "proposed",
                 f"On pourrait finalement remonter le budget a {late} euros.",
                 order, "unrelated", str(late)); order += 2
        budget["history"].append({"value": str(late), "status": "proposed"})

    date_ev = event("date", "date", d0, "proposed")
    add_spec(specs, date_ev, "date", "proposed", T["date_proposed"].format(day=d0),
             order, "starts", d0); order += 2
    add_spec(specs, date_ev, "date", "confirmed", T["date_confirmed"].format(day=d1),
             order, "supersedes", d1); order += 2
    if rng.random() < 0.45:
        add_spec(specs, date_ev, "date", "confirmed", T["date_moved"].format(day=d2),
                 order, "supersedes", d2); order += 2
        d1 = d2
    date_ev["current_value"], date_ev["current_status"] = d1, "confirmed"
    date_ev["history"] = [{"value": d0, "status": "proposed"},
                          {"value": d1, "status": "confirmed"}]

    venue_ev = event("venue", "venue", venue, "proposed")
    v_alt = rng.choice([v for v in VENUES if v != venue])
    add_spec(specs, venue_ev, "venue", "proposed", T["venue_proposed"].format(venue=v_alt),
             order, "starts", v_alt); order += 2
    add_spec(specs, venue_ev, "venue", "confirmed", T["venue_confirmed"].format(venue=venue),
             order, "supersedes", venue); order += 2
    venue_ev["current_value"], venue_ev["current_status"] = venue, "confirmed"
    venue_ev["history"] = [{"value": v_alt, "status": "proposed"},
                           {"value": venue, "status": "confirmed"}]

    supplier_ev = event("supplier", "supplier", supplier, "confirmed")
    add_spec(specs, supplier_ev, "supplier", "confirmed",
             T["supplier_proposed"].format(supplier=supplier), order, "starts", supplier)
    order += 2
    supplier_ev["history"] = [{"value": supplier, "status": "confirmed"}]

    payment_ev = event("payment", "payment", "pending", "open")
    pay = rng.choice([300, 500, 800])
    add_spec(specs, payment_ev, "payment", "open",
             T["payment_pending"].format(supplier=supplier), order, "depends", "pending")
    order += 2
    if rng.random() < 0.55:
        add_spec(specs, payment_ev, "payment", "confirmed",
                 T["payment_confirmed"].format(amount=pay, supplier=supplier),
                 order, "answers", str(pay)); order += 2
        payment_ev["current_value"] = str(pay)
        payment_ev["current_status"] = "confirmed"
        payment_ev["history"] = [{"value": "pending", "status": "open"},
                                 {"value": str(pay), "status": "confirmed"}]
    else:
        payment_ev["unresolved"] = ["payment confirmation missing"]
        payment_ev["open"].append("payment")
        payment_ev["history"] = [{"value": "pending", "status": "open"}]

    action_ev = event("action", "action", action, "proposed")
    add_spec(specs, action_ev, "action", "proposed",
             T["action_assigned"].format(owner=owner, action=action, day=day(rng)),
             order, "starts", action, owner=owner); order += 2
    if rng.random() < 0.5:
        add_spec(specs, action_ev, "action", "completed",
                 T["action_done"].format(owner=owner, action=action),
                 order, "answers", action, owner=owner); order += 2
        action_ev["current_status"] = "completed"
        action_ev["history"] = [{"value": action, "status": "proposed"},
                                {"value": action, "status": "completed"}]
    else:
        action_ev["history"] = [{"value": action, "status": "proposed"}]

    # Hard negative: a different project's same amount/venue appears as a distractor.
    neg_project = rng.choice([p for p in PROJECTS if p != project])
    neg_text = (f"Pour {neg_project}, le budget valide est {b1} euros et le lieu est "
                f"la {venue}.")
    add_spec(specs, events[0], "budget", "distractor", neg_text, order + 3,
             "unrelated", str(b1))
    specs[-1]["event_id"] = event_id(project, "hard_negative")
    events.append({"event_id": specs[-1]["event_id"], "project_id": project,
                   "type": "hard_negative", "slot": "budget",
                   "current_value": str(b1), "current_status": "unrelated",
                   "history": [], "mentions": [], "open": [], "unresolved": []})
    order += 8

    # Render specs in graph order, with side-chat and distractors between them.
    turns: list[dict] = []
    clock = datetime(2026, 2, 3, 9, 0)
    for n, spec in enumerate(sorted(specs, key=lambda s: s["order"])):
        while rng.random() < 0.45:
            clock += timedelta(minutes=rng.randint(1, 8))
            if rng.random() < 0.55:
                text = rng.choice(SIDE)
            else:
                text = f"Et pour {rng.choice(PROJECTS)}, on en est ou ?"
            turns.append({"turn_id": f"{project}_t{len(turns):03d}",
                          "project_id": project, "meeting_id": project,
                          "turn_index": len(turns),
                          "session": 1 + len(turns) // 35,
                          "speaker": rng.choice(PEOPLE),
                          "timestamp": clock.isoformat(timespec="minutes"),
                          "text": text, "gold_event": None})
        clock += timedelta(minutes=rng.randint(1, 7))
        if len(turns) and len(turns) % 41 == 0:
            clock += timedelta(days=1, hours=2)
        text = spec["text"]
        if rng.random() < 0.20:
            text = rng.choice(["euh ", "attends ", "en fait "]) + text
        turn = {"turn_id": f"{project}_t{len(turns):03d}", "project_id": project,
                "meeting_id": project, "turn_index": len(turns),
                "session": 1 + len(turns) // 35, "speaker": rng.choice(PEOPLE),
                "timestamp": clock.isoformat(timespec="minutes"), "text": text,
                "gold_event": spec["event_id"]}
        turns.append(turn)
        mid = f"{spec['event_id']}:m{len(mentions):03d}"
        mention = {"mention_id": mid, "event_id": spec["event_id"],
                   "turn_id": turn["turn_id"], "slot": spec["slot"],
                   "value": spec["value"], "status": spec["status"],
                   "evidence": text, "owner": spec.get("owner", ""),
                   "deadline": spec.get("deadline", "")}
        mentions.append(mention)
        for ev in events:
            if ev["event_id"] == spec["event_id"]:
                ev["mentions"].append(mid)
                break
        if spec["relation"] not in ("starts", "unrelated"):
            prev = [m for m in mentions if m["event_id"] == spec["event_id"]
                    and m["mention_id"] != mid]
            if prev:
                relations.append({"from_mention_id": prev[-1]["mention_id"],
                                  "to_mention_id": mid, "relation": spec["relation"],
                                  "event_id": spec["event_id"],
                                  "evidence": text})
    # Queries from current state; one unresolved query is preserved.
    for ev in events:
        if ev["type"] == "hard_negative":
            continue
        unresolved = bool(ev["unresolved"])
        queries.append({
            "query_id": f"{project}:q:{ev['type']}",
            "query": TEMPLATES[variant]["question"].format(
                slot=ev["slot"], project=project),
            "gold_event_id": ev["event_id"], "gold_value": ev["current_value"],
            "gold_status": ev["current_status"],
            "answer_expected": "unknown" if unresolved else ev["current_value"],
            "unresolved": unresolved,
            "evidence_turn_ids": [m["turn_id"] for m in mentions
                                  if m["event_id"] == ev["event_id"]],
        })
    return {"project_id": project, "split": split, "template": variant,
            "events": events, "turns": turns, "mentions": mentions,
            "relations": relations, "queries": queries}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/latent_v1")
    ap.add_argument("--groups", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    projects = PROJECTS * ((args.groups // len(PROJECTS)) + 1)
    splits = {"train": projects[:args.groups // 2],
              "validation": projects[args.groups // 2: args.groups // 2 + args.groups // 4],
              "test": projects[args.groups // 2 + args.groups // 4: args.groups]}
    manifest = {"seed": args.seed, "variant_by_split":
                {"train": "A", "validation": "B", "test": "C"},
                "splits": {}}
    for split, names in splits.items():
        all_turns, all_events, all_mentions, all_rels, all_queries = [], [], [], [], []
        for i, name in enumerate(names):
            p = make_project(f"{name}-{split[0]}{i}", i, split, rng)
            all_turns += p["turns"]; all_events += p["events"]
            all_mentions += p["mentions"]; all_rels += p["relations"]
            all_queries += p["queries"]
        d = Path(args.out) / split
        write_jsonl(d / "turns.jsonl", all_turns)
        write_jsonl(d / "events.jsonl", all_events)
        write_jsonl(d / "mentions.jsonl", all_mentions)
        write_jsonl(d / "relations.jsonl", all_rels)
        write_jsonl(d / "queries.jsonl", all_queries)
        manifest["splits"][split] = {
            "projects": len(names), "turns": len(all_turns),
            "events": len(all_events), "mentions": len(all_mentions),
            "relations": len(all_rels), "queries": len(all_queries),
        }
        print(split, manifest["splits"][split])
    (Path(args.out) / "manifest.json").write_text(json.dumps(manifest, indent=1),
                                                   encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
