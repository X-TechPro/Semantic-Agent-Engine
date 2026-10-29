"""Synthetic French meeting utterances labelled with the piscine taxonomy.

Labels come from piscine.md: the eight unit types of section 4, the unit
`status` of the semantic-unit schema, and the seven request intents of
section 5. `unknown` / `none` are added because a meeting is full of sentences
that carry neither a state nor a request.

The generator is built around three ideas that the previous datasets lacked:

1. **Template-disjoint splits.** A family is a phrasing pattern. Whole families
   are assigned to train/val/test, so the test score measures generalisation to
   unseen phrasing rather than recall of a memorised sentence.
2. **Minimal pairs.** The same content with one cue changed: "on valide" vs
   "on n'a pas encore validé", "on part sur 50k" vs "on part sur 50k ou pas".
   A tagger that only reads topic words scores at chance here.
3. **ASR surface.** Lowercase, missing final punctuation, fillers, repeated
   words — the shape of the real transcripts, not clean written French.

Every discriminating cue is lexical ("pas", "plus", "annule", "ou pas",
"je m'engage"), never punctuation, because punctuation is what ASR drops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

KINDS = ("decision", "action", "commitment", "question", "answer", "fact",
         "risk", "idea")
STATUSES = ("proposed", "confirmed", "cancelled", "completed", "unknown")
INTENTS = ("answer", "compare", "dossier", "email", "task", "schedule",
           "clarify", "none")

PEOPLE = [
    "Marie", "Karim", "Sofia", "Jean-Luc", "Awa", "Thomas", "Léa", "Nabil",
    "Chloé", "Idriss", "Camille", "Yasmine", "Hugo", "Fatou", "Étienne",
    "Nadia", "Paul", "Amina", "Rémi", "Inès", "Mehdi", "Clara", "Samir",
    "Julie", "Bakary", "Élodie", "Rachid", "Manon", "Ousmane", "Céline",
    "Grégoire", "Salma", "Vincent", "Aïcha", "Julien", "Mariam", "Bruno",
    "Lina", "Sébastien", "Khadija", "Antoine", "Nour", "Damien", "Estelle",
]
PROJECTS = [
    "Atlas", "Orion", "Zéphyr", "Baobab", "Nimbus", "Sirocco", "Aurora",
    "Mistral", "Kaïros", "Hélios", "Vega", "Oasis", "Delta", "Pixel",
    "Lumen", "Cobalt", "Faucon", "Ivoire", "Saturne", "Éclipse", "Horizon",
    "Tempo", "Nova", "Alizé", "Boréal", "Cèdre", "Dune", "Ébène", "Flux",
    "Granit", "Indigo", "Jade", "Krypton", "Lagune", "Mirage", "Nectar",
]
DELIVERABLES = [
    "le budget", "le devis", "la maquette", "le cahier des charges",
    "la présentation", "le prototype", "le planning", "le contrat",
    "la facture", "le rapport", "le deck", "l'audit", "la roadmap",
    "le support de formation", "l'API", "le jeu de données", "la recette",
    "le backlog", "l'offre commerciale", "la note de cadrage",
    "le plan de charge", "la spécification", "le benchmark", "le budget marketing",
]
DATES = [
    "vendredi", "lundi prochain", "fin novembre", "le 12 décembre",
    "avant la fin du mois", "la semaine prochaine", "mi-janvier",
    "jeudi matin", "le 3 février", "avant les vacances", "début mars",
    "le 15 juin", "après le comité", "en fin de semaine", "le 30 septembre",
]
AMOUNTS = [
    "50k", "cinquante mille euros", "80 000 euros", "12 000 euros",
    "2 500 euros", "300 euros par jour", "1,2 million", "15 %",
    "un budget de 40k", "45 000 euros", "9 000 euros", "120k",
    "30 % du budget", "7 500 euros",
]
COUNTS = ["3", "5", "12", "40", "200", "8", "25", "60", "150", "1 200"]
FILLERS = ["euh", "bah", "ben", "du coup", "en fait", "voilà", "genre",
           "tu vois", "quoi", "bref", "mh", "ah ouais", "ok"]

#: (id, kind, status, forms). `{p}` person, `{proj}` project, `{d}` deliverable,
#: `{date}` date, `{amt}` amount, `{n}` count.
FAMILIES: list[dict] = [
    # --- decision / confirmed ------------------------------------------------
    {"id": "dec_conf_a", "kind": "decision", "status": "confirmed", "forms": [
        "on valide {d} pour {proj}",
        "c'est acté pour {d} sur {proj}",
        "on part sur {amt} pour {proj}",
        "on retient {d} pour {proj}",
        "décision prise, on garde {d}"]},
    {"id": "dec_conf_b", "kind": "decision", "status": "confirmed", "forms": [
        "on confirme {d} pour {proj}",
        "c'est validé, on y va avec {d}",
        "on est d'accord sur {amt} pour {proj}",
        "on garde {d}, c'est entériné"]},
    {"id": "dec_conf_c", "kind": "decision", "status": "confirmed", "forms": [
        "on tranche pour {d}",
        "la décision c'est {d} pour {proj}",
        "on signe sur {amt}",
        "on arrête {d} comme ça"]},
    {"id": "dec_conf_d", "kind": "decision", "status": "confirmed", "forms": [
        "on choisit la version {n} de {d}",
        "on valide {amt} pour {proj}",
        "c'est voté, on prend {d}",
        "on officialise {d} pour {proj}"]},
    # --- decision / proposed -------------------------------------------------
    {"id": "dec_prop_a", "kind": "decision", "status": "proposed", "forms": [
        "je propose de partir sur {amt} pour {proj}",
        "ma proposition c'est {d} pour {proj}",
        "on pourrait retenir {d} pour {date}",
        "je mets sur la table {amt} pour {proj}",
        "on partirait sur {d} pour {proj}"]},
    {"id": "dec_prop_b", "kind": "decision", "status": "proposed", "forms": [
        "on n'a pas encore validé {d}",
        "ce n'est pas acté pour {proj}",
        "rien n'est décidé sur {d}",
        "on n'a pas tranché sur {amt}",
        "la décision sur {d} n'est pas prise",
        "ce n'est pas encore acté pour {proj}",
        "on n'a toujours pas validé {d}",
        "rien n'est acté sur {d}"]},
    {"id": "dec_prop_c", "kind": "decision", "status": "proposed", "forms": [
        "à valider : {d} pour {proj}",
        "on doit encore décider de {d}",
        "il reste à arbitrer {d}",
        "on attend la validation pour {amt}"]},
    {"id": "dec_prop_d", "kind": "decision", "status": "proposed", "forms": [
        "je te propose {d} pour {date}",
        "on peut envisager {amt} pour {proj}",
        "une option c'est {d} pour {proj}",
        "on regarde {d} avant de décider"]},
    # --- decision / cancelled ------------------------------------------------
    {"id": "dec_canc_a", "kind": "decision", "status": "cancelled", "forms": [
        "on annule {d} pour {proj}",
        "finalement on laisse tomber {d}",
        "c'est mort pour {proj}, on arrête",
        "on ne fait pas {d} cette fois",
        "on abandonne {d} pour {proj}"]},
    {"id": "dec_canc_b", "kind": "decision", "status": "cancelled", "forms": [
        "on ne part pas sur {amt}",
        "on retire {d} du périmètre",
        "on stoppe {proj}",
        "on ne valide plus {d}"]},
    {"id": "dec_canc_c", "kind": "decision", "status": "cancelled", "forms": [
        "on ne retient pas {d}",
        "on écarte {d} pour {proj}",
        "c'est annulé pour {date}",
        "on renonce à {d}"]},
    # --- decision / completed ------------------------------------------------
    {"id": "dec_done_a", "kind": "decision", "status": "completed", "forms": [
        "c'est fait, {d} est signé pour {proj}",
        "on a validé {d} hier",
        "{d} est bouclé pour {proj}",
        "la décision est passée, on a signé {d}",
        "c'est réglé pour {proj}"]},
    {"id": "dec_done_b", "kind": "decision", "status": "completed", "forms": [
        "on a tranché la semaine dernière",
        "{d} a été validé pour {proj}",
        "c'est déjà acté depuis {date}",
        "on a officialisé {d}"]},
    # --- action / confirmed --------------------------------------------------
    {"id": "act_conf_a", "kind": "action", "status": "confirmed", "forms": [
        "j'envoie {d} à {p} cet après-midi",
        "je m'occupe de {d} pour {proj}",
        "je relance {p} sur {d}",
        "on envoie {d} à {p} demain",
        "je prépare {d} pour {date}"]},
    {"id": "act_conf_b", "kind": "action", "status": "confirmed", "forms": [
        "je fais le point avec {p} {date}",
        "je reviens vers {p} pour {d}",
        "on lance {d} {date}",
        "je transmets {d} à {p}"]},
    {"id": "act_conf_c", "kind": "action", "status": "confirmed", "forms": [
        "je vérifie {d} avant {date}",
        "je mets à jour {d} pour {proj}",
        "on planifie {d} pour {date}",
        "je demande {d} à {p}"]},
    {"id": "act_conf_d", "kind": "action", "status": "confirmed", "forms": [
        "je m'en occupe",
        "je le fais today".replace("today", "aujourd'hui"),
        "on s'en charge avec {p}",
        "je prends {d} en main"]},
    # --- action / completed --------------------------------------------------
    {"id": "act_done_a", "kind": "action", "status": "completed", "forms": [
        "j'ai envoyé {d} à {p}",
        "{d} est parti chez {p}",
        "j'ai relancé {p} ce matin",
        "on a transmis {d} à {p}",
        "j'ai fait le point avec {p}"]},
    {"id": "act_done_b", "kind": "action", "status": "completed", "forms": [
        "j'ai déjà vérifié {d}",
        "{d} a été mis à jour",
        "on a planifié {d} pour {date}",
        "j'ai demandé {d} à {p}"]},
    # --- commitment / confirmed ----------------------------------------------
    {"id": "com_conf_a", "kind": "commitment", "status": "confirmed", "forms": [
        "je m'engage à livrer {d} {date}",
        "on te livre {d} {date}, promis",
        "je te garantis {d} pour {date}",
        "on s'engage sur {d} pour {proj}",
        "je te promets {d} {date}"]},
    {"id": "com_conf_b", "kind": "commitment", "status": "confirmed", "forms": [
        "{p} s'engage à fournir {d}",
        "on assure {d} pour {date}",
        "c'est promis pour {date}",
        "je m'y engage sur {proj}"]},
    # --- commitment / cancelled ----------------------------------------------
    {"id": "com_canc_a", "kind": "commitment", "status": "cancelled", "forms": [
        "je ne pourrai pas livrer {d} {date}",
        "je ne tiens pas l'engagement sur {d}",
        "on ne pourra pas assurer {d}",
        "{p} ne peut plus fournir {d}"]},
    # --- question ------------------------------------------------------------
    {"id": "q_a", "kind": "question", "status": "unknown", "forms": [
        "est-ce que {p} a validé {d}",
        "est-ce qu'on part sur {amt}",
        "qui s'occupe de {d}",
        "c'est quoi la date pour {proj}",
        "est-ce que {d} est prêt"]},
    {"id": "q_b", "kind": "question", "status": "unknown", "forms": [
        "on part sur {amt} ou pas",
        "on valide {d} ou on attend",
        "{p} est dispo {date}",
        "on a le budget pour {proj} ou pas",
        "combien on a sur {proj}"]},
    {"id": "q_c", "kind": "question", "status": "unknown", "forms": [
        "qu'est-ce qu'on fait pour {d}",
        "comment on gère {proj}",
        "pourquoi on n'a pas {d}",
        "quand est-ce qu'on livre {d}"]},
    {"id": "q_d", "kind": "question", "status": "unknown", "forms": [
        "est-ce que c'est validé pour {proj}",
        "on est d'accord sur {d}",
        "tu as eu {d} de {p}",
        "il faut valider {d} non"]},
    # --- answer --------------------------------------------------------------
    {"id": "ans_a", "kind": "answer", "status": "unknown", "forms": [
        "oui, {p} a validé",
        "non, pas encore",
        "c'est exactement ça",
        "on est d'accord, {d} est prêt",
        "pas de souci pour {date}"]},
    {"id": "ans_b", "kind": "answer", "status": "unknown", "forms": [
        "oui c'est bon pour {proj}",
        "non, {p} n'a pas répondu",
        "effectivement, {d} est validé",
        "d'accord avec toi sur {d}",
        "c'est ça, on part sur {amt}"]},
    {"id": "ans_c", "kind": "answer", "status": "unknown", "forms": [
        "voilà, c'est {p} qui gère",
        "exact, c'est {date}",
        "non, c'est {p} le responsable",
        "oui, on a bien {d}"]},
    # --- fact ----------------------------------------------------------------
    {"id": "fact_a", "kind": "fact", "status": "unknown", "forms": [
        "le budget de {proj} est de {amt}",
        "{proj} compte {n} personnes",
        "la réunion {proj} est {date}",
        "on a {n} clients sur {proj}",
        "{p} est responsable de {proj}"]},
    {"id": "fact_b", "kind": "fact", "status": "unknown", "forms": [
        "le contrat {proj} court jusqu'à {date}",
        "il y a {n} personnes dans l'équipe {proj}",
        "{d} fait {n} pages",
        "le taux horaire est de {amt}"]},
    {"id": "fact_c", "kind": "fact", "status": "unknown", "forms": [
        "{p} travaille sur {proj} depuis {date}",
        "la version actuelle de {d} date de {date}",
        "on est {n} sur {proj}",
        "le serveur {proj} tourne depuis {date}"]},
    {"id": "fact_d", "kind": "fact", "status": "unknown", "forms": [
        "c'est {p} qui a le dossier {proj}",
        "{d} est chez {p}",
        "la prochaine réunion est {date}",
        "le budget restant est de {amt}"]},
    # --- risk ----------------------------------------------------------------
    {"id": "risk_a", "kind": "risk", "status": "unknown", "forms": [
        "si on ne valide pas {date}, on perd {d}",
        "on risque de rater {date} sur {proj}",
        "il y a un risque de retard sur {d}",
        "attention, {proj} peut glisser"]},
    {"id": "risk_b", "kind": "risk", "status": "unknown", "forms": [
        "on peut perdre {amt} sur {proj}",
        "{p} risque de partir si on ne décide pas",
        "sans {d}, on ne tient pas {date}",
        "le risque c'est de dépasser {amt}"]},
    {"id": "risk_c", "kind": "risk", "status": "unknown", "forms": [
        "ça peut bloquer {proj}",
        "on n'aura pas {d} à temps",
        "si {p} ne répond pas, {proj} glisse",
        "il y a un vrai risque sur {date}"]},
    # --- idea ----------------------------------------------------------------
    {"id": "idea_a", "kind": "idea", "status": "proposed", "forms": [
        "et si on découpait {d} en deux",
        "on pourrait tester une version light de {d}",
        "une piste serait de décaler {proj} {date}",
        "et si {p} prenait {d}"]},
    {"id": "idea_b", "kind": "idea", "status": "proposed", "forms": [
        "on peut imaginer un format plus court pour {d}",
        "une autre idée c'est de fusionner {proj} et {proj2}",
        "et si on faisait ça en interne",
        "on pourrait commencer par {d}"]},
    {"id": "idea_c", "kind": "idea", "status": "proposed", "forms": [
        "idée : faire {d} en {n} étapes",
        "on pourrait demander à {p} de regarder",
        "et si on changeait l'ordre de {d}",
        "ça vaudrait le coup de tester {d}"]},
    # --- intents (requests addressed to the assistant) -----------------------
    {"id": "int_answer_a", "kind": None, "status": None, "intent": "answer", "forms": [
        "qu'est-ce qu'on a décidé sur {proj}",
        "rappelle-moi la dernière décision sur {d}",
        "quelle est la date validée pour {proj}",
        "qu'est-ce qui est acté pour {proj}"]},
    {"id": "int_answer_b", "kind": None, "status": None, "intent": "answer", "forms": [
        "dis-moi où on en est sur {d}",
        "c'est quoi la décision actuelle sur {proj}",
        "résume-moi ce qui est validé pour {proj}"]},
    {"id": "int_compare_a", "kind": None, "status": None, "intent": "compare", "forms": [
        "qu'est-ce qui a changé sur {proj} depuis la dernière réunion",
        "compare les deux dernières versions de {d}",
        "qu'est-ce qui a bougé entre {date} et aujourd'hui"]},
    {"id": "int_compare_b", "kind": None, "status": None, "intent": "compare", "forms": [
        "montre-moi les différences sur {d}",
        "compare {proj} et {proj2}",
        "qu'est-ce qui a évolué sur le budget de {proj}"]},
    {"id": "int_dossier_a", "kind": None, "status": None, "intent": "dossier", "forms": [
        "fais-moi un dossier sur {proj}",
        "prépare une synthèse sur {proj} avec les sources",
        "monte un dossier complet sur {d}"]},
    {"id": "int_dossier_b", "kind": None, "status": None, "intent": "dossier", "forms": [
        "rassemble tout ce qu'on a sur {proj}",
        "fais un état des lieux sur {proj}"]},
    {"id": "int_email_a", "kind": None, "status": None, "intent": "email", "forms": [
        "écris à {p} pour demander {d}",
        "envoie un mail à {p} sur {proj}",
        "rédige un message à {p} pour {d}"]},
    {"id": "int_email_b", "kind": None, "status": None, "intent": "email", "forms": [
        "prépare un email pour {p} au sujet de {d}",
        "écris à la personne qui gère {proj}"]},
    {"id": "int_task_a", "kind": None, "status": None, "intent": "task", "forms": [
        "prépare une tâche pour {p} sur {d}",
        "crée une tâche pour relancer {p}",
        "mets une tâche à {p} pour {d}"]},
    {"id": "int_task_b", "kind": None, "status": None, "intent": "task", "forms": [
        "ajoute une tâche : envoyer {d} à {p}",
        "planifie une tâche de vérification sur {d}"]},
    {"id": "int_schedule_a", "kind": None, "status": None, "intent": "schedule", "forms": [
        "organise un point avec {p} {date}",
        "planifie une réunion {proj} {date}",
        "bloque un créneau avec {p} {date}"]},
    {"id": "int_schedule_b", "kind": None, "status": None, "intent": "schedule", "forms": [
        "prépare un point {date} sur {proj}",
        "cale une réunion de suivi {date}"]},
    {"id": "int_clarify_a", "kind": None, "status": None, "intent": "clarify", "forms": [
        "envoie ça à la bonne personne",
        "occupe-toi de ça rapidement",
        "fais le nécessaire"]},
    {"id": "int_clarify_b", "kind": None, "status": None, "intent": "clarify", "forms": [
        "transfère ça à qui de droit",
        "règle ça au plus vite"]},
]

#: Contrasts: same content, one cue changed. `split` decides whether the
#: phrasing teaches the cue (train) or tests an unseen phrasing (test).
#: Every cue family has at least one training variant, so a failure on the test
#: variant means the cue was not learned, not that it was never seen.
PAIR_GROUPS: list[dict] = [
    {"group": "decision_vs_question", "field": "kind", "split": "train",
     "a": "on part sur {d}", "la": "decision",
     "b": "on part sur {d} ou pas", "lb": "question"},
    {"group": "decision_vs_question_b", "field": "kind", "split": "test",
     "a": "on valide {d}", "la": "decision",
     "b": "est-ce qu'on valide {d}", "lb": "question"},
    {"group": "idea_vs_decision", "field": "kind", "split": "train",
     "a": "on part sur {d}", "la": "decision",
     "b": "on pourrait partir sur {d}", "lb": "idea"},
    {"group": "idea_vs_decision_b", "field": "kind", "split": "test",
     "a": "on retient {d}", "la": "decision",
     "b": "et si on retenait {d}", "lb": "idea"},
    {"group": "fact_vs_risk", "field": "kind", "split": "train",
     "a": "{proj} glisse sur {date}", "la": "fact",
     "b": "{proj} risque de glisser sur {date}", "lb": "risk"},
    {"group": "fact_vs_risk_b", "field": "kind", "split": "test",
     "a": "{proj} est en retard", "la": "fact",
     "b": "il y a un risque que {proj} soit en retard", "lb": "risk"},
    {"group": "answer_vs_fact", "field": "kind", "split": "train",
     "a": "oui, {d} est validé", "la": "answer",
     "b": "{d} est validé", "lb": "decision"},
    {"group": "answer_vs_fact_b", "field": "kind", "split": "test",
     "a": "voilà, {d} est validé", "la": "answer",
     "b": "{d} est validé", "lb": "decision"},
    {"group": "action_vs_commitment", "field": "kind", "split": "train",
     "a": "j'envoie {d} {date}", "la": "action",
     "b": "je m'engage à envoyer {d} {date}", "lb": "commitment"},
    {"group": "action_vs_commitment_b", "field": "kind", "split": "test",
     "a": "je transmets {d} {date}", "la": "action",
     "b": "je te garantis {d} {date}", "lb": "commitment"},
    {"group": "confirmed_vs_proposed", "field": "status", "split": "train",
     "a": "on valide {d} pour {proj}", "la": "confirmed",
     "b": "on n'a pas encore validé {d} pour {proj}", "lb": "proposed"},
    {"group": "confirmed_vs_proposed_b", "field": "status", "split": "test",
     "a": "c'est acté pour {d}", "la": "confirmed",
     "b": "ce n'est pas encore acté pour {d}", "lb": "proposed"},
    {"group": "confirmed_vs_cancelled", "field": "status", "split": "train",
     "a": "on valide {d} pour {proj}", "la": "confirmed",
     "b": "on annule {d} pour {proj}", "lb": "cancelled"},
    {"group": "confirmed_vs_cancelled_b", "field": "status", "split": "test",
     "a": "on garde {d}", "la": "confirmed",
     "b": "on ne garde pas {d}", "lb": "cancelled"},
    {"group": "confirmed_vs_completed", "field": "status", "split": "train",
     "a": "on valide {d} pour {proj}", "la": "confirmed",
     "b": "on a déjà validé {d} pour {proj}", "lb": "completed"},
    {"group": "confirmed_vs_completed_b", "field": "status", "split": "test",
     "a": "on valide {d}", "la": "confirmed",
     "b": "c'est déjà fait pour {d}", "lb": "completed"},
]


def _assign_splits(families: list[dict], seed: int = 13) -> dict[str, str]:
    """Family -> split. Whole phrasings move, never individual sentences."""
    groups: dict[tuple, list[str]] = {}
    for f in families:
        key = (f.get("kind"), f.get("status"), f.get("intent"))
        groups.setdefault(key, []).append(f["id"])
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for key, ids in groups.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        n = len(ids)
        if n >= 3:
            n_test, n_dev = 1, 1
        elif n >= 2:
            n_test, n_dev = 1, 0
        else:
            n_test, n_dev = 0, 0
        for i, fid in enumerate(ids):
            out[fid] = ("test" if i < n_test else
                        "val" if i < n_test + n_dev else "train")
    return out


def _draw(rng: random.Random, people, projects) -> dict:
    proj = rng.choice(projects)
    return {"p": rng.choice(people), "proj": proj,
            "proj2": rng.choice([p for p in projects if p != proj]),
            "d": rng.choice(DELIVERABLES), "date": rng.choice(DATES),
            "amt": rng.choice(AMOUNTS), "n": rng.choice(COUNTS)}


def _fill(form: str, rng: random.Random, people=None, projects=None,
          slots: dict | None = None) -> str:
    s = slots or _draw(rng, people or PEOPLE, projects or PROJECTS)
    return form.format(**s)


_ELISION = [("de le ", "du "), ("de les ", "des "), ("à le ", "au "),
            ("à les ", "aux "), ("de l'", "de l'")]
_SUBJECTS = {"on", "je", "j'ai", "il", "elle", "nous", "vous", "ils", "c'est",
             "ça", "y'a"}


def _surface(text: str, rng: random.Random) -> str:
    """ASR-shaped surface: no clean punctuation, fillers, repetitions.

    Fillers only go where a speaker can pause — at the start, or right after
    the subject — never inside a noun phrase: "la note de en fait cadrage" is
    not something an ASR produces and it teaches the wrong thing.
    """
    t = text.lower().rstrip("?!.,") if rng.random() < 0.95 else text.lower()
    w = t.split()
    if w and rng.random() < 0.22:
        w.insert(0, rng.choice(FILLERS))
    elif len(w) > 2 and w[1] in _SUBJECTS and rng.random() < 0.15:
        w.insert(2, rng.choice(FILLERS))
    if len(w) > 3 and rng.random() < 0.10:
        i = rng.randrange(len(w) - 1)
        if w[i] not in ("de", "du", "des", "le", "la", "les", "à", "au", "aux"):
            w.insert(i, w[i])
    t = " ".join(w)
    t = re.sub(r"\bde les\b", "des", t)
    t = re.sub(r"\bde le\b", "du", t)
    t = re.sub(r"\bà les\b", "aux", t)
    t = re.sub(r"\bà le\b", "au", t)
    return " ".join(t.split())


def build(per_family: int = 220, seed: int = 42) -> dict[str, list[dict]]:
    splits = _assign_splits(FAMILIES)
    rng = random.Random(seed)
    # Unseen names in test: a model that keyed on entities cannot hide here.
    held_people = set(PEOPLE[-8:])
    held_projects = set(PROJECTS[-6:])
    data: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for fam in FAMILIES:
        split = splits[fam["id"]]
        people = [p for p in PEOPLE if (p in held_people) == (split == "test")] \
            or PEOPLE
        projects = [p for p in PROJECTS
                    if (p in held_projects) == (split == "test")] or PROJECTS
        for _ in range(per_family):
            form = rng.choice(fam["forms"])
            text = _surface(_fill(form, rng, people, projects), rng)
            tasks = (["intent"] if fam.get("intent")
                     else ["kind", "status"])
            data[split].append({
                "text": text,
                "kind": fam.get("kind"),
                "status": fam.get("status"),
                "intent": fam.get("intent"),
                "tasks": tasks,
                "family": fam["id"],
                "source": "synthetic",
            })
    for g in PAIR_GROUPS:
        if g["split"] != "train":
            continue
        for _ in range(per_family):
            slots = _draw(rng, PEOPLE, PROJECTS)
            for side, key in (("a", "la"), ("b", "lb")):
                row = {"text": _surface(_fill(g[side], rng, slots=slots), rng),
                       g["field"]: g[key],
                       "kind": g[key] if g["field"] == "kind" else None,
                       "status": g[key] if g["field"] == "status" else None,
                       "intent": None, "tasks": [g["field"]],
                       "family": g["group"], "source": "synthetic"}
                # The contrast is the thing the tagger is judged on; one copy
                # per template family would leave it ~15% of the batch.
                data["train"] += [row, dict(row)]
    for split in data:
        rng.shuffle(data[split])
    return data


def build_pairs(per_group: int = 40, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for g in PAIR_GROUPS:
        if g["split"] != "test":
            continue
        for _ in range(per_group):
            slots = _draw(rng, PEOPLE, PROJECTS)
            a = _surface(_fill(g["a"], rng, slots=slots), rng)
            b = _surface(_fill(g["b"], rng, slots=slots), rng)
            out.append({"group": g["group"], "field": g["field"],
                        "a": a, "la": g["la"], "b": b, "lb": g["lb"]})
    return out


def stats(rows: dict[str, list[dict]]) -> dict:
    out: dict = {}
    for split, items in rows.items():
        out[split] = {"n": len(items)}
        for field in ("kind", "status", "intent"):
            counts: dict[str, int] = {}
            for it in items:
                if it.get(field):
                    counts[it[field]] = counts.get(it[field], 0) + 1
            out[split][field] = dict(sorted(counts.items()))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth/tags")
    ap.add_argument("--per-family", type=int, default=220)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    data = build(args.per_family, args.seed)
    for split, items in data.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for it in build_pairs(args.pairs, args.seed + 1):
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(
        json.dumps(stats(data), indent=1), encoding="utf-8")
    print(json.dumps(stats(data), indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
