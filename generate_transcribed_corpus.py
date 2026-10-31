#!/usr/bin/env python3
"""Generate the transcribed evaluation corpus (30 meetings) and its query set.

Style: messy French transcription (fillers, false starts, no final punctuation,
lowercase). Subjects are deliberately unrelated across projects; meetings inside
a project reference the previous session by ISO date, and each project's first
meeting references the previous project's first session. Those date references
are what the graph turns into cross-meeting edges.

Every meeting has three decisions, so count questions have a multi-message gold
set. Queries are written at three difficulty levels: last/first-decision,
count/multipart, and commitment/cross-meeting.
"""
from __future__ import annotations

import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 20261031
NAMES = ["alice", "bruno", "chloé", "david", "emma", "farid", "gaëlle", "hugo",
         "inès", "karim", "léa", "mehdi", "nina", "omar", "paul", "sofiane"]
FILLERS = ["ouais", "ok ok", "attends", "mmh", "genre", "voilà quoi", "je sais pas",
           "on verra", "bref", "enfin", "d'accord", "ah oui", "carrement", "hmm"]

PROJECTS = [
    {
        "id": "mobile", "name": "Refonte mobile",
        "focus": ["les parcours de connexion et d'accueil",
                  "le mode hors ligne et la synchronisation",
                  "la bêta et les retours utilisateurs"],
        "decisions": [
            "refondre l'écran d'accueil avec une seule action principale",
            "supprimer l'écran intermédiaire de connexion",
            "mesurer le temps d'ouverture avant toute nouvelle animation",
            "mettre en cache les dix derniers écrans consultés",
            "limiter la synchronisation aux réseaux wi-fi par défaut",
            "afficher un bandeau explicite quand l'application est hors ligne",
            "ouvrir la bêta à cinquante utilisateurs internes",
            "geler les nouvelles fonctionnalités pendant deux semaines",
            "collecter les retours via un formulaire intégré",
        ],
        "risks": ["le temps de chargement dépasse trois secondes sur les téléphones anciens",
                  "la synchronisation vide la batterie quand le réseau est instable",
                  "les utilisateurs ne comprennent pas pourquoi l'application passe hors ligne",
                  "la bêta révèle trop de bugs bloquants en une semaine"],
        "objections": ["le nouvel accueil casse les habitudes des utilisateurs réguliers",
                       "le mode hors ligne ajoute une complexité difficile à tester",
                       "cinquante testeurs c'est trop peu pour conclure",
                       "geler les fonctionnalités va retarder la feuille de route"],
        "commitments": ["préparer les maquettes de l'écran d'accueil",
                        "écrire les tests de synchronisation hors ligne",
                        "recruter les cinquante bêta-testeurs",
                        "consolider les retours de la bêta"],
        "facts": ["l'écran d'accueil actuel demande quatre clics avant la première action",
                  "la version actuelle pèse quarante mégaoctets",
                  "les crashs concernent surtout les téléphones de plus de quatre ans",
                  "le taux de conversion de l'inscription est de trente pour cent"],
    },
    {
        "id": "data", "name": "Migration données",
        "focus": ["l'inventaire des bases et des dépendances",
                  "le schéma cible et les sauvegardes",
                  "la bascule en production et le retour arrière"],
        "decisions": [
            "inventorier toutes les tables utilisées par les services",
            "geler les écritures pendant la fenêtre de migration",
            "conserver sqlite pour les prototypes internes",
            "créer un schéma cible avec postgresql",
            "tester la restauration complète avant la bascule",
            "doubler la fréquence des sauvegardes pendant un mois",
            "basculer d'abord le service de facturation",
            "prévoir un retour arrière de trente minutes maximum",
            "surveiller les erreurs pendant deux semaines après la bascule",
        ],
        "risks": ["une table oubliée bloque le service de facturation",
                  "la restauration prend plus de temps que prévu",
                  "la double écriture duplique des paiements",
                  "la bascule sature le réseau interne"],
        "objections": ["la fenêtre de gel des écritures est trop longue",
                       "postgresql ajoute une charge d'exploitation",
                       "le retour arrière n'a jamais été testé pour de vrai",
                       "deux semaines de surveillance c'est trop court"],
        "commitments": ["rédiger l'inventaire des tables",
                        "écrire le script de restauration",
                        "préparer le plan de retour arrière",
                        "superviser la bascule de facturation"],
        "facts": ["la base principale contient cent vingt tables",
                  "la sauvegarde complète prend aujourd'hui quarante minutes",
                  "deux services écrivent encore directement dans sqlite",
                  "la facturation représente un tiers du trafic"],
    },
    {
        "id": "secu", "name": "Sécurité applicative",
        "focus": ["la revue des accès administrateurs",
                  "la rotation des secrets et des jetons",
                  "la réponse à incident et la communication"],
        "decisions": [
            "imposer la double authentification aux administrateurs",
            "retirer les accès des comptes inactifs depuis six mois",
            "centraliser les secrets dans un coffre dédié",
            "faire tourner les jetons de production tous les quatre-vingt-dix jours",
            "interdire les secrets dans les variables de build",
            "désigner un responsable de la réponse à incident",
            "préparer une procédure de communication en cas de fuite",
            "tester la révocation d'un jeton en condition réelle",
            "revoir les journaux d'accès chaque semaine",
        ],
        "risks": ["un jeton partagé dans un dépôt public expose les données",
                  "la rotation automatique interrompt les intégrations partenaires",
                  "la procédure d'incident n'est pas connue des équipes",
                  "les journaux sont conservés trop peu de temps"],
        "objections": ["la double authentification va ralentir les interventions",
                       "tous les comptes inactifs ne sont pas réellement inactifs",
                       "le coffre de secrets ajoute une dépendance critique",
                       "revoir les journaux chaque semaine demande du temps"],
        "commitments": ["auditer les accès administrateurs",
                        "migrer les secrets vers le coffre",
                        "rédiger la procédure d'incident",
                        "tester la révocation d'un jeton"],
        "facts": ["quarante comptes disposent encore d'un accès administrateur",
                  "le dernier audit des accès date de huit mois",
                  "douze secrets sont stockés dans des scripts",
                  "les journaux sont conservés trente jours"],
    },
    {
        "id": "rh", "name": "Recrutement",
        "focus": ["l'ouverture des postes et les profils",
                  "la grille d'entretien et l'évaluation",
                  "l'intégration et la période d'essai"],
        "decisions": [
            "ouvrir deux postes d'ingénieur backend",
            "ouvrir un poste de designer produit",
            "harmoniser la grille d'entretien sur quatre niveaux",
            "ajouter un exercice pratique de deux heures",
            "former tous les intervieweurs avant la prochaine session",
            "prévoir un parrain pour chaque nouvelle recrue",
            "fixer un point de suivi à trente jours",
            "réduire la période d'essai à deux mois pour les profils seniors",
            "centraliser les retours d'entretien dans un seul outil",
        ],
        "risks": ["les profils seniors refusent le salaire proposé",
                  "la grille d'entretien reste subjective",
                  "la nouvelle recrue est perdue sans parrain",
                  "les entretiens techniques prennent trop de temps"],
        "objections": ["deux postes backend c'est trop ambitieux",
                       "l'exercice pratique décourage les candidats",
                       "le point à trente jours arrive trop tard",
                       "la réduction de la période d'essai est risquée"],
        "commitments": ["rédiger les fiches de poste backend",
                        "former les intervieweurs",
                        "désigner les parrains",
                        "centraliser les retours d'entretien"],
        "facts": ["douze candidatures sont en cours",
                  "le délai moyen de recrutement est de six semaines",
                  "trois intervieweurs n'ont jamais été formés",
                  "le taux d'acceptation des offres est de soixante pour cent"],
    },
    {
        "id": "pricing", "name": "Pricing",
        "focus": ["l'étude des plans et des usages",
                  "les grands comptes et les remises",
                  "le lancement tarifaire et la communication"],
        "decisions": [
            "conserver trois plans publics",
            "plafonner les remises grands comptes à quinze pour cent",
            "facturer les sièges supplémentaires au prorata",
            "tester un plan annuel avec deux mois offerts",
            "communiquer le changement trente jours avant",
            "préserver les tarifs des clients existants pendant un an",
            "facturer l'usage au-delà du quota par tranches",
            "lancer le nouveau tarif d'abord sur les nouveaux comptes",
            "revoir les remises après trois mois",
        ],
        "risks": ["la hausse du plan provoque des résiliations",
                  "les grands comptes négocient les remises au cas par cas",
                  "la facturation au prorata est incomprise",
                  "le lancement se chevauche avec la refonte mobile"],
        "objections": ["trois plans c'est encore trop compliqué",
                       "un an de tarif gelé coûte trop cher",
                       "les tranches d'usage sont difficiles à expliquer",
                       "revoir les remises après trois mois crée de l'instabilité"],
        "commitments": ["préparer la grille des plans",
                        "modéliser l'impact des remises",
                        "rédiger la communication client",
                        "configurer la facturation au prorata"],
        "facts": ["le plan médian représente la moitié des revenus",
                  "les grands comptes pèsent un quart du chiffre d'affaires",
                  "la dernière hausse tarifaire date de deux ans",
                  "le taux de résiliation est de quatre pour cent"],
    },
    {
        "id": "support", "name": "Support client",
        "focus": ["la file d'attente et les délais",
                  "la base de connaissances",
                  "l'escalade et les astreintes"],
        "decisions": [
            "ajouter un second niveau de tri dès le premier message",
            "fixer un objectif de première réponse à deux heures",
            "migrer la base de connaissances vers un espace public",
            "écrire vingt articles prioritaires",
            "créer une astreinte partagée le week-end",
            "escalader automatiquement les incidents critiques",
            "mesurer la satisfaction après chaque résolution",
            "supprimer les réponses types trop génériques",
            "revoir la file d'attente chaque matin",
        ],
        "risks": ["la file explose le lundi matin",
                  "les articles publics exposent des informations internes",
                  "l'astreinte partagée fatigue les équipes",
                  "les réponses types font baisser la satisfaction"],
        "objections": ["deux heures de première réponse c'est irréaliste",
                       "la base publique demande une relecture juridique",
                       "l'astreinte week-end coûte trop cher",
                       "la satisfaction après résolution est biaisée"],
        "commitments": ["rédiger les vingt articles prioritaires",
                        "configurer l'escalade automatique",
                        "organiser l'astreinte week-end",
                        "mesurer la satisfaction sur un mois"],
        "facts": ["le délai moyen de première réponse est de six heures",
                  "la base contient soixante articles obsolètes",
                  "les incidents critiques représentent cinq pour cent du volume",
                  "la satisfaction moyenne est de trois virgule huit sur cinq"],
    },
    {
        "id": "a11y", "name": "Accessibilité",
        "focus": ["l'audit wcag et les écarts",
                  "la navigation clavier et les lecteurs d'écran",
                  "la validation et la formation"],
        "decisions": [
            "corriger les contrastes sur les écrans principaux",
            "ajouter des libellés explicites à tous les champs",
            "rendre la navigation clavier complète sur le parcours d'achat",
            "tester avec deux lecteurs d'écran",
            "former l'équipe aux bases de l'accessibilité",
            "ajouter l'accessibilité aux critères de revue",
            "publier une déclaration d'accessibilité",
            "planifier un nouvel audit dans six mois",
            "corriger les formulaires avant la prochaine version",
        ],
        "risks": ["certaines corrections cassent le design existant",
                  "les lecteurs d'écran se comportent différemment selon les navigateurs",
                  "l'équipe manque de temps pour se former",
                  "la déclaration promet plus que ce qui est fait"],
        "objections": ["les contrastes imposés sont trop stricts",
                       "tester deux lecteurs d'écran prend trop de temps",
                       "la formation doit attendre la fin du trimestre",
                       "un audit tous les six mois c'est insuffisant"],
        "commitments": ["corriger les contrastes",
                        "ajouter les libellés de champs",
                        "former l'équipe",
                        "rédiger la déclaration d'accessibilité"],
        "facts": ["l'audit relève quarante-deux écarts",
                  "douze formulaires n'ont pas de libellé",
                  "le parcours d'achat n'est pas utilisable au clavier",
                  "deux développeurs ont déjà suivi une formation"],
    },
    {
        "id": "infra", "name": "Infrastructure",
        "focus": ["la capacité des serveurs",
                  "les sauvegardes et la restauration",
                  "le réseau et la latence"],
        "decisions": [
            "ajouter deux serveurs pour la période de pointe",
            "réserver trente pour cent de capacité libre",
            "tester une restauration complète chaque mois",
            "déplacer les sauvegardes dans une seconde région",
            "changer le fournisseur de réseau principal",
            "mesurer la latence depuis trois villes",
            "activer la limitation de débit sur l'api publique",
            "planifier une fenêtre de maintenance mensuelle",
            "documenter le plan de reprise",
        ],
        "risks": ["la pointe de trafic dépasse la capacité prévue",
                  "la restauration mensuelle mobilise trop d'équipe",
                  "le nouveau réseau ne couvre pas certaines zones",
                  "la limitation de débit bloque des clients légitimes"],
        "objections": ["deux serveurs supplémentaires coûtent trop cher",
                       "trente pour cent de marge c'est excessif",
                       "changer de réseau pendant la haute saison est risqué",
                       "une maintenance mensuelle fatigue les équipes"],
        "commitments": ["commander deux serveurs",
                        "écrire le plan de restauration",
                        "tester la latence depuis trois villes",
                        "documenter le plan de reprise"],
        "facts": ["la capacité actuelle est utilisée à quatre-vingt-dix pour cent en pointe",
                  "la dernière restauration date de six mois",
                  "la latence moyenne est de cent quatre-vingts millisecondes",
                  "une panne réseau a duré vingt minutes le mois dernier"],
    },
    {
        "id": "partner", "name": "Partenariats",
        "focus": ["la shortlist des partenaires",
                  "le contrat et les conditions",
                  "l'intégration api et le support"],
        "decisions": [
            "retenir deux partenaires pour un pilote",
            "écarter le partenaire sans support en français",
            "limiter le pilote à trois mois",
            "signer un contrat avec clause de sortie à trente jours",
            "exiger une documentation api à jour",
            "nommer un référent technique par partenaire",
            "prévoir un canal de support partagé",
            "mesurer le volume traité pendant le pilote",
            "décider de la suite après le pilote",
        ],
        "risks": ["le partenaire ne fournit pas la documentation",
                  "le pilote ne couvre pas assez d'usage",
                  "la clause de sortie est refusée",
                  "le support partagé ralentit les réponses"],
        "objections": ["deux partenaires c'est trop pour un pilote",
                       "trois mois ne suffisent pas pour décider",
                       "la clause de sortie va bloquer la négociation",
                       "un référent par partenaire mobilise trop de monde"],
        "commitments": ["préparer la shortlist",
                        "négocier la clause de sortie",
                        "écrire la documentation d'intégration",
                        "mettre en place le canal de support"],
        "facts": ["six partenaires ont répondu à l'appel",
                  "deux partenaires seulement proposent un support en français",
                  "le pilote visera mille transactions par jour",
                  "le contrat type prévoit une sortie à soixante jours"],
    },
    {
        "id": "legal", "name": "Conformité",
        "focus": ["le registre rgpd et les finalités",
                  "les sous-traitants et les transferts",
                  "l'audit et les preuves"],
        "decisions": [
            "mettre à jour le registre des traitements",
            "classer les données par finalité et durée",
            "signer un accord de sous-traitance avec chaque fournisseur",
            "interdire les transferts hors union européenne sans validation",
            "nommer un référent conformité",
            "documenter les mesures de sécurité par traitement",
            "planifier un audit interne",
            "corriger les mentions légales du site",
            "former les équipes à la minimisation des données",
        ],
        "risks": ["un sous-traitant refuse l'accord",
                  "le registre reste incomplet",
                  "les transferts existants ne sont pas tous connus",
                  "l'audit révèle des traitements non déclarés"],
        "objections": ["le registre est trop lourd à maintenir",
                       "interdire tous les transferts bloque des outils",
                       "un référent conformité à temps partiel ne suffit pas",
                       "la formation arrive trop tard"],
        "commitments": ["mettre à jour le registre",
                        "signer les accords sous-traitants",
                        "documenter les mesures de sécurité",
                        "former les équipes"],
        "facts": ["le registre recense vingt-huit traitements",
                  "douze sous-traitants traitent des données personnelles",
                  "trois outils transfèrent des données hors union européenne",
                  "le dernier audit interne date de deux ans"],
    },
]

SUBJECTS = {"mobile": ["Audit UX", "Mode hors ligne", "Bêta"], "data": ["Inventaire des bases", "Schéma cible", "Bascule production"], "secu": ["Revue des accès", "Rotation des secrets", "Réponse à incident"], "rh": ["Ouverture des postes", "Grille d'entretien", "Intégration"], "pricing": ["Étude des plans", "Grands comptes", "Lancement tarifaire"], "support": ["File d'attente", "Base de connaissances", "Escalade"], "a11y": ["Audit WCAG", "Navigation clavier", "Validation"], "infra": ["Capacité serveurs", "Sauvegardes", "Réseau"], "partner": ["Shortlist", "Contrat", "Intégration API"], "legal": ["Registre RGPD", "Sous-traitants", "Audit"]}

LAST_Q = ["Quel est le dernier message de la réunion ?",
          "Qu'est-ce qui a été dit en tout dernier ?",
          "Quel message clôt la séance ?"]
FIRST_Q = ["Quelle a été la première décision prise pendant la réunion ?",
           "Sur quoi s'est-on mis d'accord au début ?",
           "Quelle décision a ouvert la réunion ?"]
COUNT_Q = ["Combien de décisions ont été prises pendant cette réunion ?",
           "On a acté combien de choses aujourd'hui ?",
           "Quel est le nombre de décisions prises ?"]
MULTI_Q = ["Quel est le dernier message, qui l'a écrit et à quelle heure ?",
           "Qui a envoyé le dernier message et quand ?",
           "Dernier message : auteur et horodatage ?"]
COMMIT_Q = ["Qui s'est engagé à {x} ?", "Qui a pris en charge {x} ?", "Qui doit s'occuper de {x} ?"]
CROSS_Q = ["Qu'avait-on décidé le {date} à propos de {x} ?",
           "Quelle était la décision du {date} sur {x} ?",
           "Rappelle-moi ce qui a été acté le {date} pour {x}."]


def messy(text: str, rng: random.Random) -> str:
    text = text.strip()
    if rng.random() < 0.55:
        text = text[0].lower() + text[1:]
    if rng.random() < 0.35:
        text = rng.choice(["euh ", "bah ", "alors ", "attends "]) + text
    if rng.random() < 0.30:
        words = text.split()
        if len(words) > 4:
            text = " ".join(words[:2] + [words[1]] + words[2:])
    if rng.random() < 0.45:
        text = text.rstrip(".")
    if rng.random() < 0.25:
        text = text.replace(", ", " ... ")
    if rng.random() < 0.25:
        text += rng.choice([" voilà quoi", " enfin bref", " tu vois", " hein"])
    return text


def main():
    rng = random.Random(SEED)
    meetings = []
    queries = []
    project_first = {}
    previous_project = None

    for pi, project in enumerate(PROJECTS):
        project_meetings = []
        for k in range(3):
            mid = f"tr-{project['id']}-r{k + 1}"
            mdate = date(2026, 1, 5) + timedelta(days=(pi * 3 + k) * 7)
            subject = SUBJECTS[project["id"]][k]
            title = f"{project['name']} · {subject}"
            focus = project["focus"][k]
            decisions = project["decisions"][3 * k:3 * k + 3]
            risks = project["risks"][(2 * k) % 4:(2 * k) % 4 + 2]
            if len(risks) < 2:
                risks += project["risks"][:2 - len(risks)]
            objections = project["objections"][(2 * k) % 4:(2 * k) % 4 + 2]
            if len(objections) < 2:
                objections += project["objections"][:2 - len(objections)]
            commitments = project["commitments"][(2 * k) % 4:(2 * k) % 4 + 2]
            if len(commitments) < 2:
                commitments += project["commitments"][:2 - len(commitments)]
            facts = [project["facts"][(2 * k + i) % 4] for i in range(4)]

            speakers = rng.sample(NAMES, 6)
            items = []  # (text, kind)

            def add(text, kind=None):
                items.append((text, kind))

            add(f"euh salut à tous on démarre la réunion {project['name']} du {mdate.isoformat()} hein")
            add("attends je vérifie le son ... voilà c'est bon on t'entend")
            reference = None
            if k > 0:
                prev = project_meetings[k - 1]
                reference = (prev["date"], prev["subject"],
                             [f"{prev['meeting_id']}:{x:03d}" for x in prev["decisions"]], prev["title"])
                add(f"comme la dernière fois le {prev['date']} on avait acté {prev['decision_text']}")
            elif previous_project is not None:
                date0, subject0, decision0, decision0_ids, title0 = project_first[previous_project["id"]]
                reference = (date0, subject0, decision0_ids, title0)
                add(f"comme dans la réunion du {date0} sur {previous_project['name']} on avait noté que {decision0}")
            else:
                add(f"bon on est là pour parler de {focus}")
            add(f"aujourd'hui faut qu'on parle de {focus} enfin surtout de ce qui bloque")
            add(f"bah en fait {facts[0]}")
            add(f"question: est-ce que {facts[1]}")
            add(f"décision: {decisions[0]} — c'est acté, on note", "decision")
            add(f"je prends en charge {commitments[0]} et je reviens avec le résultat", "commitment")
            add(rng.choice(FILLERS))
            add(f"je veux juste signaler une réserve: {objections[0]}")
            add(f"le risque c'est que {risks[0]}")
            add(f"on avait aussi noté que {facts[2]}")
            add(rng.choice(FILLERS))
            add(f"décision: {decisions[1]} — on part là-dessus", "decision")
            add(rng.choice(FILLERS))
            add(f"je suis pas d'accord sur {objections[1]} mais bon")
            add(f"et si {risks[1]} ?")
            add(f"je prends en charge {commitments[1]} aussi", "commitment")
            add(rng.choice(FILLERS))
            add(f"pour rappel {facts[3]}")
            add(rng.choice(FILLERS))
            add(f"décision: {decisions[2]} — dernière ligne droite", "decision")
            add(rng.choice(FILLERS))
            add("on est bien d'accord que ça change pas la date de livraison ?")
            add(rng.choice(FILLERS))
            add("mmh faut que je vérifie un truc avec l'équipe")
            add(rng.choice(FILLERS))
            add(f"donc récap on a acté {decisions[0]} puis {decisions[2]}")
            add(rng.choice(FILLERS))
            add("je vous envoie le compte rendu dans la journée")
            add(rng.choice(FILLERS))
            add(f"bon on s'arrête là, prochaine réunion le {(mdate + timedelta(days=7)).isoformat()}, merci à tous")

            while len(items) < 39:
                add(rng.choice(FILLERS))
            add(f"bon on s'arrête là, prochaine réunion le {(mdate + timedelta(days=7)).isoformat()}, merci à tous")

            decision_ids = [i for i, (_, kind) in enumerate(items, 1) if kind == "decision"]
            commitment_ids = [i for i, (_, kind) in enumerate(items, 1) if kind == "commitment"]
            messages = []
            start = datetime.combine(mdate, datetime.min.time()).replace(hour=9, minute=0)
            for i, (text, kind) in enumerate(items[:40], 1):
                speaker = speakers[(i - 1) % len(speakers)]
                stamp = start + timedelta(seconds=(i - 1) * 75 + rng.randint(0, 25))
                messages.append({
                    "username": speaker,
                    "message_id": i,
                    "timestamp": stamp.strftime("%H:%M:%S"),
                    "message": messy(text, rng),
                })

            # Move the closing line to the last slot, keeping message ids stable.
            messages[-1]["message"] = messy(
                f"bon on s'arrête là, prochaine réunion le {(mdate + timedelta(days=21)).isoformat()}, merci à tous", rng)

            meeting = {
                "meeting_id": mid,
                "project_id": project["id"],
                "project": project["name"],
                "title": title,
                "meeting_date": mdate.isoformat(),
                "style": "transcribed",
                "messages": messages,
            }
            meetings.append(meeting)
            first_decision_id = decision_ids[0]
            project_meetings.append({
                "meeting_id": mid, "date": mdate.isoformat(), "title": title,
                "decision_text": decisions[0], "decision_id": first_decision_id,
                "focus": focus, "subject": subject, "decisions": decision_ids,
                "commitment_id": commitment_ids[0], "commitment": commitments[0],
            })
            if k == 0:
                project_first[project["id"]] = (mdate.isoformat(), subject, decisions[0],
                                                [f"{mid}:{x:03d}" for x in decision_ids], title)

            def cid(message_id):
                return f"{mid}:{message_id:03d}"

            queries.extend([
                {"query_id": f"{mid}-q-last", "split": "test", "meeting_id": mid,
                 "difficulty": "easy", "category": "easy_temporal",
                 "query": rng.choice(LAST_Q), "target_chunk_id": cid(40),
                 "target_chunk_ids": [cid(40)], "gold_answer": "dernier message"},
                {"query_id": f"{mid}-q-first-decision", "split": "test", "meeting_id": mid,
                 "difficulty": "easy", "category": "easy_decision",
                 "query": rng.choice(FIRST_Q), "target_chunk_id": cid(first_decision_id),
                 "target_chunk_ids": [cid(first_decision_id)], "gold_answer": decisions[0]},
                {"query_id": f"{mid}-q-count", "split": "test", "meeting_id": mid,
                 "difficulty": "hard", "category": "hard_count",
                 "query": rng.choice(COUNT_Q),
                 "target_chunk_id": cid(first_decision_id),
                 "target_chunk_ids": [cid(x) for x in decision_ids],
                 "gold_answer": f"{len(decision_ids)} décisions"},
                {"query_id": f"{mid}-q-multipart", "split": "test", "meeting_id": mid,
                 "difficulty": "hard", "category": "hard_multipart",
                 "query": rng.choice(MULTI_Q), "target_chunk_id": cid(40),
                 "target_chunk_ids": [cid(40)], "gold_answer": "dernier message + auteur + heure"},
                {"query_id": f"{mid}-q-commitment", "split": "test", "meeting_id": mid,
                 "difficulty": "hard", "category": "hard_commitment",
                 "query": rng.choice(COMMIT_Q).format(x=commitments[0]),
                 "target_chunk_id": cid(commitment_ids[0]),
                 "target_chunk_ids": [cid(commitment_ids[0])], "gold_answer": commitments[0]},
            ])
            if reference is not None:
                ref_date, ref_subject, ref_ids, ref_title = reference
                ref_meeting = next((m for m in meetings if m["meeting_id"] == ref_ids[0].split(":")[0]), None)
                queries.append({
                    "query_id": f"{mid}-q-cross", "split": "test", "meeting_id": None,
                    "difficulty": "hard", "category": "hard_cross_meeting",
                    "query": rng.choice(CROSS_Q).format(date=ref_date, x=ref_subject),
                    "target_chunk_id": ref_ids[0], "target_chunk_ids": ref_ids,
                    "gold_answer": ref_subject,
                    "reference_meeting": ref_meeting["meeting_id"] if ref_meeting else None,
                })
        previous_project = project

    (DATA / "transcribed_meetings.json").write_text(
        json.dumps(meetings, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA / "transcribed_queries.json").write_text(
        json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "meetings": len(meetings),
        "messages": sum(len(m["messages"]) for m in meetings),
        "queries": len(queries),
        "hard": sum(q["difficulty"] == "hard" for q in queries),
        "easy": sum(q["difficulty"] == "easy" for q in queries),
        "cross_meeting": sum(q["category"] == "hard_cross_meeting" for q in queries),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
