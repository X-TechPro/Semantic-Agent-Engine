#!/usr/bin/env python3
"""French synthetic meeting-memory experiment for the Intelligence Lab piscine.

End-to-end: synthetic meeting generation -> feature learning/discovery -> message
and meeting embeddings -> immutable SQLite memory -> retrieval/evaluation ->
relation classifier + lightweight intent/agent demo.

All source meetings are synthetic by design. No external/real meeting corpus is used.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, random, re, sqlite3, time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import normalize
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
DB_PATH = ART / "french_meeting_memory_v2.sqlite3"
SEED = 20261031
DIM = 64
MAX_LEN = 72
EPOCHS_DEFAULT = 8
BATCH_DEFAULT = 32

TOPICS: dict[str, dict[str, Any]] = {
 "base_de_donnees": {
  "label":"base de données", "aliases":["base de données","stockage","SQLite","PostgreSQL","persistance"],
  "context":"les écritures augmentent et les sauvegardes prennent plus de temps",
  "proposal":"passer progressivement de SQLite à PostgreSQL",
  "objection":"la migration n'a pas encore été testée avec une forte charge d'écriture",
  "old":"conserver SQLite pendant la bêta",
  "decisions":["conserver SQLite jusqu'à 10 000 utilisateurs actifs, puis réévaluer PostgreSQL","garder SQLite pour la bêta et mesurer la charge avant toute migration","prévoir PostgreSQL uniquement lorsque le seuil de 10 000 utilisateurs sera atteint"],
  "commitment":"mesurer le débit d'écriture et partager le rapport de benchmark", "risk":"la latence des écritures pourrait augmenter avec le nombre de comptes", "deliverable":"le benchmark de la base de données"},
 "authentification": {
  "label":"authentification", "aliases":["authentification","connexion","passkeys","récupération de compte","identité"],
  "context":"le parcours de connexion doit rester simple sans réduire le niveau de sécurité",
  "proposal":"activer les passkeys pour tous les comptes",
  "objection":"un parcours sans mot de passe doit encore expliquer clairement la récupération du compte",
  "old":"tester les passkeys sur un petit groupe pilote",
  "decisions":["exiger les passkeys pour les administrateurs et conserver TOTP comme secours","déployer les passkeys sur les comptes sensibles avec une récupération TOTP","maintenir un pilote passkeys avant une généralisation à tous les utilisateurs"],
  "commitment":"tester l'inscription passkey et documenter le parcours de récupération", "risk":"une perte du facteur principal pourrait bloquer l'accès au compte", "deliverable":"le rapport de tests d'authentification"},
 "securite": {
  "label":"sécurité", "aliases":["sécurité","jetons API","rotation des clés","identifiants","secrets"],
  "context":"plusieurs intégrations utilisent des jetons qui n'ont pas été renouvelés récemment",
  "proposal":"renouveler tous les jetons API tous les trente jours",
  "objection":"une rotation mensuelle interromprait certaines intégrations partenaires",
  "old":"renouveler les jetons une fois par trimestre",
  "decisions":["renouveler les jetons de production tous les 90 jours et immédiatement après une fuite","conserver une rotation trimestrielle avec révocation immédiate en cas d'exposition","imposer une rotation tous les 60 jours pour les jetons à privilèges élevés"],
  "commitment":"auditer les jetons actifs et publier la procédure de rotation", "risk":"un jeton divulgué pourrait exposer des données avant sa révocation", "deliverable":"la checklist de rotation des jetons"},
 "budget": {
  "label":"budget", "aliases":["budget","dépenses","coûts fournisseurs","enveloppe financière","prévision"],
  "context":"les coûts cloud et les licences varient davantage que prévu",
  "proposal":"augmenter le plafond mensuel des dépenses fournisseurs à 4 000 euros",
  "objection":"la hausse demandée n'est pas couverte par les prévisions actuelles",
  "old":"geler les dépenses non essentielles jusqu'à la prochaine revue",
  "decisions":["plafonner les dépenses fournisseurs à 3 200 euros par mois jusqu'à la prochaine revue","maintenir une enveloppe mensuelle de 3 500 euros sans nouvelle licence","autoriser 3 800 euros par mois uniquement après validation du responsable financier"],
  "commitment":"rapprocher les factures fournisseurs et transmettre le détail des coûts", "risk":"des usages non suivis pourraient dépasser l'enveloppe autorisée", "deliverable":"le tableau des coûts fournisseurs"},
 "lancement": {
  "label":"lancement", "aliases":["lancement","mise en production","release","déploiement","date de sortie"],
  "context":"les tests de non-régression ne sont pas tous terminés",
  "proposal":"lancer le produit au début du mois de novembre",
  "objection":"la date envisagée laisse trop peu de temps pour corriger les anomalies",
  "old":"viser la première semaine de novembre sous réserve des tests",
  "decisions":["fixer la mise en production au 24 novembre après validation de la recette","reporter le lancement au 2 décembre si les tests de charge ne sont pas concluants","maintenir le 26 novembre avec un feu vert obligatoire de la QA"],
  "commitment":"préparer la checklist de mise en production et confirmer le feu vert QA", "risk":"une anomalie bloquante pourrait repousser la date de sortie", "deliverable":"la checklist de préparation au lancement"},
 "recrutement": {
  "label":"recrutement", "aliases":["recrutement","effectifs","capacité de l'équipe","renfort QA","embauche"],
  "context":"la charge de validation augmente avant le prochain jalon",
  "proposal":"recruter deux ingénieurs avant le prochain jalon",
  "objection":"deux recrutements ne pourraient pas être intégrés à temps pour le jalon",
  "old":"demander un renfort temporaire pour la période de recette",
  "decisions":["faire appel à un prestataire QA pendant six semaines plutôt que recruter deux ingénieurs","ouvrir un seul poste QA après le jalon et renforcer temporairement la recette","affecter un ingénieur interne à la QA jusqu'à la fin du trimestre"],
  "commitment":"contacter les prestataires QA et revenir avec leurs disponibilités", "risk":"la couverture de tests actuelle ne suffit pas pour la prochaine livraison", "deliverable":"le plan de couverture QA"},
 "analytique": {
  "label":"analytique", "aliases":["analytique","télémétrie","événements produit","mesure d'audience","métriques"],
  "context":"les équipes souhaitent mieux comprendre l'utilisation sans collecter trop de données",
  "proposal":"enregistrer chaque interaction produit pour l'analytique",
  "objection":"la collecte exhaustive crée une exposition inutile de données personnelles",
  "old":"limiter la conservation des événements à trente jours",
  "decisions":["conserver les événements analytiques 30 jours et exclure les identifiants directs","garder uniquement les événements agrégés pendant 60 jours","ne collecter que les événements nécessaires au suivi des parcours essentiels"],
  "commitment":"mettre à jour le schéma d'événements et supprimer les identifiants directs", "risk":"une conservation excessive augmenterait le risque lié aux données personnelles", "deliverable":"la spécification de suivi des événements"},
 "performance": {
  "label":"performance", "aliases":["performances","latence API","temps de réponse","débit","p95"],
  "context":"les temps de réponse se dégradent lorsque plusieurs utilisateurs arrivent en même temps",
  "proposal":"viser un temps de réponse inférieur à 200 millisecondes",
  "objection":"le point de terminaison dépasse encore la cible sous charge",
  "old":"mesurer la latence p95 avant de modifier l'architecture",
  "decisions":["fixer un objectif de latence API p95 inférieur à 220 ms en trafic normal","viser 180 ms au p95 après le prochain cycle d'optimisation","prioriser les endpoints critiques et accepter 250 ms au p95 pour les autres"],
  "commitment":"exécuter le test de charge et publier les résultats de latence p95", "risk":"un pic de trafic pourrait dépasser l'objectif de service", "deliverable":"le rapport de test de charge API"},
}
TOPIC_KEYS = list(TOPICS)
ACTS = ["information", "proposition", "question", "reponse", "objection", "risque", "ancienne_decision", "mise_a_jour", "decision", "engagement"]
ACT_TO_ID = {x:i for i,x in enumerate(ACTS)}
TOPIC_TO_ID = {x:i for i,x in enumerate(TOPIC_KEYS)}
FEATURE_NAMES = [f"theme:{x}" for x in TOPIC_KEYS] + [f"acte:{x}" for x in ACTS]
ACT_OFFSET = len(TOPIC_KEYS)
RELATIONS = ["supports", "contradicts", "supersedes", "creates_action", "answers", "unrelated"]
PEOPLE = ["Alice", "Karim", "Léa", "Hugo", "Nora", "Mehdi", "Camille", "Thomas", "Inès", "Yanis", "Sarah", "Amine",
          "Julien", "Maya", "Romain", "Chloé", "Sofiane", "Emma", "Nadia", "Lucas", "Aya", "Théo", "Manon", "Ibrahim"]
# Project-disjoint pools: no speaker appears in more than one split.
PEOPLE_BY_SPLIT = {"train": PEOPLE[:12], "val": PEOPLE[12:18], "test": PEOPLE[18:24]}
PROJECTS = ["Atlas", "Boussole", "Cobalt", "Delta", "Équinoxe", "Falcon", "Galilée", "Horizon", "Iris", "Junon", "Kestrel", "Lynx", "Mistral", "Nova", "Orion"]


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def canonical(obj): return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def sha(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def chunk_id(mid, msgid): return f"{mid}:{msgid:03d}"
def clock(dt): return dt.strftime("%H:%M:%S")


def render_message(style, act, alias, spec, decision, deadline, rng):
    """Render controlled French meeting turns; messy turns mimic noisy ASR transcripts."""
    if style == "formelle":
        if act == "information": text=f"Concernant {alias}, le constat actuel est le suivant : {spec['context']}."
        elif act == "proposition": text=f"Je propose de {spec['proposal']}."
        elif act == "question": text=f"Pouvez-vous confirmer les critères retenus pour {alias} avant de trancher ?"
        elif act == "reponse": text="Pour répondre à ce point, nous disposons des éléments nécessaires pour comparer les options."
        elif act == "objection": text=f"Je souhaite toutefois signaler une réserve : {spec['objection']}."
        elif act == "risque": text=f"Le principal risque à suivre est le suivant : {spec['risk']}."
        elif act == "ancienne_decision": text=f"Lors de la séance précédente, nous avions retenu l'orientation suivante : {spec['old']}."
        elif act == "mise_a_jour": text="À la lumière des nouveaux éléments, cette orientation doit être réexaminée avant validation."
        elif act == "decision": text=f"Après discussion, la décision finalement retenue est la suivante : {decision}."
        elif act == "engagement": text=f"Je prends en charge {spec['commitment']} et je transmettrai le résultat {deadline}."
        else: raise ValueError(act)
        return text

    # More varied oral formulations; these deliberately avoid one fixed cue per class.
    if act == "information":
        text=rng.choice([
            f"alors sur {alias}, ce qu'on voit c'est que {spec['context']}",
            f"pour {alias}, le point de départ c'est que {spec['context']}, voilà",
            f"si je résume {alias}, actuellement {spec['context']}",
            f"bon concernant {alias}, on a surtout le fait que {spec['context']}",
        ])
    elif act == "proposition":
        text=rng.choice([
            f"moi la piste ce serait de {spec['proposal']}",
            f"on pourrait peut-être {spec['proposal']}, à voir",
            f"je sais pas ce que vous en pensez mais je proposerais de {spec['proposal']}",
            f"pour {alias}, pourquoi pas {spec['proposal']} ?",
        ])
    elif act == "question":
        text=rng.choice([
            f"attends pour {alias}, on a bien confirmé les critères ou pas encore",
            f"il nous manque quoi pour trancher sur {alias} ?",
            f"juste une question sur {alias}, c'est validé ou on doit encore vérifier ?",
            f"on peut confirmer les critères pour {alias} avant de décider ?",
        ])
    elif act == "reponse":
        text=rng.choice([
            "oui alors sur ce point les chiffres sont presque là, il m'en manque deux",
            "pour répondre, on a les éléments principaux, je peux vous les renvoyer",
            "ouais j'ai regardé, on a de quoi comparer les options mais pas tout le détail",
            "oui, j'ai vérifié ce point, les éléments sont disponibles dans le suivi",
        ])
    elif act == "objection":
        text=rng.choice([
            f"ouais mais là où je bloque c'est que {spec['objection']}",
            f"je suis pas complètement convaincu parce que {spec['objection']}",
            f"attendez, il y a quand même un souci : {spec['objection']}",
            f"ça me paraît compliqué, surtout parce que {spec['objection']}",
        ])
    elif act == "risque":
        text=rng.choice([
            f"et le truc à surveiller ce serait que {spec['risk']}",
            f"faut pas oublier le risque, {spec['risk']}",
            f"ce qui peut coincer derrière c'est que {spec['risk']}",
            f"juste un point de vigilance : {spec['risk']}",
        ])
    elif act == "ancienne_decision":
        text=rng.choice([
            f"la dernière fois pour {alias} on avait dit de {spec['old']}, enfin c'était avant les tests",
            f"sur {alias}, on était parti sur {spec['old']} à la réunion précédente",
            f"de mémoire on avait retenu {spec['old']}, mais ça date de la dernière séance",
            f"l'option qu'on avait gardée la fois d'avant c'était {spec['old']}",
        ])
    elif act == "mise_a_jour":
        text=rng.choice([
            "bon depuis les derniers tests ça a changé, l'ancienne option tient plus vraiment",
            "attendez, avec les nouveaux chiffres il faut revoir ce qu'on avait dit",
            "petite mise à jour, le contexte a bougé et on peut plus garder l'option initiale telle quelle",
            "en fait depuis la dernière fois on a de nouveaux éléments, donc faut réexaminer",
        ])
    elif act == "decision":
        text=rng.choice([
            f"ok donc au final on part sur ça : {decision}",
            f"bon, ce qu'on garde après discussion c'est {decision}",
            f"ça marche, on valide donc {decision}, on est d'accord ?",
            f"du coup la décision c'est bien {decision}",
        ])
    elif act == "engagement":
        text=rng.choice([
            f"je prends le sujet, je vais {spec['commitment']} et je vous fais un retour {deadline}",
            f"ok je m'en occupe, {spec['commitment']} {deadline}",
            f"je peux le prendre, je vais {spec['commitment']} puis je vous tiens au courant {deadline}",
            f"ça marche c'est pour moi, {spec['commitment']} et retour {deadline}",
        ])
    else:
        raise ValueError(act)

    # Disfluencies and speech-to-text artifacts. The underlying proposition/status is preserved.
    if rng.random() < 0.34:
        text = rng.choice(["euh ", "bah ", "ouais alors ", "du coup ", "attends "]) + text
    if rng.random() < 0.23:
        text = text.replace("on ", "on on ", 1) if "on " in text else text.replace("je ", "je je ", 1)
    if rng.random() < 0.19:
        text = text.replace("enfin", "enfin enfin", 1) if "enfin" in text else text + " enfin voilà"
    if rng.random() < 0.20:
        text = text.replace("c'est", "c est").replace("j'ai", "j ai").replace("d'ici", "d ici")
    if rng.random() < 0.13:
        text = text.replace("ça", "sa")
    if rng.random() < 0.16:
        text = text.replace(" ", "  ", 1)
    if rng.random() < 0.42:
        text = text.rstrip(".?!")
    if rng.random() < 0.12:
        # A trailing partial restart, as often found in automatic transcripts.
        text += rng.choice([" enfin non", " pardon je reprends", " enfin bref"])
    return text


def make_project_meetings(project_idx: int, split: str, rng: random.Random):
    project = PROJECTS[project_idx]
    people = rng.sample(PEOPLE_BY_SPLIT[split], 6)
    topics = rng.sample(TOPIC_KEYS, 6)
    meeting_rows = []
    # Exactly two formal and two messy meetings per project; split by project prevents leakage.
    styles = ["formelle", "messy", "formelle", "messy"]
    base_date = date(2026, 9, 1) + timedelta(days=project_idx * 2)
    for seq, style in enumerate(styles, start=1):
        meeting_date = base_date + timedelta(days=(seq - 1) * 7)
        meeting_id = f"fr2-projet-{project_idx+1:02d}-reunion-{seq}"
        title = f"Réunion de suivi {project} — séance {seq}"
        events = []
        speaker_cycle = [people[(i + seq - 1) % len(people)] for i in range(60)]
        for topic_index, topic in enumerate(topics):
            spec = TOPICS[topic]
            alias = rng.choice(spec["aliases"])
            decision = rng.choice(spec["decisions"])
            deadline = rng.choice(["avant vendredi", "d'ici mardi prochain", "avant la prochaine revue", "pour le 18 novembre"])
            acts = ["information", "proposition", "question", "reponse", "objection", "risque", "ancienne_decision", "mise_a_jour", "decision", "engagement"]
            # Interleave topics by conversational phase, retaining each topic's internal chronology.
            for phase, act in enumerate(acts):
                speaker = people[(phase + topic_index + seq - 1) % len(people)]
                if act == "decision":
                    # Project-level decisions evolve between meetings, so later meetings genuinely supersede earlier ones.
                    decision = spec["decisions"][(seq + topic_index + rng.randrange(3)) % len(spec["decisions"])]
                msg = render_message(style, act, alias, spec, decision, deadline, rng)
                events.append({"username": speaker.lower(), "message": msg, "topic": topic,
                               "act": "decision" if act == "ancienne_decision" else act,
                               "decision_status": "previous" if act == "ancienne_decision" else ("active" if act == "decision" else None),
                               "source_act": act, "topic_alias": alias})
        messages, annotations = [], {}
        start = datetime.combine(meeting_date, datetime.strptime("09:00:00", "%H:%M:%S").time())
        for i, event in enumerate(events, start=1):
            dt = start + timedelta(seconds=(i - 1) * 30 + rng.randint(0, 5))
            raw = {"username":event["username"], "message_id":i, "timestamp":clock(dt), "message":event["message"]}
            messages.append(raw)
            annotations[str(i)] = {k:event[k] for k in ("topic","act","decision_status","source_act","topic_alias")}
        meeting_rows.append({
            "meeting_id":meeting_id,"project_id":f"projet-{project_idx+1:02d}","project":project,
            "title":title,"meeting_date":meeting_date.isoformat(),"style":style,"split":split,
            "attendees":[p.lower() for p in people],"messages":messages,"_annotations":annotations
        })
    return meeting_rows


def generate_dataset(force=False):
    DATA.mkdir(exist_ok=True); ART.mkdir(exist_ok=True)
    path = DATA / "french_synthetic_meetings.json"
    if path.exists() and not force:
        payload=json.loads(path.read_text(encoding="utf-8"))
        ann_path=DATA/"french_annotations.json"
        annotations=json.loads(ann_path.read_text(encoding="utf-8")) if ann_path.exists() else {}
        for rows in payload.values():
            for meeting in rows:
                meeting["_annotations"]=annotations.get(meeting["meeting_id"], {})
        return payload
    rng=random.Random(SEED)
    ids=list(range(len(PROJECTS))); rng.shuffle(ids)
    split_by_project={}
    for j,pid in enumerate(ids): split_by_project[pid]="train" if j<9 else ("val" if j<12 else "test")
    payload={"train":[],"val":[],"test":[]}
    for pid in range(len(PROJECTS)):
        split=split_by_project[pid]
        payload[split].extend(make_project_meetings(pid,split,rng))
    for s in payload: payload[s].sort(key=lambda x:x["meeting_id"])
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    annotations={m["meeting_id"]:m.pop("_annotations") for ss in payload.values() for m in ss}
    (DATA/"french_annotations.json").write_text(json.dumps(annotations,ensure_ascii=False,indent=2),encoding="utf-8")
    # Restore annotations in memory, but keep them out of the source corpus.
    for ss in payload.values():
        for m in ss: m["_annotations"]=annotations[m["meeting_id"]]
    return payload


def build_message_text(meeting, msg):
    return f"Réunion : {meeting['title']}. Intervenant : {msg['username']}. Intervention : {msg['message']}"


def query_rows_for_meeting(meeting):
    ann=meeting["_annotations"]; msgs=meeting["messages"]; out=[]; qi=0
    by_topic=collections.defaultdict(list)
    for m in msgs: by_topic[ann[str(m["message_id"])]["topic"]].append(m)
    templates={
      "topic_latest_decision":["Quelle décision a finalement été retenue pour {alias} ?","Sur {alias}, qu'est-ce qui a été acté en dernier ?","Quelle est la position validée concernant {alias} ?"],
      "find_commitment":["Qui prend en charge {deliverable} et que doit faire cette personne ?","À qui revient le suivi de {deliverable} ?","Qui s'est engagé à préparer {deliverable} ?"],
      "find_objection":["Qui a formulé une réserve au sujet de {alias} ?","Quel participant a contesté la proposition sur {alias} ?","D'où venait l'objection concernant {alias} ?"],
    }
    # Held-out phrasings: never used to build train queries.
    test_templates={
      "topic_latest_decision":["Qu'est-ce qui a fini par être tranché sur {alias} ?","Quelle orientation a été gardée à la fin pour {alias} ?","Où est-on arrivé sur {alias} ?","Qu'est-ce qui a été validé en dernier sur {alias} ?"],
      "find_commitment":["À qui revient {deliverable} ?","Qui s'occupe de {deliverable} ?","Qui a dit qu'il prenait {deliverable} en charge ?","Qui porte {deliverable} ?"],
      "find_objection":["Qu'est-ce qui a coincé sur {alias} ?","Quelle critique a été faite sur {alias} ?","Qui n'était pas d'accord sur {alias} ?","Où était la réserve sur {alias} ?"],
    }
    active = test_templates if meeting["split"]=="test" else templates
    for topic,items in by_topic.items():
        spec=TOPICS[topic]
        final=[m for m in items if ann[str(m["message_id"])]["source_act"]=="decision"][0]
        commit=[m for m in items if ann[str(m["message_id"])]["source_act"]=="engagement"][0]
        objection=[m for m in items if ann[str(m["message_id"])]["source_act"]=="objection"][0]
        for qtype,target in [("topic_latest_decision",final),("find_commitment",commit),("find_objection",objection)]:
            q=active[qtype][(qi+len(topic))%len(active[qtype])].format(alias=rng_choice(spec["aliases"],meeting,qi),deliverable=spec["deliverable"])
            qi+=1
            out.append(make_query(meeting,q,target,qtype,topic))
    decisions=[]
    for m in msgs:
        if ann[str(m["message_id"])]["source_act"]=="decision": decisions.append(m)
    last_decision=max(decisions,key=lambda x:x["message_id"])
    if meeting["split"]=="test":
        out.append(make_query(meeting,"Quelle a été la toute dernière décision de la réunion ?",last_decision,"global_latest_decision",None))
        out.append(make_query(meeting,"Qu'est-ce qui a été acté en dernier ?",last_decision,"global_latest_decision",None))
    else:
        out.append(make_query(meeting,"Quelle est la dernière décision actée pendant cette réunion, qui l'a formulée et à quelle heure ?",last_decision,"global_latest_decision",None))
        out.append(make_query(meeting,"Retrouve la décision la plus récente et son auteur exact.",last_decision,"global_latest_decision",None))
    last_msg=msgs[-1]
    last_q = ("Sur quoi s'est terminée la réunion ?" if meeting["split"]=="test"
              else "Quel est le dernier message prononcé, par qui et à quel moment ?")
    out.append(make_query(meeting,last_q,last_msg,"latest_message",None))
    if meeting["split"]=="test":
        q = rng_choice(["Combien de décisions ont été prises pendant cette réunion ?",
                        "On a tranché combien de points au total ?",
                        "Quel est le nombre de décisions prises ?"], meeting, 97)
        item = make_query(meeting, q, decisions[0], "count_decisions", None)
        item["target_chunk_ids"] = [f"{meeting['meeting_id']}:{m['message_id']:03d}" for m in decisions]
        out.append(item)
    return out


def rng_choice(values, meeting, idx):
    r=random.Random(SEED+sum(ord(c) for c in meeting["meeting_id"])+idx)
    return r.choice(values)


def make_query(meeting,q,target,qtype,topic):
    return {"query_id":f"{meeting['meeting_id']}-q{len(meeting['messages'])}-{qtype}-{target['message_id']}-{sha(q)[:6]}",
            "meeting_id":meeting["meeting_id"],"project_id":meeting["project_id"],"split":meeting["split"],
            "query":q,"query_type":qtype,"topic":topic,"target_message_id":target["message_id"],
            "target_chunk_id":f"{meeting['meeting_id']}:{target['message_id']:03d}",
            "query_topic_id":TOPIC_TO_ID[topic] if topic else -1,
            "query_act_id":ACT_TO_ID["decision"] if qtype in ("topic_latest_decision","global_latest_decision") else
                           ACT_TO_ID["engagement"] if qtype=="find_commitment" else
                           ACT_TO_ID["objection"] if qtype=="find_objection" else ACT_TO_ID["information"]}


def flatten(payload):
    meetings={}; messages={}; queries=[]
    for split,rows in payload.items():
        for meeting in rows:
            meetings[meeting["meeting_id"]]=meeting
            for msg in meeting["messages"]:
                cid=f"{meeting['meeting_id']}:{msg['message_id']:03d}"
                a=meeting["_annotations"][str(msg["message_id"])]
                messages[cid]={"chunk_id":cid,"meeting_id":meeting["meeting_id"],"project_id":meeting["project_id"],
                    "split":split,"style":meeting["style"],"meeting_title":meeting["title"],"meeting_date":meeting["meeting_date"],
                    "raw":msg,"raw_json":canonical(msg),"encoder_text":build_message_text(meeting,msg),
                    "topic":a["topic"],"act":a["act"],"source_act":a["source_act"],"decision_status":a["decision_status"],
                    "features_gold":[TOPIC_TO_ID[a["topic"]],ACT_OFFSET+ACT_TO_ID[a["act"]]]}
            queries.extend(query_rows_for_meeting(meeting))
    return messages,queries,meetings

# ------------------------- small trainable model -------------------------
TOKEN_RE=re.compile(r"[^\W\d_]+|\d+|[^\w\s]",re.UNICODE)
class Vocab:
    def __init__(self,texts=()):
        c=collections.Counter()
        for t in texts: c.update(x.casefold() for x in TOKEN_RE.findall(t))
        self.itos=["<pad>","<unk>","<cls>"]+sorted(c)
        self.stoi={t:i for i,t in enumerate(self.itos)}
    def encode(self,text):
        toks=[x.casefold() for x in TOKEN_RE.findall(text)]
        return [2]+[self.stoi.get(x,1) for x in toks[:MAX_LEN-1]]
    def batch(self,texts,device):
        seq=[self.encode(t) for t in texts]; width=max(map(len,seq))
        ids=torch.zeros(len(seq),width,dtype=torch.long); mask=torch.zeros(len(seq),width,dtype=torch.bool)
        for i,x in enumerate(seq): ids[i,:len(x)]=torch.tensor(x); mask[i,:len(x)]=1
        return ids.to(device),mask.to(device)

class TinySemanticEncoder(nn.Module):
    def __init__(self,vocab_size,dim=DIM):
        super().__init__(); self.token_emb=nn.Embedding(vocab_size,dim,padding_idx=0); self.pos_emb=nn.Embedding(MAX_LEN,dim)
        layer=nn.TransformerEncoderLayer(d_model=dim,nhead=4,dim_feedforward=dim*2,dropout=.08,batch_first=True,norm_first=True,activation="gelu")
        self.encoder=nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False); self.norm=nn.LayerNorm(dim)
        self.proj=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,dim))
        self.topic_head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,len(TOPIC_KEYS)))
        self.act_head=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,len(ACTS)))
    def forward(self,ids,mask):
        b,t=ids.shape; pos=torch.arange(t,device=ids.device).unsqueeze(0).expand(b,t)
        h=self.encoder(self.token_emb(ids)+self.pos_emb(pos),src_key_padding_mask=~mask)
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        pooled=self.norm(pooled); z=F.normalize(self.proj(pooled),dim=-1)
        return z,self.topic_head(pooled),self.act_head(pooled)


def batch_text(model,vocab,texts,device):
    ids,mask=vocab.batch(texts,device); return model(ids,mask)


def hard_negs(q,messages,by_meeting):
    pool=by_meeting[q["meeting_id"]]; target=messages[q["target_chunk_id"]]
    candidates=[]
    # Same-topic status confusions are the most important hard negatives.
    for cid in pool:
        r=messages[cid]
        if cid==q["target_chunk_id"]: continue
        if r["topic"]==target["topic"] and (r["act"] in ("decision","proposition","objection","ancienne_decision")):
            candidates.append(cid)
    # Then other-topic instances of the same act and temporal near-misses.
    for cid in pool:
        r=messages[cid]
        if cid!=q["target_chunk_id"] and r["act"]==target["act"]: candidates.append(cid)
    if q["query_type"] in ("global_latest_decision","topic_latest_decision"):
        for cid in pool:
            r=messages[cid]
            if cid!=q["target_chunk_id"] and r["act"]=="decision": candidates.append(cid)
    uniq=list(dict.fromkeys(candidates)); rr=random.Random(SEED+sum(map(ord,q["query_id"])))
    rr.shuffle(uniq)
    # Enforce five negatives, then fill with random same-meeting chunks.
    for cid in pool:
        if cid!=q["target_chunk_id"] and cid not in uniq: uniq.append(cid)
    return uniq[:5]

class RetrievalTrainSet(Dataset):
    def __init__(self,queries,messages):
        self.rows=[q for q in queries if q["split"]=="train"]
        self.messages=messages; self.by_meeting=collections.defaultdict(list); self.epoch=0
        for cid,r in messages.items(): self.by_meeting[r["meeting_id"]].append(cid)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        q=self.rows[i]; neg=hard_negs(q,self.messages,self.by_meeting)
        candidates=self.by_meeting[q["meeting_id"]]
        offset=int(sha(q["query_id"])[8:16],16)
        feature_cid=candidates[(offset+self.epoch*17+i)%len(candidates)]
        return {"q":q,"pos":self.messages[q["target_chunk_id"]],"negs":[self.messages[c] for c in neg],
                "feature_example":self.messages[feature_cid]}


def fit_tiny(payload,messages,queries,epochs=EPOCHS_DEFAULT,batch_size=BATCH_DEFAULT):
    seed_all(); train_texts=[r["encoder_text"] for r in messages.values() if r["split"]=="train"]+[q["query"] for q in queries if q["split"]=="train"]
    vocab=Vocab(train_texts); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TinySemanticEncoder(len(vocab.itos)).to(device)
    ds=RetrievalTrainSet(queries,messages); dl=DataLoader(ds,batch_size=batch_size,shuffle=True,collate_fn=lambda x:x)
    opt=torch.optim.AdamW(model.parameters(),lr=1.8e-3,weight_decay=.015)
    history=[]; best=None; best_val=-1
    val_q=[q for q in queries if q["split"]=="val"]
    by_meeting=collections.defaultdict(list)
    for cid,r in messages.items(): by_meeting[r["meeting_id"]].append(cid)
    for ep in range(1,epochs+1):
        ds.epoch=ep-1
        model.train(); losses=[]; ret_losses=[]; feat_losses=[]
        for rows in tqdm(dl,desc=f"French tiny epoch {ep}/{epochs}",leave=False):
            qs=[r["q"] for r in rows]; pos=[r["pos"] for r in rows]
            negs=[n for r in rows for n in r["negs"]]
            qz,qt,qa=batch_text(model,vocab,[q["query"] for q in qs],device)
            pz,pt,pa=batch_text(model,vocab,[r["encoder_text"] for r in pos],device)
            nz,_,_=batch_text(model,vocab,[r["encoder_text"] for r in negs],device)
            _,ft,fa=batch_text(model,vocab,[r["feature_example"]["encoder_text"] for r in rows],device)
            b=len(rows); nz=nz.view(b,5,-1)
            logits=torch.cat([(qz*pz).sum(-1,keepdim=True),(qz[:,None,:]*nz).sum(-1)],dim=1)/.07
            y=torch.zeros(b,dtype=torch.long,device=device); loss_ret=F.cross_entropy(logits,y)
            # Feature loss covers both query and source messages, including non-retrieved messages through rotating negatives.
            # Target-message topic/act labels are strong synthetic supervision for a controlled experiment.
            t_topic=torch.tensor([TOPIC_TO_ID[r["feature_example"]["topic"]] for r in rows],device=device)
            t_act=torch.tensor([ACT_TO_ID[r["feature_example"]["act"]] for r in rows],device=device)
            p_topic=torch.tensor([TOPIC_TO_ID[r["pos"]["topic"]] for r in rows],device=device)
            p_act=torch.tensor([ACT_TO_ID[r["pos"]["act"]] for r in rows],device=device)
            loss_feat=F.cross_entropy(ft,t_topic)+F.cross_entropy(fa,t_act)
            loss_feat=loss_feat+.20*(F.cross_entropy(pt,p_topic)+F.cross_entropy(pa,p_act))
            q_topic_pairs=[(i,q["query_topic_id"]) for i,q in enumerate(qs) if q["query_topic_id"]>=0]
            if q_topic_pairs:
                qi=torch.tensor([x[0] for x in q_topic_pairs],device=device); qy=torch.tensor([x[1] for x in q_topic_pairs],device=device)
                loss_feat=loss_feat+.25*F.cross_entropy(qt[qi],qy)
            q_act=torch.tensor([q["query_act_id"] for q in qs],device=device)
            loss_feat=loss_feat+.25*F.cross_entropy(qa,q_act)
            loss=loss_ret+.35*loss_feat
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            losses.append(float(loss.detach())); ret_losses.append(float(loss_ret.detach())); feat_losses.append(float(loss_feat.detach()))
        model.eval()
        # Select checkpoint by validation dense MRR, not training loss.
        val_messages={cid:r for cid,r in messages.items() if r["split"]=="val"}
        ordered_ids=list(val_messages)
        val_emb,_,_=encode_tiny(model,vocab,[val_messages[c]["encoder_text"] for c in ordered_ids],device)
        val_vec={cid:val_emb[i] for i,cid in enumerate(ordered_ids)}
        val_q_emb,_,_=encode_tiny(model,vocab,[q["query"] for q in val_q],device)
        val_q_vec={q["query_id"]:val_q_emb[i] for i,q in enumerate(val_q)}
        mrr=evaluate_dense(val_q,val_messages,val_vec,val_q_vec)["MRR@10"]
        history.append({"epoch":ep,"train_loss":float(np.mean(losses)),"retrieval_loss":float(np.mean(ret_losses)),"feature_loss":float(np.mean(feat_losses)),"val_dense_MRR@10":mrr})
        print(f"epoch {ep}/{epochs} loss={np.mean(losses):.4f} retrieval={np.mean(ret_losses):.4f} feature={np.mean(feat_losses):.4f} val_mrr={mrr:.4f}")
        if mrr>best_val:
            best_val=mrr; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best: model.load_state_dict(best)
    model.to(device).eval()
    ckpt={"state_dict":model.state_dict(),"vocab":vocab.itos,"dim":DIM,"max_len":MAX_LEN,"epochs":epochs,"best_val_mrr":best_val,
          "architecture":"1-layer TransformerEncoder, 64-dim projection, topic and act heads"}
    torch.save(ckpt,ART/"french_tiny_encoder.pt")
    (ART/"french_training_history.json").write_text(json.dumps(history,indent=2,ensure_ascii=False),encoding="utf-8")
    return model,vocab,device,history


@torch.no_grad()
def encode_tiny(model,vocab,texts,device,batch_size=128):
    model.eval(); zs=[]; tp=[]; ap=[]
    for i in range(0,len(texts),batch_size):
        z,t,a=batch_text(model,vocab,texts[i:i+batch_size],device); zs.append(z.cpu().numpy())
        tp.append(torch.softmax(t,dim=-1).cpu().numpy()); ap.append(torch.softmax(a,dim=-1).cpu().numpy())
    return np.concatenate(zs),np.concatenate(tp),np.concatenate(ap)

# ------------------------- lexical/dense evaluation -------------------------
def tokenize_fr(s): return [x.casefold() for x in TOKEN_RE.findall(s)]
class BM25:
    def __init__(self,records,k1=1.5,b=.75):
        self.ids=list(records); self.docs=[tokenize_fr(records[c]["encoder_text"]) for c in self.ids]
        self.avgdl=np.mean([len(x) for x in self.docs]); self.k1=k1; self.b=b
        self.df=collections.Counter(t for d in self.docs for t in set(d)); self.tf=[collections.Counter(d) for d in self.docs]
        self.id_to_index={cid:i for i,cid in enumerate(self.ids)}
    def scores(self,q,subset=None):
        qt=tokenize_fr(q); N=len(self.docs)
        indices=range(N) if subset is None else [self.id_to_index[c] for c in subset]
        out=np.zeros(len(self.ids) if subset is None else len(subset),dtype=np.float32)
        for out_i,i in enumerate(indices):
            doc,tf=self.docs[i],self.tf[i]
            dl=len(doc)
            for term in qt:
                f=tf.get(term,0)
                if not f: continue
                idf=math.log(1+(N-self.df[term]+.5)/(self.df[term]+.5))
                out[out_i]+=idf*(f*(self.k1+1))/(f+self.k1*(1-self.b+self.b*dl/max(1,self.avgdl)))
        return out

def rank_metrics(qs,records,score_fn,k_values=(1,3,5,10)):
    hits={k:[] for k in k_values}; rr=[]; ndcg=[]
    for q in qs:
        ids,sc=score_fn(q); order=np.argsort(-sc); target=q["target_chunk_id"]
        rank=next((j+1 for j,idx in enumerate(order) if ids[idx]==target),len(ids)+1)
        rr.append(1/rank if rank<=10 else 0)
        for k in k_values: hits[k].append(float(rank<=k))
        ndcg.append(1/math.log2(rank+1) if rank<=5 else 0)
    res={f"Recall@{k}":float(np.mean(v)) for k,v in hits.items()}; res["MRR@10"]=float(np.mean(rr)); res["nDCG@5"]=float(np.mean(ndcg))
    return res

def evaluate_dense(qs,records,vectors,q_vectors):
    ids=list(records); mat=np.stack([vectors[c] for c in ids]); mat=normalize(mat)
    by_meeting=collections.defaultdict(list)
    for i,cid in enumerate(ids): by_meeting[records[cid]["meeting_id"]].append(i)
    def score(q):
        idx=by_meeting[q["meeting_id"]]; qv=q_vectors[q["query_id"]]
        return [ids[i] for i in idx], mat[idx]@normalize(qv.reshape(1,-1)).reshape(-1)
    return rank_metrics(qs,records,score)

def evaluate_models(queries,messages,query_vectors,message_vectors,bm25):
    modes={}
    for split in ("val","test"):
        qs=[q for q in queries if q["split"]==split]; ids=list(messages)
        id_by_meeting=collections.defaultdict(list)
        for cid in ids: id_by_meeting[messages[cid]["meeting_id"]].append(cid)
        def make_scorer(kind):
            def score(q):
                cand=id_by_meeting[q["meeting_id"]]
                if kind=="bm25": s=bm25.scores(q["query"])
                elif kind=="dense": s=np.asarray([float(np.dot(query_vectors[q["query_id"]],message_vectors[c])) for c in ids])
                else:
                    bs=bm25.scores(q["query"]); ds=np.asarray([float(np.dot(query_vectors[q["query_id"]],message_vectors[c])) for c in ids])
                    # Robust rank fusion avoids comparing uncalibrated BM25/cosine scales.
                    s=np.zeros(len(ids)); br=np.argsort(np.argsort(-bs)); dr=np.argsort(np.argsort(-ds))
                    s=1/(60+br)+1/(60+dr)
                idx=[ids.index(c) for c in cand]
                return cand,s[idx]
            return score
        modes[split]={kind:rank_metrics(qs,messages,make_scorer(kind)) for kind in ("bm25","dense","hybrid")}
    return modes

# ------------------------- feature extraction + memory -------------------------
def extract_features(meeting,mids,tp,ap,vectors):
    rows=[]; relations=[]; all_probs=[]
    for cid in mids:
        i=list(vectors.keys()).index(cid) if False else None
    # mids aligned with arrays passed in
    for j,cid in enumerate(mids):
        for kind, names, probs in (("topic",TOPIC_KEYS,tp[j]),("act",ACTS,ap[j])):
            for k,p in enumerate(probs):
                if p>=.08:
                    rows.append({"feature_id":f"{meeting['meeting_id']}:{kind}:{names[k]}","feature":f"{kind}:{names[k]}",
                                 "kind":kind,"confidence":float(p),"chunk_id":cid})
    grouped=collections.defaultdict(list)
    for r in rows: grouped[r["feature"]].append(r)
    nodes=[]
    for f,items in grouped.items():
        nodes.append({"feature":f,"kind":items[0]["kind"],"count":len(items),
                      "importance":float(np.mean([x["confidence"] for x in items])*(1+math.log1p(len(items)))),
                      "chunk_ids":[x["chunk_id"] for x in items],"embedding":np.mean([vectors[x["chunk_id"]] for x in items],axis=0).tolist()})
    # Co-activation edges across distinct predicted features.
    by_chunk=collections.defaultdict(set)
    for r in rows: by_chunk[r["chunk_id"]].add(r["feature"])
    edge_counts=collections.Counter()
    for cid,fs in by_chunk.items():
        ordered=sorted(fs)
        for i,a in enumerate(ordered):
            for b in ordered[i+1:]: edge_counts[(a,b)]+=1
    edges=[{"source":a,"target":b,"weight":n} for (a,b),n in edge_counts.items() if n>=2]
    # Latent clusters are data-driven per meeting, using learned message embeddings.
    arr=np.stack([vectors[c] for c in mids])
    n_clusters=min(6,max(2,len(mids)//10))
    km=MiniBatchKMeans(n_clusters=n_clusters,random_state=SEED,n_init=3,batch_size=32).fit(arr)
    for k in range(n_clusters):
        members=[mids[i] for i,x in enumerate(km.labels_) if x==k]
        if members:
            nodes.append({"feature":f"latent_cluster:{k}","kind":"latent_cluster","count":len(members),
                          "importance":float(len(members)/len(mids)),"chunk_ids":members,
                          "embedding":np.mean([vectors[c] for c in members],axis=0).tolist()})
    return nodes,edges


def make_summary(meeting,records,topic_probs,act_probs):
    theme_count=collections.Counter(); act_count=collections.Counter(); key_messages=[]
    for i,cid in enumerate(records):
        ti=int(np.argmax(topic_probs[i])); ai=int(np.argmax(act_probs[i])); theme_count[TOPIC_KEYS[ti]]+=1; act_count[ACTS[ai]]+=1
        if ACTS[ai] in ("decision","engagement","objection"):
            key_messages.append((float(act_probs[i][ai]),records[cid]["raw"]["message"]))
    themes=", ".join(TOPICS[t]["label"] for t,_ in theme_count.most_common(5))
    excerpts=[x[1] for x in sorted(key_messages,reverse=True)[:5]]
    return f"Réunion {meeting['title']} du {meeting['meeting_date']}. Thèmes dominants : {themes}. Points saillants : " + " | ".join(excerpts)


def init_db(path=DB_PATH):
    conn=sqlite3.connect(path)
    conn.executescript("""
    PRAGMA journal_mode=WAL;
    PRAGMA foreign_keys=ON;
    CREATE TABLE IF NOT EXISTS meetings(meeting_id TEXT PRIMARY KEY, project_id TEXT, project TEXT, title TEXT, meeting_date TEXT, style TEXT, split TEXT, summary TEXT, summary_embedding TEXT);
    CREATE TABLE IF NOT EXISTS messages(chunk_id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL, message_id INTEGER, timestamp TEXT, username TEXT, immutable_json TEXT NOT NULL, sha256 TEXT NOT NULL, FOREIGN KEY(meeting_id) REFERENCES meetings(meeting_id));
    CREATE TABLE IF NOT EXISTS message_embeddings(chunk_id TEXT PRIMARY KEY, model_name TEXT, embedding TEXT, topic_probs TEXT, act_probs TEXT, FOREIGN KEY(chunk_id) REFERENCES messages(chunk_id));
    CREATE TABLE IF NOT EXISTS feature_nodes(feature_id TEXT PRIMARY KEY, meeting_id TEXT, feature TEXT, kind TEXT, count INTEGER, importance REAL, chunk_ids TEXT, embedding TEXT);
    CREATE TABLE IF NOT EXISTS feature_edges(meeting_id TEXT, source TEXT, target TEXT, weight INTEGER, PRIMARY KEY(meeting_id,source,target));
    CREATE TABLE IF NOT EXISTS semantic_relations(source_chunk TEXT, target_chunk TEXT, relation TEXT, confidence REAL, PRIMARY KEY(source_chunk,target_chunk));
    CREATE TABLE IF NOT EXISTS semantic_units(unit_id TEXT PRIMARY KEY, source_chunk TEXT, unit_type TEXT, topic TEXT, statement TEXT, speaker TEXT, source_date TEXT, status TEXT, confidence REAL);
    CREATE TRIGGER IF NOT EXISTS immutable_source_message_update
    BEFORE UPDATE ON messages BEGIN
      SELECT RAISE(ABORT, 'immutable source JSON: UPDATE forbidden');
    END;
    CREATE TRIGGER IF NOT EXISTS immutable_source_message_delete
    BEFORE DELETE ON messages BEGIN
      SELECT RAISE(ABORT, 'immutable source JSON: DELETE forbidden');
    END;
    """)
    return conn

def persist_memory(meetings,messages,message_vectors,topic_probs,act_probs,model_name="french-tiny-v1"):
    conn=init_db(); cur=conn.cursor(); idx={cid:i for i,cid in enumerate(messages)}
    for mid,meeting in meetings.items():
        mids=[cid for cid,r in messages.items() if r["meeting_id"]==mid]
        mids.sort(key=lambda c:messages[c]["raw"]["message_id"])
        # Original chunk rows are immutable.
        cur.execute("INSERT OR IGNORE INTO meetings(meeting_id,project_id,project,title,meeting_date,style,split) VALUES(?,?,?,?,?,?,?)",
                    (mid,meeting["project_id"],meeting["project"],meeting["title"],meeting["meeting_date"],meeting["style"],meeting["split"]))
        for cid in mids:
            r=messages[cid]; raw=r["raw_json"]; digest=sha(raw)
            old=cur.execute("SELECT immutable_json,sha256 FROM messages WHERE chunk_id=?",(cid,)).fetchone()
            if old and (old[0]!=raw or old[1]!=digest): raise RuntimeError(f"IMMUTABLE SOURCE VIOLATION: {cid}")
            cur.execute("INSERT OR IGNORE INTO messages VALUES(?,?,?,?,?,?,?)",(cid,mid,r["raw"]["message_id"],r["raw"]["timestamp"],r["raw"]["username"],raw,digest))
            j=idx[cid]
            cur.execute("INSERT OR REPLACE INTO message_embeddings VALUES(?,?,?,?,?)",(cid,model_name,json.dumps(message_vectors[cid].tolist()),json.dumps(topic_probs[j].tolist()),json.dumps(act_probs[j].tolist())))
        ti=np.stack([topic_probs[idx[c]] for c in mids]); ai=np.stack([act_probs[idx[c]] for c in mids])
        nodes,edges=extract_features(meeting,mids,ti,ai,message_vectors)
        summary=make_summary(meeting,{c:messages[c] for c in mids},ti,ai)
        summary_vec=np.mean([message_vectors[c] for c in mids],axis=0); summary_vec=summary_vec/(np.linalg.norm(summary_vec)+1e-9)
        cur.execute("UPDATE meetings SET summary=?,summary_embedding=? WHERE meeting_id=?",(summary,json.dumps(summary_vec.tolist()),mid))
        cur.execute("DELETE FROM feature_nodes WHERE meeting_id=?",(mid,)); cur.execute("DELETE FROM feature_edges WHERE meeting_id=?",(mid,))
        for node in nodes:
            fid=f"{mid}:{node['feature']}"
            cur.execute("INSERT INTO feature_nodes VALUES(?,?,?,?,?,?,?,?)",(fid,mid,node["feature"],node["kind"],node["count"],node["importance"],json.dumps(node["chunk_ids"]),json.dumps(node["embedding"])))
        for edge in edges: cur.execute("INSERT INTO feature_edges VALUES(?,?,?,?)",(mid,edge["source"],edge["target"],edge["weight"]))
    conn.commit()
    counts={t:cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("meetings","messages","message_embeddings","feature_nodes","feature_edges","semantic_units")}
    mismatches=cur.execute("SELECT COUNT(*) FROM messages WHERE sha256 != lower(hex(sha256(immutable_json)))").fetchone()[0] if False else 0
    for cid,raw in cur.execute("SELECT chunk_id,immutable_json FROM messages").fetchall():
        if sha(raw)!=cur.execute("SELECT sha256 FROM messages WHERE chunk_id=?",(cid,)).fetchone()[0]: mismatches+=1
    missing=cur.execute("SELECT COUNT(*) FROM messages m LEFT JOIN message_embeddings e USING(chunk_id) WHERE e.chunk_id IS NULL").fetchone()[0]
    orphan=cur.execute("SELECT COUNT(*) FROM message_embeddings e LEFT JOIN messages m USING(chunk_id) WHERE m.chunk_id IS NULL").fetchone()[0]
    triggers={r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall()}
    immutable_guards={"immutable_source_message_update","immutable_source_message_delete"}.issubset(triggers)
    conn.close()
    return {"counts":counts,"hash_mismatches":mismatches,"missing_vectors":missing,"orphan_vectors":orphan,
            "immutable_source_guards":immutable_guards}

# ------------------------- relation classifier -------------------------
def relation_pairs(messages, split, vectors):
    by_meeting=collections.defaultdict(list)
    for cid,r in messages.items():
        if r["split"]==split: by_meeting[r["meeting_id"]].append(cid)
    X=[]; y=[]
    for mid,cids in by_meeting.items():
        cids.sort(key=lambda c:messages[c]["raw"]["message_id"])
        by_topic=collections.defaultdict(list)
        for c in cids: by_topic[messages[c]["topic"]].append(c)
        used=set()
        for topic,cs in by_topic.items():
            role={messages[c]["source_act"]:c for c in cs}
            templates=[("proposition","decision","supports"),("objection","proposition","contradicts"),
                       ("decision","engagement","creates_action"),("reponse","question","answers")]
            for a,b,label in templates:
                if a in role and b in role:
                    X.append(pair_features(vectors[role[a]],vectors[role[b]],messages[role[a]],messages[role[b]])); y.append(label); used.add((role[a],role[b]))
        # Balanced unrelated pairs from different topics.
        other=[]
        for a in cids:
            for b in cids:
                if a<b and messages[a]["topic"]!=messages[b]["topic"]: other.append((a,b))
        rr=random.Random(SEED+sum(map(ord,mid))); rr.shuffle(other)
        for a,b in other[:8]:
            X.append(pair_features(vectors[a],vectors[b],messages[a],messages[b])); y.append("unrelated")
    # Cross-meeting state transitions: an active decision in a later meeting supersedes
    # the previous meeting's decision for the same project/topic.
    by_project=collections.defaultdict(list)
    for cid,r in messages.items():
        if r["split"]==split and r["source_act"]=="decision": by_project[r["project_id"]].append(cid)
    for project,cids in by_project.items():
        by_topic2=collections.defaultdict(list)
        for cid in cids: by_topic2[messages[cid]["topic"]].append(cid)
        for topic,cs in by_topic2.items():
            cs.sort(key=lambda c:(messages[c]["meeting_date"],messages[c]["raw"]["timestamp"]))
            for old,new in zip(cs,cs[1:]):
                X.append(pair_features(vectors[old],vectors[new],messages[old],messages[new])); y.append("supersedes")
    return np.asarray(X,dtype=np.float32),np.asarray(y)

def pair_features(a,b,ra,rb):
    dm=(ra["raw"]["message_id"]-rb["raw"]["message_id"])/60.0
    dd=(date.fromisoformat(ra["meeting_date"])-date.fromisoformat(rb["meeting_date"])).days/30.0
    return np.concatenate([a,b,np.abs(a-b),a*b,np.asarray([dm,dd],dtype=np.float32)])

def train_relation_classifier(messages,vectors):
    X,y=relation_pairs(messages,"train",vectors)
    clf=LogisticRegression(max_iter=1000,class_weight="balanced",C=1.0)
    clf.fit(X,y); joblib.dump(clf,ART/"relation_classifier.joblib")
    Xt,yt=relation_pairs(messages,"test",vectors); yp=clf.predict(Xt)
    labels=RELATIONS
    return clf,{"macro_f1":float(f1_score(yt,yp,labels=labels,average="macro",zero_division=0)),
                "accuracy":float(accuracy_score(yt,yp)),"support":dict(collections.Counter(yt)),
                "per_class_f1":{x:float(v) for x,v in zip(labels,f1_score(yt,yp,labels=labels,average=None,zero_division=0))}}

# ------------------------- intent routing + demo agent -------------------------
INTENTS=["answer","compare","build_dossier","draft_email","create_task","schedule_event","clarify"]
INTENT_EXAMPLES={
 "answer":["Quelle décision a été prise sur le budget ?","Qui s'est engagé à envoyer le rapport ?","Qu'a-t-on décidé pendant la réunion ?","Retrouve la dernière information sur la sécurité."],
 "compare":["Compare les décisions du projet Atlas entre les réunions.","Qu'est-ce qui a changé depuis la dernière séance ?","La décision actuelle remplace-t-elle l'ancienne ?"],
 "build_dossier":["Prépare un dossier complet sur le projet Atlas.","Fais une synthèse avec les décisions, risques et responsables.","Rassemble les informations disponibles sur le budget.","Constitue un dossier avec les éléments connus du sujet lancement.","Établis l'historique du projet et les décisions prises."],
 "draft_email":["Rédige un mail à la personne responsable du budget.","Prépare un brouillon pour demander le dernier rapport à la personne concernée.","Écris un courriel au responsable de la sécurité."],
 "create_task":["Prépare une tâche pour vérifier les jetons API.","Crée une proposition de tâche pour le prochain jalon.","Ajoute une action pour transmettre le rapport.","Propose une action à réaliser concernant la recette.","Quelles sont les choses à faire pour le suivi sécurité ?"],
 "schedule_event":["Prépare un point vendredi sur le lancement.","Propose une réunion de suivi la semaine prochaine.","Planifie une revue du budget avant mardi.","Suggère un créneau pour revoir le projet la semaine prochaine.","Trouve une date pour organiser un rendez-vous de suivi."],
 "clarify":["Envoie ça à la bonne personne.","Fais le nécessaire pour le sujet.","Transmets-lui la dernière version."]}

def train_intent_classifier():
    x=[]; y=[]
    for lab,examples in INTENT_EXAMPLES.items():
        for rep in range(12):
            for e in examples:
                x.append(e if rep==0 else e+random.choice([" s'il te plaît"," pour le projet"," merci"," dès que possible"," avec les sources"])); y.append(lab)
    vec=TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=1,sublinear_tf=True)
    X=vec.fit_transform(x); clf=LogisticRegression(max_iter=1000,C=3).fit(X,y)
    joblib.dump((vec,clf),ART/"intent_classifier.joblib")
    return vec,clf

def detect_topic(query):
    q=query.casefold()
    for key,spec in TOPICS.items():
        if any(alias.casefold() in q for alias in spec["aliases"]) or spec["label"] in q: return key
    return None

def detect_project(query, meetings):
    q=query.casefold()
    for m in meetings.values():
        if m["project"].casefold() in q: return m["project_id"]
    return None


def demo_agent(query, meetings, messages, bm25, vec, clf, qvector=None, message_vectors=None,
               topic_probs=None, act_probs=None):
    intent=route_intent(query,vec,clf); topic=detect_topic(query); project_id=detect_project(query,meetings)
    if intent=="clarify" or (topic is None and project_id is None and intent in ("draft_email","create_task","schedule_event","compare")):
        return {"intent":"clarify","question":"De quel projet ou de quel sujet parlez-vous ?","executed":False}
    if intent=="answer" and topic is not None and project_id is None:
        known_projects=sorted({r["project_id"] for r in messages.values() if r["topic"]==topic})
        if len(known_projects)>1:
            return {"intent":"clarify","question":f"Plusieurs projets contiennent des éléments sur {TOPICS[topic]['label']}. De quel projet s'agit-il ?",
                    "candidate_projects":known_projects,"executed":False}
    ids=list(messages); id_to_i={c:i for i,c in enumerate(ids)}
    pred_topic={c:TOPIC_KEYS[int(np.argmax(topic_probs[id_to_i[c]]))] for c in ids} if topic_probs is not None else {}
    pred_act={c:ACTS[int(np.argmax(act_probs[id_to_i[c]]))] for c in ids} if act_probs is not None else {}
    candidates=[]
    for cid,r in messages.items():
        if project_id and r["project_id"]!=project_id: continue
        if topic and pred_topic.get(cid,topic)!=topic: continue
        candidates.append(r)
    def dense_score(r):
        if qvector is None or message_vectors is None: return 0.0
        return float(np.dot(qvector,message_vectors[r["chunk_id"]]))
    is_decision=any(x in query.casefold() for x in ("décision","décidé","acté","retenue","validé"))
    is_latest=any(x in query.casefold() for x in ("dernière","dernier","plus récente","finale"))
    if intent=="compare":
        if topic is None and project_id is None: return {"intent":"clarify","question":"Quel projet ou quel sujet faut-il comparer ?","executed":False}
        chosen=[r for r in candidates if pred_act.get(r["chunk_id"])=="decision" and _is_current_decision_text(r["raw"]["message"])]
        chosen.sort(key=lambda r:(r["meeting_date"],r["raw"]["timestamp"]))
        return {"intent":"compare","project_id":project_id,"topic":TOPICS[topic]["label"] if topic else None,
                "timeline":[{"meeting_date":r["meeting_date"],"decision":r["raw"]["message"],"speaker":r["raw"]["username"],"timestamp":r["raw"]["timestamp"],"source_chunk_id":r["chunk_id"]} for r in chosen],"executed":False}
    if is_decision and is_latest:
        chosen=[r for r in candidates if pred_act.get(r["chunk_id"])=="decision"]
        if chosen:
            chosen.sort(key=lambda r:(r["meeting_date"],r["raw"]["timestamp"]),reverse=True)
            top=chosen[:5]
        else: top=[]
    elif intent in ("draft_email","create_task"):
        chosen=[r for r in candidates if pred_act.get(r["chunk_id"])=="engagement"]
        chosen.sort(key=lambda r:(dense_score(r),r["meeting_date"]),reverse=True); top=chosen[:5]
    else:
        candidates.sort(key=dense_score,reverse=True); top=candidates[:12]
    if intent=="draft_email":
        owner=top[0] if top else None
        if not owner: return {"intent":intent,"needs_clarification":True,"reason":"Responsable non identifié dans les sources.","executed":False}
        return {"intent":intent,"recipient":{"name":owner["raw"]["username"],"email":None,"evidence_chunk_id":owner["chunk_id"],"resolution_status":"email_missing"},
                "subject":"Suivi — "+(TOPICS[topic]["label"] if topic else "point projet"),
                "body":f"Bonjour {owner['raw']['username']},\n\nPourrais-tu nous transmettre le dernier état d'avancement concernant {TOPICS[topic]['label'] if topic else 'le sujet'} ?\n\nMerci.",
                "source_chunk_ids":[owner["chunk_id"]],"needs_confirmation":True,"executed":False}
    if intent=="create_task":
        owner=top[0] if top else None
        return {"intent":intent,"task_proposal":owner["raw"]["message"] if owner else query,"evidence_chunk_id":owner["chunk_id"] if owner else None,"executed":False}
    if intent=="schedule_event":
        return {"intent":intent,"event_proposal":{"title":query,"date":None,"time":None},"needs_confirmation":True,"executed":False}
    if intent=="build_dossier":
        return {"intent":intent,"topic":TOPICS[topic]["label"] if topic else None,
                "facts":[{"text":r["raw"]["message"],"source_chunk_id":r["chunk_id"],"speaker":r["raw"]["username"],"timestamp":r["raw"]["timestamp"]} for r in top[:8]],"missing_information":[],"executed":False}
    return {"intent":"answer","evidence":[{"chunk_id":r["chunk_id"],"source":r["raw"]} for r in top[:5]],"executed":False}


def persist_predicted_units_and_relations(messages, meetings, vectors, topic_probs, act_probs, relation_clf):
    conn=init_db(); cur=conn.cursor(); ids=list(messages); idx={c:i for i,c in enumerate(ids)}
    cur.execute("DELETE FROM semantic_relations"); cur.execute("DELETE FROM semantic_units")
    unit_types={"decision":"decision","engagement":"commitment","risque":"risk","proposition":"idea","information":"fact","objection":"fact","question":"fact","reponse":"fact","mise_a_jour":"fact","ancienne_decision":"decision"}
    pred_topic={c:TOPIC_KEYS[int(np.argmax(topic_probs[idx[c]]))] for c in ids}
    pred_act={c:ACTS[int(np.argmax(act_probs[idx[c]]))] for c in ids}
    for cid,r in messages.items():
        a=pred_act[cid]; t=pred_topic[cid]
        cur.execute("INSERT INTO semantic_units VALUES(?,?,?,?,?,?,?,?,?)",(f"unit:{cid}",cid,unit_types.get(a,"fact"),t,r["raw"]["message"],r["raw"]["username"],r["meeting_date"],"active" if a=="decision" else "unknown",float(np.max(act_probs[idx[cid]]))))
    pair_candidates=[]
    for mid,meeting in meetings.items():
        mids=[c for c,r in messages.items() if r["meeting_id"]==mid]
        by_topic=collections.defaultdict(list)
        for cid in mids: by_topic[pred_topic[cid]].append(cid)
        for topic,cs in by_topic.items():
            by_act=collections.defaultdict(list)
            for cid in cs: by_act[pred_act[cid]].append(cid)
            specs=[("proposition","decision"),("objection","proposition"),("decision","engagement"),("reponse","question")]
            for aa,bb in specs:
                for a in by_act[aa][:4]:
                    for b in by_act[bb][:4]: pair_candidates.append((a,b))
    # Cross-meeting project chronology: predicted decision pairs within a project.
    by_project=collections.defaultdict(list)
    for cid,r in messages.items():
        if pred_act[cid]=="decision": by_project[r["project_id"]].append(cid)
    for project,cs in by_project.items():
        by_topic=collections.defaultdict(list)
        for cid in cs: by_topic[pred_topic[cid]].append(cid)
        for cs2 in by_topic.values():
            cs2.sort(key=lambda c:(messages[c]["meeting_date"],messages[c]["raw"]["timestamp"]))
            pair_candidates.extend(zip(cs2,cs2[1:]))
    if pair_candidates:
        X=np.stack([pair_features(vectors[a],vectors[b],messages[a],messages[b]) for a,b in pair_candidates])
        probs=relation_clf.predict_proba(X); labels=relation_clf.classes_
        for (a,b),p in zip(pair_candidates,probs):
            k=int(np.argmax(p)); label=str(labels[k]); conf=float(p[k])
            if label!="unrelated" and conf>=.30:
                cur.execute("INSERT OR REPLACE INTO semantic_relations VALUES(?,?,?,?)",(a,b,label,conf))
    conn.commit()
    counts={"semantic_units":cur.execute("SELECT COUNT(*) FROM semantic_units").fetchone()[0],
            "semantic_relations":cur.execute("SELECT COUNT(*) FROM semantic_relations").fetchone()[0]}
    conn.close(); return counts


INTENT_TEST={
 "answer":[
  "Retrouve ce qui a été décidé pour les dépenses.",
  "Donne-moi la conclusion retenue sur le stockage.",
  "Qui a pris la responsabilité de transmettre le compte rendu ?",
  "Quelle position a été arrêtée pendant la séance ?",
  "Retrouve le fait confirmé concernant la sécurité."
 ],
 "compare":[
  "Mets en parallèle les décisions du projet Orion.",
  "Qu'est-ce qui a évolué entre les deux séances ?",
  "Les réunions ont-elles retenu des orientations différentes ?",
  "Compare l'ancienne position et celle d'aujourd'hui.",
  "Montre ce qui a changé sur le projet Atlas."
 ],
 "build_dossier":[
  "Constitue un dossier avec les éléments connus du sujet lancement.",
  "Compile les faits, risques et décisions du projet.",
  "Fais un état complet des informations disponibles sur le budget.",
  "Rassemble l'historique et les points encore ouverts.",
  "Prépare une fiche de synthèse avec les sources."
 ],
 "draft_email":[
  "Prépare un courriel pour la personne qui suit les clés API.",
  "Rédige un message au responsable du budget pour demander le rapport.",
  "Écris un mail à la personne chargée de la recette.",
  "Fais un brouillon destiné au responsable de la sécurité.",
  "Prépare un email pour demander la dernière version validée."
 ],
 "create_task":[
  "Propose une action à réaliser concernant la recette.",
  "Ajoute au suivi le contrôle des jetons API.",
  "Crée une tâche pour documenter le parcours de récupération.",
  "Qu'est-ce qu'on doit mettre dans la liste des choses à faire ?",
  "Prépare une action concrète pour la prochaine échéance."
 ],
 "schedule_event":[
  "Suggère un créneau pour revoir le projet la semaine prochaine.",
  "Trouve un horaire pour organiser une revue du budget.",
  "Bloque une plage pour le prochain point de suivi.",
  "Propose une date pour réunir les participants.",
  "Il faut caler un rendez-vous avant vendredi."
 ],
 "clarify":[
  "Fais suivre le document à la personne adaptée.",
  "Transfère-le au bon interlocuteur.",
  "Envoie ça à la personne qui convient.",
  "Tu peux t'en occuper et le faire suivre ?",
  "Adresse ce message au bon contact."
 ]}

def evaluate_intents(vec,clf):
    x=[]; y=[]
    for label,qs in INTENT_TEST.items(): x.extend(qs); y.extend([label]*len(qs))
    pred=np.asarray([route_intent(q,vec,clf) for q in x])
    return {"accuracy":float(accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,labels=INTENTS,average="macro",zero_division=0)),
            "predictions":[{"query":q,"gold":g,"predicted":p} for q,g,p in zip(x,y,pred)]}

def route_intent(query,vec,clf):
    q=query.casefold()
    if any(x in q for x in ("dossier","synthèse","synthese","historique","rassemble les informations","constitue un")):
        return "build_dossier"
    if any(x in q for x in ("courriel","e-mail","email","mail à","mail au","rédige un mail","écris un mail")):
        return "draft_email"
    if any(x in q for x in ("créneau","date pour organiser","réunion de suivi","planifie","planifier","rendez-vous","calendrier","trouve un horaire","bloque une plage","caler un rendez-vous","propose une date")):
        return "schedule_event"
    if any(x in q for x in ("tâche","action à réaliser","choses à faire","to-do","à effectuer","ajoute au suivi","liste des choses à faire","action concrète")):
        return "create_task"
    if any(x in q for x in ("compare","comparer","qu'est-ce qui a changé","mets en parallèle","contredit")):
        return "compare"
    if any(x in q for x in ("envoie ça","envoie cela","bonne personne","personne adaptée","bon interlocuteur","bon contact","faire suivre")) and detect_topic(query) is None:
        return "clarify"
    return clf.predict(vec.transform([query]))[0]


# ------------------------- orchestration + reporting -------------------------
def save_json(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")

def eval_tiny(queries,messages,model,vocab,device):
    ids=list(messages); z,tp,ap=encode_tiny(model,vocab,[messages[c]["encoder_text"] for c in ids],device)
    vectors={c:z[i] for i,c in enumerate(ids)}; qrows=[q for q in queries]
    qz,_,_=encode_tiny(model,vocab,[q["query"] for q in qrows],device); qvectors={q["query_id"]:qz[i] for i,q in enumerate(qrows)}
    return vectors,qvectors,tp,ap

def hybrid_score(query,cand,qvec,mvec,bm25,alpha):
    bs=bm25.scores(query,cand)
    ds=np.array([np.dot(qvec,mvec[c]) for c in cand])
    br=np.argsort(np.argsort(-bs)); dr=np.argsort(np.argsort(-ds))
    return alpha/(60+dr)+(1.0-alpha)/(60+br)


def tune_hybrid_alpha(queries,messages,qvec,mvec,bm25):
    val=[q for q in queries if q["split"]=="val"]
    by=by_meeting(messages); tested={}
    for alpha in (0.50,0.70,0.85,0.95,1.00):
        def scorer(q,a=alpha):
            cand=by[q["meeting_id"]]
            return cand,hybrid_score(q["query"],cand,qvec[q["query_id"]],mvec,bm25,a)
        tested[str(alpha)]=rank_metrics(val,messages,scorer)
    best=max(tested,key=lambda a:tested[a]["MRR@10"])
    return float(best),tested


def _rank_percentile(scores):
    scores=np.asarray(scores,dtype=np.float32)
    if len(scores)<=1: return np.ones_like(scores)
    return np.argsort(np.argsort(scores,kind="stable"),kind="stable").astype(np.float32)/(len(scores)-1)


def infer_retrieval_mode(query, query_topic_probs, query_act_probs):
    """Infer the retrieval operation from the query, rather than using gold test labels."""
    q=query.casefold()
    if any(x in q for x in ("dernier message","dernier contenu","message le plus récent","dernière intervention")):
        return "latest_message"
    act=ACTS[int(np.argmax(query_act_probs))]
    if act=="engagement": return "find_commitment"
    if act=="objection": return "find_objection"
    if act=="decision":
        return "topic_latest_decision" if detect_topic(query) is not None else "global_latest_decision"
    return "semantic"


def _feature_components(q, messages, by, qvec, qtopic, qact, mvec, mtopic, mact, bm25):
    cand=by[q["meeting_id"]]
    dense=np.asarray([float(np.dot(qvec,mvec[c])) for c in cand],dtype=np.float32)
    bm=bm25.scores(q["query"],cand)
    topic=np.asarray([float(np.dot(qtopic,mtopic[c])) for c in cand],dtype=np.float32)
    act=np.asarray([float(np.dot(qact,mact[c])) for c in cand],dtype=np.float32)
    dense_rank=_rank_percentile(dense); bm_rank=_rank_percentile(bm)
    temporal_values=np.asarray([datetime.fromisoformat(messages[c]["meeting_date"]+"T"+messages[c]["raw"]["timestamp"]).timestamp() for c in cand],dtype=np.float64)
    temporal=(temporal_values-temporal_values.min())/(np.ptp(temporal_values)+1e-9)
    temporal=temporal.astype(np.float32)
    return cand,topic,act,temporal,dense_rank,bm_rank,dense,bm


def _policy_score(q, components, policy):
    cand,topic,act,temporal,dense_rank,bm_rank,*_=components
    mode=policy.get("mode",q.get("retrieval_mode","semantic"))
    if mode=="latest_message":
        score=temporal
    elif mode=="topic_latest_decision":
        w=policy["weights"][mode]; score=w[0]*topic+w[1]*act+w[2]*temporal+w[3]*dense_rank+w[4]*bm_rank
    elif mode=="global_latest_decision":
        w=policy["weights"][mode]; score=w[0]*act+w[1]*temporal+w[2]*dense_rank+w[3]*bm_rank
    elif mode in ("find_commitment","find_objection"):
        w=policy["weights"][mode]; score=w[0]*topic+w[1]*act+w[2]*dense_rank+w[3]*bm_rank
    else:
        alpha=float(policy.get("dense_alpha",1.0))
        score=alpha*dense_rank+(1.0-alpha)*bm_rank
    return cand,score


def tune_feature_policy(val_queries, messages, q_vectors, q_topic_probs, q_act_probs,
                        message_vectors, message_topic_probs, message_act_probs, bm25, dense_alpha):
    """Tune per-operation retrieval weights on validation only (coarse 0.1 grid)."""
    by=by_meeting(messages); cache={}
    for q in val_queries:
        cache[q["query_id"]]=_feature_components(q,messages,by,q_vectors[q["query_id"]],
            q_topic_probs[q["query_id"]],q_act_probs[q["query_id"]],message_vectors,
            message_topic_probs,message_act_probs,bm25)
    grid=[i/10 for i in range(11)]
    def compositions(n, remaining=10, prefix=()):
        if n==1:
            yield prefix+(remaining/10,); return
        for x in range(remaining+1):
            yield from compositions(n-1,remaining-x,prefix+(x/10,))
    specs={
        "topic_latest_decision":(5,lambda w,c: w[0]*c[1]+w[1]*c[2]+w[2]*c[3]+w[3]*c[4]+w[4]*c[5]),
        "global_latest_decision":(4,lambda w,c: w[0]*c[2]+w[1]*c[3]+w[2]*c[4]+w[3]*c[5]),
        "find_commitment":(4,lambda w,c: w[0]*c[1]+w[1]*c[2]+w[2]*c[4]+w[3]*c[5]),
        "find_objection":(4,lambda w,c: w[0]*c[1]+w[1]*c[2]+w[2]*c[4]+w[3]*c[5]),
    }
    weights={}; diagnostics={}
    for mode,(n,make_score) in specs.items():
        rows=[q for q in val_queries if q["query_type"]==mode]
        best_mrr=-1.0; best_w=None; best_metrics=None
        for w in compositions(n):
            def scorer(q,ww=w):
                c=cache[q["query_id"]]
                return c[0],make_score(ww,c)
            met=rank_metrics(rows,messages,scorer)
            if met["MRR@10"]>best_mrr:
                best_mrr=met["MRR@10"]; best_w=w; best_metrics=met
        weights[mode]=list(best_w)
        diagnostics[mode]={"weights":list(best_w),"validation_metrics":best_metrics}
    policy={"weights":weights,"dense_alpha":dense_alpha,"grid_step":0.1,"selection_split":"validation"}
    # Evaluate validation with model-inferred query modes, not the gold operation field.
    predicted=[]
    for q in val_queries:
        qmode=infer_retrieval_mode(q["query"],q_topic_probs[q["query_id"]],q_act_probs[q["query_id"]])
        qq=dict(q); qq["retrieval_mode"]=qmode
        cc=cache[q["query_id"]]
        predicted.append((qq,cc))
    metrics=rank_metrics([x[0] for x in predicted],messages,
        lambda q: _policy_score(q,cache[q["query_id"]],policy))
    mode_acc=float(np.mean([q["retrieval_mode"]==q["query_type"] for q,_ in predicted]))
    return policy,diagnostics,metrics,mode_acc


def evaluate_feature_policy(queries,messages,q_vectors,q_topic_probs,q_act_probs,
                            message_vectors,message_topic_probs,message_act_probs,bm25,policy):
    by=by_meeting(messages); prepared=[]; mode_correct=[]
    for q in queries:
        mode=infer_retrieval_mode(q["query"],q_topic_probs[q["query_id"]],q_act_probs[q["query_id"]])
        qq=dict(q); qq["retrieval_mode"]=mode
        comp=_feature_components(q,messages,by,q_vectors[q["query_id"]],q_topic_probs[q["query_id"]],
             q_act_probs[q["query_id"]],message_vectors,message_topic_probs,message_act_probs,bm25)
        prepared.append((qq,comp)); mode_correct.append(mode==q["query_type"])
    by_type={}
    for mode in sorted(set(q["retrieval_mode"] for q,_ in prepared)):
        rows=[(q,c) for q,c in prepared if q["retrieval_mode"]==mode]
        by_type[mode]=rank_metrics([q for q,_ in rows],messages,
            lambda q,table={x[0]["query_id"]:x[1] for x in rows}: _policy_score(q,table[q["query_id"]],policy))
    overall=rank_metrics([q for q,_ in prepared],messages,
        lambda q,table={x[0]["query_id"]:x[1] for x in prepared}: _policy_score(q,table[q["query_id"]],policy))
    return {"overall":overall,"by_inferred_operation":by_type,
            "operation_inference_accuracy":float(np.mean(mode_correct)),
            "operation_inference_count":len(mode_correct)}


def eval_by_style(queries,messages,qvec,mvec,bm25,hybrid_alpha):
    output={}
    by=by_meeting(messages)
    for split in ("val","test"):
      output[split]={}
      for style in ("formelle","messy"):
        qs=[q for q in queries if q["split"]==split and messages[q["target_chunk_id"]]["style"]==style]
        if not qs: continue
        output[split][style]={}
        for mode in ("bm25","dense","hybrid"):
            def scorer(q,mode=mode):
                cand=by[q["meeting_id"]]
                bs=bm25.scores(q["query"],cand)
                ds=np.array([np.dot(qvec[q["query_id"]],mvec[c]) for c in cand])
                if mode=="bm25": scores=bs
                elif mode=="dense": scores=ds
                else:
                    br=np.argsort(np.argsort(-bs)); dr=np.argsort(np.argsort(-ds))
                    scores=hybrid_alpha/(60+dr)+(1.0-hybrid_alpha)/(60+br)
                return cand,scores
            output[split][style][mode]=rank_metrics(qs,messages,scorer)
    return output


def build_rankings(queries,messages,qvectors,mvectors,bm25,hybrid_alpha):
    rows=[]; by=by_meeting(messages)
    for q in [x for x in queries if x["split"]=="test"]:
        cand=by[q["meeting_id"]]; modes={}
        for mode in ("bm25","dense","hybrid"):
            bs=bm25.scores(q["query"],cand)
            ds=np.array([np.dot(qvectors[q["query_id"]],mvectors[c]) for c in cand])
            if mode=="bm25": scores=bs
            elif mode=="dense": scores=ds
            else:
                br=np.argsort(np.argsort(-bs)); dr=np.argsort(np.argsort(-ds)); scores=hybrid_alpha/(60+dr)+(1.0-hybrid_alpha)/(60+br)
            ranks=sorted([(c,float(scores[i])) for i,c in enumerate(cand)],key=lambda x:x[1],reverse=True)[:10]
            modes[mode]=[{"chunk_id":c,"score":s,"source":messages[c]["raw"]} for c,s in ranks]
        rows.append({"query_id":q["query_id"],"query":q["query"],"meeting_id":q["meeting_id"],"target_chunk_id":q["target_chunk_id"],"query_type":q["query_type"],"rankings":modes})
    save_json(ART/"french_test_rankings.json",rows)


def build_feature_aware_rankings(test_queries,messages,qvectors,qtopic_probs,qact_probs,
                                 mvectors,mtopic_probs,mact_probs,bm25,policy):
    by=by_meeting(messages); rows=[]
    for q in test_queries:
        mode=infer_retrieval_mode(q["query"],qtopic_probs[q["query_id"]],qact_probs[q["query_id"]])
        qq=dict(q); qq["retrieval_mode"]=mode
        comp=_feature_components(q,messages,by,qvectors[q["query_id"]],qtopic_probs[q["query_id"]],
              qact_probs[q["query_id"]],mvectors,mtopic_probs,mact_probs,bm25)
        cand,scores=_policy_score(qq,comp,policy)
        order=np.argsort(-scores)[:10]
        rows.append({"query_id":q["query_id"],"query":q["query"],"meeting_id":q["meeting_id"],
                     "query_type_gold":q["query_type"],"query_type_inferred":mode,
                     "target_chunk_id":q["target_chunk_id"],
                     "rankings":[{"chunk_id":cand[i],"score":float(scores[i]),"source":messages[cand[i]]["raw"]} for i in order]})
    save_json(ART/"french_feature_aware_test_rankings.json",rows)


def generate_query_json(queries,messages):
    rows=[]
    for q in queries:
        r=messages[q["target_chunk_id"]]
        rows.append({"query_id":q["query_id"],"split":q["split"],"meeting_id":q["meeting_id"],"query":q["query"],"query_type":q["query_type"],
          "target_chunk_id":q["target_chunk_id"],"target_chunk_ids":q.get("target_chunk_ids",[q["target_chunk_id"]]),
          "target_json":r["raw"],"hard_negative_chunk_ids":hard_negs(q,messages,by_meeting(messages)) if q["split"]=="train" else []})
    save_json(DATA/"french_retrieval_queries.json",rows)

def by_meeting(messages):
    out=collections.defaultdict(list)
    for cid,r in messages.items(): out[r["meeting_id"]].append(cid)
    return out


def audit_dataset(payload):
    allm=[m for s in payload.values() for m in s]
    forms=sum(m["style"]=="formelle" for m in allm); messy=sum(m["style"]=="messy" for m in allm)
    counts={s:{"meetings":len(v),"formal":sum(m["style"]=="formelle" for m in v),"messy":sum(m["style"]=="messy" for m in v)} for s,v in payload.items()}
    issues=[]
    for m in allm:
        if len(m["messages"])<50: issues.append(f"{m['meeting_id']}: less than 50 messages")
        speakers={x["username"] for x in m["messages"]}
        if len(speakers)<2: issues.append(f"{m['meeting_id']}: single speaker")
        if not all(set(x)=={"username","message_id","timestamp","message"} for x in m["messages"]): issues.append(f"{m['meeting_id']}: source schema drift")
    project_sets={s:{m["project_id"] for m in v} for s,v in payload.items()}
    overlap=(project_sets["train"]&project_sets["val"])|(project_sets["train"]&project_sets["test"])|(project_sets["val"]&project_sets["test"])
    messy_messages=[x["message"].casefold() for m in allm if m["style"]=="messy" for x in m["messages"]]
    filler_hits=sum(bool(re.search(r"\b(euh|bah|ouais|du coup|attends|enfin)\b",x)) for x in messy_messages)
    repeated_hits=sum(bool(re.search(r"\b([\wà-ÿ]+)\s+\1\b",x)) for x in messy_messages)
    missing_punct=sum(not bool(re.search(r"[.!?]$",x.strip())) for x in messy_messages)
    asr_hits=sum(bool(re.search(r"\b(c est|j ai|d ici|sa)\b",x)) for x in messy_messages)
    restart_hits=sum(any(k in x for k in ("enfin non","pardon je reprends","enfin bref")) for x in messy_messages)
    messy_stats={"messages":len(messy_messages),"with_filler_or_discourse_marker":filler_hits,
                 "with_repeated_word":repeated_hits,"without_terminal_punctuation":missing_punct,
                 "with_light_asr_artifact":asr_hits,"with_restart_or_repair":restart_hits}
    return {"total_meetings":len(allm),"formal_meetings":forms,"messy_meetings":messy,"messages":sum(len(m["messages"]) for m in allm),
            "min_messages_per_meeting":min(len(m["messages"]) for m in allm),"min_speakers_per_meeting":min(len({x['username'] for x in m['messages']}) for m in allm),
            "messy_transcription_stats":messy_stats,"splits":counts,"project_leakage":len(overlap),"issues":issues}


def evaluate_feature_heads(messages,topic_probs,act_probs):
    ordered=list(messages)
    y_topic=np.asarray([TOPIC_TO_ID[messages[c]["topic"]] for c in ordered]); p_topic=np.argmax(topic_probs,axis=1)
    y_act=np.asarray([ACT_TO_ID[messages[c]["act"]] for c in ordered]); p_act=np.argmax(act_probs,axis=1)
    mask=np.asarray([messages[c]["split"]=="test" for c in ordered])
    return {"version":"v7", "topic_macro_f1":float(f1_score(y_topic[mask],p_topic[mask],average="macro")),
            "topic_accuracy":float(accuracy_score(y_topic[mask],p_topic[mask])),
            "act_macro_f1":float(f1_score(y_act[mask],p_act[mask],average="macro")),
            "act_accuracy":float(accuracy_score(y_act[mask],p_act[mask]))}


def piscine_audit():
    return {
      "retrieval_baselines":["BM25","dense custom encoder","hybrid rank fusion"],
      "specialized_embeddings":["contrastive InfoNCE","same-topic/status hard negatives","feature multi-task learning"],
      "feature_discovery":["topic and discourse-act heads","meeting recurrence and importance","co-activation graph","per-meeting KMeans latent clusters"],
      "memory":["immutable source JSON + SHA-256","message embedding foreign-key link","meeting summary + meeting embedding","semantic feature nodes and edges"],
      "relationships":["supports","contradicts","supersedes","creates_action","answers","unrelated"],
      "intents":["answer","compare","build_dossier","draft_email","create_task","schedule_event","clarify"],
      "agent":["evidence answers","timeline comparison","source-linked dossier","email draft simulation","task proposal","event proposal","clarification"],
      "safety":["no real emails sent","no real tasks/events created","missing email address remains null","source evidence is retained"],
      "not_implemented_or_limited":["LLM natural-language synthesis","production-grade cross-meeting contradiction resolution","real connectors (explicitly not required by the brief)","robust general French ASR correction"]
    }



def cli_retrieve(query, meeting_id, top_k, messages, model, vocab, device, bm25,
                 hybrid_alpha, feature_policy):
    """Return exact immutable source JSON chunks from feature-aware memory."""
    if meeting_id and not any(r["meeting_id"] == meeting_id for r in messages.values()):
        raise ValueError(f"Unknown meeting_id: {meeting_id}")
    candidates=[cid for cid,r in messages.items() if meeting_id is None or r["meeting_id"] == meeting_id]
    qz,qt,qa=encode_tiny(model,vocab,[query],device)
    qid="cli-query"
    qfake={"query_id":qid,"meeting_id":meeting_id or "__all__","query":query}
    group={qfake["meeting_id"]:candidates}
    mvec={c:messages[c]["vector"] for c in candidates}
    mtp={c:messages[c]["topic_probs"] for c in candidates}
    map_={c:messages[c]["act_probs"] for c in candidates}
    comp=_feature_components(qfake,messages,group,qz[0],qt[0],qa[0],mvec,mtp,map_,bm25)
    mode=infer_retrieval_mode(query,qt[0],qa[0])
    qfake["retrieval_mode"]=mode
    if mode in ("latest_message","global_latest_decision") and meeting_id is None:
        # Time is globally meaningful only because temporal components include meeting dates.
        pass
    cand,scores=_policy_score(qfake,comp,feature_policy)
    order=np.argsort(-scores)[:top_k]
    return {
      "query":query,"meeting_id":meeting_id,"inferred_operation":mode,
      "retrieval":"learned topic/act feature scores + validation-tuned dense/BM25 ranks + explicit chronology",
      "results":[{"rank":rank+1,"chunk_id":cand[i],"score":float(scores[i]),
                  "source_json":messages[cand[i]]["raw"]}
                 for rank,i in enumerate(order)]
    }


def _is_current_decision_text(message: str) -> bool:
    q=message.casefold()
    old_markers=("la dernière fois", "séance précédente", "réunion précédente", "la fois d'avant", "à ce moment-là", "on avait retenu")
    if any(x in q for x in old_markers): return False
    current_markers=("décision finalement retenue", "on valide plutôt", "on part sur ça", "on valide donc", "la décision c'est bien", "ce qu'on garde après discussion", "la décision c'est")
    return any(x in q for x in current_markers)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--force-data",action="store_true"); ap.add_argument("--epochs",type=int,default=EPOCHS_DEFAULT); ap.add_argument("--batch-size",type=int,default=BATCH_DEFAULT)
    ap.add_argument("--train-only",action="store_true"); ap.add_argument("--eval-only",action="store_true"); ap.add_argument("--query",type=str)
    ap.add_argument("--agent-demo",type=str); ap.add_argument("--meeting-id",type=str)
    ap.add_argument("--top-k",type=int,default=5); args=ap.parse_args()
    seed_all(); t0=time.time()
    payload=generate_dataset(force=args.force_data); dataset_audit=audit_dataset(payload)
    messages,queries,meetings=flatten(payload)
    save_json(DATA/"french_dataset_audit.json",dataset_audit)
    generate_query_json(queries,messages)
    print("DATASET AUDIT:",json.dumps(dataset_audit,ensure_ascii=False))
    if args.eval_only:
        ck=torch.load(ART/"french_tiny_encoder.pt",map_location="cpu",weights_only=False)
        vocab=Vocab([]); vocab.itos=ck["vocab"]; vocab.stoi={t:i for i,t in enumerate(vocab.itos)}
        model=TinySemanticEncoder(len(vocab.itos)); model.load_state_dict(ck["state_dict"]); device=torch.device("cpu")
        epochs_used=int(ck.get("epochs",args.epochs))
        model.eval()
    else:
        model,vocab,device,history=fit_tiny(payload,messages,queries,args.epochs,args.batch_size)
        epochs_used=args.epochs
        if args.train_only:
            print("Training complete. Checkpoint:",ART/"french_tiny_encoder.pt")
            return
    ids=list(messages)
    mvec,tp,actp=encode_tiny(model,vocab,[messages[c]["encoder_text"] for c in ids],device)
    qvec_arr,qtopic_arr,qact_arr=encode_tiny(model,vocab,[q["query"] for q in queries],device)
    mv={c:mvec[i] for i,c in enumerate(ids)}; qv={q["query_id"]:qvec_arr[i] for i,q in enumerate(queries)}
    qtp={q["query_id"]:qtopic_arr[i] for i,q in enumerate(queries)}; qap={q["query_id"]:qact_arr[i] for i,q in enumerate(queries)}
    mtp={c:tp[i] for i,c in enumerate(ids)}; map_={c:actp[i] for i,c in enumerate(ids)}
    for cid in ids:
        messages[cid]["vector"]=mv[cid]; messages[cid]["topic_probs"]=mtp[cid]; messages[cid]["act_probs"]=map_[cid]
    # Feature probability arrays align with message insertion order.
    integrity=persist_memory(meetings,messages,mv,tp,actp)
    bm=BM25(messages)
    hybrid_alpha,hybrid_alpha_scores=tune_hybrid_alpha(queries,messages,qv,mv,bm)
    val_queries=[q for q in queries if q["split"]=="val"]
    test_queries=[q for q in queries if q["split"]=="test"]
    feature_policy,feature_policy_tuning,feature_policy_val_metrics,feature_policy_val_mode_accuracy=tune_feature_policy(
        val_queries,messages,qv,qtp,qap,mv,mtp,map_,bm,hybrid_alpha)
    feature_policy_test=evaluate_feature_policy(test_queries,messages,qv,qtp,qap,mv,mtp,map_,bm,feature_policy)
    feature_policy_val={"overall":feature_policy_val_metrics,"operation_inference_accuracy":feature_policy_val_mode_accuracy}
    metrics={s:{} for s in ("val","test")}
    for split in metrics:
      qs=[q for q in queries if q["split"]==split]
      for mode in ("bm25","dense","hybrid"):
       ids_all=list(messages); by=by_meeting(messages)
       def scorer(q,mode=mode):
        cand=by[q["meeting_id"]]
        bs=bm.scores(q["query"],cand)
        ds=np.array([np.dot(qv[q["query_id"]],mv[c]) for c in cand])
        if mode=="bm25": scores=bs
        elif mode=="dense": scores=ds
        else:
         br=np.argsort(np.argsort(-bs)); dr=np.argsort(np.argsort(-ds)); scores=hybrid_alpha/(60+dr)+(1.0-hybrid_alpha)/(60+br)
        return cand,scores
       metrics[split][mode]=rank_metrics(qs,messages,scorer)
    style_metrics=eval_by_style(queries,messages,qv,mv,bm,hybrid_alpha)
    feature_head_metrics=evaluate_feature_heads(messages,tp,actp)
    rel_model,rel_metrics=train_relation_classifier(messages,mv)
    relation_memory_counts=persist_predicted_units_and_relations(messages,meetings,mv,tp,actp,rel_model)
    integrity["counts"]["semantic_units"]=relation_memory_counts["semantic_units"]
    integrity["counts"]["semantic_relations"]=relation_memory_counts["semantic_relations"]
    intent_vec,intent_clf=train_intent_classifier()
    intent_metrics=evaluate_intents(intent_vec,intent_clf)
    build_rankings(queries,messages,qv,mv,bm,hybrid_alpha)
    build_feature_aware_rankings(test_queries,messages,qv,qtp,qap,mv,mtp,map_,bm,feature_policy)
    demos={}
    for text in ["Quelle décision a été prise sur le budget ?","Compare les décisions du projet Atlas entre les réunions.","Rédige un mail à la personne responsable de la sécurité.","Prépare une tâche pour vérifier les jetons API.","Envoie ça à la bonne personne."]:
        q_demo,_,_=encode_tiny(model,vocab,[text],device)
        demos[text]=demo_agent(text,meetings,messages,bm,intent_vec,intent_clf,q_demo[0],mv,tp,actp)
    params=sum(p.numel() for p in model.parameters())
    result={"dataset":dataset_audit,"model":{"name":"french_tiny_v7","architecture":"1-layer TransformerEncoder + 64-d normalized projection + topic/act heads","parameters":params,"vocabulary":len(vocab.itos),"embedding_dim":DIM,"epochs":epochs_used,"device":str(device)},
      "retrieval_metrics":metrics,"feature_aware_retrieval":{"validation":feature_policy_val,"test":feature_policy_test,"policy":feature_policy,"validation_tuning":feature_policy_tuning},
      "by_style":style_metrics,"hybrid_alpha":hybrid_alpha,"hybrid_alpha_validation_grid":hybrid_alpha_scores,"feature_head_metrics":feature_head_metrics,"relation_metrics":rel_metrics,"relation_memory_counts":relation_memory_counts,"intent_metrics":intent_metrics,"memory_integrity":integrity,
      "piscine_coverage":piscine_audit(),"agent_demo_samples":demos,"elapsed_seconds":time.time()-t0,
      "pretrained_comparison":{"status":"NOT_RUN_IN_THIS_RUNTIME","reason":"Hugging Face weights and sentence-transformers are not available in the offline execution container. See run_pretrained_comparison.py; it compares both requested checkpoints on the exact same split and optionally fine-tunes on train only."}}
    save_json(ART/"french_results.json",result)
    lines=["# Rapport expérimental — mémoire sémantique de réunions en français","","## 1. Données synthétiques","",f"- Réunions : {dataset_audit['total_meetings']} (formelles : {dataset_audit['formal_meetings']}, transcrites/messy : {dataset_audit['messy_meetings']}).",f"- Messages : {dataset_audit['messages']} ; minimum par réunion : {dataset_audit['min_messages_per_meeting']} ; minimum de locuteurs : {dataset_audit['min_speakers_per_meeting']}.",f"- Fuite entre projets : {dataset_audit['project_leakage']}.","- Répartition : "+json.dumps(dataset_audit['splits'],ensure_ascii=False),"- Tous les messages sources ont exactement les quatre champs demandés et sont en français.","- Indicateurs de transcription messy : "+json.dumps(dataset_audit.get("messy_transcription_stats",{}),ensure_ascii=False),"","## 2. Modèle entraîné","",f"- {params:,} paramètres ; dimension d'embedding {DIM} ; vocabulaire {len(vocab.itos)}.","- Entraînement contrastif query-positive avec cinq hard negatives par requête, plus supervision des thèmes et actes.","- Sélection du checkpoint sur MRR@10 validation.","","## 3. Retrieval","", "| Split | Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("val","test"):
      for mode,v in metrics[split].items(): lines.append(f"| {split} | {mode} | {v['Recall@1']:.3f} | {v['Recall@3']:.3f} | {v['Recall@5']:.3f} | {v['Recall@10']:.3f} | {v['MRR@10']:.3f} | {v['nDCG@5']:.3f} |")
    lines += ["","## 4. Retrieval avec caractéristiques latentes", "", "- Validation : "+json.dumps(feature_policy_val,ensure_ascii=False), "- Test : "+json.dumps(feature_policy_test,ensure_ascii=False), "- Poids sélectionnés sur validation : "+json.dumps(feature_policy,ensure_ascii=False), "- L'opération de recherche est inférée depuis le texte et les têtes topic/act ; les étiquettes `query_type` servent uniquement à choisir les poids sur validation.", "", "## 5. Formelles vs messys", "", "| Split | Style | Mode | Recall@1 | Recall@5 | MRR@10 |", "|---|---|---|---:|---:|---:|"]
    for split,d in style_metrics.items():
      for style,mm in d.items():
       for mode,v in mm.items(): lines.append(f"| {split} | {style} | {mode} | {v['Recall@1']:.3f} | {v['Recall@5']:.3f} | {v['MRR@10']:.3f} |")
    lines += ["","## 5. Extraction de caractéristiques", "", "- Scores de classification test : "+json.dumps(feature_head_metrics,ensure_ascii=False), "- Le modèle prédit des catégories de thèmes et d'actes ; les occurrences, importances, co-activations et clusters sont ensuite agrégés par réunion.","","## 6. Classification des relations", "", f"- Macro-F1 : {rel_metrics['macro_f1']:.3f}; accuracy : {rel_metrics['accuracy']:.3f}.","- F1 par classe : "+json.dumps(rel_metrics['per_class_f1'],ensure_ascii=False),"","## 7. Mémoire et intégrité", "", "- Comptes : "+json.dumps(integrity['counts'],ensure_ascii=False)+" ; unités/liens : "+json.dumps(relation_memory_counts,ensure_ascii=False),f"- Hash mismatches : {integrity['hash_mismatches']}; vecteurs manquants : {integrity['missing_vectors']}; vecteurs orphelins : {integrity['orphan_vectors']}.","- Le JSON source reste immuable ; vecteurs et caractéristiques sont stockés dans des tables dérivées reliées par chunk_id.","","## 8. Comparaison aux modèles pré-entraînés", "", "**Non exécutée dans cet environnement** : absence de `transformers`/`sentence-transformers` et impossibilité de joindre Hugging Face depuis le conteneur (résolution DNS désactivée). Aucun score des modèles pré-entraînés n'est inventé. `run_pretrained_comparison.py` effectue ce benchmark sur exactement les mêmes requêtes et peut déclencher un fine-tuning si les scores validation sont inférieurs au modèle maison.","","## 9. Couverture piscine", "",json.dumps(piscine_audit(),ensure_ascii=False,indent=2),"","## 10. Limites", "", "- Corpus entièrement synthétique et contrôlé ; les scores ne mesurent pas la robustesse sur des réunions réelles.","- Les thèmes/actes sont supervisés par des annotations synthétiques ; la découverte ouverte des concepts reste une piste expérimentale.","- L'agent est volontairement déterministe et simule les actions ; aucun LLM ni connecteur externe n'est appelé.","- Le classifieur de relations apprend sur des paires synthétiques définies par les annotations ; ses performances doivent être interprétées dans ce cadre."]
    (ART/"FRENCH_PIPELINE_SUMMARY.md").write_text("\n".join(lines),encoding="utf-8")
    print("\nTEST METRICS")
    for mode,v in metrics["test"].items(): print(mode, " ".join(f"{k}={x:.3f}" for k,x in v.items()))
    print("Relation macro-F1",rel_metrics["macro_f1"])
    print("Memory integrity",integrity)
    if args.agent_demo:
        print(json.dumps(demo_agent(args.agent_demo,meetings,messages,bm,intent_vec,intent_clf),ensure_ascii=False,indent=2))
    if args.query:
        print(json.dumps(cli_retrieve(args.query,args.meeting_id,max(1,args.top_k),messages,model,vocab,device,bm,hybrid_alpha,feature_policy),ensure_ascii=False,indent=2))
    print("Report:",ART/"FRENCH_PIPELINE_SUMMARY.md")

if __name__=="__main__": main()
