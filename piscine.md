   Intelligence Lab · Makers  #ctre\_wnd { position: fixed; bottom: 0; right: 10px; background: #fff; box-shadow: 0px 0px 40px rgba(0,0,0,0.15); border-radius: 3px 3px 0 0; z-index: 2147483647; } @media (prefers-color-scheme: dark) { #ctre\_wnd { background: #000; box-shadow: 0px 0px 40px rgba(255,255,255,0.15); } }

[![Intelligence Lab](/brand/symbole.png)![Intelligence Lab](/brand/symbole-dark.png)Intelligence Lab · Makers](/etudiant)

[![Intelligence Lab](/brand/symbole.png)![Intelligence Lab](/brand/symbole-dark.png)Intelligence Lab · Makers](/etudiant)

Piscine Makers

[Mon projet](/etudiant/projet)[Sujets](/etudiant/sujets)[Mon compte](/etudiant/compte)

JCJonathan CHAYNAÉtudiant

[Sujets](/etudiant/sujets)

[![](/api/avatars/10-e27f9310b89e.webp)Mentor référentMartial Roberge](/etudiant/mentors/10 "Directeur Général de Gilbert")

PythonNLPDeep LearningEmbeddingsTransformersContrastive LearningFine-tuningRecherche sémantiqueRAGAgents IA

EMBEDDING LAB De la mémoire sémantique à l'agent intelligent
============================================================

Construis en trois semaines le moteur d’intelligence sémantique d’un agent IA. Entraîne un modèle d’embeddings capable de comprendre les relations entre conversations, décisions, actions et personnes. Explore le contrastive learning, les Transformers et la recherche vectorielle pour développer un agent capable de répondre à des questions complexes, de constituer des dossiers documentés et de préparer des actions intelligentes, comme rédiger un e-mail à la bonne personne. Recherche tes propres datasets, compare différentes architectures et réalise une démonstration fonctionnelle.

Mon sujet

**Durée :** 3 semaines **Format :** projet individuel **Domaines :** NLP, représentation sémantique, apprentissage contrastif, recherche d'information, IA agentique

Contexte
--------

Les assistants conversationnels savent générer du texte, mais ils ne savent pas toujours **retrouver la bonne information, comprendre ce qu'elle signifie réellement, identifier les personnes concernées et choisir l'action appropriée**. Une recherche vectorielle généraliste peut rapprocher des formulations semblables qui expriment pourtant des intentions opposées : « le budget est proposé », « le budget est validé » et « le budget est annulé ».

Le problème devient plus difficile lorsqu'un assistant dispose d'une mémoire composée de réunions, d'échanges, de notes et, à terme, de contenus issus d'outils externes. Une question peut exiger de croiser plusieurs sources, de suivre l'évolution d'une décision, de distinguer un fait confirmé d'une hypothèse, puis de préparer un document ou une communication à destination des bonnes personnes.

Cette piscine propose de construire **le moteur de compréhension sémantique d'un agent**, avec les embeddings comme objet central de recherche. Le but n'est pas de développer un simple chatbot ni d'assembler des appels à une API : il s'agit d'étudier, entraîner et évaluer des représentations qui permettent à un système de **retrouver, relier, vérifier et exploiter** les informations pertinentes avant toute réponse ou action.

Problématique
-------------

**Comment entraîner des embeddings spécialisés capables de relier les questions des utilisateurs aux bonnes informations, de distinguer des relations sémantiques fines et d'alimenter un agent qui répond, construit des dossiers et prépare des actions pertinentes ?**

Le système devra traiter quatre difficultés :

1.  **Pertinence :** retrouver les éléments utiles à une demande, même si les mots employés sont différents.
2.  **Précision sémantique :** différencier proposition, décision, annulation, contradiction, engagement et simple mention.
3.  **Contexte :** relier des informations dispersées entre plusieurs conversations et conserver leur provenance, leur date et leur statut.
4.  **Intention et destinataire :** comprendre si l'utilisateur veut une réponse, une synthèse, un dossier, une tâche ou un brouillon d'e-mail, puis déterminer sur quelles informations et vers quels destinataires cette action doit porter.

Mission
-------

Développer un prototype de **Semantic Agent Engine** qui transforme des sources textuelles en unités de connaissance interrogeables, entraîne ou adapte un modèle d'embeddings et utilise ces représentations dans un agent capable de :

*   Répondre à des questions complexes en s'appuyant sur des preuves identifiables.
*   Retrouver les décisions, actions, objections et informations associées à un sujet.
*   Distinguer une proposition d'une décision confirmée et détecter les informations devenues obsolètes.
*   Comparer plusieurs conversations pour expliquer ce qui a changé.
*   Produire un **dossier structuré** à partir de plusieurs sources : contexte, faits établis, chronologie, décisions, points ouverts, responsables et références.
*   Reconnaître une demande d'action, choisir un outil adapté et produire un **brouillon structuré**.
*   Préparer, par exemple, un e-mail destiné à la **bonne personne**, avec le bon objet et les bonnes informations, en justifiant le choix du destinataire à partir des sources disponibles.
*   Signaler explicitement toute information manquante ou ambiguë au lieu d'inventer un fait, un contact ou une décision.

**Important :** le prototype peut simuler les connecteurs (messagerie, agenda, gestion de tâches, documents). Il ne doit ni envoyer de messages ni modifier de services externes. Toute action reste une proposition à confirmer.

Exemples d'interactions attendues
---------------------------------

### Retrouver et comprendre

> « Quelle décision a finalement été prise concernant le budget du projet Atlas ? »

Le moteur doit retrouver la décision **en vigueur**, pas seulement le passage lexicalement le plus proche, et fournir les extraits qui la justifient.

> « Est-ce que la réunion de jeudi contredit celle de lundi ? »

Le système retrouve les informations concernées, identifie leur relation (`supports`, `contradicts`, `supersedes`, `unrelated`) et explique la différence entre une contradiction réelle et une décision mise à jour.

> « Qui s'est engagé à transmettre le dossier et pour quand ? »

Le système associe une action à son responsable et à son échéance, avec une référence à l'intervention source.

### Construire un dossier

> « Prépare un dossier complet sur le projet Atlas : historique, décisions, budget, risques, prochaines étapes et personnes impliquées. »

L'agent doit rechercher les sources pertinentes, regrouper les informations par sujet, éliminer les doublons, résoudre les versions contradictoires et produire un document structuré. Chaque affirmation importante doit être reliée à une preuve ; les informations absentes doivent être signalées. Le dossier peut être généré en **Markdown** ou en **JSON structuré**, avec export PDF/DOCX comme extension facultative.

### Préparer une communication

> « Rédige un mail à la personne responsable du budget pour lui demander la dernière version validée avant vendredi. »

L'agent doit :

1.  Identifier l'intention : `draft_email`.
2.  Retrouver la personne responsable du budget dans les sources disponibles.
3.  Vérifier que cette attribution est explicite et suffisamment récente.
4.  Récupérer le contexte et les décisions pertinents.
5.  Préparer un brouillon d'e-mail factuel, avec objet, destinataire **si connu**, corps et références internes.
6.  Demander une clarification si plusieurs personnes correspondent ou si l'adresse n'est pas disponible.

**Aucune adresse e-mail ne doit être inventée.** Le choix du destinataire doit être traçable ; une mention d'une personne dans une réunion ne suffit pas à en faire automatiquement le responsable du sujet.

### Distinguer information et action

> « On pourrait peut-être revoir le budget. »

Cette phrase ne constitue ni une décision confirmée ni une instruction d'envoyer un message. Le système ne doit pas créer d'action à partir d'une simple hypothèse.

Axes de recherche et algorithmes
--------------------------------

### 1\. Représentation sémantique et embeddings

Étudier les représentations vectorielles de phrases, de tours de parole et d'unités métier. Comparer au minimum une approche lexicale et un encodeur pré-entraîné :

*   **Baseline lexicale :** TF-IDF, BM25.
*   **Embeddings denses :** Sentence-BERT, E5, BGE ou équivalents multilingues.
*   **Similarité :** cosinus, recherche des plus proches voisins, indexation FAISS ou équivalent.
*   **Stratégies de représentation :** chunking fixe, segmentation par tour de parole, unités sémantiques structurées.

Une attention particulière doit être portée aux requêtes courtes, aux reformulations, à la négation, aux nombres, aux dates et aux rôles des participants.

### 2\. Apprentissage contrastif et spécialisation des embeddings

**Cœur ML/DL obligatoire de la piscine :** adapter un encodeur à des requêtes et des conversations liées aux décisions, responsabilités et actions.

Construire des couples ou triplets d'entraînement :

Ancre

Positif

Négatif difficile

« Quelle date a été validée ? »

« Le lancement est confirmé au 22 novembre. »

« Le 22 novembre a été proposé, sans validation. »

« Qui doit envoyer le dossier ? »

« Marie s'engage à transmettre le dossier. »

« Marie demande à Karim d'envoyer le dossier. »

« Le budget est-il approuvé ? »

« Le budget a été voté et adopté. »

« Le budget a été rejeté après discussion. »

Expérimenter notamment **InfoNCE**, **Multiple Negatives Ranking Loss**, **Triplet Loss** ou une variante de **Supervised Contrastive Learning**. Mettre en place une stratégie de **hard negative mining** : exemples proches par le thème, mais faux par le statut, la personne, la date ou la négation.

Comparer systématiquement l'encodeur adapté au modèle initial, sur des données de test indépendantes. Étudier les effets du choix des négatifs, du contexte ajouté et de la taille du modèle.

**Point scientifique important :** une proximité vectorielle n'est pas une preuve de compatibilité logique. Le modèle d'embeddings doit améliorer le _retrieval_ ; les relations de contradiction, de validation et de remplacement peuvent nécessiter un **modèle complémentaire de classification ou de reranking**.

### 3\. Compréhension des relations et du contexte

Construire un module capable de caractériser les relations entre unités :

*   `supports` : une source étaye une affirmation ;
*   `contradicts` : deux affirmations sont incompatibles dans le même contexte ;
*   `supersedes` : une information plus récente remplace une ancienne ;
*   `creates_action` : une décision ou un engagement donne lieu à une action ;
*   `answers` : une intervention répond à une question ;
*   `unrelated` : les unités ne sont pas liées.

Approches possibles : **régression logistique sur embeddings**, petit **MLP**, **Cross-Encoder**, **Natural Language Inference (NLI)**, classification Transformer. Une relation temporelle ne doit pas être déduite de la seule similarité : exploiter aussi les dates, les identifiants de projet et les passages justificatifs.

### 4\. Mémoire sémantique et recherche hybride

Construire une mémoire interrogeable composée d'unités telles que `decision`, `action`, `commitment`, `question`, `answer`, `fact`, `risk`, `idea`.

Chaque unité conserve au minimum son texte, son type, sa source, sa date et, si disponible, son locuteur, son responsable et son extrait justificatif. Les liens entre unités doivent permettre de suivre une information d'une conversation à l'autre.

Comparer plusieurs stratégies : recherche lexicale, recherche dense, fusion hybride, filtres par type ou date, reranking. Pour une question sur « la décision actuelle », l'agent doit pouvoir retrouver **l'historique** tout en privilégiant l'état le plus récent **justifié**.

### 5\. Analyse fine des intentions utilisateur

Une même demande peut impliquer plusieurs opérations. Le système doit distinguer notamment :

Intention

Exemple

Sortie attendue

`answer`

« Qu'a-t-on décidé ? »

Réponse sourcée

`compare`

« Qu'est-ce qui a changé ? »

Comparaison chronologique

`build_dossier`

« Fais un dossier sur Atlas. »

Document structuré et sourcé

`draft_email`

« Écris au responsable du projet. »

Brouillon et destinataire justifié

`create_task`

« Prépare une tâche pour Marie. »

Proposition de tâche

`schedule_event`

« Prépare un point vendredi. »

Proposition d'événement

`clarify`

« Envoie ça à la bonne personne. » sans contexte suffisant

Question de clarification

Étudier la classification d'intentions, l'extraction de **slots** (personne, sujet, échéance, action), la détection d'ambiguïtés et le routage vers des outils simulés. Un classifieur léger, un Transformer adapté ou une approche hybride règles + ML peuvent être comparés.

### 6\. Agent de recherche et de préparation d'actions

Assembler les modules dans un **mini-agent** dont les étapes sont explicites :

    Demande utilisateur
           |
           v
    Analyse d'intention + extraction des paramètres
           |
           v
    Recherche hybride dans la mémoire
           |
           v
    Reranking + vérification des relations
           |
           v
    Résolution du contexte, des dates et des personnes
           |
           v
    Choix de l'opération
           |
           +--> Réponse sourcée
           +--> Dossier structuré
           +--> Brouillon d'e-mail
           +--> Proposition de tâche / événement
           +--> Demande de clarification
    

L'utilisation d'un LLM pour formuler les réponses et documents est autorisée, mais **la recherche, les représentations, les relations et les décisions de routage doivent être évaluées séparément**. Le LLM ne doit pas masquer les erreurs du moteur sémantique.

Données : recherche et constitution autonomes
---------------------------------------------

**Aucun dataset n'est fourni.** L'étudiant doit rechercher, sélectionner, documenter et préparer ses données, en vérifiant leurs licences et conditions d'utilisation.

Pistes à explorer, sans obligation de les utiliser : corpus de réunions ou de dialogues publics (AMI, ICSI, MeetingBank, QMSum), jeux de recherche d'information, données de paraphrases et de NLI, conversations synthétiques construites pour tester les décisions et les responsabilités.

Les corpus publics ne contiennent pas nécessairement les annotations correspondant à toutes les tâches du projet. L'étudiant devra donc concevoir un **schéma d'annotation**, produire ou adapter des exemples et justifier leur qualité. Les conversations synthétiques sont possibles, mais doivent être complétées par une évaluation qui évite de mesurer uniquement la capacité du modèle à reconnaître des formulations générées selon les mêmes patrons.

Le corpus doit comporter des cas difficiles :

*   proposition versus décision ;
*   accord conditionnel versus validation définitive ;
*   changement de date, de budget ou de responsable ;
*   même sujet discuté dans plusieurs réunions ;
*   informations obsolètes ou contradictoires ;
*   personne citée versus personne réellement responsable ;
*   demande d'information versus instruction de préparer une action ;
*   destinataire absent ou ambigu ;
*   questions sans réponse dans les sources.

Séparer entraînement, validation et test **par conversation ou groupe de réunions**, et pas seulement par phrase, pour éviter les fuites de données. Conserver, si possible, un jeu de test comportant des projets, formulations et personnes jamais vus à l'entraînement.

Ne pas utiliser de transcriptions privées ou d'informations personnelles réelles sans autorisation appropriée. Pour les e-mails, utiliser des contacts fictifs ou des données de démonstration.

Contrats de données recommandés
-------------------------------

### Unité sémantique

    {
      "unit_id": "u-014",
      "source_id": "meeting-03",
      "source_date": "2026-10-12",
      "type": "decision",
      "statement": "Le lancement du projet Atlas est reporté au 22 novembre.",
      "speaker": "Marie",
      "evidence": "On confirme le report au 22 novembre.",
      "status": "active"
    }
    

### Relation

    {
      "from_id": "u-014",
      "to_id": "u-006",
      "relation": "supersedes",
      "evidence_source_id": "meeting-03"
    }
    

### Brouillon d'e-mail préparé par l'agent

    {
      "intent": "draft_email",
      "tool_candidate": "email.create_draft",
      "recipient": {
        "name": "Marie",
        "email": null,
        "evidence_unit_ids": ["u-011"],
        "resolution_status": "email_missing"
      },
      "subject": "Projet Atlas — version validée du budget",
      "body": "Bonjour Marie,\n\nPourrais-tu me transmettre la dernière version validée du budget du projet Atlas avant vendredi ?\n\nMerci.",
      "source_unit_ids": ["u-011", "u-014"],
      "needs_confirmation": true,
      "executed": false
    }
    

Cet exemple illustre un point essentiel : **identifier la personne pertinente ne signifie pas disposer de son adresse**. Un futur connecteur de contacts ou de messagerie pourrait résoudre cette donnée ; en son absence, l'agent doit laisser le champ vide et demander une confirmation.

Démonstration attendue
----------------------

Une interface simple (CLI, Streamlit, Gradio ou API) permet de charger plusieurs conversations et de dialoguer avec le moteur.

La démonstration doit montrer, de bout en bout :

1.  Une question sur une décision précise, avec preuve et référence.
2.  Une question ambiguë opposant proposition et décision.
3.  Une comparaison entre deux réunions avec modification d'une information.
4.  La création d'un **dossier multi-sources** dont les affirmations sont traçables.
5.  La préparation d'un **e-mail à la personne pertinente**, avec justification du destinataire.
6.  Un cas où l'agent **demande une précision** plutôt que de choisir un destinataire ou d'inventer une information.
7.  Un cas où une phrase hypothétique **ne déclenche aucune action**.

Un agent simple et fiable, fondé sur une recherche robuste et des actions simulées, vaut mieux qu'un agent spectaculaire dont les réponses ne peuvent pas être vérifiées.

Évaluation expérimentale
------------------------

Les étudiants définissent un protocole reproductible et comparent leurs modèles sur les mêmes jeux de test.

Composant

Mesures pertinentes

Retrieval

Recall@K, MRR, nDCG@K

Embeddings spécialisés

Gain par rapport à l'encodeur initial, performance sur hard negatives

Relations

Macro-F1, précision/rappel par relation, matrice de confusion

Intentions et paramètres

Macro-F1, exactitude des slots, taux de clarification appropriée

Résolution des destinataires

Precision@1, taux de faux destinataires, taux d'abstention justifiée

Réponses et dossiers

Couverture des faits, exactitude des citations, taux d'affirmations non étayées

Système

Latence, mémoire, coût d'inférence

L'analyse doit inclure des erreurs qualitatives : négations, changements temporels, confusions de personnes, mauvaises attributions, documents incomplets et requêtes sans preuve disponible.

Organisation sur trois semaines
-------------------------------

### Semaine 1 — Recherche, données et baselines

*   Étudier les embeddings, la recherche sémantique, le contrastive learning et le retrieval conversationnel.
*   Rechercher les jeux de données et vérifier leurs conditions de réutilisation.
*   Définir les unités sémantiques, les relations et les intentions.
*   Constituer un premier corpus avec questions, preuves et négatifs difficiles.
*   Construire une baseline BM25 et une baseline d'embeddings pré-entraînés.
*   Mesurer les premiers résultats et identifier les confusions les plus fréquentes.

**Résultat attendu :** données documentées, protocole de test et baseline chiffrée.

### Semaine 2 — Entraînement et analyse sémantique

*   Construire des paires ou triplets contrastifs.
*   Fine-tuner un encodeur de taille raisonnable, compatible avec les ressources disponibles.
*   Expérimenter le hard negative mining.
*   Entraîner ou configurer un modèle de classification des relations.
*   Étudier le reranking et les filtres temporels ou par type.
*   Comparer les résultats avec les baselines et conduire au moins une étude d'ablation.

**Résultat attendu :** modèle adapté, résultats reproductibles et analyse des erreurs.

### Semaine 3 — Agent, dossiers et actions simulées

*   Indexer les unités et leurs relations dans une mémoire interrogeable.
*   Implémenter la détection des intentions et l'extraction des paramètres utiles.
*   Construire la génération de réponses et de dossiers avec citations.
*   Préparer des brouillons d'e-mails, tâches ou événements via des outils simulés.
*   Tester la résolution des destinataires et les demandes de clarification.
*   Mesurer les performances de bout en bout, stabiliser la démonstration et documenter les limites.

**Résultat attendu :** prototype fonctionnel, démonstration et rapport comparatif.

Livrables
---------

*   **Dépôt de code** structuré, exécutable et documenté.
*   **Documentation des datasets** : provenance, licences, préparation, annotations et splits.
*   **Baselines et modèle d'embeddings adapté**, avec scripts d'entraînement et d'inférence.
*   **Module de recherche sémantique** renvoyant les unités, scores et preuves.
*   **Module de relations** distinguant au minimum pertinence, contradiction et remplacement.
*   **Module d'intentions** identifiant les demandes de réponse, dossier et préparation d'action.
*   **Mini-agent fonctionnel** avec interface ou API, capable de générer des dossiers et des brouillons sourcés.
*   **Connecteurs simulés** et contrats JSON pour les actions, sans exécution réelle.
*   **Rapport expérimental** : comparaisons, métriques, ablations, exemples d'erreurs, coûts et limites.
*   **Démonstration reproductible** sur plusieurs conversations.

Pistes technologiques
---------------------

**Langage et apprentissage :** Python, PyTorch, Hugging Face Transformers, Sentence Transformers, scikit-learn.

**Retrieval :** BM25, FAISS, recherche hybride, Cross-Encoders, reranking.

**Modèles :** encodeurs multilingues de la famille E5/BGE/Sentence-BERT, classifieurs NLI ou Transformers légers. Le choix doit être argumenté et compatible avec le calcul disponible.

**Agent et interface :** FastAPI, Streamlit ou Gradio ; sorties JSON validées par schéma ; outils simulés pour messagerie, documents, tâches et agenda.

**Évaluation :** pytest, scripts de benchmark, mesures de latence, suivi des expériences.

Les bibliothèques et architectures ne sont pas imposées : elles doivent servir une hypothèse expérimentale explicite.

Limites et prolongements
------------------------

Cette piscine **n'exige pas** l'entraînement d'un grand modèle de langage depuis zéro, une intégration OAuth, l'envoi d'e-mails, la synchronisation avec une messagerie réelle ou la création native de documents bureautiques complexes. Ces développements détourneraient du cœur scientifique : **la qualité des représentations sémantiques et leur utilisation fiable par un agent**.

À plus long terme, le moteur pourrait être connecté à une messagerie, un annuaire de contacts, un agenda, une base documentaire ou un gestionnaire de projets. Le système devrait alors vérifier les permissions, résoudre les identités à partir de sources autorisées et demander une confirmation explicite avant toute action ayant un effet externe.

Question finale
---------------

**Peut-on entraîner un modèle d'embeddings qui ne se contente pas de reconnaître qu'on parle du même sujet, mais qui aide un agent à comprendre ce qui a été décidé, qui est concerné, quelle information est encore valable et quelle réponse ou action est réellement justifiée ?**

Mon sujet

16 oct. 2026 → 16 nov. 2026

Sujet choisi

Tu as choisi ce sujet

[Aller à mon projet](/etudiant/projet)

Pièces jointes

Aucune.
