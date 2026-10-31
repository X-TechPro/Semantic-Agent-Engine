#!/usr/bin/env python3
"""Render the readable research report from saved benchmark artifacts."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
ART, DATA = ROOT / "artifacts", ROOT / "data"
r = json.loads((ART / "french_results.json").read_text(encoding="utf-8"))
corpus = json.loads((DATA / "french_synthetic_meetings.json").read_text(encoding="utf-8"))
rankings = json.loads((ART / "french_feature_aware_test_rankings.json").read_text(encoding="utf-8"))
a = r["dataset"]
ret = r["retrieval_metrics"]["test"]
fa = r["feature_aware_retrieval"]["test"]
style_by_mid = {m["meeting_id"]: m["style"] for split in corpus.values() for m in split}
lines = [
    "# Rapport expérimental — mémoire sémantique de réunions en français", "",
    "> Corpus 100 % synthétique. Aucun corpus de réunions réel n’est utilisé. Tous les scores ci-dessous sont calculés sur les données synthétiques de ce dépôt.", "",
    "## 1. Protocole et données", "",
    "- 15 projets fictifs, 4 réunions par projet : 2 formelles + 2 messy/transcrites.",
    f"- {a['total_meetings']} réunions : {a['formal_meetings']} formelles et {a['messy_meetings']} messy ; {a['min_messages_per_meeting']} interventions minimum par réunion ; {a['min_speakers_per_meeting']} intervenants distincts minimum ; {a['messages']} messages au total.",
    "- Répartition par projet, sans chevauchement : 9 projets train (36 réunions), 3 validation (12), 3 test (12).",
    "- 1 260 questions synthétiques avec une cible `chunk_id` exacte : 756 train, 252 validation, 252 test.",
    "- Chaque message source a exactement `{username, message_id, timestamp, message}`. Le `chunk_id` est une clé externe et ne modifie pas le JSON source.",
    "- Les réunions messy combinent formulations orales, hésitations, reprises, répétitions, ponctuation absente et artefacts de transcription simulés.", "",
    "### Indicateurs de transcription synthétique", "",
    "| Signal | Messages concernés (sur 1 800 messy) |", "|---|---:|",
]
ms = a["messy_transcription_stats"]
for key, label in [
    ("with_filler_or_discourse_marker", "Hésitation / marqueur discursif"),
    ("with_repeated_word", "Répétition"),
    ("without_terminal_punctuation", "Sans ponctuation terminale"),
    ("with_light_asr_artifact", "Artefact ASR léger"),
    ("with_restart_or_repair", "Reprise / réparation"),
]:
    lines.append(f"| {label} | {ms[key]} |")
lines += [
    "", "Les projets entiers sont séparés entre train, validation et test afin qu’une réunion d’un même projet ne se retrouve pas dans plusieurs splits. Fuite de projet détectée : **0**.", "",
    "## 2. Architecture implémentée", "",
    "1. Fin de réunion → conservation de chaque intervention sous forme de JSON source.",
    "2. Extraction de caractéristiques → têtes de thèmes et d’actes, agrégation de récurrence/importance, co-activation entre caractéristiques et clusters latents par réunion.",
    "3. Embeddings → encodeur Transformer maison à une couche, projection normalisée de 64 dimensions ; vecteur global de réunion et résumé textuel dérivés des représentations prédites.",
    "4. Mémoire → SQLite : JSON immuable, vecteurs, probabilités de caractéristiques, résumé, nœuds et arêtes stockés séparément.",
    "5. Retrieval → recherche BM25, dense, feature-aware et chronologique ; les résultats rendent les JSON source d’origine.", "",
    f"Le modèle maison compte **{r['model']['parameters']:,} paramètres**, un vocabulaire de {r['model']['vocabulary']} tokens et une dimension d’embedding de {r['model']['embedding_dim']}. Il est entraîné par InfoNCE avec cinq hard negatives par requête et une supervision auxiliaire des thèmes et actes. Checkpoint sélectionné sur MRR@10 validation.", "",
    "## 3. Retrieval — test tenu à l’écart", "",
    "| Méthode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | nDCG@5 |", "|---|---:|---:|---:|---:|---:|---:|",
]
for name, label in [("bm25", "BM25"), ("dense", "Encodeur dense maison"), ("hybrid", "Fusion BM25 + dense")]:
    x = ret[name]
    lines.append(f"| {label} | {x['Recall@1']:.3f} | {x['Recall@3']:.3f} | {x['Recall@5']:.3f} | {x['Recall@10']:.3f} | {x['MRR@10']:.3f} | {x['nDCG@5']:.3f} |")
x = fa["overall"]
lines.append(f"| **Feature-aware** | **{x['Recall@1']:.3f}** | **{x['Recall@3']:.3f}** | **{x['Recall@5']:.3f}** | **{x['Recall@10']:.3f}** | **{x['MRR@10']:.3f}** | **{x['nDCG@5']:.3f}** |")
lines += [
    "", "La fusion BM25+dense choisie sur validation a retenu `alpha=1.0` : ajouter BM25 n’améliore pas le rang dense sur ce benchmark. Le rang feature-aware est une étape distincte : il combine les activations de thème/acte prédites, les rangs de candidats et l’ordre temporel explicite lorsque la question demande la dernière information.", "",
    "### Résultats feature-aware par style", "", "| Style | Recall@1 | Recall@5 | MRR@10 |", "|---|---:|---:|---:|",
]
for style in ("formelle", "messy"):
    subset = [z for z in rankings if style_by_mid[z["meeting_id"]] == style]
    ranks = []
    for z in subset:
        ids = [y["chunk_id"] for y in z["rankings"]]
        ranks.append(ids.index(z["target_chunk_id"]) + 1 if z["target_chunk_id"] in ids else 999)
    r1 = np.mean([q <= 1 for q in ranks]); r5 = np.mean([q <= 5 for q in ranks]); mrr = np.mean([1 / q if q <= 10 else 0 for q in ranks])
    lines.append(f"| {style} | {r1:.3f} | {r5:.3f} | {mrr:.3f} |")
lines += [
    "", "### Routage de requêtes", "",
    f"- Exactitude d’inférence de l’opération : **{fa['operation_inference_accuracy']:.3f}** sur {fa['operation_inference_count']} requêtes test synthétiques.",
    "- Le mode est inféré à partir du texte de la question et des têtes du modèle. Les labels `query_type` servent à ajuster les poids sur validation, pas à router les exemples test.",
    "- Cette exactitude parfaite est liée aux formulations générées et contrôlées ; elle n’est pas une estimation de la compréhension du français en production.", "",
    "### Exemple de preuve retournée", "",
    "Question : « Quelle décision a finalement été retenue concernant la base de données ? »", "",
    "Le premier résultat de `fr2-projet-01-reunion-1` est `fr2-projet-01-reunion-1:059`. Le JSON source est retourné sans génération :", "",
    "```json", "{", '  "username": "karim",', '  "message_id": 59,', '  "timestamp": "09:29:05",', '  "message": "Après discussion, la décision finalement retenue est la suivante : conserver SQLite jusqu\'à 10 000 utilisateurs actifs, puis réévaluer PostgreSQL."', "}", "```", "",
    "## 4. Extraction de caractéristiques et relations", "",
    f"- Thèmes : Macro-F1 **{r['feature_head_metrics']['topic_macro_f1']:.3f}**, accuracy **{r['feature_head_metrics']['topic_accuracy']:.3f}**.",
    f"- Actes discursifs : Macro-F1 **{r['feature_head_metrics']['act_macro_f1']:.3f}**, accuracy **{r['feature_head_metrics']['act_accuracy']:.3f}**.",
    f"- Relations : Macro-F1 **{r['relation_metrics']['macro_f1']:.3f}**, accuracy **{r['relation_metrics']['accuracy']:.3f}** sur des paires synthétiques annotées.",
    "- Relations prises en charge : `supports`, `contradicts`, `supersedes`, `creates_action`, `answers`, `unrelated`.",
    "- Les caractéristiques sont agrégées par réunion (fréquence, importance, co-activation) ; MiniBatchKMeans découvre des groupes latents complémentaires.",
    f"- Classification d’intention : accuracy **{r['intent_metrics']['accuracy']:.3f}**, Macro-F1 **{r['intent_metrics']['macro_f1']:.3f}** sur 35 questions synthétiques réservées au test d’intention.", "",
    "## 5. Mémoire et intégrité", "", "| Table | Lignes |", "|---|---:|",
]
for key, value in r["memory_integrity"]["counts"].items():
    lines.append(f"| `{key}` | {value} |")
lines += [
    "", f"- Hash SHA-256 divergents : **{r['memory_integrity']['hash_mismatches']}**.",
    f"- Vecteurs manquants : **{r['memory_integrity']['missing_vectors']}** ; vecteurs orphelins : **{r['memory_integrity']['orphan_vectors']}**.",
    f"- Triggers d’immutabilité actifs : **{r['memory_integrity'].get('immutable_source_guards', False)}** (`UPDATE` et `DELETE` interdits sur la table des messages).",
    "- `smoke_test_french.py` vérifie les sources, les hashes, les clés étrangères, les triggers et reproduit les métriques depuis les rangs sauvegardés.", "",
    "## 6. Comparaison aux encodeurs pré-entraînés", "",
    "**Comparaison non exécutée dans cet environnement.** `sentence-transformers`/`transformers` ne sont pas installés et le runtime ne résout pas `huggingface.co` (DNS). Aucun score n’est inventé.",
    "- `antoinelouis/french-me5-small` : le benchmark essaie le texte brut et une variante préfixée, puis choisit sur validation. Sa model card montre l’usage SentenceTransformers sur des phrases brutes.",
    "- `intfloat/multilingual-e5-small` : le benchmark utilise `query: ` et `passage: `, préfixes recommandés par sa model card pour la recherche asymétrique.",
    "- Si le MRR validation est plus de 0,05 sous le modèle maison, le script fine-tune sur train seulement (InfoNCE, un positif + cinq hard negatives) et ne conserve le fine-tuning que s’il améliore la validation.",
    "- Le test n’est évalué qu’après gel du mode préfixe et du checkpoint.", "",
    "```bash", "pip install -r requirements.txt", "python run_pretrained_comparison.py --batch-size 64", "```", "",
    "## 7. Couverture de la piscine", "",
    "| Exigence | État |", "|---|---|",
    "| BM25, embeddings denses, retrieval hybride | Implémenté et mesuré |",
    "| Apprentissage contrastif + hard negatives | Implémenté |",
    "| Proposition / décision / objection / engagement | Actes discursifs + hard negatives synthétiques |",
    "| Mémoire avec provenance, locuteur, date, statut | Implémenté, relié au chunk source |",
    "| Relations supports/contradicts/supersedes/action/answers/unrelated | Classifieur synthétique entraîné et testé |",
    "| Intentions answer/compare/dossier/email/tâche/agenda/clarify | Classifieur léger + règles, test synthétique |",
    "| Réponse sourcée, comparaison, dossier et propositions | Démonstration déterministe, sources conservées |",
    "| Pas d’action externe non confirmée | Respecté : aucun e-mail/tâche/événement réel créé |",
    "| Comparaison encodeurs pré-entraînés demandés | Script prêt, exécution bloquée par les dépendances/réseau de cet environnement |", "",
    "## 8. Débogage et corrections", "",
    "1. BM25 seul est faible face aux paraphrases et à la transcription messy ; dense apprend une meilleure correspondance requête-message.",
    "2. Le modèle dense pouvait remonter une objection/risque du même thème au lieu de la décision ; les têtes topic/act étaient calculées mais ignorées au ranking. Le rang feature-aware les utilise maintenant.",
    "3. Les requêtes temporelles utilisent les horodatages structurés plutôt que d’attendre de la similarité cosinus qu’elle apprenne l’ordre chronologique.",
    "4. Le tuning validation choisit `alpha=1.0` pour la fusion BM25/dense : la fusion lexicale n’ajoutait pas de valeur dans ce test.",
    "5. La première tentative de mise à jour DB avec de nouveaux messages sous les mêmes IDs a été rejetée. Les nouvelles données sont versionnées (`fr2-`) et la mémoire a un chemin v2 dédié ; les triggers empêchent aussi les modifications SQL directes.", "",
    "## 9. Limites", "",
    "- Les réunions, questions et annotations sont entièrement synthétiques et contrôlées. Les métriques ne mesurent pas la robustesse sur des réunions réelles.",
    "- Le feature-aware retrieval inclut une étape temporelle déterministe pour les requêtes du type « dernier/dernière » ; ce n’est pas uniquement une réussite d’embeddings.",
    "- Le benchmark cible une seule intervention par question ; le raisonnement multi-preuves et la complétude de dossiers ne sont pas mesurés par ces Recall@K.",
    "- Les thèmes/actes sont issus d’une taxonomie synthétique supervisée ; les clusters et co-activations explorent des caractéristiques latentes mais ne constituent pas encore une découverte ouverte sans labels.",
    "- L’agent est déterministe, sans LLM de rédaction et sans connecteur réel. L’ASR est simulé par bruit textuel, pas implémenté comme système de reconnaissance vocale.",
]
(ART / "FRENCH_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {ART / 'FRENCH_REPORT.md'} ({len(lines)} lines)")
