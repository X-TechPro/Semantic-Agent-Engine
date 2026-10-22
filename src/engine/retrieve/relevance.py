"""Content relevance — the abstention signal (6.11 "no evidence above threshold").

Why not the retrieval score: RRF returns `sum w/(k+rank)`, so on two lists with
k=60 the *best possible* score is ~0.0328 and the worst non-zero is ~0.0161.
Rank fusion carries no relevance information — a top hit always scores ~0.0328
even when the corpus contains nothing about the question. A no-answer rule built
on that number can only ever be wrong in one direction (we measured it: with a
0.05 threshold the engine abstained on 100% of queries, including answerable
ones).

So abstention here is decided on **shared content words** between the question
and the evidence, which is embedder-independent and deterministic, and is the
same signal the benchmark uses for its unanswerable-abstention metric
(eval/temporal.py) — one instrument, two consumers.
"""

from __future__ import annotations

from engine.store.fts import tokenize

#: words that carry no topical signal in French meeting speech
STOP = {
    "quel", "quelle", "quels", "quelles", "quand", "quoi", "comment", "pourquoi",
    "qui", "que", "est", "sont", "être", "etre", "dans", "sur", "pour", "par",
    "avec", "sans", "du", "des", "le", "la", "les", "un", "une", "de", "au",
    "aux", "ce", "cet", "cette", "y", "a", "ont", "son", "sa", "ses", "nos",
    "vos", "notre", "votre", "leur", "leurs", "on", "nous", "vous", "il",
    "elle", "ils", "elles", "tu", "me", "te", "se", "lui", "c'est", "celui",
    "celle", "projet", "projets", "dossier", "decision", "décision", "decisions",
    "valeurs", "valeur", "aujourd", "hui", "finalement", "exactement",
    "historiquement", "question", "quelconque", "quelle", "quels",
    "quels", "bonjour", "allouons", "allo", "qu", "combien", "quand",
    "change", "changé", "changée", "changement", "nouveau", "nouvelle",
    "version", "validée", "validé", "derniere", "dernière", "actuelle",
    "actuel", "responsable", "personne", "s'occupe", "occupe", "mail",
    "rédige", "redige", "demander", "demande", "avant", "apres", "après",
    "vendredi", "lundi", "mardi", "mercredi", "jeudi", "dimanche", "samedi",
    "est", "où", "ou", "donc", "alors", "aussi", "plus", "moins", "tres",
    "très", "peut", "être", "fait", "faire", "chose", "choses", "point",
    "points", "suivant", "suivante", "dit", "dire", "disent", "finalement",
}


def content_words(text: str) -> set[str]:
    """Lowercase tokens with the French function-words and question-scaffolding
    removed — what is left is what the sentence is *about*."""
    out: set[str] = set()
    for t in tokenize(text):
        if len(t) < 3 or t in STOP:
            continue
        out.add(t)
    return out


def overlap(query: str, evidence: str) -> float:
    """Jaccard over content words, 0.0 when either side is empty."""
    q, e = content_words(query), content_words(evidence)
    if not q or not e:
        return 0.0
    return len(q & e) / len(q | e)


def shares_topic(query: str, evidence: str, *, min_shared: int = 1,
                 floor: float = 0.06) -> bool:
    """Is this evidence about the same thing as the question?"""
    q, e = content_words(query), content_words(evidence)
    shared = q & e
    if len(shared) < min_shared:
        return False
    return overlap(query, evidence) >= floor
