"""BM25 lexical retrieval over sentences (6.6), with an FTS5 sibling.

Primary backend `bm25s` (what 6.6 wants ablated); `fts5` is kept as a second row
in the comparison and as the no-extra-dependency path. Both return the matching
**sentence** as `Hit.evidence`, aggregated to unit level by max score.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.store.fts import tokenize


@dataclass(slots=True)
class LexResult:
    unit_id: str
    score: float
    evidence: str


class Bm25Retriever:
    backend = "bm25s"

    def __init__(self, strip_accents: bool = True, stem: bool = False):
        self.strip_accents = strip_accents
        self.stem = stem
        self._index = None
        self._sent_meta: list[tuple[str, str]] = []  # (unit_id, sentence text)
        self._built_from: tuple[int, int] = (-1, -1)

    def _tokens(self, text: str) -> list[str]:
        return tokenize(text, strip_accents=self.strip_accents, stem=self.stem)

    def build(self, store) -> int:
        """(Re)build from the store's `sentences` table — deterministic."""
        import bm25s

        rows = store.db.execute(
            "SELECT unit_id, text FROM sentences ORDER BY sent_id").fetchall()
        self._sent_meta = [(r["unit_id"], r["text"]) for r in rows]
        if not self._sent_meta:
            self._index = None
            return 0
        corpus_tokens = [self._tokens(t) for _, t in self._sent_meta]
        self._index = bm25s.BM25(corpus=[t for _, t in self._sent_meta])
        self._index.index(corpus_tokens, show_progress=False)
        return len(self._sent_meta)

    def search(self, query: str, k: int = 50,
               allow_units: set[str] | None = None) -> list[LexResult]:
        if self._index is None or not query.strip():
            return []
        import bm25s

        qtok = [self._tokens(query)]
        top = min(len(self._sent_meta), max(k * 3, k))
        res = self._index.retrieve(qtok, k=top)
        docs = res.documents[0]
        scores = res.scores[0]
        out: list[LexResult] = []
        best: dict[str, LexResult] = {}
        for text, score in zip(docs, scores):
            uid = next((u for u, t in self._sent_meta if t == text), None)
            if uid is None or (allow_units is not None and uid not in allow_units):
                continue
            s = float(score)
            if uid not in best or s > best[uid].score:
                best[uid] = LexResult(uid, s, text)
        out = sorted(best.values(), key=lambda r: -r.score)[:k]
        return out


class FtsRetriever:
    backend = "fts5"

    def __init__(self, store):
        self.store = store

    def build(self, store=None) -> int:
        return self.store.fts.sync_from_sentences()

    def search(self, query: str, k: int = 50,
               allow_units: set[str] | None = None) -> list[LexResult]:
        hits = self.store.fts.search(query, k=max(k * 3, k))
        best: dict[str, LexResult] = {}
        for h in hits:
            if allow_units is not None and h.unit_id not in allow_units:
                continue
            if h.unit_id not in best or h.score > best[h.unit_id].score:
                best[h.unit_id] = LexResult(h.unit_id, h.score, h.evidence)
        return sorted(best.values(), key=lambda r: -r.score)[:k]


def build_lexical(store, cfg) -> Bm25Retriever | FtsRetriever:
    lex = cfg["lexical"]
    if lex["backend"] == "fts5":
        r = FtsRetriever(store)
        r.build(store)
        return r
    r = Bm25Retriever(strip_accents=lex["strip_accents"], stem=lex["stem"])
    r.build(store)
    return r
