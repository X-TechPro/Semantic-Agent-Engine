"""Dense retrieval over FAISS (6.4 + 6.7)."""

from __future__ import annotations

from dataclasses import dataclass

from engine.schemas import Filters


@dataclass(slots=True)
class DenseResult:
    unit_id: str
    score: float


class DenseRetriever:
    #: the only columns `vectors` has; a config typo must not be readable as "the
    #: dense arm found nothing"
    COLUMNS = ("base", "refined")

    def __init__(self, store, embedder, *, top_k: int = 50, use: str = "base"):
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        if use not in self.COLUMNS:
            # `faiss.indexes.get("refinedd")` returns None and `search` answers [],
            # so an unknown column would look like "no dense matches for this
            # corpus" rather than like a misconfigured store, and RRF would silently
            # fuse one ranking instead of two.
            raise ValueError(f"unknown vector column {use!r}; expected one of "
                             f"{list(self.COLUMNS)}")
        if use == "refined":
            # `store.vector_matrix("refined")` falls back to the base vector for any
            # row whose refined column is NULL, so asking for the refined space on a
            # store that never ran the refine job would quietly measure the frozen
            # baseline while the artefact still said "refined". Fail at construction
            # instead: this is the same "report what actually ran" rule as D-011.
            n = store.db.execute(
                "SELECT COUNT(*) c FROM vectors WHERE refined IS NOT NULL"
            ).fetchone()["c"]
            if not n:
                raise ValueError(
                    "retrieve asked for `refined` vectors but this store has none: "
                    "run the concept refine job (offline.run_offline with "
                    "concept.enabled=true, or `sae concept-refine`) or set "
                    "store.faiss.index_key: base")
        self.use = use

    def search(self, query: str, filters: Filters | None = None,
               k: int | None = None) -> list[DenseResult]:
        vec = self.embedder.embed([query], role="query")[0]
        pairs = self.store.neighbors(vec, k or self.top_k, filters=filters,
                                    use=self.use)
        return [DenseResult(uid, float(s)) for uid, s in pairs]
