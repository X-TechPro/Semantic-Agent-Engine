"""Retrieval pipeline — the 2.2 `retrieve(q, filters) -> list[Hit]` contract.

Order matters and is the thesis under test (6.7 → 6.10):

    embed(query, role=query)
      → dense top-50 (FAISS)      ┐
      → bm25 top-50 (sentences)   ┴→ RRF(k=60) → [rerank] → validity → graph expansion

`linker_enabled=False` collapses this to **Baseline A** (plain hybrid RAG): no
validity weighting, no supersede-following, no status penalty. That single flag
is what makes the §10 comparison table honest — the same code path, one switch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from engine.config import load_config
from engine.schemas import Filters, Hit, Query, Unit
from engine.retrieve.bm25 import Bm25Retriever, build_lexical
from engine.retrieve.dense import DenseRetriever
from engine.retrieve.fusion import FusedHit, rrf
from engine.retrieve.graph_expand import expand
from engine.retrieve.rerank import build_reranker
from engine.retrieve.validity import apply_validity, filters_for_query

OWNER_Q = re.compile(
    r"\b(?:qui\s+(?:s['’]?occupe|est responsable|gère|gere)|"
    r"qui s['’]?occupe|responsable de|à qui|a qui)\b", re.IGNORECASE)


@dataclass
class RetrievalStats:
    dense_hits: int = 0
    lexical_hits: int = 0
    fused: int = 0
    after_validity: int = 0
    after_expansion: int = 0
    timings: dict[str, float] = field(default_factory=dict)


class RetrievalPipeline:
    def __init__(self, store, embedder, cfg=None, *, linker_enabled: bool = True,
                 scoping: bool | None = None, reranker: Any = None):
        self.cfg = cfg or load_config()
        self.store = store
        self.embedder = embedder
        self.linker_enabled = linker_enabled
        # 10 separates "A + metadata filters, no linker" (row 6) from plain
        # hybrid RAG (row 4) and from System B (row 7). Scoping therefore has to
        # be switchable independently of the linker, or the headline cannot be
        # decomposed into "filters" vs "typed relations". Defaults to following
        # the linker so row 4 stays filter-free as the table defines it.
        self.scoping = linker_enabled if scoping is None else scoping
        rc = self.cfg["retrieve"]
        self.top_k_dense = int(rc["top_k_dense"])
        self.top_k_lex = int(rc["top_k_lexical"])
        self.rrf_k = int(rc["rrf_k"])
        self.weights = dict(rc["weights"])
        self.final_k = int(rc["final_k"])
        self.valid = dict(rc["validity"])
        self.dense = DenseRetriever(store, embedder, top_k=self.top_k_dense,
                                   use=self.cfg["store"]["faiss"]["index_key"])
        self.lexical = build_lexical(store, self.cfg)
        self.reranker = reranker if reranker is not None else build_reranker(self.cfg)
        # which reranker this pipeline really built, not which one the YAML asked
        # for: an arm must never be able to claim `cross_encoder` after the
        # builder returned None. Read back into `meta.rerank_backend` by the bench.
        self.rerank_backend = (self.reranker.name if self.reranker is not None
                               else "none")

    # --- public ------------------------------------------------------------
    def retrieve(self, query: Query | str, filters: Filters | None = None, *,
                 k: int | None = None, linker_enabled: bool | None = None,
                 scoping: bool | None = None) -> list[Hit]:
        import time

        stats = RetrievalStats()
        q = Query(text=query) if isinstance(query, str) else query
        use_linker = self.linker_enabled if linker_enabled is None else linker_enabled
        use_scoping = self.scoping if scoping is None else scoping
        # row 4 (plain RAG) must not scope; rows 6 and 7 must
        eff_filters = filters_for_query(q) if (q.project and use_scoping) else None

        t0 = time.perf_counter()
        dense = self.dense.search(q.text, filters=eff_filters, k=self.top_k_dense)
        stats.dense_hits = len(dense)
        t1 = time.perf_counter()
        lex = self.lexical.search(q.text, k=self.top_k_lex,
                                 allow_units=None if eff_filters is None else
                                 set(self.store.unit_ids_for(eff_filters)))
        stats.lexical_hits = len(lex)
        t2 = time.perf_counter()
        stats.timings["dense_s"] = t1 - t0
        stats.timings["lexical_s"] = t2 - t1

        evidence = {r.unit_id: r.evidence for r in lex}
        rankings = {"dense": [r.unit_id for r in dense], "bm25": [r.unit_id for r in lex]}
        fused = rrf(rankings, k=self.rrf_k, weights=self.weights)
        stats.fused = len(fused)
        fused = fused[: self.final_k]

        t3 = time.perf_counter()
        if self.reranker is not None:
            parts_by_id = {f.unit_id: f.parts for f in fused}
            by_id = {u.unit_id: u for u in self.store.all_units()}
            units_r = [by_id[f.unit_id] for f in fused if f.unit_id in by_id]
            ranked = self.reranker.rerank(q.text, units_r)  # [(unit_id, score)]
            fused = [FusedHit(uid, s, {**parts_by_id.get(uid, {}), "rerank": s})
                     for uid, s in ranked]
            # only when it actually ran: recording a 0.0 ms `rerank_s` for an
            # absent reranker makes a latency table claim a stage was measured
            # that never executed.
            stats.timings["rerank_s"] = time.perf_counter() - t3

        units = [(self.store.get(f.unit_id), f) for f in fused]
        units = [(u, f) for u, f in units if u is not None]
        scored = [(u, f.score) for u, f in units]
        parts_by_id = {f.unit_id: f.parts for _, f in units}

        # An owner question is not a value question. The cross-encoder ranks
        # "risk: under control" above "risk: Marie" because both are about the
        # risk; the person slot is the speech act the question asks for.
        owner_boost = float(self.cfg["retrieve"].get("owner_boost", 0.0))
        if owner_boost and OWNER_Q.search(q.text or ""):
            boosted_owner = []
            for u, s in scored:
                has_person = bool(u.slots.persons or u.slots.owner)
                if has_person:
                    s += owner_boost
                    parts_by_id.setdefault(u.unit_id, {})["owner"] = owner_boost
                boosted_owner.append((u, s))
            scored = boosted_owner

        # 6.7: an explicit slot is a *soft boost*, never a hard filter — the
        # extractor leaves subject unset on many units, so filtering on it
        # discards gold evidence (measured: gold_in_pool 0.978 -> 0.756).
        if use_scoping and q.slots.subject:
            boost = float(self.cfg["retrieve"].get("slot_boost", 1.25))
            boosted: list[tuple[Unit, float]] = []
            for u, s in scored:
                if u.slots.subject == q.slots.subject:
                    s *= boost
                    parts_by_id.setdefault(u.unit_id, {})["slot"] = boost
                elif u.slots.subject:
                    parts_by_id.setdefault(u.unit_id, {})["slot"] = 1.0
                boosted.append((u, s))
            scored = boosted

        t4 = time.perf_counter()
        if use_linker:
            valid = apply_validity(
                scored, q,
                confidence_of=self.store.supersede_confidence,
                alpha_superseded=float(self.valid["alpha_superseded"]),
                alpha_history=float(self.valid["alpha_history"]),
                beta_soft=float(self.valid["beta_soft_status"]),
                beta_mention=float(self.valid.get("beta_mention", 1.0)),
                hard_filter=bool(self.valid["hard_filter"]))
        else:
            valid = [(u, s, 1.0, s) for u, s in scored]
        stats.timings["validity_s"] = time.perf_counter() - t4

        valid.sort(key=lambda t: (-t[3], t[0].unit_id))
        stats.after_validity = len(valid)

        hits: list[Hit] = []
        for u, base, v, final in valid[: k or self.final_k]:
            sp = dict(parts_by_id.get(u.unit_id, {}))
            sp.update({"fused": base, "validity": v, "final": final})
            sp.setdefault("dense", 0.0)
            sp.setdefault("bm25", 0.0)
            if any(u.unit_id == d.unit_id for d in dense):
                sp["dense"] = float(next(d.score for d in dense if d.unit_id == u.unit_id))
            if any(u.unit_id == l.unit_id for l in lex):
                sp["bm25"] = float(next(l.score for l in lex if u.unit_id == l.unit_id))
            hits.append(Hit(unit=u, score=final, score_parts=sp,
                            evidence=evidence.get(u.unit_id) or self._evidence(u, q.text),
                            path=[]))
        if use_linker and self.valid["graph_expansion"]:
            hits = expand(self.store, hits)
        stats.after_expansion = len(hits)
        for i, h in enumerate(hits, start=1):
            hits[i - 1] = h.model_copy(update={"rank": i})
        self.last_stats = stats
        return hits

    # --- helpers -----------------------------------------------------------
    def _evidence(self, unit: Unit, query: str) -> str:
        """Best evidence sentence inside a unit (6.6: BM25 runs over sentences)."""
        rows = self.store.db.execute(
            "SELECT text FROM sentences WHERE unit_id=? ORDER BY start",
            (unit.unit_id,)).fetchall()
        sents = [r["text"] for r in rows] or [unit.text]
        from engine.store.fts import tokenize

        qt = set(tokenize(query))
        if not qt:
            return sents[0]
        return max(sents, key=lambda s: len(qt & set(tokenize(s))))

    def rebuild(self) -> None:
        self.store.rebuild_index()
        self.lexical = build_lexical(self.store, self.cfg)


def retrieve(q: Query, store, embedder, cfg=None, **kw) -> list[Hit]:
    return RetrievalPipeline(store, embedder, cfg, **kw).retrieve(q)
