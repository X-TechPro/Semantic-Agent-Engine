"""Composition root — wires the stages behind the 2.2 interfaces.

Not a stage itself: it touches each stage only through its public entry point,
so the lego rule (0.3) still holds. CLI, API, UI and tests all go through this
object, so backends are selected in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

from engine.config import load_config
from engine.embed.base import Embedder, build_embedder
from engine.ingest import IngestPipeline
from engine.read.infer import build_reader
from engine.retrieve.pipeline import RetrievalPipeline
from engine.schemas import (Filters, Hit, Query, Reading, Slots, Source, Unit,
                            make_unit_id, utcnow)
from engine.segment.chunklet_adapter import Segmenter
from engine.slots import rules as slot_rules
from engine.store.sqlite_store import SQLiteStore


@dataclass(slots=True)
class ReadUnit:
    """A bare string read as a unit: the unit + the reader's verdict."""

    unit: Unit
    reading: Reading

    @property
    def kind(self) -> str:
        return self.reading.kind

    @property
    def intent(self) -> str:
        return self.reading.intent

    @property
    def status(self) -> str:
        return self.reading.status


class Engine:
    def __init__(self, cfg=None, *, store: SQLiteStore | None = None,
                 embedder: Embedder | None = None,
                 reranker=None,
                 roster: Iterable[str] | None = None,
                 linker_enabled: bool | None = None,
                 scoping: bool | None = None,
                 path: str | Path | None = None):
        self.cfg = cfg or load_config()
        #: an injected reranker is shared across arms (the benchmark's A/B run has
        #: two of them). Without this each arm built its own 568M-parameter
        #: cross-encoder and the test-split ablation OOMed on an 8 GB card.
        self._reranker = reranker
        # 0.3: the config selects the system (Baseline A vs System B). An
        # explicit argument still wins — that is how the A/B benchmark builds
        # both rows from one config in one process.
        if linker_enabled is None:
            linker_enabled = bool(self.cfg.get("linker_enabled", True))
        self.store = store or SQLiteStore(path or self.cfg["store"]["path"],
                                          embed_version="pending")
        self.embedder = embedder or build_embedder(self.cfg)
        self.store.embed_version = self.embedder.version
        self.segmenter = Segmenter(self.cfg)
        # 6.2: rules vs trained, selected by config in ONE place. The ingest
        # pipeline gets this same object: a second `build_reader` would load a
        # second copy of the encoder into VRAM for no behavioural difference.
        self.reader, self.reader_backend = build_reader(self.cfg)
        self.linker_enabled = linker_enabled
        # 10, row 6 is "metadata filters, no linker": scoping must be settable
        # apart from the linker, or the headline cannot be decomposed. Default
        # follows the linker so Baseline A stays the filter-free row 4.
        self.scoping = linker_enabled if scoping is None else scoping
        self.ingest_pipeline = IngestPipeline(self.store, self.embedder, self.cfg,
                                             roster=roster,
                                             linker_enabled=linker_enabled,
                                             reader=self.reader)
        self._pipeline: RetrievalPipeline | None = None
        self._decision_view = None
        self.roster: set[str] = set(roster or ())
        self.projects: set[str] = set()
        self.reports: list = []
        self.pending_queries: list[Unit] = []

    # --- stages ------------------------------------------------------------
    def segment(self, source: Source) -> list[Unit]:
        return self.segmenter.segment(source)

    def read(self, unit: Unit) -> Reading:
        return self.reader.read(unit)

    def read_text(self, text: str, *, source_id: str = "adhoc",
                  source_date: date | None = None,
                  project: str | None = None) -> ReadUnit:
        """Read a bare utterance as a unit (router, CLI and tests use this)."""
        d = source_date or date.today()
        unit = Unit(unit_id=make_unit_id(source_id, 0, 0, len(text), None),
                    source_id=source_id, source_type="note", source_date=d,
                    ingested_at=utcnow(), text=text, sentences=[], project=project,
                    slots=Slots())
        reading = self.reader.read(unit)
        decorated = unit.model_copy(update={"kind": reading.kind,
                                            "type": reading.type,
                                            "status": reading.status})
        decorated = decorated.model_copy(
            update={"slots": slot_rules.extract(
                decorated, self.roster,
                subject_model=getattr(self.ingest_pipeline,
                                      "subject_model", None))})
        # the ad-hoc read path and the ingest path must see the SAME subject
        # chain, or the router plans retrieval around a slot the store never
        # holds (two call sites, two truths — the pattern behind B-13's null)
        return ReadUnit(unit=decorated, reading=reading)

    def embed(self, texts: list[str], role: str = "doc"):
        return self.embedder.embed(texts, role=role)

    def ingest(self, source: Source):
        report = self.ingest_pipeline.ingest(source)
        self.reports.append(report)
        for u in self.store.units_of_source(source.source_id):
            if u.speaker:
                self.roster.add(u.speaker)
        if source.project:
            self.projects.add(source.project)
        self.pending_queries = list(getattr(self.ingest_pipeline, "last_queries", []))
        self._decision_view = None
        return report

    def retrieve(self, query: Query, *, k: int | None = None,
                 linker_enabled: bool | None = None,
                 scoping: bool | None = None,
                 filters: Filters | None = None) -> list:
        """Retrieve through the pipeline.

        `linker_enabled=False` gives Baseline A (plain hybrid RAG: no validity
        weighting, no supersede-following) from the *same* store and candidate
        budget, which is what makes the §10 rows comparable. It is passed per
        call because the kill test and the ablations need both systems from one
        engine; the pipeline-level default comes from this engine's own setting.
        """
        return self.pipeline.retrieve(query, filters=filters, k=k,
                                      linker_enabled=linker_enabled,
                                      scoping=scoping)

    @property
    def pipeline(self) -> RetrievalPipeline:
        """The retrieval pipeline, built once on first use.

        `retrieve()` used to construct it inline, which left no way to ask an
        engine what it actually built — so the reranker's identity and per-stage
        timings were unreadable without running a query first, and a second copy
        of the construction call was one refactor away.
        """
        if self._pipeline is None:
            self._pipeline = RetrievalPipeline(self.store, self.embedder, self.cfg,
                                               linker_enabled=self.linker_enabled,
                                               scoping=self.scoping,
                                               reranker=self._reranker)
        return self._pipeline

    @property
    def rerank_backend(self) -> str:
        """The reranker this engine actually built — read back, not assumed.

        Asks the pipeline, so it is accurate before any query too: an artefact
        must not be able to claim `cross_encoder` when the config resolved to
        `none` (or vice versa) because nobody looked at the constructed object.
        """
        return self.pipeline.rerank_backend

    def ask(self, text: str, **kw):
        """One shot: read the utterance → retrieve → act. Returns an ActionResult.

        v3: for answer/history questions the node view runs beside retrieval, so
        the latest child of the matching decision is the lead and `as_of` can
        read the state on a past date. The meeting itself is never rewritten.
        """
        from engine.agent.router import Agent

        return Agent(self).handle(text, **kw)

    @property
    def decision_view(self):
        """Decision nodes over the whole store, built once (v3)."""
        from engine.threads.view import NodeView

        if self._decision_view is None:
            self._decision_view = NodeView(
                self.store.all_units(),
                state=bool(self.cfg["retrieve"].get("node_state", False)))
        return self._decision_view

    def node_hits(self, query: Query, *, as_of: date | None = None,
                  k: int = 2) -> list[Hit]:
        """The current child of the best matching node(s), as retrieval hits.

        `as_of` selects the child that was current on that date; without it the
        node's latest child is the current decision. Children are returned in
        `path` so an answer can show the whole thread without mutating it.
        """
        out: list[Hit] = []
        for m in self.decision_view.find(query.text, project=query.project,
                                        as_of=as_of, k=k):
            u = m.current
            out.append(Hit(unit=u, score=float(m.score),
                           score_parts={"node": float(m.score)},
                           evidence=u.text,
                           path=[c.unit_id for c in m.node.children]))
        return out

    def intent_of(self, text: str) -> str:
        return self.read_text(text).intent or "answer"

    # --- housekeeping ------------------------------------------------------
    def stats(self) -> dict:
        s = self.store.count()
        # Union with the store's own values rather than reporting the session sets
        # alone: `self.projects`/`self.roster` fill during ingest, so a freshly
        # opened engine over an existing corpus reported `projects: 0, roster: 0`
        # while the database held 3 and 6. The union keeps both true -- a roster
        # passed in at construction may name people who have not spoken yet, which is
        # why the store's values are added rather than substituted.
        facets = self.store.facets()
        return {**s, "embed_model_version": self.embedder.version,
                "roster": len(self.roster | facets["speakers"]),
                "projects": len(self.projects | facets["projects"]),
                "linker_enabled": self.linker_enabled,
                "linker_backend": getattr(self.ingest_pipeline, "linker_backend",
                                          "unknown"),
                # which reader actually ran, not which one the YAML asked for
                "reader_backend": getattr(self.ingest_pipeline, "reader_backend",
                                          "unknown"),
                # B-13 made the subject resolver guard-critical: 'supersedes'
                # is undecidable without a subject key, so an artefact must
                # name the resolver that actually ran, like the other two
                "subject_resolver": getattr(self.ingest_pipeline,
                                            "subject_resolver", "unknown"),
                "arch": "v3"}

    def rebuild(self) -> int:
        self._decision_view = None
        n = self.store.rebuild_index()
        if self._pipeline is not None:
            self._pipeline.rebuild()
        return n

    @property
    def vector_column(self) -> str:
        """Which vector column retrieval is reading — read back, not assumed.

        Same rule as `rerank_backend` and `reader_backend`: an artefact may only
        claim the space it actually searched, and §10 row 9 ("B + concept
        attention") is meaningless unless the recorded column says `refined`.
        """
        return str(self.cfg.get_path("store.faiss.index_key", "base"))

    def use_vector_column(self, column: str) -> None:
        """Point retrieval at `base` or `refined` and rebuild what that implies.

        The column is read once, when `RetrievalPipeline` is constructed, so
        writing refined vectors into the store changes nothing about what an
        already-built engine searches. Without this method the refine job would be
        a no-op on any live engine — a file written, an index rebuilt, and the
        same base vectors still served. The pipeline cache is dropped so the next
        `retrieve()` actually reads the column being asked for.
        """
        if column not in ("base", "refined"):
            raise ValueError(f"unknown vector column {column!r}")
        if column == "refined":
            n = self.store.db.execute(
                "SELECT COUNT(*) c FROM vectors WHERE refined IS NOT NULL"
            ).fetchone()["c"]
            if not n:
                raise ValueError(
                    "no refined vectors in this store: run the concept refine job "
                    "before asking retrieval to read that column")
        self.cfg["store"]["faiss"]["index_key"] = column
        # the pipeline holds its own reference to cfg, so it must be rebuilt rather
        # than merely re-pointed: it copied `index_key` into DenseRetriever(use=...)
        # at construction.
        self._pipeline = None

    def close(self) -> None:
        self.store.close()
