"""Ingest — the online half of the pipeline (2.1 / 6.14).

    source → segment → read → slots → [info: embed + store + link]
                                   → [query: answered AFTER the info is stored]

Two behaviours the guide makes explicit and that are enforced here:
  * info units are stored **first**, so a query living in the same message can be
    answered with the fresh fact (2.1 / 6.2.4).
  * a supersede edge writes `valid_to` / `superseded_by` on the old unit and
    never deletes anything (2.2).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from engine.config import load_config, repo_root
from engine.embed.base import Embedder, build_embedder
from engine.link.supervised import build_linker
from engine.link.supersede import evaluate, same_slot
from engine.read.baseline_rules import is_chatter
from engine.read.infer import build_reader
from engine.schemas import (Edge, Filters, IngestReport, Query, Reading, Source,
                            Unit, utcnow)
from engine.segment.chunklet_adapter import Segmenter
from engine.slots import rules as slot_rules
from engine.store.sqlite_store import SQLiteStore

ACTION_INTENTS = frozenset({"create_task", "schedule_event", "draft_email"})


class IngestPipeline:
    def __init__(self, store: SQLiteStore, embedder: Embedder | None = None,
                 cfg=None, *, roster: Iterable[str] | None = None,
                 linker_enabled: bool = True, reader=None):
        self.cfg = cfg or load_config()
        self.store = store
        self.embedder = embedder or build_embedder(self.cfg)
        self.store.embed_version = self.embedder.version
        self.segmenter = Segmenter(self.cfg)
        # 0.3 / 6.2: one construction path selects rules vs trained. `Engine`
        # passes its own reader in — loading a second copy of a trained encoder
        # in the same process would double the VRAM for no behaviour change.
        if reader is not None:
            self.reader = reader
            self.reader_backend = getattr(reader, "name", "rules")
        else:
            self.reader, self.reader_backend = build_reader(self.cfg)
        # linker_backend records which linker this run *actually* used, so a run
        # cannot report "trained" while a missing model fell it back to rules.
        self.linker, self.linker_backend = build_linker(self.cfg)
        # 6.3's subject resolvers: the shipped one is `rules`; `gold` is the
        # hand-labelled annotation supplier the real-data demos use. The five
        # extractor proposals (b lookup, c LLM, d/e span, f classifier, g graph
        # projection) were each built, wired and measured against one pre-
        # registered gate (D-026…D-036), all refused, and the code was removed
        # (D-037) — their verdict artefacts stay in `runs/`, and asking for a
        # removed resolver is still a startup error, never a silent rules run.
        self.subject_resolver = str(
            self.cfg["slots"].get("subject_resolver", "rules"))
        if self.subject_resolver not in ("rules", "gold"):
            raise ValueError(
                f"slots.subject_resolver={self.subject_resolver!r}: only 'rules' "
                "and 'gold' (the demo's annotation supplier) exist — the "
                "extractors were measured and removed (D-037)")
        self.subject_model = None
        if self.subject_resolver == "gold":
            from engine.slots.gold_subject import GoldSubjectModel
            d = self.cfg["slots"].get("subject_model_dir") or "data/gold"
            p = Path(d)
            if not p.is_absolute():
                p = repo_root() / p
            self.subject_model = GoldSubjectModel(
                p, repo_root() / "data/external/summre_test/meetings.json")
        self.roster = set(roster or ())
        self.linker_enabled = linker_enabled
        self.last_queries: list[Unit] = []
        #: how many preceding units the reader may see (6.2.3 ablation)
        self.context_units = int(getattr(reader, "context_units",
                                         self.cfg["reader"].get("context_units", 0)))

    # --- one source --------------------------------------------------------
    def ingest(self, source: Source) -> IngestReport:
        t0 = time.perf_counter()
        # Idempotence guard (D-038, measured 2026-10-22): re-ingesting a source
        # already in the store replaced every unit (INSERT OR REPLACE briefly
        # clearing its supersede stamps) and re-linked it against the NOW-FULL
        # store, minting reverse-direction edge pairs — the demo corpus's edge
        # count went 18 -> 37 on a second pass, with atlas-1 -> atlas-3 edges
        # that only exist because atlas-3 was already there. Same source_id
        # means the same content (unit ids are derived from it), so the honest
        # second pass is a no-op, reported as such.
        already = self.store.units_of_source(source.source_id)
        if already:
            return IngestReport(source_id=source.source_id, n_units=len(already),
                                n_stored=0, n_skipped=0, n_queries=0, n_edges=0,
                                n_superseded=0, relations={},
                                elapsed_s=round(time.perf_counter() - t0, 4))
        units = self.segment(source)
        relations: dict[str, int] = {}
        n_stored = n_queries = n_edges = n_superseded = 0

        info_units: list[Unit] = []
        query_units: list[Unit] = []
        n_skipped = 0
        for i, raw in enumerate(units):
            # 6.2.3's context ablation has to be wired to reach the reader at
            # all: a `context_units` key nothing passes is an inert knob.
            ctx = units[max(0, i - self.context_units):i] if self.context_units \
                else None
            u, reading = self.annotate(raw, ctx)
            if reading.kind == "query":
                query_units.append(u)
                n_queries += 1
            if reading.kind == "info" or reading.also_info:
                if is_chatter(u):
                    # conversational glue asserts no fact; indexing it lets it
                    # outrank the decisions around it (D-007)
                    n_skipped += 1
                    continue
                info_units.append(u)

        # 1. store + link every info unit (fresh facts land before queries run)
        for u in info_units:
            stored = self.store_one(u)
            n_stored += 1
            if self.linker_enabled:
                edges, sup = self._link(stored)
                n_edges += len(edges)
                n_superseded += sup
                for rel, cnt in _tally(edges).items():
                    relations[rel] = relations.get(rel, 0) + cnt
        # 2. queries are exposed to the caller, never stored as facts (2.1)
        self.last_queries = query_units
        return IngestReport(source_id=source.source_id, n_units=len(units),
                            n_stored=n_stored, n_skipped=n_skipped,
                            n_queries=n_queries, n_edges=n_edges,
                            n_superseded=n_superseded, relations=relations,
                            elapsed_s=round(time.perf_counter() - t0, 4))

    def segment(self, source: Source) -> list[Unit]:
        return self.segmenter.segment(source)

    def annotate(self, unit: Unit, ctx: list[Unit] | None = None
                 ) -> tuple[Unit, "Reading"]:
        """Run the reader + slots; return the decorated unit and the verdict.

        `ctx` is the preceding units of the same source, and it reaches the
        reader only when `reader.context_units` asks for them — 6.2.3's ablation
        has to be able to run both ways through this one call site.
        """
        reading = self.reader.read(unit, ctx)
        decorated = unit.model_copy(update={
            "kind": reading.kind, "type": reading.type, "status": reading.status,
            "valid_from": unit.source_date if (reading.kind == "info" or reading.also_info)
            else None})
        # slots see the reader's labels (owner detection needs `type`)
        decorated = decorated.model_copy(
            update={"slots": slot_rules.extract(
                decorated, self.roster, subject_model=self.subject_model)})
        return decorated, reading

    def store_one(self, unit: Unit) -> Unit:
        vec = self.embedder.embed([unit.text], role="doc")[0]
        stamped = unit.model_copy(update={"embed_model_version": self.embedder.version})
        self.store.add_units([stamped])
        self.store.add_vectors([stamped.unit_id], [vec])
        return stamped

    @staticmethod
    def _question_end(edge: Edge, units: dict[str, Unit]) -> str | None:
        """Which endpoint of an `answers` edge is the question.

        Either side can be it: the question can arrive before the answer (the
        common case) or after it, and the linker proposes the edge in whichever
        order the two units were compared.
        """
        ends = [units.get(edge.from_id), units.get(edge.to_id)]
        qs = [u.unit_id for u in ends if u is not None and u.type == "question"]
        return qs[0] if len(qs) == 1 else None

    def _one_answer_per_question(self, edges: list[Edge],
                                units: dict[str, Unit]) -> list[Edge]:
        """6.8: a question gets exactly one answer.

        The slot-matched `answers` rule reached 0.8 recall on dev but wrote **72
        edges against 5 gold pairs** — every value-bearing unit that shares a
        question's subject was called its answer. The guide's relation is
        one-to-one, so the invariant is enforced here rather than tuned away:
        within one ingest, the best-scoring candidate survives, and a question
        that already holds an answer from an earlier unit is not re-answered.

        This belongs in the ingest path and not in the linker because the linker
        is a pure function of (new, neighbours) — it cannot know that some
        earlier unit already answered this question. Only the store knows.
        """
        keep: dict[str, Edge] = {}
        for e in edges:
            if e.relation != "answers":
                continue
            qid = self._question_end(e, units)
            if qid is None:
                continue        # neither or both sides are questions: undecidable
            if self.store.edges_for(qid, direction="both", relation="answers"):
                continue        # already answered by an earlier unit
            cur = keep.get(qid)
            if cur is None or e.score > cur.score:
                keep[qid] = e
        kept = set(map(id, keep.values()))
        return [e for e in edges
                if e.relation != "answers" or id(e) in kept]

    def _link(self, new: Unit) -> tuple[list, int]:
        """6.8 online path: embed → top-k in scope → pair judge → guards → write."""
        k = int(self.cfg["link"]["k_neighbors"])
        vec = self.embedder.embed([new.text], role="doc")[0]
        filters = None
        if self.cfg["link"]["same_scope_only"] and new.project:
            filters = Filters(projects=[new.project])
        nbrs: list[Unit] = []
        seen: set[str] = set()
        for uid, _ in self.store.neighbors(vec, k + 1, filters=filters):
            if uid == new.unit_id or uid in seen:
                continue
            u = self.store.get(uid)
            if u is not None:
                seen.add(uid)
                nbrs.append(u)
        for u in self._lexical_neighbours(new, k):
            if u.unit_id not in seen:
                seen.add(u.unit_id)
                nbrs.append(u)
        edges = self.linker.link(new, nbrs)
        edges = self._one_answer_per_question(
            edges, {new.unit_id: new, **{u.unit_id: u for u in nbrs}})
        sup = 0
        for e in edges:
            if e.relation == "supersedes":
                try:
                    self.store.mark_superseded(e.to_id, e.from_id, new.source_date)
                    self.store.add_edges([e])
                    sup += 1
                except (ValueError, KeyError) as exc:
                    # a write-time refusal is itself evidence: keep the edge,
                    # mark it refused so the audit can see it
                    self.store.add_edges([e.model_copy(
                        update={"relation": "contradicts",
                                "guard": f"refused:{type(exc).__name__}"})])
                continue
            # G2 said "this incoming unit is dated EARLIER than an in-force one",
            # which is not a conflict — it is history arriving late. Re-run the
            # same guard set in the correct orientation; if the newer statement
            # supersedes this one, backdate this unit so the final valid set is
            # ingestion-order invariant (12.2) and `valid_from` keeps meaning
            # when it was *said*, not when we learned it (2.2).
            if e.relation == "contradicts" and e.guard == "G2_stale_source_date":
                newer = self.store.get(e.to_id)
                if (newer is not None and newer.is_current
                        and evaluate(newer, new)[0] == "supersedes"):
                    try:
                        self.store.mark_superseded(new.unit_id, newer.unit_id,
                                                   newer.source_date)
                    except ValueError:
                        self.store.add_edges([e])
                        continue
                    self.store.add_edges([Edge(
                        from_id=newer.unit_id, to_id=new.unit_id,
                        relation="supersedes", score=e.score,
                        evidence_source_id=newer.source_id, created_at=utcnow(),
                        guard="G2_backdated")])
                    sup += 1
                    continue
            self.store.add_edges([e])
        # Backdating cannot piggyback on a G2-labelled edge: `evaluate` returns at
        # G7 for a terse older statement (type=fact is not a superseding speech
        # act), so that branch was dead code and the 12.3 reverse-order case kept
        # two current heads. The rule is applied directly instead: an incoming
        # unit dated *earlier* than an in-force same-slot statement is history.
        if sup == 0 and new.is_current:
            # only units this pass already replaced are off-limits; a `supports`
            # or `contradicts` edge must not hide the in-force head, or the
            # older incoming unit is never backdated (measured: reverse-order
            # 12.3 kept December rank-1 because `seen` included every relation)
            seen = {e.to_id for e in edges if e.relation == "supersedes"}
            for nbr in nbrs:
                if nbr.unit_id in seen or not nbr.is_current:
                    continue
                if not same_slot(new, nbr) or new.source_date >= nbr.source_date:
                    continue
                if evaluate(nbr, new)[0] != "supersedes":
                    continue
                try:
                    self.store.mark_superseded(new.unit_id, nbr.unit_id,
                                               nbr.source_date)
                except ValueError:
                    break
                self.store.add_edges([Edge(
                    from_id=nbr.unit_id, to_id=new.unit_id, relation="supersedes",
                    score=0.9, evidence_source_id=nbr.source_id,
                    created_at=utcnow(), guard="G2_backdated")])
                sup += 1
                break
        return edges, sup

    def _lexical_neighbours(self, new: Unit, k: int) -> list[Unit]:
        """Neighbours on the *subject key*, not only on cosine (6.8 step 1)."""
        if not new.slots.subject:
            return []
        ids = self.store.unit_ids_for(Filters(
            subject_prefix=new.slots.subject,
            projects=[new.project] if new.project else []))
        out: list[Unit] = []
        for uid in ids[: k + 1]:
            if uid == new.unit_id:
                continue
            u = self.store.get(uid)
            if u is not None:
                out.append(u)
        return out

    def query_from_unit(self, unit: Unit) -> Query:
        return Query(text=unit.text, project=unit.project, slots=unit.slots,
                     source_unit_id=unit.unit_id,
                     intent="answer" if unit.type == "question" else "none")


from engine.schemas import Reading  # noqa: E402


def _tally(edges) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in edges:
        out[e.relation] = out.get(e.relation, 0) + 1
    return out
