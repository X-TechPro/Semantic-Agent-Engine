"""§6 search: interpretation, entity/event, BM25+dense RRF, graph, rerank, state.

The layers are explicit so each one can be measured (and switched off) alone:

1. `interpret_query` — intent, requested slot, entities, and resolution of
   "it / he / that / then" against the most recent unit of the project;
2. `UnitIndex` — separate entity and event lookups, entity hits ranked first;
3. `_lexical` + `_dense` — fused by reciprocal rank, not by union;
4. `_expand` — traverses typed links (`supersedes`, `answers`, `supports`,
   `creates_action`, `contradicts`) plus `same_event`; never a turn window;
5. `_rerank` — FlauBERT relation head if a trained checkpoint exists, else
   encoder cosine, declared in `parts["rerank_model"]`;
+ `_state` — current state weighting, so a cancelled value does not lose to an
   older confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.query.interpret import QueryFrame, interpret
from engine.read.v4_tags import extract_entities, point_of, slot_of
from engine.store.v4 import V4Store, event_id_for

RELATION_WEIGHTS = {
    "supersedes": 0.70, "answers": 0.85, "supports": 1.00,
    "creates_action": 0.80, "contradicts": 0.45, "unrelated": 0.0,
}
STATUS_FACTOR = {"confirmed": 1.0, "cancelled": 1.0, "completed": 1.0,
                 "proposed": 0.70, "replaced": 0.25, "unknown": 0.90}


@dataclass
class Frame:
    text: str
    intent: str
    slot: str
    entities: list[dict]
    references: list[str]
    resolved_slot: str = ""
    resolved_from: str | None = None
    project: str | None = None
    point: str = ""


@dataclass
class SearchHit:
    unit_id: str
    message_id: str
    text: str
    score: float
    parts: dict = field(default_factory=dict)
    evidence: dict | None = None
    state: dict | None = None
    status: str = "unknown"
    slot: str = ""
    rank: int = 0

    def as_dict(self) -> dict:
        return {"unit_id": self.unit_id, "message_id": self.message_id,
                "text": self.text, "score": self.score, "parts": self.parts,
                "evidence": self.evidence, "state": self.state,
                "status": self.status, "slot": self.slot, "rank": self.rank}


def interpret_query(text: str, store: V4Store,
                    project: str | None = None) -> Frame:
    """§6 layer 1. Slots use the same taxonomy as the reading layer.

    The previous version read the query slot from a legacy dict that called a
    budget question `amount`, while every unit stored `budget`; the requested
    slot then never matched a unit and layer 6 could not fire.
    """
    q: QueryFrame = interpret(text)
    ents = [e for e in extract_entities(text)]
    slot = slot_of(text, tuple(ents))
    frame = Frame(text=text, intent=q.intent,
                  slot="" if slot == "subject" else slot,
                  point=point_of(text, tuple(ents)),
                  entities=ents, references=list(q.references or []),
                  project=project)
    if q.needs_resolution or not frame.slot:
        recent = store.units_for_project(project) if project else store.all_units()
        if recent:
            prev = recent[-1]
            frame.resolved_slot = prev.get("slot") or ""
            frame.resolved_from = prev["unit_id"]
            if not frame.slot:
                frame.slot = frame.resolved_slot
            seen = {(e.get("type"), str(e.get("value")).lower()) for e in ents}
            for e in prev.get("entities") or []:
                key = (e.get("type"), str(e.get("value")).lower())
                if key not in seen:
                    seen.add(key)
                    frame.entities.append(dict(e))
    return frame


class UnitIndex:
    """Entity, event and slot lookups over the immutable units."""

    def __init__(self, units: list[dict]):
        self.units = {u["unit_id"]: u for u in units}
        self.entity: dict[tuple[str, str], list[str]] = {}
        self.event: dict[str, list[str]] = {}
        self.slot: dict[str, list[str]] = {}
        for u in units:
            uid = u["unit_id"]
            for e in u.get("entities") or []:
                key = (str(e.get("type")), str(e.get("value")).lower())
                self.entity.setdefault(key, []).append(uid)
            if u.get("slot"):
                self.slot.setdefault(u["slot"], []).append(uid)
            topic = next((str(e.get("value")) for e in u.get("entities") or []
                          if e.get("type") == "event"), "")
            if topic:
                self.event.setdefault(topic, []).append(uid)

    def by_entities(self, entities: list[dict]) -> dict[str, float]:
        hits: dict[str, float] = {}
        for e in entities:
            key = (str(e.get("type")), str(e.get("value")).lower())
            for uid in self.entity.get(key, []):
                hits[uid] = hits.get(uid, 0.0) + 1.0
        return hits

    def by_event(self, topic: str) -> list[str]:
        return list(self.event.get(topic, []))


class V4Search:
    def __init__(self, store: V4Store, linker=None, encoder=None, *,
                 vectors: dict[str, np.ndarray] | None = None,
                 reranker=None, rerank_k: int = 20,
                 rrf_k: int = 60, final_k: int = 10, candidates: int = 30,
                 text_fn=None):
        self.store = store
        self.linker = linker
        self.encoder = encoder
        self.reranker = reranker
        self.rerank_k = rerank_k
        self.rrf_k = rrf_k
        self.final_k = final_k
        self.candidates = candidates
        # what gets embedded at index time; default is the unit text alone.
        # A caller can prepend speaker/date so "when/who" turns carry that
        # information in the vector instead of only in the metadata.
        self.text_fn = text_fn
        self._units: list[dict] = []
        self._index: UnitIndex | None = None
        # unit_id -> vector. Late-chunked vectors arrive here already computed
        # from the meeting's forward pass; units without one fall back to the
        # encoder during _refresh.
        self._vecs: dict[str, np.ndarray] = dict(vectors or {})
        self._project: dict[str, str] = {}
        self._scope: str | None = None
        self._refresh()

    # --- index -------------------------------------------------------------
    def _refresh(self) -> None:
        units = self.store.all_units()
        if len(units) != len(self._units):
            self._units = units
            self._index = UnitIndex(units)
            self._project = self.store.unit_projects()
            todo = [u for u in units if u["unit_id"] not in self._vecs]
            missing = [self.text_fn(u) if self.text_fn else u["text"] for u in todo]
            if self.encoder is not None and missing:
                vecs = np.asarray(self.encoder(missing), dtype=np.float32)
                if vecs.ndim == 1:
                    vecs = vecs[None, :]
                norm = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.clip(norm, 1e-9, None)
                for u, v in zip(todo, vecs):
                    self._vecs[u["unit_id"]] = v

    def _keep(self, uid: str) -> bool:
        """A query scoped to a project never retrieves another project's units."""
        return self._scope is None or self._project.get(uid) == self._scope

    # --- layers ------------------------------------------------------------
    def _lexical(self, text: str) -> list[str]:
        chunk_to_unit = {}
        for u in self._units:
            for cid in u.get("chunk_ids") or []:
                chunk_to_unit[cid] = u["unit_id"]
        out: list[str] = []
        for row in self.store.bm25(text, k=self.candidates * 2):
            uid = chunk_to_unit.get(row["chunk_id"])
            if uid and uid not in out and self._keep(uid):
                out.append(uid)
        return out

    def _dense(self, text: str, qvec=None) -> list[str]:
        if self.encoder is None or not self._vecs:
            return []
        q = qvec if qvec is not None else np.asarray(
            self.encoder([text]), dtype=np.float32)[0]
        n = float(np.linalg.norm(q))
        if n:
            q = q / n
        scored = [(uid, float(v @ q)) for uid, v in self._vecs.items()
                  if self._keep(uid)]
        scored.sort(key=lambda t: -t[1])
        return [uid for uid, _ in scored[: self.candidates]]

    def _rrf(self, rankings: dict[str, list[str]],
             weights: dict[str, float]) -> dict[str, float]:
        fused: dict[str, float] = {}
        for name, order in rankings.items():
            w = weights.get(name, 1.0)
            for rank, uid in enumerate(order, start=1):
                fused[uid] = fused.get(uid, 0.0) + w / (self.rrf_k + rank)
        return fused

    def _expand(self, fused: dict[str, float]) -> dict[str, float]:
        out = dict(fused)
        seeds = sorted(fused, key=lambda u: -fused[u])[:8]
        for uid in seeds:
            base = fused[uid]
            for link in self.store.links_of(uid):
                other = (link["to_unit_id"] if link["from_unit_id"] == uid
                         else link["from_unit_id"])
                w = RELATION_WEIGHTS.get(link["relation"], 0.0)
                if not w or not self._keep(other):
                    continue
                add = base * w * float(link["confidence"])
                if add > out.get(other, 0.0):
                    out[other] = add
            u = self.store.unit(uid)
            if u and u.get("slot"):
                for same in self._index.slot.get(u["slot"], [])[:3]:
                    if same == uid or not self._keep(same):
                        continue
                    add = base * 0.5
                    if add > out.get(same, 0.0):
                        out[same] = add
        return out

    def _qvec(self, text: str, qvec=None):
        if self.encoder is None:
            return None
        q = qvec if qvec is not None else np.asarray(
            self.encoder([text]), dtype=np.float32)[0]
        n = float(np.linalg.norm(q))
        return q / n if n else q

    def _cos(self, uid: str, q=None) -> float:
        if q is None or uid not in self._vecs:
            return 0.0
        return float(self._vecs[uid] @ q)

    def _rerank(self, text: str, fused: dict[str, float],
                qvec=None) -> dict[str, float]:
        """§6 layer 5 as rank fusion, not min-max blending.

        Two orderings are always present (RRF fusion, encoder cosine) and the
        relation head adds a third when a checkpoint exists. Blending raw
        min-max scores moved the gold antecedent out of the top-20 even when
        the RRF pool had it (measured: RRF recall 35/40 -> final recall 15/40);
        rank fusion is monotone and cannot lose a candidate that every signal
        ranks well.
        """
        ids = list(fused)
        if not ids:
            return {}
        orders: dict[str, list[str]] = {
            "fused": sorted(ids, key=lambda u: -fused[u])}
        weights = {"fused": 1.0}
        q = self._qvec(text, qvec)
        if q is not None:
            mu = {u: self._cos(u, q) for u in ids}
            orders["cos"] = sorted(ids, key=lambda u: -mu[u])
            weights["cos"] = 0.5
        model = "cosine" if self.encoder is not None else "rrf-only"
        if (self.linker is not None and self.linker.available
                and getattr(self.linker, "model", None) is not None):
            texts = [self.store.unit(u)["text"] for u in ids]
            probs = self.linker.model.relation_probs([text] * len(ids), texts)
            rel = {u: (p.get("supersedes", 0.0) + p.get("answers", 0.0)
                       + p.get("supports", 0.0) + p.get("creates_action", 0.0))
                   for u, p in zip(ids, probs)}
            orders["rel"] = sorted(ids, key=lambda u: -rel[u])
            weights["rel"] = 1.0
            model = self.linker.model_version
        fk = self.rrf_k
        rank_maps = {name: {u: i for i, u in enumerate(order)}
                     for name, order in orders.items()}
        out = {}
        for u in ids:
            out[u] = sum(w / (fk + rank_maps[name][u] + 1)
                         for name, w in weights.items())
        for u in ids:
            fused[u] = out[u]
        self.last_rerank_model = model
        return out

    def _state(self, hits: list[tuple[str, float]], mode: str
               ) -> list[tuple[str, float]]:
        """§6 layer 6: the current state's evidence wins its slot.

        The architecture's example is exact: "the latest cancellation must not
        lose to an older confirmation". So the unit that owns the current state
        for its slot gets a boost, a superseded unit is demoted, and a proposal
        is discounted rather than hidden.
        """
        superseded = self.store.superseded_units()
        out = []
        for uid, score in hits:
            u = self.store.unit(uid) or {}
            status = u.get("status") or "unknown"
            factor = STATUS_FACTOR.get(status, 0.9)
            if mode != "history" and u.get("slot"):
                cur = self.store.current_state(event_id_for(u, self.store),
                                               u["slot"])
                if cur and cur.get("evidence_unit") == uid:
                    factor = max(factor, 1.5)
            if uid in superseded:
                factor = 1.2 if mode == "history" else 0.15
            out.append((uid, score * factor))
        return out

    # --- public ------------------------------------------------------------
    def _current_owners(self, slot: str) -> set[str]:
        """Units that own the current state of a slot, across projects.

        A "current" question must not rank an older confirmation above the
        later cancellation (§6 layer 6). This is a hard ordering rule, not a
        score nudge, because the architecture states it as a rule.
        """
        out: set[str] = set()
        for u in self._units:
            if u.get("slot") != slot or not self._keep(u["unit_id"]):
                continue
            cur = self.store.current_state(event_id_for(u, self.store), slot)
            if cur and cur.get("evidence_unit"):
                out.add(str(cur["evidence_unit"]))
        return out

    def diagnose(self, text: str, project: str | None = None) -> dict:
        """Per-layer *ranked* pools, so recall@20 is measured per layer."""
        self._refresh()
        self._scope = project
        frame = interpret_query(text, self.store, project)
        ent = {u: s for u, s in self._index.by_entities(frame.entities).items()
               if self._keep(u)}
        q = self._qvec(frame.text)
        pool_ent = sorted(ent, key=lambda u: (-ent[u], -self._cos(u, q), u))
        topic = next((str(e.get("value")) for e in frame.entities
                      if e.get("type") == "event"), "")
        pools = {
            "entity": pool_ent,
            "event": [u for u in self._index.by_event(topic) if self._keep(u)]
            if topic else [],
            "bm25": self._lexical(frame.text),
            "dense": self._dense(frame.text),
        }
        rankings = {k: v for k, v in pools.items() if v}
        fused = self._rrf(rankings, {"entity": 1.4, "event": 1.2,
                                     "bm25": 1.0, "dense": 1.0})
        expanded = self._expand(fused)
        return {"frame": frame, "pools": pools,
                "rrf": sorted(fused, key=lambda u: -fused[u]),
                "graph": sorted(expanded, key=lambda u: -expanded[u])}

    def retrieve(self, text: str, k: int | None = None, *, mode: str = "current",
                 project: str | None = None, qvec=None) -> list[SearchHit]:
        self._refresh()
        self._scope = project
        frame = interpret_query(text, self.store, project)
        rankings: dict[str, list[str]] = {}
        ent = {u: s for u, s in self._index.by_entities(frame.entities).items()
               if self._keep(u)}
        if ent:
            q = self._qvec(frame.text, qvec)
            rankings["entity"] = sorted(
                ent, key=lambda u: (-ent[u], -self._cos(u, q), u))
        topic = next((str(e.get("value")) for e in frame.entities
                      if e.get("type") == "event"), "")
        if topic:
            ev = [u for u in self._index.by_event(topic) if self._keep(u)]
            if ev:
                rankings["event"] = ev
        lex = self._lexical(frame.text)
        if lex:
            rankings["bm25"] = lex
        dense = self._dense(frame.text, qvec)
        if dense:
            rankings["dense"] = dense
        if not rankings:
            return []
        weights = {"entity": 1.4, "event": 1.2, "bm25": 1.0, "dense": 1.0}
        fused = self._rrf(rankings, weights)
        fused = self._expand(fused)
        fused = self._rerank(frame.text, fused, qvec)
        if frame.slot:
            for uid in list(fused):
                u = self.store.unit(uid) or {}
                if u.get("slot") == frame.slot:
                    fused[uid] *= 1.5
        if frame.point:
            for uid in list(fused):
                u = self.store.unit(uid) or {}
                if u.get("point") == frame.point:
                    fused[uid] *= 2.5
        ranked = sorted(fused.items(), key=lambda t: -t[1])
        if mode != "link":
            ranked = self._state(ranked, mode)
            ranked.sort(key=lambda t: -t[1])
        if mode == "current" and frame.slot:
            owners = self._current_owners(frame.slot)
            if owners:
                best = max(s for _, s in ranked) or 1.0
                ranked = [(uid, max(s, best * 1.05) if uid in owners else s)
                          for uid, s in ranked]
                ranked.sort(key=lambda t: -t[1])

        if self.reranker is not None and len(ranked) > 1:
            head = ranked[: self.rerank_k]
            texts = [self.store.unit(u)["text"] for u, _ in head]
            scores = self.reranker(frame.text, texts)
            order = sorted(range(len(head)), key=lambda i: -scores[i])
            pos = {i: p for p, i in enumerate(order)}
            fused_rank = sorted(
                range(len(head)),
                key=lambda i: -(1.0 / (self.rrf_k + i + 1)
                                + 1.0 / (self.rrf_k + pos[i] + 1)))
            head = [head[i] for i in fused_rank]
            ranked = head + ranked[self.rerank_k:]

        hits: list[SearchHit] = []
        for rank, (uid, score) in enumerate(ranked[: k or self.final_k], start=1):
            u = self.store.unit(uid) or {}
            state = (self.store.current_state(event_id_for(u, self.store),
                                              u.get("slot"))
                     if u.get("slot") else None)
            hits.append(SearchHit(
                unit_id=uid, message_id=u.get("message_id", ""),
                text=u.get("text", ""), score=round(score, 6),
                parts={"entity": ent.get(uid, 0.0),
                       "rerank_model": getattr(self, "last_rerank_model", "")},
                evidence=u.get("evidence"), state=state,
                status=u.get("status") or "unknown",
                slot=u.get("slot") or "", rank=rank))
        return hits


__all__ = ["V4Search", "SearchHit", "Frame", "UnitIndex", "interpret_query",
           "RELATION_WEIGHTS", "STATUS_FACTOR"]
