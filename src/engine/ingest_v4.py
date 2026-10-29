"""v4 ingestion: immutable message → sentence chunks → units → links → state.

One pass, in the architecture's order (§2 → §3 → §4 → §5). The pieces the old
ingest dropped and this one keeps:

* the message timestamp is the unit date (it used `date.today()`, which made
  every temporal view wrong);
* `kind` is the speech act, `intent` is read instead of hardcoded `"none"`;
* every unit carries the span its tags came from, and a confidence;
* a bare confirmation is attached to the preceding proposal instead of becoming
  an isolated unit;
* typed links and state rows are appended, not just modelled in the schema.
"""
from __future__ import annotations

from datetime import date

from engine.link.thread import detect_topic
from engine.read.v4_tags import read_tags
from engine.schemas_v4 import ChunkRecord, MessageRecord, SemanticUnit
import re

from engine.segment.sentences_v4 import (is_confirmation, semantic_groups,
                                         split_long_spans)
from engine.read.v4_tags import point_of

QUESTION_TURN = re.compile(r"\b(?:quand|combien|qui|où|quel(?:le)?)\b", re.I)
from engine.store.v4 import V4Store
from engine.threads.state_v4 import derive, event_key


def _as_date(timestamp: str) -> date:
    try:
        return date.fromisoformat(str(timestamp)[:10])
    except Exception:
        return date.today()


class V4Ingestor:
    def __init__(self, store: V4Store, linker=None, link_window: int = 24,
                 tag_fn=None, long_split: bool = False,
                 max_words: int = 70):
        self.store = store
        self.linker = linker
        self.link_window = link_window
        # §3 is pluggable: the keyword reader by default, the fine-tuned Laya
        # heads when one is passed. Same signature as `read_tags`.
        self.tag_fn = tag_fn or read_tags
        self.long_split = long_split
        self.max_words = max_words
        self._chunker = None

    def ingest(self, m: MessageRecord) -> dict:
        """Ingest one immutable message. Idempotent by message_id."""
        self.store.add_message(m)
        source_date = _as_date(m.timestamp)
        spans = None
        merge = True
        if self.long_split:
            spans = split_long_spans(m.text, max_words=self.max_words,
                                     chunker=self._chunker_fn())
            merge = False
        groups = semantic_groups(m.text, topic_fn=detect_topic,
                                 intent_fn=lambda t: self.tag_fn(t).intent,
                                 spans=spans, merge=merge)
        unit_ids: list[str] = []
        n_chunks = n_links = n_states = 0
        # prev spans turns, not just sentences: an answer must be able to
        # inherit the point of the question it follows
        recent = self.store.units_for_project(m.project_id, 1)
        prev: SemanticUnit | None = _as_unit(recent[-1]) if recent else None

        # §2: a bare confirmation belongs to the preceding *message*, not only
        # to the preceding sentence of the same message ("OK, validé" after a
        # proposal posted earlier).
        if is_confirmation(m.text.strip()):
            recent = self.store.units_for_project(m.project_id, 1)
            if recent:
                prev = _as_unit(recent[-1])

        for gi, group in enumerate(groups):
            text = m.text[group[0].start:group[-1].end]
            chunk_ids: list[str] = []
            for si, s in enumerate(group):
                cid = f"{m.message_id}:s{gi}.{si}"
                self.store.add_chunk(ChunkRecord(cid, m.message_id, s.start,
                                                 s.end, s.text, si))
                chunk_ids.append(cid)
                n_chunks += 1

            tags = self.tag_fn(text, speaker=m.speaker,
                               source_date=source_date,
                               source_type=m.source_type)
            point = getattr(tags, "point", "") or point_of(text, tags.entities)
            if not point and prev is not None and is_confirmation(text):
                point = getattr(prev, "point", "")
            if (not point and prev is not None
                    and QUESTION_TURN.search(prev.text or "")):
                # an answer that names no object ("le 15 août") is about the
                # object of the question it answers ("les clés")
                point = getattr(prev, "point", "")
            kind = tags.kind
            status = tags.status
            slot, value = tags.slot, tags.value

            # §2: "OK, validé" belongs to the preceding proposal.
            confirmation = is_confirmation(text)
            if confirmation and prev is not None:
                kind = "answer"
                status = "confirmed"
                slot = prev.slot or slot
                value = prev.value or value

            entities = list(tags.entities)
            if m.speaker:
                entities.append({"type": "person", "value": m.speaker,
                                 "start": 0, "end": 0})
            if m.project_id:
                entities.append({"type": "project", "value": m.project_id,
                                 "start": 0, "end": 0})
            seen: set[tuple[str, str]] = set()
            deduped = []
            for e in entities:
                key = (e["type"], str(e["value"]).lower())
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(e)

            unit = SemanticUnit(
                unit_id=f"{m.message_id}:u{gi}", message_id=m.message_id,
                chunk_ids=tuple(chunk_ids), text=text, kind=kind, status=status,
                intent=tags.intent, entities=tuple(deduped),
                owner=tags.owner, deadline=tags.deadline, slot=slot,
                point=point, value=value,
                evidence=None, confidence=tags.confidence,
                model_version=tags.model_version, origin=m.origin)
            if tags.evidence:
                from engine.schemas_v4 import Evidence
                s, e, t = tags.evidence
                unit = SemanticUnit(**{**unit.__dict__,
                                       "evidence": Evidence(s, e, t)})
            self.store.add_unit(unit)
            unit_ids.append(unit.unit_id)

            # §4: candidates are the recent units of the same project
            prior = (self.store.current_state(
                event_key(m.project_id, unit.slot, unit.point), unit.slot)
                     if unit.slot else None)
            prior_value = prior.get("value") if prior else None
            if self.linker is not None and self.linker.available:
                cands = self._candidates(m, unit.unit_id)
                links = self.linker.link(unit, cands,
                                         evidence=unit.evidence.text
                                         if unit.evidence else text)
                superseded = []
                for link in links:
                    self.store.add_link(link)
                    n_links += 1
                    if link.relation == "supersedes":
                        old = self.store.unit(link.to_unit_id)
                        if old is not None:
                            superseded.append(old)
                for row in derive(unit, m.project_id,
                                  [_as_unit(o) for o in superseded if o],
                                  prior_value=prior_value):
                    self.store.add_state(row)
                    n_states += 1
            else:
                for row in derive(unit, m.project_id, prior_value=prior_value):
                    self.store.add_state(row)
                    n_states += 1

            prev = unit

        return {"unit_ids": unit_ids, "chunks": n_chunks, "links": n_links,
                "states": n_states}

    def _chunker_fn(self):
        """Lazy: a long turn needs candidate sentences before it can be split."""
        if self._chunker is None:
            from chonkie import SemanticChunker

            # similarity_window=1 compares adjacent windows. The default of 3
            # compares 3x8-word groups in noisy ASR and almost never crosses
            # the threshold (measured: 8 of 42 long turns split, then 42/42).
            self._chunker = SemanticChunker(chunk_size=512, threshold=0.6,
                                            similarity_window=1)
        return self._chunker

    def _candidates(self, m: MessageRecord, unit_id: str) -> list[SemanticUnit]:
        rows = self.store.units_for_project(m.project_id, self.link_window)
        out = []
        for r in rows:
            if r["unit_id"] == unit_id or r["message_id"] == m.message_id:
                continue
            out.append(_as_unit(r))
        return out


def _as_unit(row: dict) -> SemanticUnit:
    from engine.schemas_v4 import Evidence
    ev = row.get("evidence")
    return SemanticUnit(
        unit_id=row["unit_id"], message_id=row["message_id"],
        chunk_ids=tuple(row.get("chunk_ids") or []), text=row["text"],
        kind=row.get("kind") or "fact", status=row.get("status") or "unknown",
        intent=row.get("intent") or "answer",
        entities=tuple(row.get("entities") or []), owner=row.get("owner") or "",
        deadline=row.get("deadline") or "", slot=row.get("slot") or "",
        point=row.get("point") or "", value=row.get("value") or "",
        evidence=Evidence(ev["start"], ev["end"], ev["text"]) if ev else None,
        confidence=float(row.get("confidence") or 0.0),
        model_version=row.get("model_version") or "v4-rules-1",
        origin=row.get("origin") or "real")


__all__ = ["V4Ingestor", "_as_unit"]
