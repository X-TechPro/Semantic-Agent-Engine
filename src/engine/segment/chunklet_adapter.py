"""`segment(source) -> list[Unit]` — the only segmentation interface (2.2).

Backends (6.1):
  * ``chunklet`` (default): sentence-per-unit for meetings/notes, chunklet's
    document chunker for documents, chunklet's CodeChunker for code.
  * ``turns``: one unit per speaker turn (no sentence splitting).
  * ``fixed``: the fixed-window baseline (200 words / stride 100).
"""

from __future__ import annotations

from datetime import date, datetime

from engine.config import Config, load_config
from engine.schemas import Source, Span, Unit, Slots, make_unit_id, utcnow
from engine.segment import turns as T
from engine.segment.baseline_fixed import chunk_fixed_window
from engine.segment.code_chunker import chunk_code

_MAX_UNIT_TOKENS_HARD = 120  # a unit must stay retrievable, not a whole file


class Segmenter:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        seg = self.cfg["segment"]
        self.backend = seg["backend"]
        self.lang = seg["lang"]
        self.max_turn_tokens = seg["max_turn_tokens"]
        self.asr_max_tokens = seg["asr_max_tokens"]
        self.asr_min_tokens = seg["asr_min_tokens"]
        self.doc_max_tokens = seg["doc_max_tokens"]
        self.doc_min_tokens = seg["doc_min_tokens"]
        self.max_unit_tokens = seg["max_unit_tokens"]
        self.merge_short_turns = True

    # --- public contract ---------------------------------------------------
    def segment(self, source: Source) -> list[Unit]:
        if source.source_type == "code":
            return self._segment_code(source)
        if source.source_type == "document":
            return self._segment_document(source)
        return self._segment_meeting(source)

    # --- meetings / notes --------------------------------------------------
    def _segment_meeting(self, source: Source) -> list[Unit]:
        units: list[Unit] = []
        seq = 0
        for turn in source.turns:
            text = (turn.text or "").strip()
            if not text:
                continue
            speaker = turn.speaker or source.default_speaker
            tindex = turn.turn_index if turn.turn_index is not None else seq
            # a turn may be *about* another project (distractor); the unit takes
            # the discussed project, not the meeting's, or the 6.8.3 G0
            # cross-project guard can never fire
            for piece, spans in self._pieces(text):
                units.append(self._make_unit(
                    source, seq, piece, spans, speaker=speaker, turn_index=tindex,
                    gold=turn.gold_event_ids, project=turn.project or source.project))
                seq += 1
        return units

    def _over_unit_budget(self, text: str, cap: int | None = None) -> bool:
        """6.1: a unit must stay retrievable. `max_unit_tokens` is the budget and
        `_MAX_UNIT_TOKENS_HARD` is the ceiling under which a looser config still
        cannot ship a whole file as one unit."""
        limit = min(self.max_unit_tokens, _MAX_UNIT_TOKENS_HARD) if cap is None else cap
        return T.count_tokens(text) > limit

    def _windowed(self, text: str, cap: int | None = None) -> list[tuple[str, list[Span]]]:
        window = cap or self.asr_max_tokens
        return [(w.text, [Span.whole(w.text)])
                for w in T.split_by_token_window(text, window, self.asr_min_tokens)]

    def _pieces(self, text: str) -> list[tuple[str, list[Span]]]:
        """Split one speaker turn into (unit_text, sentence_spans).

        Never merges across speakers by construction: this only ever sees a
        single turn. A turn of several short sentences stays *one* unit when it
        is under `max_turn_tokens` (6.1: "one speaker turn if the turn is a
        single short utterance"), otherwise one unit per sentence.
        """
        if T.is_unpunctuated(text):
            ws = T.split_by_token_window(text, self.asr_max_tokens, self.asr_min_tokens)
            if len(ws) <= 1:
                return [(text, [Span.whole(text)])]
            return [(w.text, [Span.whole(w.text)]) for w in ws]

        sents = T.split_sentences(text, self.lang)
        if len(sents) <= 1:
            # chunklet refuses a boundary when the sentence after the period starts
            # lower-case (D-003; measured: 12 lower-cased ASR sentences came back as
            # ONE sentence). Without this fallback a punctuated-but-lowercase
            # transcript ships as a single unretrievable unit.
            if self._over_unit_budget(text):
                parts = self._windowed(text)
                if len(parts) > 1:
                    return parts
            return [(text, [Span.whole(text)])]

        total = sum(T.count_tokens(s.text) for s in sents)
        if self.merge_short_turns and total <= self.max_turn_tokens:
            spans = [Span(start=s.start, end=s.end, surface=s.text) for s in sents]
            return [(text, spans)]

        out: list[tuple[str, list[Span]]] = []
        for s in sents:
            # oversized sentence -> ASR-style window split, offsets are local
            if self._over_unit_budget(s.text):
                out.extend(self._windowed(s.text))
            else:
                out.append((s.text, [Span.whole(s.text)]))
        return out

    # --- documents ---------------------------------------------------------
    def _segment_document(self, source: Source) -> list[Unit]:
        text = (source.text or "").strip()
        if not text:
            return []
        sents = T.split_sentences(text, self.lang)
        if not sents:
            sents = [T.Sentence(0, len(text), text)]
        groups = self._group_sentences(sents)
        units: list[Unit] = []
        seq = 0
        for group in groups:
            start, end = group[0].start, group[-1].end
            chunk = text[start:end]
            # spans are re-based onto the chunk, and `Span` is a pydantic model
            # whose third field is `surface`: the earlier positional call raised
            # TypeError, so no document could ever be ingested.
            spans = [Span(start=s.start - start, end=s.end - start, surface=s.text)
                     for s in group]
            units.append(self._make_unit(source, seq, chunk, spans,
                                         speaker=source.default_speaker,
                                         turn_index=None, gold=[]))
            seq += 1
        return units

    def _group_sentences(self, sents: list[T.Sentence]) -> list[list[T.Sentence]]:
        """Merge adjacent sentences while the chunk stays inside the token budget.

        6.1 asks for cosine >= tau_merge on top of adjacency; that needs an
        embedder, which segmentation must not depend on (lego rule). The
        embedder-aware variant lives in `chunklet_adapter.DocumentSemanticMerger`
        and is selected by config when an embedder is supplied.
        """
        groups: list[list[T.Sentence]] = []
        cur: list[T.Sentence] = []
        cur_tokens = 0
        for s in sents:
            n = T.count_tokens(s.text)
            if cur and cur_tokens + n > self.doc_max_tokens:
                groups.append(cur)
                cur, cur_tokens = [], 0
            cur.append(s)
            cur_tokens += n
        if cur:
            groups.append(cur)
        # min_tokens guard: fold tiny trailing chunks into the previous one
        merged: list[list[T.Sentence]] = []
        for g in groups:
            toks = sum(T.count_tokens(s.text) for s in g)
            if merged and toks < self.doc_min_tokens:
                merged[-1] = merged[-1] + g
            else:
                merged.append(g)
        return merged

    # --- code --------------------------------------------------------------
    def _segment_code(self, source: Source) -> list[Unit]:
        chunks = chunk_code(source.code or "", language=source.language,
                            max_tokens=self.doc_max_tokens)
        units: list[Unit] = []
        for seq, (text, meta) in enumerate(chunks):
            spans = [Span.whole(text)]
            units.append(self._make_unit(source, seq, text, spans,
                                         speaker=None, turn_index=None, gold=[],
                                         extra={"code_symbol": meta.get("symbol")}))
        return units

    # --- factory -----------------------------------------------------------
    def _make_unit(self, source: Source, seq: int, text: str, spans: list[Span],
                   *, speaker: str | None, turn_index: int | None,
                   gold: list[str], extra: dict | None = None,
                   project: str | None = None) -> Unit:
        start = spans[0].start if spans else 0
        end = spans[-1].end if spans else len(text)
        return Unit(
            unit_id=make_unit_id(source.source_id, seq, start, end, speaker),
            source_id=source.source_id,
            source_type=source.source_type,
            source_date=source.source_date,
            ingested_at=utcnow(),
            text=text,
            sentences=spans,
            speaker=speaker,
            project=project if project is not None else source.project,
            turn_index=turn_index,
            slots=Slots(),
            gold_event_ids=list(gold),
        )


class DocumentSemanticMerger:
    """6.1 document rule: merge adjacent sentences while cosine >= tau_merge.

    Takes an `embed` callable so segmentation does not import the embedder
    stage's internals.
    """

    def __init__(self, embed, tau_merge: float = 0.55, max_tokens: int = 256,
                 min_tokens: int = 40):
        self.embed = embed
        self.tau = tau_merge
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens

    def group(self, sents: list[T.Sentence]) -> list[list[T.Sentence]]:
        import numpy as np

        vecs = np.asarray(self.embed([s.text for s in sents], role="doc"))
        groups: list[list[T.Sentence]] = []
        cur: list[T.Sentence] = []
        cur_tokens = 0
        for i, s in enumerate(sents):
            n = T.count_tokens(s.text)
            if cur:
                sim = float(vecs[i] @ vecs[i - 1])
                if sim < self.tau or cur_tokens + n > self.max_tokens:
                    groups.append(cur)
                    cur, cur_tokens = [], 0
            cur.append(s)
            cur_tokens += n
        if cur:
            groups.append(cur)
        merged: list[list[T.Sentence]] = []
        for g in groups:
            if merged and sum(T.count_tokens(x.text) for x in g) < self.min_tokens:
                merged[-1] = merged[-1] + g
            else:
                merged.append(g)
        return merged


def segment(source: Source, cfg: Config | None = None) -> list[Unit]:
    """Module-level entry point for the 2.2 contract."""
    cfg = cfg or load_config()
    if cfg["segment"]["backend"] == "fixed":
        from engine.segment.baseline_fixed import segment_fixed

        return segment_fixed(source, cfg)
    return Segmenter(cfg).segment(source)


__all__ = ["Segmenter", "DocumentSemanticMerger", "segment"]
