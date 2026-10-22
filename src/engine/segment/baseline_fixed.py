"""Fixed-window segmentation baseline (BUILD_GUIDE 6.1: 200 words, stride 100).

Used as the comparison row against semantic/chunklet segmentation; the research
notes claim a fixed window often matches semantic chunking on retrieval.
"""

from __future__ import annotations

from engine.config import load_config
from engine.schemas import Slots, Source, Span, Unit, make_unit_id, utcnow
from engine.segment import turns as T

_WORD = T._WORD


def chunk_fixed_window(text: str, window_words: int = 200,
                       stride_words: int = 100) -> list[tuple[int, int, str]]:
    """Return (start, end, surface) windows over whitespace words.

    Both parameters are clamped to >= 1. A stride of 0 — reachable straight from
    `segment.fixed_stride_words` in a config — used to be an infinite loop that
    grew a list until the kernel killed the process (measured: `timeout -s KILL 6`
    exits 137 on a six-word string). The old `max(i + stride, 1)` guard only
    covered the first step; once `i` left 0 it advanced by 0 forever.

    `window_words=0` was the quieter half of the same defect: `j == i` meant
    `words[j - 1]` sliced the *previous* word, producing an inverted span and an
    empty surface (measured: `(2, 1, '')`) — a unit that stores nothing and would
    still be embedded and linked.
    """
    window_words = max(1, int(window_words))
    stride_words = max(1, int(stride_words))
    words = list(_WORD.finditer(text))
    if not words:
        return []
    out: list[tuple[int, int, str]] = []
    n = len(words)
    i = 0
    while i < n:
        j = min(i + window_words, n)
        start = words[i].start()
        end = words[j - 1].end()
        out.append((start, end, text[start:end]))
        if j >= n:
            break
        i += stride_words                      # >= 1, so this always advances
    return out


def segment_fixed(source: Source, cfg=None, window_words: int | None = None,
                  stride_words: int | None = None) -> list[Unit]:
    cfg = cfg or load_config()
    seg = cfg["segment"]
    # `is None`, not `or`: `0 or cfg[...]` silently replaced a caller's explicit
    # window with the config value instead of honouring it (or clamping it where
    # `chunk_fixed_window` could see it).
    if window_words is None:
        window_words = seg["fixed_window_words"]
    if stride_words is None:
        stride_words = seg["fixed_stride_words"]
    text = source.body()
    units: list[Unit] = []
    for seq, (start, end, surface) in enumerate(
            chunk_fixed_window(text, window_words, stride_words)):
        surface = surface.strip()
        units.append(Unit(
            unit_id=make_unit_id(source.source_id, seq, start, end,
                                 source.default_speaker),
            source_id=source.source_id,
            source_type=source.source_type,
            source_date=source.source_date,
            ingested_at=utcnow(),
            text=surface,
            sentences=[Span.whole(surface)],
            speaker=source.default_speaker,
            project=source.project,
            slots=Slots(),
        ))
    return units
