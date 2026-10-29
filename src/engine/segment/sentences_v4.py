"""v4 sentence spans and semantic-unit grouping (ARCH_V4 §2).

Two layers, in this order:

1. exact sentence spans with offsets — `text[start:end] == span.text`;
2. groups of *adjacent* spans that stay on the same topic and intent.

Two rules are load-bearing and were inverted in the previous version:

* a short confirmation ("OK, validé") attaches to the preceding span; it is
  never its own unit;
* a span joins the previous group because the topic/intent is unchanged, not
  because it is short. Two consecutive sentences on one subject therefore make
  one unit, and a topic switch makes two.

If the text has no punctuation (ASR), the whole message is one span: a
boundary is never invented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

SENT_END = re.compile(r"[^.!?]+[.!?]?", re.UNICODE)
CONFIRM = re.compile(
    r"^\s*(?:ok|okay|oui|ouais|d'accord|d’accord|valid[ée]|approuv[ée]|"
    r"confirm[ée]|entendu|parfait|c'est bon|ça marche|ca marche)\b", re.I)
#: A confirmation longer than this is a statement about something, not an ack.
CONFIRM_MAX_CHARS = 24
#: "ok tu la lances malik" starts like an ack and is an instruction. A real
#: ack adds no actor and no value.
CONFIRM_NEW_CONTENT = re.compile(
    r"\b(?:je|tu|il|elle|on|nous|vous|ils|elles)\b|\d", re.I)


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    text: str


def split_sentences(text: str) -> list[SentenceSpan]:
    text = text or ""
    out: list[SentenceSpan] = []
    for m in SENT_END.finditer(text):
        raw = m.group(0)
        if not raw.strip():
            continue
        start = m.start() + (len(raw) - len(raw.lstrip()))
        end = m.start() + len(raw.rstrip())
        out.append(SentenceSpan(start, end, text[start:end]))
    if not out:
        return [SentenceSpan(0, len(text), text)] if text else []
    return out


def is_confirmation(text: str) -> bool:
    text = (text or "").strip()
    return (bool(text) and len(text) <= CONFIRM_MAX_CHARS
            and bool(CONFIRM.search(text))
            and not CONFIRM_NEW_CONTENT.search(text))


def _words(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def split_long_spans(text: str, max_words: int = 70, window_words: int = 8,
                     chunker=None) -> list[SentenceSpan]:
    """Split spans that are too long to read in one go.

    Unpunctuated ASR gives one span, so a semantic chunker has no candidate
    sentences to compare and returns the whole turn (measured: 0 of 12 long
    turns split). Fixed word windows give it candidates; it merges them by
    similarity and the returned offsets are mapped back to the original text,
    so `text[start:end] == span.text` still holds.
    """
    spans = split_sentences(text)
    if chunker is None:
        return spans
    out: list[SentenceSpan] = []
    for span in spans:
        if len(span.text.split()) <= max_words:
            out.append(span)
            continue
        offs = [(span.start + a, span.start + b)
                for a, b in _words(span.text)]
        if len(offs) <= window_words:
            out.append(span)
            continue
        windows = [offs[i:i + window_words]
                   for i in range(0, len(offs), window_words)]
        cand = "\n".join(" ".join(text[a:b] for a, b in w) for w in windows)
        try:
            chunks = chunker(cand)
        except Exception:
            out.append(span)
            continue
        cursor = 0
        for ch in chunks:
            n = len(ch.text.split())
            if n <= 0:
                continue
            first, last = cursor, min(len(offs), cursor + n)
            a, b = offs[first][0], offs[last - 1][1]
            out.append(SentenceSpan(a, b, text[a:b]))
            cursor = last
    return out


def semantic_groups(
    text: str,
    topic_fn: Callable[[str], str | None] | None = None,
    intent_fn: Callable[[str], str] | None = None,
    spans: list[SentenceSpan] | None = None,
    merge: bool = True,
) -> list[list[SentenceSpan]]:
    """Group adjacent spans while topic and intent stay unchanged (§2).

    `merge=False` keeps every span its own unit. Long turns are split because
    they contain more than one statement; merging them back by topic would
    undo that.
    """
    spans = spans if spans is not None else split_sentences(text)
    if not spans:
        return []
    if not merge:
        return [[s] for s in spans]
    if topic_fn is None or len(spans) == 1:
        return _merge_confirmations(spans)

    groups: list[list[SentenceSpan]] = []
    cur_topic: str | None = None
    cur_intent: str | None = None
    for s in spans:
        if is_confirmation(s.text) and groups:
            groups[-1].append(s)
            continue
        topic = topic_fn(s.text)
        intent = intent_fn(s.text) if intent_fn else None
        if (groups and topic is not None and topic == cur_topic
                and intent == cur_intent):
            groups[-1].append(s)
            continue
        groups.append([s])
        cur_topic, cur_intent = topic, intent
    return groups


def _merge_confirmations(spans: list[SentenceSpan]) -> list[list[SentenceSpan]]:
    groups: list[list[SentenceSpan]] = []
    for s in spans:
        if groups and is_confirmation(s.text):
            groups[-1].append(s)
        else:
            groups.append([s])
    return groups


__all__ = ["SentenceSpan", "split_sentences", "semantic_groups",
           "is_confirmation", "CONFIRM"]
