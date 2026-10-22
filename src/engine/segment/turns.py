"""Speaker-turn + sentence segmentation with owned character offsets.

Why this module exists rather than calling chunklet directly (6.1): the engine
needs `Unit.sentences` to be *exact* character spans, and BUILD_GUIDE 6.1
acceptance is "100% span reconstruction". Measured behaviour of chunklet 3.0.1
`SentenceSplitter(lang="fr")` (docs/DECISIONS.md D-003):

    "M. Dupont est arrivé. env. 30 personnes étaient là. etc. On a attendu."
      -> ['M. Dupont est arrivé. env.', '30 personnes étaient là. etc.', 'On a attendu.']

i.e. the abbreviation `env.` is glued to the *previous* sentence. We repair the
known classes and always recompute offsets ourselves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# French abbreviations that must not end a sentence when the next fragment
# starts lower-case / with a digit.
ABBREVIATIONS = {
    "M.", "MM.", "Dr.", "Me.", "Mlle.", "etc.", "env.", "p. ex.", "ex.",
    "J.-C.", "av. J.-C.", "etc.", "S.A.", "S.A.R.L.", "SA.", "cf.", "i.e.",
    "a.m.", "p.m.", "St.", "Ste.", "tél.", "no.", "n°", "vol.", "bl.",
}
_ABBR_TAIL = re.compile(
    r"(?:^|\s)((?:[A-Za-zÀ-ÿ]{1,6}\.){1,3})\s*$"
)
_SENTENCE_END = re.compile(r"[.!?…]+[\"'»)\]]?\s*$")
_WORD = re.compile(r"[^\s]+")


def count_tokens(text: str) -> int:
    """Cheap deterministic token count (whitespace words). See D-003: chunklet
    requires a token_counter for token-based chunking; we standardise on this so
    unit ids and chunk sizes are reproducible without a tokenizer download."""
    return len(_WORD.findall(text))


def _looks_like_abbreviation(tail: str) -> bool:
    t = tail.strip()
    if t in ABBREVIATIONS:
        return True
    # single initial: "J.", "M."
    if re.fullmatch(r"[A-Za-zÀ-ÿ]{1,3}\.", t):
        return True
    if re.fullmatch(r"(?:[A-Za-zÀ-ÿ]{1,4}\.){1,2}", t):
        return True
    return False


@dataclass(slots=True)
class Sentence:
    start: int
    end: int
    text: str


def repair_glued_abbreviations(sents: list[str]) -> list[str]:
    """Move a trailing abbreviation into the following sentence when the
    following sentence does not start with a capital (the chunklet failure)."""
    out: list[str] = []
    i = 0
    while i < len(sents):
        cur = sents[i]
        nxt = sents[i + 1] if i + 1 < len(sents) else None
        m = _ABBR_TAIL.search(cur)
        if m and nxt is not None:
            tail = m.group(1)
            rest = cur[: m.start(1)].rstrip()
            starts_open = bool(nxt) and (nxt[0].islower() or nxt[0].isdigit())
            if rest and _looks_like_abbreviation(tail) and starts_open:
                out.append(rest)
                sents[i + 1] = f"{tail} {nxt.lstrip()}"
                i += 1
                continue
        out.append(cur)
        i += 1
    return out


def _iter_chunklet_sentences(text: str, lang: str = "fr") -> list[str]:
    from chunklet import SentenceSplitter

    ss = SentenceSplitter(lang=lang)
    return [s for s in ss.split_text(text) if s.strip()]


def split_sentences(text: str, lang: str = "fr") -> list[Sentence]:
    """Return exact character spans for each sentence of `text`."""
    raw = _iter_chunklet_sentences(text, lang)
    raw = repair_glued_abbreviations(raw)
    return align_spans(text, raw)


def align_spans(text: str, surfaces: list[str]) -> list[Sentence]:
    """Map returned sentence strings back onto `text` as disjoint, ordered
    character spans. chunklet may normalise whitespace, so we search forward
    from the cursor and, failing that, fall back to a whitespace-insensitive
    match that still yields exact offsets."""
    out: list[Sentence] = []
    cursor = 0
    for surf in surfaces:
        s = surf.strip()
        if not s:
            continue
        idx = text.find(s, cursor)
        if idx == -1:
            idx = _fuzzy_find(text, s, cursor)
            if idx is None:
                # Last resort: keep the surface but anchor at the cursor; the
                # reconstruction test will surface this as a real failure.
                idx = min(cursor, len(text))
                span_text = text[idx:idx + len(s)]
                out.append(Sentence(idx, idx + len(span_text), span_text))
                cursor = idx + len(span_text)
                continue
        out.append(Sentence(idx, idx + len(s), text[idx:idx + len(s)]))
        cursor = idx + len(s)
    return out


_WS = re.compile(r"\s+")


def _fuzzy_find(text: str, needle: str, start: int) -> int | None:
    """Find `needle` ignoring whitespace differences; returns the exact offset."""
    compact = _WS.sub("", text)
    target = _WS.sub("", needle)
    if not target:
        return None
    pos = compact.find(target, _WS.sub("", text[:start]).__len__())
    if pos == -1:
        pos = compact.find(target)
        if pos == -1:
            return None
    # map compact index back to original index
    seen = 0
    for i, ch in enumerate(text):
        if not ch.isspace():
            if seen == pos:
                return i
            seen += 1
    return None


def split_by_token_window(text: str, max_tokens: int = 40,
                           min_tokens: int = 8) -> list[Sentence]:
    """ASR fallback (6.1): unpunctuated turns split on a fixed token window."""
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    if not words:
        return []
    out: list[Sentence] = []
    i = 0
    while i < len(words):
        j = min(i + max_tokens, len(words))
        if j < len(words) and (len(words) - j) < min_tokens:
            j = len(words)
        start = words[i][0]
        end = words[j - 1][1]
        out.append(Sentence(start, end, text[start:end]))
        i = j
    return out


def is_unpunctuated(text: str) -> bool:
    return not bool(re.search(r"[.!?…]", text))


def has_sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END.search(text.strip()))
