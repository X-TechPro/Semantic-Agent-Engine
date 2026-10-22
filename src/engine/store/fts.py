"""FTS5 lexical index over **sentences** (2.4 / 6.6).

`bm25s` is the primary lexical backend (it is what 6.6 asks to ablate against),
but SQLite FTS5 is kept in the store so the comparison row exists without an
extra process, and so `/query` can run on a store with no bm25s index built.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, strip_accents: bool = True, stem: bool = False) -> list[str]:
    """French-friendly analyser: lowercase, optional accent folding, keep digits.

    6.6: "keep numbers and dates as tokens ('22', 'novembre')". Light stemming is
    an ablation knob (suffix stripper), off by default.
    """
    if strip_accents:
        import unicodedata

        text = "".join(c if not unicodedata.combining(c) else ""
                       for c in unicodedata.normalize("NFD", text))
    toks = [t.lower() for t in _TOKEN.findall(text)]
    if stem:
        toks = [_light_stem(t) for t in toks]
    return toks


_SUFFIXES = ("issements", "issement", "ations", "ation", "ements", "ement",
             "euses", "euse", "istes", "iste", "antes", "ante", "és", "ée",
             "ées", "era", "iez", "ions", "ais", "ait", "ant", "és", "er",
             "ez", "ée", "es", "ée", "s")


def _light_stem(tok: str) -> str:
    for s in _SUFFIXES:
        if len(tok) > len(s) + 3 and tok.endswith(s):
            return tok[: -len(s)]
    return tok


def fts_quote(text: str) -> str:
    """Escape a user query for FTS5 MATCH: tokenise and AND the terms."""
    toks = [t for t in tokenize(text) if t]
    if not toks:
        return ""
    return " AND ".join(f'"{t}"' for t in toks)


def fts_quote_or(text: str) -> str:
    toks = [t for t in tokenize(text) if t]
    if not toks:
        return ""
    return " OR ".join(f'"{t}"' for t in toks)


@dataclass(slots=True)
class LexicalHit:
    unit_id: str
    score: float
    evidence: str


class FtsIndex:
    def __init__(self, db: sqlite3.Connection):
        self.db = db

    def sync_from_sentences(self) -> int:
        """Rebuild the FTS5 table from `sentences` (deterministic)."""
        self.db.execute("DELETE FROM sentences_fts")
        self.db.execute(
            """INSERT INTO sentences_fts(text, unit_id, sent_id)
               SELECT text, unit_id, sent_id FROM sentences""")
        self.db.commit()
        return self.db.execute("SELECT COUNT(*) FROM sentences_fts").fetchone()[0]

    def search(self, query: str, k: int = 50, *, require_all: bool = True) -> list[LexicalHit]:
        q = fts_quote(query) if require_all else fts_quote_or(query)
        if not q:
            return []
        rows = self.db.execute(
            """SELECT unit_id, text, bm25(sentences_fts) AS rank FROM sentences_fts
               WHERE sentences_fts MATCH ? ORDER BY rank LIMIT ?""", (q, k)).fetchall()
        out: list[LexicalHit] = []
        for r in rows:
            # bm25() returns *negative* relevance (lower is better); flip it so
            # every lexical backend in this repo yields "higher is better".
            out.append(LexicalHit(unit_id=r[0], score=float(-r[2]), evidence=r[1]))
        return out
