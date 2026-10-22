"""Code chunking via chunklet's CodeChunker, with tree-sitter when available.

6.1: "one function/method/class = one chunk; file-level docstring = one chunk.
No ML." tree-sitter grammars are optional (`pip install tree-sitter-python`), so
the default path is chunklet's language-agnostic structural chunker.
"""

from __future__ import annotations

import re

from engine.segment import turns as T

_DEF_RE = re.compile(
    r"^[ \t]*(?:async[ \t]+)?(?:def|class|function|func|sub|type|interface|"
    r"struct|enum|impl|fn)[ \t]+(?P<name>[A-Za-z_À-ÿ][\wÀ-ÿ]*)",
    re.MULTILINE,
)


def chunk_code(code: str, language: str | None = None, max_tokens: int = 256
               ) -> list[tuple[str, dict]]:
    """Return (text, metadata) chunks. Deterministic, no model, no network."""
    if not code or not code.strip():
        return []
    try:
        from chunklet import CodeChunker

        chunks = CodeChunker(max_tokens=max_tokens,
                             token_counter=T.count_tokens).chunk_text(
            code, base_metadata={"language": language or "unknown"})
        out: list[tuple[str, dict]] = []
        for c in chunks:
            text = str(c["chunk"]) if isinstance(c, dict) and "chunk" in c else str(c)
            if not text.strip():
                continue
            meta = {"symbol": _symbol_of(text), "source": "chunklet"}
            if language:
                meta["language"] = language
            out.append((text.strip("\n"), meta))
        if out:
            return out
    except Exception:
        pass  # chunklet unsupported/failed -> regex fallback below
    return _fallback_split(code, max_tokens)


def _symbol_of(text: str) -> str | None:
    """Name of the first definition in a chunk, or None if it declares none.

    Read from the capture group, NOT from the text after the match: `_DEF_RE`
    includes the name inside what it matches, so `text[m.end():]` started at the
    parameter list and the lookup failed on every real input (measured: `def
    alpha(a, b)`, `class Beta`, `async def gamma`, `func delta` all returned
    None), silently stripping the symbol from every code chunk.
    """
    m = _DEF_RE.search(text)
    return m.group("name") if m else None


def _fallback_split(code: str, max_tokens: int) -> list[tuple[str, dict]]:
    blocks = re.split(r"\n(?=(?:async[ \t]+)?(?:def|class|function|func)\s)", code)
    out: list[tuple[str, dict]] = []
    for b in blocks:
        b = b.strip("\n")
        if not b:
            continue
        if T.count_tokens(b) <= max_tokens:
            out.append((b, {"symbol": _symbol_of(b), "source": "regex"}))
        else:  # oversize block: keep line windows so offsets stay contiguous
            lines = b.split("\n")
            cur: list[str] = []
            n = 0
            for ln in lines:
                cur.append(ln)
                n += T.count_tokens(ln)
                if n >= max_tokens:
                    out.append(("\n".join(cur), {"symbol": None, "source": "regex-window"}))
                    cur, n = [], 0
            if cur:
                out.append(("\n".join(cur), {"symbol": None, "source": "regex-window"}))
    return out
