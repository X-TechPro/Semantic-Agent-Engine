"""Cross-encoder link baseline over the same candidate pool as the scorer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Candidate:
    turn: int
    text: str
    speaker: str = ""
    timestamp: str = ""


class CrossEncoderLinker:
    def __init__(self, model_id: str = "BAAI/bge-reranker-v2-m3",
                 device: str = "cuda"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, device=device)

    @staticmethod
    def _meta(c: Candidate) -> str:
        bits = [c.speaker or "-", c.timestamp or "-"]
        return "[" + "|".join(bits) + "] " + c.text

    @staticmethod
    def _context(turns: list[dict], i: int, radius: int = 1) -> str:
        lo, hi = max(0, i - radius), min(len(turns), i + radius + 1)
        return " || ".join(str(turns[k].get("text") or "") for k in range(lo, hi))

    def scores(self, query: str, candidates: list[Candidate], turns: list[dict],
               style: str = "plain") -> list[float]:
        if style == "plain":
            pairs = [(query, c.text) for c in candidates]
        elif style == "metadata":
            q = query
            pairs = [(q, self._meta(c)) for c in candidates]
        elif style == "context":
            qctx = query
            pairs = [(qctx, self._context(turns, c.turn)) for c in candidates]
        elif style == "metadata_context":
            qctx = query
            pairs = [(qctx, self._meta(c) + " || " + self._context(turns, c.turn))
                     for c in candidates]
        else:
            raise ValueError(style)
        arr = self.model.predict(pairs, batch_size=32, show_progress_bar=False)
        return [float(x) for x in arr]

    def rank(self, query: str, candidates: list[Candidate], turns: list[dict],
             style: str = "plain") -> list[int]:
        s = self.scores(query, candidates, turns, style)
        return [c.turn for _, c in sorted(zip(s, candidates), key=lambda x: -x[0])]


__all__ = ["Candidate", "CrossEncoderLinker"]
