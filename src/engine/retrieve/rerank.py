"""Reranker (6.9) — three backends behind one interface.

  * ``none`` (default)      — no reranking; RRF order is kept. This is Baseline A.
  * ``overlap``             — deterministic lexical overlap + metadata prefix
                              bonus. No model, usable in CI and in the "LLM-off"
                              ablation, and it is what the unit tests exercise.
  * ``cross_encoder``       — bge-reranker-v2-m3 (Apache-2.0, verified in
                              docs/MODELCARDS.md) through
                              ``sentence_transformers.CrossEncoder``.

6.9 asks for a *metadata prefix* on the rerank input
(``[decision|confirmed|2026-10-05|Marie] text``); that is implemented here and
shared by both model-backed paths.
"""

from __future__ import annotations

from engine.schemas import Unit


def metadata_prefix(unit: Unit) -> str:
    parts = [unit.type or "other", unit.status or "mention",
             unit.source_date.isoformat(), unit.speaker or "-"]
    return "[" + "|".join(parts) + "] "


def build_reranker(cfg):
    backend = cfg["retrieve"]["rerank"]["backend"]
    candidates = int(cfg["retrieve"]["rerank"]["candidates"])
    if backend in (None, "", "none"):
        return None
    if backend == "overlap":
        return OverlapReranker(candidates=candidates)
    if backend in ("cross_encoder", "bge-reranker-v2-m3"):
        return CrossEncoderReranker(model_id=cfg["retrieve"]["rerank"].get("model_id")
                                    or "BAAI/bge-reranker-v2-m3",
                                    candidates=candidates,
                                    device=cfg.get("device", "auto"))
    raise ValueError(f"unknown rerank backend {backend!r}")


class OverlapReranker:
    name = "overlap"

    def __init__(self, candidates: int = 30):
        self.candidates = candidates

    def rerank(self, query: str, units: list[Unit]) -> list[tuple[str, float]]:
        from engine.store.fts import tokenize

        qt = set(tokenize(query))
        out: list[tuple[str, float]] = []
        for u in units[: self.candidates]:
            ut = set(tokenize(u.text))
            jac = len(qt & ut) / len(qt | ut) if (qt or ut) else 0.0
            # a current, confirmed unit is a better answer than a mention
            prior = 1.0 if (u.is_current and u.status == "confirmed") else 0.8
            out.append((u.unit_id, round(jac * prior, 6)))
        out.sort(key=lambda t: (-t[1], t[0]))
        return out


class CrossEncoderReranker:
    name = "cross_encoder"

    def __init__(self, model_id: str, candidates: int = 30, device: str = "auto"):
        from sentence_transformers import CrossEncoder

        if device in (None, "auto"):
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.model = CrossEncoder(model_id, device=device)
        self.candidates = candidates

    def rerank(self, query: str, units: list[Unit]) -> list[tuple[str, float]]:
        pairs = [(query, metadata_prefix(u) + u.text) for u in units[: self.candidates]]
        scores = self.model.predict(pairs, show_progress_bar=False)
        out = [(u.unit_id, float(s)) for u, s in zip(units[: self.candidates], scores)]
        out.sort(key=lambda t: (-t[1], t[0]))
        return out
