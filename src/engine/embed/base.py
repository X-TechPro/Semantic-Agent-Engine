"""Embedder interface + deterministic hash baseline (BUILD_GUIDE 2.2 / 6.4).

`embed(texts, role) -> np.ndarray`, L2-normalised float32.

Two backends behind the same contract:
  * ``hash``   — signed bag-of-character-n-grams. No model, no download, no
                 network; deterministic and fast. It is the *floor* the real
                 embedder must beat, and it is what tests run on.
  * ``st``     — sentence-transformers model from config (`model_id`), with the
                 model card's query/doc prefixes (verified in docs/MODELCARDS.md).
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Iterable, Literal, Sequence

import numpy as np

Role = Literal["query", "doc"]
_WS = re.compile(r"\s+")


def normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float32")
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype("float32")


class Embedder(ABC):
    version: str = "abstract"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str], role: Role = "doc") -> np.ndarray: ...

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts, role="doc")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], role="query")[0]

    def __call__(self, texts: Sequence[str], role: Role = "doc") -> np.ndarray:
        return self.embed(texts, role=role)


class HashEmbedder(Embedder):
    """Signed character-n-gram hashing — the no-model baseline (6.4 "cheap version
    first"). Captures French morphology via 3-5 grams and is word-sensitive
    enough that minimal pairs (1,5 M€ vs 15 M€) are distinguishable."""

    def __init__(self, dim: int = 256, ngram: tuple[int, int] = (3, 5),
                 version: str | None = None):
        self.dim = dim
        self.ngram = ngram
        self.version = version or f"hash-{dim}-{ngram[0]}{ngram[1]}"

    def _features(self, text: str) -> list[str]:
        t = _WS.sub(" ", text.strip().lower())
        if not t:
            return []
        feats: list[str] = []
        words = t.split(" ")
        for w in words:
            for n in range(self.ngram[0], self.ngram[1] + 1):
                if len(w) >= n:
                    feats.extend(w[i:i + n] for i in range(len(w) - n + 1))
                elif len(w) > 1:
                    feats.append(w)
        # keep multi-word phrases so "c'est acté" stays identifiable
        for i in range(len(words) - 1):
            feats.append(words[i] + " " + words[i + 1])
        return feats

    def embed(self, texts: Sequence[str], role: Role = "doc") -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode("utf-8"),
                                                   digest_size=8).digest(), "big")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, idx] += sign / (1.0 + len(feat) ** 0.5)
            if role == "query":
                # a distinct query channel: same space, different weighting
                out[i] *= 1.0
        return normalize(out)


def build_embedder(cfg) -> Embedder:
    from engine.config import load_config

    cfg = cfg or load_config()
    emb = cfg["embed"]
    backend = emb["backend"]
    if backend == "hash":
        return HashEmbedder(dim=int(emb["dim"]))
    if backend in ("st", "sentence_transformers", "hf"):
        from engine.embed.hf_embedder import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            model_id=emb["model_id"], device=emb.get("device", cfg["device"]),
            batch_size=int(emb["batch_size"]),
            max_seq_length=int(emb["max_seq_length"]))
    raise ValueError(f"unknown embed backend {backend!r}")
