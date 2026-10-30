"""sentence-transformers embedder behind the 2.2 contract (6.4).

Prefix conventions are taken from docs/MODELCARDS.md, which the M0 verifier
regenerates from the live model cards — never hard-coded from memory.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from engine.embed.base import Embedder, Role, normalize

# Verified prefixes (see docs/MODELCARDS.md). Empty string = no instruction.
KNOWN_PREFIXES: dict[str, dict[str, str]] = {
    "intfloat/multilingual-e5-base": {"query": "query: ", "doc": "passage: "},
    "intfloat/multilingual-e5-large": {"query": "query: ", "doc": "passage: "},
    "BAAI/bge-m3": {"query": "", "doc": ""},
    "BAAI/bge-m3-sparse": {"query": "", "doc": ""},
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {"query": "", "doc": ""},
    "Qwen/Qwen3-Embedding-0.6B": {"query": "Instruct: Given a web search query, "
                                "retrieve relevant passages that answer the query\nQuery: ",
                              "doc": ""},
}


def _global_hf_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "huggingface"


def guard_hf_cache() -> None:
    """Never keep a second copy of the HF cache inside the repository.

    A repo-local HF_HOME (models/hf, 8.7 GB once) duplicates the user's cache.
    The environment is read by huggingface_hub at import time, so this runs
    before sentence_transformers is imported and redirects the variable when it
    points inside the checkout. One warning, once per process.
    """
    target = os.environ.get("HF_HOME") or os.environ.get("HF_HUB_CACHE")
    if not target:
        return
    root = Path(__file__).resolve().parents[3]
    try:
        inside = root in Path(target).resolve().parents
    except OSError:
        inside = False
    if not inside:
        return
    fallback = _global_hf_home()
    os.environ["HF_HOME"] = str(fallback)
    os.environ.pop("HF_HUB_CACHE", None)
    print(f"hf cache: {target} is inside the repo; using {fallback} "
          f"(unset HF_HOME to silence this)", file=sys.stderr)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_id: str, device: str | None = "auto",
                 batch_size: int = 64, max_seq_length: int | None = 512,
                 prefixes: dict[str, str] | None = None):
        if not model_id:
            raise ValueError("embed.model_id must be set for the st backend")
        guard_hf_cache()
        from sentence_transformers import SentenceTransformer

        if device in (None, "auto"):
            device = _pick_device()
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_id, device=device,
                                         model_kwargs={"trust_remote_code": False})
        if max_seq_length:
            try:
                self.model.max_seq_length = max_seq_length
            except Exception:
                pass
        self.prefixes = prefixes or _prefixes_for(model_id)
        # sentence-transformers 5.x renamed this; the old name emits a
        # FutureWarning on every construction (seen in the e5 bake-off log).
        dim_fn = (getattr(self.model, "get_embedding_dimension", None)
                  or self.model.get_sentence_embedding_dimension)
        self.dim = int(dim_fn())
        try:
            info = self.model.sentence_transformer_config or {}
        except Exception:
            info = {}
        self.version = f"{model_id}@{self.dim}d"

    def embed(self, texts: Sequence[str], role: Role = "doc") -> np.ndarray:
        prefix = self.prefixes.get(role, "")
        prepared = [f"{prefix}{t}" for t in texts]
        vecs = self.model.encode(
            prepared, batch_size=self.batch_size, normalize_embeddings=False,
            convert_to_numpy=True, show_progress_bar=False,
            prompt=prefix or None,  # ST v5 API; ignored gracefully below
        ) if _st_supports_prompt() else self.model.encode(
            prepared, batch_size=self.batch_size, normalize_embeddings=False,
            convert_to_numpy=True, show_progress_bar=False)
        return normalize(np.asarray(vecs, dtype="float32"))


def _prefixes_for(model_id: str) -> dict[str, str]:
    """Prefixes for a hub id OR for a locally fine-tuned directory.

    `KNOWN_PREFIXES` is keyed by hub id, so pointing `embed.model_id` at a
    fine-tuned local path used to fall through to `{"query": "", "doc": ""}` —
    Qwen3's `Instruct:` prompt would vanish and the benchmark would measure a
    different model than the engine runs. A fine-tune writes this sidecar next to
    its weights, so the convention travels with the artefact.
    """
    import json
    from pathlib import Path

    from engine.config import repo_root

    raw = Path(model_id)
    for cand in (raw, repo_root() / raw):
        side = cand / "sae_prefixes.json"
        if side.is_file():
            try:
                data = json.loads(side.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                break
            return {"query": data.get("query", ""), "doc": data.get("doc", "")}
    return KNOWN_PREFIXES.get(model_id, {"query": "", "doc": ""})


def _st_supports_prompt() -> bool:
    import inspect

    from sentence_transformers import SentenceTransformer

    try:
        return "prompt" in inspect.signature(SentenceTransformer.encode).parameters
    except (ValueError, TypeError):  # pragma: no cover
        return False


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
