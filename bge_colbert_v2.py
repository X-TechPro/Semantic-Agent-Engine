#!/usr/bin/env python3
"""ColBERTv2 residual compression for the fine-tuned BGE-M3 token index.

Keeps BGE-M3 as the encoder; quantises its projected 1024-d token matrix to
`n_centroids` centroids plus one residual sign bit per dimension. Scoring is
MaxSim over the reconstructed tokens.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
HYBRID_DIR = ART / "hybrid"


class CompressedColBERTIndex:
    def __init__(self, base, n_centroids: int = 2048, device: str = "cuda"):
        self.base = base
        self.device = device
        self.n_centroids = n_centroids
        self.path = HYBRID_DIR / f"{base._cache_path.stem}-v2-k{n_centroids}.pt"
        self.centroids: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None
        self.assign: torch.Tensor | None = None
        self.codes: np.ndarray | None = None
        self.tokens: torch.Tensor | None = None
        self.doc_index = base.doc_index.to(self.device) if base.doc_index is not None else None
        self.stats: dict[str, Any] = {}
        self._load_or_build()

    def _load_or_build(self):
        self.base.build()
        if self.path.exists():
            try:
                payload = torch.load(self.path, map_location="cpu", weights_only=False)
                if payload.get("ids") == self.base.ids and payload.get("raw_tokens") == int(self.base.tokens.size(0)):
                    self.centroids = payload["centroids"].float()
                    self.scale = payload["scale"].float()
                    self.assign = payload["assign"].long()
                    self.codes = payload["codes"].numpy()
                    self.stats = payload.get("stats", {})
                    self.tokens = self._decompress()
                    return
            except Exception:
                pass
        raw = self.base.tokens.float().cpu().numpy()
        k = min(self.n_centroids, max(2, len(raw) // 8))
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=1, random_state=0).fit(raw)
        centroids = km.cluster_centers_.astype(np.float32)
        assign = km.labels_.astype(np.int64)
        residual = raw - centroids[assign]
        scale = np.abs(residual).mean(0).astype(np.float32) + 1e-6
        signs = residual >= 0
        codes = np.packbits(signs, axis=1)
        self.centroids = torch.from_numpy(centroids).to(torch.float16)
        self.scale = torch.from_numpy(scale).to(torch.float16)
        self.assign = torch.from_numpy(assign)
        self.codes = codes
        self.tokens = self._decompress()
        raw_bytes = int(raw.nbytes)
        compressed_bytes = int(self.centroids.numpy().nbytes + self.assign.numpy().nbytes + codes.nbytes + self.scale.numpy().nbytes)
        self.stats = {"centroids": int(k), "bits_per_dim": 1, "raw_tokens": int(len(raw)),
                      "raw_bytes_fp32": raw_bytes, "compressed_bytes": compressed_bytes,
                      "ratio_fp32": round(raw_bytes / compressed_bytes, 2)}
        HYBRID_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"ids": self.base.ids, "raw_tokens": int(len(raw)), "centroids": self.centroids,
                    "scale": self.scale, "assign": self.assign,
                    "codes": torch.from_numpy(codes), "offsets": self.base.offsets,
                    "stats": self.stats}, self.path)
        (ART / "bge_colbert_v2_stats.json").write_text(json.dumps(self.stats, indent=2), encoding="utf-8")

    def _decompress(self) -> torch.Tensor:
        assert self.centroids is not None and self.scale is not None and self.assign is not None
        dim = self.centroids.size(1)
        bits = np.unpackbits(self.codes, axis=1)[:, :dim].astype(np.float32)
        signs = torch.from_numpy(bits * 2.0 - 1.0)
        tokens = self.centroids[self.assign].float() + signs * self.scale.float().unsqueeze(0)
        tokens = F.normalize(tokens, dim=-1).to(torch.float16)
        return tokens.to(self.device)

    @torch.inference_mode()
    def score(self, query: str) -> np.ndarray:
        base = self.base
        base._ensure_model()
        enc = base._tokenizer([query], truncation=True, max_length=64, return_tensors="pt").to(base.device)
        hidden = base._model(**enc).last_hidden_state.float()
        mask = enc["attention_mask"].bool()
        if base.projection is not None:
            hidden = hidden[:, 1:]
            mask = mask[:, 1:]
            hidden = base.projection(hidden)
        q = F.normalize(hidden[0][mask[0]], dim=-1).to(torch.float16)
        sims = q @ self.tokens.T
        maxima = torch.full((q.size(0), len(base.ids)), -1e4, device=self.device, dtype=sims.dtype)
        maxima.scatter_reduce_(1, self.doc_index.unsqueeze(0).expand(q.size(0), -1), sims,
                               reduce="amax")
        return maxima.sum(0).float().cpu().numpy()
