#!/usr/bin/env python3
"""Rexu v3: pplx contextual tokens + late interaction.

pplx-context(full meeting) -> contextual tokens -> Dense(1024->128) -> MaxSim.
No BGE-M3. Rexu v3.1 is archived under archive/rexu_v3_1/.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
HF = ART / "hf_cache"
MODEL = "perplexity-ai/pplx-embed-context-v1-0.6b"
LATE_HEAD = ART / "pplx_v3_late_head_multi.pt"


class LateHead(nn.Module):
    def __init__(self, dim: int = 1024, out: int = 128):
        super().__init__(); self.fc = nn.Linear(dim, out, bias=False)
    def forward(self, x): return F.normalize(self.fc(x), dim=-1)


class RexuV3Retriever:
    def __init__(self, engine, slug: str, max_tokens: int = 15000):
        self.engine = engine
        self.slug = slug
        self.max_tokens = max_tokens
        ds = engine._get_dataset(slug)
        self.ds = ds
        self.records = ds["records"]
        self.meetings = ds["meetings"]
        self.device = engine.device
        self._model = None
        self._tok = None
        self._late = None
        self._docs_mean: dict[str, torch.Tensor] | None = None
        self._docs_tokens: dict[str, torch.Tensor] | None = None
        suffix = "" if max_tokens == 15000 else f"_w{max_tokens}"
        self._cache = ART / f"pplx_v3_docs_{slug}{suffix}.pt"

    def _ensure(self):
        if self._model is not None:
            return
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(
            MODEL, trust_remote_code=True, cache_dir=str(HF)).to(self.device).eval()
        self._tok = self._model.tokenizer
        if self._tok.sep_token is None:
            raise RuntimeError("pplx tokenizer has no separator token")
        self._late = LateHead().to(self.device)
        self._late.load_state_dict(torch.load(LATE_HEAD, map_location="cpu",
                                               weights_only=False)["late"])
        self._late.eval()
        if self._docs_mean is None:
            self._load_or_build_docs()

    @torch.inference_mode()
    def _encode_one(self, text: str):
        text = text + self._tok.sep_token
        enc = self._tok([text], padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(self.device)
        hidden = self._model(**enc).last_hidden_state.float()[0].cpu()
        mask = enc["attention_mask"][0].cpu().bool()
        sep_id = self._tok.sep_token_id
        seps = ((enc["input_ids"][0].cpu() == sep_id) & mask).nonzero(as_tuple=True)[0].tolist()
        end = seps[0] if seps else int(mask.sum())
        toks = hidden[:end][mask[:end]]
        if toks.size(0) == 0:
            toks = torch.zeros(1, hidden.size(-1))
        return F.normalize(toks.mean(0), dim=-1), F.normalize(toks, dim=-1)

    @torch.inference_mode()
    def _encode_meeting(self, chunks: list[str]):
        text = self._tok.sep_token.join(chunks)
        enc = self._tok([text], padding=True, truncation=True, max_length=max(16384, self.max_tokens + 1024),
                        return_tensors="pt").to(self.device)
        hidden = self._model(**enc).last_hidden_state.float()[0].cpu()
        ids = enc["input_ids"][0].cpu(); mask = enc["attention_mask"][0].cpu().bool()
        sep_id = self._tok.sep_token_id
        seps = ((ids == sep_id) & mask).nonzero(as_tuple=True)[0].tolist()
        segments, start = [], 0
        for pos in seps:
            segments.append((start, pos)); start = pos + 1
        segments.append((start, int(mask.sum())))
        means, toks = [], []
        for a, b in segments:
            t = hidden[a:b][mask[a:b]]
            if t.size(0) == 0:
                t = torch.zeros(1, hidden.size(-1))
            toks.append(F.normalize(t, dim=-1))
            means.append(F.normalize(t.mean(0), dim=-1))
        return torch.stack(means), toks

    def _windows(self, chunks: list[str]) -> list[list[str]]:
        max_tokens = self.max_tokens
        """Split one meeting into consecutive windows that fit the encoder's 16k limit.

        Short meetings stay a single window (identical to the previous behaviour).
        Long meetings (e.g. QMSum, up to ~1.4k turns) are split at turn boundaries; the
        later windows lose earlier-window context but no turn is silently dropped.
        """
        lengths = [len(self._tok(self.records[c]["encoder_text"] + self._tok.sep_token)["input_ids"])
                   for c in chunks]
        if sum(lengths) <= max_tokens:
            return [chunks]
        windows, cur, cur_len = [], [], 0
        for c, n in zip(chunks, lengths):
            if cur and cur_len + n > max_tokens:
                windows.append(cur); cur, cur_len = [], 0
            cur.append(c); cur_len += n
        if cur:
            windows.append(cur)
        return windows

    def _load_or_build_docs(self):
        if self._cache.exists():
            payload = torch.load(self._cache, map_location="cpu", weights_only=False)
            if payload.get("ids") == list(self.records):
                self._docs_mean = payload["mean"]
                self._docs_tokens = payload["tokens"]
                return
        means, tokens = {}, {}
        for mid, chunks in self.ds["by_meeting"].items():
            for window in self._windows(chunks):
                texts = [self.records[c]["encoder_text"] for c in window]
                m, ts = self._encode_meeting(texts)
                if len(m) != len(window):
                    raise RuntimeError(f"{mid}: encoded {len(m)} segments for {len(window)} chunks")
                for c, mv, tv in zip(window, m, ts):
                    means[c] = mv
                    tokens[c] = tv
        if len(means) != len(self.records):
            raise RuntimeError(f"doc encoding covered {len(means)} of {len(self.records)} chunks")
            if len(means) % 600 == 0:
                print(f"  rexu v3 docs: {len(means)}/{len(self.records)}", flush=True)
        self._docs_mean = means
        self._docs_tokens = tokens
        torch.save({"ids": list(self.records), "mean": means, "tokens": tokens}, self._cache)

    @torch.inference_mode()
    def _score(self, query: str, method: str, candidates: list[str]):
        self._ensure()
        _, qtokens = self._encode_one(query)
        qv = self._late(qtokens.to(self.device))
        dv = [self._late(self._docs_tokens[c].to(self.device)) for c in candidates]
        return torch.tensor([(qv @ d.T).max(dim=1).values.sum() for d in dv]).cpu().numpy()

    def search(self, query: str, top_k: int = 10, meeting_id: str | None = None,
               method: str = "rexu_v3"):
        candidates = self.ds["by_meeting"].get(meeting_id, []) if meeting_id else list(self.records)
        scores = self._score(query, method, candidates)
        order = np.argsort(-scores)[:top_k]
        results = []
        for rank, i in enumerate(order, 1):
            cid = candidates[i]; r = self.records[cid]
            results.append({"rank": rank, "chunk_id": cid, "score": float(scores[i]),
                            "meeting_id": r["meeting_id"], "meeting_title": r["meeting_title"],
                            "meeting_date": r["meeting_date"], "username": r["username"],
                            "timestamp": r["timestamp"], "message": r["message_text"],
                            "source_json": r["raw"]})
        explanation = "Rexu v3 · pplx-context full-meeting tokens + 1024→128 late interaction (MaxSim); no BGE-M3"
        return {"query": query, "method": method, "explanation": explanation,
                "candidate_count": len(candidates), "results": results}


_lock = threading.Lock()
_retrievers: dict[tuple[int, str], RexuV3Retriever] = {}


def get_rexu_v3(engine, slug: str) -> RexuV3Retriever:
    key = (id(engine), slug)
    with _lock:
        if key not in _retrievers:
            _retrievers[key] = RexuV3Retriever(engine, slug)
        return _retrievers[key]
