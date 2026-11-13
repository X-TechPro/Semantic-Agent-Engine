"""ColBERTv2-style late-chunked multi-vector index.

Late chunking (Günther et al., 2024): encode the whole meeting once, then derive
each chunk's token vectors from the meeting-level contextual states instead of
encoding each chunk in isolation. ColBERTv2 (Santhanam et al., 2022): quantise
every token vector to a centroid plus one bit per residual dimension.

The backbone is multilingual BGE-M3, not the English MS-MARCO ColBERTv2
checkpoint, because the corpus is French. Scoring is the ColBERT MaxSim
function over the reconstructed late-chunked vectors. PLAID's inverted index is
not implemented; scoring is exhaustive over the compressed token matrix, and
the compression ratios below refer to the serialised token codes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ART = ROOT / "artifacts"
HYBRID_DIR = ART / "hybrid"
HYBRID_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VERSION = "colbertv2-late-v1"


class LateChunkedColBERTv2:
    def __init__(self, records: dict[str, dict[str, Any]], meetings: dict[str, dict[str, Any]],
                 slug: str, device: str = "cuda", model_name: str = "BAAI/bge-m3",
                 max_len: int = 8192, n_centroids: int = 1024,
                 late: bool = True, compress: bool = True):
        self.records = records
        self.meetings = meetings
        self.ids = list(records)
        self.id_to_pos = {cid: i for i, cid in enumerate(self.ids)}
        self.slug = slug
        self.device = device if torch.cuda.is_available() and device != "cpu" else "cpu"
        self.model_name = model_name
        self.max_len = max_len
        self.n_centroids = n_centroids
        self.late = late
        self.compress = compress
        self._model = None
        self._tokenizer = None
        self.tokens: torch.Tensor | None = None
        self.offsets: torch.Tensor | None = None
        self.doc_index: torch.Tensor | None = None
        self.late_vecs: torch.Tensor | None = None
        self.stats: dict[str, Any] = {}
        mode = f"{'late' if late else 'chunk'}-{'q' if compress else 'raw'}"
        self._cache_path = HYBRID_DIR / f"{slug}-colbertv2-{mode}-L{max_len}-K{n_centroids}.pt"
        self._load_cache()

    def _load_cache(self):
        if not self._cache_path.exists():
            return
        try:
            payload = torch.load(self._cache_path, map_location="cpu", weights_only=False)
        except Exception:
            return
        if (payload.get("version") != CACHE_VERSION or payload.get("ids") != self.ids
                or payload.get("model") != self.model_name):
            return
        if payload.get("compressed", True):
            self.tokens, self.offsets, self.late_vecs = self._decompress(payload)
        else:
            self.tokens = payload["tokens"].to(torch.float16)
            self.offsets = payload["offsets"].to(torch.long)
            self.late_vecs = payload["late_vecs"].to(torch.float16)
        self.doc_index = torch.repeat_interleave(
            torch.arange(len(self.ids)), self.offsets[1:] - self.offsets[:-1])
        self.stats = payload.get("stats", {})

    def _ensure_model(self):
        if self._model is not None:
            return
        import os
        os.environ.setdefault("HF_HOME", str(ART / "hf_cache"))
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=str(ART / "hf_cache"))
        self._model = AutoModel.from_pretrained(
            self.model_name, cache_dir=str(ART / "hf_cache"), dtype=torch.float16
        ).to(self.device).eval()

    @staticmethod
    def _pack_signs(residual: torch.Tensor) -> np.ndarray:
        signs = (residual >= 0).to(torch.uint8).cpu().numpy()
        return np.packbits(signs, axis=1)

    @staticmethod
    def _decompress(payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assign = payload["assign"].to(torch.long)
        centroids = payload["centroids"].to(torch.float32)
        scale = payload["scale"].to(torch.float32)
        dim = centroids.size(1)
        bits = np.unpackbits(payload["codes"].numpy(), axis=1)[:, :dim].astype(np.float32)
        signs = torch.from_numpy(bits * 2.0 - 1.0)
        tokens = centroids[assign] + signs * scale.unsqueeze(0)
        tokens = F.normalize(tokens, dim=-1).to(torch.float16)
        return tokens, payload["offsets"].to(torch.long), payload["late_vecs"].to(torch.float16)

    def _meeting_windows(self, meeting_id: str, budget: int) -> list[tuple[str, dict[str, tuple[int, int]], list[str]]]:
        """Split a meeting into consecutive windows whose documents fit `budget` tokens.

        Each window repeats the meeting title. A meeting that fits is returned as one
        window, identical to _meeting_document. Turns are never split across windows.
        """
        self._ensure_model()
        meeting = self.meetings[meeting_id]
        prefix = f"Réunion : {meeting.get('title', meeting_id)}.\n"
        cids = meeting["messages"]
        segments = {c: f"{self.records[c]['username']} : {self.records[c]['message_text']}\n" for c in cids}
        count = lambda s: len(self._tokenizer(s, add_special_tokens=False)["input_ids"])
        prefix_tokens = count(prefix)
        windows: list[list[str]] = []
        current: list[str] = []
        used = prefix_tokens
        for c in cids:
            n = count(segments[c])
            if current and used + n > budget:
                windows.append(current); current = []; used = prefix_tokens
            current.append(c); used += n
        if current:
            windows.append(current)
        out = []
        for win in windows:
            parts, spans, position = [prefix], {}, len(prefix)
            for c in win:
                seg = segments[c]
                spans[c] = (position, position + len(seg))
                parts.append(seg); position += len(seg)
            out.append(("".join(parts), spans, win))
        return out

    def _meeting_document(self, meeting_id: str) -> tuple[str, dict[str, tuple[int, int]]]:
        meeting = self.meetings[meeting_id]
        prefix = f"Réunion : {meeting.get('title', meeting_id)}.\n"
        parts = [prefix]
        spans: dict[str, tuple[int, int]] = {}
        position = len(prefix)
        for cid in meeting["messages"]:
            record = self.records[cid]
            segment = f"{record['username']} : {record['message_text']}\n"
            spans[cid] = (position, position + len(segment))
            parts.append(segment)
            position += len(segment)
        return "".join(parts), spans

    @torch.inference_mode()
    def build(self):
        if self.tokens is not None:
            return
        self._ensure_model()
        assert self._tokenizer is not None and self._model is not None
        started = time.time()
        token_rows: list[torch.Tensor] = []
        late_rows: list[torch.Tensor] = []
        ordered_ids: list[str] = []
        truncated = 0
        fallback = 0
        if self.late:
            for index, mid in enumerate(self.meetings, 1):
                for document, spans, window_ids in self._meeting_windows(mid, self.max_len - 8):
                    enc = self._tokenizer(document, return_offsets_mapping=True, truncation=True,
                                          max_length=self.max_len, return_tensors="pt")
                    offsets = enc["offset_mapping"][0].tolist()
                    if enc["input_ids"].size(1) >= self.max_len:
                        truncated += 1
                    enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
                    hidden = self._model(**enc).last_hidden_state[0].float()
                    hidden = F.normalize(hidden, dim=-1)
                    for cid in window_ids:
                        start, end = spans[cid]
                        token_idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= start and b <= end]
                        if token_idx:
                            rows = hidden[token_idx].to(torch.float16).cpu()
                        else:
                            rows = self._encode_chunk(self.records[cid]["encoder_text"])
                            fallback += 1
                        token_rows.append(rows)
                        late_rows.append(F.normalize(rows.float().mean(0), dim=-1).cpu())
                        ordered_ids.append(cid)
                if index % 10 == 0 or index == len(self.meetings):
                    print(f"  late-chunk index: {index}/{len(self.meetings)} meetings", flush=True)
        else:
            for index, cid in enumerate(self.ids, 1):
                rows = self._encode_chunk(self.records[cid]["encoder_text"])
                token_rows.append(rows)
                late_rows.append(F.normalize(rows.float().mean(0), dim=-1).cpu())
                ordered_ids.append(cid)
                if index % 400 == 0 or index == len(self.ids):
                    print(f"  chunk index: {index}/{len(self.ids)} chunks", flush=True)

        token_map = dict(zip(ordered_ids, token_rows))
        late_map = dict(zip(ordered_ids, late_rows))
        token_rows = [token_map[cid] for cid in self.ids]
        late_rows = [late_map[cid] for cid in self.ids]
        if self.compress:
            self.tokens, self.late_vecs = self._compress(token_rows, late_rows, started, truncated, fallback)
        else:
            self.tokens, self.late_vecs = self._save_raw(token_rows, late_rows, started, truncated, fallback)
        self.offsets = torch.tensor([0] + list(np.cumsum([r.size(0) for r in token_rows])), dtype=torch.long)
        self.doc_index = torch.repeat_interleave(
            torch.arange(len(self.ids)), self.offsets[1:] - self.offsets[:-1])

    @torch.inference_mode()
    def _encode_chunk(self, text: str) -> torch.Tensor:
        assert self._tokenizer is not None and self._model is not None
        enc = self._tokenizer([text], padding=True, truncation=True, max_length=128,
                              return_tensors="pt").to(self.device)
        hidden = self._model(**enc).last_hidden_state[0].float()
        hidden = F.normalize(hidden, dim=-1)
        return hidden[enc["attention_mask"][0].bool()].to(torch.float16).cpu()

    def _compress(self, token_rows: list[torch.Tensor], late_rows: list[torch.Tensor],
                  started: float, truncated: int, fallback: int):
        from sklearn.cluster import MiniBatchKMeans
        all_tokens = torch.cat(token_rows, dim=0).to(torch.float32)
        count, dim = all_tokens.shape
        k = max(2, min(self.n_centroids, max(2, count // 8)))
        sample = all_tokens.numpy()
        if sample.shape[0] > 60000:
            pick = np.random.default_rng(0).choice(sample.shape[0], 60000, replace=False)
            sample = sample[pick]
        km = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=4096, n_init=3, max_iter=100)
        km.fit(sample)
        centroids = torch.from_numpy(km.cluster_centers_).to(torch.float32)

        device = self.device
        cent_dev = centroids.to(device)
        assignments = torch.empty(count, dtype=torch.int16)
        residual_sum = torch.zeros(dim, dtype=torch.float32)
        for start in range(0, count, 8192):
            batch = all_tokens[start:start + 8192].to(device)
            assign = (batch @ cent_dev.T).argmax(dim=1)
            residual = batch - cent_dev[assign]
            assignments[start:start + batch.size(0)] = assign.to(torch.int16).cpu()
            residual_sum += residual.abs().sum(0).float().cpu()
        scale = (residual_sum / max(1, count)).clamp_min(1e-6)

        codes = []
        squared = 0.0
        cosine = 0.0
        for start in range(0, count, 8192):
            batch = all_tokens[start:start + 8192].to(device)
            assign = assignments[start:start + batch.size(0)].to(device).long()
            residual = batch - cent_dev[assign]
            codes.append(self._pack_signs(residual))
            signs = torch.where(residual >= 0, 1.0, -1.0)
            reconstructed = F.normalize(cent_dev[assign] + signs * scale.to(device), dim=-1)
            original = F.normalize(batch, dim=-1)
            squared += float(F.mse_loss(reconstructed, original, reduction="sum"))
            cosine += float((reconstructed * original).sum())
        packed = np.concatenate(codes, axis=0)

        payload = {
            "version": CACHE_VERSION, "ids": self.ids, "model": self.model_name,
            "max_len": self.max_len, "n_centroids": k, "compressed": True,
            "codes": torch.from_numpy(packed), "assign": assignments,
            "centroids": centroids.to(torch.float16), "scale": scale.to(torch.float16),
            "offsets": torch.tensor([0] + list(np.cumsum([r.size(0) for r in token_rows])), dtype=torch.long),
            "late_vecs": torch.stack(late_rows).to(torch.float16),
            "stats": {
                "meetings": len(self.meetings), "chunks": len(token_rows), "tokens": int(count),
                "dimension": int(dim), "centroids": int(k), "bits_per_dim": 1,
                "truncated_meetings": truncated, "fallback_chunks": fallback,
                "token_fp16_bytes": int(count * dim * 2),
                "code_bytes": int(packed.nbytes + assignments.numpy().nbytes
                                  + centroids.numpy().nbytes + scale.numpy().nbytes),
                "compression_ratio": round(float(count * dim * 2) / float(
                    max(1, packed.nbytes + assignments.numpy().nbytes
                        + centroids.numpy().nbytes + scale.numpy().nbytes)), 2),
                "reconstruction_mse": squared / max(1, count),
                "reconstruction_cosine": cosine / max(1, count),
                "build_seconds": round(time.time() - started, 1),
            },
        }
        torch.save(payload, self._cache_path)
        self.stats = payload["stats"]
        tokens, offsets, late_vecs = self._decompress(payload)
        return tokens, late_vecs

    def _save_raw(self, token_rows: list[torch.Tensor], late_rows: list[torch.Tensor],
                  started: float, truncated: int, fallback: int):
        counts = [r.size(0) for r in token_rows]
        tokens = torch.cat(token_rows, dim=0).to(torch.float16)
        late_vecs = torch.stack(late_rows).to(torch.float16)
        offsets = torch.tensor([0] + list(np.cumsum(counts)), dtype=torch.long)
        stats = {
            "meetings": len(self.meetings), "chunks": len(token_rows), "tokens": int(tokens.size(0)),
            "dimension": int(tokens.size(1)), "centroids": 0, "bits_per_dim": 0,
            "truncated_meetings": truncated, "fallback_chunks": fallback,
            "token_fp16_bytes": int(tokens.numpy().nbytes), "code_bytes": int(tokens.numpy().nbytes),
            "compression_ratio": 1.0, "reconstruction_mse": 0.0, "reconstruction_cosine": 1.0,
            "build_seconds": round(time.time() - started, 1),
        }
        payload = {"version": CACHE_VERSION, "ids": self.ids, "model": self.model_name,
                   "max_len": self.max_len, "compressed": False, "tokens": tokens,
                   "offsets": offsets, "late_vecs": late_vecs, "stats": stats}
        torch.save(payload, self._cache_path)
        self.stats = stats
        return tokens, late_vecs

    def _to_device(self):
        if self.tokens is None:
            self.build()
        device = torch.device(self.device)
        if self.tokens.device != device:
            self.tokens = self.tokens.to(device)
            self.doc_index = self.doc_index.to(device)
            self.late_vecs = self.late_vecs.to(device)

    @torch.inference_mode()
    def score(self, query: str) -> dict[str, np.ndarray]:
        self._to_device()
        self._ensure_model()
        assert self._tokenizer is not None and self.tokens is not None and self.doc_index is not None
        enc = self._tokenizer([query], truncation=True, max_length=64, return_tensors="pt").to(self.device)
        hidden = self._model(**enc).last_hidden_state.float()
        q = F.normalize(hidden[0][enc["attention_mask"][0].bool()], dim=-1).to(torch.float16)
        sims = q @ self.tokens.T
        maxima = torch.full((q.size(0), len(self.ids)), -1e4, device=self.device, dtype=torch.float16)
        maxima.scatter_reduce_(1, self.doc_index.unsqueeze(0).expand(q.size(0), -1), sims,
                               reduce="amax", include_self=True)
        maxsim = maxima.sum(dim=0).float().cpu().numpy()
        q_mean = F.normalize(q.float().mean(0), dim=-1)
        late = (q_mean @ self.late_vecs.float().T).cpu().numpy()
        return {"maxsim": maxsim, "late_vec": late}
