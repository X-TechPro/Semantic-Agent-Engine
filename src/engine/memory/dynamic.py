"""Dynamic mention writer: projected embeddings + learned event codes.

No hand-written point lexicon. A small projection is trained on real old/new
pairs; mentions are clustered into event codes; status is a learned classifier
over turn text. The event store then applies the latest/latest-confirmed rule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from torch import nn

from engine.memory.event_store import EventStore, MentionRecord


class Projection(nn.Module):
    def __init__(self, dim: int, out: int = 128):
        super().__init__()
        self.fc = nn.Linear(dim, out, bias=False)

    def forward(self, x):
        z = self.fc(x)
        return nn.functional.normalize(z, dim=-1)


@dataclass
class DynamicWriter:
    projection: Projection | None = None
    status_model: object | None = None
    threshold: float = 0.45

    def fit(self, mids: list[str], emb: dict[str, np.ndarray],
            pairs: dict[str, list[tuple[str, int, int]]],
            labels: dict[str, list[int]], texts: dict[str, list[str]],
            epochs: int = 120, seed: int = 42) -> "DynamicWriter":
        torch.manual_seed(seed)
        dim = int(emb[mids[0]].shape[1])
        self.projection = Projection(dim)
        opt = torch.optim.AdamW(self.projection.parameters(), lr=2e-3,
                                weight_decay=1e-3)
        positives = []
        for mid in mids:
            for _topic, old, new in pairs.get(mid, []):
                if old < len(emb[mid]) and new < len(emb[mid]):
                    positives.append((mid, old, new))
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            opt.zero_grad()
            losses = []
            for mid, old, new in positives:
                a = torch.tensor(emb[mid][old][None], dtype=torch.float32)
                b = torch.tensor(emb[mid][new][None], dtype=torch.float32)
                # negatives: turns from other meetings / distant turns
                neg_mid = next(m for m in mids if m != mid)
                idx = rng.integers(0, len(emb[neg_mid]), size=8)
                neg = torch.tensor(emb[neg_mid][idx], dtype=torch.float32)
                za, zb, zn = (self.projection(a)[0], self.projection(b)[0],
                              self.projection(neg))
                pos = torch.dot(za, zb) / 0.1
                negs = (zn @ za) / 0.1
                losses.append(-pos + torch.logsumexp(
                    torch.cat([pos[None], negs]), dim=0))
            if losses:
                loss = torch.stack(losses).mean()
                loss.backward(); opt.step()
        self.projection.eval()
        # status: learned from binary change labels, no hand cues
        train_texts, ys = [], []
        for mid in mids:
            for i, t in enumerate(texts[mid]):
                train_texts.append(t); ys.append(int(labels[mid][i]))
        self.status_model = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                            max_features=8000, sublinear_tf=True)
        X = self.status_model.fit_transform(train_texts)
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                                      solver="liblinear").fit(X, ys)
        return self

    def status(self, text: str) -> str:
        if self.status_model is None:
            return "open"
        p = float(self.clf.predict_proba(self.status_model.transform([text]))[0, 1])
        return "confirmed" if p >= 0.5 else "open"

    def event_codes(self, mid: str, emb: np.ndarray,
                    threshold: float | None = None) -> np.ndarray:
        z = self.project(emb)
        t = self.threshold if threshold is None else threshold
        if len(z) < 2:
            return np.zeros(len(z), dtype=int)
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=t, metric="cosine",
            linkage="average")
        return model.fit_predict(z)

    def project(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = self.projection(torch.tensor(x, dtype=torch.float32))
        return z.numpy()

    def mentions(self, mid: str, turns: list[dict], emb: np.ndarray,
                 threshold: float | None = None) -> list[MentionRecord]:
        codes = self.event_codes(mid, emb, threshold)
        z = self.project(emb)
        out = []
        for i, turn in enumerate(turns):
            text = turn.get("text", "")
            out.append(MentionRecord(
                mention_id=f"{mid}:{i}:c{int(codes[i])}", project=mid,
                turn=i, timestamp="2021-10-05", text=text,
                point=f"c{int(codes[i])}", value=text.strip(),
                status=self.status(text), evidence=text,
                embedding=z[i].tolist()))
        return out


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    z = np.load(Path(path))
    return {k: z[k] for k in z.files}


__all__ = ["DynamicWriter", "Projection", "load_embeddings"]
