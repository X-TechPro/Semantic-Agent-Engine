"""Learned continuity scorer: pair features -> same-thread + relation.

The embedding only retrieves candidates. This scorer decides whether a distant
candidate continues/answers/supersedes the query, using text, speaker, time and
context features. It is trained on the synthetic relation graph and evaluated on
the real SUMM-RE pairs.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9€]+")
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
STOP = set("le la les un une des de du d au aux et ou mais donc pour dans sur avec sans ce cette ces il elle ils elles on nous vous je tu se sa son ses leur leurs est sont a ai as ont été être avoir plus moins très bien tout tous toute toutes".split())


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text or "") if t.lower() not in STOP}


def _nums(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUM_RE.findall(text or "")}


@dataclass
class PairMeta:
    project: str = ""
    speaker: str = ""
    turn: int = 0
    timestamp: str = ""
    field: str = ""


def features(q_text: str, c_text: str, q_emb: np.ndarray, c_emb: np.ndarray,
             q: PairMeta, c: PairMeta) -> list[float]:
    qa, ca = _tokens(q_text), _tokens(c_text)
    qn, cn = _nums(q_text), _nums(c_text)
    union = qa | ca
    inter = qa & ca
    cos = float(np.dot(q_emb, c_emb))
    return [
        cos,
        1.0 - cos,
        len(inter) / max(1, len(union)),
        len(inter) / max(1, min(len(qa), len(ca))),
        float(len(qn & cn)),
        float(len(qn)), float(len(cn)),
        1.0 if qn and (qn & cn) else 0.0,
        1.0 if q.speaker and q.speaker == c.speaker else 0.0,
        1.0 if q.project and q.project == c.project else 0.0,
        math.log1p(abs(q.turn - c.turn)),
        abs(len(q_text) - len(c_text)) / max(1, len(q_text) + len(c_text)),
        1.0 if q.field and q.field == c.field else 0.0,
    ]


@dataclass
class PairScore:
    relation: str
    p_same: float
    probs: dict[str, float]


class RelationScorer:
    def __init__(self):
        self.same = None
        self.relation = None
        self.classes_: list[str] = []

    def fit(self, rows: list[dict], *, seed: int = 42) -> "RelationScorer":
        X = np.asarray([r["x"] for r in rows], dtype=np.float64)
        y_same = np.asarray([0 if r["relation"] == "unrelated" else 1
                             for r in rows], dtype=int)
        y_rel = np.asarray([r["relation"] for r in rows])
        self.same = LogisticRegression(class_weight="balanced", max_iter=4000,
                                       solver="liblinear", random_state=seed)
        self.same.fit(X, y_same)
        self.relation = LogisticRegression(class_weight="balanced", max_iter=4000,
                                           solver="lbfgs", random_state=seed)
        self.relation.fit(X, y_rel)
        self.classes_ = list(self.relation.classes_)
        return self

    def score_one(self, x: list[float]) -> PairScore:
        arr = np.asarray([x], dtype=np.float64)
        ps = float(self.same.predict_proba(arr)[0, 1])
        pr = self.relation.predict_proba(arr)[0]
        probs = {c: float(p) for c, p in zip(self.classes_, pr)}
        relation = self.classes_[int(np.argmax(pr))]
        # A same-thread probability is the useful continuous edge weight.
        return PairScore(relation=relation, p_same=ps, probs=probs)

    def score_many(self, xs: list[list[float]]) -> list[PairScore]:
        if not xs:
            return []
        arr = np.asarray(xs, dtype=np.float64)
        ps = self.same.predict_proba(arr)[:, 1]
        pr = self.relation.predict_proba(arr)
        out = []
        for i in range(len(xs)):
            probs = {c: float(p) for c, p in zip(self.classes_, pr[i])}
            out.append(PairScore(self.classes_[int(np.argmax(pr[i]))],
                                 float(ps[i]), probs))
        return out


__all__ = ["RelationScorer", "PairMeta", "PairScore", "features"]
