"""Trained pair classifier for the linker (BUILD_GUIDE 6.8 baseline (b) / (c)).

A logistic regression over hand-built pair features answers one question:
*does this pair look like a supersede?*  It only ever **proposes**; the
deterministic guards in `supersede.evaluate` still decide, exactly as 6.11
demands ("a model never overrules a guard").

Why a separate model at all: measured, the rules linker scores `supersedes`
precision 1.0 on the Atlas holdout but 0.45 on dev, and the dev temporal@1 gain
of the relation layer is ~0 while the test gain is +0.148. The headroom numbers
say the architecture converts what it is given, so the binding constraint is
relation quality, not the validity filter.

No pickle: coefficients are stored as JSON, so `models/linker_lr.json` is
inspectable and portable across sklearn versions.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any

from engine.schemas import Unit

#: feature order — shared by training and inference, do not reorder casually
FEATURES: tuple[str, ...] = (
    "same_project", "same_subject", "same_attribute", "both_value_bearing",
    "source_assertive", "target_current", "target_is_mention",
    "typed_amount", "typed_date", "typed_owner", "typed_persons",
    "typed_equal", "numeric_sets_equal", "topical_overlap", "jaccard",
    "date_signed_gap_days", "abs_gap_days", "same_date",
    "len_ratio", "status_changed", "type_equal",
)

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_SCAFFOLD = {
    "projet", "budget", "lancement", "date", "prestataire", "stack", "technique",
    "rapport", "tarif", "recrutement", "conformite", "perimetre", "strategie",
    "prevu", "finalement", "environnements", "risque", "marche", "francophones",
}


def _numbers(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\d+(?:[.,]\d+)?", (text or "").replace("\u00a0", " ")))


def _words(text: str) -> set[str]:
    import unicodedata

    out = set()
    for w in _WORD.findall((text or "").lower()):
        w = "".join(c for c in unicodedata.normalize("NFD", w)
                    if not unicodedata.combining(c))
        if len(w) > 3 and w not in _SCAFFOLD:
            out.add(w)
    return out


def build_features(new: Unit, old: Unit) -> dict[str, float]:
    """One row per candidate pair, from Unit fields only (no embedder import)."""
    from engine.link.supersede import (SUPERSEDING_TYPES, VALUE_BEARING_TYPES,
                                       comparable_values, same_slot)
    from engine.slots.subjects import fold

    s_new, s_old = new.slots, old.slots
    comparable, equal = comparable_values(new, old)
    tn = {x for x in (s_new.amount, s_new.date_value, s_new.owner) if x}
    to = {x for x in (s_old.amount, s_old.date_value, s_old.owner) if x}
    wn, wo = _words(new.text), _words(old.text)
    inter = len(wn & wo)
    union = len(wn | wo)
    gap = (new.source_date - old.source_date).days
    an = new.status in {"confirmed", "cancelled"}
    ao = old.status in {"confirmed", "cancelled"}
    attr_n = (s_new.subject or "").rsplit(".", 1)[-1]
    attr_o = (s_old.subject or "").rsplit(".", 1)[-1]
    return {
        "same_project": float(bool(new.project) and new.project == old.project),
        "same_subject": float(same_slot(new, old)),
        "same_attribute": float(attr_n == attr_o and bool(attr_n)),
        "both_value_bearing": float(new.type in SUPERSEDING_TYPES
                                    and old.type in VALUE_BEARING_TYPES),
        "source_assertive": float(an),
        "target_current": float(old.is_current),
        "target_is_mention": float(old.status == "mention"),
        "typed_amount": float(bool(s_new.amount and s_old.amount)),
        "typed_date": float(bool(s_new.date_value and s_old.date_value)),
        "typed_owner": float(bool(s_new.owner and s_old.owner)),
        "typed_persons": float(bool(set(s_new.persons) & set(s_old.persons))),
        "typed_equal": float(equal and bool(tn and to)),
        "numeric_sets_equal": float(_numbers(new.text) == _numbers(old.text)
                                    and bool(_numbers(new.text))),
        "topical_overlap": inter / max(1, min(len(wn), len(wo))) if (wn and wo) else 0.0,
        "jaccard": inter / union if union else 0.0,
        "date_signed_gap_days": max(-3650.0, min(3650.0, float(gap))) / 365.0,
        "abs_gap_days": min(3650.0, abs(float(gap))) / 365.0,
        "same_date": float(gap == 0),
        "len_ratio": min(len(new.text), len(old.text)) / max(1, max(len(new.text),
                                                                    len(old.text))),
        "status_changed": float(an != ao),
        "type_equal": float(new.type == old.type),
    }


def to_row(feats: dict[str, float]) -> list[float]:
    return [float(feats.get(k, 0.0)) for k in FEATURES]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class PairClassifier:
    """JSON-stored logistic regression: p = sigmoid(w·x + b)."""

    def __init__(self, weights: list[float], bias: float, *, threshold: float = 0.5,
                 meta: dict[str, Any] | None = None):
        if len(weights) != len(FEATURES):
            raise ValueError(f"expected {len(FEATURES)} weights, got {len(weights)}")
        self.weights = weights
        self.bias = bias
        self.threshold = threshold
        self.meta = meta or {}

    def score(self, new: Unit, old: Unit) -> float:
        x = to_row(build_features(new, old))
        return sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, x)))

    def proposes_supersede(self, new: Unit, old: Unit) -> bool:
        return self.score(new, old) >= self.threshold

    def explain(self, new: Unit, old: Unit, top: int = 5) -> list[tuple[str, float, float]]:
        x = dict(zip(FEATURES, to_row(build_features(new, old))))
        contrib = sorted(((f, x[f], x[f] * w) for f, w in zip(FEATURES, self.weights)),
                         key=lambda t: -abs(t[2]))
        return contrib[:top]

    # --- persistence -------------------------------------------------------
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"features": list(FEATURES), "weights": self.weights,
                                    "bias": self.bias, "threshold": self.threshold,
                                    "meta": self.meta}, indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "PairClassifier | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if d.get("features") != list(FEATURES):
            return None  # stale feature order -> refuse rather than mispredict
        return cls(list(d["weights"]), float(d["bias"]),
                   threshold=float(d.get("threshold", 0.5)), meta=d.get("meta") or {})
