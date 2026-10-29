"""§4 heads over the FlauBERT encoder: relation (6) + status (4).

Why a separate module: `inokufu/flaubert-base-uncased-xnli-sts` is a bare
`FlaubertModel` (196 tensors, no `classifier`, no `sequence_summary`).
`AutoModelForSequenceClassification.from_pretrained` therefore builds a
**randomly initialised** 2-logit head, and any ranking read from it is a random
projection — that is what produced the earlier "0-shot FlauBERT" numbers.

So the heads are defined here, trained here (`train_relation.py`), and loaded
from a checkpoint here. An untrained model reports `trained=False`; callers must
not present its logits as a relation prediction.

Encoder pooling matches the checkpoint's own config (`summary_type: first`):
the first token's hidden state, since the projection weights that config
promises are absent from the repository.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from engine.schemas_v4 import RELATIONS

MODEL_ID = "inokufu/flaubert-base-uncased-xnli-sts"
RELATION_LABELS: tuple[str, ...] = RELATIONS
STATUS_LABELS: tuple[str, ...] = ("proposed", "confirmed", "cancelled", "completed")


class Heads(nn.Module):
    """Two heads over the encoder's pooled states.

    The pair head reads the standard sentence-pair interaction
    `[u, v, |u-v|, u*v]`, not the `[CLS]` state of a concatenation. Reason,
    measured: this checkpoint has no trained pooler (`summary_use_proj` is in
    its config but every `sequence_summary.*` weight is absent), so `[CLS]`
    on a pair is an untrained position and the head stayed flat — 33rd place
    for a gold antecedent the pipeline had ranked 1st. The interaction feature
    is the pair representation the encoder can actually support.
    """

    def __init__(self, hidden: int = 768, n_relation: int = 6, n_status: int = 4,
                 n_change: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.n_relation = n_relation
        self.n_status = n_status
        self.n_change = n_change
        self.dropout = nn.Dropout(dropout)
        self.pair = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_relation))
        self.status = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_status))
        self.change = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_change))

    def forward_pair(self, features: torch.Tensor) -> torch.Tensor:
        return self.pair(features)

    def forward_unit(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.status(pooled)

    def forward_change(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.change(pooled)


@dataclass
class RelationPrediction:
    relation: str
    confidence: float
    probabilities: dict[str, float]


class FlaubertRelationModel:
    """Cross-encoder for relations + encoder for unit status."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cuda",
                 max_length: int = 256, seed: int = 42):
        from transformers import AutoModel, AutoTokenizer

        torch.manual_seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() or
                                   device == "cpu" else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.encoder = AutoModel.from_pretrained(model_id).to(self.device)
        hidden = int(getattr(self.encoder.config, "emb_dim", 768))
        self.heads = Heads(hidden).to(self.device)
        self.max_length = max_length
        self.model_id = model_id
        self.trained = False
        #: prior-matched cutoff for the change head; a class-weighted 2-way head
        #: is not calibrated at 0.5 (measured: recall 1.0 at 0.5). Set by
        #: training, read from the checkpoint, never guessed at inference.
        self.change_threshold = 0.5
        self.encoder_finetuned = False

    # --- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Save heads *and* the fine-tuned encoder.

        Saving only the heads silently discarded the encoder fine-tuning:
        reloading rebuilt the base encoder, so every re-evaluated fold scored a
        different model than the one that was trained (measured: change AUC
        0.73 in memory vs 0.62 after reload). The encoder is stored in half
        precision because it is an artefact, not a source file.
        """
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.heads.state_dict(), p / "heads.pt")
        torch.save({k: v.half() for k, v in self.encoder.state_dict().items()},
                   p / "encoder.pt")
        (p / "config.json").write_text(json.dumps({
            "model_id": self.model_id,
            "hidden": self.heads.hidden,
            "relation_labels": list(RELATION_LABELS),
            "status_labels": list(STATUS_LABELS),
            "change_labels": ["no_change", "change"],
            "change_threshold": self.change_threshold,
            "max_length": self.max_length,
            "trained": True,
            "encoder_finetuned": True,
            "encoder_dtype": "float16",
        }, indent=1), encoding="utf-8")
        return p

    def load(self, path: str | Path) -> "FlaubertRelationModel":
        p = Path(path)
        cfg_path = p / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"no head checkpoint at {p}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if cfg.get("relation_labels") != list(RELATION_LABELS):
            raise ValueError("checkpoint relation labels do not match the schema")
        if cfg.get("status_labels") != list(STATUS_LABELS):
            raise ValueError("checkpoint status labels do not match the schema")
        state = torch.load(p / "heads.pt", map_location=self.device,
                           weights_only=True)
        self.heads.load_state_dict(state)
        enc_path = p / "encoder.pt"
        if enc_path.exists():
            enc = torch.load(enc_path, map_location=self.device,
                             weights_only=True)
            enc = {k: v.float() for k, v in enc.items()}
            self.encoder.load_state_dict(enc)
            self.encoder_finetuned = True
        self.heads.eval()
        self.encoder.eval()
        self.trained = bool(cfg.get("trained", True))
        self.change_threshold = float(cfg.get("change_threshold", 0.5))
        return self

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cuda",
                        model_id: str = MODEL_ID) -> "FlaubertRelationModel":
        return cls(model_id=model_id, device=device).load(path)

    # --- encoding ----------------------------------------------------------
    def _batch(self, pairs: bool, a: list[str], b: list[str] | None = None):
        return self.tok(a, b, padding=True, truncation=True,
                        max_length=self.max_length, return_tensors="pt").to(self.device)

    def pooled(self, texts: list[str], batch: int = 16) -> torch.Tensor:
        """Mean-pooled encoder state, gradients intact (training path)."""
        out = []
        for s in range(0, len(texts), batch):
            enc = self._batch(False, texts[s:s + batch])
            h = self.encoder(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            out.append((h * mask).sum(1) / mask.sum(1).clamp(min=1))
        return torch.cat(out) if out else torch.zeros(0, self.heads.hidden,
                                                     device=self.device)

    def pair_features(self, queries: list[str], candidates: list[str]) -> torch.Tensor:
        u = self.pooled(queries)
        v = self.pooled(candidates)
        return torch.cat([u, v, (u - v).abs(), u * v], dim=-1)

    def embed(self, texts: list[str], batch: int = 32) -> torch.Tensor:
        """Mean-pooled L2-normalised encoder states — the honest cosine arm."""
        with torch.no_grad():
            v = self.pooled(texts, batch)
        return nn.functional.normalize(v, dim=-1).cpu()

    # --- inference ---------------------------------------------------------
    def relation_probs(self, queries: list[str], candidates: list[str]
                       ) -> list[dict[str, float]]:
        if not self.trained:
            raise RuntimeError(
                "relation head is untrained; train it with "
                "`python -m engine.link.train_relation` before reading it")
        out: list[dict[str, float]] = []
        self.heads.eval()
        with torch.no_grad():
            for s in range(0, len(queries), 16):
                feat = self.pair_features(queries[s:s + 16],
                                          candidates[s:s + 16])
                p = torch.softmax(self.heads.forward_pair(feat).float(), dim=-1)
                for row in p.cpu().tolist():
                    out.append(dict(zip(RELATION_LABELS, row)))
        return out

    def status_probs(self, texts: list[str]) -> list[dict[str, float]]:
        if not self.trained:
            raise RuntimeError("status head is untrained")
        out: list[dict[str, float]] = []
        self.heads.eval()
        with torch.no_grad():
            for s in range(0, len(texts), 16):
                p = torch.softmax(
                    self.heads.forward_unit(self.pooled(texts[s:s + 16])).float(),
                    dim=-1)
                for row in p.cpu().tolist():
                    out.append(dict(zip(STATUS_LABELS, row)))
        return out

    def change_probs(self, texts: list[str]) -> list[float]:
        """P(this turn changes something). The primary target, not a relation."""
        if not self.trained:
            raise RuntimeError("change head is untrained")
        out: list[float] = []
        self.heads.eval()
        with torch.no_grad():
            for s in range(0, len(texts), 16):
                p = torch.softmax(
                    self.heads.forward_change(self.pooled(texts[s:s + 16])).float(),
                    dim=-1)
                out += [float(row[1]) for row in p.cpu().tolist()]
        return out

    def predict_relation(self, query: str, candidate: str) -> RelationPrediction:
        probs = self.relation_probs([query], [candidate])[0]
        label = max(probs, key=probs.get)
        return RelationPrediction(label, float(probs[label]), probs)

    def predict_status(self, text: str) -> tuple[str, float]:
        probs = self.status_probs([text])[0]
        label = max(probs, key=probs.get)
        return label, float(probs[label])


__all__ = ["FlaubertRelationModel", "Heads", "RelationPrediction", "MODEL_ID",
           "RELATION_LABELS", "STATUS_LABELS"]
