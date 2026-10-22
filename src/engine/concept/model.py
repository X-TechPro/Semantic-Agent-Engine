"""Concept attention (6.13) — refine unit vectors with their neighbourhood.

The hypothesis is narrow and testable: a small transformer over *frozen* unit
embeddings, conditioned on the unit's neighbours and its metadata, ranks the
in-force unit above the stale one better than the frozen space does on its own.
"Attention over concepts instead of tokens" — the tokens were already embedded;
what is missing is that a unit's meaning depends on which other units are in
scope.

Two constraints the rest of the engine puts on this module, both discovered by
reading the call sites rather than guessing:

* **The topic vector must keep the base dimension.** Retrieval embeds the query
  with the real embedder and searches the refined index
  (`store.neighbors(vec, ..., use="refined")`), so a refined vector of any other
  width would not raise — FAISS would answer with nothing, and every refined
  query would look like "no results" rather than a broken model. The transformer
  projects down for speed and back up for storage.
* **Refined vectors are L2-normalised.** `IndexFlatIP` over unit-length rows is
  cosine; `embed/base.normalise()` is the engine's convention and the base
  vectors already obey it. Feeding raw logits into the same index would silently
  change the metric between the base and refined arms being compared.

Default-off: §0.4 keeps the measured rules baseline until the trained version
beats it, and §6.13 says to drop this if it does not clear +3 pts on temporal@1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

#: metadata vocabularies the encoder consumes. Kept as explicit tuples because a
#: checkpoint stores its index sets: reading them from a live enum would let a
#: schema change silently re-map every unit's metadata.
SOURCE_TYPES = ("meeting", "document", "code", "note")
KINDS = ("info", "query")
TYPES = ("decision", "action", "commitment", "question", "answer", "fact", "risk",
         "idea", "other")
STATUSES = ("proposed", "hypothesis", "conditional", "confirmed", "cancelled",
            "mention")


@dataclass(frozen=True)
class ConceptConfig:
    dim: int = 768                 # base vector width (the embedder's dim)
    d_model: int = 256             # working width; 6.13's "project down for speed"
    layers: int = 2                # 6.13 allows 2-4
    heads: int = 4                 # 6.13 allows 4-8
    dropout: float = 0.0           # zero by default: the suite must stay exact
    max_context: int = 17          # unit + k neighbours (k=16)
    meta_time_buckets: int = 8     # relative-time embedding granularity
    max_step: float = 0.20         # hard cap on ‖delta‖ before renormalising

    def __post_init__(self) -> None:
        if self.d_model % self.heads:
            raise ValueError(f"d_model {self.d_model} not divisible by heads "
                             f"{self.heads}")


class RelativeTime(nn.Module):
    """Age of the unit relative to the newest unit in the context set.

    `source_date` is the bi-temporal fact the whole engine turns on, and a raw
    day-count is unbounded, so the model sees a bucketed embedding plus the
    signed log-gap: the bucket carries "same week / same month / long ago" and
    the scalar keeps the ordering that supersede guards depend on.
    """

    def __init__(self, buckets: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(buckets + 1, d_model)   # +1 = unknown date
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, bucket: torch.Tensor, log_gap: torch.Tensor) -> torch.Tensor:
        emb = self.emb(bucket)
        return emb + self.scale * log_gap.unsqueeze(-1)


class ConceptAttention(nn.Module):
    """One forward pass over a context set -> refined vectors for every member.

    Inputs per unit (6.13): frozen base vector, metadata (source type, kind,
    type, status, relative time), position in thread. Outputs per unit: `topic`
    (same width as the base vector, for retrieval), `state` (d_model, consumed by
    the linker/validity side) and `validity` (scalar in [0,1]).
    """

    def __init__(self, cfg: ConceptConfig):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.dim, cfg.d_model)
        # the `_emb` suffix is load-bearing, not style: `nn.Module.type(dst_type)`
        # is torch's dtype-cast method, so an attribute literally named `type`
        # shadowed it and the first forward pass died inside `Module._apply` with
        # `TypeError: dtype must be a type, str, or dtype object`.
        self.src_emb = nn.Embedding(len(SOURCE_TYPES) + 1, cfg.d_model)
        self.kind_emb = nn.Embedding(len(KINDS) + 1, cfg.d_model)
        self.type_emb = nn.Embedding(len(TYPES) + 1, cfg.d_model)
        self.status_emb = nn.Embedding(len(STATUSES) + 1, cfg.d_model)
        self.time = RelativeTime(cfg.meta_time_buckets, cfg.d_model)
        # learned positional table, capped by max_context; a sinusoid would do but
        # 6.13 asks for "position in thread", i.e. a role that has to be learned.
        self.pos_emb = nn.Embedding(cfg.max_context, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.heads, dim_feedforward=cfg.d_model * 2,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True)
        # norm_first already forces this off; stating it keeps the warning that
        # otherwise prints on every construction out of the run logs.
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        # `topic` is a RESIDUAL on the frozen base vector, and this head is
        # zero-initialised so the delta starts at exactly 0.
        #
        # Measured reason: with the usual "project the hidden state into a new
        # vector" head, a freshly initialised model emits a random rotation of the
        # space — cosine-to-base 0.0105 and drift 1.4063 against a 0.25 bound on a
        # 3-project store, which the refine job then (correctly) refused to publish.
        # The drift penalty was spent un-rotating the initialisation instead of
        # refining, and through a 32-dim bottleneck it cannot represent the identity
        # map it would need to recover. 6.13 asks for refined vectors, not a second
        # embedding space: a unit may move a little from its base, so it must start
        # not moving at all. Zero-init makes the model begin AS the measured
        # baseline (0.4), and it only earns the right to move when a loss pays for it.
        self.out_topic = nn.Linear(cfg.d_model, cfg.dim)
        nn.init.zeros_(self.out_topic.weight)
        nn.init.zeros_(self.out_topic.bias)
        self.out_state = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_validity = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model // 2),
                                          nn.GELU(),
                                          nn.Linear(cfg.d_model // 2, 1))

    # --- vocabulary helpers (module-level so a checkpoint is self-describing) --
    @staticmethod
    def index_of(table: tuple[str, ...], value: str | None) -> int:
        """Unknown label -> 0, which is a real embedding row rather than a crash.

        A unit can legitimately carry `type=None` before the reader has run, and
        an IndexError inside a nightly refine job would look like model corruption.
        """
        if value is None:
            return 0
        try:
            return table.index(value) + 1
        except ValueError:
            return 0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """batch: x [B,N,d], meta_* [B,N], gap_days [B,N], turn [B,N], mask [B,N]."""
        x = self.in_proj(batch["x"])
        x = (x + self.src_emb(batch["src"]) + self.kind_emb(batch["kind"])
             + self.type_emb(batch["type"]) + self.status_emb(batch["status"]))
        x = x + self.time(batch["bucket"], batch["log_gap"])
        x = x + self.pos_emb(batch["pos"].clamp(max=self.cfg.max_context - 1))
        # padding rows are masked out of attention; a row of all-pad would make
        # softmax return NaNs, so the first position is never masked away.
        pad = batch["mask"] == 0
        pad[:, 0] = False
        h = self.encoder(x, src_key_padding_mask=pad)
        h = self.norm(h)
        # The residual, with a structural cap on how far it may move. Two facts,
        # both measured rather than assumed:
        #
        # * `out_topic` is zero-initialised and the base vector is *added*, so the
        #   refined vector equals the frozen base at step 0 (measured: cosine-to-base
        #   1.000000, drift 0.00000012). Summing rather than replacing is what makes
        #   the model begin AS the measured baseline (0.4); without the sum, a
        #   zero-initialised head emits all-zero vectors.
        # * `max_step` normalises the delta to at most this length, so
        #   ‖refined − base‖ is bounded by construction rather than by a penalty the
        #   optimiser can outbid. With only the soft hinge, a real 8-epoch run on 20
        #   projects reached drift_p95 1.2238 and cosine-to-base 0.5081 against a
        #   0.25 / 0.9688 contract: the contrastive term simply paid the hinge and
        #   rotated the space anyway, which is re-embedding, not refining. 6.13's own
        #   risk list ("drift invalidates ANN index") is a constraint on the geometry,
        #   so it belongs in the geometry.
        delta = self.out_topic(h)
        step = delta.norm(dim=-1, keepdim=True)
        delta = delta * (self.cfg.max_step / step.clamp(min=self.cfg.max_step))
        topic = F.normalize(batch["x"] + delta, dim=-1)
        validity = torch.sigmoid(self.out_validity(h).squeeze(-1))
        # `state` must go through `out_state`: returning the raw hidden state made
        # the head dead code (measured: `out_state.weight.grad is None` after a
        # backward through every loss term, including the triplet that exists to
        # train it), so the checkpoint would have carried an untrained projection
        # while the linker was documented to read it.
        state = self.out_state(h)
        # masked rows must not contribute a vector at all: they would otherwise
        # enter the drift term with a meaningless value.
        keep = batch["mask"].to(topic.dtype)
        return {"topic": topic, "topic_masked": topic * keep.unsqueeze(-1),
                "state": state, "state_masked": state * keep.unsqueeze(-1),
                "validity": validity * keep, "keep": keep}


def log_gap_days(days: float | None) -> float:
    """Signed log-gap used by `RelativeTime`; None -> 0.0 (unknown date)."""
    if days is None:
        return 0.0
    return math.copysign(math.log1p(abs(float(days))), float(days))


__all__ = ["ConceptAttention", "ConceptConfig", "RelativeTime", "log_gap_days",
           "SOURCE_TYPES", "KINDS", "TYPES", "STATUSES"]
