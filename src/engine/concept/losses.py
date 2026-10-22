"""The four losses of 6.13, written so each can be measured alone.

Reporting them separately is not decoration: the kill test says whether the
refined space helps, and these four say *why* it helped or failed. A single
weighted sum that improved temporal@1 would not distinguish "the hard negatives
taught it proposed-vs-confirmed" from "the drift bound was so loose that the
vectors became a different space and every stored ANN index is stale".

**The anchoring constraint that shapes all of them.** Retrieval embeds a query
with the frozen embedder (`DenseRetriever.search` -> `embedder.embed([query],
role="query")`) and searches the refined index. So the query vector is a *base*
vector, while the stored side is *refined*. The refined space therefore has to
remain a small perturbation of the base space or the two sides are not
comparable — which is precisely why 6.13 pairs the refinement with a drift bound
rather than trusting it. Consequently:

* `contrastive` pulls the centre's **refined** vector toward its neighbours'
  **frozen** vectors, never toward other refined vectors. A topic-to-topic term
  would let the whole space rotate freely and score well here while destroying
  live retrieval.
* `drift` bounds how far that pull may carry a vector. With weight 0 the two
  terms above are unconstrained, so the weight is part of the contract and is
  recorded in the checkpoint, not left at an optimiser default.

1. `contrastive` — relation-aware positives, and 6.13's named hard negatives:
   **same topic, different state**.
2. `validity`     — a superseded unit's validity falls, the in-force unit's stays high.
3. `incremental`  — encode a prefix then extend it, vs one-shot encoding of the
   whole set. The corpus grows meeting by meeting; a refinement that only holds
   with the full context present at once would be wrong after the next ingest.
4. `state`        — triplet margin over the model's own state vectors, giving the
   state head an objective (6.13 hands that vector to the linker).
5. `drift`        — ||refined − base||, with a hinge past `max_norm`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: the loss terms this module knows about. `train()` validates its weights dict
#: against it, because `total()` looks weights up with `.get(name, 0.0)`: without a
#: guard, a typo'd key trains with that term silently disabled (D-011's inert knob).
TERMS = ("contrastive", "validity", "incremental", "state", "drift")


def contrastive(topic: torch.Tensor, base: torch.Tensor, mask: torch.Tensor,
                pos: torch.Tensor, neg: torch.Tensor, *, temperature: float = 0.07
                ) -> torch.Tensor:
    """InfoNCE: centre's refined topic vector vs its neighbours' frozen vectors.

    `pos` / `neg` are [B, m] neighbour-index tensors padded with -1. Hard
    negatives stay *in the denominator* on purpose — excluding them would train
    the model to ignore exactly the pairs it is supposed to separate. The centre's
    own column is always excluded (a query must not retrieve itself as evidence of
    itself), which the -1 padding does naturally by scattering onto column 0.

    Rows without a positive are skipped rather than scored, so a corpus with
    nothing to learn still returns 0.0 instead of a number that looks like progress.
    """
    q = F.normalize(topic[:, 0], dim=-1)
    k = F.normalize(base, dim=-1)
    logits = (q.unsqueeze(1) * k).sum(-1) / temperature           # [B, N]
    valid_col = mask != 0
    denom_mask = valid_col.clone()
    denom_mask[:, 0] = False
    for idx in (pos, neg):
        keep = idx >= 0
        rows = torch.arange(idx.shape[0], device=idx.device).unsqueeze(1).expand_as(idx)
        cols = torch.where(keep, idx, torch.zeros_like(idx)).clamp(max=logits.shape[1] - 1)
        denom_mask[rows[keep], cols[keep]] = False
    denom = logits.masked_fill(~denom_mask, -torch.inf)
    d_lse = torch.logsumexp(denom, dim=1)
    pos_ok = pos >= 0
    gather = torch.where(pos_ok, pos, torch.zeros_like(pos)).clamp(max=logits.shape[1] - 1)
    pos_logits = torch.gather(logits, 1, gather)                  # [B, m]
    per = torch.logaddexp(pos_logits, d_lse.unsqueeze(1)) - pos_logits
    per = torch.where(pos_ok, per, torch.zeros_like(per))
    live = pos_ok.any(dim=1) & torch.isfinite(d_lse)
    if not bool(live.any()):
        return torch.zeros((), device=topic.device)
    return (per.sum(dim=1) / pos_ok.sum(dim=1).clamp(min=1))[live].mean()


def validity(validity_pred: torch.Tensor, target: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """BCE on the validity head, over unmasked rows.

    `target` carries -1 where the ledger says nothing about a unit; those rows are
    dropped, never mapped to 0.0. An unlabelled `proposed` unit is not a
    "not-valid" unit, and silently co-labelling them would teach the head the
    opposite of what 6.13 asks it to represent.
    """
    keep = (mask != 0) & (target >= 0)
    if not bool(keep.any()):
        return torch.zeros((), device=validity_pred.device)
    return F.binary_cross_entropy(validity_pred[keep].clamp(1e-6, 1 - 1e-6),
                                  target[keep])


def incremental(full: torch.Tensor, prefix_then_suffix: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
    """1 − cosine(one-shot full set, prefix encoded then extended by the suffix).

    Measured on the *topic* vectors because those are what retrieval stores; a
    disagreement here means the nightly refine job and a live ingest would write
    different geometry for the same unit.
    """
    a = F.normalize(full, dim=-1)
    b = F.normalize(prefix_then_suffix, dim=-1)
    cos = (a * b).sum(-1)
    keep = mask != 0
    if not bool(keep.any()):
        return torch.zeros((), device=full.device)
    return 1.0 - cos[keep].mean()


def triplet(states: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor,
            neg: torch.Tensor, *, margin: float = 0.3) -> torch.Tensor:
    """Triplet margin over the model's own state vectors — the state head's job.

    Without this term `out_state` receives no gradient at all (measured:
    `out_state.weight`/`out_state.bias` come back with `grad is None` from a
    topic+validity loss), and 6.13 hands that vector to the linker as an input. An
    untrained projection would be a random rotation of the hidden state, which is
    worse than not claiming the output exists.

    Only the centre row anchors a triplet: averaging over every neighbour slot
    would weight the same centre sixteen times and silently inflate its pull.
    A row needs at least one positive *and* one negative to contribute, so a
    batch of all-positive context sets returns 0.0 rather than a fake margin.
    """
    s = F.normalize(states, dim=-1)
    n = s.shape[1]
    d = 1.0 - (s[:, 0, None] * s).sum(-1)                        # [B, N] from centre
    valid_slot = mask != 0

    def _pair(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        keep = (idx >= 0) & (idx < n) & valid_slot.gather(1, idx.clamp(min=0, max=n - 1))
        keep = keep & (idx != 0)          # the centre is never its own positive/negative
        return idx.clamp(min=0, max=n - 1), keep

    p_idx, p_ok = _pair(pos)
    q_idx, q_ok = _pair(neg)
    dp = (torch.gather(d, 1, p_idx) * p_ok).sum(1) / p_ok.sum(1).clamp(min=1)
    dn = (torch.gather(d, 1, q_idx) * q_ok).sum(1) / q_ok.sum(1).clamp(min=1)
    live = p_ok.any(dim=1) & q_ok.any(dim=1)
    if not bool(live.any()):
        return torch.zeros((), device=states.device)
    return F.relu(margin + dp[live] - dn[live]).mean()


def drift(refined: torch.Tensor, base: torch.Tensor, mask: torch.Tensor,
          *, max_norm: float = 0.25) -> torch.Tensor:
    """Mean squared displacement, plus a hinge once it exceeds `max_norm`.

    The hinge is the point: a plain L2 penalty trades drift against accuracy and
    lets the optimiser buy accuracy with unbounded movement. `max_norm` states how
    far a refined vector may move from its frozen base before the loss stops being
    a trade-off and becomes a violation — and since a live query is a *base*
    vector, unbounded movement silently breaks retrieval while this number looks fine.
    """
    d = refined - base
    sq = (d * d).sum(-1)
    keep = mask.to(sq.dtype)
    n = keep.sum().clamp(min=1.0)
    mean = (sq * keep).sum() / n
    # `+ eps` inside the sqrt is load-bearing, not tidiness. `sqrt` has an infinite
    # derivative at 0 and `relu`'s derivative there is 0, so the hinge's backward is
    # `0 * inf = NaN`: measured, `relu(sqrt(0) - 0.25)` returns 0.0 forward and NaN
    # backward, and `drift` on a refined vector exactly equal to its base gave a
    # finite value with a non-finite gradient. The zero-initialised residual head
    # makes refined == base at step 0 *by construction*, so without this every
    # training run would NaN on the very first backward.
    over = F.relu((sq + 1e-12).sqrt() - max_norm)
    hinge = (over * keep).sum() / n
    return mean + hinge


def total(parts: dict[str, torch.Tensor], weights: dict[str, float]
          ) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum + the per-term values actually used.

    Every term is reported even when its weight is zero, so an ablation that turns
    one off still shows its number and the table cannot quietly drop a row.
    """
    dev = next(iter(parts.values())).device
    out = torch.zeros((), device=dev)
    seen: dict[str, float] = {}
    for name, value in parts.items():
        w = float(weights.get(name, 0.0))
        out = out + w * value
        seen[name] = float(value.detach())
        seen[f"w_{name}"] = w
    seen["total"] = float(out.detach())
    return out, seen


__all__ = ["TERMS", "contrastive", "validity", "incremental", "triplet", "drift",
           "total"]
