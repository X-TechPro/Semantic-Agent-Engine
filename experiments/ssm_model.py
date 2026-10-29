"""A plain selective state-space model in pure torch.

No mamba_ssm, no CUDA kernels: the recurrence is written out so it is auditable.

    h_t = exp(dt_t * A) * h_{t-1} + (dt_t * B_t) * x_t
    y_t = <h_t, C_t> + D * x_t

`dt_t`, `B_t`, `C_t` come from the input, so the state is selective (it can
decide to keep or overwrite). A is diagonal and negative — the S4/S6 choice that
makes the scan cheap. Around it: a causal depthwise convolution and a gated
output, both standard.

This is deliberately the *simple* version: a sequential loop over time, no
parallel scan. At the corpus sizes here (a few hundred timesteps per batch) the
loop is not the bottleneck.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(4, math.ceil(d_model / 16))
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                              groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner,
                                self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        # A = -exp(A_log): decay rates 1..d_state, shared across channels
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      ).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)
        with torch.no_grad():
            self.dt_proj.bias.fill_(-4.0)      # softplus(-4) ~ 0.018

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        u, z = self.in_proj(x).chunk(2, dim=-1)             # (B,L,di) each
        u = self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)
        dt_raw, Bp, Cp = self.x_proj(u).split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))               # (B,L,di)
        A = -torch.exp(self.A_log)                          # (di,ds)
        h = x.new_zeros(B, self.d_inner, self.d_state)
        ys = []
        for t in range(L):
            dt_t = dt[:, t]                                 # (B,di)
            a = torch.exp(dt_t.unsqueeze(-1) * A)           # (B,di,ds)
            b = dt_t.unsqueeze(-1) * Bp[:, t].unsqueeze(1)  # (B,di,ds)
            h = a * h + b * u[:, t].unsqueeze(-1)
            ys.append((h * Cp[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + self.D * u
        y = y * F.silu(z)
        return self.out_proj(y)


class SSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.ssm(self.norm(x)))


class SSMLM(nn.Module):
    """Token embedding -> N selective SSM blocks -> next-token logits."""

    def __init__(self, vocab: int, d_model: int = 128, layers: int = 2,
                 d_state: int = 16, dropout: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=0)
        self.blocks = nn.ModuleList([
            SSMBlock(d_model, d_state, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, vocab)
        self.d_state = d_state

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.emb(ids))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.drop(self.norm(x)))


__all__ = ["SelectiveSSM", "SSMBlock", "SSMLM"]
