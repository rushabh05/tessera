"""A small per-modality encoder: Linear -> GroupNorm -> GELU -> Linear.

GroupNorm, never BatchNorm - the same rule the architecture design settled on
project-wide, for small-batch stability (a leave-one-replica-out fold's smallest
class can have a handful of examples) and because BatchNorm statistics depend on
what else is in the batch, which is exactly the kind of batch-composition
dependence this project's own falsification work (the base paper's
`trace(cov(Z))`) criticises elsewhere - a model whose own encoder had the same
defect would be indefensible.
"""

from __future__ import annotations

import torch
from torch import nn


class SmallMLPEncoder(nn.Module):
    """Maps one modality's flat feature slice to a shared embedding dimension."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, groups: int = 4) -> None:
        super().__init__()
        # GroupNorm needs the channel count divisible by the group count; a small
        # hidden_dim (e.g. 8) with groups=4 still works, but assert rather than let
        # torch raise a less legible error at the first forward pass.
        if hidden_dim % groups != 0:
            groups = 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
