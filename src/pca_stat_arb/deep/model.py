"""CNN + Transformer signal extractor and allocation (Guijarro-Ordonez, Pelger & Zanotti, 2021).

The CNN picks up local shapes in a residual path (jumps, trends, kinks); the Transformer lets
every point of the path attend to every other, so the model can relate e.g. a recent drop to
the level a month ago. A linear head turns the summary into one score per stock.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels_in: int, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels_in, channels, kernel_size=2, padding=1)
        self.norm = nn.InstanceNorm1d(channels)
        self.skip = nn.Conv1d(channels_in, channels, kernel_size=1) if channels_in != channels else nn.Identity()

    def forward(self, x):  # x: (batch, channels_in, L)
        out = self.conv(x)[..., : x.shape[-1]]  # causal-ish: drop the extra padded step
        return torch.relu(self.norm(out)) + self.skip(x)


class CNNTransformer(nn.Module):
    def __init__(self, lookback: int, filters: int = 8, heads: int = 4, dropout: float = 0.25):
        super().__init__()
        self.conv = nn.Sequential(ConvBlock(1, filters), ConvBlock(filters, filters))
        self.position = nn.Parameter(torch.zeros(1, lookback, filters))
        layer = nn.TransformerEncoderLayer(
            d_model=filters, nhead=heads, dim_feedforward=2 * filters,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Linear(filters, 1)

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        """paths: (batch, L) cumulative residual paths -> (batch,) scores."""
        h = self.conv(paths.unsqueeze(1)).transpose(1, 2)  # (batch, L, filters)
        h = self.transformer(h + self.position)
        return self.head(h[:, -1]).squeeze(-1)


def allocate(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Scores (days, N) -> residual weights with sum |w| = 1 per day; invalid stocks get 0."""
    s = torch.where(valid, scores, torch.zeros_like(scores))
    return s / s.abs().sum(dim=-1, keepdim=True).clamp_min(1e-8)
