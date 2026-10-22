"""Training: maximise the Sharpe ratio of net daily P&L, with costs on the implied stock trades."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from pca_stat_arb.deep.config import DeepConfig
from pca_stat_arb.deep.model import CNNTransformer, allocate


@dataclass
class Tensors:
    """Model inputs and targets for a range of days, on one device."""
    X: torch.Tensor        # (T, N, L) features at close t
    valid: torch.Tensor    # (T, N) tradable with a full feature path at close t
    target: torch.Tensor   # (T, N) residual return earned by a position set at close t
    Q: torch.Tensor        # (T, N, m)
    B: torch.Tensor        # (T, N, m)


def make_tensors(X, valid, eps, Q, B, lag: int, device) -> Tensors:
    """Targets: a position decided at close t is in place from close t+lag and earns eps_{t+1+lag}."""
    T = X.shape[0]
    target = np.zeros_like(eps)
    shift = 1 + lag
    target[: T - shift] = np.nan_to_num(eps[shift:])
    as_t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
    return Tensors(as_t(X), as_t(valid, torch.bool), as_t(target), as_t(Q), as_t(B))


def portfolio_path(model, d: Tensors, days: slice, cost_bps: float):
    """Net daily returns of the model's portfolio over consecutive `days` (differentiable)."""
    X, valid = d.X[days], d.valid[days]
    D, N, L = X.shape
    scores = torch.zeros(D, N, device=X.device)
    scores[valid] = model(X[valid])
    w = allocate(scores, valid)                                        # residual weights
    w_stock = w - torch.einsum("dnm,dm->dn", d.Q[days], torch.einsum("dnm,dn->dm", d.B[days], w))
    gross = (w * d.target[days]).sum(dim=1)
    turnover = (w_stock[1:] - w_stock[:-1]).abs().sum(dim=1)
    net = gross.clone()
    net[1:] = net[1:] - turnover * cost_bps / 1e4
    return net, gross, w


def sharpe(x: torch.Tensor) -> torch.Tensor:
    return x.mean() / x.std().clamp_min(1e-8) * np.sqrt(252)


@torch.no_grad()
def evaluate(model, d: Tensors, days: range, cfg: DeepConfig) -> float:
    model.eval()
    net, _, _ = portfolio_path(model, d, slice(days.start, days.stop), cfg.cost_bps)
    return float(sharpe(net[1:]))


def train_one(d: Tensors, train_days: range, val_days: range, cfg: DeepConfig, seed: int, log=None):
    """Train one network; keep the epoch with the best validation net Sharpe."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = CNNTransformer(cfg.lookback, cfg.filters, cfg.heads, cfg.dropout).to(d.X.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    # Start from the untrained network, so a model that never improves is still well defined.
    best, best_state, stale = evaluate(model, d, val_days, cfg), copy.deepcopy(model.state_dict()), 0
    starts = np.arange(train_days.start, train_days.stop - cfg.block_days + 1, cfg.block_days // 2)
    for epoch in range(cfg.max_epochs):
        model.train()
        for s in rng.permutation(starts):
            net, _, _ = portfolio_path(model, d, slice(s, s + cfg.block_days), cfg.cost_bps)
            loss = -sharpe(net[1:])
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = evaluate(model, d, val_days, cfg)
        if log:
            log(f"    seed {seed} epoch {epoch:2d}  val net Sharpe {score:+.2f}")
        if score > best:
            best, best_state, stale = score, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best
