"""Walk-forward evaluation: at the start of each year, train on all earlier data, then trade
that year with the frozen model. Every trading day is scored by a model that never saw it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pca_stat_arb.accounting import pnl_from_holdings
from pca_stat_arb.deep.config import DeepConfig
from pca_stat_arb.deep.model import allocate
from pca_stat_arb.deep.residual_panel import ResidualPanel, cumulative_residual_features
from pca_stat_arb.deep.train import make_tensors, train_one


@dataclass
class WalkForwardResult:
    residual_weights: pd.DataFrame   # (dates x stocks) decided at each close, 0 outside trading years
    stock_weights: pd.DataFrame      # implied stock holdings (before execution lag)
    pnl: pd.DataFrame                # pnl_from_holdings output over the trading years
    folds: pd.DataFrame              # per year: training range, validation Sharpe per seed
    models: dict                     # year -> list of ensemble members' state_dicts (on CPU)


def run_walkforward(
    returns: pd.DataFrame, panel: ResidualPanel, cfg: DeepConfig, last_year: int, device="cpu", log=print,
    checkpoint_dir=None,
) -> WalkForwardResult:
    """checkpoint_dir: if given, each finished year is saved there and reused on a rerun with
    the same config, so an interrupted run resumes instead of starting over."""
    X, valid = cumulative_residual_features(panel, cfg.lookback)
    d = make_tensors(X, valid, panel.eps, panel.Q, panel.B, cfg.execution_lag, device)
    dates = panel.index
    T, N = valid.shape
    horizon = 1 + cfg.execution_lag          # a decision at t is labelled with eps_{t+horizon}
    first = int(np.argmax(valid.any(axis=1)))  # first day with any usable features

    W_res = np.zeros((T, N))
    folds, saved = [], {}
    for year in range(cfg.first_trade_year, last_year + 1):
        trade = np.flatnonzero(dates.year == year)
        if trade.size == 0:
            continue
        start = trade[0]
        # Labels of decision days must be known before the year starts: stop `horizon` days early.
        fit_end = start - horizon + 1
        val = range(fit_end - cfg.val_days, fit_end)
        train = range(first, val.start - horizon + 1)
        ckpt = None
        if checkpoint_dir is not None:
            key = hashlib.sha1(f"{cfg}|{dates[0]}|{list(panel.columns)}".encode()).hexdigest()[:10]
            ckpt = Path(checkpoint_dir) / f"fold_{year}_{key}.pt"
            if ckpt.exists():
                saved_fold = torch.load(ckpt)
                W_res[trade[0] : trade[-1] + 1] = saved_fold["weights"].numpy()
                folds.append(saved_fold["fold"])
                saved[year] = saved_fold["models"]
                log(f"{year}: loaded from checkpoint")
                continue
        log(f"{year}: train {dates[train.start].date()}..{dates[train.stop - 1].date()}, "
            f"validate ..{dates[val.stop - 1].date()}, trade {dates[trade[0]].date()}..{dates[trade[-1]].date()}")
        models, val_scores = [], []
        for seed in cfg.seeds:
            m, score = train_one(d, train, val, cfg, seed, log=log)
            models.append(m)
            val_scores.append(score)
            log(f"    seed {seed}: best validation net Sharpe {score:+.2f}")
        with torch.no_grad():
            for s in range(trade[0], trade[-1] + 1, 21):  # a few weeks at a time, to keep memory low
                sl = slice(s, min(s + 21, trade[-1] + 1))
                v = d.valid[sl]
                scores = torch.zeros(v.shape, device=device)
                for m in models:  # ensemble: average the members' scores
                    scores[v] += m(d.X[sl][v]) / len(models)
                W_res[sl] = allocate(scores, v).cpu().numpy()
        saved[year] = [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in models]
        folds.append({"year": year, "train_start": str(dates[train.start].date()), "train_end": str(dates[train.stop - 1].date()),
                      **{f"val_sharpe_seed{s}": sc for s, sc in zip(cfg.seeds, val_scores)}})
        if ckpt is not None:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"weights": torch.from_numpy(W_res[trade[0] : trade[-1] + 1].copy()), "fold": folds[-1], "models": saved[year]},
                       ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)

    W_stock = np.zeros_like(W_res)
    for t in np.flatnonzero(np.abs(W_res).sum(axis=1) > 0):
        W_stock[t] = panel.stock_weights(t, W_res[t])
    R = returns.reindex(index=dates, columns=panel.columns).to_numpy(dtype=float)
    pnl = pnl_from_holdings(dates, R, W_stock, cfg.cost_bps, execution_lag=cfg.execution_lag)
    first_trade = np.flatnonzero(dates.year == cfg.first_trade_year)[0]
    pnl = pnl.iloc[first_trade + 1 :]
    return WalkForwardResult(
        residual_weights=pd.DataFrame(W_res, index=dates, columns=panel.columns),
        stock_weights=pd.DataFrame(W_stock, index=dates, columns=panel.columns),
        pnl=pnl,
        folds=pd.DataFrame(folds),
        models=saved,
    )
