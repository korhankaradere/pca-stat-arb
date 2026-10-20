"""Daily walk-forward backtest of the PCA stat-arb strategy.

Timing: at the close of day t the model is fit on returns up to and including t, target
weights are set, and they earn the returns of day t+1. Nothing from t+1 or later is ever
used at t.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pca_stat_arb.config import StrategyConfig
from pca_stat_arb.pca import fit_eigenportfolios
from pca_stat_arb.residuals import fit_residuals, s_scores
from pca_stat_arb.signals import FLAT, update_positions


@dataclass
class BacktestResult:
    weights: pd.DataFrame        # executed dollar weights per stock, held from close t to close t+1
    states: pd.DataFrame         # +1 / 0 / -1 signal state per stock (dated when the signal fired)
    s_scores: pd.DataFrame       # s-score per stock (NaN if ineligible)
    # gross, cost, net (fractions of capital) and stock_turnover, etf_turnover, all dated t+1
    pnl: pd.DataFrame
    explained_variance: pd.Series  # share of variance explained by the kept PCA factors
    etf_weights: pd.DataFrame | None = None  # executed dollar holdings of hedge ETFs (ETF model)

    @property
    def net_returns(self) -> pd.Series:
        return self.pnl["net"]


def volume_scaling(volume: pd.DataFrame, window: int) -> pd.DataFrame:
    """AL's trading-time factor <V>/V_t, using a trailing average that ends at t (no lookahead).

    Returns on heavy-volume days are shrunk and quiet-day returns amplified, so moves backed
    by news/volume count for less in the mean-reversion signal.
    """
    v = volume.where(volume > 0)
    return v.rolling(window, min_periods=window).mean() / v


def run_backtest(
    returns: pd.DataFrame,
    cfg: StrategyConfig,
    volume: pd.DataFrame | None = None,
    etf_returns: pd.DataFrame | None = None,
    progress: bool = False,
) -> BacktestResult:
    """Run the strategy over a (dates x tickers) frame of daily simple returns.

    A stock is tradable on day t only if it has a full pca_window of returns ending at t.
    Positions in stocks that drop out are closed. Capital is held fixed at 1 (no compounding),
    so P&L is in units of starting capital, as in AL.

    volume (same shape as returns) is required when cfg.volume_adjust is set.
    etf_returns (dates x ETFs, same index) is required when cfg.factor_model == "etf".
    """
    R = returns.to_numpy(dtype=float)
    T, N = R.shape
    W = cfg.pca_window
    use_etf = cfg.factor_model == "etf"
    if use_etf:
        if etf_returns is None:
            raise ValueError("factor_model='etf' needs an etf_returns frame")
        E = etf_returns.reindex(returns.index).to_numpy(dtype=float)
        etf_hold = np.zeros((T, E.shape[1]))

    if cfg.volume_adjust:
        if volume is None:
            raise ValueError("volume_adjust=True needs a volume frame")
        scale = volume_scaling(volume.reindex_like(returns), cfg.volume_window).to_numpy(dtype=float)
        R_signal = R * scale
    else:
        R_signal = R

    weights = np.zeros((T, N))
    states = np.zeros((T, N), dtype=np.int8)
    scores = np.full((T, N), np.nan)
    explained = np.full(T, np.nan)
    state = np.zeros(N, dtype=np.int8)
    # Frozen mode: row i holds the stock-space hedge attached to the open position in stock i.
    hedges = np.zeros((N, N)) if cfg.hedge_mode == "frozen" else None

    for t in range(W - 1, T - 1):
        window = R[t - W + 1 : t + 1]
        sig_recent = R_signal[t - cfg.residual_window + 1 : t + 1]
        usable = ~np.isnan(window).any(axis=0) & ~np.isnan(sig_recent).any(axis=0)
        idx = np.flatnonzero(usable)
        etf_recent = E[t - cfg.residual_window + 1 : t + 1] if use_etf else None
        if idx.size <= cfg.n_factors + 1 or (use_etf and np.isnan(etf_recent).any()):
            state[:] = FLAT
            if hedges is not None:
                hedges[:] = 0.0
            continue

        if use_etf:
            eig = None
            factors = etf_recent
        else:
            eig = fit_eigenportfolios(window[:, idx], cfg.n_factors)
            factors = window[-cfg.residual_window :, idx] @ eig.weights
        fit = fit_residuals(sig_recent[:, idx], factors)
        s = s_scores(fit, cfg.kappa_min, cfg.center_mean)

        new_state = np.zeros(N, dtype=np.int8)
        new_state[idx] = update_positions(
            state[idx], s, cfg.s_open, cfg.s_close_long, cfg.s_close_short
        )

        # Each open name: +/- position_size dollars of the stock, hedged by shorting
        # beta_ij * position_size dollars of eigenportfolio j. In stock space the hedge for a
        # position w_i is -w_i * Q @ beta_i, so the whole book nets to w - Q @ (B^T @ w).
        if use_etf:
            # ETF factors are $1 holdings in each ETF, so the book's hedge is simply -B^T w
            # dollars of each ETF: one netted position per ETF instead of a basket of stocks.
            w = new_state[idx] * cfg.position_size
            weights[t, idx] = w
            etf_hold[t] = -(fit.betas.T @ w)
        elif hedges is None:
            w = new_state[idx] * cfg.position_size
            weights[t, idx] = w - eig.weights @ (fit.betas.T @ w)
        else:
            changed = np.flatnonzero(new_state != state)
            hedges[changed] = 0.0
            pos_in_idx = {g: k for k, g in enumerate(idx)}
            for g in changed:
                if new_state[g] != FLAT:
                    k = pos_in_idx[g]
                    hedges[g, idx] = -new_state[g] * cfg.position_size * (eig.weights @ fit.betas[k])
            weights[t] = new_state * cfg.position_size + hedges.sum(axis=0)

        if cfg.rebalance_band > 0:
            prev = weights[t - 1]
            weights[t] = np.where(np.abs(weights[t] - prev) > cfg.rebalance_band, weights[t], prev)
            # Don't keep stale weight in a stock we can no longer price.
            weights[t, ~usable] = np.where(np.isnan(R[t, ~usable]), 0.0, weights[t, ~usable])
            if use_etf:
                prev_e = etf_hold[t - 1]
                etf_hold[t] = np.where(np.abs(etf_hold[t] - prev_e) > cfg.rebalance_band, etf_hold[t], prev_e)

        state = new_state
        states[t] = state
        scores[t, idx] = s
        if eig is not None:
            explained[t] = eig.explained_variance_ratio.sum()

        if progress and t % 250 == 0:
            print(f"{returns.index[t].date()}  universe={idx.size}  open={np.count_nonzero(state)}")

    # Execution lag: holdings decided at close t are only in place from close t+lag.
    if cfg.execution_lag:
        weights = _delay(weights, cfg.execution_lag)
        if use_etf:
            etf_hold = _delay(etf_hold, cfg.execution_lag)

    # P&L: holdings in place at close t earn the return of t+1. A missing next-day return
    # (e.g. delisting in yfinance data) is treated as 0, which is optimistic; CRSP fixes this.
    gross = np.zeros(T)
    gross[1:] = (weights[:-1] * np.nan_to_num(R[1:])).sum(axis=1)
    stock_to = np.zeros(T)
    stock_to[1:] = np.abs(np.diff(weights, axis=0, prepend=0.0)).sum(axis=1)[:-1]
    etf_to = np.zeros(T)
    if use_etf:
        gross[1:] += (etf_hold[:-1] * np.nan_to_num(E[1:])).sum(axis=1)
        etf_to[1:] = np.abs(np.diff(etf_hold, axis=0, prepend=0.0)).sum(axis=1)[:-1]
    # Trades at close t are charged on day t+1's P&L.
    cost = stock_to * cfg.cost_bps / 1e4 + etf_to * cfg.etf_cost_bps / 1e4

    idx_dates = returns.index
    pnl = pd.DataFrame(
        {"gross": gross, "cost": cost, "net": gross - cost, "stock_turnover": stock_to, "etf_turnover": etf_to},
        index=idx_dates,
    )
    pnl = pnl.iloc[W:]  # first day with a possible position return

    return BacktestResult(
        weights=pd.DataFrame(weights, index=idx_dates, columns=returns.columns),
        states=pd.DataFrame(states, index=idx_dates, columns=returns.columns),
        s_scores=pd.DataFrame(scores, index=idx_dates, columns=returns.columns),
        pnl=pnl,
        explained_variance=pd.Series(explained, index=idx_dates, name="explained_variance"),
        etf_weights=pd.DataFrame(etf_hold, index=idx_dates, columns=etf_returns.columns) if use_etf else None,
    )


def _delay(holdings: np.ndarray, lag: int) -> np.ndarray:
    """Shift holdings `lag` rows later; nothing is held during the first `lag` rows."""
    out = np.zeros_like(holdings)
    out[lag:] = holdings[:-lag]
    return out
