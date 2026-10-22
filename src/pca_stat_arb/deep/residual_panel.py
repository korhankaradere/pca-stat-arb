"""Tradable PCA residual returns.

At each close t the factor model is fitted on data up to t. The residual of day t+1 is then
the part of R_{t+1} not explained by the factors with those (already known) loadings:

    eps_{t+1} = R_{t+1} - B_t (Q_t^T R_{t+1})

A position w in residuals is a real portfolio: holding w_net = w - Q_t B_t^T w in stocks from
close t earns exactly w . eps_{t+1}. So the network can be trained on residual returns while
P&L and costs are computed on the stock holdings it implies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pca_stat_arb.pca import fit_eigenportfolios
from pca_stat_arb.residuals import fit_residuals


@dataclass
class ResidualPanel:
    index: pd.DatetimeIndex
    columns: pd.Index
    eps: np.ndarray        # (T, N) residual return of day t (NaN where undefined)
    Q: np.ndarray          # (T, N, m) eigenportfolio weights fitted at close t (0 off-universe)
    B: np.ndarray          # (T, N, m) factor betas fitted at close t (0 off-universe)
    tradable: np.ndarray   # (T, N) bool: in the model universe at close t

    def stock_weights(self, t: int, w: np.ndarray) -> np.ndarray:
        """Stock holdings at close t implied by residual weights w (length N)."""
        return w - self.Q[t] @ (self.B[t].T @ w)


def build_residual_panel(returns: pd.DataFrame, n_factors: int, pca_window: int, beta_window: int) -> ResidualPanel:
    R = returns.to_numpy(dtype=float)
    T, N = R.shape
    eps = np.full((T, N), np.nan, dtype=np.float32)
    Q = np.zeros((T, N, n_factors), dtype=np.float32)
    B = np.zeros((T, N, n_factors), dtype=np.float32)
    tradable = np.zeros((T, N), dtype=bool)

    for t in range(pca_window - 1, T):
        window = R[t - pca_window + 1 : t + 1]
        idx = np.flatnonzero(~np.isnan(window).any(axis=0))
        if idx.size <= n_factors + 1:
            continue
        eig = fit_eigenportfolios(window[:, idx], n_factors)
        recent = window[-beta_window:, idx]
        fit = fit_residuals(recent, recent @ eig.weights)
        Q[t, idx], B[t, idx], tradable[t, idx] = eig.weights, fit.betas, True
        if t + 1 < T:
            r_next = R[t + 1, idx]
            f_next = eig.weights.T @ np.nan_to_num(r_next)
            eps[t + 1, idx] = r_next - fit.betas @ f_next  # NaN stays NaN for a missing return
    return ResidualPanel(returns.index, returns.columns, eps, Q, B, tradable)


def cumulative_residual_features(panel: ResidualPanel, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Network inputs: for each close t and stock, the cumulative residual path over the last
    `lookback` days (eps_{t-L+1..t}, all known at close t), standardised across stocks.

    Returns X (T, N, L) float32 and valid (T, N) bool. Invalid entries of X are 0.
    """
    T, N = panel.eps.shape
    X = np.zeros((T, N, lookback), dtype=np.float32)
    valid = np.zeros((T, N), dtype=bool)
    for t in range(lookback, T):
        win = panel.eps[t - lookback + 1 : t + 1]                 # (L, N)
        ok = panel.tradable[t] & ~np.isnan(win).any(axis=0)
        if not ok.any():
            continue
        path = np.cumsum(win[:, ok], axis=0).T                     # (n_ok, L)
        scale = path.std()
        X[t, ok] = path / (scale if scale > 0 else 1.0)
        valid[t] = ok
    return X, valid
