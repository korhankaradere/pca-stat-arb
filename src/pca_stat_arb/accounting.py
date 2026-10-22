"""P&L, trading costs and execution lag for a sequence of holdings.

Shared by every strategy in the package, so all of them are scored by the same rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def delay(holdings: np.ndarray, lag: int) -> np.ndarray:
    """Shift holdings `lag` rows later; nothing is held during the first `lag` rows."""
    if lag == 0:
        return holdings
    out = np.zeros_like(holdings)
    out[lag:] = holdings[:-lag]
    return out


def pnl_from_holdings(
    index: pd.DatetimeIndex,
    stock_returns: np.ndarray,
    stock_weights: np.ndarray,
    cost_bps: float,
    etf_returns: np.ndarray | None = None,
    etf_weights: np.ndarray | None = None,
    etf_cost_bps: float = 0.0,
    execution_lag: int = 0,
) -> pd.DataFrame:
    """Daily P&L of holdings decided at each close, as fractions of (fixed) capital.

    Row t of the weights is decided at close t. With execution_lag = k it is only in place
    from close t+k, and holdings in place at close t earn the return of day t+1. Trades at
    close t are charged on day t+1. A missing next-day return (e.g. a delisting in yfinance
    data) counts as 0, which is optimistic; CRSP fixes this.

    Returns columns gross, cost, net, stock_turnover, etf_turnover, all dated t+1.
    """
    T = len(index)
    W = delay(stock_weights, execution_lag)
    gross = np.zeros(T)
    gross[1:] = (W[:-1] * np.nan_to_num(stock_returns[1:])).sum(axis=1)
    stock_to = np.zeros(T)
    stock_to[1:] = np.abs(np.diff(W, axis=0, prepend=0.0)).sum(axis=1)[:-1]
    etf_to = np.zeros(T)
    if etf_weights is not None:
        H = delay(etf_weights, execution_lag)
        gross[1:] += (H[:-1] * np.nan_to_num(etf_returns[1:])).sum(axis=1)
        etf_to[1:] = np.abs(np.diff(H, axis=0, prepend=0.0)).sum(axis=1)[:-1]
    cost = stock_to * cost_bps / 1e4 + etf_to * etf_cost_bps / 1e4
    return pd.DataFrame(
        {"gross": gross, "cost": cost, "net": gross - cost, "stock_turnover": stock_to, "etf_turnover": etf_to},
        index=index,
    )
