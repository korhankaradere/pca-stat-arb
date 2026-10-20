"""Performance statistics for a daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(cum_pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of cumulative (additive) P&L."""
    return float((cum_pnl - cum_pnl.cummax()).min())


def summary(daily: pd.Series) -> pd.Series:
    """Annualised stats for daily P&L expressed as a fraction of fixed capital."""
    daily = daily.dropna()
    mean, std = daily.mean(), daily.std(ddof=1)
    cum = daily.cumsum()
    return pd.Series(
        {
            "ann_return": mean * TRADING_DAYS,
            "ann_vol": std * np.sqrt(TRADING_DAYS),
            "sharpe": mean / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
            "max_drawdown": max_drawdown(cum),
            "hit_rate": float((daily > 0).mean()),
            "n_days": len(daily),
        }
    )


def turnover(weights: pd.DataFrame) -> pd.Series:
    """Daily one-way turnover, sum |w_t - w_{t-1}|, in units of capital."""
    return weights.diff().abs().sum(axis=1)


def split_summary(daily: pd.Series, split_date: str, include_test: bool = True) -> pd.DataFrame:
    """summary() for the in-sample period, and optionally the held-out test and the full span."""
    split = pd.Timestamp(split_date)
    parts = {"in_sample": daily[daily.index < split]}
    if include_test:
        parts["test"] = daily[daily.index >= split]
        parts["full"] = daily
    return pd.DataFrame({k: summary(v) for k, v in parts.items()}).T
