"""Shared plumbing for running a config end to end (data loading + backtest)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pca_stat_arb.backtest import BacktestResult, run_backtest
from pca_stat_arb.config import Config
from pca_stat_arb.data import get_source


def load_market_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    source = get_source(cfg.data)
    returns = source.load_returns(cfg.data.start, cfg.data.end)
    try:
        volume = source.load_volume(cfg.data.start, cfg.data.end)
    except NotImplementedError:
        volume = None
    return returns, volume


def run_variant(cfg: Config, overrides: dict, returns: pd.DataFrame, volume: pd.DataFrame | None) -> BacktestResult:
    strategy = replace(cfg.strategy, **overrides)
    return run_backtest(returns, strategy, volume=volume)
