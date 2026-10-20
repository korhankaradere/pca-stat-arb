"""Data sources. Everything downstream consumes a (dates x tickers) frame of daily simple returns."""

from pca_stat_arb.config import DataConfig
from pca_stat_arb.data.base import ReturnsSource


def get_source(cfg: DataConfig) -> ReturnsSource:
    if cfg.source == "yfinance":
        from pca_stat_arb.data.yfinance_source import YFinanceSource
        from pca_stat_arb.data.universe import load_universe

        return YFinanceSource(tickers=load_universe(cfg.universe))
    if cfg.source == "crsp":
        from pca_stat_arb.data.crsp_source import CRSPSource

        return CRSPSource()
    raise ValueError(f"unknown data source: {cfg.source!r}")


__all__ = ["ReturnsSource", "get_source"]
