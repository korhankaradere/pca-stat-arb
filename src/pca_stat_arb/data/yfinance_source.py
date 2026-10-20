"""Yahoo Finance via yfinance: free, and fine for development, but survivorship-biased."""

from __future__ import annotations

import hashlib

import pandas as pd
import yfinance as yf

from pca_stat_arb.data.base import CACHE_DIR, ReturnsSource


class YFinanceSource(ReturnsSource):
    name = "yfinance"

    def __init__(self, tickers: list[str]):
        self.tickers = sorted(set(tickers))

    def _cache_path(self, field: str, start: str, end: str):
        key = hashlib.sha1(",".join(self.tickers).encode()).hexdigest()[:10]
        return CACHE_DIR / f"yf_{field}_{start}_{end}_{key}.parquet"

    def _download(self, start: str, end: str) -> None:
        raw = yf.download(
            self.tickers, start=start, end=end, auto_adjust=True,
            progress=False, threads=True, group_by="column",
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for field in ("Close", "Volume"):
            frame = raw[field].dropna(axis=1, how="all").sort_index()
            frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
            frame.to_parquet(self._cache_path(field.lower(), start, end))
        missing = set(self.tickers) - set(raw["Close"].dropna(axis=1, how="all").columns)
        if missing:
            print(f"yfinance returned no data for {len(missing)} tickers: {sorted(missing)[:10]}")

    def _load(self, field: str, start: str, end: str, refresh: bool) -> pd.DataFrame:
        path = self._cache_path(field, start, end)
        if refresh or not path.exists():
            self._download(start, end)
        return pd.read_parquet(path)

    def load_prices(self, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
        """Dividend- and split-adjusted closes, cached to parquet."""
        return self._load("close", start, end, refresh)

    def load_returns(self, start: str, end: str) -> pd.DataFrame:
        prices = self.load_prices(start, end)
        # fill_method=None: don't invent zero returns across gaps in a stock's history.
        returns = prices.pct_change(fill_method=None).iloc[1:]
        # Yahoo backfills prices for some stocks before their US listing (e.g. SW, FERG, AMCR)
        # with zero volume. Those days weren't tradable, so treat them as missing.
        volume = self.load_volume(start, end)
        return returns.where(volume.reindex_like(returns) > 0)

    def load_volume(self, start: str, end: str) -> pd.DataFrame:
        """Daily share volume, aligned to load_returns."""
        return self._load("volume", start, end, refresh=False).iloc[1:]
