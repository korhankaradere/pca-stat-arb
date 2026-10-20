"""Stock universes."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

UNIVERSE_DIR = Path(__file__).resolve().parents[3] / "data" / "universes"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Select Sector SPDR ETFs that have traded since December 1998, so the hedge set is the same
# throughout the sample. (XLRE, 2015, and XLC, 2018, are left out for that reason.)
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def fetch_sp500_current() -> pd.DataFrame:
    """Scrape today's S&P 500 constituents from Wikipedia.

    Caveat: this is today's membership applied to the past. Stocks that were dropped or
    went bust are missing, so backtests on it are survivorship-biased.
    """
    html = requests.get(SP500_URL, headers={"User-Agent": "pca-stat-arb research"}, timeout=30).text
    table = pd.read_html(StringIO(html), attrs={"id": "constituents"})[0]
    table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)  # BRK.B -> BRK-B (Yahoo style)
    return table[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]].rename(
        columns={"Symbol": "ticker", "Security": "name", "GICS Sector": "sector", "GICS Sub-Industry": "sub_industry"}
    )


def load_universe(name: str, refresh: bool = False) -> list[str]:
    """Load a universe's tickers, snapshotting it to data/universes/ on first use so runs are reproducible."""
    if name != "sp500_current":
        raise ValueError(f"unknown universe: {name!r}")
    path = UNIVERSE_DIR / f"{name}.csv"
    if refresh or not path.exists():
        UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
        fetch_sp500_current().to_csv(path, index=False)
    return pd.read_csv(path)["ticker"].tolist()
