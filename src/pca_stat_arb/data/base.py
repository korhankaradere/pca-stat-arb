"""Interface every data source implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


class ReturnsSource(ABC):
    """Produces daily total (dividend-adjusted) simple returns.

    Contract for load_returns:
      * index: trading dates (DatetimeIndex, ascending, no duplicates)
      * columns: one per security identifier (ticker for yfinance, PERMNO for CRSP)
      * values: simple returns; NaN where the security was not trading
    """

    name: str

    @abstractmethod
    def load_returns(self, start: str, end: str) -> pd.DataFrame: ...

    def load_volume(self, start: str, end: str) -> pd.DataFrame:
        """Daily share volume with the same index/columns as load_returns (optional)."""
        raise NotImplementedError(f"{self.name} does not provide volume")
