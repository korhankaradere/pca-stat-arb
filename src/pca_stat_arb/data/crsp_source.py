"""CRSP daily stock file via WRDS (planned, not implemented yet).

Why it matters: CRSP gives a point-in-time universe including delisted stocks, and
delisting returns. Both are needed for a survivorship-free backtest.

Planned implementation:
  1. conn = wrds.Connection()  (Imperial WRDS credentials; install with `pip install -e .[crsp]`)
  2. Daily returns: crsp.dsf (permno, date, ret, prc, shrout), joined to crsp.dsenames for
     shrcd in (10, 11) (ordinary common shares) and exchcd in (1, 2, 3) (NYSE/AMEX/NASDAQ).
  3. Merge delisting returns from crsp.dsedelist into the last trading day.
  4. Universe each day: e.g. top-N by market cap (|prc| * shrout) as of the prior month end,
     or historical S&P 500 membership from crsp.dsp500list. AL used stocks above $1bn cap.
  5. Pivot to (date x permno) returns, meeting the ReturnsSource contract.
"""

from __future__ import annotations

import pandas as pd

from pca_stat_arb.data.base import ReturnsSource


class CRSPSource(ReturnsSource):
    name = "crsp"

    def load_returns(self, start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("CRSP loader is planned; see module docstring")
