"""Bang-bang trading rule on s-scores (Avellaneda & Lee, 2010, Sec. 5)."""

from __future__ import annotations

import numpy as np

LONG, FLAT, SHORT = 1, 0, -1


def update_positions(
    state: np.ndarray,
    s: np.ndarray,
    s_open: float,
    s_close_long: float,
    s_close_short: float,
) -> np.ndarray:
    """Move each name's position (+1 / 0 / -1) one day forward given today's s-score.

    Closes are checked before opens, so a long that closes can flip straight into a short
    on the same day if s is high enough. A NaN s-score (no longer mean-reverting) closes
    any open position.
    """
    state = state.copy()
    valid = np.isfinite(s)

    state[~valid] = FLAT
    close_long = (state == LONG) & valid & (s > -s_close_long)
    close_short = (state == SHORT) & valid & (s < s_close_short)
    state[close_long | close_short] = FLAT

    flat = (state == FLAT) & valid
    state[flat & (s < -s_open)] = LONG
    state[flat & (s > s_open)] = SHORT
    return state
