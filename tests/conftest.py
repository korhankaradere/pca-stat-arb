import numpy as np
import pandas as pd
import pytest


def simulate_factor_market(T=600, N=80, m=3, kappa=25.0, seed=0, return_factors=False):
    """Returns driven by m factors plus idiosyncratic residuals whose cumulative sum is OU.

    Returns (returns_frame, true_loadings), plus the factor-return frame if return_factors.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    # As in real equities, the market factor dominates the others.
    factors = rng.normal(0, 1, size=(T, m)) * np.array([0.012] + [0.005] * (m - 1))
    # Factor 1 is a "market" (all loadings positive); the others are zero-mean "sector/style" factors.
    loadings = rng.normal(0.0, 1.0, size=(N, m))
    loadings[:, 0] = rng.normal(1.0, 0.3, size=N)

    # Discretised OU for the cumulative residual X; daily residual = dX.
    b = np.exp(-kappa * dt)
    sigma_eq = 0.05
    X = np.zeros((T + 1, N))
    for t in range(T):
        X[t + 1] = b * X[t] + rng.normal(0, sigma_eq * np.sqrt(1 - b**2), size=N)
    resid = np.diff(X, axis=0)

    R = factors @ loadings.T + resid
    idx = pd.bdate_range("2015-01-01", periods=T)
    frame = pd.DataFrame(R, index=idx, columns=[f"S{i}" for i in range(N)])
    if return_factors:
        return frame, loadings, pd.DataFrame(factors, index=idx, columns=[f"F{j}" for j in range(m)])
    return frame, loadings


@pytest.fixture
def factor_market():
    return simulate_factor_market()


@pytest.fixture
def factor_market_with_factors():
    """Same market, plus the true factor returns (stand-ins for tradable ETFs)."""
    return simulate_factor_market(return_factors=True)
