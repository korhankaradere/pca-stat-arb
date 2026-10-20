"""Eigenportfolios from the empirical correlation matrix (Avellaneda & Lee, 2010, Sec. 2.1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Eigenportfolios:
    weights: np.ndarray       # (N, m): dollar weight of stock i in eigenportfolio j, Q_i^(j) = v_i^(j) / sigma_i
    eigenvalues: np.ndarray   # (N,): all eigenvalues of the correlation matrix, descending
    eigenvectors: np.ndarray  # (N, m): leading eigenvectors

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        """Share of total variance explained by each kept factor."""
        m = self.weights.shape[1]
        return self.eigenvalues[:m] / self.eigenvalues.sum()


def fit_eigenportfolios(returns: np.ndarray, n_factors: int) -> Eigenportfolios:
    """Fit eigenportfolios on a (T, N) window of returns with no missing values.

    Returns are standardised per stock, the correlation matrix is diagonalised, and
    each leading eigenvector is turned into dollar weights by dividing by each stock's
    volatility. That makes the eigenportfolio return F_j = sum_i Q_i^(j) R_i.
    """
    if np.isnan(returns).any():
        raise ValueError("returns window contains NaNs; filter the universe first")
    mu = returns.mean(axis=0)
    sigma = returns.std(axis=0, ddof=1)
    sigma = np.where(sigma > 0, sigma, np.nan)
    z = (returns - mu) / sigma
    corr = (z.T @ z) / (returns.shape[0] - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)  # ascending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    v = eigenvectors[:, order[:n_factors]]

    # Sign convention: make each eigenvector's weights sum to a positive number so that the
    # first factor reads as "the market" and the factor plots are comparable across days.
    v = v * np.sign(v.sum(axis=0, keepdims=True) + 1e-300)

    return Eigenportfolios(weights=v / sigma[:, None], eigenvalues=eigenvalues, eigenvectors=v)
