"""Factor regression, OU fit of the residual process, and s-scores (Avellaneda & Lee, 2010, Sec. 4 & App. A)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass
class ResidualFit:
    alpha: np.ndarray      # (N,) regression intercept (daily drift)
    betas: np.ndarray      # (N, m) loadings on the factor returns
    kappa: np.ndarray      # (N,) annualised speed of mean reversion
    m: np.ndarray          # (N,) OU equilibrium level of the cumulative residual
    sigma_eq: np.ndarray   # (N,) equilibrium std of the OU process
    x_last: np.ndarray     # (N,) last value of the cumulative residual


def fit_residuals(stock_returns: np.ndarray, factor_returns: np.ndarray) -> ResidualFit:
    """Regress each stock on the factors, then fit an AR(1) to the cumulative residuals.

    stock_returns: (L, N), factor_returns: (L, m). All N regressions share one design
    matrix, so they are solved in a single least-squares call.

    Model:  R_i = alpha_i + sum_j beta_ij F_j + eps_i,   X_i(k) = sum_{l<=k} eps_i(l)
            X(k+1) = a + b X(k) + zeta,   kappa = -ln(b) * 252,
            m = a / (1 - b),   sigma_eq = sqrt(Var(zeta) / (1 - b^2))
    Fits with b outside (0, 1) are not mean-reverting; their kappa/m/sigma_eq are NaN.
    """
    L = stock_returns.shape[0]
    design = np.column_stack([np.ones(L), factor_returns])
    coef, *_ = np.linalg.lstsq(design, stock_returns, rcond=None)
    eps = stock_returns - design @ coef
    X = np.cumsum(eps, axis=0)

    x, y = X[:-1], X[1:]
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    var_x = ((x - x_mean) ** 2).sum(axis=0)
    b = ((x - x_mean) * (y - y_mean)).sum(axis=0) / np.where(var_x > 0, var_x, np.nan)
    a = y_mean - b * x_mean
    zeta = y - (a + b * x)
    var_zeta = zeta.var(axis=0, ddof=1)

    ok = (b > 0) & (b < 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(ok, -np.log(b) * TRADING_DAYS, np.nan)
        m = np.where(ok, a / (1 - b), np.nan)
        sigma_eq = np.where(ok, np.sqrt(var_zeta / (1 - b**2)), np.nan)

    return ResidualFit(
        alpha=coef[0], betas=coef[1:].T, kappa=kappa, m=m, sigma_eq=sigma_eq, x_last=X[-1]
    )


def s_scores(fit: ResidualFit, kappa_min: float, center_mean: bool = True) -> np.ndarray:
    """s = (X_last - m) / sigma_eq, NaN where the residual is not mean-reverting fast enough.

    With centring, m is replaced by m - mean(m) across eligible stocks, which AL found
    removes a systematic bias from the regression intercepts.
    """
    eligible = np.isfinite(fit.kappa) & (fit.kappa > kappa_min) & (fit.sigma_eq > 0)
    m = fit.m.copy()
    if center_mean and eligible.any():
        m = m - np.nanmean(m[eligible])
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (fit.x_last - m) / fit.sigma_eq
    return np.where(eligible, s, np.nan)
